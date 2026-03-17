"""
Hidden Markov Model (HMM) Based Market Regime Detection
========================================================

This module implements a STATISTICAL method for detecting market regimes
using Hidden Markov Models, as opposed to the rule-based approach.

WHAT IS A HIDDEN MARKOV MODEL?
==============================

Imagine you're outside a building and can't see inside. You can only see
people's umbrellas when they exit. Based on whether people have umbrellas,
you want to guess the WEATHER inside (sunny, rainy, cloudy).

- The WEATHER is the "hidden state" - you can't directly observe it
- The UMBRELLAS are the "observations" - you can see them
- The HMM learns the relationship between states and observations

In our case:
- Hidden states = Market regimes (bull, bear, sideways)
- Observations = Returns, volatility, volume (what we can measure)

WHY USE HMM INSTEAD OF RULE-BASED?
==================================

Rule-Based:
+ Simple, interpretable, fast
+ You control the rules explicitly
- May miss subtle patterns
- Rules are fixed, not data-driven

HMM-Based:
+ Learns patterns from data
+ Can capture complex relationships
+ Provides probability distributions over regimes
- "Black box" - harder to interpret
- Requires training data
- May overfit

USING BOTH is ideal - they complement each other.

HOW HMM WORKS (Simplified):
===========================

1. HIDDEN STATES: 
   The market can be in one of K states (e.g., K=3 for bull/bear/sideways)
   
2. TRANSITION MATRIX:
   Probability of moving from one state to another
   P(tomorrow=bull | today=bull) = 0.95  (bull markets persist)
   P(tomorrow=bear | today=bull) = 0.03  (rare to flip directly)
   
3. EMISSION MODEL:
   What observations look like in each state
   Bull: mean_return=+0.1%, std=0.5%
   Bear: mean_return=-0.2%, std=2.5%
   
4. INFERENCE:
   Given observations, compute most likely sequence of states
   Uses "Viterbi algorithm" or "forward-backward algorithm"

Author: Prince Alexander John
Date: February 2026
Project: Regime-Aware Experience Replay for DRL Portfolio Management
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Optional, Dict, Union
from dataclasses import dataclass
import warnings

# Import from our rule-based module for consistency
from .rule_based_detector import MarketRegime, RegimeDetectionConfig

# HMM library - if not installed, provide helpful error
try:
    from hmmlearn.hmm import GaussianHMM
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False
    warnings.warn(
        "hmmlearn not installed. Install with: pip install hmmlearn\n"
        "HMM regime detection will not be available."
    )


@dataclass
class HMMConfig:
    """
    Configuration for HMM-based regime detection.
    
    Parameters:
    -----------
    n_regimes : int
        Number of hidden states (regimes) to detect.
        Default 3 for bull/bear/sideways.
        Could use 4 to add "crisis" regime.
        
    n_iterations : int
        Maximum iterations for EM algorithm during training.
        More iterations = better fit but slower.
        
    covariance_type : str
        Type of covariance matrix for emission distributions.
        - 'full': Each state has full covariance matrix (most flexible)
        - 'diag': Diagonal covariance (features independent within state)
        - 'spherical': Single variance per state (simplest)
        
    features : list
        Which features to use for regime detection.
        Default: ['returns', 'volatility', 'volume_change']
        
    random_state : int
        Random seed for reproducibility
        
    min_variance : float
        Minimum variance to add for numerical stability
    """
    n_regimes: int = 3
    n_iterations: int = 100
    covariance_type: str = 'full'
    features: List[str] = None  # Set in __post_init__
    random_state: int = 42
    min_variance: float = 1e-3
    
    def __post_init__(self):
        if self.features is None:
            self.features = ['returns', 'volatility', 'volume_change']


class HMMRegimeDetector:
    """
    Hidden Markov Model based regime detector.
    
    This detector LEARNS regime patterns from historical data,
    as opposed to the rule-based detector which uses predefined rules.
    
    Workflow:
    1. Prepare features from price data
    2. Train HMM on training data (learns transition/emission parameters)
    3. Use trained model to predict regimes on new data
    
    IMPORTANT: Train only on TRAINING data, never on validation/test!
    
    Example Usage:
    --------------
    >>> detector = HMMRegimeDetector()
    >>> 
    >>> # Step 1: Prepare training data (2010-2019)
    >>> train_prices = download_data('2010-01-01', '2019-12-31')
    >>> 
    >>> # Step 2: Fit the model
    >>> detector.fit(train_prices)
    >>> 
    >>> # Step 3: Predict on all data
    >>> all_prices = download_data('2010-01-01', '2023-12-31')
    >>> regimes = detector.predict(all_prices)
    """
    
    def __init__(self, config: Optional[HMMConfig] = None):
        """
        Initialize the HMM regime detector.
        
        Parameters:
        -----------
        config : HMMConfig, optional
            Configuration parameters. Uses defaults if not provided.
        """
        if not HMM_AVAILABLE:
            raise ImportError(
                "hmmlearn is required for HMM regime detection.\n"
                "Install with: pip install hmmlearn"
            )
        
        self.config = config or HMMConfig()
        
        # The actual HMM model (created during fit)
        self._model: Optional[GaussianHMM] = None
        
        # Mapping from HMM state indices to MarketRegime
        # HMM returns states as integers (0, 1, 2)
        # We need to map these to meaningful labels (bull, bear, sideways)
        self._state_mapping: Dict[int, MarketRegime] = {}
        
        # Feature statistics for normalization
        self._feature_means: Optional[np.ndarray] = None
        self._feature_stds: Optional[np.ndarray] = None
        
        # Store training diagnostics
        self._training_log_likelihood: Optional[float] = None
        self._is_fitted: bool = False
    
    def _prepare_features(
        self,
        price_data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
        fit_normalisation: bool = False
    ) -> np.ndarray:
        """
        Prepare feature matrix for HMM from price data.

        WHAT FEATURES DO WE USE?
        ------------------------

        1. RETURNS (daily percentage change)
           - Most direct measure of price movement
           - Bull: positive on average
           - Bear: negative on average

        2. VOLATILITY (rolling std of returns)
           - Measures uncertainty/fear
           - Bear markets have higher volatility
           - Bull markets are calmer

        3. VOLUME CHANGE (percentage change in volume)
           - High volume = important moves
           - Can signal regime transitions

        WHY NORMALISE?
        --------------
        HMMs work best when features are on similar scales.
        We use z-score normalisation: (x - mean) / std

        Parameters:
        -----------
        price_data : pd.DataFrame or Dict[str, pd.DataFrame]
            Must have 'close' and 'volume' columns.
            If dict, uses SPY or the first ticker.

        fit_normalisation : bool
            If True, compute and store normalisation parameters
            If False, use previously stored parameters
            (True during training, False during inference)

        Returns:
        --------
        np.ndarray
            Feature matrix of shape (n_days, n_features)
        """
        # Handle dict input (e.g., from download_ohlcv_data)
        if isinstance(price_data, dict):
            if 'SPY' in price_data:
                price_data = price_data['SPY']
            else:
                price_data = list(price_data.values())[0]

        # Find price column (handle different naming conventions)
        price_col = None
        for col in ['close', 'Close', 'Adj Close', 'adj_close']:
            if col in price_data.columns:
                price_col = col
                break
        if price_col is None:
            # Fallback to first column
            price_col = price_data.columns[0]

        # Find volume column
        volume_col = None
        for col in ['volume', 'Volume']:
            if col in price_data.columns:
                volume_col = col
                break

        features = {}

        # Feature 1: Daily returns
        if 'returns' not in price_data.columns:
            features['returns'] = price_data[price_col].pct_change()
        else:
            features['returns'] = price_data['returns']

        # Feature 2: Rolling volatility (20-day)
        if 'volatility' not in price_data.columns:
            features['volatility'] = features['returns'].rolling(20).std()
        else:
            features['volatility'] = price_data['volatility']

        # Feature 3: Volume change
        if volume_col is not None:
            features['volume_change'] = price_data[volume_col].pct_change()
        else:
            # If no volume, use zero (neutral)
            features['volume_change'] = pd.Series(0, index=price_data.index)
        
        # Combine into DataFrame
        feature_df = pd.DataFrame(features)
        
        # Drop NaN rows (from rolling calculations)
        feature_df = feature_df.dropna()
        
        # Convert to numpy
        X = feature_df.values
        
        # Handle infinities and extreme values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        X = np.clip(X, -10, 10)  # Clip extreme values
        
        # Normalise features
        if fit_normalisation:
            # Compute and store normalisation parameters
            self._feature_means = X.mean(axis=0)
            self._feature_stds = X.std(axis=0)
            # Prevent division by zero
            self._feature_stds = np.maximum(self._feature_stds, 1e-8)
        
        if self._feature_means is not None:
            X = (X - self._feature_means) / self._feature_stds
        
        return X, feature_df.index
    
    def fit(
        self,
        price_data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
        verbose: bool = True
    ) -> 'HMMRegimeDetector':
        """
        Train the HMM on historical price data.
        
        THE TRAINING PROCESS:
        ---------------------
        
        HMM uses the Expectation-Maximisation (EM) algorithm:
        
        1. INITIALISATION:
           - Randomly initialise parameters (or use k-means)
           - Parameters: transition matrix, emission means/covariances
           
        2. E-STEP (Expectation):
           - Given current parameters, compute probability of each state
             at each time step (forward-backward algorithm)
             
        3. M-STEP (Maximisation):
           - Given state probabilities, update parameters to maximise
             likelihood of observed data
             
        4. REPEAT until convergence (or max iterations)
        
        WHAT THE MODEL LEARNS:
        ----------------------
        
        1. Transition Matrix A (3x3 for 3 regimes):
           A[i,j] = P(next_state=j | current_state=i)
           
           Example:
                        Bull  Bear  Side
           From Bull  [ 0.95  0.02  0.03 ]  # Bull persists
           From Bear  [ 0.03  0.90  0.07 ]  # Bear persists
           From Side  [ 0.15  0.10  0.75 ]  # Sideways often transitions
           
        2. Emission Means (what features look like in each state):
           Bull:  [returns=+0.001, vol=0.005, volume=0.0]
           Bear:  [returns=-0.002, vol=0.020, volume=0.1]
           Side:  [returns=0.0,    vol=0.010, volume=0.0]
           
        3. Emission Covariances (how features vary in each state)
        
        Parameters:
        -----------
        price_data : pd.DataFrame or Dict[str, pd.DataFrame]
            Training data. ONLY use training period!
            If dict, uses SPY or the first ticker.

        verbose : bool
            Whether to print training progress
            
        Returns:
        --------
        self
            Returns self for method chaining
        """
        if verbose:
            print("=" * 50)
            print("TRAINING HMM REGIME DETECTOR")
            print("=" * 50)
        
        # Step 1: Prepare features
        if verbose:
            print("\n1. Preparing features...")
        
        X, indices = self._prepare_features(price_data, fit_normalisation=True)
        n_samples, n_features = X.shape
        
        if verbose:
            print(f"   Samples: {n_samples}")
            print(f"   Features: {n_features}")
        
        # Step 2: Create and configure HMM
        if verbose:
            print("\n2. Configuring HMM model...")
            print(f"   Number of regimes: {self.config.n_regimes}")
            print(f"   Covariance type: {self.config.covariance_type}")
        
        self._model = GaussianHMM(
            n_components=self.config.n_regimes,
            covariance_type=self.config.covariance_type,
            n_iter=self.config.n_iterations,
            random_state=self.config.random_state,
            verbose=False,  # We'll handle our own logging
            tol=1e-4  # Convergence tolerance
        )
        
        # Step 3: Train the model
        if verbose:
            print("\n3. Training (EM algorithm)...")
        
        # Fit the model
        self._model.fit(X)
        
        # Store training diagnostics
        self._training_log_likelihood = self._model.score(X)
        
        if verbose:
            print(f"   Training complete!")
            print(f"   Log-likelihood: {self._training_log_likelihood:.2f}")
        
        # Step 4: Map states to regime labels
        if verbose:
            print("\n4. Mapping states to regimes...")
        
        self._map_states_to_regimes(X)
        
        if verbose:
            print("   State mapping:")
            for state_idx, regime in self._state_mapping.items():
                print(f"     State {state_idx} -> {regime.value}")
        
        self._is_fitted = True
        
        if verbose:
            print("\n" + "=" * 50)
            print("TRAINING COMPLETE")
            print("=" * 50)
        
        return self
    
    def _map_states_to_regimes(self, X: np.ndarray) -> None:
        """
        Map HMM state indices to meaningful regime labels.
        
        THE CHALLENGE:
        --------------
        HMM returns states as integers (0, 1, 2) but these are arbitrary.
        State 0 might be bull in one training run and bear in another.
        
        We need to determine which state corresponds to which regime
        based on the LEARNED CHARACTERISTICS of each state.
        
        THE APPROACH:
        -------------
        Look at the mean RETURN for each state:
        - Highest mean return → Bull
        - Lowest mean return → Bear
        - Middle → Sideways
        
        This makes intuitive sense and is robust.
        """
        # Get the learned means for each state
        state_means = self._model.means_
        
        # The first feature is returns (index 0)
        # Get mean return for each state
        mean_returns = state_means[:, 0]
        
        # Sort states by mean return
        # argsort returns indices that would sort the array
        sorted_indices = np.argsort(mean_returns)
        
        # Handle edge cases first
        if self.config.n_regimes == 1:
            # Single regime: default to sideways (neutral)
            self._state_mapping = {sorted_indices[0]: MarketRegime.SIDEWAYS}
            return

        if self.config.n_regimes == 2:
            # Two regimes: bull and bear only
            self._state_mapping = {
                sorted_indices[0]: MarketRegime.BEAR,   # Lowest returns
                sorted_indices[1]: MarketRegime.BULL,   # Highest returns
            }
            return

        # Standard case: n_regimes >= 3
        # Assign labels based on mean returns
        # Lowest return → Bear
        # Highest return → Bull
        # Middle → Sideways
        self._state_mapping = {
            sorted_indices[0]: MarketRegime.BEAR,      # Lowest returns
            sorted_indices[-1]: MarketRegime.BULL,     # Highest returns
        }

        # Handle sideways (all middle states)
        for i in range(1, len(sorted_indices) - 1):
            self._state_mapping[sorted_indices[i]] = MarketRegime.SIDEWAYS
    
    def predict(
        self,
        price_data: Union[pd.DataFrame, Dict[str, pd.DataFrame]]
    ) -> pd.Series:
        """
        Predict regimes for price data using trained model.

        Can be used on:
        - Training data (to label training experiences)
        - Validation data (to tune hyperparameters)
        - Test data (final evaluation)
        - Live data (real-time regime detection)

        Parameters:
        -----------
        price_data : pd.DataFrame or Dict[str, pd.DataFrame]
            Price data to predict regimes for. If dict, uses SPY or first ticker.
            
        Returns:
        --------
        pd.Series
            Regime predictions, indexed by date
        """
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        # Handle dict input (e.g., from download_ohlcv_data)
        if isinstance(price_data, dict):
            if 'SPY' in price_data:
                price_data = price_data['SPY']
            else:
                price_data = list(price_data.values())[0]

        # Prepare features (don't refit normalisation)
        X, indices = self._prepare_features(price_data, fit_normalisation=False)
        
        # Predict most likely state sequence (Viterbi algorithm)
        state_sequence = self._model.predict(X)
        
        # Map states to regime labels
        regimes = [self._state_mapping[s] for s in state_sequence]
        
        # Create Series with date index
        regime_series = pd.Series(regimes, index=indices, name='regime')
        
        # For dates that were dropped (NaN from feature calculation),
        # we need to fill them. Use forward fill.
        full_index = price_data.index
        regime_series = regime_series.reindex(full_index, method='ffill')
        
        # Fill any remaining NaN at the start with the first valid value
        regime_series = regime_series.bfill()

        return regime_series

    def detect(
        self,
        price_data: Union[pd.DataFrame, Dict[str, pd.DataFrame]]
    ) -> pd.Series:
        """
        Alias for predict() to match RuleBasedRegimeDetector API.

        This allows UnifiedRegimeDetector to call detect() on both detectors.
        """
        return self.predict(price_data)

    def predict_proba(
        self,
        price_data: Union[pd.DataFrame, Dict[str, pd.DataFrame]]
    ) -> pd.DataFrame:
        """
        Get probability distribution over regimes.

        Unlike predict() which returns the MOST LIKELY regime,
        this returns PROBABILITIES for each regime.

        Useful for:
        - Uncertainty quantification
        - Soft transitions between regimes
        - Risk management (high uncertainty = be cautious)

        Example output:

                      bull_prob  bear_prob  sideways_prob
        2020-01-01     0.85       0.05        0.10
        2020-01-02     0.82       0.08        0.10
        ...

        Parameters:
        -----------
        price_data : pd.DataFrame or Dict[str, pd.DataFrame]
            Price data. If dict, uses SPY or first ticker.

        Returns:
        --------
        pd.DataFrame
            Probability for each regime at each time step
        """
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        # Handle dict input (e.g., from download_ohlcv_data)
        if isinstance(price_data, dict):
            if 'SPY' in price_data:
                price_data = price_data['SPY']
            else:
                price_data = list(price_data.values())[0]

        # Prepare features
        X, indices = self._prepare_features(price_data, fit_normalisation=False)
        
        # Get state probabilities (posterior probabilities)
        # This uses the forward-backward algorithm
        state_probs = self._model.predict_proba(X)
        
        # Map to regime names
        columns = {}
        for state_idx, regime in self._state_mapping.items():
            columns[f'{regime.value}_prob'] = state_probs[:, state_idx]
        
        prob_df = pd.DataFrame(columns, index=indices)
        
        return prob_df
    
    def get_transition_matrix(self) -> pd.DataFrame:
        """
        Get the learned transition probability matrix.
        
        Returns a matrix where entry [i, j] is the probability
        of transitioning from regime i to regime j.
        
        Returns:
        --------
        pd.DataFrame
            Transition matrix with regime labels
        """
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Get transition matrix from model
        trans_mat = self._model.transmat_
        
        # Create DataFrame with regime labels
        regime_order = [self._state_mapping[i] for i in range(self.config.n_regimes)]
        regime_names = [r.value for r in regime_order]
        
        trans_df = pd.DataFrame(
            trans_mat,
            index=regime_names,
            columns=regime_names
        )
        
        return trans_df
    
    def get_emission_parameters(self) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Get the learned emission parameters for each regime.
        
        Shows what features look like in each regime:
        - Means: average value of each feature
        - Covariances: how features vary and co-vary
        
        Returns:
        --------
        dict
            Emission parameters organised by regime
        """
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        params = {}
        
        for state_idx, regime in self._state_mapping.items():
            # Get mean and covariance for this state
            mean = self._model.means_[state_idx]
            
            if self.config.covariance_type == 'full':
                cov = self._model.covars_[state_idx]
            elif self.config.covariance_type == 'diag':
                cov = np.diag(self._model.covars_[state_idx])
            else:
                cov = self._model.covars_[state_idx] * np.eye(len(mean))
            
            # Denormalise to get interpretable values
            denorm_mean = mean * self._feature_stds + self._feature_means
            
            params[regime.value] = {
                'mean': denorm_mean,
                'mean_normalised': mean,
                'covariance': cov,
                'feature_names': self.config.features
            }
        
        return params
    
    def get_current_regime(
        self, 
        price_data: pd.DataFrame
    ) -> Tuple[MarketRegime, Dict[str, float]]:
        """
        Get current regime with confidence scores.
        
        Useful for live trading decisions.
        
        Parameters:
        -----------
        price_data : pd.DataFrame
            Recent price history
            
        Returns:
        --------
        tuple
            (regime, probabilities)
            regime: Most likely current regime
            probabilities: Dict of regime -> probability
        """
        probs = self.predict_proba(price_data)
        last_probs = probs.iloc[-1].to_dict()
        
        regimes = self.predict(price_data)
        current_regime = regimes.iloc[-1]
        
        return current_regime, last_probs


# =============================================================================
# TESTING / DEMONSTRATION CODE
# =============================================================================

if __name__ == "__main__":
    """
    Demonstrate HMM regime detection.
    
    Run this file directly:
        python hmm_detector.py
    """
    if not HMM_AVAILABLE:
        print("hmmlearn not installed. Cannot run demonstration.")
        print("Install with: pip install hmmlearn")
        exit(1)
    
    print("=" * 60)
    print("HMM REGIME DETECTOR - DEMONSTRATION")
    print("=" * 60)
    
    # Create synthetic data with clear regimes
    np.random.seed(42)
    n_days = 1000
    dates = pd.date_range('2018-01-01', periods=n_days, freq='B')
    
    # Generate prices with regime-dependent characteristics
    price = 100.0
    prices = [price]
    volumes = [1000000]
    
    for i in range(1, n_days):
        # Define regime periods
        if i < 300:  # Bull
            drift, vol = 0.001, 0.008
        elif i < 500:  # Bear
            drift, vol = -0.0015, 0.025
        elif i < 700:  # Sideways
            drift, vol = 0.0, 0.012
        else:  # Bull again
            drift, vol = 0.0012, 0.007
        
        ret = drift + vol * np.random.randn()
        price = price * (1 + ret)
        prices.append(price)
        
        # Volume also varies by regime
        base_vol = 1000000
        vol_noise = base_vol * 0.3 * np.random.randn()
        if i >= 300 and i < 500:  # Higher volume in bear
            base_vol *= 1.5
        volumes.append(max(base_vol + vol_noise, 100000))
    
    price_df = pd.DataFrame({
        'close': prices,
        'volume': volumes
    }, index=dates)
    
    print(f"\nGenerated {n_days} days of synthetic data")
    
    # Split into train and test
    train_data = price_df.iloc[:700]
    test_data = price_df.iloc[700:]
    
    print(f"Training data: {len(train_data)} days")
    print(f"Test data: {len(test_data)} days")
    
    # Create and train detector
    detector = HMMRegimeDetector()
    detector.fit(train_data, verbose=True)
    
    # Predict on all data
    regimes = detector.predict(price_df)
    
    print("\n" + "-" * 40)
    print("REGIME PREDICTIONS")
    print("-" * 40)
    
    regime_counts = regimes.value_counts()
    for regime in MarketRegime:
        count = regime_counts.get(regime, 0)
        pct = count / len(regimes) * 100
        print(f"{regime.value:10s}: {count:4d} days ({pct:5.1f}%)")
    
    print("\n" + "-" * 40)
    print("TRANSITION MATRIX")
    print("-" * 40)
    print("\nProbability of transitioning from row to column:")
    trans_matrix = detector.get_transition_matrix()
    print(trans_matrix.round(3).to_string())
    
    print("\n" + "-" * 40)
    print("EMISSION PARAMETERS (Feature Means by Regime)")
    print("-" * 40)
    params = detector.get_emission_parameters()
    for regime, p in params.items():
        print(f"\n{regime.upper()}:")
        for i, feat in enumerate(p['feature_names']):
            print(f"  {feat}: {p['mean'][i]:.6f}")
    
    print("\n" + "=" * 60)
    print("HMM demonstration complete!")
    print("=" * 60)