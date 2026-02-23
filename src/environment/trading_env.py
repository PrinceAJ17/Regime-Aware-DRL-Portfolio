"""
Trading Environment Module
==========================

This module implements the Gymnasium environment for portfolio management.
This is where your RL agent interacts with the simulated market.

How It Works (The Big Picture)
------------------------------
1. Environment loads historical market data and pre-calculated features
2. Agent observes the current state (market features + portfolio state)
3. Agent outputs an action (desired portfolio weights)
4. Environment executes trades, deducts costs, advances time
5. Environment calculates reward based on portfolio performance
6. Repeat until episode ends (252 days or bankruptcy)

Key Components
--------------
- **State Space**: What the agent observes (market + portfolio features)
- **Action Space**: What the agent can do (set portfolio weights)
- **Reward Function**: How we measure success (risk-adjusted returns)
- **Transition Dynamics**: How the world changes (prices move, time passes)

Usage
-----
>>> from src.environment.trading_env import TradingEnvironment
>>> from src.data.downloader import download_ohlcv_data
>>> from src.data.features import calculate_all_features
>>>
>>> # Prepare data
>>> data = download_ohlcv_data(['SPY', 'QQQ', 'TLT', 'GLD', 'BIL'], '2015-01-01', '2019-12-31')
>>> features = calculate_all_features(data)
>>>
>>> # Create environment
>>> env = TradingEnvironment(price_data=data, feature_data=features)
>>>
>>> # Use with Stable Baselines 3
>>> from stable_baselines3 import PPO
>>> model = PPO("MlpPolicy", env, verbose=1)
>>> model.learn(total_timesteps=100000)
"""

import logging
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

# Set up logging
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION DEFAULTS
# =============================================================================

# Default assets (from your project specification)
DEFAULT_TICKERS = ['SPY', 'QQQ', 'TLT', 'GLD', 'BIL']

# Episode configuration
DEFAULT_EPISODE_LENGTH = 252  # Trading days in a year
DEFAULT_INITIAL_CAPITAL = 1_000_000  # Starting portfolio value

# Transaction costs (from your specification)
DEFAULT_TRANSACTION_COST = 0.001  # 0.1% = 10 basis points total (commission + slippage)

# Reward function parameters
DEFAULT_RISK_PENALTY_COEFFICIENT = 0.1  # λ_risk
DEFAULT_COST_PENALTY_COEFFICIENT = 1.0  # λ_cost  
DEFAULT_DRAWDOWN_THRESHOLD = 0.10  # 10% drawdown before penalty kicks in

# Termination conditions
DEFAULT_BANKRUPTCY_THRESHOLD = 0.10  # End episode if portfolio drops to 10% of initial


# =============================================================================
# TRADING ENVIRONMENT CLASS
# =============================================================================

class TradingEnvironment(gym.Env):
    """
    A Gymnasium environment for portfolio management with multiple assets.
    
    This environment simulates trading a portfolio of assets (ETFs) based on
    historical market data. The agent observes market features and portfolio
    state, then decides how to allocate capital across assets.
    
    Attributes
    ----------
    tickers : List[str]
        List of asset ticker symbols.
    n_assets : int
        Number of assets in the portfolio.
    episode_length : int
        Number of trading days per episode.
    initial_capital : float
        Starting portfolio value.
    
    Observation Space
    -----------------
    A dictionary containing:
    - 'market_features': Technical indicators for each asset
    - 'portfolio_weights': Current allocation to each asset
    - 'portfolio_value': Current total portfolio value (normalised)
    
    Action Space
    ------------
    Continuous Box of shape (n_assets,) representing target portfolio weights.
    Values are in [-1, 1] and get transformed to valid weights via softmax.
    """
    
    # Gymnasium requires this metadata
    metadata = {'render_modes': ['human']}
    
    def __init__(
        self,
        price_data: Dict[str, pd.DataFrame],
        feature_data: Dict[str, pd.DataFrame],
        tickers: List[str] = None,
        episode_length: int = DEFAULT_EPISODE_LENGTH,
        initial_capital: float = DEFAULT_INITIAL_CAPITAL,
        transaction_cost: float = DEFAULT_TRANSACTION_COST,
        risk_penalty_coef: float = DEFAULT_RISK_PENALTY_COEFFICIENT,
        cost_penalty_coef: float = DEFAULT_COST_PENALTY_COEFFICIENT,
        drawdown_threshold: float = DEFAULT_DRAWDOWN_THRESHOLD,
        bankruptcy_threshold: float = DEFAULT_BANKRUPTCY_THRESHOLD,
        random_start: bool = True,
        start_idx: Optional[int] = None,
        reward_scaling: float = 1.0,
    ):
        """
        Initialise the trading environment.
        
        Parameters
        ----------
        price_data : Dict[str, pd.DataFrame]
            Dictionary mapping tickers to OHLCV DataFrames.
            Output from download_ohlcv_data().
        
        feature_data : Dict[str, pd.DataFrame]
            Dictionary mapping tickers to feature DataFrames.
            Output from calculate_all_features().
        
        tickers : List[str], optional
            Which tickers to trade. Default: all tickers in price_data.
        
        episode_length : int
            Number of trading days per episode. Default: 252 (1 year).
        
        initial_capital : float
            Starting portfolio value in dollars. Default: 1,000,000.
        
        transaction_cost : float
            Cost as fraction of trade value. Default: 0.001 (0.1%).
        
        risk_penalty_coef : float
            Coefficient for drawdown penalty in reward. Default: 0.1.
        
        cost_penalty_coef : float
            Coefficient for transaction cost penalty in reward. Default: 1.0.
        
        drawdown_threshold : float
            Drawdown level before penalty applies. Default: 0.10 (10%).
        
        bankruptcy_threshold : float
            Portfolio value (as fraction of initial) that ends episode.
            Default: 0.10 (10%).
        
        random_start : bool
            If True, start each episode at a random date. Default: True.
        
        start_idx : int, optional
            If provided (and random_start=False), start episodes at this index.
        
        reward_scaling : float
            Multiply rewards by this factor. Default: 1.0.
        """
        super().__init__()
        
        # Store configuration
        self.tickers = tickers or list(price_data.keys())
        self.n_assets = len(self.tickers)
        self.episode_length = episode_length
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.risk_penalty_coef = risk_penalty_coef
        self.cost_penalty_coef = cost_penalty_coef
        self.drawdown_threshold = drawdown_threshold
        self.bankruptcy_threshold = bankruptcy_threshold
        self.random_start = random_start
        self.fixed_start_idx = start_idx
        self.reward_scaling = reward_scaling
        
        # Process and align data
        self._prepare_data(price_data, feature_data)

        # Expose price_data for external access (e.g., regime detection)
        self.price_data = price_data 
        
        # Define observation and action spaces
        self._define_spaces()
        
        # Initialise state variables (will be set in reset())
        self.current_step = 0
        self.start_idx = 0
        self.portfolio_weights = None
        self.portfolio_value = None
        self.peak_value = None
        self.cash = None
        self.shares = None
        self.episode_returns = []
        
        logger.info(f"TradingEnvironment initialised:")
        logger.info(f"  Assets: {self.tickers}")
        logger.info(f"  Episode length: {self.episode_length} days")
        logger.info(f"  Data length: {self.n_steps} days")
        logger.info(f"  Feature dimension per asset: {self.n_features}")
    
    def _prepare_data(
        self,
        price_data: Dict[str, pd.DataFrame],
        feature_data: Dict[str, pd.DataFrame]
    ) -> None:
        """
        Prepare and align price and feature data.
        
        This method:
        1. Extracts adjusted close prices for each ticker
        2. Aligns all data to the same date index
        3. Drops rows with any NaN values
        4. Converts to numpy arrays for fast access
        
        Why Alignment Matters
        ---------------------
        Different assets might have slightly different trading dates (holidays, etc.).
        We need all data aligned to the same dates so that on each step, we have
        valid data for ALL assets.
        """
        # Extract adjusted close prices
        price_dfs = []
        for ticker in self.tickers:
            if ticker not in price_data:
                raise ValueError(f"Ticker {ticker} not found in price_data")
            df = price_data[ticker]
            price_col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
            price_dfs.append(df[price_col].rename(ticker))
        
        # Combine into single DataFrame (aligns on date index)
        self.prices_df = pd.concat(price_dfs, axis=1)
        
        # Extract features for each ticker
        feature_dfs = []
        for ticker in self.tickers:
            if ticker not in feature_data:
                raise ValueError(f"Ticker {ticker} not found in feature_data")
            # Add ticker prefix to column names to avoid conflicts
            df = feature_data[ticker].add_prefix(f"{ticker}_")
            feature_dfs.append(df)
        
        # Combine all features
        self.features_df = pd.concat(feature_dfs, axis=1)
        
        # Align prices and features to same index
        common_index = self.prices_df.index.intersection(self.features_df.index)
        self.prices_df = self.prices_df.loc[common_index]
        self.features_df = self.features_df.loc[common_index]
        
        # Drop rows with any NaN (warm-up period for indicators)
        valid_mask = ~(self.prices_df.isna().any(axis=1) | self.features_df.isna().any(axis=1))
        self.prices_df = self.prices_df[valid_mask]
        self.features_df = self.features_df[valid_mask]
        
        # Store dimensions
        self.n_steps = len(self.prices_df)
        self.n_features = len(self.features_df.columns)
        
        # Convert to numpy for fast indexing
        self.prices = self.prices_df.values  # Shape: (n_steps, n_assets)
        self.features = self.features_df.values  # Shape: (n_steps, n_features)
        self.dates = self.prices_df.index.to_numpy()
        
        # Calculate daily returns for reward calculation
        self.returns = np.zeros_like(self.prices)
        self.returns[1:] = (self.prices[1:] - self.prices[:-1]) / self.prices[:-1]
        
        # Check we have enough data
        min_required = self.episode_length + 1
        if self.n_steps < min_required:
            raise ValueError(
                f"Not enough data: have {self.n_steps} days, need at least {min_required} "
                f"for episode_length={self.episode_length}"
            )
        
        logger.info(f"Data prepared: {self.n_steps} valid trading days, "
                   f"{self.n_features} features total")
    
    def _define_spaces(self) -> None:
        """
        Define the observation and action spaces.
        
        Observation Space
        -----------------
        We use a flat observation (1D array) containing:
        - Market features for all assets (n_features values)
        - Current portfolio weights (n_assets values)
        - Normalised portfolio value (1 value)
        
        Total observation size: n_features + n_assets + 1
        
        Why Flat Instead of Dict?
        -------------------------
        While Gymnasium supports Dict observation spaces, many RL algorithms
        work better with flat observations. We can always restructure internally.
        
        Action Space
        ------------
        Continuous values in [-1, 1] for each asset. These get transformed
        into valid portfolio weights (non-negative, sum to 1) using softmax.
        
        Why Not Directly Output Weights?
        ---------------------------------
        The softmax transformation ensures outputs are always valid weights,
        even if the neural network outputs arbitrary values. This makes
        learning more stable.
        """
        # Observation: features + portfolio weights + normalised value
        obs_dim = self.n_features + self.n_assets + 1
        
        # Observation space: all values normalised roughly to [-10, 10]
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )
        
        # Action space: one value per asset, will be transformed to weights
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.n_assets,),
            dtype=np.float32
        )
    
    def _action_to_weights(self, action: np.ndarray) -> np.ndarray:
        """
        Convert raw action (arbitrary values) to valid portfolio weights.
        
        We use softmax to ensure:
        1. All weights are non-negative (no shorting)
        2. Weights sum to 1.0 (fully invested)
        
        Parameters
        ----------
        action : np.ndarray
            Raw action from the agent, shape (n_assets,).
        
        Returns
        -------
        np.ndarray
            Valid portfolio weights, shape (n_assets,), summing to 1.0.
        
        How Softmax Works
        -----------------
        softmax(x)_i = exp(x_i) / sum(exp(x_j))
        
        This transforms any values into a valid probability distribution.
        Example:
            action = [1.0, 0.5, -0.5, 0.0, 0.0]
            weights ≈ [0.33, 0.20, 0.07, 0.12, 0.12] (sums to ~1.0)
        """
        # Clip to prevent numerical overflow in exp
        action = np.clip(action, -10, 10)
        
        # Softmax transformation
        exp_action = np.exp(action)
        weights = exp_action / np.sum(exp_action)
        
        return weights
    
    def _get_observation(self) -> np.ndarray:
        """
        Construct the observation for the current step.
        
        Returns
        -------
        np.ndarray
            Observation vector containing:
            - Market features (normalised)
            - Current portfolio weights
            - Normalised portfolio value
        """
        # Get current market features
        market_features = self.features[self.current_step]
        
        # Portfolio weights (already normalised, sum to 1)
        weights = self.portfolio_weights
        
        # Normalised portfolio value (relative to initial capital)
        # This tells the agent "am I up or down from the start?"
        normalised_value = np.array([self.portfolio_value / self.initial_capital - 1.0])
        
        # Concatenate all components
        observation = np.concatenate([
            market_features,
            weights,
            normalised_value
        ]).astype(np.float32)
        
        return observation
    
    def _calculate_portfolio_value(self) -> float:
        """
        Calculate current portfolio value based on share holdings and prices.
        
        Returns
        -------
        float
            Total portfolio value = sum(shares * prices) + cash
        """
        current_prices = self.prices[self.current_step]
        holdings_value = np.sum(self.shares * current_prices)
        return holdings_value + self.cash
    
    def _execute_trades(self, target_weights: np.ndarray) -> float:
        """
        Execute trades to achieve target portfolio weights.
        
        This method:
        1. Calculates current portfolio value
        2. Determines target dollar amounts for each asset
        3. Calculates required trades
        4. Executes trades and deducts transaction costs
        
        Parameters
        ----------
        target_weights : np.ndarray
            Desired portfolio weights, shape (n_assets,).
        
        Returns
        -------
        float
            Total transaction cost incurred.
        
        How Trading Works
        -----------------
        Current state: 
            - Portfolio value = $1,000,000
            - Current weights = [0.20, 0.20, 0.20, 0.20, 0.20]
            
        Target weights = [0.40, 0.30, 0.10, 0.10, 0.10]
        
        Target values:
            - SPY: $400,000 (currently $200,000 → buy $200,000)
            - QQQ: $300,000 (currently $200,000 → buy $100,000)
            - TLT: $100,000 (currently $200,000 → sell $100,000)
            - GLD: $100,000 (currently $200,000 → sell $100,000)
            - BIL: $100,000 (currently $200,000 → sell $100,000)
        
        Total traded = $200k + $100k + $100k + $100k + $100k = $600,000
        Transaction cost = $600,000 × 0.1% = $600
        """
        current_prices = self.prices[self.current_step]
        
        # Current portfolio value (before trades)
        current_value = self._calculate_portfolio_value()
        
        # Target dollar amount for each asset
        target_values = target_weights * current_value
        
        # Current dollar amount for each asset
        current_values = self.shares * current_prices
        
        # Required trades (positive = buy, negative = sell)
        trade_values = target_values - current_values
        
        # Calculate transaction cost (proportional to absolute trade value)
        total_traded = np.sum(np.abs(trade_values))
        transaction_cost = total_traded * self.transaction_cost
        
        # Execute trades
        # First, calculate new share counts
        # We assume we can trade fractional shares (common with ETFs)
        new_shares = target_values / current_prices
        
        # Update holdings
        self.shares = new_shares
        
        # Deduct transaction cost from cash
        # (In a more complex model, this might come from selling shares)
        self.cash = 0  # Fully invested model - no separate cash
        
        # Adjust portfolio value for transaction costs
        self.portfolio_value = current_value - transaction_cost
        
        # Update weights
        self.portfolio_weights = target_weights.copy()
        
        return transaction_cost
    
    def _calculate_reward(
        self,
        prev_value: float,
        curr_value: float,
        transaction_cost: float
    ) -> float:
        """
        Calculate the reward for this step.
        
        The reward function from your specification:
        
            reward = R_p - λ_cost × turnover - λ_risk × max(0, DD - DD_threshold)
        
        Where:
        - R_p = Portfolio return
        - turnover = Transaction cost (proxy for turnover)
        - DD = Current drawdown from peak
        - DD_threshold = Acceptable drawdown before penalty
        
        Parameters
        ----------
        prev_value : float
            Portfolio value before this step.
        curr_value : float
            Portfolio value after this step.
        transaction_cost : float
            Transaction costs incurred.
        
        Returns
        -------
        float
            Reward value (positive = good, negative = bad).
        """
        # Portfolio return
        portfolio_return = (curr_value - prev_value) / prev_value
        
        # Transaction cost penalty
        # Normalise by portfolio value to make it scale-invariant
        cost_penalty = self.cost_penalty_coef * (transaction_cost / prev_value)
        
        # Drawdown penalty
        # Update peak value
        self.peak_value = max(self.peak_value, curr_value)
        
        # Calculate current drawdown
        drawdown = (self.peak_value - curr_value) / self.peak_value
        
        # Penalty only if drawdown exceeds threshold
        excess_drawdown = max(0, drawdown - self.drawdown_threshold)
        drawdown_penalty = self.risk_penalty_coef * excess_drawdown
        
        # Total reward
        reward = portfolio_return - cost_penalty - drawdown_penalty
        
        # Apply scaling
        reward *= self.reward_scaling
        
        return reward
    
    def _check_termination(self) -> Tuple[bool, bool]:
        """
        Check if the episode should end.
        
        Returns
        -------
        terminated : bool
            True if episode ended due to environment rules (bankruptcy).
        truncated : bool
            True if episode ended due to time limit (episode_length reached).
        """
        terminated = False
        truncated = False
        
        # Check bankruptcy
        if self.portfolio_value < self.initial_capital * self.bankruptcy_threshold:
            terminated = True
            logger.debug(f"Episode terminated: bankruptcy "
                        f"(value={self.portfolio_value:.2f}, "
                        f"threshold={self.initial_capital * self.bankruptcy_threshold:.2f})")
        
        # Check time limit
        steps_taken = self.current_step - self.start_idx
        if steps_taken >= self.episode_length:
            truncated = True
            logger.debug(f"Episode truncated: reached {self.episode_length} steps")
        
        # Check if we've run out of data
        if self.current_step >= self.n_steps - 1:
            truncated = True
            logger.debug("Episode truncated: end of data")
        
        return terminated, truncated
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment to start a new episode.
        
        Parameters
        ----------
        seed : int, optional
            Random seed for reproducibility.
        options : dict, optional
            Additional options (not used currently).
        
        Returns
        -------
        observation : np.ndarray
            Initial observation for the new episode.
        info : dict
            Additional information about the initial state.
        
        What Happens on Reset
        ---------------------
        1. Choose a starting date (random or fixed)
        2. Initialise portfolio with equal weights
        3. Set portfolio value to initial capital
        4. Return the initial observation
        """
        # Set random seed if provided
        super().reset(seed=seed)
        
        # Determine starting index
        if self.random_start:
            # Random start, but ensure we have enough data for a full episode
            max_start = self.n_steps - self.episode_length - 1
            max_start = max(0, max_start)
            self.start_idx = self.np_random.integers(0, max_start + 1)
        elif self.fixed_start_idx is not None:
            self.start_idx = self.fixed_start_idx
        else:
            self.start_idx = 0
        
        self.current_step = self.start_idx
        
        # Initialise portfolio
        self.portfolio_value = self.initial_capital
        self.peak_value = self.initial_capital
        self.cash = 0  # Fully invested
        
        # Start with equal weights
        self.portfolio_weights = np.ones(self.n_assets) / self.n_assets
        
        # Calculate initial share holdings
        current_prices = self.prices[self.current_step]
        value_per_asset = self.portfolio_value / self.n_assets
        self.shares = value_per_asset / current_prices
        
        # Reset episode tracking
        self.episode_returns = []
        
        # Get initial observation
        observation = self._get_observation()
        
        # Info dictionary
        info = {
            'start_date': str(self.dates[self.start_idx]),
            'initial_value': self.portfolio_value,
            'initial_weights': self.portfolio_weights.copy(),
        }
        
        logger.debug(f"Episode reset: starting at index {self.start_idx} "
                    f"({self.dates[self.start_idx]})")
        
        return observation, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.
        
        Parameters
        ----------
        action : np.ndarray
            Raw action from the agent, shape (n_assets,).
            Will be converted to portfolio weights.
        
        Returns
        -------
        observation : np.ndarray
            New observation after taking the action.
        reward : float
            Reward for this step.
        terminated : bool
            Whether the episode ended due to environment rules.
        truncated : bool
            Whether the episode ended due to time limit.
        info : dict
            Additional information about this step.
        
        What Happens in a Step
        ----------------------
        1. Convert action to target weights
        2. Execute trades (rebalance portfolio)
        3. Advance time by one day
        4. Update portfolio value based on new prices
        5. Calculate reward
        6. Check if episode should end
        7. Return results
        """
        # Store previous state for reward calculation
        prev_value = self.portfolio_value
        prev_weights = self.portfolio_weights.copy()
        
        # Convert action to portfolio weights
        target_weights = self._action_to_weights(action)
        
        # Execute trades to achieve target weights
        transaction_cost = self._execute_trades(target_weights)
        
        # Advance to next time step
        self.current_step += 1
        
        # Update portfolio value based on new prices
        # The price change happens overnight; our shares gain/lose value
        current_prices = self.prices[self.current_step]
        self.portfolio_value = np.sum(self.shares * current_prices) + self.cash
        
        # Calculate reward
        reward = self._calculate_reward(prev_value, self.portfolio_value, transaction_cost)
        
        # Track returns for analysis
        step_return = (self.portfolio_value - prev_value) / prev_value
        self.episode_returns.append(step_return)
        
        # Check termination conditions
        terminated, truncated = self._check_termination()
        
        # Get new observation
        observation = self._get_observation()
        
        # Build info dictionary
        info = {
            'date': str(self.dates[self.current_step]),
            'portfolio_value': self.portfolio_value,
            'portfolio_weights': self.portfolio_weights.copy(),
            'step_return': step_return,
            'transaction_cost': transaction_cost,
            'drawdown': (self.peak_value - self.portfolio_value) / self.peak_value,
        }
        
        # Add final info if episode ended
        if terminated or truncated:
            info['final_value'] = self.portfolio_value
            info['total_return'] = (self.portfolio_value - self.initial_capital) / self.initial_capital
            info['episode_length'] = self.current_step - self.start_idx
            if self.episode_returns:
                info['sharpe_ratio'] = self._calculate_episode_sharpe()
        
        return observation, reward, terminated, truncated, info
    
    def _calculate_episode_sharpe(self, risk_free_rate: float = 0.0) -> float:
        """
        Calculate the Sharpe ratio for the episode.
        
        Parameters
        ----------
        risk_free_rate : float
            Daily risk-free rate (default 0).
        
        Returns
        -------
        float
            Annualised Sharpe ratio.
        """
        if len(self.episode_returns) < 2:
            return 0.0
        
        returns = np.array(self.episode_returns)
        mean_return = np.mean(returns) - risk_free_rate
        std_return = np.std(returns)
        
        if std_return == 0:
            return 0.0
        
        # Annualise (252 trading days)
        sharpe = (mean_return / std_return) * np.sqrt(252)
        return sharpe
    
    def render(self, mode: str = 'human') -> None:
        """
        Render the environment (optional, for debugging).
        
        Currently just prints the state. Could be extended to create
        visualisations.
        """
        print(f"\n{'='*50}")
        print(f"Step: {self.current_step - self.start_idx} / {self.episode_length}")
        print(f"Date: {self.dates[self.current_step]}")
        print(f"Portfolio Value: ${self.portfolio_value:,.2f}")
        print(f"Return from Start: {(self.portfolio_value/self.initial_capital - 1)*100:.2f}%")
        print(f"Drawdown: {(self.peak_value - self.portfolio_value)/self.peak_value*100:.2f}%")
        print(f"Weights: {dict(zip(self.tickers, self.portfolio_weights.round(3)))}")
        print(f"{'='*50}")
    
    def get_portfolio_history(self) -> pd.DataFrame:
        """
        Get a DataFrame of portfolio performance for analysis.
        
        This is useful for evaluating trained agents.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with columns: date, value, return, drawdown
        """
        # This would need to be called after running a full episode
        # with tracking enabled. For now, return episode returns.
        return pd.DataFrame({
            'return': self.episode_returns
        })


# =============================================================================
# HELPER FUNCTION TO CREATE ENVIRONMENT
# =============================================================================

def create_trading_env(
    price_data: Dict[str, pd.DataFrame],
    feature_data: Dict[str, pd.DataFrame],
    **kwargs
) -> TradingEnvironment:
    """
    Convenience function to create a trading environment.
    
    Parameters
    ----------
    price_data : Dict[str, pd.DataFrame]
        Output from download_ohlcv_data().
    feature_data : Dict[str, pd.DataFrame]
        Output from calculate_all_features().
    **kwargs
        Additional arguments passed to TradingEnvironment.
    
    Returns
    -------
    TradingEnvironment
        Configured trading environment.
    
    Example
    -------
    >>> env = create_trading_env(price_data, feature_data, episode_length=252)
    """
    return TradingEnvironment(
        price_data=price_data,
        feature_data=feature_data,
        **kwargs
    )


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    'TradingEnvironment',
    'create_trading_env',
    'DEFAULT_TICKERS',
    'DEFAULT_EPISODE_LENGTH',
    'DEFAULT_INITIAL_CAPITAL',
    'DEFAULT_TRANSACTION_COST',
]


# =============================================================================
# SCRIPT EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Quick test of the environment
    logging.basicConfig(level=logging.INFO)
    
    print("="*60)
    print("Trading Environment Module")
    print("="*60)
    print()
    print("This module provides the TradingEnvironment class for RL training.")
    print()
    print("Usage:")
    print("  >>> from src.environment.trading_env import TradingEnvironment")
    print("  >>> env = TradingEnvironment(price_data, feature_data)")
    print("  >>> obs, info = env.reset()")
    print("  >>> action = env.action_space.sample()")
    print("  >>> obs, reward, terminated, truncated, info = env.step(action)")
    print()
    print("With Stable Baselines 3:")
    print("  >>> from stable_baselines3 import PPO")
    print("  >>> model = PPO('MlpPolicy', env, verbose=1)")
    print("  >>> model.learn(total_timesteps=100000)")