"""
Regime Detection Module
=======================

This module contains market regime detection implementations.

IMPLEMENTED DETECTORS:
======================

1. RuleBasedRegimeDetector
   - Uses SMA crossover (50/200) + volatility
   - No training required
   - Simple and interpretable

2. HMMRegimeDetector  
   - Hidden Markov Model for regime detection
   - Learns regimes from data
   - More sophisticated but requires training

3. UnifiedRegimeDetector
   - Unified interface for both methods
   - Easy to swap detection methods

MARKET REGIMES:
===============
- BULL: Rising prices, low volatility, SMA50 > SMA200
- BEAR: Falling prices, high volatility, SMA50 < SMA200  
- SIDEWAYS: No clear trend, moderate volatility

USAGE:
======
>>> from src.regime import UnifiedRegimeDetector, MarketRegime
>>>
>>> # Rule-based detection (no training needed)
>>> detector = UnifiedRegimeDetector(method='rule_based')
>>> current_regime = detector.detect_current(price_data)

Author: Prince Alexander John
Date: February 2026
"""

from enum import Enum
from typing import Optional, Dict, Union
import pandas as pd
import numpy as np
import warnings


# =============================================================================
# Try to import detectors and canonical MarketRegime, fall back gracefully
# =============================================================================

RuleBasedRegimeDetector = None
HMMRegimeDetector = None
RegimeDetectionConfig = None
MarketRegime = None

try:
    from .rule_based_detector import (
        RuleBasedRegimeDetector,
        RegimeDetectionConfig,
        MarketRegime,
    )
except ImportError as e:
    pass

# Fallback MarketRegime if rule_based_detector couldn't be imported
if MarketRegime is None:
    class MarketRegime(Enum):
        """Market regime labels (fallback definition)."""
        BULL = 'bull'
        BEAR = 'bear'
        SIDEWAYS = 'sideways'

        def __str__(self):
            return self.value

try:
    from .hmm_detector import HMMRegimeDetector
except ImportError as e:
    pass


# =============================================================================
# UnifiedRegimeDetector - Always available (with fallback implementation)
# =============================================================================

class UnifiedRegimeDetector:
    """
    Unified interface for regime detection.
    
    Provides a single API regardless of which detection method you're using.
    Falls back to a simple built-in detector if specialized modules aren't available.
    
    Example:
    --------
    >>> detector = UnifiedRegimeDetector(method='rule_based')
    >>> current_regime = detector.detect_current(price_data)  # Returns 'bull', 'bear', or 'sideways'
    """
    
    def __init__(self, method: str = 'rule_based', **kwargs):
        """
        Initialize detector.
        
        Parameters:
        -----------
        method : str
            'rule_based' or 'hmm'
        """
        self.method = method
        self._detector = None
        self._is_fitted = False
        
        # Configuration
        self.sma_short = kwargs.get('sma_short', 50)
        self.sma_long = kwargs.get('sma_long', 200)
        self.volatility_window = kwargs.get('volatility_window', 20)
        
        # Cache for regime detection
        self._regime_cache: Optional[pd.Series] = None
        self._cache_key: Optional[str] = None
        
        # Try to use specialized detectors if available
        if method == 'rule_based':
            if RuleBasedRegimeDetector is not None:
                try:
                    self._detector = RuleBasedRegimeDetector(**kwargs)
                    self._is_fitted = True
                except Exception as e:
                    warnings.warn(f"Could not create RuleBasedRegimeDetector: {e}")
            
            if self._detector is None:
                # Use built-in fallback
                self._is_fitted = True
                
        elif method == 'hmm':
            if HMMRegimeDetector is not None:
                try:
                    self._detector = HMMRegimeDetector(**kwargs)
                except Exception as e:
                    warnings.warn(f"Could not create HMMRegimeDetector: {e}. Falling back to rule-based.")
                    self.method = 'rule_based'
                    self._is_fitted = True
            else:
                warnings.warn("HMMRegimeDetector not available. Falling back to rule-based.")
                self.method = 'rule_based'
                self._is_fitted = True
        else:
            raise ValueError(f"Unknown method: {method}. Use 'rule_based' or 'hmm'")
    
    def fit(self, price_data, verbose: bool = True):
        """
        Fit the detector (only needed for HMM).
        
        Parameters:
        -----------
        price_data : pd.DataFrame or Dict[str, pd.DataFrame]
            Historical price data
        verbose : bool
            Print progress
        """
        if self._detector is not None and hasattr(self._detector, 'fit'):
            self._detector.fit(price_data, verbose=verbose)
        self._is_fitted = True
        return self
    
    def detect(self, price_data: Union[pd.DataFrame, Dict[str, pd.DataFrame]]) -> pd.Series:
        """
        Detect regimes for all dates in the data.
        
        Parameters:
        -----------
        price_data : pd.DataFrame or Dict[str, pd.DataFrame]
            Price data with OHLCV columns
            
        Returns:
        --------
        pd.Series
            Series of MarketRegime values indexed by date
        """
        # Use specialized detector if available
        if self._detector is not None:
            return self._detector.detect(price_data)
        
        # Fallback: built-in simple detector
        return self._simple_detect(price_data)
    
    def _simple_detect(self, price_data: Union[pd.DataFrame, Dict[str, pd.DataFrame]]) -> pd.Series:
        """
        Simple built-in regime detection (fallback).
        
        Uses SMA crossover + volatility threshold.
        """
        # Handle dict input
        if isinstance(price_data, dict):
            if 'SPY' in price_data:
                df = price_data['SPY']
            else:
                df = list(price_data.values())[0]
        else:
            df = price_data
        
        # Find price column
        price_col = None
        for col in ['close', 'Close', 'Adj Close', 'adj_close']:
            if col in df.columns:
                price_col = col
                break
        
        if price_col is None:
            price_col = df.columns[0]
        
        prices = df[price_col].values
        
        # Calculate SMAs
        sma_short = self._rolling_mean(prices, self.sma_short)
        sma_long = self._rolling_mean(prices, self.sma_long)
        
        # Calculate volatility
        returns = np.diff(prices) / prices[:-1]
        returns = np.insert(returns, 0, 0)
        volatility = self._rolling_std(returns, self.volatility_window)
        vol_threshold = np.nanpercentile(volatility, 75)
        
        # Classify regimes
        regimes = []
        for i in range(len(prices)):
            if np.isnan(sma_short[i]) or np.isnan(sma_long[i]):
                regimes.append(MarketRegime.SIDEWAYS)
            elif sma_short[i] > sma_long[i] * 1.02:  # 2% above
                regimes.append(MarketRegime.BULL)
            elif sma_short[i] < sma_long[i] * 0.98:  # 2% below
                regimes.append(MarketRegime.BEAR)
            else:
                regimes.append(MarketRegime.SIDEWAYS)
        
        return pd.Series(regimes, index=df.index, name='regime')
    
    def detect_current(self, price_data: Union[pd.DataFrame, Dict[str, pd.DataFrame]]) -> str:
        """
        Get the current (most recent) regime as a string.
        
        This is the main method used during RL training to label experiences.
        
        Parameters:
        -----------
        price_data : pd.DataFrame or Dict
            Recent price history
            
        Returns:
        --------
        str
            Current regime: 'bull', 'bear', or 'sideways'
        """
        regimes = self.detect(price_data)
        current = regimes.iloc[-1]
        
        # Convert to string if enum
        if hasattr(current, 'value'):
            return current.value
        return str(current)
    
    def detect_at_index(
        self,
        price_data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
        index: int
    ) -> str:
        """
        Get regime at a specific index as a string.
        
        Parameters:
        -----------
        price_data : pd.DataFrame or Dict
            Price data
        index : int
            Index position
            
        Returns:
        --------
        str
            Regime at that index: 'bull', 'bear', or 'sideways'
        """
        regimes = self.detect(price_data)
        if 0 <= index < len(regimes):
            current = regimes.iloc[index]
            if hasattr(current, 'value'):
                return current.value
            return str(current)
        return 'sideways'
    
    def get_statistics(self, price_data: Union[pd.DataFrame, Dict[str, pd.DataFrame]]) -> Dict:
        """Get statistics about detected regimes."""
        regimes = self.detect(price_data)
        
        total = len(regimes)
        counts = {}
        percentages = {}
        
        for regime in MarketRegime:
            count = (regimes == regime).sum()
            counts[regime.value] = int(count)
            percentages[regime.value] = count / total * 100 if total > 0 else 0
        
        return {
            'total_days': total,
            'counts': counts,
            'percentages': percentages
        }
    
    @staticmethod
    def _rolling_mean(data: np.ndarray, window: int) -> np.ndarray:
        """Compute rolling mean."""
        result = np.full_like(data, np.nan, dtype=float)
        for i in range(window - 1, len(data)):
            result[i] = np.mean(data[i - window + 1:i + 1])
        return result
    
    @staticmethod
    def _rolling_std(data: np.ndarray, window: int) -> np.ndarray:
        """Compute rolling standard deviation."""
        result = np.full_like(data, np.nan, dtype=float)
        for i in range(window - 1, len(data)):
            result[i] = np.std(data[i - window + 1:i + 1])
        return result


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    'MarketRegime',
    'UnifiedRegimeDetector',
    'RuleBasedRegimeDetector',
    'RegimeDetectionConfig',
    'HMMRegimeDetector',
]