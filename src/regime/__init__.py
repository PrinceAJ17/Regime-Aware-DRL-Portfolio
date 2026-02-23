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
>>> regimes = detector.detect(price_data)
>>>
>>> # HMM detection (requires training)
>>> detector = UnifiedRegimeDetector(method='hmm')
>>> detector.fit(train_data)
>>> regimes = detector.detect(test_data)

Author: [Your Name]
Date: February 2026
"""

# Import from submodules
try:
    from .rule_based_detector import (
        RuleBasedRegimeDetector,
        RegimeDetectionConfig,
        MarketRegime
    )
except ImportError:
    RuleBasedRegimeDetector = None
    RegimeDetectionConfig = None
    MarketRegime = None

try:
    from .hmm_detector import HMMRegimeDetector
except ImportError:
    HMMRegimeDetector = None

# Create a simple MarketRegime enum if not imported
if MarketRegime is None:
    from enum import Enum
    
    class MarketRegime(Enum):
        """Market regime labels."""
        BULL = 'bull'
        BEAR = 'bear'
        SIDEWAYS = 'sideways'


class UnifiedRegimeDetector:
    """
    Unified interface for regime detection.
    
    Provides a single API regardless of which detection method you're using.
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
        
        if method == 'rule_based':
            if RuleBasedRegimeDetector is not None:
                self._detector = RuleBasedRegimeDetector(**kwargs)
                self._is_fitted = True  # Rule-based doesn't need fitting
            else:
                raise ImportError("RuleBasedRegimeDetector not available")
        elif method == 'hmm':
            if HMMRegimeDetector is not None:
                self._detector = HMMRegimeDetector(**kwargs)
            else:
                raise ImportError("HMMRegimeDetector not available. Install hmmlearn.")
        else:
            raise ValueError(f"Unknown method: {method}. Use 'rule_based' or 'hmm'")
    
    def fit(self, price_data, verbose: bool = True):
        """Fit the detector (only needed for HMM)."""
        if self.method == 'hmm' and self._detector is not None:
            self._detector.fit(price_data, verbose=verbose)
        self._is_fitted = True
        return self
    
    def detect(self, price_data):
        """Detect regimes for the given price data."""
        if self._detector is None:
            raise ValueError("Detector not initialized")
        
        if self.method == 'hmm' and not self._is_fitted:
            raise ValueError("HMM detector must be fitted first. Call fit()")
        
        return self._detector.detect(price_data)
    
    def detect_current(self, price_data) -> str:
        """
        Detect the current (most recent) regime.
        
        Returns:
        --------
        str
            'bull', 'bear', or 'sideways'
        """
        regimes = self.detect(price_data)
        
        if hasattr(regimes, 'iloc'):
            # pandas Series
            current = regimes.iloc[-1]
        elif hasattr(regimes, '__getitem__'):
            # list or array
            current = regimes[-1]
        else:
            current = regimes
        
        # Convert to string if enum
        if hasattr(current, 'value'):
            return current.value
        return str(current)


def compare_detection_methods(price_data, train_end_date=None):
    """Compare rule-based and HMM detection methods."""
    results = {}
    
    # Rule-based
    if RuleBasedRegimeDetector is not None:
        rule_detector = UnifiedRegimeDetector(method='rule_based')
        results['rule_based'] = rule_detector.detect(price_data)
    
    # HMM (if available)
    if HMMRegimeDetector is not None:
        try:
            hmm_detector = UnifiedRegimeDetector(method='hmm')
            if train_end_date:
                train_data = price_data[:train_end_date]
            else:
                split_idx = int(len(price_data) * 0.7)
                train_data = price_data.iloc[:split_idx]
            
            hmm_detector.fit(train_data, verbose=False)
            results['hmm'] = hmm_detector.detect(price_data)
        except Exception as e:
            print(f"HMM detection failed: {e}")
    
    return results


__all__ = [
    'MarketRegime',
    'RuleBasedRegimeDetector',
    'RegimeDetectionConfig',
    'HMMRegimeDetector',
    'UnifiedRegimeDetector',
    'compare_detection_methods',
]