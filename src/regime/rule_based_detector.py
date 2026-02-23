"""
Rule-Based Market Regime Detection
===================================

This module implements a simple, interpretable method for detecting market regimes
(bull, bear, sideways) using classic technical analysis indicators.

CONCEPT: What is a Market Regime?
---------------------------------
A market regime is a distinct "state" of the market characterised by:
- Trend direction (up, down, or flat)
- Volatility level (high or low)
- Typical duration (weeks to months)

Think of regimes like weather seasons - each has distinct characteristics and
requires different strategies to navigate successfully.

WHY THIS MATTERS FOR REINFORCEMENT LEARNING:
--------------------------------------------
Standard RL assumes the environment is stationary (doesn't change over time).
But financial markets are NON-STATIONARY - they shift between regimes.

If we train on data that's 60% bull market:
- Agent becomes expert at bull markets
- Agent fails catastrophically in bear markets
- This is called "regime bias" in training

By detecting regimes, we can ensure balanced training across all conditions.

TECHNICAL INDICATORS USED:
--------------------------
1. SMA-50: 50-day Simple Moving Average (short-term trend)
2. SMA-200: 200-day Simple Moving Average (long-term trend)  
3. Volatility: 20-day rolling standard deviation of returns

REGIME RULES:
-------------
- BULL: SMA-50 > SMA-200 AND Volatility < 75th percentile
- BEAR: SMA-50 < SMA-200 AND Volatility > 75th percentile
- SIDEWAYS: Everything else (transitions, uncertain periods)

Author: [Your Name]
Date: February 2026
Project: Regime-Aware Experience Replay for DRL Portfolio Management
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Optional, Dict, Union
from enum import Enum
from dataclasses import dataclass


class MarketRegime(Enum):
    """
    Enumeration of possible market regimes.
    
    Using an Enum (enumeration) instead of strings like 'bull' or 'bear' because:
    1. Prevents typos - MarketRegime.BUL would raise an error, but 'bul' wouldn't
    2. IDE autocomplete support
    3. Type safety
    4. Easy to iterate over all regimes
    
    Example:
        >>> regime = MarketRegime.BULL
        >>> print(regime.value)  # 'bull'
        >>> print(regime.name)   # 'BULL'
    """
    BULL = 'bull'
    BEAR = 'bear'
    SIDEWAYS = 'sideways'


@dataclass
class RegimeDetectionConfig:
    """
    Configuration parameters for regime detection.
    
    Using a dataclass for configuration because:
    1. Clear documentation of all parameters
    2. Type hints for each parameter
    3. Immutable by default (safer)
    4. Easy to serialise to/from YAML
    
    Parameters:
    -----------
    sma_short_period : int
        Period for short-term moving average. Default 50 days.
        Shorter period = more responsive to recent changes.
        
    sma_long_period : int
        Period for long-term moving average. Default 200 days.
        Longer period = smoother, shows "big picture" trend.
        
    volatility_period : int
        Period for volatility calculation. Default 20 days (~1 trading month).
        
    volatility_threshold_percentile : float
        Percentile threshold for high/low volatility classification.
        Default 75 means: if current volatility > 75% of historical volatility,
        it's considered "high volatility".
        
    min_regime_duration : int
        Minimum number of days a regime must persist before we switch.
        Prevents rapid flip-flopping between regimes.
        Default 5 days.
    """
    sma_short_period: int = 50
    sma_long_period: int = 200
    volatility_period: int = 20
    volatility_threshold_percentile: float = 75.0
    min_regime_duration: int = 5


class RuleBasedRegimeDetector:
    """
    Detects market regimes using technical analysis rules.
    
    This is the SIMPLER of our two regime detection methods.
    It's easy to understand and interpret, making it good for:
    - Debugging (you can see exactly why a regime was assigned)
    - Baseline comparison
    - Real-time detection (fast computation)
    
    The TRADEOFF is that it may miss subtle regime changes that
    a statistical model (like HMM) might catch.
    
    Example Usage:
    --------------
    >>> detector = RuleBasedRegimeDetector()
    >>> prices = pd.DataFrame({'close': [...], 'high': [...], 'low': [...]})
    >>> regimes = detector.detect(prices)
    >>> print(regimes)  # Series of MarketRegime values
    
    >>> # Get current regime for live trading
    >>> current = detector.get_current_regime(prices)
    >>> print(f"Current regime: {current.value}")
    """
    
    def __init__(self, config: Optional[RegimeDetectionConfig] = None):
        """
        Initialise the regime detector.
        
        Parameters:
        -----------
        config : RegimeDetectionConfig, optional
            Configuration parameters. If None, uses defaults.
        """
        self.config = config or RegimeDetectionConfig()
        
        # Store the volatility threshold once calculated
        # (we calculate it from historical data)
        self._volatility_threshold: Optional[float] = None
        
        # Store regime history for analysis
        self._regime_history: List[Tuple[pd.Timestamp, MarketRegime]] = []
    
    def _calculate_sma(self, prices: pd.Series, period: int) -> pd.Series:
        """
        Calculate Simple Moving Average.
        
        WHAT IS SMA?
        ------------
        SMA is the arithmetic mean of prices over a specified period.
        It "smooths out" price data to show the underlying trend.
        
        Formula: SMA(n) = (P1 + P2 + ... + Pn) / n
        
        WHY USE IT?
        -----------
        - Removes day-to-day noise
        - Shows direction of trend
        - Widely used by traders (self-fulfilling prophecy effect)
        
        Parameters:
        -----------
        prices : pd.Series
            Price series (typically closing prices)
        period : int
            Number of periods to average
            
        Returns:
        --------
        pd.Series
            Moving average values (first `period-1` values will be NaN)
            
        Example:
        --------
        If prices = [10, 11, 12, 13, 14] and period = 3:
        SMA = [NaN, NaN, 11, 12, 13]
        
        The first SMA value (11) = (10 + 11 + 12) / 3
        """
        return prices.rolling(window=period, min_periods=period).mean()
    
    def _calculate_volatility(self, prices: pd.Series, period: int) -> pd.Series:
        """
        Calculate rolling volatility (standard deviation of returns).
        
        WHAT IS VOLATILITY?
        -------------------
        Volatility measures how much prices fluctuate.
        High volatility = prices jumping around a lot = uncertainty
        Low volatility = prices moving steadily = confidence
        
        We calculate it as the standard deviation of RETURNS, not prices.
        This is because:
        - Returns are percentage changes (scale-independent)
        - A $1 move on a $10 stock is more significant than on a $100 stock
        
        Formula:
        --------
        1. Calculate daily returns: r_t = (P_t - P_{t-1}) / P_{t-1}
        2. Calculate rolling standard deviation of returns
        
        ANNUALISED VS DAILY:
        --------------------
        We calculate daily volatility. To annualise:
        Annual_Vol = Daily_Vol × sqrt(252)  # 252 trading days per year
        
        But we use daily for regime detection since we're comparing
        recent volatility to historical volatility on the same scale.
        
        Parameters:
        -----------
        prices : pd.Series
            Price series
        period : int
            Lookback period for volatility calculation
            
        Returns:
        --------
        pd.Series
            Rolling volatility values
        """
        # Calculate percentage returns
        # pct_change() computes (current - previous) / previous
        returns = prices.pct_change()
        
        # Calculate rolling standard deviation
        volatility = returns.rolling(window=period, min_periods=period).std()
        
        return volatility
    
    def _determine_volatility_threshold(self, volatility: pd.Series) -> float:
        """
        Determine the threshold for "high" vs "low" volatility.
        
        WHY A PERCENTILE THRESHOLD?
        ---------------------------
        We use a percentile rather than a fixed number because:
        - Volatility levels vary across time periods
        - What's "high" for 2015 might be "normal" for 2020
        - Percentiles adapt to the data's distribution
        
        Using 75th percentile means:
        - If current vol > 75% of historical observations → HIGH
        - If current vol ≤ 75% of historical observations → LOW
        
        Parameters:
        -----------
        volatility : pd.Series
            Historical volatility values
            
        Returns:
        --------
        float
            The volatility threshold value
        """
        # Remove NaN values before calculating percentile
        valid_volatility = volatility.dropna()
        
        threshold = np.percentile(
            valid_volatility, 
            self.config.volatility_threshold_percentile
        )
        
        return threshold
    
    def _classify_regime(
        self, 
        sma_short: float, 
        sma_long: float, 
        volatility: float,
        vol_threshold: float
    ) -> MarketRegime:
        """
        Classify the current market regime based on indicators.
        
        DECISION LOGIC:
        ---------------
        
        The logic combines TREND and VOLATILITY:
        
        1. BULL MARKET:
           - Trend: SMA-50 > SMA-200 (short-term above long-term)
           - Volatility: Below threshold (calm, confident market)
           - Interpretation: Prices rising steadily with low fear
           
        2. BEAR MARKET:
           - Trend: SMA-50 < SMA-200 (short-term below long-term)
           - Volatility: Above threshold (fearful, panicky market)
           - Interpretation: Prices falling with high uncertainty
           
        3. SIDEWAYS MARKET:
           - Everything else
           - Could be: trending up but volatile (uncertain rally)
           - Could be: trending down but calm (orderly decline)
           - Could be: no clear trend
           - Interpretation: Transitional or range-bound market
        
        WHY BOTH CONDITIONS?
        --------------------
        Using BOTH trend AND volatility is more robust than either alone:
        
        - Trend alone misses: Early stage bear markets (prices falling 
          but SMA hasn't crossed yet), or volatile rallies
          
        - Volatility alone misses: Whether we're going up or down
        
        Together, they capture the "character" of the market better.
        
        Parameters:
        -----------
        sma_short : float
            Current short-term SMA value
        sma_long : float
            Current long-term SMA value
        volatility : float
            Current volatility value
        vol_threshold : float
            Threshold for high/low volatility
            
        Returns:
        --------
        MarketRegime
            The classified regime
        """
        # Check for NaN values (happens at start of data)
        if pd.isna(sma_short) or pd.isna(sma_long) or pd.isna(volatility):
            return MarketRegime.SIDEWAYS  # Default when we can't determine
        
        # Trend condition: Is short-term MA above or below long-term MA?
        trend_bullish = sma_short > sma_long
        trend_bearish = sma_short < sma_long
        
        # Volatility condition: Is current volatility high or low?
        volatility_low = volatility < vol_threshold
        volatility_high = volatility >= vol_threshold
        
        # Apply classification rules
        if trend_bullish and volatility_low:
            return MarketRegime.BULL
        elif trend_bearish and volatility_high:
            return MarketRegime.BEAR
        else:
            return MarketRegime.SIDEWAYS
    
    def detect(
        self, 
        price_data: pd.DataFrame,
        price_column: str = 'close'
    ) -> pd.Series:
        """
        Detect regimes for entire price history.
        
        This is the MAIN METHOD you'll use during training to label
        all historical experiences with their regimes.
        
        Parameters:
        -----------
        price_data : pd.DataFrame
            DataFrame with price data. Must have the specified price column.
            Index should be DatetimeIndex.
            
        price_column : str
            Name of the column containing prices to use. Default 'close'.
            
        Returns:
        --------
        pd.Series
            Series of MarketRegime values, indexed same as input.
            
        Example:
        --------
        >>> detector = RuleBasedRegimeDetector()
        >>> prices = pd.DataFrame({
        ...     'close': [100, 101, 102, 103, 104, ...],  # 200+ days
        ... }, index=pd.date_range('2020-01-01', periods=500))
        >>> regimes = detector.detect(prices)
        >>> print(regimes.value_counts())
        BULL        250
        SIDEWAYS    150
        BEAR        100
        """
        prices = price_data[price_column]
        
        # Step 1: Calculate indicators
        sma_short = self._calculate_sma(prices, self.config.sma_short_period)
        sma_long = self._calculate_sma(prices, self.config.sma_long_period)
        volatility = self._calculate_volatility(prices, self.config.volatility_period)
        
        # Step 2: Determine volatility threshold from the data
        self._volatility_threshold = self._determine_volatility_threshold(volatility)
        
        # Step 3: Classify each day
        regimes = []
        for i in range(len(prices)):
            regime = self._classify_regime(
                sma_short.iloc[i],
                sma_long.iloc[i],
                volatility.iloc[i],
                self._volatility_threshold
            )
            regimes.append(regime)
        
        # Create Series with same index as input
        regime_series = pd.Series(regimes, index=prices.index, name='regime')
        
        # Step 4: Apply minimum duration filter (optional smoothing)
        regime_series = self._apply_minimum_duration(regime_series)
        
        return regime_series
    
    def _apply_minimum_duration(self, regimes: pd.Series) -> pd.Series:
        """
        Smooth regime labels to prevent rapid flip-flopping.
        
        WHY DO THIS?
        ------------
        Without smoothing, you might get:
        Day 1: BULL
        Day 2: SIDEWAYS  # Just a volatile day
        Day 3: BULL
        Day 4: SIDEWAYS  # Another blip
        Day 5: BULL
        
        This rapid switching is probably noise, not real regime changes.
        We want sustained changes before declaring a new regime.
        
        The algorithm:
        1. Find runs of consecutive same regimes
        2. If a run is shorter than min_duration, replace with previous regime
        
        Parameters:
        -----------
        regimes : pd.Series
            Raw regime classifications
            
        Returns:
        --------
        pd.Series
            Smoothed regime classifications
        """
        if self.config.min_regime_duration <= 1:
            return regimes
        
        smoothed = regimes.copy()
        
        # Simple approach: forward fill short runs
        # More sophisticated approaches exist but this works well enough
        i = 0
        while i < len(smoothed):
            # Find end of current regime run
            current_regime = smoothed.iloc[i]
            j = i + 1
            while j < len(smoothed) and smoothed.iloc[j] == current_regime:
                j += 1
            
            run_length = j - i
            
            # If run is too short and not at the start, replace with previous
            if run_length < self.config.min_regime_duration and i > 0:
                previous_regime = smoothed.iloc[i - 1]
                smoothed.iloc[i:j] = previous_regime
            
            i = j
        
        return smoothed
    
    def get_current_regime(
        self, 
        price_data: pd.DataFrame,
        price_column: str = 'close'
    ) -> MarketRegime:
        """
        Get the current (most recent) market regime.
        
        Use this during live trading or for labelling new experiences.
        
        Parameters:
        -----------
        price_data : pd.DataFrame
            Recent price history (need at least `sma_long_period` days)
            
        price_column : str
            Name of price column
            
        Returns:
        --------
        MarketRegime
            Current regime
        """
        regimes = self.detect(price_data, price_column)
        return regimes.iloc[-1]
    
    def get_regime_statistics(
        self, 
        regimes: pd.Series,
        price_data: Optional[pd.DataFrame] = None
    ) -> Dict[str, any]:
        """
        Calculate statistics about detected regimes.
        
        Useful for:
        - Understanding your training data composition
        - Verifying regime detection is working sensibly
        - Report/documentation
        
        Parameters:
        -----------
        regimes : pd.Series
            Detected regime series
        price_data : pd.DataFrame, optional
            Original price data for additional statistics
            
        Returns:
        --------
        dict
            Dictionary of statistics
        """
        stats = {}
        
        # Count regimes
        regime_counts = regimes.value_counts()
        total_days = len(regimes)
        
        stats['total_days'] = total_days
        stats['regime_counts'] = {r.value: int(regime_counts.get(r, 0)) 
                                   for r in MarketRegime}
        stats['regime_percentages'] = {r.value: regime_counts.get(r, 0) / total_days * 100 
                                        for r in MarketRegime}
        
        # Calculate regime durations
        regime_durations = self._calculate_regime_durations(regimes)
        stats['regime_durations'] = regime_durations
        
        # Number of regime transitions
        transitions = (regimes != regimes.shift(1)).sum() - 1  # -1 for first day
        stats['num_transitions'] = int(transitions)
        stats['avg_regime_duration'] = total_days / (transitions + 1) if transitions > 0 else total_days
        
        return stats
    
    def _calculate_regime_durations(self, regimes: pd.Series) -> Dict[str, Dict[str, float]]:
        """
        Calculate how long each regime typically lasts.
        
        Returns statistics like:
        - Average bull market duration: 45 days
        - Average bear market duration: 20 days
        """
        durations = {r.value: [] for r in MarketRegime}
        
        current_regime = regimes.iloc[0]
        current_duration = 1
        
        for i in range(1, len(regimes)):
            if regimes.iloc[i] == current_regime:
                current_duration += 1
            else:
                durations[current_regime.value].append(current_duration)
                current_regime = regimes.iloc[i]
                current_duration = 1
        
        # Don't forget the last run
        durations[current_regime.value].append(current_duration)
        
        # Calculate statistics
        duration_stats = {}
        for regime, durs in durations.items():
            if durs:
                duration_stats[regime] = {
                    'mean': np.mean(durs),
                    'median': np.median(durs),
                    'min': np.min(durs),
                    'max': np.max(durs),
                    'count': len(durs)
                }
            else:
                duration_stats[regime] = {
                    'mean': 0, 'median': 0, 'min': 0, 'max': 0, 'count': 0
                }
        
        return duration_stats


# =============================================================================
# TESTING / DEMONSTRATION CODE
# =============================================================================

if __name__ == "__main__":
    """
    Demonstrate the regime detector with sample data.
    
    Run this file directly to see how regime detection works:
        python rule_based_detector.py
    """
    import matplotlib.pyplot as plt
    
    print("=" * 60)
    print("RULE-BASED REGIME DETECTOR - DEMONSTRATION")
    print("=" * 60)
    
    # Create sample price data that shows different regimes
    np.random.seed(42)
    
    # Generate synthetic price data with clear regimes
    n_days = 500
    dates = pd.date_range('2020-01-01', periods=n_days, freq='B')  # Business days
    
    # Start price
    price = 100.0
    prices = [price]
    
    # Create different regime periods
    for i in range(1, n_days):
        if i < 150:  # Bull market (steady rise, low volatility)
            drift = 0.001  # 0.1% daily drift up
            vol = 0.005    # 0.5% daily volatility
        elif i < 250:  # Bear market (fall, high volatility)
            drift = -0.002  # -0.2% daily drift
            vol = 0.025     # 2.5% daily volatility (5x higher)
        elif i < 350:  # Sideways (no drift, moderate volatility)
            drift = 0.0
            vol = 0.01
        else:  # Bull market again
            drift = 0.0015
            vol = 0.008
        
        # Generate random return
        ret = drift + vol * np.random.randn()
        price = price * (1 + ret)
        prices.append(price)
    
    price_df = pd.DataFrame({'close': prices}, index=dates)
    
    print(f"\nGenerated {n_days} days of synthetic price data")
    print(f"Date range: {dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}")
    
    # Detect regimes
    detector = RuleBasedRegimeDetector()
    regimes = detector.detect(price_df)
    
    # Get statistics
    stats = detector.get_regime_statistics(regimes)
    
    print("\n" + "-" * 40)
    print("REGIME STATISTICS")
    print("-" * 40)
    print(f"Total trading days: {stats['total_days']}")
    print(f"\nRegime distribution:")
    for regime, count in stats['regime_counts'].items():
        pct = stats['regime_percentages'][regime]
        print(f"  {regime:10s}: {count:4d} days ({pct:5.1f}%)")
    
    print(f"\nNumber of regime transitions: {stats['num_transitions']}")
    print(f"Average regime duration: {stats['avg_regime_duration']:.1f} days")
    
    print("\n" + "-" * 40)
    print("REGIME DURATION DETAILS")
    print("-" * 40)
    for regime, dur_stats in stats['regime_durations'].items():
        print(f"\n{regime.upper()}:")
        print(f"  Episodes: {dur_stats['count']}")
        print(f"  Mean duration: {dur_stats['mean']:.1f} days")
        print(f"  Median: {dur_stats['median']:.1f} days")
        print(f"  Range: {dur_stats['min']:.0f} - {dur_stats['max']:.0f} days")
    
    print("\n" + "=" * 60)
    print("Regime detection demonstration complete!")
    print("=" * 60)