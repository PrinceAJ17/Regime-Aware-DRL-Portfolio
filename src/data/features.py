"""
Feature Engineering Module
==========================

This module transforms raw OHLCV data into features that your RL agent can learn from.

Why Feature Engineering?
------------------------
Raw price data is not suitable for machine learning because:
1. Prices are non-stationary (they trend upward over time)
2. Absolute values don't capture patterns (a $1 move means different things at different price levels)
3. Important information is hidden (trends, momentum, volatility)

Features solve these problems by extracting meaningful, stationary signals from raw data.

What This Module Provides
-------------------------
1. **Returns**: Percentage price changes (stationary, scale-independent)
2. **Technical Indicators**: RSI, MACD, Bollinger Bands, ATR, OBV
3. **Volatility Measures**: Rolling standard deviation of returns
4. **Relative Strength**: Performance vs benchmark (SPY)
5. **Normalisation**: Scale all features to similar ranges

Usage
-----
>>> from src.data.features import calculate_all_features
>>> from src.data.downloader import download_ohlcv_data
>>>
>>> # Download raw data
>>> data = download_ohlcv_data(['SPY', 'QQQ'], '2020-01-01', '2023-12-31')
>>>
>>> # Calculate features for each ticker
>>> features = calculate_all_features(data)
>>>
>>> # Access features for a specific ticker
>>> spy_features = features['SPY']
>>> print(spy_features.columns.tolist())
"""

import logging
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

# We use pandas-ta for technical indicators
# It provides reliable, well-tested implementations
import pandas_ta as ta

# Set up logging
logger = logging.getLogger(__name__)

# Default cache directory for computed features
DEFAULT_FEATURE_CACHE_DIR = Path("data/features_cache")


# =============================================================================
# FEATURE CACHING
# =============================================================================

def _get_cache_key(data: Dict[str, pd.DataFrame], **kwargs) -> str:
    """Generate a unique cache key based on data shape and parameters."""
    parts = []
    for ticker in sorted(data.keys()):
        df = data[ticker]
        parts.append(f"{ticker}_{len(df)}_{df.index.min().date()}_{df.index.max().date()}")
    params = "_".join(f"{k}={v}" for k, v in sorted(kwargs.items()))
    content = "|".join(parts) + "|" + params
    return hashlib.md5(content.encode()).hexdigest()[:16]


def save_features_to_cache(
    features: Dict[str, pd.DataFrame],
    cache_key: str,
    cache_dir: Union[str, Path] = DEFAULT_FEATURE_CACHE_DIR
) -> Path:
    """Save computed features to disk cache."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"features_{cache_key}.parquet"

    # Combine all features into one DataFrame with ticker prefix
    combined = pd.concat(
        {ticker: df for ticker, df in features.items()},
        axis=1
    )
    combined.to_parquet(cache_path)
    logger.info(f"Cached features to {cache_path}")
    return cache_path


def load_features_from_cache(
    cache_key: str,
    tickers: List[str],
    cache_dir: Union[str, Path] = DEFAULT_FEATURE_CACHE_DIR
) -> Optional[Dict[str, pd.DataFrame]]:
    """Load cached features from disk. Returns None if cache miss."""
    cache_dir = Path(cache_dir)
    cache_path = cache_dir / f"features_{cache_key}.parquet"

    if not cache_path.exists():
        return None

    try:
        combined = pd.read_parquet(cache_path)
        features = {}
        for ticker in tickers:
            if ticker in combined.columns.get_level_values(0):
                features[ticker] = combined[ticker]
        if len(features) == len(tickers):
            logger.info(f"Loaded features from cache: {cache_path}")
            return features
    except Exception as e:
        logger.warning(f"Failed to load feature cache: {e}")
    return None


def get_or_calculate_features(
    data: Dict[str, pd.DataFrame],
    benchmark_ticker: str = 'SPY',
    normalise: bool = True,
    normalisation_window: int = 60,
    use_cache: bool = True,
    cache_dir: Union[str, Path] = DEFAULT_FEATURE_CACHE_DIR
) -> Dict[str, pd.DataFrame]:
    """
    Get features from cache or calculate them.

    This is the preferred function for batch experiments - it avoids
    recalculating features repeatedly for the same data.
    """
    cache_key = _get_cache_key(
        data,
        benchmark=benchmark_ticker,
        normalise=normalise,
        norm_window=normalisation_window
    )

    if use_cache:
        cached = load_features_from_cache(cache_key, list(data.keys()), cache_dir)
        if cached is not None:
            return cached

    # Calculate features
    features = calculate_all_features(
        data,
        benchmark_ticker=benchmark_ticker,
        normalise=normalise,
        normalisation_window=normalisation_window
    )

    if use_cache:
        save_features_to_cache(features, cache_key, cache_dir)

    return features


# =============================================================================
# CONFIGURATION CONSTANTS
# =============================================================================

# Default parameters matching your project specification
DEFAULT_RSI_PERIOD = 14
DEFAULT_MACD_FAST = 12
DEFAULT_MACD_SLOW = 26
DEFAULT_MACD_SIGNAL = 9
DEFAULT_BB_PERIOD = 20
DEFAULT_BB_STD = 2.0
DEFAULT_ATR_PERIOD = 14
DEFAULT_VOLATILITY_WINDOWS = [5, 10, 20]
DEFAULT_RELATIVE_STRENGTH_WINDOWS = [5, 10, 20]
DEFAULT_NORMALISATION_WINDOW = 60


# =============================================================================
# RETURNS CALCULATION
# =============================================================================

def calculate_returns(
    prices: pd.Series,
    periods: int = 1,
    log_returns: bool = False
) -> pd.Series:
    """
    Calculate percentage returns from a price series.
    
    Why Returns Instead of Prices?
    ------------------------------
    Prices are non-stationary — they trend upward over time. A neural network
    trained on prices from 2010-2015 would see very different numbers than
    prices from 2020-2023. Returns solve this:
    
    - Price: $100 → $101 → $99    (different scale than $300 → $303 → $297)
    - Return: +1% → -2%            (same scale regardless of price level)
    
    Parameters
    ----------
    prices : pd.Series
        Series of prices (typically 'Adj Close').
    periods : int
        Number of periods for return calculation. Default 1 (daily returns).
    log_returns : bool
        If True, calculate log returns: ln(P_t / P_{t-1})
        If False, calculate simple returns: (P_t - P_{t-1}) / P_{t-1}
        
        Log returns have nice mathematical properties (they're additive),
        but simple returns are more intuitive. We use simple returns by default.
    
    Returns
    -------
    pd.Series
        Series of returns, same index as input.
        First `periods` values will be NaN (no prior price to compare to).
    
    Examples
    --------
    >>> prices = pd.Series([100, 101, 99, 102])
    >>> calculate_returns(prices)
    0         NaN
    1    0.010000  # 1% gain
    2   -0.019802  # ~2% loss
    3    0.030303  # ~3% gain
    dtype: float64
    """
    if log_returns:
        # Log return: ln(P_t / P_{t-1})
        returns = np.log(prices / prices.shift(periods))
    else:
        # Simple return: (P_t - P_{t-1}) / P_{t-1}
        returns = prices.pct_change(periods=periods)
    
    return returns


def calculate_cumulative_returns(
    prices: pd.Series,
    window: Optional[int] = None
) -> pd.Series:
    """
    Calculate cumulative returns over a window or from the start.
    
    Parameters
    ----------
    prices : pd.Series
        Series of prices.
    window : Optional[int]
        Rolling window size. If None, calculates cumulative return from start.
    
    Returns
    -------
    pd.Series
        Cumulative returns.
    
    Examples
    --------
    >>> prices = pd.Series([100, 110, 105, 115])
    >>> calculate_cumulative_returns(prices)  # From start
    0    0.00
    1    0.10  # 10% total return
    2    0.05  # 5% total return
    3    0.15  # 15% total return
    
    >>> calculate_cumulative_returns(prices, window=2)  # Rolling 2-day
    0         NaN
    1    0.100000  # 110/100 - 1
    2   -0.045455  # 105/110 - 1
    3    0.095238  # 115/105 - 1
    """
    if window is None:
        # Cumulative return from start
        return prices / prices.iloc[0] - 1
    else:
        # Rolling window return
        return prices / prices.shift(window) - 1


# =============================================================================
# TECHNICAL INDICATORS
# =============================================================================

def calculate_rsi(
    prices: pd.Series,
    period: int = DEFAULT_RSI_PERIOD
) -> pd.Series:
    """
    Calculate the Relative Strength Index (RSI).
    
    What is RSI?
    ------------
    RSI measures momentum — the speed and magnitude of recent price changes.
    It oscillates between 0 and 100.
    
    - RSI > 70: "Overbought" — price may have risen too fast, might fall
    - RSI < 30: "Oversold" — price may have fallen too fast, might rise
    - RSI ≈ 50: Neutral
    
    How It Works
    ------------
    1. Calculate daily gains and losses
    2. Average gains over `period` days
    3. Average losses over `period` days
    4. RS = Average Gain / Average Loss
    5. RSI = 100 - (100 / (1 + RS))
    
    Why It's Useful
    ---------------
    RSI helps identify potential reversal points. If a stock has been going up
    relentlessly (high RSI), it might be due for a pullback. Your agent can
    learn these patterns.
    
    Parameters
    ----------
    prices : pd.Series
        Series of prices (typically 'Adj Close').
    period : int
        Lookback period for RSI calculation. Standard is 14 days.
    
    Returns
    -------
    pd.Series
        RSI values between 0 and 100.
    """
    return ta.rsi(prices, length=period)


def calculate_macd(
    prices: pd.Series,
    fast: int = DEFAULT_MACD_FAST,
    slow: int = DEFAULT_MACD_SLOW,
    signal: int = DEFAULT_MACD_SIGNAL
) -> pd.DataFrame:
    """
    Calculate MACD (Moving Average Convergence Divergence).
    
    What is MACD?
    -------------
    MACD measures the relationship between two moving averages of price.
    It helps identify trend changes and momentum.
    
    Components:
    - MACD Line: Fast EMA - Slow EMA (typically 12-day EMA minus 26-day EMA)
    - Signal Line: EMA of the MACD Line (typically 9-day EMA)
    - Histogram: MACD Line - Signal Line
    
    How to Interpret
    ----------------
    - MACD crosses above Signal → Bullish signal (trend turning up)
    - MACD crosses below Signal → Bearish signal (trend turning down)
    - Histogram positive and growing → Strengthening uptrend
    - Histogram negative and shrinking → Weakening downtrend
    
    Parameters
    ----------
    prices : pd.Series
        Series of prices.
    fast : int
        Fast EMA period (default 12).
    slow : int
        Slow EMA period (default 26).
    signal : int
        Signal line period (default 9).
    
    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - 'MACD_{fast}_{slow}_{signal}': The MACD line
        - 'MACDh_{fast}_{slow}_{signal}': The histogram
        - 'MACDs_{fast}_{slow}_{signal}': The signal line
    """
    return ta.macd(prices, fast=fast, slow=slow, signal=signal)


def calculate_bollinger_bands(
    prices: pd.Series,
    period: int = DEFAULT_BB_PERIOD,
    std: float = DEFAULT_BB_STD
) -> pd.DataFrame:
    """
    Calculate Bollinger Bands.
    
    What are Bollinger Bands?
    -------------------------
    Bollinger Bands are volatility bands placed above and below a moving average.
    They expand during volatile periods and contract during calm periods.
    
    Components:
    - Middle Band: Simple Moving Average (SMA) of price
    - Upper Band: SMA + (standard deviation × multiplier)
    - Lower Band: SMA - (standard deviation × multiplier)
    
    How to Interpret
    ----------------
    - Price near Upper Band → Potentially overbought (or strong uptrend)
    - Price near Lower Band → Potentially oversold (or strong downtrend)
    - Bands squeezing (getting narrow) → Low volatility, breakout may be coming
    - Bands expanding → High volatility, trend in progress
    
    Useful Derived Features
    -----------------------
    - %B: Where price is within the bands (0 = lower band, 1 = upper band)
    - Bandwidth: How wide the bands are (volatility measure)
    
    Parameters
    ----------
    prices : pd.Series
        Series of prices.
    period : int
        Lookback period for SMA and standard deviation (default 20).
    std : float
        Number of standard deviations for bands (default 2.0).
    
    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - 'BBL_{period}_{std}': Lower band
        - 'BBM_{period}_{std}': Middle band (SMA)
        - 'BBU_{period}_{std}': Upper band
        - 'BBB_{period}_{std}': Bandwidth
        - 'BBP_{period}_{std}': Percent B (%B)
    """
    return ta.bbands(prices, length=period, std=std)


def calculate_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = DEFAULT_ATR_PERIOD
) -> pd.Series:
    """
    Calculate Average True Range (ATR).
    
    What is ATR?
    ------------
    ATR measures volatility by looking at the full range of price movement,
    including gaps between days.
    
    True Range is the greatest of:
    1. Current High - Current Low
    2. |Current High - Previous Close|
    3. |Current Low - Previous Close|
    
    ATR is the moving average of True Range.
    
    Why It's Useful
    ---------------
    - ATR tells you how much an asset typically moves per day
    - High ATR = volatile, large daily swings
    - Low ATR = calm, small daily movements
    - Your agent can use this to adjust position sizes or expectations
    
    Parameters
    ----------
    high : pd.Series
        High prices.
    low : pd.Series
        Low prices.
    close : pd.Series
        Close prices.
    period : int
        Lookback period (default 14).
    
    Returns
    -------
    pd.Series
        ATR values.
    """
    return ta.atr(high=high, low=low, close=close, length=period)


def calculate_obv(
    close: pd.Series,
    volume: pd.Series
) -> pd.Series:
    """
    Calculate On-Balance Volume (OBV).
    
    What is OBV?
    ------------
    OBV uses volume flow to predict price changes. The theory is that
    volume precedes price — smart money moves volume before the price follows.
    
    How It Works
    ------------
    - If price closes up: Add day's volume to cumulative OBV
    - If price closes down: Subtract day's volume from cumulative OBV
    - If price unchanged: OBV stays the same
    
    How to Interpret
    ----------------
    - OBV rising while price is flat → Accumulation (potential bullish)
    - OBV falling while price is flat → Distribution (potential bearish)
    - OBV confirms price trend → Trend is healthy
    - OBV diverges from price → Potential reversal coming
    
    Parameters
    ----------
    close : pd.Series
        Close prices.
    volume : pd.Series
        Trading volume.
    
    Returns
    -------
    pd.Series
        OBV values (cumulative).
    """
    return ta.obv(close=close, volume=volume)


# =============================================================================
# VOLATILITY MEASURES
# =============================================================================

def calculate_volatility(
    prices: pd.Series,
    windows: List[int] = None,
    annualise: bool = True,
    trading_days: int = 252
) -> pd.DataFrame:
    """
    Calculate rolling volatility (standard deviation of returns).
    
    What is Volatility?
    -------------------
    Volatility measures how much returns vary from their average.
    High volatility = returns swing wildly; Low volatility = returns are stable.
    
    It's calculated as the standard deviation of returns, typically annualised.
    
    Why Multiple Windows?
    ---------------------
    Different window sizes capture different information:
    - 5-day volatility: Very recent, reactive to immediate events
    - 10-day volatility: Short-term, smooths out daily noise
    - 20-day volatility: Medium-term, captures monthly patterns
    
    Your agent can learn that certain volatility patterns precede moves.
    
    Parameters
    ----------
    prices : pd.Series
        Series of prices.
    windows : List[int]
        Rolling window sizes (default [5, 10, 20]).
    annualise : bool
        If True, multiply by sqrt(trading_days) to annualise.
    trading_days : int
        Number of trading days per year (default 252).
    
    Returns
    -------
    pd.DataFrame
        DataFrame with columns 'volatility_{window}' for each window.
    """
    if windows is None:
        windows = DEFAULT_VOLATILITY_WINDOWS
    
    # First calculate returns
    returns = calculate_returns(prices)
    
    result = pd.DataFrame(index=prices.index)
    
    for window in windows:
        vol = returns.rolling(window=window).std()
        
        if annualise:
            vol = vol * np.sqrt(trading_days)
        
        result[f'volatility_{window}d'] = vol
    
    return result


def calculate_realised_volatility(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 20
) -> pd.Series:
    """
    Calculate realised volatility using Parkinson estimator.
    
    Why Parkinson Instead of Standard Deviation?
    ---------------------------------------------
    Standard volatility only uses close prices. But intraday, prices swing
    to the high and low. Parkinson's formula uses high and low prices,
    giving a more accurate volatility estimate with less data.
    
    Formula: sqrt( (1/n) * sum( (ln(High/Low))^2 / (4 * ln(2)) ) )
    
    Parameters
    ----------
    high : pd.Series
        High prices.
    low : pd.Series
        Low prices.
    close : pd.Series
        Close prices (used for index alignment).
    window : int
        Rolling window size.
    
    Returns
    -------
    pd.Series
        Realised volatility values.
    """
    # Parkinson volatility estimator
    log_hl_ratio = np.log(high / low)
    parkinson = (log_hl_ratio ** 2) / (4 * np.log(2))
    realised_vol = np.sqrt(parkinson.rolling(window=window).mean())
    
    # Annualise
    realised_vol = realised_vol * np.sqrt(252)
    
    return realised_vol


# =============================================================================
# RELATIVE STRENGTH
# =============================================================================

def calculate_relative_strength(
    prices: pd.Series,
    benchmark_prices: pd.Series,
    windows: List[int] = None
) -> pd.DataFrame:
    """
    Calculate relative strength vs a benchmark.
    
    What is Relative Strength?
    --------------------------
    Relative strength measures how an asset performs compared to a benchmark
    (typically the broad market, like SPY).
    
    - Relative Strength > 1: Asset outperformed the benchmark
    - Relative Strength < 1: Asset underperformed the benchmark
    - Relative Strength = 1: Asset performed the same as benchmark
    
    Why It's Useful
    ---------------
    Markets have different regimes. In a bull market, some stocks lead (high RS)
    and others lag. Your agent can learn to overweight strong performers.
    
    Parameters
    ----------
    prices : pd.Series
        Prices of the asset to analyse.
    benchmark_prices : pd.Series
        Prices of the benchmark (e.g., SPY).
    windows : List[int]
        Rolling window sizes for comparison (default [5, 10, 20]).
    
    Returns
    -------
    pd.DataFrame
        DataFrame with columns 'relative_strength_{window}d' for each window.
    """
    if windows is None:
        windows = DEFAULT_RELATIVE_STRENGTH_WINDOWS
    
    result = pd.DataFrame(index=prices.index)
    
    for window in windows:
        # Calculate returns over the window
        asset_return = prices / prices.shift(window) - 1
        benchmark_return = benchmark_prices / benchmark_prices.shift(window) - 1
        
        # Relative strength ratio
        # Add small epsilon to avoid division by zero
        rs = (1 + asset_return) / (1 + benchmark_return + 1e-10)
        
        result[f'relative_strength_{window}d'] = rs
    
    return result


# =============================================================================
# NORMALISATION
# =============================================================================

def normalise_zscore(
    series: pd.Series,
    window: int = DEFAULT_NORMALISATION_WINDOW
) -> pd.Series:
    """
    Normalise a series using rolling z-score.

    What is Z-Score Normalisation?
    ------------------------------
    Z-score transforms data so it has mean=0 and standard deviation=1.

    Formula: z = (x - mean) / std

    We use a ROLLING window so that:
    - The normalisation adapts to changing market conditions
    - Values are relative to recent history, not all-time history
    - The agent sees comparable scales even if volatility changes

    Why Rolling?
    ------------
    Markets change over time. Volatility in 2008 was much higher than 2017.
    If we normalised using all-time statistics, features from calm periods
    would look very different from features during crises.

    Rolling z-score keeps everything relative to recent history, making
    the agent's job easier.

    Note: pandas rolling() defaults to center=False, meaning it only uses
    PAST data (no look-ahead bias).

    Parameters
    ----------
    series : pd.Series
        Series to normalise.
    window : int
        Rolling window size (default 60 days ≈ 3 months).

    Returns
    -------
    pd.Series
        Z-score normalised series.
    """
    rolling_mean = series.rolling(window=window).mean()
    rolling_std = series.rolling(window=window).std()

    # Avoid division by zero
    rolling_std = rolling_std.replace(0, np.nan)

    return (series - rolling_mean) / rolling_std


def normalise_minmax(
    series: pd.Series,
    window: int = DEFAULT_NORMALISATION_WINDOW,
    feature_range: tuple = (0, 1)
) -> pd.Series:
    """
    Normalise a series using rolling min-max scaling.

    What is Min-Max Normalisation?
    ------------------------------
    Min-max scaling transforms data to a fixed range, typically [0, 1].

    Formula: x_scaled = (x - min) / (max - min)

    Useful for features that have natural bounds (like RSI which is 0-100)
    or when you want values in a specific range.

    Note: pandas rolling() defaults to center=False, meaning it only uses
    PAST data (no look-ahead bias).

    Parameters
    ----------
    series : pd.Series
        Series to normalise.
    window : int
        Rolling window size.
    feature_range : tuple
        Desired output range (default (0, 1)).

    Returns
    -------
    pd.Series
        Min-max normalised series.
    """
    rolling_min = series.rolling(window=window).min()
    rolling_max = series.rolling(window=window).max()

    # Avoid division by zero
    range_diff = rolling_max - rolling_min
    range_diff = range_diff.replace(0, np.nan)

    # Scale to [0, 1]
    scaled = (series - rolling_min) / range_diff

    # Scale to desired range
    min_val, max_val = feature_range
    return scaled * (max_val - min_val) + min_val


# =============================================================================
# SPREAD PROXY (MICROSTRUCTURE)
# =============================================================================

def calculate_spread_proxy(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series
) -> pd.Series:
    """
    Calculate a bid-ask spread proxy from OHLC data.
    
    Why a Proxy?
    ------------
    Real bid-ask spread data requires expensive Level 2 market data feeds.
    We don't have that, but we can estimate the spread from OHLC prices.
    
    The daily high-low range captures some spread information because:
    - Wide spreads contribute to larger high-low ranges
    - Illiquid stocks (wide spreads) tend to have larger intraday ranges
    
    This is the Corwin-Schultz estimator, simplified:
    Spread ≈ (High - Low) / Close
    
    It's not perfect, but it gives the agent some information about
    market microstructure without requiring expensive data.
    
    Parameters
    ----------
    high : pd.Series
        High prices.
    low : pd.Series
        Low prices.
    close : pd.Series
        Close prices.
    
    Returns
    -------
    pd.Series
        Spread proxy values.
    """
    return (high - low) / close


# =============================================================================
# MAIN FEATURE CALCULATION FUNCTION
# =============================================================================

def calculate_features_for_ticker(
    df: pd.DataFrame,
    benchmark_prices: Optional[pd.Series] = None,
    normalise: bool = True,
    normalisation_window: int = DEFAULT_NORMALISATION_WINDOW
) -> pd.DataFrame:
    """
    Calculate all features for a single ticker's OHLCV DataFrame.
    
    This function orchestrates all the individual feature calculations
    and combines them into a single DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame with columns: Open, High, Low, Close, Adj Close, Volume.
    benchmark_prices : Optional[pd.Series]
        Benchmark prices for relative strength calculation.
        If None, relative strength features are not calculated.
    normalise : bool
        Whether to apply z-score normalisation to features.
    normalisation_window : int
        Window for rolling normalisation.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with all calculated features.
    
    Features Included
    -----------------
    - returns: Daily returns
    - rsi_14: RSI with 14-day period
    - macd, macd_signal, macd_hist: MACD components
    - bb_upper, bb_middle, bb_lower, bb_percent, bb_width: Bollinger Bands
    - atr_14: Average True Range
    - obv: On-Balance Volume
    - volatility_5d, volatility_10d, volatility_20d: Rolling volatility
    - spread_proxy: Bid-ask spread proxy
    """
    features = pd.DataFrame(index=df.index)
    
    # Get price columns
    close = df['Adj Close'] if 'Adj Close' in df.columns else df['Close']
    high = df['High']
    low = df['Low']
    volume = df['Volume']
    
    # -------------------------------------------------------------------------
    # Returns
    # -------------------------------------------------------------------------
    features['returns'] = calculate_returns(close)
    
    # -------------------------------------------------------------------------
    # RSI
    # -------------------------------------------------------------------------
    rsi = calculate_rsi(close, period=DEFAULT_RSI_PERIOD)
    features['rsi_14'] = rsi
    
    # -------------------------------------------------------------------------
    # MACD
    # -------------------------------------------------------------------------
    macd_df = calculate_macd(close)
    if macd_df is not None and not macd_df.empty:
        # Get the column names (they include the parameters)
        macd_cols = macd_df.columns.tolist()
        macd_line_col = [c for c in macd_cols if c.startswith('MACD_') and 'h' not in c and 's' not in c.lower()]
        macd_hist_col = [c for c in macd_cols if 'MACDh' in c]
        macd_signal_col = [c for c in macd_cols if 'MACDs' in c]
        
        if macd_line_col:
            features['macd'] = macd_df[macd_line_col[0]]
        if macd_hist_col:
            features['macd_hist'] = macd_df[macd_hist_col[0]]
        if macd_signal_col:
            features['macd_signal'] = macd_df[macd_signal_col[0]]
    
    # -------------------------------------------------------------------------
    # Bollinger Bands
    # -------------------------------------------------------------------------
    bb_df = calculate_bollinger_bands(close)
    if bb_df is not None and not bb_df.empty:
        bb_cols = bb_df.columns.tolist()
        
        # Extract each band component
        for col in bb_cols:
            if 'BBL_' in col:
                features['bb_lower'] = bb_df[col]
            elif 'BBM_' in col:
                features['bb_middle'] = bb_df[col]
            elif 'BBU_' in col:
                features['bb_upper'] = bb_df[col]
            elif 'BBB_' in col:
                features['bb_width'] = bb_df[col]
            elif 'BBP_' in col:
                features['bb_percent'] = bb_df[col]
    
    # -------------------------------------------------------------------------
    # ATR
    # -------------------------------------------------------------------------
    atr = calculate_atr(high, low, close, period=DEFAULT_ATR_PERIOD)
    features['atr_14'] = atr
    
    # -------------------------------------------------------------------------
    # OBV
    # -------------------------------------------------------------------------
    obv = calculate_obv(close, volume)
    features['obv'] = obv
    
    # -------------------------------------------------------------------------
    # Volatility
    # -------------------------------------------------------------------------
    volatility_df = calculate_volatility(close, windows=DEFAULT_VOLATILITY_WINDOWS)
    for col in volatility_df.columns:
        features[col] = volatility_df[col]
    
    # -------------------------------------------------------------------------
    # Relative Strength (if benchmark provided)
    # -------------------------------------------------------------------------
    if benchmark_prices is not None:
        # Align benchmark with our index
        benchmark_aligned = benchmark_prices.reindex(df.index, method='ffill')
        
        rs_df = calculate_relative_strength(close, benchmark_aligned)
        for col in rs_df.columns:
            features[col] = rs_df[col]
    
    # -------------------------------------------------------------------------
    # Spread Proxy
    # -------------------------------------------------------------------------
    features['spread_proxy'] = calculate_spread_proxy(high, low, close)
    
    # -------------------------------------------------------------------------
    # Normalisation
    # -------------------------------------------------------------------------
    if normalise:
        # Columns that should NOT be normalised
        # (they're already in interpretable ranges or need special handling)
        skip_normalisation = ['returns', 'bb_percent']  # bb_percent is already 0-1
        
        # Columns that are bounded (use min-max instead of z-score)
        bounded_features = ['rsi_14']  # RSI is 0-100
        
        for col in features.columns:
            if col in skip_normalisation:
                continue
            elif col in bounded_features:
                # Min-max normalise to [0, 1]
                features[col] = normalise_minmax(
                    features[col], 
                    window=normalisation_window,
                    feature_range=(0, 1)
                )
            else:
                # Z-score normalise
                features[col] = normalise_zscore(
                    features[col], 
                    window=normalisation_window
                )
    
    return features


def calculate_all_features(
    data: Dict[str, pd.DataFrame],
    benchmark_ticker: str = 'SPY',
    normalise: bool = True,
    normalisation_window: int = DEFAULT_NORMALISATION_WINDOW
) -> Dict[str, pd.DataFrame]:
    """
    Calculate features for all tickers in a data dictionary.
    
    This is the main function you'll use. It takes the output from
    download_ohlcv_data() and calculates all features for each ticker.
    
    Parameters
    ----------
    data : Dict[str, pd.DataFrame]
        Dictionary mapping ticker symbols to OHLCV DataFrames.
        This is the output format from download_ohlcv_data().
    benchmark_ticker : str
        Ticker to use as benchmark for relative strength (default 'SPY').
    normalise : bool
        Whether to normalise features.
    normalisation_window : int
        Window for rolling normalisation.
    
    Returns
    -------
    Dict[str, pd.DataFrame]
        Dictionary mapping ticker symbols to feature DataFrames.
    
    Example
    -------
    >>> from src.data.downloader import download_ohlcv_data
    >>> from src.data.features import calculate_all_features
    >>>
    >>> data = download_ohlcv_data(['SPY', 'QQQ', 'TLT'], '2020-01-01', '2023-12-31')
    >>> features = calculate_all_features(data)
    >>>
    >>> spy_features = features['SPY']
    >>> print(spy_features.columns.tolist())
    >>> print(spy_features.head())
    """
    # Get benchmark prices for relative strength calculation
    if benchmark_ticker in data:
        benchmark_df = data[benchmark_ticker]
        benchmark_prices = benchmark_df['Adj Close'] if 'Adj Close' in benchmark_df.columns else benchmark_df['Close']
    else:
        logger.warning(f"Benchmark ticker '{benchmark_ticker}' not in data. "
                      f"Relative strength features will not be calculated.")
        benchmark_prices = None
    
    features = {}
    
    for ticker, df in data.items():
        logger.info(f"Calculating features for {ticker}...")
        
        # For the benchmark itself, we don't calculate relative strength
        # (it would always be 1.0)
        if ticker == benchmark_ticker:
            ticker_benchmark = None
        else:
            ticker_benchmark = benchmark_prices
        
        features[ticker] = calculate_features_for_ticker(
            df=df,
            benchmark_prices=ticker_benchmark,
            normalise=normalise,
            normalisation_window=normalisation_window
        )
        
        logger.info(f"  Created {len(features[ticker].columns)} features, "
                   f"{len(features[ticker])} rows for {ticker}")
    
    return features


def get_feature_names() -> List[str]:
    """
    Get a list of all feature names that will be created.
    
    This is useful for understanding the feature space and for
    documentation purposes.
    
    Returns
    -------
    List[str]
        List of feature column names.
    """
    return [
        'returns',
        'rsi_14',
        'macd',
        'macd_signal',
        'macd_hist',
        'bb_lower',
        'bb_middle',
        'bb_upper',
        'bb_width',
        'bb_percent',
        'atr_14',
        'obv',
        'volatility_5d',
        'volatility_10d',
        'volatility_20d',
        'relative_strength_5d',
        'relative_strength_10d',
        'relative_strength_20d',
        'spread_proxy',
    ]


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    'calculate_returns',
    'calculate_cumulative_returns',
    'calculate_rsi',
    'calculate_macd',
    'calculate_bollinger_bands',
    'calculate_atr',
    'calculate_obv',
    'calculate_volatility',
    'calculate_realised_volatility',
    'calculate_relative_strength',
    'calculate_spread_proxy',
    'normalise_zscore',
    'normalise_minmax',
    'calculate_features_for_ticker',
    'calculate_all_features',
    'get_feature_names',
]


# =============================================================================
# SCRIPT EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Example usage when running this file directly
    logging.basicConfig(level=logging.INFO)
    
    print("="*60)
    print("Feature Engineering Module")
    print("="*60)
    print()
    print("This module calculates technical indicators and features from OHLCV data.")
    print()
    print("Available features:")
    for feature in get_feature_names():
        print(f"  - {feature}")
    print()
    print("Usage:")
    print("  >>> from src.data.downloader import download_ohlcv_data")
    print("  >>> from src.data.features import calculate_all_features")
    print("  >>> data = download_ohlcv_data(['SPY'], '2020-01-01', '2023-12-31')")
    print("  >>> features = calculate_all_features(data)")