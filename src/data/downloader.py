"""
Data Downloader Module
======================

This module handles downloading financial data from external sources,
primarily Yahoo Finance for price data.

Why This Module Exists
----------------------
Our RL agent needs historical market data to learn from. This module:
1. Downloads OHLCV (Open, High, Low, Close, Volume) data from Yahoo Finance
2. Caches data locally to avoid repeated downloads
3. Handles errors gracefully (network issues, invalid tickers)
4. Provides a clean interface for the rest of your project

Key Concepts
------------
**Ticker**: A short code identifying a financial instrument.
    - SPY = S&P 500 ETF
    - QQQ = Nasdaq 100 ETF
    - AAPL = Apple Inc.

**OHLCV**: The five pieces of daily price data:
    - Open: First trade price of the day
    - High: Highest trade price of the day
    - Low: Lowest trade price of the day
    - Close: Last trade price of the day (most commonly used)
    - Volume: Total shares traded

**Adjusted Close**: The close price adjusted for dividends and stock splits.
    This is what you should use for calculating returns, as it reflects
    the actual value an investor would have received.

Usage
-----
>>> from src.data.downloader import download_ohlcv_data
>>> 
>>> # Download a single ticker
>>> spy_data = download_ohlcv_data(
...     tickers=['SPY'],
...     start_date='2020-01-01',
...     end_date='2023-12-31'
... )
>>> 
>>> # Download multiple tickers
>>> data = download_ohlcv_data(
...     tickers=['SPY', 'QQQ', 'TLT', 'GLD', 'BIL'],
...     start_date='2010-01-01',
...     end_date='2023-12-31'
... )
>>> 
>>> # Access individual ticker data
>>> spy_prices = data['SPY']
>>> print(spy_prices.head())
"""

import os
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union

import pandas as pd
import yfinance as yf

# =============================================================================
# LOGGING SETUP
# =============================================================================
# Logging is like keeping a diary of what your program does.
# Instead of using print() everywhere (which clutters output), we use logging.
# This lets us:
#   - Control verbosity (show only errors, or show everything)
#   - Save logs to files for debugging later
#   - Include timestamps automatically

# Create a logger for this module
# __name__ is a special Python variable that equals 'src.data.downloader' here
logger = logging.getLogger(__name__)

#Create (or fetch) a logger named after this file. Logging is a professional system for 
#recording what your program does, like a diary with timestamps, severity levels and the ability
#to save to files.
# __name__ is a special Python variable containing the current module's name ("src.data.downloader"). This means that every log message shows exactly which file it came from.

# ````
# logger.debug("Detailed debugging info")      # For development
# logger.info("Normal operation info")         # General updates
# logger.warning("Something unexpected")       # Not an error, but notable
# logger.error("Something failed")             # An error occurred
# logger.critical("System is crashing")        # Catastrophic failure
# ````

# We can control what we see by using the basicConfig function.
# logging.basicConfig(level=logging.INFO) -- ONLY shows INFO and above

# =============================================================================
# CONFIGURATION CONSTANTS
# =============================================================================
# These are default values that can be overridden when calling functions.
# Putting them at the top makes them easy to find and change.

# Default directory for cached data (relative to project root)
DEFAULT_CACHE_DIR = Path("data/raw")

# How many days before we consider cached data "stale" and re-download
DEFAULT_CACHE_EXPIRY_DAYS = 1

# Column names we expect from Yahoo Finance
OHLCV_COLUMNS = ['Open', 'High', 'Low', 'Close', 'Volume']

# The column we use for "price" in most calculations
# Adjusted Close accounts for dividends and splits
PRICE_COLUMN = 'Adj Close'


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _ensure_directory_exists(path: Path) -> None:
    """
    Create a directory if it doesn't exist.
    
    This is a helper function (note the underscore prefix, which is Python
    convention for "internal use" functions).
    
    Parameters
    ----------
    path : Path
        The directory path to create.
    
    Why This Exists
    ---------------
    When we try to save a file to 'data/raw/SPY.csv', the 'data/raw' directory
    must exist first. This function ensures it does.
    
    The exist_ok=True parameter means "don't raise an error if it already exists."
    """
    path.mkdir(parents=True, exist_ok=True)
    # a) path:Path -- The input is a Path object(like Path("data/raw"))
    # b) -> None -- This function return nothing (it just does something)
    # c) path.mkdir() -- Creates this directory
    # d) parents=True -- Also create parent folders if needed
    # ````
    # path = Path("data/raw/training")
    # path.mkdir(parents=True, exist_ok=True)
    # ````
    # If nothing exists then it would create data/ then create data/raw and finally create 
    # data/raw/training/. Without parents=True, it would crash if data/ or data/raw didn't exist.

    # e) exist_ok= True -- Don't crash if the folder already exists. Without exist_ok = True, the 
    # second call that would call the creation of the folder again would crash with FileExistsError. 



def _get_cache_filepath(ticker: str, cache_dir: Path) -> Path:
    """
    Generate the filepath where a ticker's data should be cached.
    
    Parameters
    ----------
    ticker : str
        The ticker symbol (e.g., 'SPY').
    cache_dir : Path
        The directory for cached files.
    
    Returns
    -------
    Path
        The full filepath (e.g., 'data/raw/SPY.csv').
    
    Example
    -------
    >>> _get_cache_filepath('SPY', Path('data/raw'))
    Path('data/raw/SPY.csv')
    """
    # a) ticker: str -- The ticker symbol as a string
    # b) cache_dir: Path -- The directory where files are stored
    # c) -> Path -- Returns a Path object
    # d) cache_dir / f"{ticker}.csv"

    return cache_dir / f"{ticker}.csv"


def _is_cache_valid(filepath: Path, max_age_days: int = 1) -> bool:
    """
    Check if a cached file exists and is recent enough to use.
    
    Parameters
    ----------
    filepath : Path
        Path to the cached file.
    max_age_days : int
        Maximum age in days before cache is considered stale.
    
    Returns
    -------
    bool
        True if cache exists and is fresh, False otherwise.
    
    Why Cache Expiry?
    -----------------
    Markets trade every day. If you downloaded data yesterday, you're missing
    today's prices. We set an expiry so that:
    - During development: Data is re-downloaded periodically to stay current
    - During experiments: You can set a longer expiry to avoid re-downloads
    
    During final experiments, I would want to download once and set
    max_age_days to a very large number (like 365).
    """
    # Step 1: Check if file exists at all
    if not filepath.exists():
        return False
    
    # Step 2: Get when the file was last modified

    file_modified_time = datetime.fromtimestamp(filepath.stat().st_mtime)
    # filepath.stat() -- Gets fie metadata (size, time, permissions)
    # st_mtime -- The modification time as a Unix timestamp
    # datetime.foretimestamp() -- Converts that number to a readable datetime	
    
    # Step 3: Calculate age. Subtracting two datetimes gives a timedelta duration.
    age = datetime.now() - file_modified_time
    
    # Step 4: Compare to maximum allowed age
    return age < timedelta(days=max_age_days)


# =============================================================================
# CORE DOWNLOAD FUNCTIONS
# =============================================================================

def download_single_ticker(
    ticker: str,
    start_date: str,
    end_date: str,
    progress: bool = False
) -> Optional[pd.DataFrame]:
    """
    Download OHLCV data for a single ticker from Yahoo Finance.
    
    This is the lowest-level download function. It directly calls the
    yfinance library and handles any errors that might occur.
    
    Parameters
    ----------
    ticker : str
        The ticker symbol to download (e.g., 'SPY', 'AAPL').
    start_date : str
        Start date in 'YYYY-MM-DD' format.
    end_date : str
        End date in 'YYYY-MM-DD' format.
    progress : bool
        Whether to show a progress bar (default False for cleaner output).
    
    Returns
    -------
    Optional[pd.DataFrame]
        DataFrame with OHLCV data, or None if download failed.
        The DataFrame has a DatetimeIndex and columns:
        ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    
    How It Works
    ------------
    1. We call yf.download() with the ticker and date range
    2. yfinance sends a request to Yahoo's servers
    3. Yahoo returns the data (or an error)
    4. We validate that we got what we expected
    5. We return the DataFrame (or None if something went wrong)
    
    Error Handling
    --------------
    Many things can go wrong:
    - Network is down → yfinance raises an exception
    - Ticker doesn't exist → yfinance returns empty DataFrame
    - Yahoo's servers are slow → request times out
    
    Instead of crashing, we catch these errors and return None,
    letting the calling code decide what to do.
    
    """
    logger.info(f"Downloading {ticker} from {start_date} to {end_date}")
    
    try:
        #yf.download() is the main function from yfinance
        #It returns a pandas DataFrame with the requested data
        df = yf.download(
            tickers=ticker,
            start=start_date,
            end=end_date,
            progress=progress, #Would determine whether a progress bar would be hidden or shown.
            auto_adjust=False, 
            #Would determine if both Close and Adj Close columns are kept or not. 
            #If auto_adjust = True then only close would be kept, we would lose out on raw prices. 
            #Adj Close would show true prices from stock splits, dividends and/or capital gain distributions.  
        )
        if df.empty:
            logger.warning(f"No data returned for {ticker}. "
                          f"Check if the ticker symbol is correct.")
            return None
            #Check if we got any data. Return None if we fail to retrieve data.
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            #When you download ONE ticker, yfinance SHOULD return simple columns
            #[Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
            #But sometimes, it would return a MultiIndex (two-level columns)
            #[('Open', 'SPY'), ('High', 'SPY'), ('Low', 'SPY'), ...]
            #get_level_values(0) is used to extract all values from the first level of a MultiIndex.
            #The code would flatten it back to simple columns.
            #This happens as a byproduct of a quirk in yfinance where MultiIndex is sued to distinguish between multiple tickers, sometimes even being done for single tickers.

        
        # Validate that we have the expected columns
        expected_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
        missing_columns = set(expected_columns) - set(df.columns)
        #Don't return None here — partial data might still be useful
        #Here, we are calculating what the missing column would be using set() and subtraction. 
        #set() would convert the expected_columns array and the df.columns array into sets and find the missing column between the two.
        
        if missing_columns:
            logger.warning(f"Missing columns for {ticker}: {missing_columns}")
            # Don't return None here — partial data might still be useful
        
        logger.info(f"Successfully downloaded {len(df)} rows for {ticker}")
        return df
        
    except Exception as e:
        #We do not want ONE failed download to crash the entire 
        #program, if SPY fails but QQQ and TLT works then we should still get those.
        logger.error(f"Error downloading {ticker}: {str(e)}")
        return None


def download_multiple_tickers(
    tickers: List[str],
    start_date: str,
    end_date: str,
    progress: bool = True
) -> Dict[str, pd.DataFrame]:
    """
    Download OHLCV data for multiple tickers.
    
    This function downloads each ticker separately, which is slightly slower
    than downloading all at once, but gives us better error handling and
    cleaner data structures.

    Example
    -------
    >>> tickers = ['SPY', 'QQQ', 'TLT']
    >>> data = download_multiple_tickers(tickers, '2020-01-01', '2020-12-31')
    >>> for ticker, df in data.items():
    ...     print(f"{ticker}: {len(df)} days")
    SPY: 253 days
    QQQ: 253 days
    TLT: 253 days
    """
    results = {}
    #This dictionary would store successful downloads, it would start empty and add to it as each download succeeds.

    for i, ticker in enumerate(tickers):
        if progress:
            print(f"Downloading {ticker} ({i+1}/{len(tickers)})...")
        #This loop would loop through the ticker symbols while having the enumerate function giving both an index and a value in a loop. We need the index for progress reporting. 

        """"
            tickers = ['SPY', 'QQQ', 'TLT']

            # Without enumerate:
            for ticker in tickers:
                print(ticker)
            # Output: SPY, QQQ, TLT (no index)

            # With enumerate:
            for i, ticker in enumerate(tickers):
                print(f"{i}: {ticker}")
            # Output: 0: SPY, 1: QQQ, 2: TLT (with index)
        """
        df = download_single_ticker(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            progress=False  # Don't show individual progress bars from yfinance
        )
        """
            Parameters
            ----------
            tickers : List[str]
                List of ticker symbols to download.
            start_date : str
                Start date in 'YYYY-MM-DD' format.
            end_date : str
                End date in 'YYYY-MM-DD' format.
            progress : bool
                Whether to show progress information.
            
            Returns
            -------
            Dict[str, pd.DataFrame]
                Dictionary mapping ticker symbols to their DataFrames.
                Tickers that failed to download are not included.
        """
        if df is not None:
            results[ticker] = df
        else:
            logger.warning(f"Skipping {ticker} due to download failure")
        # Here we are calling our download_single_ticker function. 
        # What happens here is that once the function is called, the ticker's entire OHLCV information would be copied into the dataframe "df". 
        # This dataframe would they become the value in the key-pair value for that particular ticker in the dictionary "results". 
        # If the download is not successful maybe because of an incorrect ticker symbol then the function would return an error and move on to the next. 
    # Report summary
    successful = len(results)
    failed = len(tickers) - successful
    
    if failed > 0:
        logger.warning(f"Downloaded {successful}/{len(tickers)} tickers. "
                      f"{failed} failed.")
    else:
        logger.info(f"Successfully downloaded all {successful} tickers.")
    
    return results


# =============================================================================
# CACHING FUNCTIONS
# =============================================================================

def save_to_cache(
    data: Dict[str, pd.DataFrame],
    cache_dir: Union[str, Path] = DEFAULT_CACHE_DIR
) -> None:
    """
    Save downloaded data to CSV files for later use.
    
    Why Cache Data?
    ---------------
    1. **Speed**: Loading from disk is faster than downloading from internet
    2. **Reliability**: Works even when offline or if Yahoo is down
    3. **Reproducibility**: Ensures you're using the exact same data across runs
    4. **Courtesy**: Reduces load on Yahoo's servers (too many requests = blocked)
    
    Parameters
    ----------
    data : Dict[str, pd.DataFrame]
        Dictionary mapping ticker symbols to DataFrames.
    cache_dir : Union[str, Path]
        Directory to save files in.
    
    File Format
    -----------
    We use CSV because it's:
    - Human-readable (you can open it in Excel)
    - Universal (any tool can read it)
    - Simple (no special software needed)
    
    For larger datasets, you might use Parquet (faster, smaller files),
    but CSV is fine for our purposes.
    
    Example
    -------
    >>> data = download_multiple_tickers(['SPY', 'QQQ'], '2020-01-01', '2020-12-31')
    >>> save_to_cache(data)
    # Creates data/raw/SPY.csv and data/raw/QQQ.csv
    """
    cache_dir = Path(cache_dir)
    _ensure_directory_exists(cache_dir)
    
    for ticker, df in data.items():
        filepath = _get_cache_filepath(ticker, cache_dir)
        
        # index=True saves the DatetimeIndex as a column
        # This is important — we need to recover the dates when loading!
        df.to_csv(filepath, index=True)
        
        logger.info(f"Saved {ticker} to {filepath}")


def load_from_cache(
    tickers: List[str],
    cache_dir: Union[str, Path] = DEFAULT_CACHE_DIR
) -> Dict[str, pd.DataFrame]:
    """
    Load previously downloaded data from CSV files.
    
    Parameters
    ----------
    tickers : List[str]
        List of ticker symbols to load.
    cache_dir : Union[str, Path]
        Directory where cached files are stored.
    
    Returns
    -------
    Dict[str, pd.DataFrame]
        Dictionary mapping ticker symbols to DataFrames.
        Tickers without cached data are not included.
    
    Notes
    -----
    When loading, we need to:
    1. Parse the date column back to datetime
    2. Set it as the index
    3. Ensure the index is a proper DatetimeIndex
    
    Example
    -------
    >>> data = load_from_cache(['SPY', 'QQQ'])
    >>> spy = data.get('SPY')  # Returns None if not in cache
    """
    cache_dir = Path(cache_dir)
    results = {}
    
    for ticker in tickers:
        filepath = _get_cache_filepath(ticker, cache_dir)
        
        if filepath.exists():
            try:
                # Read CSV with the first column as index (the dates)
                # parse_dates=True converts the index to DatetimeIndex
                df = pd.read_csv(
                    filepath,
                    index_col=0,
                    parse_dates=True
                )
                
                results[ticker] = df
                logger.info(f"Loaded {ticker} from cache ({len(df)} rows)")
                
            except Exception as e:
                logger.error(f"Error loading {ticker} from cache: {str(e)}")
        else:
            logger.debug(f"No cache found for {ticker}")
    
    return results


# =============================================================================
# MAIN PUBLIC FUNCTION
# =============================================================================

def download_ohlcv_data(
    tickers: List[str],
    start_date: str,
    end_date: str,
    cache_dir: Union[str, Path] = DEFAULT_CACHE_DIR,
    use_cache: bool = True,
    cache_expiry_days: int = DEFAULT_CACHE_EXPIRY_DAYS,
    force_download: bool = False
) -> Dict[str, pd.DataFrame]:
    """
    Download OHLCV data with intelligent caching.
    
    This is the main function you should use. It:
    1. Checks if data is already cached and recent
    2. Downloads only what's missing or stale
    3. Saves new downloads to cache
    4. Returns all requested data
    
    Parameters
    ----------
    tickers : List[str]
        List of ticker symbols to download.
        Example: ['SPY', 'QQQ', 'TLT', 'GLD', 'BIL']
    
    start_date : str
        Start date in 'YYYY-MM-DD' format.
        Example: '2010-01-01'
    
    end_date : str
        End date in 'YYYY-MM-DD' format.
        Example: '2023-12-31'
    
    cache_dir : Union[str, Path]
        Directory for cached files.
        Default: 'data/raw'
    
    use_cache : bool
        Whether to use cached data. Set to False to always download fresh.
        Default: True
    
    cache_expiry_days : int
        How many days before cached data is considered stale.
        Default: 1 (re-download daily)
        Set higher (e.g., 365) during final experiments for reproducibility.
    
    force_download : bool
        If True, ignore cache and download everything fresh.
        Default: False
    
    Returns
    -------
    Dict[str, pd.DataFrame]
        Dictionary mapping ticker symbols to DataFrames.
        
        Each DataFrame has:
        - DatetimeIndex (the trading dates)
        - Columns: Open, High, Low, Close, Adj Close, Volume
    
    Examples
    --------
    Basic usage:
    >>> data = download_ohlcv_data(
    ...     tickers=['SPY', 'QQQ'],
    ...     start_date='2020-01-01',
    ...     end_date='2023-12-31'
    ... )
    >>> spy = data['SPY']
    >>> print(spy.head())
    
    Force fresh download:
    >>> data = download_ohlcv_data(
    ...     tickers=['SPY'],
    ...     start_date='2020-01-01',
    ...     end_date='2023-12-31',
    ...     force_download=True
    ... )
    
    Long cache for reproducibility:
    >>> data = download_ohlcv_data(
    ...     tickers=['SPY'],
    ...     start_date='2020-01-01',
    ...     end_date='2023-12-31',
    ...     cache_expiry_days=365  # Only re-download after a year
    ... )
    """
    cache_dir = Path(cache_dir)
    
    # Determine which tickers we need to download
    if force_download or not use_cache:
        # Download everything fresh
        tickers_to_download = tickers
        cached_data = {}
    else:
        # Check what's in cache and what's expired
        cached_data = {}
        tickers_to_download = []
        
        for ticker in tickers:
            filepath = _get_cache_filepath(ticker, cache_dir)
            
            if _is_cache_valid(filepath, cache_expiry_days):
                # Cache is valid — we'll load it
                logger.debug(f"{ticker}: Using cached data")
            else:
                # Cache is missing or stale — need to download
                tickers_to_download.append(ticker)
                if filepath.exists():
                    logger.debug(f"{ticker}: Cache expired, will re-download")
                else:
                    logger.debug(f"{ticker}: Not in cache, will download")
        
        # Load valid cached data
        if tickers and not tickers_to_download:
            # All data is in cache
            logger.info("All data found in cache")
        
        cached_tickers = [t for t in tickers if t not in tickers_to_download]
        if cached_tickers:
            cached_data = load_from_cache(cached_tickers, cache_dir)
    
    # Download any missing/stale tickers
    if tickers_to_download:
        print(f"\nDownloading {len(tickers_to_download)} ticker(s) from Yahoo Finance...")
        print(f"Date range: {start_date} to {end_date}\n")
        
        downloaded_data = download_multiple_tickers(
            tickers=tickers_to_download,
            start_date=start_date,
            end_date=end_date,
            progress=True
        )
        
        # Save newly downloaded data to cache
        if downloaded_data and use_cache:
            save_to_cache(downloaded_data, cache_dir)
    else:
        downloaded_data = {}
    
    # Combine cached and downloaded data
    all_data = {**cached_data, **downloaded_data}
    
    # Report what we ended up with
    print(f"\n{'='*50}")
    print(f"Data Summary")
    print(f"{'='*50}")
    
    for ticker in tickers:
        if ticker in all_data:
            df = all_data[ticker]
            print(f"{ticker}: {len(df):,} trading days "
                  f"({df.index.min().date()} to {df.index.max().date()})")
        else:
            print(f"{ticker}: FAILED TO LOAD")
    
    print(f"{'='*50}\n")
    
    return all_data


# =============================================================================
# CONVENIENCE FUNCTIONS FOR YOUR PROJECT
# =============================================================================

def download_project_data(
    cache_dir: Union[str, Path] = DEFAULT_CACHE_DIR,
    force_download: bool = False
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Download all data needed for this project (training, validation, test sets).
    
    This is a convenience function that downloads data for all three periods
    defined in your project specification.
    
    Parameters
    ----------
    cache_dir : Union[str, Path]
        Directory for cached files.
    force_download : bool
        If True, ignore cache and download everything fresh.
    
    Returns
    -------
    Dict[str, Dict[str, pd.DataFrame]]
        Nested dictionary:
        {
            'training': {'SPY': df, 'QQQ': df, ...},
            'validation': {'SPY': df, 'QQQ': df, ...},
            'testing': {'SPY': df, 'QQQ': df, ...}
        }
    
    Example
    -------
    >>> all_data = download_project_data()
    >>> training_spy = all_data['training']['SPY']
    >>> test_spy = all_data['testing']['SPY']
    """
    # Your 5 assets from the project specification
    TICKERS = ['SPY', 'QQQ', 'TLT', 'GLD', 'BIL']
    
    # Date ranges from your specification
    PERIODS = {
        'training': ('2010-01-01', '2019-12-31'),
        'validation': ('2020-01-01', '2021-12-31'),
        'testing': ('2022-01-01', '2023-12-31'),
    }
    
    all_data = {}
    
    for period_name, (start_date, end_date) in PERIODS.items():
        print(f"\n{'#'*60}")
        print(f"# Downloading {period_name.upper()} data")
        print(f"# {start_date} to {end_date}")
        print(f"{'#'*60}")
        
        # Create a subdirectory for each period
        period_cache_dir = Path(cache_dir) / period_name
        
        all_data[period_name] = download_ohlcv_data(
            tickers=TICKERS,
            start_date=start_date,
            end_date=end_date,
            cache_dir=period_cache_dir,
            force_download=force_download
        )
    
    return all_data


def get_combined_dataframe(
    data: Dict[str, pd.DataFrame],
    column: str = 'Adj Close'
) -> pd.DataFrame:
    """
    Combine multiple tickers into a single DataFrame.
    
    This is useful when you want to see all prices side by side,
    or calculate correlations between assets.
    
    Parameters
    ----------
    data : Dict[str, pd.DataFrame]
        Dictionary mapping tickers to DataFrames (from download_ohlcv_data).
    column : str
        Which column to extract from each ticker's DataFrame.
        Default: 'Adj Close' (recommended for return calculations)
    
    Returns
    -------
    pd.DataFrame
        DataFrame with dates as index and tickers as columns.
    
    Example
    -------
    >>> data = download_ohlcv_data(['SPY', 'QQQ', 'TLT'], '2020-01-01', '2023-12-31')
    >>> prices = get_combined_dataframe(data)
    >>> print(prices.head())
                     SPY        QQQ        TLT
    Date
    2020-01-02  324.87    215.45    138.20
    2020-01-03  322.41    213.01    139.87
    ...
    
    >>> # Calculate correlation matrix
    >>> returns = prices.pct_change().dropna()
    >>> print(returns.corr())
    """
    # Extract the specified column from each ticker's DataFrame
    combined = pd.DataFrame({
        ticker: df[column]
        for ticker, df in data.items()
        if column in df.columns
    })
    
    # Sort columns alphabetically for consistency
    combined = combined.sort_index(axis=1)
    
    return combined


# =============================================================================
# MODULE EXPORTS
# =============================================================================
# __all__ defines what gets imported when someone does:
# from src.data.downloader import *

__all__ = [
    'download_ohlcv_data',
    'download_single_ticker',
    'download_multiple_tickers',
    'download_project_data',
    'get_combined_dataframe',
    'save_to_cache',
    'load_from_cache',
]


# =============================================================================
# SCRIPT EXECUTION
# =============================================================================
# This block runs only when you execute the file directly:
# python src/data/downloader.py
#
# It does NOT run when you import the module:
# from src.data.downloader import download_ohlcv_data

if __name__ == "__main__":
    # Set up basic logging so we can see what's happening
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("="*60)
    print("Portfolio RL - Data Downloader")
    print("="*60)
    print()
    print("This will download all required data for the project.")
    print("Data will be saved to data/raw/")
    print()
    
    # Download all project data
    all_data = download_project_data()
    
    print("\nDownload complete!")
    print("\nTo use this data in your code:")
    print(">>> from src.data.downloader import download_ohlcv_data")
    print(">>> data = download_ohlcv_data(['SPY'], '2020-01-01', '2023-12-31')")