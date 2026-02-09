#!/usr/bin/env python3
"""
Download Data Script
====================

This script downloads all the financial data needed for the project.

Usage
-----
From the project root directory:
    python scripts/download_data.py

Options:
    --force     Force re-download even if cached data exists

What It Downloads
-----------------
- SPY (S&P 500 ETF)
- QQQ (Nasdaq 100 ETF)
- TLT (20+ Year Treasury Bond ETF)
- GLD (Gold ETF)
- BIL (1-3 Month T-Bill ETF)

For three time periods:
- Training: 2010-01-01 to 2019-12-31
- Validation: 2020-01-01 to 2021-12-31
- Testing: 2022-01-01 to 2023-12-31

Output
------
Data is saved to:
    data/raw/training/
    data/raw/validation/
    data/raw/testing/
"""

import sys
import argparse
import logging
from pathlib import Path

# Add the project root to Python path so we can import our modules
# This is necessary when running scripts from the scripts/ folder
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Now we can import our downloader
from src.data.downloader import download_project_data, download_ohlcv_data


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for the script."""
    level = logging.DEBUG if verbose else logging.INFO
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )


def main():
    """Main entry point for the download script."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Download financial data for the Portfolio RL project.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--force', '-f',
        action='store_true',
        help='Force re-download even if cached data exists'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Show detailed logging output'
    )
    
    parser.add_argument(
        '--tickers',
        nargs='+',
        default=None,
        help='Specific tickers to download (default: all project tickers)'
    )
    
    parser.add_argument(
        '--start-date',
        default=None,
        help='Start date (YYYY-MM-DD) for custom date range'
    )
    
    parser.add_argument(
        '--end-date',
        default=None,
        help='End date (YYYY-MM-DD) for custom date range'
    )
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(args.verbose)
    
    # Print header
    print()
    print("=" * 60)
    print("Portfolio RL - Data Downloader")
    print("=" * 60)
    print()
    
    # Determine what to download
    if args.tickers and args.start_date and args.end_date:
        # Custom download
        print(f"Downloading custom data:")
        print(f"  Tickers: {args.tickers}")
        print(f"  Period: {args.start_date} to {args.end_date}")
        print()
        
        data = download_ohlcv_data(
            tickers=args.tickers,
            start_date=args.start_date,
            end_date=args.end_date,
            force_download=args.force
        )
        
    else:
        # Download all project data
        print("Downloading all project data:")
        print("  Assets: SPY, QQQ, TLT, GLD, BIL")
        print("  Periods: Training (2010-2019), Validation (2020-2021), Testing (2022-2023)")
        print()
        
        if args.force:
            print("  [Force mode: ignoring cached data]")
            print()
        
        data = download_project_data(force_download=args.force)
    
    # Print summary
    print()
    print("=" * 60)
    print("Download Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()