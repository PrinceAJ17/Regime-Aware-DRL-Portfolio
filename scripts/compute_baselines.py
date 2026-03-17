"""
Compute Baseline Portfolio Metrics
==================================

Computes performance metrics for standard benchmark portfolios on the test period (2022-2023).
These serve as comparison points for RL agent performance.

Baselines:
1. SPY Buy-and-Hold: 100% allocation to SPY
2. Equal-Weight: 20% each to SPY, QQQ, TLT, GLD, BIL, rebalanced monthly
3. 60/40 Portfolio: 60% SPY, 40% TLT, rebalanced monthly
4. Risk Parity (simple): Inverse-volatility weighted, rebalanced monthly

Output: results/baselines.json

Usage:
    python scripts/compute_baselines.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.downloader import download_ohlcv_data
from src.utils.metrics import calculate_all_metrics


TICKERS = ["SPY", "QQQ", "TLT", "GLD", "BIL"]
# Test period dates - matches agent evaluation period
# After the warmup buffer fix, agents now download test data from 2021-08-01
# but start evaluation at 2022-01-01 (via start_idx), so baselines should match.
TEST_START = "2022-01-03"  # First trading day of 2022 - matches agent test period!
TEST_END = "2023-12-29"    # Last trading day of 2023
INITIAL_CAPITAL = 1_000_000
TRANSACTION_COST = 0.001  # 10 bps


def load_test_data():
    """Load test period price data (same dates as agent evaluation)."""
    print(f"Loading test data: {TEST_START} to {TEST_END}")
    print("NOTE: Dates match agent evaluation period (2022-01-01 to 2023-12-31)")
    price_data = download_ohlcv_data(
        TICKERS, TEST_START, TEST_END,
        cache_dir="data/raw/baselines",  # Separate cache for baselines
        cache_expiry_days=365,
        quiet=True
    )
    return price_data


def compute_returns(price_data: dict) -> pd.DataFrame:
    """Compute daily returns for all tickers."""
    closes = {}
    for ticker, df in price_data.items():
        closes[ticker] = df["Close"]

    prices_df = pd.DataFrame(closes)
    returns_df = prices_df.pct_change().dropna()
    return returns_df


def simulate_portfolio(returns_df: pd.DataFrame, weights: np.ndarray,
                       rebalance_freq: str = None) -> dict:
    """
    Simulate a portfolio with given weights.

    Parameters
    ----------
    returns_df : pd.DataFrame
        Daily returns for each asset.
    weights : np.ndarray
        Target portfolio weights (must sum to 1).
    rebalance_freq : str, optional
        Rebalancing frequency: 'M' for monthly, 'W' for weekly, None for buy-and-hold.

    Returns
    -------
    dict
        Portfolio metrics and daily values.
    """
    n_days = len(returns_df)
    portfolio_values = np.zeros(n_days + 1)
    portfolio_values[0] = INITIAL_CAPITAL

    daily_returns = []
    current_weights = weights.copy()

    # Track rebalancing days
    if rebalance_freq:
        # Use 'ME' (month-end) instead of deprecated 'M'
        freq = 'ME' if rebalance_freq == 'M' else rebalance_freq
        rebalance_dates = returns_df.resample(freq).last().index
    else:
        rebalance_dates = []

    total_transaction_costs = 0.0

    for i, (date, row) in enumerate(returns_df.iterrows()):
        asset_returns = row.values

        # Portfolio return for the day
        portfolio_return = np.sum(current_weights * asset_returns)
        portfolio_values[i + 1] = portfolio_values[i] * (1 + portfolio_return)
        daily_returns.append(portfolio_return)

        # Update weights based on asset price changes (drift)
        new_values = current_weights * (1 + asset_returns)
        current_weights = new_values / np.sum(new_values)

        # Rebalance if needed
        if rebalance_freq and date in rebalance_dates:
            # Transaction cost = sum of absolute weight changes
            weight_changes = np.abs(weights - current_weights)
            cost_pct = np.sum(weight_changes) * TRANSACTION_COST
            cost_dollar = portfolio_values[i + 1] * cost_pct
            portfolio_values[i + 1] -= cost_dollar
            total_transaction_costs += cost_dollar
            current_weights = weights.copy()

    # Compute metrics
    daily_returns = np.array(daily_returns)
    metrics = calculate_all_metrics(daily_returns)

    # Add additional info
    metrics["initial_value"] = INITIAL_CAPITAL
    metrics["final_value"] = portfolio_values[-1]
    metrics["total_transaction_costs"] = total_transaction_costs
    metrics["n_days"] = n_days

    return {
        "metrics": metrics,
        "daily_values": portfolio_values.tolist(),
        "daily_returns": daily_returns.tolist(),
    }


def compute_risk_parity_weights(returns_df: pd.DataFrame, lookback: int = 60) -> np.ndarray:
    """Compute inverse-volatility weights based on recent returns."""
    recent_returns = returns_df.iloc[-lookback:]
    vols = recent_returns.std()
    inv_vols = 1 / vols
    weights = inv_vols / inv_vols.sum()
    return weights.values


def main():
    # Load data
    price_data = load_test_data()
    returns_df = compute_returns(price_data)

    print(f"Test period: {returns_df.index[0].date()} to {returns_df.index[-1].date()}")
    print(f"Trading days: {len(returns_df)}")
    print()

    baselines = {}

    # 1. SPY Buy-and-Hold
    print("Computing SPY Buy-and-Hold...")
    spy_weights = np.array([1.0, 0.0, 0.0, 0.0, 0.0])  # 100% SPY
    spy_result = simulate_portfolio(returns_df, spy_weights, rebalance_freq=None)
    baselines["spy_buy_hold"] = {
        "name": "SPY Buy-and-Hold",
        "weights": dict(zip(TICKERS, spy_weights.tolist())),
        "rebalance": "None",
        **spy_result,
    }
    print(f"  Sharpe: {spy_result['metrics']['sharpe_ratio']:.3f}, "
          f"Return: {spy_result['metrics']['cumulative_return']:.1%}, "
          f"MDD: {spy_result['metrics']['max_drawdown']:.1%}")

    # 2. Equal-Weight (monthly rebalance)
    print("Computing Equal-Weight...")
    equal_weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    equal_result = simulate_portfolio(returns_df, equal_weights, rebalance_freq="M")
    baselines["equal_weight"] = {
        "name": "Equal-Weight (Monthly)",
        "weights": dict(zip(TICKERS, equal_weights.tolist())),
        "rebalance": "Monthly",
        **equal_result,
    }
    print(f"  Sharpe: {equal_result['metrics']['sharpe_ratio']:.3f}, "
          f"Return: {equal_result['metrics']['cumulative_return']:.1%}, "
          f"MDD: {equal_result['metrics']['max_drawdown']:.1%}")

    # 3. 60/40 Portfolio (monthly rebalance)
    print("Computing 60/40 Portfolio...")
    sixty_forty_weights = np.array([0.6, 0.0, 0.4, 0.0, 0.0])  # 60% SPY, 40% TLT
    sixty_forty_result = simulate_portfolio(returns_df, sixty_forty_weights, rebalance_freq="M")
    baselines["sixty_forty"] = {
        "name": "60/40 SPY/TLT (Monthly)",
        "weights": dict(zip(TICKERS, sixty_forty_weights.tolist())),
        "rebalance": "Monthly",
        **sixty_forty_result,
    }
    print(f"  Sharpe: {sixty_forty_result['metrics']['sharpe_ratio']:.3f}, "
          f"Return: {sixty_forty_result['metrics']['cumulative_return']:.1%}, "
          f"MDD: {sixty_forty_result['metrics']['max_drawdown']:.1%}")

    # 4. Risk Parity (simple inverse-vol, monthly rebalance)
    print("Computing Risk Parity...")
    # Use first 60 days to compute initial weights, then rebalance monthly
    rp_weights = compute_risk_parity_weights(returns_df, lookback=60)
    rp_result = simulate_portfolio(returns_df, rp_weights, rebalance_freq="M")
    baselines["risk_parity"] = {
        "name": "Risk Parity (Monthly)",
        "weights": dict(zip(TICKERS, rp_weights.tolist())),
        "rebalance": "Monthly",
        **rp_result,
    }
    print(f"  Sharpe: {rp_result['metrics']['sharpe_ratio']:.3f}, "
          f"Return: {rp_result['metrics']['cumulative_return']:.1%}, "
          f"MDD: {rp_result['metrics']['max_drawdown']:.1%}")

    # Save results
    output_path = PROJECT_ROOT / "results" / "baselines.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert numpy types for JSON
    def convert_numpy(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Not JSON serialisable: {type(obj)}")

    with open(output_path, "w") as f:
        json.dump(baselines, f, indent=2, default=convert_numpy)

    print(f"\nResults saved to: {output_path}")

    # Summary table
    print("\n" + "=" * 70)
    print(f"BASELINE COMPARISON (Test Period: {TEST_START} to {TEST_END})")
    print("=" * 70)
    print(f"{'Strategy':<30} {'Sharpe':>10} {'Return':>12} {'MDD':>10} {'Sortino':>10}")
    print("-" * 70)

    for key, data in baselines.items():
        m = data["metrics"]
        print(f"{data['name']:<30} {m['sharpe_ratio']:>10.3f} "
              f"{m['cumulative_return']:>11.1%} {m['max_drawdown']:>10.1%} "
              f"{m.get('sortino_ratio', 0):>10.3f}")

    print("=" * 70)


if __name__ == "__main__":
    main()
