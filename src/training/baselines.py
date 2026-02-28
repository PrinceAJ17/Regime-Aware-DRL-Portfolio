"""
Baseline Strategies
===================

Non-RL baseline strategies for comparison with trained agents.

Implements:
- buy_and_hold: 100% SPY
- equal_weight: 20% each asset, monthly rebalance
- sixty_forty: 60% SPY / 40% TLT, monthly rebalance
- momentum: top 2 assets by trailing 12-1 month return, monthly rebalance

Author: Prince Alexander John
Date: February 2026
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict

from src.utils.metrics import calculate_all_metrics


def run_all_baselines(
    price_data: Dict[str, pd.DataFrame],
    output_path: Path = Path("results/experiments/baselines.json"),
) -> Dict[str, dict]:
    """Run all baselines, save results, and return metrics."""
    results = {
        "buy_and_hold": run_buy_and_hold(price_data),
        "equal_weight": run_equal_weight(price_data),
        "sixty_forty": run_sixty_forty(price_data),
        "momentum": run_momentum(price_data),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=_serialise)

    return results


def _get_close(df: pd.DataFrame) -> pd.Series:
    """Get close price series from a DataFrame."""
    for col in ["Adj Close", "Close", "close"]:
        if col in df.columns:
            return df[col]
    return df.iloc[:, 0]


def _align_prices(price_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Align all tickers to common dates, return close prices DataFrame."""
    closes = {}
    for ticker, df in price_data.items():
        closes[ticker] = _get_close(df)
    aligned = pd.DataFrame(closes).dropna()
    return aligned


def run_buy_and_hold(price_data: Dict[str, pd.DataFrame]) -> dict:
    """100% SPY, no rebalancing."""
    spy_close = _get_close(price_data["SPY"])
    returns = spy_close.pct_change().dropna().values
    metrics = calculate_all_metrics(returns)
    metrics["strategy"] = "buy_and_hold"
    return metrics


def run_equal_weight(price_data: Dict[str, pd.DataFrame]) -> dict:
    """20% each asset, monthly rebalance."""
    prices = _align_prices(price_data)
    returns = prices.pct_change().dropna()

    n_assets = len(prices.columns)
    weights = np.ones(n_assets) / n_assets

    # Monthly rebalance: reset weights at month boundaries
    portfolio_returns = []
    current_weights = weights.copy()

    for i, (date, row) in enumerate(returns.iterrows()):
        daily_returns = row.values
        port_ret = np.sum(current_weights * daily_returns)
        portfolio_returns.append(port_ret)

        # Update weights based on price movement
        current_weights = current_weights * (1 + daily_returns)
        current_weights = current_weights / current_weights.sum()

        # Rebalance at month boundaries
        if i + 1 < len(returns):
            next_date = returns.index[i + 1]
            if next_date.month != date.month:
                current_weights = weights.copy()

    metrics = calculate_all_metrics(np.array(portfolio_returns))
    metrics["strategy"] = "equal_weight"
    return metrics


def run_sixty_forty(price_data: Dict[str, pd.DataFrame]) -> dict:
    """60% SPY, 40% TLT, monthly rebalance."""
    prices = _align_prices({"SPY": price_data["SPY"], "TLT": price_data["TLT"]})
    returns = prices.pct_change().dropna()

    target_weights = np.array([0.6, 0.4])
    portfolio_returns = []
    current_weights = target_weights.copy()

    for i, (date, row) in enumerate(returns.iterrows()):
        daily_returns = row.values
        port_ret = np.sum(current_weights * daily_returns)
        portfolio_returns.append(port_ret)

        current_weights = current_weights * (1 + daily_returns)
        current_weights = current_weights / current_weights.sum()

        if i + 1 < len(returns):
            next_date = returns.index[i + 1]
            if next_date.month != date.month:
                current_weights = target_weights.copy()

    metrics = calculate_all_metrics(np.array(portfolio_returns))
    metrics["strategy"] = "sixty_forty"
    return metrics


def run_momentum(price_data: Dict[str, pd.DataFrame]) -> dict:
    """Top 2 assets by trailing 12-1 month return, monthly rebalance."""
    prices = _align_prices(price_data)
    returns = prices.pct_change().dropna()
    tickers = list(prices.columns)

    # Pre-compute trailing 12-1 month returns (skip most recent month)
    momentum_lookback = 252  # ~12 months
    skip_recent = 21  # ~1 month

    portfolio_returns = []
    current_weights = np.ones(len(tickers)) / len(tickers)
    last_rebalance_month = None

    for i in range(len(returns)):
        date = returns.index[i]
        daily_returns = returns.iloc[i].values

        # Portfolio return for today
        port_ret = np.sum(current_weights * daily_returns)
        portfolio_returns.append(port_ret)

        # Rebalance at month boundaries if enough history
        rebalance = False
        if i + 1 < len(returns):
            next_date = returns.index[i + 1]
            if next_date.month != date.month:
                rebalance = True

        if rebalance and i >= momentum_lookback:
            # Compute momentum scores
            scores = {}
            for j, ticker in enumerate(tickers):
                # 12-month return minus 1-month return
                start = max(0, i - momentum_lookback)
                end = i - skip_recent
                if end > start:
                    cumret = (1 + returns.iloc[start:end, j]).prod() - 1
                    scores[j] = cumret
                else:
                    scores[j] = 0.0

            # Pick top 2
            sorted_assets = sorted(scores, key=scores.get, reverse=True)
            top2 = sorted_assets[:2]

            current_weights = np.zeros(len(tickers))
            for idx in top2:
                current_weights[idx] = 0.5
        elif not rebalance:
            # Drift weights
            current_weights = current_weights * (1 + daily_returns)
            if current_weights.sum() > 0:
                current_weights = current_weights / current_weights.sum()

    metrics = calculate_all_metrics(np.array(portfolio_returns))
    metrics["strategy"] = "momentum"
    return metrics


def _serialise(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Not serialisable: {type(obj)}")
