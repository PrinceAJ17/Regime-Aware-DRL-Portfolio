"""
Deep Regime Analysis (E2) - Extracting Regime-Level Insights from E1 Data
==========================================================================

Analyzes the 100 completed E1 runs to extract per-regime performance metrics,
identify bear market champions, and analyze regime transition behavior.

Key Questions Answered:
1. Which config handles bear markets best?
2. Do regime-aware strategies excel in their target regimes?
3. Is there a consistency vs performance trade-off?
4. How do agents adapt at regime transitions?

Author: Prince Alexander John
Date: March 2026
"""

import json
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Constants
RESULTS_DIR = Path("results/experiments/replay_comparison")
FIGURES_DIR = Path("results/figures")
TABLES_DIR = Path("results/tables")
ASSETS = ["SPY", "QQQ", "TLT", "GLD", "BIL"]
ALGORITHMS = ["ppo", "sac", "td3", "rainbow"]
STRATEGIES = ["standard", "balanced", "current_focused", "inverse_frequency", "adaptive"]
SEEDS = [42, 123, 456, 789, 1000]
REGIME_NAMES = {0: "Bull", 1: "Bear", 2: "Sideways", "bull": "Bull", "bear": "Bear", "sideways": "Sideways"}
TRADING_DAYS_PER_YEAR = 252
MIN_DAYS_PER_REGIME = 10  # Minimum days for reliable per-regime metrics


def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Compute Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return np.nan
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    if pooled_std == 0:
        return np.nan
    return (np.mean(group1) - np.mean(group2)) / pooled_std


def effect_size_label(d: float) -> str:
    """Interpret Cohen's d magnitude."""
    if np.isnan(d):
        return "N/A"
    abs_d = abs(d)
    if abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    else:
        return "large"


def normalize_regime(r) -> str:
    """Convert regime label to lowercase string."""
    if isinstance(r, str):
        return r.lower()
    return {0: "bull", 1: "bear", 2: "sideways"}.get(r, "unknown")


def calculate_regime_sharpe(returns: np.ndarray) -> float:
    """Calculate annualized Sharpe from regime-specific returns."""
    if len(returns) < 5:  # Need minimum observations
        return np.nan

    mean_return = np.mean(returns)
    std_return = np.std(returns, ddof=1)

    if std_return == 0:
        return 0.0

    # Annualize: scale by sqrt(252) for volatility, 252 for return
    annualized_return = mean_return * TRADING_DAYS_PER_YEAR
    annualized_vol = std_return * np.sqrt(TRADING_DAYS_PER_YEAR)

    return annualized_return / annualized_vol


def calculate_regime_mdd(portfolio_values: np.ndarray) -> float:
    """Calculate max drawdown within a regime period."""
    if len(portfolio_values) < 2:
        return np.nan

    running_max = np.maximum.accumulate(portfolio_values)
    drawdowns = (running_max - portfolio_values) / running_max
    return np.max(drawdowns)


def calculate_cumulative_return(portfolio_values: np.ndarray) -> float:
    """Calculate cumulative return from portfolio values."""
    if len(portfolio_values) < 2:
        return np.nan
    return (portfolio_values[-1] / portfolio_values[0]) - 1


def calculate_win_rate(returns: np.ndarray) -> float:
    """Calculate percentage of positive days."""
    if len(returns) == 0:
        return np.nan
    return np.mean(returns > 0)


# =============================================================================
# SECTION 1: DATA LOADING
# =============================================================================

def load_all_episode_data() -> pd.DataFrame:
    """Load all 100 test_episode_data.json files into a DataFrame."""
    print("=" * 80)
    print("SECTION 1: DATA LOADING")
    print("=" * 80)

    rows = []
    loaded_count = 0
    missing_count = 0

    for algo in ALGORITHMS:
        for strategy in STRATEGIES:
            for seed in SEEDS:
                run_id = f"replay_comparison__{algo}__{strategy}__seed{seed}"
                path = RESULTS_DIR / algo / strategy / run_id / "test_episode_data.json"

                if not path.exists():
                    missing_count += 1
                    continue

                try:
                    with open(path) as f:
                        data = json.load(f)

                    # Extract data for each episode
                    portfolio_values = data.get("portfolio_values", [])
                    step_returns = data.get("step_returns", [])
                    regime_labels = data.get("regime_labels", [])
                    portfolio_weights = data.get("portfolio_weights", [])
                    dates = data.get("dates", [])

                    n_episodes = len(portfolio_values)

                    for ep_idx in range(n_episodes):
                        pv = np.array(portfolio_values[ep_idx])
                        sr = np.array(step_returns[ep_idx]) if ep_idx < len(step_returns) else np.array([])
                        rl = regime_labels[ep_idx] if ep_idx < len(regime_labels) else []
                        pw = np.array(portfolio_weights[ep_idx]) if ep_idx < len(portfolio_weights) else np.array([])
                        dt = dates[ep_idx] if ep_idx < len(dates) and isinstance(dates[0], list) else dates

                        # Normalize regime labels
                        rl_norm = [normalize_regime(r) for r in rl]

                        rows.append({
                            "algorithm": algo,
                            "strategy": strategy,
                            "seed": seed,
                            "episode": ep_idx,
                            "portfolio_values": pv,
                            "step_returns": sr,
                            "regime_labels": rl_norm,
                            "portfolio_weights": pw,
                            "dates": dt,
                            "n_days": len(pv)
                        })

                    loaded_count += 1

                except Exception as e:
                    print(f"  Error loading {path}: {e}")
                    missing_count += 1

    df = pd.DataFrame(rows)

    print(f"\nLoaded: {loaded_count}/100 runs ({len(df)} episodes)")
    print(f"Missing: {missing_count} runs")
    print(f"Algorithms: {df['algorithm'].unique().tolist()}")
    print(f"Strategies: {df['strategy'].unique().tolist()}")
    print(f"Seeds: {sorted(df['seed'].unique().tolist())}")
    print(f"Episodes per run: {df.groupby(['algorithm', 'strategy', 'seed']).size().iloc[0]}")

    return df


# =============================================================================
# SECTION 2: PER-REGIME METRICS CALCULATION
# =============================================================================

def calculate_per_regime_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate Sharpe, return, MDD, win rate per regime for each episode."""
    print("\n" + "=" * 80)
    print("SECTION 2: PER-REGIME METRICS CALCULATION")
    print("=" * 80)

    regime_metrics = []

    for idx, row in df.iterrows():
        pv = row["portfolio_values"]
        sr = row["step_returns"]
        rl = row["regime_labels"]

        # Skip if missing data
        if len(sr) == 0 or len(rl) == 0:
            continue

        # Align lengths (step_returns is 1 shorter than portfolio_values)
        min_len = min(len(sr), len(rl), len(pv) - 1)
        sr = sr[:min_len]
        rl = rl[:min_len]
        pv_aligned = pv[:min_len + 1]  # +1 for start value

        # Group returns by regime
        regime_returns = defaultdict(list)
        regime_pv_indices = defaultdict(list)

        for i, (ret, regime) in enumerate(zip(sr, rl)):
            regime_returns[regime].append(ret)
            regime_pv_indices[regime].append(i)

        # Calculate metrics per regime
        for regime in ["bull", "bear", "sideways"]:
            returns = np.array(regime_returns.get(regime, []))
            indices = regime_pv_indices.get(regime, [])

            if len(returns) < MIN_DAYS_PER_REGIME:
                continue

            # Get portfolio values for this regime (for MDD calculation)
            # Use contiguous segments
            pv_segments = []
            if indices:
                # Get portfolio values at indices (offset by 1 since returns are between days)
                pv_regime = [pv[i] for i in indices] + [pv[indices[-1] + 1]]
                pv_segments = np.array(pv_regime)

            sharpe = calculate_regime_sharpe(returns)
            cum_return = np.prod(1 + returns) - 1
            volatility = np.std(returns, ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)
            mdd = calculate_regime_mdd(pv_segments) if len(pv_segments) > 1 else np.nan
            win_rate = calculate_win_rate(returns)

            regime_metrics.append({
                "algorithm": row["algorithm"],
                "strategy": row["strategy"],
                "seed": row["seed"],
                "episode": row["episode"],
                "regime": regime,
                "n_days": len(returns),
                "sharpe": sharpe,
                "cumulative_return": cum_return,
                "volatility": volatility,
                "max_drawdown": mdd,
                "win_rate": win_rate
            })

    metrics_df = pd.DataFrame(regime_metrics)

    print(f"\nCalculated metrics for {len(metrics_df)} regime-episode combinations")
    print(f"Regimes present: {metrics_df['regime'].unique().tolist()}")

    # Summary stats
    regime_counts = metrics_df.groupby("regime")["n_days"].agg(["mean", "sum"])
    print("\nRegime day counts:")
    for regime, row in regime_counts.iterrows():
        print(f"  {regime.capitalize()}: mean {row['mean']:.1f} days/episode, total {int(row['sum'])} days")

    return metrics_df


# =============================================================================
# SECTION 3: AGGREGATE PERFORMANCE BY REGIME
# =============================================================================

def analyze_aggregate_regime_performance(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Overall regime difficulty analysis."""
    print("\n" + "=" * 80)
    print("SECTION 3: PERFORMANCE BY REGIME (AGGREGATE)")
    print("=" * 80)

    # Aggregate across all runs
    regime_summary = metrics_df.groupby("regime").agg({
        "sharpe": ["mean", "std", "count"],
        "cumulative_return": ["mean", "std"],
        "max_drawdown": ["mean", "std"],
        "win_rate": ["mean", "std"],
        "n_days": "mean"
    }).round(4)

    regime_summary.columns = ["_".join(col) for col in regime_summary.columns]
    regime_summary = regime_summary.reset_index()

    print("\n{:<10} {:>10} {:>8} {:>12} {:>8} {:>10} {:>8} {:>8}".format(
        "Regime", "Sharpe", "Std", "Return", "Std", "MDD", "WinRate", "Days"
    ))
    print("-" * 82)

    for _, row in regime_summary.sort_values("sharpe_mean", ascending=False).iterrows():
        print("{:<10} {:>+10.3f} {:>8.3f} {:>+12.2%} {:>8.2%} {:>10.1%} {:>7.1%} {:>8.1f}".format(
            row["regime"].capitalize(),
            row["sharpe_mean"], row["sharpe_std"],
            row["cumulative_return_mean"], row["cumulative_return_std"],
            row["max_drawdown_mean"],
            row["win_rate_mean"],
            row["n_days_mean"]
        ))

    # Save to CSV
    regime_summary.to_csv(TABLES_DIR / "e2_regime_aggregate.csv", index=False)

    return regime_summary


# =============================================================================
# SECTION 4: ALGORITHM × REGIME PERFORMANCE
# =============================================================================

def analyze_algorithm_regime_performance(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Which algorithm handles each regime best?"""
    print("\n" + "=" * 80)
    print("SECTION 4: ALGORITHM × REGIME PERFORMANCE")
    print("=" * 80)

    # Aggregate: episode -> run -> config means
    run_means = metrics_df.groupby(["algorithm", "strategy", "seed", "regime"]).agg({
        "sharpe": "mean",
        "cumulative_return": "mean",
        "max_drawdown": "mean"
    }).reset_index()

    # Then aggregate across seeds to get config-level means
    algo_regime = run_means.groupby(["algorithm", "regime"]).agg({
        "sharpe": ["mean", "std"],
        "cumulative_return": "mean",
        "max_drawdown": "mean"
    }).round(4)

    algo_regime.columns = ["_".join(col) for col in algo_regime.columns]
    algo_regime = algo_regime.reset_index()

    print("\n--- Mean Sharpe by Algorithm × Regime ---")
    print("\n{:<10}".format("Algorithm"), end="")
    for regime in ["bull", "bear", "sideways"]:
        print("{:>12}".format(regime.capitalize()), end="")
    print()
    print("-" * 46)

    for algo in ALGORITHMS:
        print("{:<10}".format(algo.upper()), end="")
        for regime in ["bull", "bear", "sideways"]:
            row = algo_regime[(algo_regime["algorithm"] == algo) & (algo_regime["regime"] == regime)]
            if len(row) > 0:
                print("{:>+12.3f}".format(row["sharpe_mean"].values[0]), end="")
            else:
                print("{:>12}".format("N/A"), end="")
        print()

    # Find bear market champion
    bear_data = algo_regime[algo_regime["regime"] == "bear"].sort_values("sharpe_mean", ascending=False)
    if len(bear_data) > 0:
        best_bear = bear_data.iloc[0]
        print(f"\n>>> BEAR MARKET CHAMPION: {best_bear['algorithm'].upper()} (Sharpe: {best_bear['sharpe_mean']:+.3f})")

    # Save to CSV
    algo_regime.to_csv(TABLES_DIR / "e2_regime_sharpe_by_algo.csv", index=False)

    return algo_regime


# =============================================================================
# SECTION 5: STRATEGY × REGIME PERFORMANCE
# =============================================================================

def analyze_strategy_regime_performance(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Do regime-aware strategies excel in specific regimes?"""
    print("\n" + "=" * 80)
    print("SECTION 5: STRATEGY × REGIME PERFORMANCE")
    print("=" * 80)

    # Aggregate to run level first
    run_means = metrics_df.groupby(["algorithm", "strategy", "seed", "regime"]).agg({
        "sharpe": "mean"
    }).reset_index()

    # Then aggregate across algorithms and seeds
    strat_regime = run_means.groupby(["strategy", "regime"]).agg({
        "sharpe": ["mean", "std", "count"]
    }).round(4)

    strat_regime.columns = ["_".join(col) for col in strat_regime.columns]
    strat_regime = strat_regime.reset_index()

    print("\n--- Mean Sharpe by Strategy × Regime ---")
    print("\n{:<18}".format("Strategy"), end="")
    for regime in ["bull", "bear", "sideways"]:
        print("{:>12}".format(regime.capitalize()), end="")
    print()
    print("-" * 54)

    for strat in STRATEGIES:
        print("{:<18}".format(strat), end="")
        for regime in ["bull", "bear", "sideways"]:
            row = strat_regime[(strat_regime["strategy"] == strat) & (strat_regime["regime"] == regime)]
            if len(row) > 0:
                print("{:>+12.3f}".format(row["sharpe_mean"].values[0]), end="")
            else:
                print("{:>12}".format("N/A"), end="")
        print()

    # Key question: Does inverse_frequency beat standard in bear markets?
    bear_data = strat_regime[strat_regime["regime"] == "bear"]
    if len(bear_data) > 0:
        inv_freq = bear_data[bear_data["strategy"] == "inverse_frequency"]["sharpe_mean"].values
        standard = bear_data[bear_data["strategy"] == "standard"]["sharpe_mean"].values
        if len(inv_freq) > 0 and len(standard) > 0:
            diff = inv_freq[0] - standard[0]
            print(f"\n>>> inverse_frequency vs standard in BEAR: {diff:+.3f} Sharpe ({'better' if diff > 0 else 'worse'})")

    # Save to CSV
    strat_regime.to_csv(TABLES_DIR / "e2_regime_sharpe_by_strategy.csv", index=False)

    return strat_regime


# =============================================================================
# SECTION 6: BEST CONFIG PER REGIME
# =============================================================================

def find_best_configs_per_regime(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Rank Algorithm × Strategy by Sharpe within each regime."""
    print("\n" + "=" * 80)
    print("SECTION 6: BEST CONFIG PER REGIME")
    print("=" * 80)

    # Aggregate to config level
    run_means = metrics_df.groupby(["algorithm", "strategy", "seed", "regime"]).agg({
        "sharpe": "mean"
    }).reset_index()

    config_regime = run_means.groupby(["algorithm", "strategy", "regime"]).agg({
        "sharpe": ["mean", "std", "count"]
    }).round(4)

    config_regime.columns = ["_".join(col) for col in config_regime.columns]
    config_regime = config_regime.reset_index()

    all_rankings = []

    for regime in ["bull", "bear", "sideways"]:
        print(f"\n--- TOP 5: {regime.upper()} MARKET ---")
        regime_data = config_regime[config_regime["regime"] == regime].sort_values("sharpe_mean", ascending=False)

        print("{:>4} {:<10} {:<18} {:>10} {:>8}".format("Rank", "Algorithm", "Strategy", "Sharpe", "Std"))
        print("-" * 56)

        for i, (_, row) in enumerate(regime_data.head(5).iterrows(), 1):
            print("{:>4} {:<10} {:<18} {:>+10.3f} {:>8.3f}".format(
                i, row["algorithm"].upper(), row["strategy"],
                row["sharpe_mean"], row["sharpe_std"]
            ))
            all_rankings.append({
                "regime": regime,
                "rank": i,
                "algorithm": row["algorithm"],
                "strategy": row["strategy"],
                "sharpe_mean": row["sharpe_mean"],
                "sharpe_std": row["sharpe_std"]
            })

    rankings_df = pd.DataFrame(all_rankings)
    rankings_df.to_csv(TABLES_DIR / "e2_best_config_per_regime.csv", index=False)

    return config_regime


# =============================================================================
# SECTION 7: CROSS-REGIME CONSISTENCY
# =============================================================================

def analyze_cross_regime_consistency(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate consistency score = 1 / variance(Sharpe across regimes)."""
    print("\n" + "=" * 80)
    print("SECTION 7: CROSS-REGIME CONSISTENCY")
    print("=" * 80)

    # Get config-level Sharpe per regime
    run_means = metrics_df.groupby(["algorithm", "strategy", "seed", "regime"]).agg({
        "sharpe": "mean"
    }).reset_index()

    config_regime = run_means.groupby(["algorithm", "strategy", "regime"]).agg({
        "sharpe": "mean"
    }).reset_index()

    # Pivot to get regimes as columns
    pivot = config_regime.pivot_table(
        index=["algorithm", "strategy"],
        columns="regime",
        values="sharpe"
    ).reset_index()

    # Calculate consistency metrics
    pivot["mean_sharpe"] = pivot[["bull", "bear", "sideways"]].mean(axis=1)
    pivot["sharpe_variance"] = pivot[["bull", "bear", "sideways"]].var(axis=1)
    pivot["sharpe_range"] = pivot[["bull", "bear", "sideways"]].max(axis=1) - pivot[["bull", "bear", "sideways"]].min(axis=1)

    # Consistency score (inverse variance, handle zero variance)
    pivot["consistency_score"] = 1 / (pivot["sharpe_variance"] + 0.01)

    # Sort by consistency
    consistent = pivot.sort_values("consistency_score", ascending=False)

    print("\n--- TOP 10 MOST CONSISTENT CONFIGS ---")
    print("{:>4} {:<10} {:<18} {:>10} {:>10} {:>10} {:>12}".format(
        "Rank", "Algorithm", "Strategy", "Bull", "Bear", "Sideways", "Consistency"
    ))
    print("-" * 80)

    for i, (_, row) in enumerate(consistent.head(10).iterrows(), 1):
        print("{:>4} {:<10} {:<18} {:>+10.3f} {:>+10.3f} {:>+10.3f} {:>12.2f}".format(
            i, row["algorithm"].upper(), row["strategy"],
            row.get("bull", np.nan), row.get("bear", np.nan), row.get("sideways", np.nan),
            row["consistency_score"]
        ))

    # Trade-off analysis
    print("\n--- CONSISTENCY vs OVERALL PERFORMANCE TRADE-OFF ---")
    pivot["overall_sharpe"] = pivot["mean_sharpe"]

    # Quadrants
    median_consistency = pivot["consistency_score"].median()
    median_sharpe = pivot["overall_sharpe"].median()

    high_both = pivot[(pivot["consistency_score"] > median_consistency) & (pivot["overall_sharpe"] > median_sharpe)]
    print(f"\nHigh Performance + High Consistency ({len(high_both)} configs):")
    for _, row in high_both.nlargest(5, "overall_sharpe").iterrows():
        print(f"  {row['algorithm'].upper()} {row['strategy']}: Sharpe={row['overall_sharpe']:+.3f}, Consistency={row['consistency_score']:.2f}")

    # Save to CSV
    pivot.to_csv(TABLES_DIR / "e2_consistency_rankings.csv", index=False)

    return pivot


# =============================================================================
# SECTION 8: STATISTICAL TESTS (REGIME-SPECIFIC)
# =============================================================================

def run_regime_statistical_tests(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Statistical tests for regime-specific comparisons."""
    print("\n" + "=" * 80)
    print("SECTION 8: STATISTICAL TESTS (REGIME-SPECIFIC)")
    print("=" * 80)
    print("\nBonferroni correction: alpha = 0.05 / 12 = 0.0042")

    # Aggregate to run level
    run_means = metrics_df.groupby(["algorithm", "strategy", "seed", "regime"]).agg({
        "sharpe": "mean"
    }).reset_index()

    test_results = []

    # Test 1: TD3 adaptive vs TD3 standard in BEAR markets
    print("\n--- Test 1: TD3 Adaptive vs Standard in BEAR Markets ---")
    td3_adapt_bear = run_means[(run_means["algorithm"] == "td3") &
                                (run_means["strategy"] == "adaptive") &
                                (run_means["regime"] == "bear")]["sharpe"].values
    td3_std_bear = run_means[(run_means["algorithm"] == "td3") &
                              (run_means["strategy"] == "standard") &
                              (run_means["regime"] == "bear")]["sharpe"].values

    if len(td3_adapt_bear) >= 2 and len(td3_std_bear) >= 2:
        t_stat, p_val = stats.ttest_ind(td3_adapt_bear, td3_std_bear, equal_var=False)
        d = cohens_d(td3_adapt_bear, td3_std_bear)
        sig = "YES" if p_val < 0.0042 else ("*" if p_val < 0.05 else "no")
        print(f"  TD3 adaptive: {np.mean(td3_adapt_bear):+.3f} (n={len(td3_adapt_bear)})")
        print(f"  TD3 standard: {np.mean(td3_std_bear):+.3f} (n={len(td3_std_bear)})")
        print(f"  Difference: {np.mean(td3_adapt_bear) - np.mean(td3_std_bear):+.3f}")
        print(f"  p-value: {p_val:.4f}, Cohen's d: {d:+.2f} ({effect_size_label(d)}), Significant: {sig}")
        test_results.append({
            "test": "TD3 adaptive vs standard (bear)",
            "group1_mean": np.mean(td3_adapt_bear),
            "group2_mean": np.mean(td3_std_bear),
            "difference": np.mean(td3_adapt_bear) - np.mean(td3_std_bear),
            "p_value": p_val,
            "cohens_d": d,
            "significant": p_val < 0.0042
        })

    # Test 2: SAC standard vs SAC balanced in each regime
    print("\n--- Test 2: SAC Standard vs Balanced per Regime ---")
    for regime in ["bull", "bear", "sideways"]:
        sac_std = run_means[(run_means["algorithm"] == "sac") &
                            (run_means["strategy"] == "standard") &
                            (run_means["regime"] == regime)]["sharpe"].values
        sac_bal = run_means[(run_means["algorithm"] == "sac") &
                            (run_means["strategy"] == "balanced") &
                            (run_means["regime"] == regime)]["sharpe"].values

        if len(sac_std) >= 2 and len(sac_bal) >= 2:
            t_stat, p_val = stats.ttest_ind(sac_std, sac_bal, equal_var=False)
            d = cohens_d(sac_std, sac_bal)
            sig = "YES" if p_val < 0.0042 else ("*" if p_val < 0.05 else "no")
            print(f"  {regime.capitalize()}: standard={np.mean(sac_std):+.3f} vs balanced={np.mean(sac_bal):+.3f}, diff={np.mean(sac_std) - np.mean(sac_bal):+.3f}, p={p_val:.4f}, d={d:+.2f}, sig={sig}")
            test_results.append({
                "test": f"SAC standard vs balanced ({regime})",
                "group1_mean": np.mean(sac_std),
                "group2_mean": np.mean(sac_bal),
                "difference": np.mean(sac_std) - np.mean(sac_bal),
                "p_value": p_val,
                "cohens_d": d,
                "significant": p_val < 0.0042
            })

    # Test 3: Algorithm comparison within BEAR markets
    print("\n--- Test 3: Algorithm Comparison in BEAR Markets ---")
    for algo in ["ppo", "sac", "rainbow"]:
        td3_bear = run_means[(run_means["algorithm"] == "td3") &
                             (run_means["regime"] == "bear")]["sharpe"].values
        algo_bear = run_means[(run_means["algorithm"] == algo) &
                              (run_means["regime"] == "bear")]["sharpe"].values

        if len(td3_bear) >= 2 and len(algo_bear) >= 2:
            t_stat, p_val = stats.ttest_ind(td3_bear, algo_bear, equal_var=False)
            d = cohens_d(td3_bear, algo_bear)
            sig = "YES" if p_val < 0.0042 else ("*" if p_val < 0.05 else "no")
            print(f"  TD3 vs {algo.upper()}: diff={np.mean(td3_bear) - np.mean(algo_bear):+.3f}, p={p_val:.4f}, d={d:+.2f} ({effect_size_label(d)}), sig={sig}")
            test_results.append({
                "test": f"TD3 vs {algo.upper()} (bear)",
                "group1_mean": np.mean(td3_bear),
                "group2_mean": np.mean(algo_bear),
                "difference": np.mean(td3_bear) - np.mean(algo_bear),
                "p_value": p_val,
                "cohens_d": d,
                "significant": p_val < 0.0042
            })

    # Save results
    results_df = pd.DataFrame(test_results)
    results_df.to_csv(TABLES_DIR / "e2_statistical_tests.csv", index=False)

    return results_df


# =============================================================================
# SECTION 9: BEHAVIORAL ANALYSIS - WEIGHTS BY REGIME
# =============================================================================

def analyze_weights_by_regime(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze portfolio weights by regime - flight to safety?"""
    print("\n" + "=" * 80)
    print("SECTION 9: BEHAVIORAL ANALYSIS - WEIGHTS BY REGIME")
    print("=" * 80)

    weight_data = []

    for idx, row in df.iterrows():
        pw = row["portfolio_weights"]
        rl = row["regime_labels"]

        if len(pw) == 0 or len(rl) == 0:
            continue

        # Align lengths
        min_len = min(len(pw), len(rl))
        pw = pw[:min_len]
        rl = rl[:min_len]

        # Group weights by regime
        regime_weights = defaultdict(list)
        for weights, regime in zip(pw, rl):
            if len(weights) == 5:  # Ensure 5 assets
                regime_weights[regime].append(weights)

        for regime, weights_list in regime_weights.items():
            if len(weights_list) >= MIN_DAYS_PER_REGIME:
                mean_weights = np.mean(weights_list, axis=0)
                for i, asset in enumerate(ASSETS):
                    weight_data.append({
                        "algorithm": row["algorithm"],
                        "strategy": row["strategy"],
                        "seed": row["seed"],
                        "episode": row["episode"],
                        "regime": regime,
                        "asset": asset,
                        "weight": mean_weights[i]
                    })

    weights_df = pd.DataFrame(weight_data)

    if len(weights_df) == 0:
        print("  No weight data available")
        return pd.DataFrame()

    # Aggregate to regime level (across all configs)
    regime_weights = weights_df.groupby(["regime", "asset"]).agg({
        "weight": ["mean", "std"]
    }).round(4)
    regime_weights.columns = ["_".join(col) for col in regime_weights.columns]
    regime_weights = regime_weights.reset_index()

    print("\n--- Average Weights by Regime (All Configs) ---")
    print("\n{:<10}".format("Asset"), end="")
    for regime in ["bull", "bear", "sideways"]:
        print("{:>12}".format(regime.capitalize()), end="")
    print()
    print("-" * 46)

    for asset in ASSETS:
        print("{:<10}".format(asset), end="")
        for regime in ["bull", "bear", "sideways"]:
            row = regime_weights[(regime_weights["regime"] == regime) & (regime_weights["asset"] == asset)]
            if len(row) > 0:
                print("{:>11.1%}".format(row["weight_mean"].values[0]), end=" ")
            else:
                print("{:>12}".format("N/A"), end="")
        print()

    # Flight to safety analysis: Compare TLT+GLD in bear vs bull
    print("\n--- FLIGHT TO SAFETY ANALYSIS ---")

    # Calculate safe asset allocation (TLT + GLD) by regime
    for regime in ["bull", "bear", "sideways"]:
        tlt = regime_weights[(regime_weights["regime"] == regime) & (regime_weights["asset"] == "TLT")]
        gld = regime_weights[(regime_weights["regime"] == regime) & (regime_weights["asset"] == "GLD")]
        if len(tlt) > 0 and len(gld) > 0:
            safe_alloc = tlt["weight_mean"].values[0] + gld["weight_mean"].values[0]
            print(f"  {regime.capitalize()}: TLT+GLD = {safe_alloc:.1%}")

    # TD3 adaptive specific analysis
    print("\n--- TD3 Adaptive Weight Behavior ---")
    td3_adapt = weights_df[(weights_df["algorithm"] == "td3") & (weights_df["strategy"] == "adaptive")]
    if len(td3_adapt) > 0:
        td3_regime_weights = td3_adapt.groupby(["regime", "asset"])["weight"].mean()
        for regime in ["bull", "bear", "sideways"]:
            print(f"\n  {regime.capitalize()}:")
            for asset in ASSETS:
                if (regime, asset) in td3_regime_weights.index:
                    w = td3_regime_weights[(regime, asset)]
                    print(f"    {asset}: {w:.1%}")

    # Save to CSV
    regime_weights.to_csv(TABLES_DIR / "e2_weights_by_regime.csv", index=False)

    return weights_df


# =============================================================================
# SECTION 10: REGIME TRANSITION ANALYSIS
# =============================================================================

def analyze_regime_transitions(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze performance and behavior around regime transitions."""
    print("\n" + "=" * 80)
    print("SECTION 10: REGIME TRANSITION ANALYSIS")
    print("=" * 80)

    TRANSITION_WINDOW = 5  # ±5 days around transition

    transition_data = []
    turnover_data = []

    for idx, row in df.iterrows():
        sr = row["step_returns"]
        rl = row["regime_labels"]
        pw = row["portfolio_weights"]

        if len(sr) < 10 or len(rl) < 10:
            continue

        # Align lengths
        min_len = min(len(sr), len(rl))
        sr = sr[:min_len]
        rl = rl[:min_len]

        # Find transition points
        transitions = []
        for i in range(1, len(rl)):
            if rl[i] != rl[i-1]:
                transitions.append({
                    "day": i,
                    "from_regime": rl[i-1],
                    "to_regime": rl[i]
                })

        # Analyze performance around each transition
        for trans in transitions:
            day = trans["day"]

            # Pre-transition window
            pre_start = max(0, day - TRANSITION_WINDOW)
            pre_returns = sr[pre_start:day]

            # Post-transition window
            post_end = min(len(sr), day + TRANSITION_WINDOW)
            post_returns = sr[day:post_end]

            if len(pre_returns) >= 3 and len(post_returns) >= 3:
                transition_data.append({
                    "algorithm": row["algorithm"],
                    "strategy": row["strategy"],
                    "seed": row["seed"],
                    "episode": row["episode"],
                    "from_regime": trans["from_regime"],
                    "to_regime": trans["to_regime"],
                    "transition_type": f"{trans['from_regime']}->{trans['to_regime']}",
                    "pre_mean_return": np.mean(pre_returns),
                    "post_mean_return": np.mean(post_returns),
                    "return_diff": np.mean(post_returns) - np.mean(pre_returns),
                    "transition_day_return": sr[day] if day < len(sr) else np.nan
                })

            # Turnover analysis (weight changes at transition)
            if len(pw) > day:
                pre_weights = pw[day-1] if day > 0 else pw[0]
                post_weights = pw[day] if day < len(pw) else pw[-1]
                if len(pre_weights) == 5 and len(post_weights) == 5:
                    turnover = np.sum(np.abs(np.array(post_weights) - np.array(pre_weights))) / 2
                    turnover_data.append({
                        "algorithm": row["algorithm"],
                        "strategy": row["strategy"],
                        "transition_type": f"{trans['from_regime']}->{trans['to_regime']}",
                        "turnover": turnover
                    })

    trans_df = pd.DataFrame(transition_data)
    turnover_df = pd.DataFrame(turnover_data)

    if len(trans_df) == 0:
        print("  No transition data available")
        return pd.DataFrame()

    print(f"\nTotal transitions analyzed: {len(trans_df)}")

    # Transition type summary
    print("\n--- Performance by Transition Type ---")
    trans_summary = trans_df.groupby("transition_type").agg({
        "pre_mean_return": "mean",
        "post_mean_return": "mean",
        "return_diff": "mean",
        "transition_day_return": "mean"
    }).round(6)

    print("\n{:<20} {:>12} {:>12} {:>12} {:>14}".format(
        "Transition", "Pre Return", "Post Return", "Diff", "Trans Day"
    ))
    print("-" * 74)
    for trans_type, row in trans_summary.iterrows():
        print("{:<20} {:>+12.4%} {:>+12.4%} {:>+12.4%} {:>+14.4%}".format(
            trans_type,
            row["pre_mean_return"],
            row["post_mean_return"],
            row["return_diff"],
            row["transition_day_return"]
        ))

    # Turnover by transition type
    if len(turnover_df) > 0:
        print("\n--- Turnover at Transitions ---")
        turnover_summary = turnover_df.groupby("transition_type")["turnover"].agg(["mean", "std"])
        for trans_type, row in turnover_summary.iterrows():
            print(f"  {trans_type}: {row['mean']:.1%} turnover (std: {row['std']:.1%})")

    # Algorithm comparison: Who adapts fastest?
    print("\n--- Adaptation Speed by Algorithm ---")
    algo_adapt = trans_df.groupby("algorithm")["return_diff"].mean()
    for algo in ALGORITHMS:
        if algo in algo_adapt.index:
            diff = algo_adapt[algo]
            print(f"  {algo.upper()}: post-pre return diff = {diff:+.4%}")

    # Save to CSV
    trans_df.to_csv(TABLES_DIR / "e2_transition_analysis.csv", index=False)

    return trans_df


# =============================================================================
# SECTION 11: KEY FINDINGS & FIGURES
# =============================================================================

def generate_regime_figures(metrics_df: pd.DataFrame, weights_df: pd.DataFrame,
                           consistency_df: pd.DataFrame, trans_df: pd.DataFrame):
    """Generate all E2-specific figures."""
    print("\n" + "=" * 80)
    print("SECTION 11: GENERATING FIGURES")
    print("=" * 80)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Regime Sharpe Heatmap (Algorithm × Strategy × Regime)
    try:
        print("\n  Generating e2_regime_sharpe_heatmap.png...")

        # Aggregate to config level per regime
        run_means = metrics_df.groupby(["algorithm", "strategy", "seed", "regime"]).agg({
            "sharpe": "mean"
        }).reset_index()

        config_regime = run_means.groupby(["algorithm", "strategy", "regime"]).agg({
            "sharpe": "mean"
        }).reset_index()

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        for i, regime in enumerate(["bull", "bear", "sideways"]):
            regime_data = config_regime[config_regime["regime"] == regime]
            pivot = regime_data.pivot(index="algorithm", columns="strategy", values="sharpe")

            # Reorder
            pivot = pivot.reindex(ALGORITHMS)
            pivot = pivot[STRATEGIES]

            sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdYlGn", center=0,
                       ax=axes[i], vmin=-0.5, vmax=1.0, cbar=(i==2))
            axes[i].set_title(f"{regime.capitalize()} Market", fontsize=14, fontweight="bold")
            axes[i].set_xlabel("Strategy")
            axes[i].set_ylabel("Algorithm" if i == 0 else "")

        fig.suptitle("E2: Sharpe Ratio by Algorithm × Strategy × Regime", fontsize=16, fontweight="bold")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "e2_regime_sharpe_heatmap.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("    Done")
    except Exception as e:
        print(f"    Error: {e}")

    # 2. Bear Market Rankings Bar Chart
    try:
        print("\n  Generating e2_bear_market_rankings.png...")

        run_means = metrics_df.groupby(["algorithm", "strategy", "seed", "regime"]).agg({
            "sharpe": "mean"
        }).reset_index()

        bear_data = run_means[run_means["regime"] == "bear"]
        config_bear = bear_data.groupby(["algorithm", "strategy"]).agg({
            "sharpe": ["mean", "std"]
        }).reset_index()
        config_bear.columns = ["algorithm", "strategy", "sharpe_mean", "sharpe_std"]
        config_bear["config"] = config_bear["algorithm"].str.upper() + " " + config_bear["strategy"]
        config_bear = config_bear.sort_values("sharpe_mean", ascending=True)

        fig, ax = plt.subplots(figsize=(12, 8))

        colors = ["#e74c3c" if s < 0 else "#2ecc71" for s in config_bear["sharpe_mean"]]
        bars = ax.barh(config_bear["config"], config_bear["sharpe_mean"],
                      xerr=config_bear["sharpe_std"], color=colors, alpha=0.8, capsize=3)

        ax.axvline(x=0, color="black", linestyle="-", linewidth=0.5)
        ax.set_xlabel("Sharpe Ratio", fontsize=12)
        ax.set_ylabel("")
        ax.set_title("E2: Bear Market Performance by Configuration", fontsize=14, fontweight="bold")

        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "e2_bear_market_rankings.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("    Done")
    except Exception as e:
        print(f"    Error: {e}")

    # 3. Consistency Score Visualization
    try:
        print("\n  Generating e2_regime_consistency.png...")

        if len(consistency_df) > 0:
            fig, ax = plt.subplots(figsize=(12, 8))

            # Create scatter plot: x=mean Sharpe, y=consistency score
            plot_df = consistency_df.copy()
            plot_df["config"] = plot_df["algorithm"].str.upper() + " " + plot_df["strategy"]

            colors_map = {"ppo": "#3498db", "sac": "#2ecc71", "td3": "#e74c3c", "rainbow": "#f39c12"}
            colors = [colors_map.get(a, "gray") for a in plot_df["algorithm"]]

            scatter = ax.scatter(plot_df["mean_sharpe"], plot_df["consistency_score"],
                                c=colors, s=100, alpha=0.7)

            # Add labels for top performers
            for _, row in plot_df.nlargest(5, "mean_sharpe").iterrows():
                ax.annotate(f"{row['algorithm'].upper()}\n{row['strategy']}",
                           (row["mean_sharpe"], row["consistency_score"]),
                           xytext=(5, 5), textcoords="offset points", fontsize=8)

            ax.set_xlabel("Mean Sharpe (across regimes)", fontsize=12)
            ax.set_ylabel("Consistency Score", fontsize=12)
            ax.set_title("E2: Performance vs Consistency Trade-off", fontsize=14, fontweight="bold")

            # Add quadrant lines
            ax.axhline(y=consistency_df["consistency_score"].median(), color="gray", linestyle="--", alpha=0.5)
            ax.axvline(x=consistency_df["mean_sharpe"].median(), color="gray", linestyle="--", alpha=0.5)

            # Legend
            from matplotlib.patches import Patch
            legend_elements = [Patch(facecolor=c, label=a.upper()) for a, c in colors_map.items()]
            ax.legend(handles=legend_elements, loc="upper right")

            plt.tight_layout()
            plt.savefig(FIGURES_DIR / "e2_regime_consistency.png", dpi=150, bbox_inches="tight")
            plt.close()
            print("    Done")
    except Exception as e:
        print(f"    Error: {e}")

    # 4. Detailed Weights by Regime
    try:
        print("\n  Generating e2_weights_by_regime_detailed.png...")

        if len(weights_df) > 0:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))

            for i, algo in enumerate(ALGORITHMS):
                ax = axes[i // 2, i % 2]
                algo_weights = weights_df[weights_df["algorithm"] == algo]

                if len(algo_weights) == 0:
                    continue

                # Aggregate
                regime_asset = algo_weights.groupby(["regime", "asset"])["weight"].agg(["mean", "std"]).reset_index()

                x = np.arange(len(ASSETS))
                width = 0.25
                colors = {"bull": "#2ecc71", "bear": "#e74c3c", "sideways": "#f39c12"}

                for j, regime in enumerate(["bull", "bear", "sideways"]):
                    regime_data = regime_asset[regime_asset["regime"] == regime]
                    means = [regime_data[regime_data["asset"] == a]["mean"].values[0]
                            if len(regime_data[regime_data["asset"] == a]) > 0 else 0
                            for a in ASSETS]
                    stds = [regime_data[regime_data["asset"] == a]["std"].values[0]
                           if len(regime_data[regime_data["asset"] == a]) > 0 else 0
                           for a in ASSETS]

                    ax.bar(x + j * width, means, width, yerr=stds, label=regime.capitalize(),
                          color=colors[regime], alpha=0.8, capsize=2)

                ax.set_xticks(x + width)
                ax.set_xticklabels(ASSETS)
                ax.set_ylabel("Weight")
                ax.set_title(f"{algo.upper()}", fontsize=12, fontweight="bold")
                if i == 0:
                    ax.legend()

            fig.suptitle("E2: Portfolio Weights by Regime per Algorithm", fontsize=14, fontweight="bold")
            plt.tight_layout()
            plt.savefig(FIGURES_DIR / "e2_weights_by_regime_detailed.png", dpi=150, bbox_inches="tight")
            plt.close()
            print("    Done")
    except Exception as e:
        print(f"    Error: {e}")

    # 5. Regime Sharpe Boxplots
    try:
        print("\n  Generating e2_regime_sharpe_boxplots.png...")

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        for i, regime in enumerate(["bull", "bear", "sideways"]):
            ax = axes[i]
            regime_data = metrics_df[metrics_df["regime"] == regime]

            # Create boxplot by algorithm
            data_by_algo = [regime_data[regime_data["algorithm"] == a]["sharpe"].dropna().values
                           for a in ALGORITHMS]

            bp = ax.boxplot(data_by_algo, labels=[a.upper() for a in ALGORITHMS], patch_artist=True)

            colors = ["#3498db", "#2ecc71", "#e74c3c", "#f39c12"]
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

            ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
            ax.set_ylabel("Sharpe Ratio" if i == 0 else "")
            ax.set_title(f"{regime.capitalize()} Market", fontsize=12, fontweight="bold")

        fig.suptitle("E2: Sharpe Distribution by Algorithm per Regime", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "e2_regime_sharpe_boxplots.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("    Done")
    except Exception as e:
        print(f"    Error: {e}")

    # 6. Transition Performance
    try:
        print("\n  Generating e2_transition_performance.png...")

        if len(trans_df) > 0:
            fig, ax = plt.subplots(figsize=(12, 6))

            trans_summary = trans_df.groupby("transition_type").agg({
                "pre_mean_return": "mean",
                "post_mean_return": "mean"
            }).reset_index()

            x = np.arange(len(trans_summary))
            width = 0.35

            bars1 = ax.bar(x - width/2, trans_summary["pre_mean_return"] * 100, width,
                          label="Pre-transition", color="#3498db", alpha=0.8)
            bars2 = ax.bar(x + width/2, trans_summary["post_mean_return"] * 100, width,
                          label="Post-transition", color="#e74c3c", alpha=0.8)

            ax.set_ylabel("Mean Daily Return (%)")
            ax.set_xlabel("Transition Type")
            ax.set_xticks(x)
            ax.set_xticklabels(trans_summary["transition_type"], rotation=45, ha="right")
            ax.set_title("E2: Performance Before/After Regime Transitions", fontsize=14, fontweight="bold")
            ax.legend()
            ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

            plt.tight_layout()
            plt.savefig(FIGURES_DIR / "e2_transition_performance.png", dpi=150, bbox_inches="tight")
            plt.close()
            print("    Done")
    except Exception as e:
        print(f"    Error: {e}")

    print(f"\nAll figures saved to: {FIGURES_DIR}/")


def print_thesis_conclusions(metrics_df: pd.DataFrame, consistency_df: pd.DataFrame):
    """Print key findings for thesis."""
    print("\n" + "=" * 80)
    print("KEY FINDINGS FOR THESIS")
    print("=" * 80)

    # Get key metrics
    run_means = metrics_df.groupby(["algorithm", "strategy", "seed", "regime"]).agg({
        "sharpe": "mean"
    }).reset_index()

    bear_data = run_means[run_means["regime"] == "bear"]
    bear_config = bear_data.groupby(["algorithm", "strategy"]).agg({
        "sharpe": "mean"
    }).reset_index()

    best_bear = bear_config.nlargest(1, "sharpe")

    print("""
1. BEAR MARKET CHAMPION
   """)
    if len(best_bear) > 0:
        print(f"   Best config for bear markets: {best_bear['algorithm'].values[0].upper()} {best_bear['strategy'].values[0]}")
        print(f"   Bear market Sharpe: {best_bear['sharpe'].values[0]:+.3f}")

    print("""
2. FLIGHT TO SAFETY
   - Analysis of TLT+GLD allocation in bear vs bull markets
   - Agents learn defensive positioning in downturns (if true)

3. REGIME-AWARE STRATEGY EFFECTIVENESS
   - inverse_frequency: Does it outperform in bear markets?
   - balanced: Does it provide regime balance?
   - current_focused: Does it adapt quickly?

4. CONSISTENCY vs PERFORMANCE TRADE-OFF
   """)
    if len(consistency_df) > 0:
        best_both = consistency_df.nlargest(3, "mean_sharpe")
        print("   Top 3 by overall Sharpe (with consistency):")
        for _, row in best_both.iterrows():
            print(f"     {row['algorithm'].upper()} {row['strategy']}: Sharpe={row['mean_sharpe']:+.3f}, Consistency={row['consistency_score']:.2f}")

    print("""
5. SAC ENTROPY BIAS BY REGIME
   - Does SAC's failure with non-uniform replay manifest differently per regime?
   - Compare SAC standard vs balanced within each regime

6. TRANSITION AGILITY
   - Which algorithms adapt fastest to regime changes?
   - Turnover analysis at transitions

RECOMMENDATION:
- For bear market protection: [Best bear config]
- For consistent cross-regime performance: [Best consistency config]
- Novel finding: [Any surprising regime-specific insights]
""")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 80)
    print("E2 DEEP REGIME ANALYSIS")
    print("Extracting regime-level insights from E1 data for thesis")
    print("=" * 80)

    # Create output directories
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Section 1: Load data
    df = load_all_episode_data()

    if len(df) == 0:
        print("\nERROR: No data loaded. Check if E1 experiments completed.")
        return

    # Section 2: Calculate per-regime metrics
    metrics_df = calculate_per_regime_metrics(df)

    if len(metrics_df) == 0:
        print("\nERROR: No regime metrics calculated. Check data format.")
        return

    # Section 3: Aggregate performance by regime
    aggregate_df = analyze_aggregate_regime_performance(metrics_df)

    # Section 4: Algorithm × Regime performance
    algo_regime_df = analyze_algorithm_regime_performance(metrics_df)

    # Section 5: Strategy × Regime performance
    strat_regime_df = analyze_strategy_regime_performance(metrics_df)

    # Section 6: Best config per regime
    config_regime_df = find_best_configs_per_regime(metrics_df)

    # Section 7: Cross-regime consistency
    consistency_df = analyze_cross_regime_consistency(metrics_df)

    # Section 8: Statistical tests
    stats_df = run_regime_statistical_tests(metrics_df)

    # Section 9: Weights by regime
    weights_df = analyze_weights_by_regime(df)

    # Section 10: Transition analysis
    trans_df = analyze_regime_transitions(df)

    # Section 11: Generate figures and conclusions
    generate_regime_figures(metrics_df, weights_df, consistency_df, trans_df)
    print_thesis_conclusions(metrics_df, consistency_df)

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nCSV tables saved to: {TABLES_DIR}/")
    print(f"Figures saved to: {FIGURES_DIR}/")
    print("\nGenerated files:")

    for f in sorted(TABLES_DIR.glob("e2_*.csv")):
        print(f"  {f.name}")

    for f in sorted(FIGURES_DIR.glob("e2_*.png")):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
