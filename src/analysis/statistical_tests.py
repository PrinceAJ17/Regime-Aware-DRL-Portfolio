"""
Statistical Tests
=================

Statistical analysis for comparing replay strategies:
- Welch's t-test with Bonferroni correction
- Cohen's d effect size
- Bootstrap confidence intervals
- Tests for multiple metrics (Sharpe, Sortino, MDD)

Author: Prince Alexander John
Date: February 2026
"""

import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
from typing import Tuple, Optional

from .results_loader import load_all_results


TABLES_DIR = Path("results/tables")


# =============================================================================
# Effect Size Calculations
# =============================================================================

def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """
    Compute Cohen's d effect size for two independent samples.

    Uses pooled standard deviation (Cohen's original formulation).
    Interpretation:
        |d| < 0.2: negligible
        0.2 <= |d| < 0.5: small
        0.5 <= |d| < 0.8: medium
        |d| >= 0.8: large

    Parameters
    ----------
    group1 : np.ndarray
        First group (treatment).
    group2 : np.ndarray
        Second group (control/baseline).

    Returns
    -------
    float
        Cohen's d effect size.
    """
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return np.nan

    var1 = np.var(group1, ddof=1)
    var2 = np.var(group2, ddof=1)

    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    if pooled_std == 0:
        return np.nan

    return (np.mean(group1) - np.mean(group2)) / pooled_std


def effect_size_interpretation(d: float) -> str:
    """Return interpretation of Cohen's d magnitude."""
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


def bootstrap_ci(data: np.ndarray, stat_func=np.mean, n_bootstrap: int = 10000,
                 confidence: float = 0.95, seed: int = 42) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval for a statistic.

    Parameters
    ----------
    data : np.ndarray
        Sample data.
    stat_func : callable
        Statistic to compute (default: np.mean).
    n_bootstrap : int
        Number of bootstrap samples.
    confidence : float
        Confidence level (default: 0.95 for 95% CI).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    Tuple[float, float, float]
        (point_estimate, lower_bound, upper_bound)
    """
    rng = np.random.RandomState(seed)
    n = len(data)

    if n < 2:
        return np.nan, np.nan, np.nan

    boot_stats = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        sample = rng.choice(data, size=n, replace=True)
        boot_stats[i] = stat_func(sample)

    alpha = 1 - confidence
    lower = np.percentile(boot_stats, 100 * alpha / 2)
    upper = np.percentile(boot_stats, 100 * (1 - alpha / 2))
    point = stat_func(data)

    return point, lower, upper


def run_all_tests(output_dir: Path = TABLES_DIR) -> dict:
    """Run all statistical tests and save results."""
    output_dir.mkdir(parents=True, exist_ok=True)
    df = load_all_results()

    if df.empty:
        print("No results for statistical testing.")
        return {}

    all_results = {}

    # 1. Basic t-tests for Sharpe (original)
    sharpe_tests = test_replay_vs_standard(df)
    if not sharpe_tests.empty:
        sharpe_tests.to_csv(output_dir / "statistical_tests_sharpe.csv", index=False)
        print(f"  Sharpe tests saved to {output_dir / 'statistical_tests_sharpe.csv'}")
        all_results["sharpe_tests"] = sharpe_tests

    # 2. Multi-metric tests (Sharpe, Sortino, MDD)
    multi_tests = test_all_metrics(df)
    if not multi_tests.empty:
        multi_tests.to_csv(output_dir / "statistical_tests_all_metrics.csv", index=False)
        print(f"  Multi-metric tests saved to {output_dir / 'statistical_tests_all_metrics.csv'}")
        all_results["multi_metric_tests"] = multi_tests

    # 3. Win rate analysis
    win_rates = compute_win_rates(df)
    if not win_rates.empty:
        win_rates.to_csv(output_dir / "win_rates.csv", index=False)
        print(f"  Win rates saved to {output_dir / 'win_rates.csv'}")
        all_results["win_rates"] = win_rates

    # Keep backwards compatibility: return sharpe_tests as primary result
    return all_results


def test_replay_vs_standard(df: pd.DataFrame) -> pd.DataFrame:
    """
    Welch's t-test: each regime-aware strategy vs standard baseline.

    Per algorithm: 4 comparisons (balanced, current_focused, inverse_frequency, adaptive)
    Total: 4 algorithms x 4 strategies = 16 tests
    Bonferroni alpha = 0.05 / 16 = 0.003125
    """
    e1 = df[df["experiment"] == "replay_comparison"].copy()
    if e1.empty or "sharpe_ratio" not in e1.columns:
        return pd.DataFrame()

    strategies = ["balanced", "current_focused", "inverse_frequency", "adaptive"]
    algorithms = sorted(e1["algorithm"].unique())
    n_tests = len(strategies) * len(algorithms)
    bonferroni_alpha = 0.05 / max(n_tests, 1)

    results = []

    for algo in algorithms:
        algo_df = e1[e1["algorithm"] == algo]
        standard_sharpe = algo_df[algo_df["strategy"] == "standard"]["sharpe_ratio"].dropna().values

        if len(standard_sharpe) < 2:
            continue

        for strat in strategies:
            strat_sharpe = algo_df[algo_df["strategy"] == strat]["sharpe_ratio"].dropna().values

            if len(strat_sharpe) < 2:
                continue

            t_stat, p_value = stats.ttest_ind(strat_sharpe, standard_sharpe, equal_var=False)
            improvement = np.mean(strat_sharpe) - np.mean(standard_sharpe)
            effect_d = cohens_d(strat_sharpe, standard_sharpe)

            results.append({
                "algorithm": algo,
                "strategy": strat,
                "vs": "standard",
                "metric": "sharpe_ratio",
                "mean_strategy": np.mean(strat_sharpe),
                "std_strategy": np.std(strat_sharpe, ddof=1),
                "mean_standard": np.mean(standard_sharpe),
                "std_standard": np.std(standard_sharpe, ddof=1),
                "improvement": improvement,
                "improvement_pct": (improvement / abs(np.mean(standard_sharpe)) * 100
                                    if np.mean(standard_sharpe) != 0 else 0),
                "t_statistic": t_stat,
                "p_value": p_value,
                "cohens_d": effect_d,
                "effect_size": effect_size_interpretation(effect_d),
                "n_strategy": len(strat_sharpe),
                "n_standard": len(standard_sharpe),
                "significant_raw": p_value < 0.05,
                "significant_bonferroni": p_value < bonferroni_alpha,
                "bonferroni_alpha": bonferroni_alpha,
            })

    return pd.DataFrame(results)


def test_all_metrics(df: pd.DataFrame, metrics: list = None) -> pd.DataFrame:
    """
    Run statistical tests for multiple metrics.

    Parameters
    ----------
    df : pd.DataFrame
        Results dataframe from load_all_results().
    metrics : list, optional
        Metrics to test. Default: ['sharpe_ratio', 'sortino_ratio', 'max_drawdown'].

    Returns
    -------
    pd.DataFrame
        Combined test results for all metrics.
    """
    if metrics is None:
        metrics = ["sharpe_ratio", "sortino_ratio", "max_drawdown"]

    e1 = df[df["experiment"] == "replay_comparison"].copy()
    if e1.empty:
        return pd.DataFrame()

    strategies = ["balanced", "current_focused", "inverse_frequency", "adaptive"]
    algorithms = sorted(e1["algorithm"].unique())
    n_tests = len(strategies) * len(algorithms) * len(metrics)
    bonferroni_alpha = 0.05 / max(n_tests, 1)

    results = []

    for metric in metrics:
        if metric not in e1.columns:
            continue

        # For MDD, lower is better, so flip the sign for interpretation
        flip_sign = metric == "max_drawdown"

        for algo in algorithms:
            algo_df = e1[e1["algorithm"] == algo]
            standard_vals = algo_df[algo_df["strategy"] == "standard"][metric].dropna().values

            if len(standard_vals) < 2:
                continue

            for strat in strategies:
                strat_vals = algo_df[algo_df["strategy"] == strat][metric].dropna().values

                if len(strat_vals) < 2:
                    continue

                t_stat, p_value = stats.ttest_ind(strat_vals, standard_vals, equal_var=False)
                improvement = np.mean(strat_vals) - np.mean(standard_vals)
                effect_d = cohens_d(strat_vals, standard_vals)

                # Bootstrap CI for the strategy
                point, ci_lower, ci_upper = bootstrap_ci(strat_vals)

                # For MDD, improvement means reduction (more negative diff is better)
                if flip_sign:
                    improvement = -improvement  # Negative MDD diff = improvement
                    effect_d = -effect_d if not np.isnan(effect_d) else effect_d

                results.append({
                    "metric": metric,
                    "algorithm": algo,
                    "strategy": strat,
                    "vs": "standard",
                    "mean_strategy": np.mean(strat_vals),
                    "ci_lower": ci_lower,
                    "ci_upper": ci_upper,
                    "std_strategy": np.std(strat_vals, ddof=1),
                    "mean_standard": np.mean(standard_vals),
                    "std_standard": np.std(standard_vals, ddof=1),
                    "improvement": improvement,
                    "t_statistic": t_stat,
                    "p_value": p_value,
                    "cohens_d": effect_d,
                    "effect_size": effect_size_interpretation(effect_d),
                    "n_strategy": len(strat_vals),
                    "n_standard": len(standard_vals),
                    "significant_bonferroni": p_value < bonferroni_alpha,
                    "bonferroni_alpha": bonferroni_alpha,
                })

    return pd.DataFrame(results)


def compute_win_rates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute win rates: fraction of runs where strategy beats standard.

    For each (algorithm, strategy) pair, computes:
    - Fraction of seeds where strategy Sharpe > standard Sharpe
    - Uses paired comparison when possible (same seeds)
    """
    e1 = df[df["experiment"] == "replay_comparison"].copy()
    if e1.empty:
        return pd.DataFrame()

    strategies = ["balanced", "current_focused", "inverse_frequency", "adaptive"]
    algorithms = sorted(e1["algorithm"].unique())

    results = []

    for algo in algorithms:
        algo_df = e1[e1["algorithm"] == algo]
        standard_df = algo_df[algo_df["strategy"] == "standard"]

        for strat in strategies:
            strat_df = algo_df[algo_df["strategy"] == strat]

            # Get common seeds for paired comparison
            standard_by_seed = standard_df.set_index("seed")["sharpe_ratio"]
            strat_by_seed = strat_df.set_index("seed")["sharpe_ratio"]

            common_seeds = set(standard_by_seed.index) & set(strat_by_seed.index)

            if len(common_seeds) < 2:
                continue

            wins = 0
            losses = 0
            ties = 0

            for seed in common_seeds:
                s_val = standard_by_seed.loc[seed]
                t_val = strat_by_seed.loc[seed]

                if pd.isna(s_val) or pd.isna(t_val):
                    continue

                if t_val > s_val + 0.01:  # Small threshold to avoid floating point ties
                    wins += 1
                elif t_val < s_val - 0.01:
                    losses += 1
                else:
                    ties += 1

            total = wins + losses + ties
            if total == 0:
                continue

            results.append({
                "algorithm": algo,
                "strategy": strat,
                "wins": wins,
                "losses": losses,
                "ties": ties,
                "total": total,
                "win_rate": wins / total,
                "loss_rate": losses / total,
            })

    return pd.DataFrame(results)
