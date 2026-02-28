"""
Statistical Tests
=================

Welch's t-test with Bonferroni correction for comparing replay strategies.

Tests each regime-aware strategy vs standard (baseline) per algorithm.

Author: Prince Alexander John
Date: February 2026
"""

import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path

from .results_loader import load_all_results


TABLES_DIR = Path("results/tables")


def run_all_tests(output_dir: Path = TABLES_DIR) -> pd.DataFrame:
    """Run all statistical tests and save results."""
    output_dir.mkdir(parents=True, exist_ok=True)
    df = load_all_results()

    if df.empty:
        print("No results for statistical testing.")
        return pd.DataFrame()

    results = test_replay_vs_standard(df)

    if not results.empty:
        results.to_csv(output_dir / "statistical_tests_e1.csv", index=False)
        print(f"  Statistical tests saved to {output_dir / 'statistical_tests_e1.csv'}")

    return results


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
                "n_strategy": len(strat_sharpe),
                "n_standard": len(standard_sharpe),
                "significant_raw": p_value < 0.05,
                "significant_bonferroni": p_value < bonferroni_alpha,
                "bonferroni_alpha": bonferroni_alpha,
            })

    return pd.DataFrame(results)
