"""
Deep Analysis of E7 (Timestep Ablation) Experiment Results
===========================================================

Tests impact of training duration on overfitting:
- Timesteps: 100k, 200k, 300k, 400k, 500k
- Algorithm: TD3 only
- Strategies: adaptive, standard
- Seeds: 42, 123, 456

Author: Prince Alexander John
Date: March 2026
"""

import json
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
import matplotlib.pyplot as plt


def cohens_d(group1, group2):
    """Compute Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    if n1 == 0 or n2 == 0:
        return np.nan
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return np.nan
    return (np.mean(group1) - np.mean(group2)) / pooled_std


def effect_size_label(d):
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


def load_e7_results() -> pd.DataFrame:
    """Load E7 results from experiment directories."""
    results = []
    base_dir = Path("results/experiments/timestep_ablation/td3")

    if not base_dir.exists():
        print(f"Warning: {base_dir} does not exist. No results to analyze.")
        return pd.DataFrame()

    for strategy_dir in base_dir.iterdir():
        if not strategy_dir.is_dir():
            continue
        strategy = strategy_dir.name

        for run_dir in strategy_dir.iterdir():
            if not run_dir.is_dir():
                continue

            # Load run config to get timesteps
            config_file = run_dir / "run_config.json"
            test_file = run_dir / "test_results.json"
            val_file = run_dir / "val_results.json"

            if not config_file.exists() or not test_file.exists():
                continue

            try:
                config = json.loads(config_file.read_text(encoding="utf-8"))
                test_results = json.loads(test_file.read_text(encoding="utf-8"))

                val_sharpe = None
                if val_file.exists():
                    val_results = json.loads(val_file.read_text(encoding="utf-8"))
                    val_sharpe = val_results.get("sharpe_ratio")

                results.append({
                    "timesteps": config.get("total_timesteps"),
                    "strategy": strategy,
                    "seed": config.get("seed"),
                    "test_sharpe": test_results.get("sharpe_ratio"),
                    "val_sharpe": val_sharpe,
                    "test_mdd": test_results.get("max_drawdown"),
                    "test_return": test_results.get("cumulative_return"),
                    "test_sortino": test_results.get("sortino_ratio"),
                    "test_calmar": test_results.get("calmar_ratio"),
                })
            except Exception as e:
                print(f"Warning: Failed to load {run_dir}: {e}")
                continue

    return pd.DataFrame(results)


def analyze_results(df: pd.DataFrame):
    """Analyze E7 results."""
    print("=" * 80)
    print("DEEP ANALYSIS OF E7 (TIMESTEP ABLATION) EXPERIMENT")
    print("=" * 80)
    print(f"\nTotal runs: {len(df)}")
    print(f"Algorithm: TD3 only")
    print(f"Timesteps: {sorted(df['timesteps'].unique().tolist())}")
    print(f"Strategies: {df['strategy'].unique().tolist()}")
    print(f"Seeds: {sorted(df['seed'].unique().tolist())}")

    # =========================================================================
    # 1. SUMMARY BY TIMESTEPS
    # =========================================================================
    print("\n" + "=" * 80)
    print("1. PERFORMANCE BY TRAINING DURATION (Aggregated Across Strategies)")
    print("=" * 80)

    ts_summary = df.groupby('timesteps').agg({
        'test_sharpe': ['mean', 'std', 'min', 'max'],
        'val_sharpe': ['mean'],
        'test_mdd': ['mean']
    }).round(3)
    ts_summary.columns = ['_'.join(col) for col in ts_summary.columns]

    print("\n{:>10} {:>12} {:>8} {:>8} {:>8} {:>12} {:>10}".format(
        "Timesteps", "TestSharpe", "Std", "Min", "Max", "ValSharpe", "TestMDD"
    ))
    print("-" * 76)
    for timesteps, row in ts_summary.iterrows():
        print("{:>10}k {:>+11.3f} {:>8.3f} {:>+8.3f} {:>+8.3f} {:>+11.3f} {:>9.1f}%".format(
            int(timesteps // 1000), row['test_sharpe_mean'], row['test_sharpe_std'],
            row['test_sharpe_min'], row['test_sharpe_max'],
            row['val_sharpe_mean'], row['test_mdd_mean'] * 100
        ))

    # =========================================================================
    # 2. SUMMARY BY TIMESTEPS x STRATEGY
    # =========================================================================
    print("\n" + "=" * 80)
    print("2. PERFORMANCE BY TIMESTEPS x STRATEGY")
    print("=" * 80)

    pivot = df.groupby(['timesteps', 'strategy'])['test_sharpe'].agg(['mean', 'std']).round(3)
    pivot = pivot.reset_index()

    print("\n{:>10} {:<12} {:>12} {:>8}".format("Timesteps", "Strategy", "TestSharpe", "Std"))
    print("-" * 46)

    for _, row in pivot.sort_values(['timesteps', 'strategy']).iterrows():
        print("{:>9}k {:<12} {:>+12.3f} {:>8.3f}".format(
            int(row['timesteps'] // 1000), row['strategy'], row['mean'], row['std']
        ))

    # =========================================================================
    # 3. VAL-TEST GAP ANALYSIS (OVERFITTING)
    # =========================================================================
    print("\n" + "=" * 80)
    print("3. VAL-TEST GAP ANALYSIS (OVERFITTING INDICATOR)")
    print("=" * 80)

    df['val_test_gap'] = df['val_sharpe'] - df['test_sharpe']

    gap_summary = df.groupby('timesteps')['val_test_gap'].agg(['mean', 'std']).round(3)
    gap_summary = gap_summary.reset_index()

    print("\n{:>10} {:>15} {:>10}".format("Timesteps", "Val-Test Gap", "Std"))
    print("-" * 40)
    for _, row in gap_summary.iterrows():
        print("{:>9}k {:>+15.3f} {:>10.3f}".format(
            int(row['timesteps'] // 1000), row['mean'], row['std']
        ))

    # =========================================================================
    # 4. OPTIMAL TIMESTEP IDENTIFICATION
    # =========================================================================
    print("\n" + "=" * 80)
    print("4. OPTIMAL TIMESTEP IDENTIFICATION")
    print("=" * 80)

    # By test performance
    best_test = df.groupby('timesteps')['test_sharpe'].mean().idxmax()
    best_test_value = df.groupby('timesteps')['test_sharpe'].mean().max()

    # By minimum overfitting (smallest gap)
    min_gap_ts = df.groupby('timesteps')['val_test_gap'].mean().idxmin()
    min_gap_value = df.groupby('timesteps')['val_test_gap'].mean().min()

    print(f"\nBest Test Sharpe: {int(best_test // 1000)}k timesteps (Sharpe: {best_test_value:+.3f})")
    print(f"Minimum Overfitting: {int(min_gap_ts // 1000)}k timesteps (Gap: {min_gap_value:+.3f})")

    # Per strategy
    print("\nOptimal by Strategy:")
    for strategy in df['strategy'].unique():
        strat_df = df[df['strategy'] == strategy]
        best_ts = strat_df.groupby('timesteps')['test_sharpe'].mean().idxmax()
        best_val = strat_df.groupby('timesteps')['test_sharpe'].mean().max()
        print(f"  {strategy}: {int(best_ts // 1000)}k timesteps (Test Sharpe: {best_val:+.3f})")

    # =========================================================================
    # 5. STATISTICAL TESTS
    # =========================================================================
    print("\n" + "=" * 80)
    print("5. STATISTICAL TESTS: TIMESTEP EFFECT")
    print("=" * 80)

    # ANOVA across timesteps
    groups = [df[df['timesteps'] == ts]['test_sharpe'].values for ts in sorted(df['timesteps'].unique())]
    f_stat, p_val = stats.f_oneway(*groups)
    print(f"\nOne-way ANOVA across timesteps: F={f_stat:.3f}, p={p_val:.4f}")
    sig = "YES" if p_val < 0.05 else "no"
    print(f"Significant difference between timesteps? {sig}")

    # Pairwise: 200k vs 500k (likely best vs standard)
    ts_200k = df[df['timesteps'] == 200_000]['test_sharpe'].values
    ts_500k = df[df['timesteps'] == 500_000]['test_sharpe'].values
    if len(ts_200k) > 0 and len(ts_500k) > 0:
        t_stat, p_val = stats.ttest_ind(ts_200k, ts_500k, equal_var=False)
        d = cohens_d(ts_200k, ts_500k)
        print(f"\n200k vs 500k: t={t_stat:.3f}, p={p_val:.4f}")
        diff = np.mean(ts_200k) - np.mean(ts_500k)
        print(f"  Difference: {diff:+.3f} (200k {'better' if diff > 0 else 'worse'})")
        print(f"  Cohen's d: {d:+.3f} ({effect_size_label(d)})")

    # =========================================================================
    # 6. COMPARISON TO E1 500k BASELINE
    # =========================================================================
    print("\n" + "=" * 80)
    print("6. COMPARISON TO E1 TD3 RESULTS (500k baseline)")
    print("=" * 80)

    # E1 TD3 results (500k baseline)
    E1_TD3 = {
        "adaptive": {"mean": 0.733, "std": 0.206},
        "standard": {"mean": 0.632, "std": 0.166},
    }

    print("\n{:<12} {:>12} {:>12} {:>10}".format(
        "Strategy", "E1 500k", "E7 500k", "Difference"
    ))
    print("-" * 50)

    for strategy in ['adaptive', 'standard']:
        e1_mean = E1_TD3[strategy]['mean']
        e7_500k = df[(df['timesteps'] == 500_000) & (df['strategy'] == strategy)]['test_sharpe'].mean()
        diff = e7_500k - e1_mean
        print("{:<12} {:>+12.3f} {:>+12.3f} {:>+10.3f}".format(
            strategy, e1_mean, e7_500k, diff
        ))

    # =========================================================================
    # 7. KEY FINDINGS
    # =========================================================================
    print("\n" + "=" * 80)
    print("7. KEY FINDINGS FOR THESIS")
    print("=" * 80)

    # Find optimal
    overall_best_ts = df.groupby('timesteps')['test_sharpe'].mean().idxmax()
    overall_best_val = df.groupby('timesteps')['test_sharpe'].mean().max()
    ts_500k_val = df[df['timesteps'] == 500_000]['test_sharpe'].mean()

    # Gap analysis
    gap_at_best = df[df['timesteps'] == overall_best_ts]['val_test_gap'].mean()
    gap_at_500k = df[df['timesteps'] == 500_000]['val_test_gap'].mean()

    print(f"""
KEY FINDINGS:

1. OPTIMAL TRAINING DURATION
   - Best test Sharpe at: {int(overall_best_ts // 1000)}k timesteps ({overall_best_val:+.3f})
   - Standard 500k timesteps: {ts_500k_val:+.3f}
   - Improvement: {overall_best_val - ts_500k_val:+.3f} ({(overall_best_val - ts_500k_val) / abs(ts_500k_val) * 100:.1f}%)

2. OVERFITTING ANALYSIS
   - Val-Test gap at {int(overall_best_ts // 1000)}k: {gap_at_best:+.3f}
   - Val-Test gap at 500k: {gap_at_500k:+.3f}
   - Shorter training {'reduces' if gap_at_best < gap_at_500k else 'does not reduce'} overfitting

3. PRACTICAL RECOMMENDATIONS
   - Use {int(overall_best_ts // 1000)}k timesteps for TD3 training
   - Saves {(500_000 - overall_best_ts) / 500_000 * 100:.0f}% training time
   - Implement early stopping based on validation performance

THESIS PARAGRAPH:
"We investigated the impact of training duration on generalization.
Test Sharpe peaked at {int(overall_best_ts // 1000)}k timesteps ({overall_best_val:+.3f}),
after which performance {'degraded' if ts_500k_val < overall_best_val else 'plateaued'}.
The validation-test gap {'increased' if gap_at_500k > gap_at_best else 'remained stable'}
from {gap_at_best:+.3f} to {gap_at_500k:+.3f} at longer training durations,
indicating {'overfitting beyond the optimal point' if gap_at_500k > gap_at_best else 'stable generalization'}.
This suggests that the commonly used 500k timesteps may be {'excessive' if overall_best_ts < 500_000 else 'appropriate'}
for this task."
""")

    return df


def generate_plots(df: pd.DataFrame):
    """Generate E7 analysis plots."""
    if df.empty:
        print("No data to plot.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('E7: Timestep Ablation Analysis (TD3)', fontsize=14, fontweight='bold')

    timesteps_k = sorted(df['timesteps'].unique() / 1000)

    # Plot 1: Test Sharpe vs Timesteps by Strategy
    ax1 = axes[0, 0]
    for strategy in ['adaptive', 'standard']:
        strat_df = df[df['strategy'] == strategy]
        means = strat_df.groupby('timesteps')['test_sharpe'].mean()
        stds = strat_df.groupby('timesteps')['test_sharpe'].std()
        ts_k = means.index / 1000
        ax1.errorbar(ts_k, means.values, yerr=stds.values, marker='o', capsize=5, label=strategy)
    ax1.set_xlabel('Timesteps (k)')
    ax1.set_ylabel('Test Sharpe Ratio')
    ax1.set_title('Test Performance vs Training Duration')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    # Plot 2: Val-Test Gap vs Timesteps
    ax2 = axes[0, 1]
    df['val_test_gap'] = df['val_sharpe'] - df['test_sharpe']
    gap_means = df.groupby('timesteps')['val_test_gap'].mean()
    gap_stds = df.groupby('timesteps')['val_test_gap'].std()
    ts_k = gap_means.index / 1000
    ax2.errorbar(ts_k, gap_means.values, yerr=gap_stds.values, marker='s', capsize=5, color='red')
    ax2.set_xlabel('Timesteps (k)')
    ax2.set_ylabel('Val - Test Sharpe (Gap)')
    ax2.set_title('Overfitting Gap vs Training Duration')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    # Plot 3: Val and Test Sharpe vs Timesteps
    ax3 = axes[1, 0]
    test_means = df.groupby('timesteps')['test_sharpe'].mean()
    val_means = df.groupby('timesteps')['val_sharpe'].mean()
    ts_k = test_means.index / 1000
    ax3.plot(ts_k, val_means.values, marker='o', label='Validation', color='green')
    ax3.plot(ts_k, test_means.values, marker='s', label='Test', color='blue')
    ax3.set_xlabel('Timesteps (k)')
    ax3.set_ylabel('Sharpe Ratio')
    ax3.set_title('Validation vs Test Performance')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.fill_between(ts_k, val_means.values, test_means.values, alpha=0.2, color='red', label='Gap')

    # Plot 4: Box plot of Test Sharpe by Timesteps
    ax4 = axes[1, 1]
    timesteps_list = sorted(df['timesteps'].unique())
    box_data = [df[df['timesteps'] == ts]['test_sharpe'].values for ts in timesteps_list]
    bp = ax4.boxplot(box_data, labels=[f'{int(ts//1000)}k' for ts in timesteps_list], patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
    ax4.set_xlabel('Timesteps')
    ax4.set_ylabel('Test Sharpe Ratio')
    ax4.set_title('Test Sharpe Distribution by Training Duration')
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()

    # Save
    out_path = Path("results/figures/e7_timestep_analysis.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {out_path}")
    plt.close()


def main():
    df = load_e7_results()

    if df.empty:
        print("\nNo E7 results found. Run the experiment first:")
        print("  python scripts/launch_experiment.py --experiments timestep_ablation --workers 4")
        return

    df = analyze_results(df)
    generate_plots(df)

    # Save summary
    summary = df.groupby(['timesteps', 'strategy']).agg({
        'test_sharpe': ['mean', 'std'],
        'val_sharpe': ['mean'],
        'val_test_gap': ['mean'],
        'test_mdd': ['mean'],
        'test_return': ['mean'],
        'test_sortino': ['mean'],
        'test_calmar': ['mean']
    }).round(3)
    summary.to_csv("results/e7_analysis_summary.csv")
    print(f"\nSummary saved to: results/e7_analysis_summary.csv")


if __name__ == "__main__":
    main()
