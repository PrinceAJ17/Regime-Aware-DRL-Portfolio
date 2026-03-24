"""
Deep Analysis of E4 (Feature Ablation) Experiment Results
==========================================================

Tests impact of feature set on performance:
- Feature sets: full, reduced
- Algorithms: SAC, PPO
- Strategy: balanced (fixed)
- Seeds: 42, 123, 456, 789, 1000

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
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
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


def load_e4_results() -> pd.DataFrame:
    """Load E4 results from experiment directories."""
    results = []
    base_dir = Path("results/experiments/feature_ablation")

    if not base_dir.exists():
        print(f"Warning: {base_dir} does not exist. No results to analyze.")
        return pd.DataFrame()

    for algo_dir in base_dir.iterdir():
        if not algo_dir.is_dir():
            continue
        algorithm = algo_dir.name

        for strategy_dir in algo_dir.iterdir():
            if not strategy_dir.is_dir():
                continue
            strategy = strategy_dir.name

            for run_dir in strategy_dir.iterdir():
                if not run_dir.is_dir():
                    continue

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
                        "algorithm": algorithm,
                        "strategy": strategy,
                        "feature_set": config.get("feature_set"),
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
    """Analyze E4 results."""
    print("=" * 80)
    print("DEEP ANALYSIS OF E4 (FEATURE ABLATION) EXPERIMENT")
    print("=" * 80)
    print(f"\nTotal runs: {len(df)}")
    print(f"Algorithms: {df['algorithm'].unique().tolist()}")
    print(f"Feature sets: {df['feature_set'].unique().tolist()}")
    print(f"Seeds: {sorted(df['seed'].unique().tolist())}")

    # =========================================================================
    # 1. SUMMARY BY FEATURE SET
    # =========================================================================
    print("\n" + "=" * 80)
    print("1. PERFORMANCE BY FEATURE SET (Aggregated Across Algorithms)")
    print("=" * 80)

    fs_summary = df.groupby('feature_set').agg({
        'test_sharpe': ['mean', 'std', 'min', 'max', 'count'],
        'val_sharpe': ['mean'],
        'test_mdd': ['mean']
    }).round(3)
    fs_summary.columns = ['_'.join(col) for col in fs_summary.columns]

    print("\n{:<12} {:>12} {:>8} {:>8} {:>8} {:>6} {:>12} {:>10}".format(
        "FeatureSet", "TestSharpe", "Std", "Min", "Max", "N", "ValSharpe", "TestMDD"
    ))
    print("-" * 86)
    for feature_set, row in fs_summary.iterrows():
        print("{:<12} {:>+12.3f} {:>8.3f} {:>+8.3f} {:>+8.3f} {:>6} {:>+12.3f} {:>9.1f}%".format(
            feature_set, row['test_sharpe_mean'], row['test_sharpe_std'],
            row['test_sharpe_min'], row['test_sharpe_max'],
            int(row['test_sharpe_count']),
            row['val_sharpe_mean'], row['test_mdd_mean'] * 100
        ))

    # =========================================================================
    # 2. SUMMARY BY ALGORITHM
    # =========================================================================
    print("\n" + "=" * 80)
    print("2. PERFORMANCE BY ALGORITHM")
    print("=" * 80)

    algo_summary = df.groupby('algorithm').agg({
        'test_sharpe': ['mean', 'std', 'min', 'max', 'count'],
        'val_sharpe': ['mean'],
        'test_mdd': ['mean']
    }).round(3)
    algo_summary.columns = ['_'.join(col) for col in algo_summary.columns]

    print("\n{:<10} {:>12} {:>8} {:>8} {:>8} {:>6} {:>12} {:>10}".format(
        "Algorithm", "TestSharpe", "Std", "Min", "Max", "N", "ValSharpe", "TestMDD"
    ))
    print("-" * 84)
    for algorithm, row in algo_summary.iterrows():
        print("{:<10} {:>+12.3f} {:>8.3f} {:>+8.3f} {:>+8.3f} {:>6} {:>+12.3f} {:>9.1f}%".format(
            algorithm.upper(), row['test_sharpe_mean'], row['test_sharpe_std'],
            row['test_sharpe_min'], row['test_sharpe_max'],
            int(row['test_sharpe_count']),
            row['val_sharpe_mean'], row['test_mdd_mean'] * 100
        ))

    # =========================================================================
    # 3. FEATURE SET x ALGORITHM INTERACTION
    # =========================================================================
    print("\n" + "=" * 80)
    print("3. FEATURE SET x ALGORITHM INTERACTION")
    print("=" * 80)

    pivot = df.groupby(['algorithm', 'feature_set'])['test_sharpe'].agg(['mean', 'std']).round(3)
    pivot = pivot.reset_index()

    print("\n{:<10} {:<12} {:>12} {:>8}".format("Algorithm", "FeatureSet", "TestSharpe", "Std"))
    print("-" * 46)

    for _, row in pivot.sort_values(['algorithm', 'feature_set']).iterrows():
        print("{:<10} {:<12} {:>+12.3f} {:>8.3f}".format(
            row['algorithm'].upper(), row['feature_set'], row['mean'], row['std']
        ))

    # =========================================================================
    # 4. STATISTICAL TESTS: FULL vs REDUCED
    # =========================================================================
    print("\n" + "=" * 80)
    print("4. STATISTICAL TESTS: FULL vs REDUCED FEATURES")
    print("=" * 80)

    full_data = df[df['feature_set'] == 'full']['test_sharpe'].values
    reduced_data = df[df['feature_set'] == 'reduced']['test_sharpe'].values

    if len(full_data) > 0 and len(reduced_data) > 0:
        t_stat, p_val = stats.ttest_ind(full_data, reduced_data, equal_var=False)
        d = cohens_d(full_data, reduced_data)
        diff = np.mean(full_data) - np.mean(reduced_data)

        print(f"\nOverall: Full vs Reduced")
        print(f"  Full mean:    {np.mean(full_data):+.3f} (std: {np.std(full_data):.3f})")
        print(f"  Reduced mean: {np.mean(reduced_data):+.3f} (std: {np.std(reduced_data):.3f})")
        print(f"  Difference:   {diff:+.3f}")
        print(f"  t-statistic:  {t_stat:.3f}")
        print(f"  p-value:      {p_val:.4f}")
        print(f"  Cohen's d:    {d:+.3f} ({effect_size_label(d)})")
        print(f"  Significant (p<0.05)? {'YES' if p_val < 0.05 else 'no'}")

    # Per algorithm
    print("\nPer-Algorithm Tests:")
    for algo in df['algorithm'].unique():
        algo_full = df[(df['algorithm'] == algo) & (df['feature_set'] == 'full')]['test_sharpe'].values
        algo_reduced = df[(df['algorithm'] == algo) & (df['feature_set'] == 'reduced')]['test_sharpe'].values

        if len(algo_full) > 0 and len(algo_reduced) > 0:
            t_stat, p_val = stats.ttest_ind(algo_full, algo_reduced, equal_var=False)
            d = cohens_d(algo_full, algo_reduced)
            diff = np.mean(algo_full) - np.mean(algo_reduced)
            sig = "YES" if p_val < 0.05 else "no"
            print(f"  {algo.upper()}: full {np.mean(algo_full):+.3f} vs reduced {np.mean(algo_reduced):+.3f} "
                  f"(diff: {diff:+.3f}, p={p_val:.4f}, d={d:+.2f}, sig: {sig})")

    # =========================================================================
    # 5. VAL-TEST GAP ANALYSIS
    # =========================================================================
    print("\n" + "=" * 80)
    print("5. VAL-TEST GAP ANALYSIS (OVERFITTING INDICATOR)")
    print("=" * 80)

    df['val_test_gap'] = df['val_sharpe'] - df['test_sharpe']

    gap_by_fs = df.groupby('feature_set')['val_test_gap'].agg(['mean', 'std']).round(3)
    print("\n{:<12} {:>15} {:>10}".format("FeatureSet", "Val-Test Gap", "Std"))
    print("-" * 40)
    for fs, row in gap_by_fs.iterrows():
        print("{:<12} {:>+15.3f} {:>10.3f}".format(fs, row['mean'], row['std']))

    # =========================================================================
    # 6. COMPARISON WITH E1 BASELINES
    # =========================================================================
    print("\n" + "=" * 80)
    print("6. COMPARISON WITH E1 BASELINES (balanced strategy)")
    print("=" * 80)

    # E1 baseline values from CLAUDE.md
    E1_BASELINES = {
        "ppo": {"balanced": 0.168},
        "sac": {"balanced": -0.014},
    }

    print("\n{:<10} {:<12} {:>12} {:>12} {:>10}".format(
        "Algorithm", "FeatureSet", "E4 Sharpe", "E1 Sharpe", "Difference"
    ))
    print("-" * 60)

    for algo in df['algorithm'].unique():
        if algo in E1_BASELINES:
            e1_val = E1_BASELINES[algo]["balanced"]
            for fs in ['full', 'reduced']:
                e4_mean = df[(df['algorithm'] == algo) & (df['feature_set'] == fs)]['test_sharpe'].mean()
                diff = e4_mean - e1_val
                print("{:<10} {:<12} {:>+12.3f} {:>+12.3f} {:>+10.3f}".format(
                    algo.upper(), fs, e4_mean, e1_val, diff
                ))

    # =========================================================================
    # 7. KEY FINDINGS
    # =========================================================================
    print("\n" + "=" * 80)
    print("7. KEY FINDINGS FOR THESIS")
    print("=" * 80)

    full_mean = df[df['feature_set'] == 'full']['test_sharpe'].mean()
    reduced_mean = df[df['feature_set'] == 'reduced']['test_sharpe'].mean()
    diff = full_mean - reduced_mean
    pct_diff = (diff / abs(reduced_mean)) * 100 if reduced_mean != 0 else 0

    full_gap = df[df['feature_set'] == 'full']['val_test_gap'].mean()
    reduced_gap = df[df['feature_set'] == 'reduced']['val_test_gap'].mean()

    print(f"""
KEY FINDINGS:

1. FEATURE SET COMPARISON
   - Full features: {full_mean:+.3f} mean Sharpe
   - Reduced features: {reduced_mean:+.3f} mean Sharpe
   - Difference: {diff:+.3f} ({pct_diff:+.1f}%)
   - Full features {'outperform' if diff > 0 else 'underperform'} reduced features

2. OVERFITTING BY FEATURE SET
   - Full features gap: {full_gap:+.3f}
   - Reduced features gap: {reduced_gap:+.3f}
   - {'Reduced features generalize better' if reduced_gap < full_gap else 'Full features generalize better'}

3. ALGORITHM SENSITIVITY
""")

    for algo in df['algorithm'].unique():
        algo_full = df[(df['algorithm'] == algo) & (df['feature_set'] == 'full')]['test_sharpe'].mean()
        algo_reduced = df[(df['algorithm'] == algo) & (df['feature_set'] == 'reduced')]['test_sharpe'].mean()
        algo_diff = algo_full - algo_reduced
        print(f"   - {algo.upper()}: diff = {algo_diff:+.3f} (full {'better' if algo_diff > 0 else 'worse'})")

    # Thesis paragraph
    better_fs = 'full' if full_mean > reduced_mean else 'reduced'
    print(f"""
THESIS PARAGRAPH:
"We investigated whether a reduced feature set could match full feature performance.
The full feature set achieved a mean Sharpe of {full_mean:+.3f} compared to {reduced_mean:+.3f}
for the reduced set, a difference of {diff:+.3f} ({pct_diff:+.1f}%).
{'This suggests that additional features provide useful information for the policy.'
if better_fs == 'full' else
'This suggests that simpler features may reduce overfitting and improve generalization.'}
The validation-test gap was {full_gap:+.3f} for full features vs {reduced_gap:+.3f} for reduced,
indicating {'similar' if abs(full_gap - reduced_gap) < 0.2 else 'different'} generalization behavior."
""")

    return df


def generate_plots(df: pd.DataFrame):
    """Generate E4 analysis plots."""
    if df.empty:
        print("No data to plot.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('E4: Feature Ablation Analysis', fontsize=14, fontweight='bold')

    # Plot 1: Bar chart - Test Sharpe by Feature Set
    ax1 = axes[0, 0]
    fs_means = df.groupby('feature_set')['test_sharpe'].mean()
    fs_stds = df.groupby('feature_set')['test_sharpe'].std()
    colors = ['#2ecc71', '#e74c3c']
    bars = ax1.bar(fs_means.index, fs_means.values, yerr=fs_stds.values, capsize=5, color=colors, alpha=0.8)
    ax1.set_xlabel('Feature Set')
    ax1.set_ylabel('Test Sharpe Ratio')
    ax1.set_title('Test Performance by Feature Set')
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax1.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar, val in zip(bars, fs_means.values):
        ax1.annotate(f'{val:+.3f}', xy=(bar.get_x() + bar.get_width()/2, val),
                     ha='center', va='bottom' if val > 0 else 'top', fontsize=10)

    # Plot 2: Grouped bar chart - Feature Set x Algorithm
    ax2 = axes[0, 1]
    algorithms = sorted(df['algorithm'].unique())
    feature_sets = ['full', 'reduced']
    x = np.arange(len(algorithms))
    width = 0.35

    for i, fs in enumerate(feature_sets):
        means = [df[(df['algorithm'] == algo) & (df['feature_set'] == fs)]['test_sharpe'].mean()
                 for algo in algorithms]
        stds = [df[(df['algorithm'] == algo) & (df['feature_set'] == fs)]['test_sharpe'].std()
                for algo in algorithms]
        ax2.bar(x + (i - 0.5) * width, means, width, yerr=stds, capsize=3,
                label=fs, color=colors[i], alpha=0.8)

    ax2.set_xlabel('Algorithm')
    ax2.set_ylabel('Test Sharpe Ratio')
    ax2.set_title('Test Performance by Algorithm x Feature Set')
    ax2.set_xticks(x)
    ax2.set_xticklabels([a.upper() for a in algorithms])
    ax2.legend()
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax2.grid(True, alpha=0.3, axis='y')

    # Plot 3: Val-Test Gap by Feature Set
    ax3 = axes[1, 0]
    df['val_test_gap'] = df['val_sharpe'] - df['test_sharpe']
    gap_means = df.groupby('feature_set')['val_test_gap'].mean()
    gap_stds = df.groupby('feature_set')['val_test_gap'].std()
    bars = ax3.bar(gap_means.index, gap_means.values, yerr=gap_stds.values, capsize=5,
                   color=['#3498db', '#9b59b6'], alpha=0.8)
    ax3.set_xlabel('Feature Set')
    ax3.set_ylabel('Val - Test Sharpe (Gap)')
    ax3.set_title('Overfitting Gap by Feature Set')
    ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax3.grid(True, alpha=0.3, axis='y')

    # Plot 4: Box plot of Test Sharpe by Feature Set
    ax4 = axes[1, 1]
    box_data = [df[df['feature_set'] == fs]['test_sharpe'].values for fs in feature_sets]
    bp = ax4.boxplot(box_data, tick_labels=['Full', 'Reduced'], patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax4.set_xlabel('Feature Set')
    ax4.set_ylabel('Test Sharpe Ratio')
    ax4.set_title('Test Sharpe Distribution by Feature Set')
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()

    # Save
    out_path = Path("results/figures/e4_feature_ablation.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {out_path}")
    plt.close()


def save_summary(df: pd.DataFrame):
    """Save CSV summary."""
    if df.empty:
        return

    summary = df.groupby(['algorithm', 'feature_set']).agg({
        'test_sharpe': ['mean', 'std', 'count'],
        'val_sharpe': ['mean'],
        'test_mdd': ['mean'],
        'test_return': ['mean'],
        'test_sortino': ['mean'],
        'test_calmar': ['mean']
    }).round(3)
    summary.columns = ['_'.join(col) for col in summary.columns]
    summary = summary.reset_index()

    # Add val-test gap
    summary['val_test_gap'] = (summary['val_sharpe_mean'] - summary['test_sharpe_mean']).round(3)

    out_path = Path("results/e4_analysis_summary.csv")
    summary.to_csv(out_path, index=False)
    print(f"Summary saved to: {out_path}")


def main():
    df = load_e4_results()

    if df.empty:
        print("\nNo E4 results found. Run the experiment first:")
        print("  python scripts/launch_experiment.py --experiments feature_ablation --workers 4")
        return

    df = analyze_results(df)
    generate_plots(df)
    save_summary(df)


if __name__ == "__main__":
    main()
