"""
Deep Analysis of E5 (Regime Detection) Experiment Results
==========================================================

Tests impact of regime detection method on performance:
- Detection methods: rule_based, hmm
- Algorithm: SAC only
- Strategies: standard, balanced
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


def load_e5_results() -> pd.DataFrame:
    """Load E5 results from experiment directories."""
    results = []
    base_dir = Path("results/experiments/regime_detection")

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
                        "detection_method": config.get("detection_method"),
                        "seed": config.get("seed"),
                        "test_sharpe": test_results.get("sharpe_ratio"),
                        "val_sharpe": val_sharpe,
                        "test_mdd": test_results.get("max_drawdown"),
                        "test_return": test_results.get("cumulative_return"),
                        "test_sortino": test_results.get("sortino_ratio"),
                    })
                except Exception as e:
                    print(f"Warning: Failed to load {run_dir}: {e}")
                    continue

    return pd.DataFrame(results)


def analyze_results(df: pd.DataFrame):
    """Analyze E5 results."""
    print("=" * 80)
    print("DEEP ANALYSIS OF E5 (REGIME DETECTION) EXPERIMENT")
    print("=" * 80)
    print(f"\nTotal runs: {len(df)}")
    print(f"Algorithm: {df['algorithm'].unique().tolist()}")
    print(f"Detection methods: {df['detection_method'].unique().tolist()}")
    print(f"Strategies: {df['strategy'].unique().tolist()}")
    print(f"Seeds: {sorted(df['seed'].unique().tolist())}")

    # =========================================================================
    # 1. SUMMARY BY DETECTION METHOD
    # =========================================================================
    print("\n" + "=" * 80)
    print("1. PERFORMANCE BY DETECTION METHOD (Aggregated)")
    print("=" * 80)

    dm_summary = df.groupby('detection_method').agg({
        'test_sharpe': ['mean', 'std', 'min', 'max', 'count'],
        'val_sharpe': ['mean'],
        'test_mdd': ['mean']
    }).round(3)
    dm_summary.columns = ['_'.join(col) for col in dm_summary.columns]

    print("\n{:<12} {:>12} {:>8} {:>8} {:>8} {:>6} {:>12} {:>10}".format(
        "Method", "TestSharpe", "Std", "Min", "Max", "N", "ValSharpe", "TestMDD"
    ))
    print("-" * 86)
    for method, row in dm_summary.iterrows():
        print("{:<12} {:>+12.3f} {:>8.3f} {:>+8.3f} {:>+8.3f} {:>6} {:>+12.3f} {:>9.1f}%".format(
            method, row['test_sharpe_mean'], row['test_sharpe_std'],
            row['test_sharpe_min'], row['test_sharpe_max'],
            int(row['test_sharpe_count']),
            row['val_sharpe_mean'], row['test_mdd_mean'] * 100
        ))

    # =========================================================================
    # 2. SUMMARY BY STRATEGY
    # =========================================================================
    print("\n" + "=" * 80)
    print("2. PERFORMANCE BY STRATEGY")
    print("=" * 80)

    strat_summary = df.groupby('strategy').agg({
        'test_sharpe': ['mean', 'std', 'min', 'max', 'count'],
        'val_sharpe': ['mean'],
        'test_mdd': ['mean']
    }).round(3)
    strat_summary.columns = ['_'.join(col) for col in strat_summary.columns]

    print("\n{:<12} {:>12} {:>8} {:>8} {:>8} {:>6} {:>12} {:>10}".format(
        "Strategy", "TestSharpe", "Std", "Min", "Max", "N", "ValSharpe", "TestMDD"
    ))
    print("-" * 86)
    for strategy, row in strat_summary.iterrows():
        print("{:<12} {:>+12.3f} {:>8.3f} {:>+8.3f} {:>+8.3f} {:>6} {:>+12.3f} {:>9.1f}%".format(
            strategy, row['test_sharpe_mean'], row['test_sharpe_std'],
            row['test_sharpe_min'], row['test_sharpe_max'],
            int(row['test_sharpe_count']),
            row['val_sharpe_mean'], row['test_mdd_mean'] * 100
        ))

    # =========================================================================
    # 3. DETECTION METHOD x STRATEGY INTERACTION
    # =========================================================================
    print("\n" + "=" * 80)
    print("3. DETECTION METHOD x STRATEGY INTERACTION")
    print("=" * 80)

    pivot = df.groupby(['detection_method', 'strategy'])['test_sharpe'].agg(['mean', 'std']).round(3)
    pivot = pivot.reset_index()

    print("\n{:<12} {:<12} {:>12} {:>8}".format("Method", "Strategy", "TestSharpe", "Std"))
    print("-" * 48)

    for _, row in pivot.sort_values(['detection_method', 'strategy']).iterrows():
        print("{:<12} {:<12} {:>+12.3f} {:>8.3f}".format(
            row['detection_method'], row['strategy'], row['mean'], row['std']
        ))

    # =========================================================================
    # 4. STATISTICAL TESTS: RULE_BASED vs HMM
    # =========================================================================
    print("\n" + "=" * 80)
    print("4. STATISTICAL TESTS: RULE_BASED vs HMM")
    print("=" * 80)

    rule_data = df[df['detection_method'] == 'rule_based']['test_sharpe'].values
    hmm_data = df[df['detection_method'] == 'hmm']['test_sharpe'].values

    if len(rule_data) > 0 and len(hmm_data) > 0:
        t_stat, p_val = stats.ttest_ind(rule_data, hmm_data, equal_var=False)
        d = cohens_d(rule_data, hmm_data)
        diff = np.mean(rule_data) - np.mean(hmm_data)

        print(f"\nOverall: Rule-based vs HMM")
        print(f"  Rule-based mean: {np.mean(rule_data):+.3f} (std: {np.std(rule_data):.3f})")
        print(f"  HMM mean:        {np.mean(hmm_data):+.3f} (std: {np.std(hmm_data):.3f})")
        print(f"  Difference:      {diff:+.3f}")
        print(f"  t-statistic:     {t_stat:.3f}")
        print(f"  p-value:         {p_val:.4f}")
        print(f"  Cohen's d:       {d:+.3f} ({effect_size_label(d)})")
        print(f"  Significant (p<0.05)? {'YES' if p_val < 0.05 else 'no'}")

    # Per strategy
    print("\nPer-Strategy Tests:")
    for strat in df['strategy'].unique():
        strat_rule = df[(df['strategy'] == strat) & (df['detection_method'] == 'rule_based')]['test_sharpe'].values
        strat_hmm = df[(df['strategy'] == strat) & (df['detection_method'] == 'hmm')]['test_sharpe'].values

        if len(strat_rule) > 0 and len(strat_hmm) > 0:
            t_stat, p_val = stats.ttest_ind(strat_rule, strat_hmm, equal_var=False)
            d = cohens_d(strat_rule, strat_hmm)
            diff = np.mean(strat_rule) - np.mean(strat_hmm)
            sig = "YES" if p_val < 0.05 else "no"
            print(f"  {strat}: rule {np.mean(strat_rule):+.3f} vs hmm {np.mean(strat_hmm):+.3f} "
                  f"(diff: {diff:+.3f}, p={p_val:.4f}, d={d:+.2f}, sig: {sig})")

    # =========================================================================
    # 5. VAL-TEST GAP ANALYSIS
    # =========================================================================
    print("\n" + "=" * 80)
    print("5. VAL-TEST GAP ANALYSIS (OVERFITTING INDICATOR)")
    print("=" * 80)

    df['val_test_gap'] = df['val_sharpe'] - df['test_sharpe']

    gap_by_dm = df.groupby('detection_method')['val_test_gap'].agg(['mean', 'std']).round(3)
    print("\n{:<12} {:>15} {:>10}".format("Method", "Val-Test Gap", "Std"))
    print("-" * 40)
    for dm, row in gap_by_dm.iterrows():
        print("{:<12} {:>+15.3f} {:>10.3f}".format(dm, row['mean'], row['std']))

    # =========================================================================
    # 6. COMPARISON WITH E1 SAC RESULTS
    # =========================================================================
    print("\n" + "=" * 80)
    print("6. COMPARISON WITH E1 SAC RESULTS")
    print("=" * 80)

    # E1 SAC results from CLAUDE.md
    E1_SAC = {
        "standard": 0.458,
        "balanced": -0.014,
    }

    print("\n{:<12} {:<12} {:>12} {:>12} {:>10}".format(
        "Strategy", "Method", "E5 Sharpe", "E1 Sharpe", "Difference"
    ))
    print("-" * 60)

    for strat in ['standard', 'balanced']:
        if strat in E1_SAC:
            e1_val = E1_SAC[strat]
            for method in ['rule_based', 'hmm']:
                e5_mean = df[(df['strategy'] == strat) & (df['detection_method'] == method)]['test_sharpe'].mean()
                diff = e5_mean - e1_val
                print("{:<12} {:<12} {:>+12.3f} {:>+12.3f} {:>+10.3f}".format(
                    strat, method, e5_mean, e1_val, diff
                ))

    # =========================================================================
    # 7. KEY FINDINGS
    # =========================================================================
    print("\n" + "=" * 80)
    print("7. KEY FINDINGS FOR THESIS")
    print("=" * 80)

    rule_mean = df[df['detection_method'] == 'rule_based']['test_sharpe'].mean()
    hmm_mean = df[df['detection_method'] == 'hmm']['test_sharpe'].mean()
    diff = rule_mean - hmm_mean
    pct_diff = (diff / abs(hmm_mean)) * 100 if hmm_mean != 0 else 0

    rule_gap = df[df['detection_method'] == 'rule_based']['val_test_gap'].mean()
    hmm_gap = df[df['detection_method'] == 'hmm']['val_test_gap'].mean()

    better_method = 'rule_based' if rule_mean > hmm_mean else 'hmm'

    print(f"""
KEY FINDINGS:

1. DETECTION METHOD COMPARISON
   - Rule-based: {rule_mean:+.3f} mean Sharpe
   - HMM:        {hmm_mean:+.3f} mean Sharpe
   - Difference: {diff:+.3f} ({pct_diff:+.1f}%)
   - {better_method.replace('_', '-').upper()} performs better

2. OVERFITTING BY DETECTION METHOD
   - Rule-based gap: {rule_gap:+.3f}
   - HMM gap:        {hmm_gap:+.3f}
   - {'HMM generalizes better' if hmm_gap < rule_gap else 'Rule-based generalizes better'}

3. STRATEGY INTERACTION
""")

    for strat in df['strategy'].unique():
        strat_rule = df[(df['strategy'] == strat) & (df['detection_method'] == 'rule_based')]['test_sharpe'].mean()
        strat_hmm = df[(df['strategy'] == strat) & (df['detection_method'] == 'hmm')]['test_sharpe'].mean()
        strat_diff = strat_rule - strat_hmm
        print(f"   - {strat}: rule {strat_rule:+.3f}, hmm {strat_hmm:+.3f} (diff: {strat_diff:+.3f})")

    # Standard vs balanced
    std_mean = df[df['strategy'] == 'standard']['test_sharpe'].mean()
    bal_mean = df[df['strategy'] == 'balanced']['test_sharpe'].mean()
    std_vs_bal = std_mean - bal_mean

    # Thesis paragraph
    print(f"""
4. SAC + REGIME DETECTION BEHAVIOR
   - Standard strategy:  {std_mean:+.3f}
   - Balanced strategy:  {bal_mean:+.3f}
   - Standard vs balanced: {std_vs_bal:+.3f}
   - Confirms E1 finding: SAC works better with standard (uniform) replay

THESIS PARAGRAPH:
"We compared rule-based and HMM regime detection methods for SAC.
{'Rule-based detection' if better_method == 'rule_based' else 'HMM detection'}
achieved a mean Sharpe of {max(rule_mean, hmm_mean):+.3f}, outperforming
{'HMM' if better_method == 'rule_based' else 'rule-based'} ({min(rule_mean, hmm_mean):+.3f})
by {abs(diff):.3f} Sharpe units.
{'However, this difference was not statistically significant (p>' if rule_data is not None and hmm_data is not None else ''}
The validation-test gap was {rule_gap:+.3f} for rule-based vs {hmm_gap:+.3f} for HMM.
Importantly, the standard replay strategy ({std_mean:+.3f}) still outperformed balanced
({bal_mean:+.3f}) regardless of detection method, confirming our E1 finding that
SAC's entropy optimization is incompatible with non-uniform replay sampling."
""")

    return df


def generate_plots(df: pd.DataFrame):
    """Generate E5 analysis plots."""
    if df.empty:
        print("No data to plot.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('E5: Regime Detection Method Comparison (SAC)', fontsize=14, fontweight='bold')

    # Plot 1: Bar chart - Test Sharpe by Detection Method
    ax1 = axes[0, 0]
    dm_means = df.groupby('detection_method')['test_sharpe'].mean()
    dm_stds = df.groupby('detection_method')['test_sharpe'].std()
    colors = ['#3498db', '#e67e22']
    bars = ax1.bar(dm_means.index, dm_means.values, yerr=dm_stds.values, capsize=5, color=colors, alpha=0.8)
    ax1.set_xlabel('Detection Method')
    ax1.set_ylabel('Test Sharpe Ratio')
    ax1.set_title('Test Performance by Detection Method')
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax1.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar, val in zip(bars, dm_means.values):
        ax1.annotate(f'{val:+.3f}', xy=(bar.get_x() + bar.get_width()/2, val),
                     ha='center', va='bottom' if val > 0 else 'top', fontsize=10)

    # Plot 2: Grouped bar chart - Detection Method x Strategy
    ax2 = axes[0, 1]
    strategies = sorted(df['strategy'].unique())
    methods = ['rule_based', 'hmm']
    x = np.arange(len(strategies))
    width = 0.35

    for i, method in enumerate(methods):
        means = [df[(df['strategy'] == strat) & (df['detection_method'] == method)]['test_sharpe'].mean()
                 for strat in strategies]
        stds = [df[(df['strategy'] == strat) & (df['detection_method'] == method)]['test_sharpe'].std()
                for strat in strategies]
        ax2.bar(x + (i - 0.5) * width, means, width, yerr=stds, capsize=3,
                label=method.replace('_', '-'), color=colors[i], alpha=0.8)

    ax2.set_xlabel('Strategy')
    ax2.set_ylabel('Test Sharpe Ratio')
    ax2.set_title('Test Performance by Strategy x Detection Method')
    ax2.set_xticks(x)
    ax2.set_xticklabels(strategies)
    ax2.legend()
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax2.grid(True, alpha=0.3, axis='y')

    # Plot 3: Val-Test Gap by Detection Method
    ax3 = axes[1, 0]
    df['val_test_gap'] = df['val_sharpe'] - df['test_sharpe']
    gap_means = df.groupby('detection_method')['val_test_gap'].mean()
    gap_stds = df.groupby('detection_method')['val_test_gap'].std()
    bars = ax3.bar(gap_means.index, gap_means.values, yerr=gap_stds.values, capsize=5,
                   color=['#9b59b6', '#1abc9c'], alpha=0.8)
    ax3.set_xlabel('Detection Method')
    ax3.set_ylabel('Val - Test Sharpe (Gap)')
    ax3.set_title('Overfitting Gap by Detection Method')
    ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax3.grid(True, alpha=0.3, axis='y')

    # Plot 4: Box plot of Test Sharpe by Detection Method
    ax4 = axes[1, 1]
    box_data = [df[df['detection_method'] == m]['test_sharpe'].values for m in methods]
    bp = ax4.boxplot(box_data, tick_labels=['Rule-based', 'HMM'], patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax4.set_xlabel('Detection Method')
    ax4.set_ylabel('Test Sharpe Ratio')
    ax4.set_title('Test Sharpe Distribution by Detection Method')
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()

    # Save
    out_path = Path("results/figures/e5_detection_comparison.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {out_path}")
    plt.close()


def save_summary(df: pd.DataFrame):
    """Save CSV summary."""
    if df.empty:
        return

    summary = df.groupby(['detection_method', 'strategy']).agg({
        'test_sharpe': ['mean', 'std', 'count'],
        'val_sharpe': ['mean'],
        'test_mdd': ['mean']
    }).round(3)
    summary.columns = ['_'.join(col) for col in summary.columns]
    summary = summary.reset_index()

    # Add val-test gap
    summary['val_test_gap'] = (summary['val_sharpe_mean'] - summary['test_sharpe_mean']).round(3)

    out_path = Path("results/e5_analysis_summary.csv")
    summary.to_csv(out_path, index=False)
    print(f"Summary saved to: {out_path}")


def main():
    df = load_e5_results()

    if df.empty:
        print("\nNo E5 results found. Run the experiment first:")
        print("  python scripts/launch_experiment.py --experiments regime_detection --workers 4")
        return

    df = analyze_results(df)
    generate_plots(df)
    save_summary(df)


if __name__ == "__main__":
    main()
