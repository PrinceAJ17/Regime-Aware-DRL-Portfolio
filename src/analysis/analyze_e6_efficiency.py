"""
Deep Analysis of E6 (Sample Efficiency) Experiment Results
===========================================================

Tests performance with extended training (1M timesteps):
- Algorithms: SAC, TD3
- Strategies: standard, balanced, inverse_frequency
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


def load_e6_results() -> pd.DataFrame:
    """Load E6 results from experiment directories."""
    results = []
    base_dir = Path("results/experiments/sample_efficiency")

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
                        "seed": config.get("seed"),
                        "timesteps": config.get("total_timesteps"),
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
    """Analyze E6 results."""
    print("=" * 80)
    print("DEEP ANALYSIS OF E6 (SAMPLE EFFICIENCY) EXPERIMENT")
    print("=" * 80)
    print(f"\nTotal runs: {len(df)}")
    print(f"Algorithms: {df['algorithm'].unique().tolist()}")
    print(f"Strategies: {df['strategy'].unique().tolist()}")
    print(f"Timesteps: {df['timesteps'].unique().tolist()}")
    print(f"Seeds: {sorted(df['seed'].unique().tolist())}")

    # =========================================================================
    # 1. SUMMARY BY ALGORITHM
    # =========================================================================
    print("\n" + "=" * 80)
    print("1. PERFORMANCE BY ALGORITHM (1M timesteps)")
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

    print("\n{:<18} {:>12} {:>8} {:>8} {:>8} {:>6} {:>12} {:>10}".format(
        "Strategy", "TestSharpe", "Std", "Min", "Max", "N", "ValSharpe", "TestMDD"
    ))
    print("-" * 92)
    for strategy, row in strat_summary.iterrows():
        print("{:<18} {:>+12.3f} {:>8.3f} {:>+8.3f} {:>+8.3f} {:>6} {:>+12.3f} {:>9.1f}%".format(
            strategy, row['test_sharpe_mean'], row['test_sharpe_std'],
            row['test_sharpe_min'], row['test_sharpe_max'],
            int(row['test_sharpe_count']),
            row['val_sharpe_mean'], row['test_mdd_mean'] * 100
        ))

    # =========================================================================
    # 3. ALGORITHM x STRATEGY INTERACTION
    # =========================================================================
    print("\n" + "=" * 80)
    print("3. ALGORITHM x STRATEGY INTERACTION")
    print("=" * 80)

    pivot = df.groupby(['algorithm', 'strategy'])['test_sharpe'].agg(['mean', 'std']).round(3)
    pivot = pivot.reset_index()

    print("\n{:<10} {:<18} {:>12} {:>8}".format("Algorithm", "Strategy", "TestSharpe", "Std"))
    print("-" * 52)

    for _, row in pivot.sort_values(['algorithm', 'strategy']).iterrows():
        print("{:<10} {:<18} {:>+12.3f} {:>8.3f}".format(
            row['algorithm'].upper(), row['strategy'], row['mean'], row['std']
        ))

    # =========================================================================
    # 4. COMPARISON WITH E1 (500k vs 1M timesteps)
    # =========================================================================
    print("\n" + "=" * 80)
    print("4. COMPARISON WITH E1 (500k timesteps baseline)")
    print("=" * 80)

    # E1 results (500k timesteps baseline)
    E1_RESULTS = {
        "td3": {
            "standard": 0.632,
            "balanced": 0.438,
            "inverse_frequency": 0.254,
        },
        "sac": {
            "standard": 0.458,
            "balanced": -0.014,
            "inverse_frequency": 0.134,
        }
    }

    print("\n{:<10} {:<18} {:>12} {:>12} {:>10} {:>12}".format(
        "Algorithm", "Strategy", "E6 (1M)", "E1 (500k)", "Diff", "Improve?"
    ))
    print("-" * 78)

    for algo in df['algorithm'].unique():
        if algo in E1_RESULTS:
            for strat in df['strategy'].unique():
                if strat in E1_RESULTS[algo]:
                    e1_val = E1_RESULTS[algo][strat]
                    e6_mean = df[(df['algorithm'] == algo) & (df['strategy'] == strat)]['test_sharpe'].mean()
                    diff = e6_mean - e1_val
                    improve = "YES" if diff > 0 else "no"
                    print("{:<10} {:<18} {:>+12.3f} {:>+12.3f} {:>+10.3f} {:>12}".format(
                        algo.upper(), strat, e6_mean, e1_val, diff, improve
                    ))

    # =========================================================================
    # 5. STATISTICAL TESTS: SAC vs TD3
    # =========================================================================
    print("\n" + "=" * 80)
    print("5. STATISTICAL TESTS: SAC vs TD3")
    print("=" * 80)

    sac_data = df[df['algorithm'] == 'sac']['test_sharpe'].values
    td3_data = df[df['algorithm'] == 'td3']['test_sharpe'].values

    if len(sac_data) > 0 and len(td3_data) > 0:
        t_stat, p_val = stats.ttest_ind(sac_data, td3_data, equal_var=False)
        d = cohens_d(td3_data, sac_data)  # TD3 - SAC
        diff = np.mean(td3_data) - np.mean(sac_data)

        print(f"\nOverall: TD3 vs SAC")
        print(f"  TD3 mean: {np.mean(td3_data):+.3f} (std: {np.std(td3_data):.3f})")
        print(f"  SAC mean: {np.mean(sac_data):+.3f} (std: {np.std(sac_data):.3f})")
        print(f"  Difference (TD3 - SAC): {diff:+.3f}")
        print(f"  t-statistic: {t_stat:.3f}")
        print(f"  p-value:     {p_val:.4f}")
        print(f"  Cohen's d:   {d:+.3f} ({effect_size_label(d)})")
        print(f"  Significant (p<0.05)? {'YES' if p_val < 0.05 else 'no'}")

    # Per strategy
    print("\nPer-Strategy Comparison:")
    for strat in df['strategy'].unique():
        strat_sac = df[(df['strategy'] == strat) & (df['algorithm'] == 'sac')]['test_sharpe'].values
        strat_td3 = df[(df['strategy'] == strat) & (df['algorithm'] == 'td3')]['test_sharpe'].values

        if len(strat_sac) > 0 and len(strat_td3) > 0:
            t_stat, p_val = stats.ttest_ind(strat_sac, strat_td3, equal_var=False)
            d = cohens_d(strat_td3, strat_sac)
            diff = np.mean(strat_td3) - np.mean(strat_sac)
            sig = "YES" if p_val < 0.05 else "no"
            print(f"  {strat}: TD3 {np.mean(strat_td3):+.3f} vs SAC {np.mean(strat_sac):+.3f} "
                  f"(diff: {diff:+.3f}, p={p_val:.4f}, d={d:+.2f}, sig: {sig})")

    # =========================================================================
    # 6. VAL-TEST GAP ANALYSIS
    # =========================================================================
    print("\n" + "=" * 80)
    print("6. VAL-TEST GAP ANALYSIS (OVERFITTING with 1M timesteps)")
    print("=" * 80)

    df['val_test_gap'] = df['val_sharpe'] - df['test_sharpe']

    gap_by_algo = df.groupby('algorithm')['val_test_gap'].agg(['mean', 'std']).round(3)
    print("\n{:<10} {:>15} {:>10}".format("Algorithm", "Val-Test Gap", "Std"))
    print("-" * 38)
    for algo, row in gap_by_algo.iterrows():
        print("{:<10} {:>+15.3f} {:>10.3f}".format(algo.upper(), row['mean'], row['std']))

    # E1 gaps for comparison (approximate from known val-test differences)
    E1_GAPS = {"sac": 1.1, "td3": 0.9}

    print("\nComparison with E1 (500k) gaps:")
    for algo in df['algorithm'].unique():
        if algo in E1_GAPS:
            e6_gap = df[df['algorithm'] == algo]['val_test_gap'].mean()
            e1_gap = E1_GAPS[algo]
            diff = e6_gap - e1_gap
            print(f"  {algo.upper()}: E6 gap {e6_gap:+.3f} vs E1 gap ~{e1_gap:+.1f} "
                  f"(diff: {diff:+.3f}, {'more' if diff > 0 else 'less'} overfitting)")

    # =========================================================================
    # 7. KEY FINDINGS
    # =========================================================================
    print("\n" + "=" * 80)
    print("7. KEY FINDINGS FOR THESIS")
    print("=" * 80)

    td3_mean = df[df['algorithm'] == 'td3']['test_sharpe'].mean()
    sac_mean = df[df['algorithm'] == 'sac']['test_sharpe'].mean()
    overall_mean = df['test_sharpe'].mean()

    td3_gap = df[df['algorithm'] == 'td3']['val_test_gap'].mean()
    sac_gap = df[df['algorithm'] == 'sac']['val_test_gap'].mean()

    # E1 comparison
    e1_td3_mean = np.mean([E1_RESULTS['td3'][s] for s in E1_RESULTS['td3']])
    e1_sac_mean = np.mean([E1_RESULTS['sac'][s] for s in E1_RESULTS['sac']])
    td3_improvement = td3_mean - e1_td3_mean
    sac_improvement = sac_mean - e1_sac_mean

    print(f"""
KEY FINDINGS:

1. EXTENDED TRAINING (1M timesteps) EFFECT
   - Overall mean Sharpe: {overall_mean:+.3f}
   - TD3: {td3_mean:+.3f} (E1 500k: {e1_td3_mean:+.3f}, diff: {td3_improvement:+.3f})
   - SAC: {sac_mean:+.3f} (E1 500k: {e1_sac_mean:+.3f}, diff: {sac_improvement:+.3f})
   - {'Longer training helped' if td3_improvement > 0 and sac_improvement > 0 else 'Longer training did not help'}

2. ALGORITHM RANKING (1M timesteps)
   - TD3 mean: {td3_mean:+.3f}
   - SAC mean: {sac_mean:+.3f}
   - TD3 {'better' if td3_mean > sac_mean else 'worse'} by {abs(td3_mean - sac_mean):.3f}

3. OVERFITTING WITH EXTENDED TRAINING
   - TD3 gap: {td3_gap:+.3f}
   - SAC gap: {sac_gap:+.3f}
   - Gaps are {'larger' if max(td3_gap, sac_gap) > 1.2 else 'similar'} to E1

4. STRATEGY RANKING
""")

    for strat in sorted(df['strategy'].unique()):
        strat_mean = df[df['strategy'] == strat]['test_sharpe'].mean()
        print(f"   - {strat}: {strat_mean:+.3f}")

    best_strat = df.groupby('strategy')['test_sharpe'].mean().idxmax()
    best_strat_val = df.groupby('strategy')['test_sharpe'].mean().max()

    # Thesis paragraph
    print(f"""
THESIS PARAGRAPH:
"We extended training to 1 million timesteps to evaluate sample efficiency.
TD3 achieved a mean Sharpe of {td3_mean:+.3f}, {'improving' if td3_improvement > 0 else 'declining'}
by {abs(td3_improvement):.3f} from the 500k baseline. SAC achieved {sac_mean:+.3f},
{'improving' if sac_improvement > 0 else 'declining'} by {abs(sac_improvement):.3f}.
The validation-test gap {'increased' if max(td3_gap, sac_gap) > 1.2 else 'remained stable'} at
{td3_gap:+.3f} (TD3) and {sac_gap:+.3f} (SAC), suggesting that
{'longer training leads to increased overfitting' if max(td3_gap, sac_gap) > 1.2 else 'overfitting is independent of training duration'}.
The {best_strat} strategy performed best ({best_strat_val:+.3f}), consistent with E1 findings.
This suggests that {'500k timesteps may be sufficient' if td3_improvement < 0.1 else '1M timesteps provides marginal improvement'}
for this task."
""")

    return df


def generate_plots(df: pd.DataFrame):
    """Generate E6 analysis plots."""
    if df.empty:
        print("No data to plot.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('E6: Sample Efficiency Analysis (1M timesteps)', fontsize=14, fontweight='bold')

    # Plot 1: Bar chart - Test Sharpe by Algorithm
    ax1 = axes[0, 0]
    algo_means = df.groupby('algorithm')['test_sharpe'].mean()
    algo_stds = df.groupby('algorithm')['test_sharpe'].std()
    colors = ['#3498db', '#e74c3c']
    bars = ax1.bar(algo_means.index, algo_means.values, yerr=algo_stds.values, capsize=5, color=colors, alpha=0.8)
    ax1.set_xlabel('Algorithm')
    ax1.set_ylabel('Test Sharpe Ratio')
    ax1.set_title('Test Performance by Algorithm (1M timesteps)')
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax1.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar, val in zip(bars, algo_means.values):
        ax1.annotate(f'{val:+.3f}', xy=(bar.get_x() + bar.get_width()/2, val),
                     ha='center', va='bottom' if val > 0 else 'top', fontsize=10)

    # Plot 2: Grouped bar chart - Algorithm x Strategy
    ax2 = axes[0, 1]
    strategies = sorted(df['strategy'].unique())
    algorithms = sorted(df['algorithm'].unique())
    x = np.arange(len(strategies))
    width = 0.35

    for i, algo in enumerate(algorithms):
        means = [df[(df['strategy'] == strat) & (df['algorithm'] == algo)]['test_sharpe'].mean()
                 for strat in strategies]
        stds = [df[(df['strategy'] == strat) & (df['algorithm'] == algo)]['test_sharpe'].std()
                for strat in strategies]
        ax2.bar(x + (i - 0.5) * width, means, width, yerr=stds, capsize=3,
                label=algo.upper(), color=colors[i], alpha=0.8)

    ax2.set_xlabel('Strategy')
    ax2.set_ylabel('Test Sharpe Ratio')
    ax2.set_title('Test Performance by Strategy x Algorithm')
    ax2.set_xticks(x)
    ax2.set_xticklabels(strategies, rotation=15)
    ax2.legend()
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax2.grid(True, alpha=0.3, axis='y')

    # Plot 3: Comparison with E1 (500k baseline)
    ax3 = axes[1, 0]
    E1_RESULTS = {
        "td3": {"standard": 0.632, "balanced": 0.438, "inverse_frequency": 0.254},
        "sac": {"standard": 0.458, "balanced": -0.014, "inverse_frequency": 0.134}
    }

    # Create comparison data
    comparison_data = []
    for algo in algorithms:
        if algo in E1_RESULTS:
            e1_mean = np.mean([E1_RESULTS[algo][s] for s in E1_RESULTS[algo]])
            e6_mean = df[df['algorithm'] == algo]['test_sharpe'].mean()
            comparison_data.append({
                'algo': algo.upper(),
                'e1': e1_mean,
                'e6': e6_mean
            })

    if comparison_data:
        comp_df = pd.DataFrame(comparison_data)
        x_comp = np.arange(len(comp_df))
        ax3.bar(x_comp - 0.2, comp_df['e1'], 0.4, label='E1 (500k)', color='#95a5a6', alpha=0.8)
        ax3.bar(x_comp + 0.2, comp_df['e6'], 0.4, label='E6 (1M)', color='#2ecc71', alpha=0.8)
        ax3.set_xlabel('Algorithm')
        ax3.set_ylabel('Test Sharpe Ratio')
        ax3.set_title('E1 (500k) vs E6 (1M) Comparison')
        ax3.set_xticks(x_comp)
        ax3.set_xticklabels(comp_df['algo'])
        ax3.legend()
        ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax3.grid(True, alpha=0.3, axis='y')

    # Plot 4: Box plot of Test Sharpe by Algorithm
    ax4 = axes[1, 1]
    box_data = [df[df['algorithm'] == algo]['test_sharpe'].values for algo in algorithms]
    bp = ax4.boxplot(box_data, tick_labels=[a.upper() for a in algorithms], patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax4.set_xlabel('Algorithm')
    ax4.set_ylabel('Test Sharpe Ratio')
    ax4.set_title('Test Sharpe Distribution (1M timesteps)')
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()

    # Save
    out_path = Path("results/figures/e6_sample_efficiency.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {out_path}")
    plt.close()


def save_summary(df: pd.DataFrame):
    """Save CSV summary."""
    if df.empty:
        return

    summary = df.groupby(['algorithm', 'strategy']).agg({
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

    out_path = Path("results/e6_analysis_summary.csv")
    summary.to_csv(out_path, index=False)
    print(f"Summary saved to: {out_path}")


def main():
    df = load_e6_results()

    if df.empty:
        print("\nNo E6 results found. Run the experiment first:")
        print("  python scripts/launch_experiment.py --experiments sample_efficiency --workers 4")
        return

    df = analyze_results(df)
    generate_plots(df)
    save_summary(df)


if __name__ == "__main__":
    main()
