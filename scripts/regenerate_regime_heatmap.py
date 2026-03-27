"""
Regenerate e2_regime_heatmap.png with Sharpe ratios instead of mean rewards.

Uses the same data loading and Sharpe calculation methodology as analyze_e2_regime.py.
Produces a single heatmap: rows = algorithm/strategy, columns = bull/bear/sideways.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Constants (matching analyze_e2_regime.py exactly)
RESULTS_DIR = Path("results/experiments/replay_comparison")
FIGURES_DIR = Path("results/figures")
REPORT_FIGURES_DIR = Path("report/figures")
ALGORITHMS = ["ppo", "sac", "td3", "rainbow"]
STRATEGIES = ["standard", "balanced", "current_focused", "inverse_frequency", "adaptive"]
SEEDS = [42, 123, 456, 789, 1000]
TRADING_DAYS_PER_YEAR = 252
MIN_DAYS_PER_REGIME = 10


def normalize_regime(r) -> str:
    if isinstance(r, str):
        return r.lower()
    return {0: "bull", 1: "bear", 2: "sideways"}.get(r, "unknown")


def calculate_regime_sharpe(returns: np.ndarray) -> float:
    """Calculate annualized Sharpe from regime-specific returns.
    Identical to analyze_e2_regime.py lines 81-96."""
    if len(returns) < 5:
        return np.nan
    mean_return = np.mean(returns)
    std_return = np.std(returns, ddof=1)
    if std_return == 0:
        return 0.0
    annualized_return = mean_return * TRADING_DAYS_PER_YEAR
    annualized_vol = std_return * np.sqrt(TRADING_DAYS_PER_YEAR)
    return annualized_return / annualized_vol


def load_and_compute():
    """Load all 100 E1 runs, compute per-regime Sharpe for each config."""
    print("Loading E1 data and computing per-regime Sharpe ratios...")

    regime_metrics = []
    loaded = 0

    for algo in ALGORITHMS:
        for strategy in STRATEGIES:
            for seed in SEEDS:
                run_id = f"replay_comparison__{algo}__{strategy}__seed{seed}"
                path = RESULTS_DIR / algo / strategy / run_id / "test_episode_data.json"

                if not path.exists():
                    print(f"  Missing: {path}")
                    continue

                with open(path) as f:
                    data = json.load(f)

                portfolio_values = data.get("portfolio_values", [])
                step_returns = data.get("step_returns", [])
                regime_labels = data.get("regime_labels", [])
                n_episodes = len(portfolio_values)

                for ep_idx in range(n_episodes):
                    pv = np.array(portfolio_values[ep_idx])
                    sr = np.array(step_returns[ep_idx]) if ep_idx < len(step_returns) else np.array([])
                    rl = regime_labels[ep_idx] if ep_idx < len(regime_labels) else []
                    rl_norm = [normalize_regime(r) for r in rl]

                    # Align lengths
                    min_len = min(len(sr), len(rl_norm), len(pv) - 1)
                    sr = sr[:min_len]
                    rl_norm = rl_norm[:min_len]

                    # Group returns by regime
                    regime_returns = defaultdict(list)
                    for ret, regime in zip(sr, rl_norm):
                        regime_returns[regime].append(ret)

                    # Compute Sharpe per regime
                    for regime in ["bull", "bear", "sideways"]:
                        returns = np.array(regime_returns.get(regime, []))
                        if len(returns) < MIN_DAYS_PER_REGIME:
                            continue
                        sharpe = calculate_regime_sharpe(returns)
                        regime_metrics.append({
                            "algorithm": algo,
                            "strategy": strategy,
                            "seed": seed,
                            "episode": ep_idx,
                            "regime": regime,
                            "sharpe": sharpe
                        })

                loaded += 1

    print(f"Loaded {loaded}/100 runs, {len(regime_metrics)} regime-episode records")
    return pd.DataFrame(regime_metrics)


def validate_against_known(metrics_df):
    """Cross-validate computed values against known CSV data."""
    print("\n--- Validation against e2_regime_sharpe_by_algo.csv ---")
    known = pd.read_csv("results/tables/e2_regime_sharpe_by_algo.csv")

    algo_regime = metrics_df.groupby(["algorithm", "regime"])["sharpe"].mean().reset_index()

    for _, krow in known.iterrows():
        computed = algo_regime[
            (algo_regime["algorithm"] == krow["algorithm"]) &
            (algo_regime["regime"] == krow["regime"])
        ]["sharpe"].values

        if len(computed) > 0:
            diff = abs(computed[0] - krow["sharpe_mean"])
            status = "OK" if diff < 0.01 else f"MISMATCH (diff={diff:.4f})"
            print(f"  {krow['algorithm']:8s} {krow['regime']:10s}: known={krow['sharpe_mean']:+.4f}, computed={computed[0]:+.4f} -> {status}")
        else:
            print(f"  {krow['algorithm']:8s} {krow['regime']:10s}: MISSING")

    # Also validate specific known values from CLAUDE.md
    print("\n--- Validation against CLAUDE.md key values ---")
    # TD3 bear mean should be ~+0.931
    td3_bear = algo_regime[(algo_regime["algorithm"] == "td3") & (algo_regime["regime"] == "bear")]["sharpe"].values[0]
    print(f"  TD3 bear: {td3_bear:+.4f} (expected ~+0.931)")

    # Bull aggregate should be ~+1.564
    bull_all = metrics_df[metrics_df["regime"] == "bull"]["sharpe"].mean()
    print(f"  Bull aggregate: {bull_all:+.4f} (expected ~+1.564)")

    # Sideways aggregate should be ~-1.286
    sideways_all = metrics_df[metrics_df["regime"] == "sideways"]["sharpe"].mean()
    print(f"  Sideways aggregate: {sideways_all:+.4f} (expected ~-1.286)")

    # Bear aggregate should be ~+0.487
    bear_all = metrics_df[metrics_df["regime"] == "bear"]["sharpe"].mean()
    print(f"  Bear aggregate: {bear_all:+.4f} (expected ~+0.487)")


def generate_heatmap(metrics_df):
    """Generate the heatmap matching the current figure layout but with Sharpe ratios."""

    # Aggregate: mean Sharpe per algorithm × strategy × regime
    config_regime = metrics_df.groupby(["algorithm", "strategy", "regime"])["sharpe"].mean().reset_index()

    # Create row labels matching current figure: "ALGO / strategy"
    algo_display = {"ppo": "PPO", "sac": "SAC", "td3": "TD3", "rainbow": "Rainbow"}
    strategy_display = {
        "standard": "standard",
        "balanced": "balanced",
        "current_focused": "current_focused",
        "inverse_frequency": "inverse_freq",
        "adaptive": "adaptive"
    }

    config_regime["row_label"] = config_regime.apply(
        lambda r: f"{algo_display[r['algorithm']]} / {strategy_display[r['strategy']]}", axis=1
    )

    # Pivot to heatmap format
    pivot = config_regime.pivot(index="row_label", columns="regime", values="sharpe")
    pivot = pivot[["bull", "bear", "sideways"]]  # Column order
    pivot.columns = ["Bull", "Bear", "Sideways"]

    # Row order: PPO configs, SAC configs, TD3 configs, Rainbow configs
    row_order = []
    for algo in ["PPO", "SAC", "TD3", "Rainbow"]:
        for strat in ["standard", "balanced", "current_focused", "inverse_freq", "adaptive"]:
            row_order.append(f"{algo} / {strat}")

    pivot = pivot.reindex(row_order)

    # Generate figure
    fig, ax = plt.subplots(figsize=(10, 12))

    sns.heatmap(
        pivot,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        center=0,
        vmin=-2.0,
        vmax=2.0,
        linewidths=0.5,
        linecolor="white",
        ax=ax,
        cbar_kws={"label": "Annualized Sharpe Ratio", "shrink": 0.8}
    )

    ax.set_title("E2: Sharpe Ratio by Algorithm × Strategy × Regime", fontsize=14, fontweight="bold")
    ax.set_xlabel("Market Regime", fontsize=12)
    ax.set_ylabel("Algorithm / Strategy", fontsize=12)

    # Add horizontal lines to separate algorithm groups
    for i in range(1, 4):
        ax.axhline(y=i * 5, color="black", linewidth=2)

    plt.tight_layout()

    # Save
    out_path = FIGURES_DIR / "e2_regime_heatmap.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {out_path}")

    # Copy to report/figures/
    report_path = REPORT_FIGURES_DIR / "e2_regime_heatmap.png"
    if REPORT_FIGURES_DIR.exists():
        plt.savefig(report_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {report_path}")

    plt.close()

    # Print the pivot table for verification
    print("\n--- Heatmap values (Sharpe ratios) ---")
    print(pivot.to_string())


if __name__ == "__main__":
    metrics_df = load_and_compute()
    validate_against_known(metrics_df)
    generate_heatmap(metrics_df)
    print("\nDone!")
