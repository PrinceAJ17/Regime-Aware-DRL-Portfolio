"""
Tier 2 Analysis Plots
=====================

Generates:
1. Equity curves (portfolio value over time)
2. Portfolio weight allocations
3. Regime transition analysis

Author: Prince Alexander John
Date: March 2026
"""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
from datetime import datetime

RESULTS_DIR = Path("results/experiments/replay_comparison")
FIGURES_DIR = Path("results/figures")
ASSETS = ["SPY", "QQQ", "TLT", "GLD", "BIL"]


def load_episode_data(algo: str, strategy: str, seed: int, split: str = "test") -> dict:
    """Load episode data for a specific run."""
    run_id = f"replay_comparison__{algo}__{strategy}__seed{seed}"
    path = RESULTS_DIR / algo / strategy / run_id / f"{split}_episode_data.json"
    if path.exists():
        with open(path) as f:
            data = json.load(f)
        # Parse dates - they're nested lists of ISO strings
        if "dates" in data and data["dates"]:
            # Use first episode's dates
            dates = data["dates"][0] if isinstance(data["dates"][0], list) else data["dates"]
            data["dates"] = [pd.to_datetime(d).to_pydatetime() for d in dates]
        return data
    return {}


def load_baselines() -> dict:
    """Load baseline results."""
    path = Path("results/baselines.json")
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


# =============================================================================
# 1. EQUITY CURVES
# =============================================================================

TOP5_CONFIGS = [
    ("td3", "standard",        "#2ecc71", "TD3 Standard"),
    ("td3", "current_focused", "#27ae60", "TD3 Current-Focused"),
    ("td3", "adaptive",        "#1abc9c", "TD3 Adaptive"),
    ("sac", "standard",        "#3498db", "SAC Standard"),
    ("td3", "balanced",        "#16a085", "TD3 Balanced"),
]

SEEDS = [42, 123, 456, 789, 1000]


def _load_mean_pv_across_seeds(algo, strategy):
    """Load portfolio values averaged across all seeds and episodes."""
    all_pvs = []
    for seed in SEEDS:
        data = load_episode_data(algo, strategy, seed, "test")
        if not data or "portfolio_values" not in data:
            continue
        pvs = np.array(data["portfolio_values"])
        mean_pv = pvs.mean(axis=0)  # average across episodes
        all_pvs.append(mean_pv)
    if not all_pvs:
        return None, None
    # Align lengths and average across seeds
    min_len = min(len(pv) for pv in all_pvs)
    all_pvs = np.array([pv[:min_len] for pv in all_pvs])
    # Also get dates from first successful load
    for seed in SEEDS:
        data = load_episode_data(algo, strategy, seed, "test")
        if data and "dates" in data:
            dates = data["dates"][:min_len]
            return all_pvs.mean(axis=0), dates
    return all_pvs.mean(axis=0), None


def plot_equity_curves_single(output_dir: Path):
    """Plot equity curves for top 5 configurations (mean across all seeds)."""
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 7))

    dates = None
    for algo, strategy, color, label in TOP5_CONFIGS:
        mean_pv, d = _load_mean_pv_across_seeds(algo, strategy)
        if mean_pv is None:
            continue
        if dates is None and d is not None:
            dates = d

        plot_len = min(len(dates), len(mean_pv)) if dates else len(mean_pv)
        x = dates[:plot_len] if dates else list(range(plot_len))
        ax.plot(x, mean_pv[:plot_len], color=color, lw=2.5, label=label)

    # Add SPY baseline
    baselines = load_baselines()
    if "spy_buy_hold" in baselines and dates:
        spy_values = baselines["spy_buy_hold"].get("daily_values", [])
        if spy_values:
            plot_len = min(len(spy_values), len(dates))
            ax.plot(dates[:plot_len], spy_values[:plot_len], color="#e74c3c",
                    lw=2, linestyle="--", label="SPY Buy-and-Hold")

    ax.axhline(y=1_000_000, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Portfolio Value ($)", fontsize=12)
    ax.set_title("Top 5 Configurations vs SPY (Test Period 2022-2023, Mean Across Seeds)",
                 fontsize=14, fontweight="bold")
    ax.legend(loc="upper left", fontsize=10)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x/1e6:.2f}M"))

    plt.tight_layout()
    plt.savefig(output_dir / "equity_curves_top5.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  equity_curves_top5.png")


def plot_equity_curves_comparison(output_dir: Path):
    """Plot equity curves comparing top 5 configs (mean across seeds)."""
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 7))

    dates = None
    for algo, strategy, color, label in TOP5_CONFIGS:
        mean_pv, d = _load_mean_pv_across_seeds(algo, strategy)
        if mean_pv is None:
            continue
        if dates is None and d is not None:
            dates = d

        plot_len = min(len(dates), len(mean_pv)) if dates else len(mean_pv)
        x = dates[:plot_len] if dates else list(range(plot_len))
        ax.plot(x, mean_pv[:plot_len], color=color, lw=2.5, label=label)

    # Add SPY baseline
    baselines = load_baselines()
    if "spy_buy_hold" in baselines and dates:
        spy_values = baselines["spy_buy_hold"].get("daily_values", [])
        if spy_values:
            plot_len = min(len(spy_values), len(dates))
            ax.plot(dates[:plot_len], spy_values[:plot_len], color="#e74c3c",
                    lw=2, linestyle="--", label="SPY Buy-and-Hold")

    ax.axhline(y=1_000_000, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Portfolio Value ($)", fontsize=12)
    ax.set_title("Top 5 Configurations: Equity Curves (Test Period 2022-2023)",
                 fontsize=14, fontweight="bold")
    ax.legend(loc="upper left", fontsize=10)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x/1e6:.2f}M"))

    plt.tight_layout()
    plt.savefig(output_dir / "equity_curves_top5_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  equity_curves_top5_comparison.png")


# =============================================================================
# 2. PORTFOLIO WEIGHTS
# =============================================================================

def plot_portfolio_weights(output_dir: Path):
    """Plot portfolio weight allocation over time for top 5 configs."""
    output_dir.mkdir(parents=True, exist_ok=True)

    asset_colors = ["#e74c3c", "#3498db", "#2ecc71", "#f1c40f", "#95a5a6"]

    fig, axes = plt.subplots(5, 1, figsize=(14, 20), sharex=True)

    for idx, (algo, strategy, _, label) in enumerate(TOP5_CONFIGS):
        ax = axes[idx]
        data = load_episode_data(algo, strategy, 42, "test")
        if not data or "portfolio_weights" not in data:
            ax.set_title(f"{label}: No data", fontsize=12)
            continue

        weights = np.array(data["portfolio_weights"])
        dates = data.get("dates", list(range(weights.shape[1])))
        # Average across episodes
        w = weights.mean(axis=0)

        min_len = min(len(dates), len(w))
        dates_plot = dates[:min_len]
        w = w[:min_len]

        if idx == 0:
            ax.stackplot(dates_plot, w.T, labels=ASSETS, colors=asset_colors, alpha=0.8)
        else:
            ax.stackplot(dates_plot, w.T, colors=asset_colors, alpha=0.8)
        ax.set_ylabel("Weight", fontsize=10)
        ax.set_title(f"{label}", fontsize=12, fontweight="bold")
        ax.set_ylim(0, 1)

    axes[0].legend(loc="upper right", ncol=5, fontsize=9)
    axes[-1].set_xlabel("Date", fontsize=12)
    fig.suptitle("Portfolio Allocation Over Time: Top 5 Configurations",
                 fontsize=14, fontweight="bold", y=1.01)

    plt.tight_layout()
    plt.savefig(output_dir / "portfolio_weights_top5.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  portfolio_weights_top5.png")


def plot_average_weights_by_regime(output_dir: Path):
    """Bar chart: average portfolio weights by regime for top 5 configs."""
    output_dir.mkdir(parents=True, exist_ok=True)

    int_to_name = {0: "Bull", 1: "Bear", 2: "Sideways"}
    str_to_name = {"bull": "Bull", "bear": "Bear", "sideways": "Sideways"}
    regime_colors = {"Bull": "#2ecc71", "Bear": "#e74c3c", "Sideways": "#f39c12"}

    fig, axes = plt.subplots(1, 5, figsize=(22, 5), sharey=True)

    for idx, (algo, strategy, _, label) in enumerate(TOP5_CONFIGS):
        ax = axes[idx]
        data = load_episode_data(algo, strategy, 42, "test")
        if not data or "portfolio_weights" not in data or "regime_labels" not in data:
            ax.set_title(label, fontsize=10)
            continue

        weights = np.array(data["portfolio_weights"])[0]
        regimes = data["regime_labels"][0]

        results = {"Bull": [], "Bear": [], "Sideways": []}
        for w, r in zip(weights, regimes):
            if isinstance(r, str):
                rn = str_to_name.get(r.lower(), None)
            else:
                rn = int_to_name.get(r, None)
            if rn:
                results[rn].append(w)

        x = np.arange(len(ASSETS))
        width = 0.25
        i = 0
        for regime in ["Bull", "Bear", "Sideways"]:
            ws = results.get(regime, [])
            if ws:
                mean_w = np.mean(ws, axis=0)
                ax.bar(x + i * width, mean_w, width, label=regime if idx == 0 else None,
                       color=regime_colors[regime])
                i += 1

        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.set_xticks(x + width)
        ax.set_xticklabels(ASSETS, fontsize=8, rotation=45)
        if idx == 0:
            ax.set_ylabel("Average Weight", fontsize=11)

    axes[0].legend(fontsize=9)
    fig.suptitle("Portfolio Weights by Regime: Top 5 Configurations",
                 fontsize=14, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_dir / "weights_by_regime.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  weights_by_regime.png")


# =============================================================================
# 3. REGIME ANALYSIS
# =============================================================================

def plot_regime_timeline(output_dir: Path):
    """Plot regime labels over time with top 5 portfolio value overlay."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load regime labels from first config
    ref_data = load_episode_data("td3", "standard", 42, "test")
    if not ref_data or "regime_labels" not in ref_data:
        print("  No regime data available")
        return

    regimes = ref_data["regime_labels"][0]
    dates = ref_data.get("dates", list(range(len(regimes))))
    min_len = min(len(dates), len(regimes))
    dates = dates[:min_len]
    regimes = regimes[:min_len]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                                    gridspec_kw={'height_ratios': [3, 1]})

    # Top: Portfolio values for top 5
    for algo, strategy, color, label in TOP5_CONFIGS:
        mean_pv, _ = _load_mean_pv_across_seeds(algo, strategy)
        if mean_pv is None:
            continue
        plot_len = min(len(dates), len(mean_pv))
        ax1.plot(dates[:plot_len], mean_pv[:plot_len], color=color, lw=2, label=label)

    # Add SPY buy-and-hold
    baselines = load_baselines()
    if "spy_buy_hold" in baselines:
        spy_values = baselines["spy_buy_hold"].get("daily_values", [])
        if spy_values:
            plot_len = min(len(spy_values), len(dates))
            ax1.plot(dates[:plot_len], spy_values[:plot_len], color="#e74c3c", lw=2,
                    linestyle="--", label="SPY Buy-and-Hold")

    ax1.set_ylabel("Portfolio Value ($)", fontsize=12)
    ax1.set_title("Top 5 Configurations with Regime Labels (Test Period 2022-2023)",
                  fontsize=14, fontweight="bold")
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x/1e6:.2f}M"))
    ax1.legend(loc="upper left", fontsize=9)

    # Color background by regime (handle both string and int)
    def normalize_regime(r):
        if isinstance(r, str):
            return r.lower()
        return {0: "bull", 1: "bear", 2: "sideways"}.get(r, "unknown")

    regime_colors = {"bull": "#d5f5e3", "bear": "#fadbd8", "sideways": "#fef9e7"}

    prev_regime = normalize_regime(regimes[0])
    start_idx = 0
    for i, r in enumerate(regimes):
        r_norm = normalize_regime(r)
        if r_norm != prev_regime or i == len(regimes) - 1:
            ax1.axvspan(dates[start_idx], dates[i],
                       alpha=0.3, color=regime_colors.get(prev_regime, "white"))
            start_idx = i
            prev_regime = r_norm

    # Bottom: Regime labels
    normalized_regimes = [normalize_regime(r) for r in regimes]
    ax2.fill_between(dates, 0, 1, where=[r == "bull" for r in normalized_regimes],
                     color="#2ecc71", alpha=0.7, label="Bull")
    ax2.fill_between(dates, 0, 1, where=[r == "bear" for r in normalized_regimes],
                     color="#e74c3c", alpha=0.7, label="Bear")
    ax2.fill_between(dates, 0, 1, where=[r == "sideways" for r in normalized_regimes],
                     color="#f39c12", alpha=0.7, label="Sideways")
    ax2.set_ylabel("Regime", fontsize=12)
    ax2.set_xlabel("Date", fontsize=12)
    ax2.set_yticks([])
    ax2.legend(loc="upper right", ncol=3)

    plt.tight_layout()
    plt.savefig(output_dir / "regime_timeline.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  regime_timeline.png")


def plot_regime_distribution(output_dir: Path):
    """Pie chart of regime distribution in test period."""
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_episode_data("td3", "standard", 42, "test")
    if not data or "regime_labels" not in data:
        print("  No regime data available")
        return

    regimes = data["regime_labels"][0]

    # Handle both string and int regime labels
    counts = {"bull": 0, "bear": 0, "sideways": 0}
    for r in regimes:
        r_str = str(r).lower() if isinstance(r, str) else {0: "bull", 1: "bear", 2: "sideways"}.get(r, "unknown")
        if r_str in counts:
            counts[r_str] += 1

    # Filter out zero counts
    labels = []
    sizes = []
    colors_list = []
    color_map = {"bull": "#2ecc71", "bear": "#e74c3c", "sideways": "#f39c12"}
    name_map = {"bull": "Bull", "bear": "Bear", "sideways": "Sideways"}

    for regime_id in ["bull", "bear", "sideways"]:
        if counts[regime_id] > 0:
            labels.append(name_map[regime_id])
            sizes.append(counts[regime_id])
            colors_list.append(color_map[regime_id])

    if not sizes:
        print("  No regime data to plot")
        return

    fig, ax = plt.subplots(figsize=(8, 8))
    wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors_list,
                                       autopct='%1.1f%%', startangle=90,
                                       textprops={'fontsize': 12})
    ax.set_title("Regime Distribution (Test Period 2022-2023)", fontsize=14, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_dir / "regime_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  regime_distribution.png")


def plot_sharpe_by_strategy_bars(output_dir: Path):
    """Bar chart of mean test Sharpe by algorithm and strategy."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load from E1 analysis
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from analyze_e1_deep import load_e1_results
    df = load_e1_results()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    algos = ["ppo", "sac", "td3", "rainbow"]
    strategies = ["standard", "balanced", "current_focused", "inverse_frequency", "adaptive"]
    colors = ["#3498db", "#2ecc71", "#9b59b6", "#e74c3c", "#f39c12"]

    for i, algo in enumerate(algos):
        ax = axes[i]
        algo_df = df[df["algorithm"] == algo]

        means = []
        stds = []
        for strat in strategies:
            strat_data = algo_df[algo_df["strategy"] == strat]["test_sharpe"]
            means.append(strat_data.mean())
            stds.append(strat_data.std())

        x = np.arange(len(strategies))
        bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors, alpha=0.8)

        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        ax.set_title(f"{algo.upper()}", fontsize=14, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([s.replace("_", "\n") for s in strategies], fontsize=9)
        ax.set_ylabel("Test Sharpe Ratio")

        # Add value labels
        for bar, mean in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                   f"{mean:.2f}", ha='center', va='bottom', fontsize=9)

    fig.suptitle("E1: Test Sharpe by Algorithm and Strategy", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "sharpe_by_strategy_bars.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  sharpe_by_strategy_bars.png")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("GENERATING TIER 2 PLOTS")
    print("=" * 60)

    output_dir = FIGURES_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n1. Equity Curves")
    plot_equity_curves_single(output_dir)
    # plot_equity_curves_comparison(output_dir)  # Removed: duplicate of equity_curves_top5

    print("\n2. Portfolio Weights")
    # plot_portfolio_weights(output_dir)  # Removed: not used in report
    plot_average_weights_by_regime(output_dir)

    print("\n3. Regime Analysis")
    plot_regime_timeline(output_dir)
    # plot_regime_distribution(output_dir)  # Removed: not used in report

    print("\n4. Summary Bar Charts")
    plot_sharpe_by_strategy_bars(output_dir)

    print(f"\nAll figures saved to: {output_dir}/")


if __name__ == "__main__":
    main()
