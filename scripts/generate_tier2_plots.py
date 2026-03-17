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

def plot_equity_curves_single(output_dir: Path):
    """Plot equity curves for best configuration (TD3 adaptive)."""
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 6))

    # Load TD3 adaptive seed 123 (best single run: +1.006 Sharpe)
    data = load_episode_data("td3", "adaptive", 123, "test")
    if not data:
        print("  No data for TD3 adaptive seed 123")
        return

    # Plot all 5 episodes
    portfolio_values = np.array(data["portfolio_values"])
    dates = data.get("dates", list(range(len(portfolio_values[0]))))

    # Align lengths
    min_len = min(len(dates), len(portfolio_values[0]))
    dates = dates[:min_len]

    for i, pv_full in enumerate(portfolio_values):
        pv = pv_full[:min_len]  # Truncate to match dates
        alpha = 0.3 if i > 0 else 1.0
        lw = 1 if i > 0 else 2
        label = "TD3 Adaptive" if i == 0 else None
        ax.plot(dates, pv, color="#2ecc71", alpha=alpha, lw=lw, label=label)

    # Add baseline (SPY)
    baselines = load_baselines()
    if "spy_buy_hold" in baselines:
        spy_values = baselines["spy_buy_hold"].get("daily_values", [])
        if spy_values:
            # Align lengths - use minimum of both
            plot_len = min(len(spy_values), len(dates))
            ax.plot(dates[:plot_len], spy_values[:plot_len], color="#e74c3c", lw=2, label="SPY Buy-and-Hold")

    ax.axhline(y=1_000_000, color="gray", linestyle=":", alpha=0.5, label="Initial Capital")
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Portfolio Value ($)", fontsize=12)
    ax.set_title("TD3 Adaptive vs SPY Buy-and-Hold (Test Period 2022-2023)", fontsize=14, fontweight="bold")
    ax.legend(loc="upper left")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x/1e6:.2f}M"))

    plt.tight_layout()
    plt.savefig(output_dir / "equity_curve_td3_adaptive.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  equity_curve_td3_adaptive.png")


def plot_equity_curves_comparison(output_dir: Path):
    """Plot equity curves comparing all algorithms (best strategy each)."""
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 6))

    configs = [
        ("td3", "adaptive", "#2ecc71", "TD3 Adaptive"),
        ("sac", "standard", "#3498db", "SAC Standard"),
        ("ppo", "current_focused", "#9b59b6", "PPO Current-Focused"),
        ("rainbow", "inverse_frequency", "#f39c12", "Rainbow Inverse-Freq"),
    ]

    dates = None
    for algo, strategy, color, label in configs:
        # Use seed 42 for consistency
        data = load_episode_data(algo, strategy, 42, "test")
        if not data:
            continue

        pv = np.array(data["portfolio_values"])
        mean_pv = pv.mean(axis=0)

        if dates is None:
            dates = data.get("dates", list(range(len(mean_pv))))

        # Align lengths
        plot_len = min(len(dates), len(mean_pv))
        ax.plot(dates[:plot_len], mean_pv[:plot_len], color=color, lw=2, label=label)

    # Add SPY baseline
    baselines = load_baselines()
    if "spy_buy_hold" in baselines and dates:
        spy_values = baselines["spy_buy_hold"].get("daily_values", [])
        if spy_values and len(spy_values) >= len(dates):
            ax.plot(dates, spy_values[:len(dates)], color="#e74c3c", lw=2,
                    label="SPY Buy-and-Hold")

    ax.axhline(y=1_000_000, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Portfolio Value ($)", fontsize=12)
    ax.set_title("Algorithm Comparison: Equity Curves (Test Period)", fontsize=14, fontweight="bold")
    ax.legend(loc="upper left")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x/1e6:.2f}M"))

    plt.tight_layout()
    plt.savefig(output_dir / "equity_curves_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  equity_curves_comparison.png")


# =============================================================================
# 2. PORTFOLIO WEIGHTS
# =============================================================================

def plot_portfolio_weights(output_dir: Path):
    """Plot portfolio weight allocation over time for TD3 adaptive."""
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_episode_data("td3", "adaptive", 42, "test")
    if not data or "portfolio_weights" not in data:
        print("  No weight data available")
        return

    weights = np.array(data["portfolio_weights"])  # Shape: (5 episodes, T, 5 assets)
    dates = data.get("dates", list(range(weights.shape[1])))

    # Use first episode
    w = weights[0]  # Shape: (T, 5)

    # Align lengths
    min_len = min(len(dates), len(w))
    dates = dates[:min_len]
    w = w[:min_len]

    fig, ax = plt.subplots(figsize=(14, 6))

    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f1c40f", "#95a5a6"]

    ax.stackplot(dates, w.T, labels=ASSETS, colors=colors, alpha=0.8)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Portfolio Weight", fontsize=12)
    ax.set_title("TD3 Adaptive: Portfolio Allocation Over Time", fontsize=14, fontweight="bold")
    ax.legend(loc="upper left", ncol=5)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(output_dir / "portfolio_weights_td3_adaptive.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  portfolio_weights_td3_adaptive.png")


def plot_average_weights_by_regime(output_dir: Path):
    """Bar chart: average portfolio weights by regime."""
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_episode_data("td3", "adaptive", 42, "test")
    if not data or "portfolio_weights" not in data or "regime_labels" not in data:
        print("  No regime weight data available")
        return

    weights = np.array(data["portfolio_weights"])[0]  # First episode
    regimes = data["regime_labels"][0]  # First episode

    # Handle both string and int regime labels
    int_to_name = {0: "Bull", 1: "Bear", 2: "Sideways"}
    str_to_name = {"bull": "Bull", "bear": "Bear", "sideways": "Sideways"}

    results = {"Bull": [], "Bear": [], "Sideways": [], "Unknown": []}
    for w, r in zip(weights, regimes):
        if isinstance(r, str):
            regime_name = str_to_name.get(r.lower(), "Unknown")
        else:
            regime_name = int_to_name.get(r, "Unknown")
        results[regime_name].append(w)

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(ASSETS))
    width = 0.25
    colors = {"Bull": "#2ecc71", "Bear": "#e74c3c", "Sideways": "#f39c12", "Unknown": "#95a5a6"}

    i = 0
    for regime in ["Bull", "Bear", "Sideways"]:
        ws = results.get(regime, [])
        if ws:
            mean_w = np.mean(ws, axis=0)
            ax.bar(x + i * width, mean_w, width, label=regime, color=colors[regime])
            i += 1

    ax.set_xlabel("Asset", fontsize=12)
    ax.set_ylabel("Average Weight", fontsize=12)
    ax.set_title("TD3 Adaptive: Average Portfolio Weights by Regime", fontsize=14, fontweight="bold")
    ax.set_xticks(x + width)
    ax.set_xticklabels(ASSETS)
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / "weights_by_regime.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  weights_by_regime.png")


# =============================================================================
# 3. REGIME ANALYSIS
# =============================================================================

def plot_regime_timeline(output_dir: Path):
    """Plot regime labels over time with portfolio value overlay."""
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_episode_data("td3", "adaptive", 42, "test")
    if not data or "regime_labels" not in data:
        print("  No regime data available")
        return

    pv = np.array(data["portfolio_values"])[0]
    regimes = data["regime_labels"][0]
    dates = data.get("dates", list(range(len(pv))))

    # Align lengths
    min_len = min(len(dates), len(pv), len(regimes))
    dates = dates[:min_len]
    pv = pv[:min_len]
    regimes = regimes[:min_len]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                                    gridspec_kw={'height_ratios': [3, 1]})

    # Top: Portfolio value
    ax1.plot(dates, pv, color="#2ecc71", lw=2, label="TD3 Adaptive")

    # Add SPY buy-and-hold
    baselines = load_baselines()
    if "spy_buy_hold" in baselines:
        spy_values = baselines["spy_buy_hold"].get("daily_values", [])
        if spy_values:
            plot_len = min(len(spy_values), len(dates))
            ax1.plot(dates[:plot_len], spy_values[:plot_len], color="#e74c3c", lw=2,
                    label="SPY Buy-and-Hold")

    ax1.set_ylabel("Portfolio Value ($)", fontsize=12)
    ax1.set_title("TD3 Adaptive vs SPY: Portfolio Value with Regime Labels", fontsize=14, fontweight="bold")
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x/1e6:.2f}M"))
    ax1.legend(loc="upper left")

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

    data = load_episode_data("td3", "adaptive", 42, "test")
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
    from analyze_e1_deep import E1_RESULTS
    df = pd.DataFrame(E1_RESULTS)

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
    plot_equity_curves_comparison(output_dir)

    print("\n2. Portfolio Weights")
    plot_portfolio_weights(output_dir)
    plot_average_weights_by_regime(output_dir)

    print("\n3. Regime Analysis")
    plot_regime_timeline(output_dir)
    plot_regime_distribution(output_dir)

    print("\n4. Summary Bar Charts")
    plot_sharpe_by_strategy_bars(output_dir)

    print(f"\nAll figures saved to: {output_dir}/")


if __name__ == "__main__":
    main()
