"""
Plot Generation
===============

Creates all figures for the research report.

Output: results/figures/

Author: Prince Alexander John
Date: February 2026
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from .results_loader import load_all_results, load_training_curves, load_baselines


FIGURES_DIR = Path("results/figures")


def generate_all_plots(output_dir: Path = FIGURES_DIR):
    """Generate all research figures."""
    output_dir.mkdir(parents=True, exist_ok=True)
    df = load_all_results()

    if df.empty:
        print("No results to generate plots from.")
        return

    if "replay_comparison" in df["experiment"].values:
        plot_e1_sharpe_boxplots(df, output_dir)
        plot_e1_learning_curves(output_dir)

    if any(e in df["experiment"].values for e in ["replay_comparison", "regime_performance"]):
        plot_e2_regime_heatmap(df, output_dir)

    if "robustness" in df["experiment"].values:
        plot_e3_robustness_curves(df, output_dir)

    if "sample_efficiency" in df["experiment"].values:
        plot_e6_sample_efficiency(output_dir)

    plot_baseline_comparison(df, output_dir)

    print(f"Figures saved to {output_dir}/")


def plot_e1_sharpe_boxplots(df: pd.DataFrame, output_dir: Path):
    """Box plots of Sharpe ratio: one subplot per algorithm, boxes per strategy."""
    e1 = df[df["experiment"] == "replay_comparison"].copy()
    if e1.empty or "sharpe_ratio" not in e1.columns:
        return

    algos = sorted(e1["algorithm"].unique())
    n = len(algos)
    if n == 0:
        return

    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), sharey=True, squeeze=False)

    for i, algo in enumerate(algos):
        algo_df = e1[e1["algorithm"] == algo]
        ax = axes[0][i]
        sns.boxplot(data=algo_df, x="strategy", y="sharpe_ratio", ax=ax)
        ax.set_title(algo.upper(), fontsize=12, fontweight="bold")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=45)

    axes[0][0].set_ylabel("Sharpe Ratio")
    fig.suptitle("E1: Replay Strategy Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "e1_sharpe_boxplots.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  e1_sharpe_boxplots.png")


def plot_e1_learning_curves(output_dir: Path):
    """Learning curves: eval reward vs timesteps, grouped by strategy per algorithm."""
    curves = load_training_curves(experiment="replay_comparison")
    if curves.empty or "eval_reward" not in curves.columns:
        return

    algos = sorted(curves["algorithm"].unique())
    n = len(algos)
    if n == 0:
        return

    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=True, squeeze=False)

    for i, algo in enumerate(algos):
        ax = axes[0][i]
        algo_df = curves[curves["algorithm"] == algo]

        for strat, group in algo_df.groupby("strategy"):
            # Average over seeds
            means = group.groupby("timestep")["eval_reward"].mean()
            stds = group.groupby("timestep")["eval_reward"].std().fillna(0)
            ax.plot(means.index, means.values, label=strat)
            ax.fill_between(means.index, means - stds, means + stds, alpha=0.15)

        ax.set_title(algo.upper(), fontsize=12, fontweight="bold")
        ax.set_xlabel("Timesteps")
        if i == 0:
            ax.set_ylabel("Eval Reward")
        ax.legend(fontsize=7)

    fig.suptitle("E1: Learning Curves", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "e1_learning_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  e1_learning_curves.png")


def plot_e2_regime_heatmap(df: pd.DataFrame, output_dir: Path):
    """Heatmap: rows = (algorithm, strategy), columns = regime, values = mean reward."""
    e2 = df[df["experiment"].isin(["replay_comparison", "regime_performance"])].copy()
    if e2.empty:
        return

    data = []
    for (algo, strat), group in e2.groupby(["algorithm", "strategy"]):
        row = {"label": f"{algo.upper()} / {strat}"}
        for regime in ["bull", "bear", "sideways"]:
            col = f"{regime}_mean_reward"
            if col in group.columns:
                row[regime] = group[col].mean()
        data.append(row)

    if not data:
        return

    heatmap_df = pd.DataFrame(data).set_index("label")
    heatmap_df = heatmap_df.dropna(how="all")
    if heatmap_df.empty:
        return

    fig, ax = plt.subplots(figsize=(8, max(4, len(heatmap_df) * 0.5)))
    sns.heatmap(heatmap_df, annot=True, fmt=".4f", cmap="RdYlGn", ax=ax,
                linewidths=0.5, linecolor="grey")
    ax.set_title("E2: Mean Reward by Regime", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "e2_regime_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  e2_regime_heatmap.png")


def plot_e3_robustness_curves(df: pd.DataFrame, output_dir: Path):
    """Line plot: Sharpe ratio vs noise_level for each strategy."""
    e3 = df[df["experiment"] == "robustness"].copy()
    if e3.empty or "sharpe_ratio" not in e3.columns:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    for strat, group in e3.groupby("strategy"):
        means = group.groupby("noise_level")["sharpe_ratio"].mean()
        stds = group.groupby("noise_level")["sharpe_ratio"].std().fillna(0)
        ax.errorbar(means.index, means.values, yerr=stds.values,
                    label=strat, marker="o", capsize=3)

    ax.set_xlabel("Noise Level")
    ax.set_ylabel("Sharpe Ratio")
    ax.set_title("E3: Robustness to Regime Detection Noise (SAC)", fontsize=14, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "e3_robustness.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  e3_robustness.png")


def plot_e6_sample_efficiency(output_dir: Path):
    """Performance at different training timestep checkpoints."""
    import json

    # Load checkpoint metrics
    results_dir = Path("results/experiments")
    manifest_path = results_dir / "manifest.json"
    if not manifest_path.exists():
        return

    with open(manifest_path) as f:
        manifest = json.load(f)

    rows = []
    for entry in manifest.get("runs", []):
        if entry["experiment"] != "sample_efficiency" or entry["status"] != "completed":
            continue
        cp_path = Path(entry["result_dir"]) / "checkpoint_metrics.json"
        if cp_path.exists():
            with open(cp_path) as f:
                checkpoints = json.load(f)
            for ts_str, metrics in checkpoints.items():
                rows.append({
                    "algorithm": entry["algorithm"],
                    "strategy": entry["strategy"],
                    "timestep": int(ts_str),
                    "eval_reward": metrics.get("eval_reward"),
                })

    if not rows:
        return

    cp_df = pd.DataFrame(rows)

    algos = sorted(cp_df["algorithm"].unique())
    n = len(algos)
    if n == 0:
        return

    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), sharey=True, squeeze=False)
    for i, algo in enumerate(algos):
        ax = axes[0][i]
        algo_df = cp_df[cp_df["algorithm"] == algo]
        for strat, group in algo_df.groupby("strategy"):
            means = group.groupby("timestep")["eval_reward"].mean()
            ax.plot(means.index, means.values, marker="o", label=strat)
        ax.set_title(algo.upper(), fontsize=12, fontweight="bold")
        ax.set_xlabel("Timesteps")
        if i == 0:
            ax.set_ylabel("Eval Reward")
        ax.legend(fontsize=8)

    fig.suptitle("E6: Sample Efficiency", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "e6_sample_efficiency.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  e6_sample_efficiency.png")


def plot_baseline_comparison(df: pd.DataFrame, output_dir: Path):
    """Bar chart comparing best RL agents vs baselines."""
    baselines = load_baselines()

    if df.empty and baselines.empty:
        return

    # Get best RL result per algorithm (using best strategy by mean Sharpe)
    rl_data = []
    if not df.empty and "sharpe_ratio" in df.columns:
        best = df.groupby("algorithm")["sharpe_ratio"].mean()
        for algo, sharpe in best.items():
            rl_data.append({"name": f"RL: {algo.upper()}", "sharpe_ratio": sharpe})

    baseline_data = []
    if not baselines.empty and "sharpe_ratio" in baselines.columns:
        for _, row in baselines.iterrows():
            name = row.get("strategy", "baseline")
            baseline_data.append({"name": name, "sharpe_ratio": row["sharpe_ratio"]})

    all_data = baseline_data + rl_data
    if not all_data:
        return

    plot_df = pd.DataFrame(all_data)

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#95a5a6"] * len(baseline_data) + ["#3498db"] * len(rl_data)
    bars = ax.bar(plot_df["name"], plot_df["sharpe_ratio"], color=colors)
    ax.set_ylabel("Sharpe Ratio")
    ax.set_title("RL Agents vs Baseline Strategies", fontsize=14, fontweight="bold")
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / "baseline_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  baseline_comparison.png")
