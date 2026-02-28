"""
Comparison Tables
=================

Generates summary CSV tables for the research report.

Output: results/tables/

Author: Prince Alexander John
Date: February 2026
"""

import pandas as pd
from pathlib import Path

from .results_loader import load_all_results


TABLES_DIR = Path("results/tables")


def generate_all_tables(output_dir: Path = TABLES_DIR):
    """Generate all comparison tables."""
    output_dir.mkdir(parents=True, exist_ok=True)
    df = load_all_results()

    if df.empty:
        print("No results to generate tables from.")
        return

    if "replay_comparison" in df["experiment"].values:
        generate_e1_table(df, output_dir)
    if any(e in df["experiment"].values for e in ["replay_comparison", "regime_performance"]):
        generate_e2_table(df, output_dir)
    if "robustness" in df["experiment"].values:
        generate_e3_table(df, output_dir)
    if "feature_ablation" in df["experiment"].values:
        generate_e4_table(df, output_dir)
    if "regime_detection" in df["experiment"].values:
        generate_e5_table(df, output_dir)
    if "sample_efficiency" in df["experiment"].values:
        generate_e6_table(df, output_dir)

    print(f"Tables saved to {output_dir}/")


def generate_e1_table(df: pd.DataFrame, output_dir: Path):
    """
    E1: Main comparison — Algorithm x Strategy Sharpe ratio.

    Rows: algorithms, Columns: strategies
    Values: mean +/- std across seeds
    """
    e1 = df[df["experiment"] == "replay_comparison"].copy()
    if e1.empty or "sharpe_ratio" not in e1.columns:
        return

    pivot = e1.groupby(["algorithm", "strategy"])["sharpe_ratio"].agg(["mean", "std"]).reset_index()
    pivot["display"] = pivot.apply(
        lambda r: f"{r['mean']:.3f} +/- {r['std']:.3f}" if pd.notna(r["mean"]) else "N/A",
        axis=1,
    )
    table = pivot.pivot(index="algorithm", columns="strategy", values="display")
    table.to_csv(output_dir / "e1_replay_comparison.csv")

    # Also save raw numeric version
    raw = e1.groupby(["algorithm", "strategy"]).agg(
        sharpe_mean=("sharpe_ratio", "mean"),
        sharpe_std=("sharpe_ratio", "std"),
        reward_mean=("mean_reward", "mean"),
        mdd_mean=("max_drawdown", "mean"),
        n_seeds=("seed", "count"),
    ).reset_index()
    raw.to_csv(output_dir / "e1_replay_comparison_raw.csv", index=False)

    print(f"  E1 table: {output_dir / 'e1_replay_comparison.csv'}")


def generate_e2_table(df: pd.DataFrame, output_dir: Path):
    """E2: Per-regime reward breakdown."""
    e2 = df[df["experiment"].isin(["replay_comparison", "regime_performance"])].copy()
    if e2.empty:
        return

    rows = []
    for (algo, strat), group in e2.groupby(["algorithm", "strategy"]):
        row = {"algorithm": algo, "strategy": strat}
        for regime in ["bull", "bear", "sideways"]:
            col = f"{regime}_mean_reward"
            if col in group.columns:
                row[f"{regime}_mean"] = group[col].mean()
                row[f"{regime}_std"] = group[col].std()
        rows.append(row)

    table = pd.DataFrame(rows)
    table.to_csv(output_dir / "e2_regime_performance.csv", index=False)
    print(f"  E2 table: {output_dir / 'e2_regime_performance.csv'}")


def generate_e3_table(df: pd.DataFrame, output_dir: Path):
    """E3: Robustness — Sharpe vs noise level."""
    e3 = df[df["experiment"] == "robustness"].copy()
    if e3.empty or "sharpe_ratio" not in e3.columns:
        return

    table = e3.groupby(["noise_level", "strategy"]).agg(
        sharpe_mean=("sharpe_ratio", "mean"),
        sharpe_std=("sharpe_ratio", "std"),
        n_seeds=("seed", "count"),
    ).reset_index()
    table.to_csv(output_dir / "e3_robustness.csv", index=False)
    print(f"  E3 table: {output_dir / 'e3_robustness.csv'}")


def generate_e4_table(df: pd.DataFrame, output_dir: Path):
    """E4: Feature ablation — full vs reduced features."""
    e4 = df[df["experiment"] == "feature_ablation"].copy()
    if e4.empty or "sharpe_ratio" not in e4.columns:
        return

    table = e4.groupby(["feature_set", "algorithm"]).agg(
        sharpe_mean=("sharpe_ratio", "mean"),
        sharpe_std=("sharpe_ratio", "std"),
        reward_mean=("mean_reward", "mean"),
        n_seeds=("seed", "count"),
    ).reset_index()
    table.to_csv(output_dir / "e4_feature_ablation.csv", index=False)
    print(f"  E4 table: {output_dir / 'e4_feature_ablation.csv'}")


def generate_e5_table(df: pd.DataFrame, output_dir: Path):
    """E5: Detection method comparison."""
    e5 = df[df["experiment"] == "regime_detection"].copy()
    if e5.empty or "sharpe_ratio" not in e5.columns:
        return

    table = e5.groupby(["detection_method", "strategy"]).agg(
        sharpe_mean=("sharpe_ratio", "mean"),
        sharpe_std=("sharpe_ratio", "std"),
        n_seeds=("seed", "count"),
    ).reset_index()
    table.to_csv(output_dir / "e5_detection_comparison.csv", index=False)
    print(f"  E5 table: {output_dir / 'e5_detection_comparison.csv'}")


def generate_e6_table(df: pd.DataFrame, output_dir: Path):
    """E6: Sample efficiency — performance at checkpoints."""
    import json

    e6 = df[df["experiment"] == "sample_efficiency"].copy()
    if e6.empty:
        return

    # Load checkpoint metrics from files
    rows = []
    for _, run in e6.iterrows():
        cp_path = Path(run.get("result_dir", run["run_id"])) / "checkpoint_metrics.json"
        if not cp_path.exists():
            # Try constructing path from run_id parts
            result_dir = Path("results/experiments") / run["experiment"] / run["algorithm"] / run["strategy"] / run["run_id"]
            cp_path = result_dir / "checkpoint_metrics.json"

        if cp_path.exists():
            with open(cp_path) as f:
                checkpoints = json.load(f)
            for ts_str, metrics in checkpoints.items():
                rows.append({
                    "algorithm": run["algorithm"],
                    "strategy": run["strategy"],
                    "seed": run["seed"],
                    "checkpoint_timestep": int(ts_str),
                    "eval_reward": metrics.get("eval_reward"),
                    "sharpe_ratio": metrics.get("sharpe_ratio"),
                })

    if rows:
        table = pd.DataFrame(rows)
        summary = table.groupby(["algorithm", "strategy", "checkpoint_timestep"]).agg(
            reward_mean=("eval_reward", "mean"),
            reward_std=("eval_reward", "std"),
            sharpe_mean=("sharpe_ratio", "mean"),
        ).reset_index()
        summary.to_csv(output_dir / "e6_sample_efficiency.csv", index=False)
        print(f"  E6 table: {output_dir / 'e6_sample_efficiency.csv'}")
