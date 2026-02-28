"""
Results Loader
==============

Loads experiment results from output directories into pandas DataFrames.

The key output is a single DataFrame with one row per completed run.

Author: Prince Alexander John
Date: February 2026
"""

import json
import pandas as pd
from pathlib import Path
from typing import Optional

RESULTS_DIR = Path("results/experiments")


def load_all_results(results_dir: Path = RESULTS_DIR) -> pd.DataFrame:
    """
    Load all experiment results into a single DataFrame.

    Reads the manifest and per-run JSON files for completed runs.

    Returns
    -------
    pd.DataFrame
        One row per completed run with columns:
        run_id, experiment, algorithm, strategy, seed, mean_reward,
        sharpe_ratio, max_drawdown, cumulative_return, sortino_ratio,
        bull_mean_reward, bear_mean_reward, sideways_mean_reward,
        wall_time_seconds, noise_level, feature_set, detection_method
    """
    manifest_path = results_dir / "manifest.json"
    if not manifest_path.exists():
        print(f"No manifest found at {manifest_path}")
        return pd.DataFrame()

    with open(manifest_path) as f:
        manifest = json.load(f)

    rows = []
    for entry in manifest.get("runs", []):
        if entry["status"] != "completed":
            continue

        result_dir = Path(entry["result_dir"])
        row = _load_single_run(result_dir, entry)
        if row is not None:
            rows.append(row)

    if not rows:
        print("No completed runs found.")
        return pd.DataFrame()

    return pd.DataFrame(rows)


def _load_single_run(result_dir: Path, manifest_entry: dict) -> Optional[dict]:
    """Load results from a single completed run."""
    row = {
        "run_id": manifest_entry["run_id"],
        "experiment": manifest_entry["experiment"],
        "algorithm": manifest_entry["algorithm"],
        "strategy": manifest_entry["strategy"],
        "seed": manifest_entry["seed"],
        "wall_time_seconds": manifest_entry.get("wall_time_seconds", 0),
    }

    # Load eval results
    eval_path = result_dir / "eval_results.json"
    if eval_path.exists():
        with open(eval_path) as f:
            eval_data = json.load(f)
        row["mean_reward"] = eval_data.get("mean_reward")
        row["std_reward"] = eval_data.get("std_reward")
        row["sharpe_ratio"] = eval_data.get("sharpe_ratio")
        row["max_drawdown"] = eval_data.get("max_drawdown")
        row["cumulative_return"] = eval_data.get("cumulative_return")
        row["sortino_ratio"] = eval_data.get("sortino_ratio")

    # Load regime-specific results
    regime_path = result_dir / "regime_eval_results.json"
    if regime_path.exists():
        with open(regime_path) as f:
            regime_data = json.load(f)
        for regime in ["bull", "bear", "sideways"]:
            if regime in regime_data and regime_data[regime]:
                row[f"{regime}_mean_reward"] = regime_data[regime].get("mean_reward")
                row[f"{regime}_std_reward"] = regime_data[regime].get("std_reward")

    # Load run config for experiment-specific params
    config_path = result_dir / "run_config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        row["noise_level"] = config.get("noise_level", 0.0)
        row["feature_set"] = config.get("feature_set", "full")
        row["detection_method"] = config.get("detection_method", "rule_based")

    return row


def load_training_curves(
    results_dir: Path = RESULTS_DIR,
    experiment: str = None,
    algorithm: str = None,
    strategy: str = None,
) -> pd.DataFrame:
    """
    Load training metric time series for learning curve plots.

    Returns DataFrame with columns:
        run_id, timestep, eval_reward, sharpe_ratio, max_drawdown
    """
    manifest_path = results_dir / "manifest.json"
    if not manifest_path.exists():
        return pd.DataFrame()

    with open(manifest_path) as f:
        manifest = json.load(f)

    rows = []
    for entry in manifest.get("runs", []):
        if entry["status"] != "completed":
            continue
        if experiment and entry["experiment"] != experiment:
            continue
        if algorithm and entry["algorithm"] != algorithm:
            continue
        if strategy and entry["strategy"] != strategy:
            continue

        metrics_path = Path(entry["result_dir"]) / "training_metrics.json"
        if not metrics_path.exists():
            continue

        with open(metrics_path) as f:
            metrics = json.load(f)

        timesteps = metrics.get("timesteps", [])
        eval_rewards = metrics.get("eval_rewards", [])
        sharpe_ratios = metrics.get("sharpe_ratios", [])
        max_drawdowns = metrics.get("max_drawdowns", [])

        for i, ts in enumerate(timesteps):
            row = {
                "run_id": entry["run_id"],
                "experiment": entry["experiment"],
                "algorithm": entry["algorithm"],
                "strategy": entry["strategy"],
                "seed": entry["seed"],
                "timestep": ts,
                "eval_reward": eval_rewards[i] if i < len(eval_rewards) else None,
                "sharpe_ratio": sharpe_ratios[i] if i < len(sharpe_ratios) else None,
                "max_drawdown": max_drawdowns[i] if i < len(max_drawdowns) else None,
            }
            rows.append(row)

    return pd.DataFrame(rows) if rows else pd.DataFrame()


def load_baselines(results_dir: Path = RESULTS_DIR) -> pd.DataFrame:
    """Load baseline strategy results."""
    baseline_path = results_dir / "baselines.json"
    if baseline_path.exists():
        with open(baseline_path) as f:
            data = json.load(f)
        return pd.DataFrame(data.values())
    return pd.DataFrame()
