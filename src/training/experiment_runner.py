"""
Experiment Runner
=================

Orchestrates training experiments with checkpointing, resumability,
and optional parallelisation.

Manifest file: results/experiments/manifest.json
Each entry tracks: run_id, status, start_time, end_time, error, config, result_dir

Author: Prince Alexander John
Date: February 2026
"""

import json
import time
import traceback
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict
from dataclasses import dataclass, field, asdict
from concurrent.futures import ProcessPoolExecutor, as_completed

from .run_config import RunConfig, EXPERIMENT_GENERATORS


MANIFEST_PATH = Path("results/experiments/manifest.json")


# =============================================================================
# Manifest
# =============================================================================

@dataclass
class RunRecord:
    """One entry in the manifest."""
    run_id: str
    experiment: str
    algorithm: str
    strategy: str
    seed: int
    status: str = "pending"
    start_time: str = ""
    end_time: str = ""
    wall_time_seconds: float = 0.0
    error: str = ""
    config: dict = field(default_factory=dict)
    result_dir: str = ""


class Manifest:
    """JSON manifest tracking all run statuses for resumability."""

    def __init__(self, path: Path = MANIFEST_PATH):
        self.path = path
        self._records: Dict[str, RunRecord] = {}
        self._load()

    def _load(self):
        if self.path.exists():
            with open(self.path, "r") as f:
                data = json.load(f)
            for entry in data.get("runs", []):
                rec = RunRecord(**{k: v for k, v in entry.items() if k in RunRecord.__dataclass_fields__})
                self._records[rec.run_id] = rec

    def _save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "last_updated": datetime.now().isoformat(),
            "summary": self._summary(),
            "runs": [asdict(r) for r in self._records.values()],
        }
        with open(self.path, "w") as f:
            json.dump(data, f, indent=2)

    def _summary(self) -> dict:
        counts = {"pending": 0, "running": 0, "completed": 0, "failed": 0}
        for r in self._records.values():
            counts[r.status] = counts.get(r.status, 0) + 1
        counts["total"] = len(self._records)
        return counts

    def register_runs(self, configs: List[RunConfig]):
        for cfg in configs:
            if cfg.run_id not in self._records:
                self._records[cfg.run_id] = RunRecord(
                    run_id=cfg.run_id,
                    experiment=cfg.experiment,
                    algorithm=cfg.algorithm,
                    strategy=cfg.strategy,
                    seed=cfg.seed,
                    config=cfg.to_dict(),
                    result_dir=str(cfg.output_dir),
                )
        self._save()

    def is_completed(self, run_id: str) -> bool:
        rec = self._records.get(run_id)
        return rec is not None and rec.status == "completed"

    def mark_running(self, run_id: str):
        self._records[run_id].status = "running"
        self._records[run_id].start_time = datetime.now().isoformat()
        self._save()

    def mark_completed(self, run_id: str, wall_time: float):
        self._records[run_id].status = "completed"
        self._records[run_id].end_time = datetime.now().isoformat()
        self._records[run_id].wall_time_seconds = wall_time
        self._save()

    def mark_failed(self, run_id: str, error: str):
        self._records[run_id].status = "failed"
        self._records[run_id].end_time = datetime.now().isoformat()
        self._records[run_id].error = error[:500]
        self._save()

    def get_pending_ids(self, experiment: str = None) -> List[str]:
        result = []
        for run_id, rec in self._records.items():
            if rec.status in ("pending", "running"):
                if experiment is None or rec.experiment == experiment:
                    result.append(run_id)
        return result

    def print_progress(self):
        s = self._summary()
        completed = s["completed"]
        total = s["total"]
        failed = s["failed"]
        pct = (completed / total * 100) if total > 0 else 0

        # Progress bar
        bar_width = 30
        filled = int(bar_width * completed / total) if total > 0 else 0
        bar = "#" * filled + "-" * (bar_width - filled)

        print(f"\n{'=' * 70}")
        print(f" Experiment Progress")
        print(f"{'=' * 70}")
        print(f"  [{bar}] {completed}/{total} ({pct:.1f}%)")

        if failed > 0:
            print(f"  Failed: {failed} | Running: {s['running']} | Pending: {s['pending']}")
        else:
            print(f"  Running: {s['running']} | Pending: {s['pending']}")

        # Time estimates from completed runs
        completed_records = [r for r in self._records.values() if r.status == "completed"]
        if completed_records:
            times = [r.wall_time_seconds for r in completed_records]
            avg_time = sum(times) / len(times)
            total_elapsed = sum(times)
            remaining = total - completed - failed
            eta_seconds = avg_time * remaining

            print(f"  Total elapsed: {_format_time(total_elapsed)} | "
                  f"Avg per run: {_format_time(avg_time)}")
            if remaining > 0:
                print(f"  ETA: ~{_format_time(eta_seconds)} remaining ({remaining} runs left)")

        # Per-experiment breakdown
        print(f"\n  {'Experiment':<25} {'Done':>6} {'Failed':>8} {'Total':>7}")
        print(f"  {'-' * 48}")

        exp_counts: Dict[str, Dict[str, int]] = {}
        for r in self._records.values():
            if r.experiment not in exp_counts:
                exp_counts[r.experiment] = {"total": 0, "completed": 0, "failed": 0}
            exp_counts[r.experiment]["total"] += 1
            if r.status == "completed":
                exp_counts[r.experiment]["completed"] += 1
            elif r.status == "failed":
                exp_counts[r.experiment]["failed"] += 1

        for exp, counts in sorted(exp_counts.items()):
            print(f"  {exp:<25} {counts['completed']:>6} {counts['failed']:>8} {counts['total']:>7}")

        # Per-algorithm breakdown (if any completed)
        if completed_records:
            print(f"\n  {'Algorithm':<12} {'Completed':>10} {'Avg Time':>12} {'Avg Sharpe':>12}")
            print(f"  {'-' * 48}")
            algo_stats: Dict[str, list] = {}
            for r in completed_records:
                if r.algorithm not in algo_stats:
                    algo_stats[r.algorithm] = []
                algo_stats[r.algorithm].append(r.wall_time_seconds)

            for algo in sorted(algo_stats):
                times = algo_stats[algo]
                avg = sum(times) / len(times)
                print(f"  {algo.upper():<12} {len(times):>10} {_format_time(avg):>12}")

        print(f"{'=' * 70}")


def _format_time(seconds: float) -> str:
    """Format seconds into human-readable string."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        m, s = divmod(int(seconds), 60)
        return f"{m}m {s:02d}s"
    else:
        h, remainder = divmod(int(seconds), 3600)
        m, s = divmod(remainder, 60)
        return f"{h}h {m:02d}m"


def _progress_bar(completed: int, total: int, width: int = 30) -> str:
    """Render an ASCII progress bar."""
    filled = int(width * completed / total) if total > 0 else 0
    bar = "#" * filled + "-" * (width - filled)
    pct = completed / total * 100 if total > 0 else 0
    return f"[{bar}] {completed}/{total} ({pct:.1f}%)"


# =============================================================================
# Noisy Regime Detector (for E3 robustness experiment)
# =============================================================================

class NoisyRegimeDetector:
    """Wraps a regime detector and randomly flips labels with given probability."""

    def __init__(self, detector, noise_level: float, seed: int = 42):
        self.detector = detector
        self.noise_level = noise_level
        self.rng = np.random.RandomState(seed)

    def detect(self, price_data):
        regimes = self.detector.detect(price_data)
        if self.noise_level <= 0:
            return regimes

        from src.regime import MarketRegime
        all_regimes = list(MarketRegime)
        noisy = regimes.copy()
        for i in range(len(noisy)):
            if self.rng.random() < self.noise_level:
                current = noisy.iloc[i]
                others = [r for r in all_regimes if r != current]
                noisy.iloc[i] = self.rng.choice(others)
        return noisy

    def detect_current(self, price_data):
        regimes = self.detect(price_data)
        current = regimes.iloc[-1]
        return current.value if hasattr(current, "value") else str(current)

    def detect_at_index(self, price_data, index):
        regimes = self.detect(price_data)
        if 0 <= index < len(regimes):
            current = regimes.iloc[index]
            return current.value if hasattr(current, "value") else str(current)
        return "sideways"

    def get_statistics(self, price_data):
        return self.detector.get_statistics(price_data)

    def fit(self, price_data, **kwargs):
        if hasattr(self.detector, "fit"):
            self.detector.fit(price_data, **kwargs)
        return self


# =============================================================================
# Single Run Execution (top-level for ProcessPoolExecutor pickling)
# =============================================================================

def _json_serialisable(obj):
    """JSON serialiser for numpy types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    raise TypeError(f"Not JSON serialisable: {type(obj)}")


def execute_single_run(run_config_dict: dict) -> dict:
    """
    Execute a single training run. Top-level function for pickling.

    Returns dict with: run_id, status, wall_time, error
    """
    import sys
    warnings.filterwarnings("ignore")

    project_root = Path(__file__).parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from .run_config import RunConfig
    from .agent_factory import create_agent, create_training_config
    from src.data.downloader import download_ohlcv_data
    from src.data.features import calculate_all_features
    from src.environment.trading_env import TradingEnvironment
    from src.regime import UnifiedRegimeDetector

    run_config = RunConfig.from_dict(run_config_dict)
    run_id = run_config.run_id
    output_dir = run_config.output_dir

    start_time = time.time()

    try:
        TICKERS = ["SPY", "QQQ", "TLT", "GLD", "BIL"]

        # 1. Load all three data splits
        #    - train_price_data : 2010-2019  (agent learns here)
        #    - val_price_data   : 2020-2021  (generalisation check)
        #    - test_price_data  : 2022-2023  (held-out, primary reported metric)
        # Cache is keyed by ticker filename only, so each split must use its own
        # subdirectory to prevent the val/test requests hitting the training cache.
        print("Loading data splits...")
        train_price_data = download_ohlcv_data(
            TICKERS, "2010-01-01", "2019-12-31", cache_dir="data/raw/training", cache_expiry_days=365
        )
        val_price_data = download_ohlcv_data(
            TICKERS, "2020-01-01", "2021-12-31", cache_dir="data/raw/validation", cache_expiry_days=365
        )
        test_price_data = download_ohlcv_data(
            TICKERS, "2022-01-01", "2023-12-31", cache_dir="data/raw/testing", cache_expiry_days=365
        )

        # 2. Calculate features for each split separately
        train_features = calculate_all_features(train_price_data)
        val_features   = calculate_all_features(val_price_data)
        test_features  = calculate_all_features(test_price_data)

        # E4: reduced feature set applied to all splits equally
        if run_config.experiment == "feature_ablation" and run_config.feature_set == "reduced":
            train_features = _reduce_features(train_features)
            val_features   = _reduce_features(val_features)
            test_features  = _reduce_features(test_features)

        # 3. Create environments — one per split
        train_env = TradingEnvironment(
            price_data=train_price_data,
            feature_data=train_features,
            episode_length=252,
            random_start=True,
        )
        val_env = TradingEnvironment(
            price_data=val_price_data,
            feature_data=val_features,
            episode_length=252,
            random_start=False,
            start_idx=0,
        )
        test_env = TradingEnvironment(
            price_data=test_price_data,
            feature_data=test_features,
            episode_length=252,
            random_start=False,
            start_idx=0,
        )

        # 4. Regime detector — fit ONLY on training data to prevent look-ahead bias.
        # For HMM: learns transition matrix and emission params from training data.
        # For rule_based: anchors the 75th-percentile volatility threshold to the
        # training distribution so val/test detect() calls use the same threshold.
        detector = UnifiedRegimeDetector(method=run_config.detection_method)
        detector.fit(train_price_data, verbose=False)

        # E3: noise wrapper applies during training only; clean detector used for eval
        train_detector = detector
        if run_config.noise_level > 0:
            train_detector = NoisyRegimeDetector(detector, run_config.noise_level, run_config.seed)

        # 5. Apply algorithm-specific wrappers to val and test envs
        if run_config.algorithm == "rainbow":
            from src.agents.rainbow_agent import DiscreteActionWrapper
            val_env  = DiscreteActionWrapper(val_env)
            test_env = DiscreteActionWrapper(test_env)

        # PPO regime-aware: train env is wrapped internally by the agent;
        # val and test envs must match with the CLEAN detector + their own price data
        if run_config.algorithm == "ppo" and run_config.strategy != "standard":
            from src.agents.ppo_agent import RegimeObservationWrapper
            val_env  = RegimeObservationWrapper(val_env,  detector, price_data=val_price_data)
            test_env = RegimeObservationWrapper(test_env, detector, price_data=test_price_data)

        # 6. Create agent (uses train_detector which may have noise for E3)
        agent = create_agent(run_config, train_env, train_detector)

        # 7. Train on training data only
        training_config = create_training_config(run_config)
        metrics = agent.train(training_config=training_config)

        # 8. Evaluate on VALIDATION set (2020-2021) — generalisation check
        val_results = agent.evaluate(val_env, n_episodes=10)
        val_financial = _compute_financial_metrics(agent, val_env, n_episodes=5)
        val_results.update(val_financial)

        val_regime_results = {}
        try:
            val_regime_results = agent.evaluate_by_regime(val_env, n_episodes=10)
        except Exception:
            pass

        # 9. Evaluate on TEST set (2022-2023) — PRIMARY out-of-sample metric
        test_results = agent.evaluate(test_env, n_episodes=10)
        test_financial = _compute_financial_metrics(agent, test_env, n_episodes=5)
        test_results.update(test_financial)

        test_regime_results = {}
        try:
            test_regime_results = agent.evaluate_by_regime(test_env, n_episodes=10)
        except Exception:
            pass

        # 10. Save everything
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            agent.save(str(output_dir / "model"))
        except Exception:
            pass

        metrics.save(output_dir / "training_metrics.json")

        # val_results.json  → 2020-2021 validation (generalisation check)
        with open(output_dir / "val_results.json", "w") as f:
            json.dump(val_results, f, indent=2, default=_json_serialisable)

        with open(output_dir / "val_regime_results.json", "w") as f:
            json.dump(val_regime_results, f, indent=2, default=_json_serialisable)

        # test_results.json → 2022-2023 test (PRIMARY metric reported in paper)
        with open(output_dir / "test_results.json", "w") as f:
            json.dump(test_results, f, indent=2, default=_json_serialisable)

        with open(output_dir / "test_regime_results.json", "w") as f:
            json.dump(test_regime_results, f, indent=2, default=_json_serialisable)

        # Keep eval_results.json pointing to val for backwards compat with analysis scripts
        with open(output_dir / "eval_results.json", "w") as f:
            json.dump(val_results, f, indent=2, default=_json_serialisable)

        with open(output_dir / "regime_eval_results.json", "w") as f:
            json.dump(val_regime_results, f, indent=2, default=_json_serialisable)

        with open(output_dir / "run_config.json", "w") as f:
            json.dump(run_config.to_dict(), f, indent=2)

        # E6: checkpoint metrics
        if run_config.eval_checkpoints and hasattr(metrics, "timesteps"):
            _save_checkpoint_metrics(output_dir, metrics, run_config.eval_checkpoints)

        wall_time = time.time() - start_time
        return {
            "run_id": run_id,
            "status": "completed",
            "wall_time": wall_time,
            "error": "",
            # TEST (out-of-sample) is the primary metric
            "sharpe_ratio": test_results.get("sharpe_ratio"),
            "max_drawdown": test_results.get("max_drawdown"),
            "mean_reward": test_results.get("mean_reward"),
            # Validation metrics included for overfitting check
            "val_sharpe_ratio": val_results.get("sharpe_ratio"),
            "val_max_drawdown": val_results.get("max_drawdown"),
        }

    except Exception as e:
        wall_time = time.time() - start_time
        error_msg = f"{type(e).__name__}: {e}"

        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "error.log", "w", encoding="utf-8") as f:
            f.write(traceback.format_exc())

        return {"run_id": run_id, "status": "failed", "wall_time": wall_time, "error": error_msg}


def _compute_financial_metrics(agent, eval_env, n_episodes: int = 5) -> dict:
    """Run evaluation episodes and compute financial metrics (Sharpe, MDD, etc.).

    Uses info['step_return'] (net-of-transaction-cost portfolio return) rather than
    the shaped RL reward, so that Sharpe/Sortino/MDD reflect actual financial performance
    without distortion from the drawdown penalty term in the reward function.
    """
    from src.utils.metrics import calculate_all_metrics

    all_episode_returns = []

    for _ in range(n_episodes):
        obs, info = eval_env.reset()
        episode_returns = []
        done = False

        while not done:
            action, _ = agent.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            # Use actual net-of-cost portfolio return, not the shaped RL reward.
            # step_return = (portfolio_value - transaction_cost - prev_value) / prev_value
            step_return = info.get("step_return", reward)
            episode_returns.append(step_return)
            done = terminated or truncated

        all_episode_returns.append(episode_returns)

    # Compute metrics on the concatenated returns from all episodes
    flat_returns = np.concatenate(all_episode_returns)
    if len(flat_returns) < 2:
        return {}

    metrics = calculate_all_metrics(flat_returns)

    # Convert numpy types
    result = {}
    for k, v in metrics.items():
        if isinstance(v, (np.floating, np.integer)):
            result[k] = float(v)
        elif isinstance(v, float) and np.isfinite(v):
            result[k] = v
        elif isinstance(v, (int, float)):
            result[k] = v

    return result


def _reduce_features(feature_data: dict) -> dict:
    """Keep only basic features for E4 ablation."""
    reduced = {}
    for ticker, df in feature_data.items():
        basic_cols = [c for c in df.columns if any(
            tag in c.lower() for tag in ["return", "volume", "close", "open", "high", "low"]
        )]
        if basic_cols:
            reduced[ticker] = df[basic_cols]
        else:
            reduced[ticker] = df.iloc[:, :5]
    return reduced


def _save_checkpoint_metrics(output_dir: Path, metrics, checkpoints: List[int]):
    """Save metrics at each evaluation checkpoint for E6."""
    checkpoint_data = {}
    if not metrics.timesteps:
        return

    for cp in checkpoints:
        # Find closest timestep
        closest_idx = None
        for i, ts in enumerate(metrics.timesteps):
            if ts <= cp:
                closest_idx = i

        if closest_idx is not None:
            checkpoint_data[str(cp)] = {
                "timestep": metrics.timesteps[closest_idx],
                "eval_reward": metrics.eval_rewards[closest_idx] if closest_idx < len(metrics.eval_rewards) else None,
                "sharpe_ratio": metrics.sharpe_ratios[closest_idx] if closest_idx < len(metrics.sharpe_ratios) else None,
                "max_drawdown": metrics.max_drawdowns[closest_idx] if closest_idx < len(metrics.max_drawdowns) else None,
            }

    with open(output_dir / "checkpoint_metrics.json", "w") as f:
        json.dump(checkpoint_data, f, indent=2, default=_json_serialisable)


# =============================================================================
# Experiment Runner
# =============================================================================

class ExperimentRunner:
    """
    Orchestrates experiment execution with resumability.

    Usage:
        runner = ExperimentRunner(experiments=["replay_comparison"])
        runner.run(max_workers=4)
    """

    def __init__(
        self,
        experiments: Optional[List[str]] = None,
        manifest_path: Path = MANIFEST_PATH,
    ):
        self.manifest = Manifest(manifest_path)

        if experiments is None:
            experiments = list(EXPERIMENT_GENERATORS.keys())

        self.run_configs: Dict[str, RunConfig] = {}
        for exp in experiments:
            if exp not in EXPERIMENT_GENERATORS:
                raise ValueError(
                    f"Unknown experiment: {exp}. "
                    f"Valid: {list(EXPERIMENT_GENERATORS.keys())}"
                )
            for cfg in EXPERIMENT_GENERATORS[exp]():
                self.run_configs[cfg.run_id] = cfg

        self.manifest.register_runs(list(self.run_configs.values()))

    def run(self, max_workers: int = 4, retry_failed: bool = False):
        """Run all pending experiments."""
        pending_ids = []
        for run_id in self.run_configs:
            if self.manifest.is_completed(run_id):
                continue
            rec = self.manifest._records.get(run_id)
            if rec and rec.status == "failed" and not retry_failed:
                continue
            pending_ids.append(run_id)

        total = len(self.run_configs)
        skip = total - len(pending_ids)

        # Count by algorithm and experiment
        from collections import Counter
        algo_counts = Counter(c.algorithm for c in self.run_configs.values())
        exp_counts = Counter(c.experiment for c in self.run_configs.values())
        algo_summary = ", ".join(f"{a.upper()}: {n}" for a, n in sorted(algo_counts.items()))
        exp_summary = ", ".join(f"{e}: {n}" for e, n in sorted(exp_counts.items()))

        print(f"\n{'=' * 70}")
        print(f" Experiment Runner")
        print(f"{'=' * 70}")
        print(f"  Total runs:  {total} ({algo_summary})")
        print(f"  Experiments: {exp_summary}")
        print(f"  Completed:   {skip} | To run: {len(pending_ids)}")
        print(f"  Workers:     {max_workers}")
        print(f"{'=' * 70}\n")

        if not pending_ids:
            print("All runs already completed!")
            return

        if max_workers <= 1:
            self._run_sequential(pending_ids)
        else:
            self._run_parallel(pending_ids, max_workers)

        self.manifest.print_progress()

    def _run_sequential(self, run_ids: List[str]):
        total = len(run_ids)
        start_time = time.time()
        run_times = []
        completed_count = 0
        failed_count = 0

        for i, run_id in enumerate(run_ids, 1):
            cfg = self.run_configs[run_id]

            print(f"\n[{i}/{total}] {run_id}")
            print(f"  Algorithm: {cfg.algorithm.upper()} | Strategy: {cfg.strategy} | "
                  f"Seed: {cfg.seed} | {cfg.total_timesteps:,} timesteps")

            self.manifest.mark_running(run_id)
            result = execute_single_run(cfg.to_dict())

            if result["status"] == "completed":
                self.manifest.mark_completed(run_id, result["wall_time"])
                run_times.append(result["wall_time"])
                completed_count += 1

                # Show result metrics (test = primary out-of-sample, val = generalisation check)
                test_sharpe = result.get("sharpe_ratio")
                test_mdd    = result.get("max_drawdown")
                val_sharpe  = result.get("val_sharpe_ratio")
                test_str = f"Test Sharpe: {test_sharpe:.3f} | MDD: {test_mdd:.1%}" if test_sharpe is not None else ""
                val_str  = f"Val Sharpe: {val_sharpe:.3f}" if val_sharpe is not None else ""
                metrics_str = " | ".join(s for s in [test_str, val_str] if s)
                print(f"  Completed in {_format_time(result['wall_time'])} | {metrics_str}")
            else:
                self.manifest.mark_failed(run_id, result["error"])
                failed_count += 1
                print(f"  FAILED in {_format_time(result['wall_time'])}: {result['error'][:120]}")

            # Progress summary
            done = completed_count + failed_count
            print(f"\n  {_progress_bar(done, total)}")
            elapsed = time.time() - start_time
            print(f"  Elapsed: {_format_time(elapsed)}", end="")

            if run_times:
                avg = sum(run_times) / len(run_times)
                remaining = total - done
                eta = avg * remaining
                print(f" | Avg: {_format_time(avg)}/run", end="")
                if remaining > 0:
                    print(f" | ETA: ~{_format_time(eta)}", end="")
            if failed_count > 0:
                print(f" | {failed_count} failed", end="")
            print()

    def _run_parallel(self, run_ids: List[str], max_workers: int):
        total = len(run_ids)
        start_time = time.time()
        run_times = []
        completed_count = 0
        failed_count = 0

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for run_id in run_ids:
                cfg = self.run_configs[run_id]
                self.manifest.mark_running(run_id)
                future = executor.submit(execute_single_run, cfg.to_dict())
                futures[future] = run_id

            for future in as_completed(futures):
                run_id = futures[future]
                cfg = self.run_configs[run_id]

                try:
                    result = future.result()
                    if result["status"] == "completed":
                        self.manifest.mark_completed(run_id, result["wall_time"])
                        run_times.append(result["wall_time"])
                        completed_count += 1

                        test_sharpe = result.get("sharpe_ratio")
                        test_mdd    = result.get("max_drawdown")
                        val_sharpe  = result.get("val_sharpe_ratio")
                        metrics_str = ""
                        if test_sharpe is not None:
                            metrics_str = f"Test: {test_sharpe:.3f}"
                            if val_sharpe is not None:
                                metrics_str += f" | Val: {val_sharpe:.3f}"
                            if test_mdd is not None:
                                metrics_str += f" | MDD: {test_mdd:.1%}"
                        done = completed_count + failed_count
                        print(f"  [{done}/{total}] {cfg.algorithm.upper()}/{cfg.strategy} "
                              f"s{cfg.seed} -- {_format_time(result['wall_time'])} | {metrics_str}")
                    else:
                        self.manifest.mark_failed(run_id, result["error"])
                        failed_count += 1
                        done = completed_count + failed_count
                        print(f"  [{done}/{total}] FAILED: {run_id}: {result['error'][:80]}")
                except Exception as e:
                    self.manifest.mark_failed(run_id, str(e))
                    failed_count += 1
                    done = completed_count + failed_count
                    print(f"  [{done}/{total}] CRASHED: {run_id}: {e}")

                # Periodic progress (every 5 completions)
                done = completed_count + failed_count
                if done % 5 == 0 or done == total:
                    elapsed = time.time() - start_time
                    avg = sum(run_times) / len(run_times) if run_times else 0
                    remaining = total - done
                    eta = (avg * remaining / max_workers) if avg > 0 else 0
                    print(f"\n  {_progress_bar(done, total)}")
                    print(f"  Elapsed: {_format_time(elapsed)} | "
                          f"Avg: {_format_time(avg)}/run | "
                          f"ETA: ~{_format_time(eta)}\n")
