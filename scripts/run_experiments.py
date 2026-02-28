"""
Run Experiments CLI
===================

Usage:
    python scripts/run_experiments.py                                    # Run all experiments
    python scripts/run_experiments.py --experiments replay_comparison    # Run E1 only
    python scripts/run_experiments.py --workers 1                        # Sequential mode
    python scripts/run_experiments.py --dry-run                          # List runs without executing
    python scripts/run_experiments.py --status                           # Show progress
    python scripts/run_experiments.py --retry-failed                     # Retry failed runs

Author: Prince Alexander John
Date: February 2026
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


VALID_EXPERIMENTS = [
    "replay_comparison",
    "regime_performance",
    "robustness",
    "feature_ablation",
    "regime_detection",
    "sample_efficiency",
]


def main():
    parser = argparse.ArgumentParser(description="Run training experiments")

    parser.add_argument(
        "--experiments", nargs="+", default=None,
        choices=VALID_EXPERIMENTS,
        help="Which experiments to run (default: all)",
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Max concurrent workers (default: 1 = sequential)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show runs without executing",
    )
    parser.add_argument(
        "--retry-failed", action="store_true",
        help="Retry previously failed runs",
    )
    parser.add_argument(
        "--status", action="store_true",
        help="Show current progress and exit",
    )
    parser.add_argument(
        "--baselines", action="store_true",
        help="Run baseline strategies only",
    )
    parser.add_argument(
        "--timesteps", type=int, default=None,
        help="Override total_timesteps for all runs (useful for smoke tests)",
    )

    args = parser.parse_args()

    if args.status:
        from src.training.experiment_runner import Manifest, MANIFEST_PATH
        manifest = Manifest(MANIFEST_PATH)
        manifest.print_progress()
        return

    if args.baselines:
        from src.training.baselines import run_all_baselines
        from src.data.downloader import download_ohlcv_data

        print("Running baseline strategies...")
        price_data = download_ohlcv_data(
            tickers=["SPY", "QQQ", "TLT", "GLD", "BIL"],
            start_date="2010-01-01",
            end_date="2019-12-31",
        )
        results = run_all_baselines(price_data)
        print("\nBaseline Results:")
        for name, metrics in results.items():
            sr = metrics.get("sharpe_ratio", "N/A")
            mdd = metrics.get("max_drawdown", "N/A")
            cr = metrics.get("cumulative_return", "N/A")
            print(f"  {name}: Sharpe={sr:.3f}, MDD={mdd:.3f}, Return={cr:.3f}")
        return

    from src.training.experiment_runner import ExperimentRunner

    runner = ExperimentRunner(experiments=args.experiments)

    # Apply timestep override (for smoke tests / quick runs)
    if args.timesteps is not None:
        for cfg in runner.run_configs.values():
            cfg.total_timesteps = args.timesteps
        print(f"[Override] total_timesteps set to {args.timesteps:,} for all runs")

    if args.dry_run:
        print(f"\nDry run — would execute {len(runner.run_configs)} runs:\n")
        for run_id in sorted(runner.run_configs.keys()):
            cfg = runner.run_configs[run_id]
            print(f"  {run_id}  ({cfg.total_timesteps:,} timesteps)")

        # Summary by experiment
        from collections import Counter
        exp_counts = Counter(c.experiment for c in runner.run_configs.values())
        print(f"\nSummary:")
        for exp, count in sorted(exp_counts.items()):
            print(f"  {exp}: {count} runs")
        print(f"  Total: {sum(exp_counts.values())} runs")
        return

    runner.run(max_workers=args.workers, retry_failed=args.retry_failed)


if __name__ == "__main__":
    main()
