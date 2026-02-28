"""
Analyze Results CLI
===================

Usage:
    python scripts/analyze_results.py                # All tables + plots + stats
    python scripts/analyze_results.py --summary      # Print quick summary
    python scripts/analyze_results.py --tables-only   # CSV tables only
    python scripts/analyze_results.py --plots-only    # Figures only
    python scripts/analyze_results.py --stats         # Statistical tests only

Author: Prince Alexander John
Date: February 2026
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def main():
    parser = argparse.ArgumentParser(description="Analyze experiment results")

    parser.add_argument("--tables-only", action="store_true")
    parser.add_argument("--plots-only", action="store_true")
    parser.add_argument("--stats", action="store_true")
    parser.add_argument("--summary", action="store_true")

    args = parser.parse_args()

    from src.analysis.results_loader import load_all_results
    from src.analysis.comparison_tables import generate_all_tables
    from src.analysis.plots import generate_all_plots
    from src.analysis.statistical_tests import run_all_tests

    if args.summary:
        df = load_all_results()
        if df.empty:
            print("No completed results found.")
            return

        print(f"\nLoaded {len(df)} completed runs\n")
        print(f"Experiments: {df['experiment'].value_counts().to_dict()}")
        print(f"Algorithms:  {df['algorithm'].value_counts().to_dict()}")

        if "sharpe_ratio" in df.columns:
            print(f"\nMean Sharpe by strategy:")
            print(df.groupby("strategy")["sharpe_ratio"].agg(["mean", "std", "count"]).round(4).to_string())

            print(f"\nMean Sharpe by algorithm:")
            print(df.groupby("algorithm")["sharpe_ratio"].agg(["mean", "std", "count"]).round(4).to_string())

        if "mean_reward" in df.columns:
            print(f"\nMean reward by strategy:")
            print(df.groupby("strategy")["mean_reward"].agg(["mean", "std"]).round(4).to_string())

        return

    do_all = not args.tables_only and not args.plots_only and not args.stats

    if do_all or args.tables_only:
        print("Generating comparison tables...")
        generate_all_tables()

    if do_all or args.plots_only:
        print("Generating plots...")
        generate_all_plots()

    if do_all or args.stats:
        print("Running statistical tests...")
        results = run_all_tests()
        if not results.empty:
            sig = results[results["significant_bonferroni"]]
            print(f"  {len(sig)}/{len(results)} significant results (Bonferroni corrected)")


if __name__ == "__main__":
    main()
