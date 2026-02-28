"""
Analysis Module
===============

Loads experiment results and generates tables, plots, and statistical tests.

Usage:
    from src.analysis.results_loader import load_all_results
    from src.analysis.comparison_tables import generate_all_tables
    from src.analysis.plots import generate_all_plots
    from src.analysis.statistical_tests import run_all_tests
"""

from .results_loader import load_all_results, load_baselines
from .comparison_tables import generate_all_tables
from .plots import generate_all_plots
from .statistical_tests import run_all_tests

__all__ = [
    "load_all_results",
    "load_baselines",
    "generate_all_tables",
    "generate_all_plots",
    "run_all_tests",
]
