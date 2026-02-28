"""
Training Orchestration Module
==============================

Provides experiment running, agent creation, and baseline strategies.

Usage:
    from src.training.experiment_runner import ExperimentRunner
    from src.training.run_config import RunConfig, generate_all_runs
    from src.training.agent_factory import create_agent, create_training_config
    from src.training.baselines import run_all_baselines
"""

from .run_config import RunConfig, generate_all_runs, generate_runs, EXPERIMENT_GENERATORS
from .agent_factory import create_agent, create_training_config
from .experiment_runner import ExperimentRunner, Manifest, MANIFEST_PATH
from .baselines import run_all_baselines

__all__ = [
    "RunConfig",
    "generate_all_runs",
    "generate_runs",
    "EXPERIMENT_GENERATORS",
    "create_agent",
    "create_training_config",
    "ExperimentRunner",
    "Manifest",
    "MANIFEST_PATH",
    "run_all_baselines",
]
