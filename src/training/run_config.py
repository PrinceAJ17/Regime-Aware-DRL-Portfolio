"""
Run Configuration
=================

Defines RunConfig (the atomic unit of work) and experiment generators
for all 6 experiments defined in configs/experiment_config.yaml.

Author: Prince Alexander John
Date: February 2026
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any


@dataclass
class RunConfig:
    """Configuration for a single training run."""
    experiment: str
    algorithm: str
    strategy: str
    seed: int
    total_timesteps: int

    # Experiment-specific parameters
    noise_level: float = 0.0              # E3: robustness
    feature_set: str = "full"             # E4: feature_ablation
    detection_method: str = "rule_based"  # E5: regime_detection
    eval_checkpoints: Optional[List[int]] = None  # E6: sample_efficiency

    @property
    def run_id(self) -> str:
        """Unique identifier for this run."""
        parts = [self.experiment, self.algorithm, self.strategy]
        if self.experiment == "robustness":
            parts.append(f"noise{self.noise_level}")
        if self.experiment == "feature_ablation":
            parts.append(self.feature_set)
        if self.experiment == "regime_detection":
            parts.append(self.detection_method)
        parts.append(f"seed{self.seed}")
        return "__".join(parts)

    @property
    def output_dir(self) -> Path:
        """Directory for this run's outputs."""
        return Path("results/experiments") / self.experiment / self.algorithm / self.strategy / self.run_id

    def to_dict(self) -> Dict[str, Any]:
        return {
            "experiment": self.experiment,
            "algorithm": self.algorithm,
            "strategy": self.strategy,
            "seed": self.seed,
            "total_timesteps": self.total_timesteps,
            "noise_level": self.noise_level,
            "feature_set": self.feature_set,
            "detection_method": self.detection_method,
            "eval_checkpoints": self.eval_checkpoints,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RunConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# =============================================================================
# Experiment Generators
# =============================================================================

ALGORITHMS = ["ppo", "sac", "td3", "rainbow"]
STRATEGIES = ["standard", "balanced", "current_focused", "inverse_frequency", "adaptive"]
SEEDS = [42, 123, 456, 789, 1000]


def generate_e1_runs() -> List[RunConfig]:
    """E1: Replay Strategy Comparison — 100 runs."""
    runs = []
    for algo in ALGORITHMS:
        for strategy in STRATEGIES:
            for seed in SEEDS:
                runs.append(RunConfig(
                    experiment="replay_comparison",
                    algorithm=algo,
                    strategy=strategy,
                    seed=seed,
                    total_timesteps=500_000,
                ))
    return runs


def generate_e2_runs() -> List[RunConfig]:
    """E2: Regime Performance — 60 runs."""
    runs = []
    for algo in ALGORITHMS:
        for strategy in ["standard", "balanced", "inverse_frequency"]:
            for seed in SEEDS:
                runs.append(RunConfig(
                    experiment="regime_performance",
                    algorithm=algo,
                    strategy=strategy,
                    seed=seed,
                    total_timesteps=500_000,
                ))
    return runs


def generate_e3_runs() -> List[RunConfig]:
    """E3: Robustness to Detection Noise (TD3 only) — 36 runs."""
    runs = []
    for noise in [0.0, 0.1, 0.2, 0.3]:
        for strategy in ["standard", "balanced", "inverse_frequency"]:
            for seed in [42, 123, 456]:
                runs.append(RunConfig(
                    experiment="robustness",
                    algorithm="td3",
                    strategy=strategy,
                    seed=seed,
                    total_timesteps=500_000,
                    noise_level=noise,
                ))
    return runs


def generate_e4_runs() -> List[RunConfig]:
    """E4: Feature Ablation — 20 runs."""
    runs = []
    for feature_set in ["full", "reduced"]:
        for algo in ["sac", "ppo"]:
            for seed in SEEDS:
                runs.append(RunConfig(
                    experiment="feature_ablation",
                    algorithm=algo,
                    strategy="balanced",
                    seed=seed,
                    total_timesteps=500_000,
                    feature_set=feature_set,
                ))
    return runs


def generate_e5_runs() -> List[RunConfig]:
    """E5: Regime Detection Method Comparison (SAC only) — 12 runs."""
    runs = []
    for method in ["rule_based", "hmm"]:
        for strategy in ["standard", "balanced"]:
            for seed in [42, 123, 456]:
                runs.append(RunConfig(
                    experiment="regime_detection",
                    algorithm="sac",
                    strategy=strategy,
                    seed=seed,
                    total_timesteps=500_000,
                    detection_method=method,
                ))
    return runs


def generate_e6_runs() -> List[RunConfig]:
    """E6: Sample Efficiency — 30 runs."""
    runs = []
    for strategy in ["standard", "balanced", "inverse_frequency"]:
        for algo in ["sac", "td3"]:
            for seed in SEEDS:
                runs.append(RunConfig(
                    experiment="sample_efficiency",
                    algorithm=algo,
                    strategy=strategy,
                    seed=seed,
                    total_timesteps=1_000_000,
                    eval_checkpoints=[50_000, 100_000, 250_000, 500_000, 1_000_000],
                ))
    return runs


EXPERIMENT_GENERATORS = {
    "replay_comparison": generate_e1_runs,
    "regime_performance": generate_e2_runs,
    "robustness": generate_e3_runs,
    "feature_ablation": generate_e4_runs,
    "regime_detection": generate_e5_runs,
    "sample_efficiency": generate_e6_runs,
}


def generate_runs(experiment: str) -> List[RunConfig]:
    """Generate runs for a specific experiment."""
    if experiment not in EXPERIMENT_GENERATORS:
        raise ValueError(f"Unknown experiment: {experiment}. Valid: {list(EXPERIMENT_GENERATORS.keys())}")
    return EXPERIMENT_GENERATORS[experiment]()


def generate_all_runs() -> List[RunConfig]:
    """Generate all 258 runs across all experiments."""
    all_runs = []
    for gen in EXPERIMENT_GENERATORS.values():
        all_runs.extend(gen())
    return all_runs
