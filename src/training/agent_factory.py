"""
Agent Factory
=============

Creates fully configured agents from RunConfig. Handles all per-agent
quirks: replay buffer wrappers, observation wrappers, discrete action
wrappers, and regime detection setup.

Agent-specific behaviour:
- PPO: On-policy, no replay buffer. Uses RegimeObservationWrapper for regime awareness.
- SAC: Off-policy. Uses RegimeAwareReplayBuffer + SB3-compatible wrapper.
- TD3: Off-policy. Uses RegimeAwareReplayBuffer + base_agent ReplayBufferWrapper.
- Rainbow: Custom training loop. Requires DiscreteActionWrapper on env.

Author: Prince Alexander John
Date: February 2026
"""

import gymnasium as gym
from typing import Optional

from src.agents.base_agent import TrainingConfig
from src.agents.ppo_agent import PPOAgent, PPOConfig
from src.agents.sac_agent import SACAgent, SACConfig
from src.agents.td3_agent import TD3Agent
from src.agents.rainbow_agent import RainbowAgent, DiscreteActionWrapper
from src.replay import RegimeAwareReplayBuffer, RegimeAwarePrioritizedBuffer

from .run_config import RunConfig


def create_agent(
    run_config: RunConfig,
    train_env: gym.Env,
    regime_detector,
):
    """
    Create a fully configured agent from a RunConfig.

    Parameters
    ----------
    run_config : RunConfig
        The run configuration.
    train_env : gym.Env
        The training environment (unwrapped).
    regime_detector
        Pre-configured UnifiedRegimeDetector.

    Returns
    -------
    BaseAgent
        Configured agent ready for training.
    """
    algo = run_config.algorithm
    is_regime_aware = (run_config.strategy != "standard")

    if algo == "ppo":
        return _create_ppo(run_config, train_env, regime_detector, is_regime_aware)
    elif algo == "sac":
        return _create_sac(run_config, train_env, regime_detector, is_regime_aware)
    elif algo == "td3":
        return _create_td3(run_config, train_env, regime_detector, is_regime_aware)
    elif algo == "rainbow":
        return _create_rainbow(run_config, train_env, regime_detector, is_regime_aware)
    else:
        raise ValueError(f"Unknown algorithm: {algo}")


def _create_ppo(run_config, train_env, detector, is_regime_aware):
    """PPO: on-policy, regime awareness via observation augmentation."""
    ppo_config = PPOConfig(
        add_regime_to_obs=is_regime_aware,
        track_regime_metrics=True,
    )
    return PPOAgent(
        env=train_env,
        name=run_config.run_id,
        ppo_config=ppo_config,
        regime_detector=detector if is_regime_aware else None,
        verbose=0,
    )


def _create_sac(run_config, train_env, detector, is_regime_aware):
    """SAC: off-policy, uses RegimeAwareReplayBuffer."""
    replay_buffer = None
    if is_regime_aware:
        replay_buffer = RegimeAwareReplayBuffer(
            buffer_size=1_000_000,
            strategy=run_config.strategy,
            random_seed=run_config.seed,
        )

    sac_config = SACConfig(
        use_regime_aware_replay=is_regime_aware,
        replay_strategy=run_config.strategy if is_regime_aware else "balanced",
    )
    return SACAgent(
        env=train_env,
        name=run_config.run_id,
        sac_config=sac_config,
        regime_detector=detector,
        replay_buffer=replay_buffer,
        verbose=0,
    )


def _create_td3(run_config, train_env, detector, is_regime_aware):
    """TD3: off-policy, uses ReplayBufferWrapper from base_agent."""
    replay_buffer = None
    if is_regime_aware:
        replay_buffer = RegimeAwareReplayBuffer(
            buffer_size=1_000_000,
            strategy=run_config.strategy,
            random_seed=run_config.seed,
        )

    return TD3Agent(
        env=train_env,
        name=run_config.run_id,
        config=None,
        replay_buffer=replay_buffer,
        regime_detector=detector,
        seed=run_config.seed,
        verbose=0,
    )


def _create_rainbow(run_config, train_env, detector, is_regime_aware):
    """Rainbow: custom loop, needs DiscreteActionWrapper."""
    discrete_env = DiscreteActionWrapper(train_env)

    replay_buffer = None
    if is_regime_aware:
        # Use RegimeAwarePrioritizedBuffer which combines PER with regime sampling
        # and returns the 7-tuple format Rainbow's update() expects
        replay_buffer = RegimeAwarePrioritizedBuffer(
            capacity=200_000,
            strategy=run_config.strategy,
            random_seed=run_config.seed,
        )

    return RainbowAgent(
        env=discrete_env,
        name=run_config.run_id,
        config=None,
        replay_buffer=replay_buffer,
        seed=run_config.seed,
        verbose=0,
        regime_detector=detector if is_regime_aware else None,
    )


def create_training_config(run_config: RunConfig) -> TrainingConfig:
    """Create TrainingConfig from RunConfig."""
    return TrainingConfig(
        total_timesteps=run_config.total_timesteps,
        eval_freq=50_000,
        n_eval_episodes=10,
        save_freq=run_config.total_timesteps,  # Save only at end
        save_path=str(run_config.output_dir / "checkpoints"),
        early_stopping=False,  # Run full timesteps for fair comparison
        tensorboard_log=None,
        seed=run_config.seed,
        device="auto",
    )
