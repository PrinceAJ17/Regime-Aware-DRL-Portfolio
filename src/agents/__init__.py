"""
RL Agents Module
================

This module contains all reinforcement learning agent implementations.

IMPLEMENTED ALGORITHMS:
=======================

Week 4 (PPO & SAC):
-------------------
1. PPOAgent - Proximal Policy Optimization
   - On-policy algorithm
   - Regime-aware through observation augmentation
   - Stable, widely-used baseline

2. SACAgent - Soft Actor-Critic  
   - Off-policy algorithm
   - DIRECTLY uses your RegimeAwareReplayBuffer!
   - Maximum entropy for exploration

Week 5 (TD3 & Rainbow) - To be implemented:
-------------------------------------------
3. TD3Agent - Twin Delayed DDPG
   - Off-policy, continuous actions
   - Uses your RegimeAwareReplayBuffer

4. RainbowAgent - Rainbow DQN
   - Off-policy, discrete actions
   - Combines multiple DQN improvements
   - Needs integration with PER + regime-awareness

YOUR NOVEL CONTRIBUTION:
========================

The key innovation is how each algorithm interfaces with your 
regime-aware replay buffer:

| Algorithm | Replay Type | Regime Integration |
|-----------|-------------|-------------------|
| PPO       | Rollout     | Observation augmentation |
| SAC       | Replay      | RegimeAwareReplayBuffer |
| TD3       | Replay      | RegimeAwareReplayBuffer |
| Rainbow   | PER         | Regime + Priority sampling |

USAGE EXAMPLES:
===============

# SAC with regime-aware replay (recommended for your research)
>>> from src.agents import SACAgent, SACConfig
>>> from src.replay import RegimeAwareReplayBuffer
>>> from src.regime import UnifiedRegimeDetector
>>>
>>> env = TradingEnv(...)
>>> detector = UnifiedRegimeDetector(method='rule_based')
>>> buffer = RegimeAwareReplayBuffer(buffer_size=100000, strategy='balanced')
>>>
>>> agent = SACAgent(
...     env=env,
...     regime_detector=detector,
...     replay_buffer=buffer,
...     sac_config=SACConfig(replay_strategy='balanced')
... )
>>> agent.train(TrainingConfig(total_timesteps=100000))

# PPO with regime-augmented observations
>>> from src.agents import PPOAgent, PPOConfig
>>>
>>> agent = PPOAgent(
...     env=env,
...     regime_detector=detector,
...     ppo_config=PPOConfig(add_regime_to_obs=True)
... )
>>> agent.train(TrainingConfig(total_timesteps=100000))

# Factory functions for quick setup
>>> from src.agents import create_sac_agent, create_ppo_agent
>>>
>>> sac = create_sac_agent(env, use_regime_aware=True, replay_strategy='inverse_frequency')
>>> ppo = create_ppo_agent(env, add_regime_to_obs=True)

Author: [Your Name]
Date: February 2026
"""

# Base classes and configs
from .base_agent import (
    BaseAgent,
    TrainingConfig,
    TrainingMetrics
)

# PPO Agent
from .ppo_agent import (
    PPOAgent,
    PPOConfig,
    RegimeObservationWrapper,
    RegimeMetricsCallback,
    create_ppo_agent
)

# SAC Agent
from .sac_agent import (
    SACAgent,
    SACConfig,
    RegimeAwareReplayBufferWrapper,
    RegimeTrackingCallback,
    create_sac_agent
)

# Convenience: all agents
AVAILABLE_AGENTS = {
    'ppo': PPOAgent,
    'sac': SACAgent,
    # 'td3': TD3Agent,  # Week 5
    # 'rainbow': RainbowAgent,  # Week 5
}

__all__ = [
    # Base
    'BaseAgent',
    'TrainingConfig',
    'TrainingMetrics',
    
    # PPO
    'PPOAgent',
    'PPOConfig',
    'RegimeObservationWrapper',
    'RegimeMetricsCallback',
    'create_ppo_agent',
    
    # SAC
    'SACAgent',
    'SACConfig',
    'RegimeAwareReplayBufferWrapper',
    'RegimeTrackingCallback',
    'create_sac_agent',
    
    # Registry
    'AVAILABLE_AGENTS',
]