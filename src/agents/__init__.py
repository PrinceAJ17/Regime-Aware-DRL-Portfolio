"""
RL Agents for Portfolio Management.

This module provides wrappers around reinforcement learning algorithms
for use in portfolio optimization with regime-aware experience replay.

Agents:
-------
- PPOAgent: Proximal Policy Optimization (on-policy, stable)
- SACAgent: Soft Actor-Critic (off-policy, sample-efficient)
- TD3Agent: Twin Delayed DDPG (off-policy, stable)
- RainbowAgent: Rainbow DQN (off-policy, discrete actions)

All agents support:
- Custom replay buffers (including RegimeAwareReplayBuffer)
- Checkpointing and model loading
- TensorBoard logging
- Evaluation and comparison

Usage:
------
>>> from src.agents import PPOAgent, SACAgent, TD3Agent, RainbowAgent
>>> 
>>> # For continuous actions (PPO, SAC, TD3)
>>> agent = SACAgent(trading_env, replay_buffer=regime_aware_buffer)
>>> agent.train(total_timesteps=1_000_000)
>>> 
>>> # For discrete actions (Rainbow)
>>> from src.agents.rainbow_agent import DiscreteActionWrapper
>>> discrete_env = DiscreteActionWrapper(trading_env)
>>> agent = RainbowAgent(discrete_env, replay_buffer=regime_aware_buffer)
>>> agent.train(total_timesteps=1_000_000)

Author: Prince Alexander John
Date: February 2026
Project: Regime-Aware Experience Replay for DRL Portfolio Management
"""

from .base_agent import BaseAgent, ReplayBufferWrapper
from .ppo_agent import PPOAgent
from .sac_agent import SACAgent
from .td3_agent import TD3Agent
from .rainbow_agent import RainbowAgent, DiscreteActionWrapper

__all__ = [
    # Base class
    "BaseAgent",
    "ReplayBufferWrapper",
    
    # Continuous action agents
    "PPOAgent",
    "SACAgent", 
    "TD3Agent",
    
    # Discrete action agents
    "RainbowAgent",
    "DiscreteActionWrapper",
]

# Agent registry for easy iteration
AGENTS = {
    "ppo": PPOAgent,
    "sac": SACAgent,
    "td3": TD3Agent,
    "rainbow": RainbowAgent,
}

# Agents that use replay buffers (benefit from regime-aware replay)
REPLAY_BUFFER_AGENTS = ["sac", "td3", "rainbow"]

# Agents that need continuous actions
CONTINUOUS_AGENTS = ["ppo", "sac", "td3"]

# Agents that need discrete actions
DISCRETE_AGENTS = ["rainbow"]
