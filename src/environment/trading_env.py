"""
Trading Environment Module
==========================

This module implements the core trading environment that your RL agents interact with.

What is a Trading Environment?
------------------------------
A trading environment simulates the financial market. The RL agent:
1. Observes the current state (prices, indicators, portfolio)
2. Takes an action (rebalance the portfolio)
3. Receives a reward (profit/loss adjusted for risk)
4. Observes the next state

This is a Gymnasium environment, meaning it follows a standard interface
that any RL algorithm can interact with.

Key Components
--------------
- State space: What the agent sees (observation)
- Action space: What the agent can do (portfolio weights)
- Reward function: How we measure success
- Transition dynamics: How the world changes

Why Custom Environment?
-----------------------
Standard RL environments (CartPole, Atari games) don't capture the nuances
of financial markets:
- Transaction costs
- Non-stationary dynamics
- Multi-asset portfolios
- Risk management

We need a custom environment tailored to portfolio management.

Usage
-----
>>> from src.environment.trading_env import TradingEnvironment
>>> env = TradingEnvironment(data, config)
>>> state, info = env.reset()
>>> action = agent.predict(state)
>>> next_state, reward, terminated, truncated, info = env.step(action)
"""

# TODO: Implement in Week 2
#
# Class to implement:
# - TradingEnvironment(gymnasium.Env)
#     - __init__(self, data, config)
#     - reset(self, seed=None)
#     - step(self, action)
#     - _get_observation(self)
#     - _calculate_reward(self)
#     - _execute_trades(self, action)

pass
