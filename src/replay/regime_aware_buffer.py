"""
Regime-Aware Replay Buffer Module
==================================

THIS IS YOUR CORE RESEARCH CONTRIBUTION.

This module implements the novel regime-aware experience replay buffer
that is the main innovation of your project.

What is Experience Replay?
--------------------------
In reinforcement learning, the agent collects experiences as it interacts
with the environment:
    (state, action, reward, next_state, done)

Standard experience replay stores these in a buffer and samples them
randomly during training. This breaks temporal correlations and stabilises
learning.

The Problem with Standard Replay
--------------------------------
Financial markets have different "regimes" (bull, bear, sideways). If your
training data is 60% bull market, 25% sideways, 15% bear market, then
standard random sampling means your agent:
- Becomes expert at bull markets (lots of examples)
- Struggles with bear markets (few examples)
- Fails catastrophically when regime changes

This is the non-stationarity problem in financial RL.

Our Solution: Regime-Aware Replay
---------------------------------
Instead of random sampling, we:
1. Label each experience with its market regime
2. Store experiences in regime-specific buffers
3. Sample using intelligent strategies:
   - Balanced: Equal samples from each regime
   - Current-focused: Oversample current regime
   - Inverse-frequency: Oversample rare regimes
   - Adaptive: Based on learning signal (TD error)

This ensures the agent learns effectively across ALL market conditions.

Why This is Novel
-----------------
Prior work (Nixon Raj, 2025) added regime signals to the agent's observations.
We do something different: we modify HOW the agent learns, not just WHAT it sees.
This is a more fundamental change to the learning process.

Usage
-----
>>> from src.replay.regime_aware_buffer import RegimeAwareReplayBuffer
>>> buffer = RegimeAwareReplayBuffer(capacity=1_000_000, regime_detector=detector)
>>> buffer.add(state, action, reward, next_state, done)  # Auto-labels regime
>>> batch = buffer.sample(batch_size=256, strategy='balanced')
"""

# TODO: Implement in Week 3 (This is your most important week!)
#
# Classes to implement:
#
# 1. RegimeAwareReplayBuffer
#    - __init__(self, capacity, regime_detector)
#    - add(self, state, action, reward, next_state, done)
#    - sample(self, batch_size, strategy='balanced')
#    - __len__(self)
#
# 2. Sampling strategies (in sampling.py):
#    - balanced_sampling()
#    - current_focused_sampling()
#    - inverse_frequency_sampling()
#    - adaptive_sampling()
#
# Key implementation details:
# - Use collections.deque for each regime buffer (efficient FIFO)
# - Track regime counts for inverse-frequency weighting
# - Store TD errors for adaptive sampling (requires callback from training)

pass
