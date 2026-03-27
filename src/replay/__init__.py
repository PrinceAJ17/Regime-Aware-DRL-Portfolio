"""
Experience Replay Module
========================

This module contains replay buffer implementations.

IMPLEMENTED BUFFERS:
====================

1. StandardReplayBuffer (baseline)
   - Uniform random sampling
   - What every RL library uses by default

2. RegimeAwareReplayBuffer (YOUR NOVEL CONTRIBUTION!)
   - Labels experiences by market regime
   - Four intelligent sampling strategies:
     * Balanced: Equal samples from each regime
     * Current-Focused: Prioritize current market regime
     * Inverse-Frequency: Oversample rare regimes
     * Adaptive: Weight by TD-error per regime

WHY THIS MATTERS:
=================

Standard replay samples uniformly:
- If 60% of data is bull market, 60% of training is bull market
- Agent becomes expert at bull, fails at bear

Your regime-aware replay:
- Ensures balanced learning across ALL market conditions
- Agent handles regime changes gracefully

USAGE:
======
>>> from src.replay import RegimeAwareReplayBuffer, MarketRegime
>>>
>>> # Create buffer with balanced sampling
>>> buffer = RegimeAwareReplayBuffer(buffer_size=100000, strategy='balanced')
>>>
>>> # Add experiences with regime labels
>>> buffer.add(state, action, reward, next_state, done, MarketRegime.BULL)
>>>
>>> # Sample with intelligent strategy
>>> batch = buffer.sample(batch_size=64)

Author: Prince Alexander John
Date: February 2026
"""

# Re-export from canonical modules (no duplicate implementations)
from .standard_buffer import Experience, StandardReplayBuffer
from .regime_aware_buffer import (
    MarketRegime,
    SamplingStrategy,
    RegimeExperience,
    RegimeAwareReplayBuffer,
    RegimeAwarePrioritizedBuffer,
)


__all__ = [
    'MarketRegime',
    'SamplingStrategy',
    'Experience',
    'RegimeExperience',
    'StandardReplayBuffer',
    'RegimeAwareReplayBuffer',
    'RegimeAwarePrioritizedBuffer',
]
