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

Author: [Your Name]
Date: February 2026
"""

from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Union
from collections import deque
import numpy as np
import random


class MarketRegime(Enum):
    """Market regime labels."""
    BULL = 'bull'
    BEAR = 'bear'
    SIDEWAYS = 'sideways'


class SamplingStrategy(Enum):
    """Sampling strategies for regime-aware replay."""
    BALANCED = 'balanced'
    CURRENT_FOCUSED = 'current_focused'
    INVERSE_FREQUENCY = 'inverse_frequency'
    ADAPTIVE = 'adaptive'


@dataclass
class Experience:
    """Standard experience tuple."""
    state: np.ndarray
    action: np.ndarray
    reward: float
    next_state: np.ndarray
    done: bool
    info: Optional[Dict[str, Any]] = None


@dataclass
class RegimeExperience:
    """Experience with regime label."""
    state: np.ndarray
    action: np.ndarray
    reward: float
    next_state: np.ndarray
    done: bool
    regime: MarketRegime
    info: Optional[Dict[str, Any]] = None
    td_error: float = 1.0


class StandardReplayBuffer:
    """
    Standard uniform random replay buffer (baseline).
    
    This is what every RL library uses by default.
    We use this as a BASELINE to compare against your regime-aware version.
    """
    
    def __init__(self, buffer_size: int = 1_000_000, random_seed: Optional[int] = None):
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=buffer_size)
        
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)
    
    def add(self, state, action, reward, next_state, done, info=None):
        """Add experience to buffer."""
        exp = Experience(
            state=np.array(state, dtype=np.float32),
            action=np.array(action, dtype=np.float32),
            reward=float(reward),
            next_state=np.array(next_state, dtype=np.float32),
            done=bool(done),
            info=info
        )
        self.buffer.append(exp)
    
    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        """Sample batch uniformly at random."""
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
        
        experiences = random.sample(list(self.buffer), batch_size)
        
        return {
            'states': np.array([e.state for e in experiences]),
            'actions': np.array([e.action for e in experiences]),
            'rewards': np.array([e.reward for e in experiences]),
            'next_states': np.array([e.next_state for e in experiences]),
            'dones': np.array([e.done for e in experiences])
        }
    
    def __len__(self):
        return len(self.buffer)
    
    def is_full(self):
        return len(self.buffer) >= self.buffer_size


class RegimeAwareReplayBuffer:
    """
    Regime-aware experience replay buffer.
    
    THIS IS YOUR CORE NOVEL CONTRIBUTION!
    
    Key features:
    1. Experiences labeled with market regime at storage time
    2. Separate storage per regime
    3. Intelligent sampling strategies
    """
    
    def __init__(
        self,
        buffer_size: int = 1_000_000,
        strategy: Union[str, SamplingStrategy] = 'balanced',
        current_focus_ratio: float = 0.6,
        random_seed: Optional[int] = None
    ):
        self.buffer_size = buffer_size
        self.per_regime_size = buffer_size // 3
        
        # Convert string to enum if needed
        if isinstance(strategy, str):
            strategy = SamplingStrategy(strategy)
        self.strategy = strategy
        self.current_focus_ratio = current_focus_ratio
        
        # Separate buffers per regime
        self.buffers: Dict[MarketRegime, deque] = {
            MarketRegime.BULL: deque(maxlen=self.per_regime_size),
            MarketRegime.BEAR: deque(maxlen=self.per_regime_size),
            MarketRegime.SIDEWAYS: deque(maxlen=self.per_regime_size)
        }
        
        # TD errors for adaptive sampling
        self.td_errors: Dict[MarketRegime, List[float]] = {
            MarketRegime.BULL: [],
            MarketRegime.BEAR: [],
            MarketRegime.SIDEWAYS: []
        }
        
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)
    
    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        regime: Union[MarketRegime, str],
        info: Optional[Dict] = None,
        td_error: float = 1.0
    ):
        """Add experience with regime label."""
        # Convert string to enum
        if isinstance(regime, str):
            regime = MarketRegime(regime)
        
        exp = RegimeExperience(
            state=np.array(state, dtype=np.float32),
            action=np.array(action, dtype=np.float32),
            reward=float(reward),
            next_state=np.array(next_state, dtype=np.float32),
            done=bool(done),
            regime=regime,
            info=info,
            td_error=td_error
        )
        
        self.buffers[regime].append(exp)
        self.td_errors[regime].append(td_error)
        
        # Keep TD errors list in sync with buffer
        if len(self.td_errors[regime]) > len(self.buffers[regime]):
            self.td_errors[regime] = self.td_errors[regime][-len(self.buffers[regime]):]
    
    def sample(
        self,
        batch_size: int,
        strategy: Optional[Union[str, SamplingStrategy]] = None,
        current_regime: Optional[MarketRegime] = None
    ) -> Dict[str, np.ndarray]:
        """
        Sample batch using specified strategy.
        
        Parameters:
        -----------
        batch_size : int
            Number of experiences to sample
        strategy : str or SamplingStrategy, optional
            Override default strategy
        current_regime : MarketRegime, optional
            Required for current_focused strategy
        """
        if strategy is None:
            strategy = self.strategy
        elif isinstance(strategy, str):
            strategy = SamplingStrategy(strategy)
        
        # Calculate samples per regime
        if strategy == SamplingStrategy.BALANCED:
            samples_per_regime = self._balanced_allocation(batch_size)
        elif strategy == SamplingStrategy.CURRENT_FOCUSED:
            samples_per_regime = self._current_focused_allocation(batch_size, current_regime)
        elif strategy == SamplingStrategy.INVERSE_FREQUENCY:
            samples_per_regime = self._inverse_frequency_allocation(batch_size)
        elif strategy == SamplingStrategy.ADAPTIVE:
            samples_per_regime = self._adaptive_allocation(batch_size)
        else:
            samples_per_regime = self._balanced_allocation(batch_size)
        
        # Sample from each regime
        experiences = []
        regimes = []
        
        for regime, n_samples in samples_per_regime.items():
            if n_samples > 0 and len(self.buffers[regime]) > 0:
                actual_samples = min(n_samples, len(self.buffers[regime]))
                sampled = random.sample(list(self.buffers[regime]), actual_samples)
                experiences.extend(sampled)
                regimes.extend([regime.value] * actual_samples)
        
        if not experiences:
            # Return empty batch
            return {
                'states': np.array([]),
                'actions': np.array([]),
                'rewards': np.array([]),
                'next_states': np.array([]),
                'dones': np.array([]),
                'regimes': np.array([])
            }
        
        # Convert to numpy arrays
        regime_to_idx = {'bull': 0, 'bear': 1, 'sideways': 2}
        
        return {
            'states': np.array([e.state for e in experiences]),
            'actions': np.array([e.action for e in experiences]),
            'rewards': np.array([e.reward for e in experiences]),
            'next_states': np.array([e.next_state for e in experiences]),
            'dones': np.array([e.done for e in experiences]),
            'regimes': np.array([regime_to_idx[r] for r in regimes])
        }
    
    def _balanced_allocation(self, batch_size: int) -> Dict[MarketRegime, int]:
        """Equal samples from each regime."""
        n_per_regime = batch_size // 3
        remainder = batch_size % 3
        
        allocation = {
            MarketRegime.BULL: n_per_regime,
            MarketRegime.BEAR: n_per_regime,
            MarketRegime.SIDEWAYS: n_per_regime
        }
        
        # Distribute remainder
        for i, regime in enumerate([MarketRegime.BULL, MarketRegime.BEAR, MarketRegime.SIDEWAYS]):
            if i < remainder:
                allocation[regime] += 1
        
        return allocation
    
    def _current_focused_allocation(
        self, 
        batch_size: int, 
        current_regime: Optional[MarketRegime]
    ) -> Dict[MarketRegime, int]:
        """Focus on current regime."""
        if current_regime is None:
            return self._balanced_allocation(batch_size)
        
        n_current = int(batch_size * self.current_focus_ratio)
        n_other = batch_size - n_current
        n_per_other = n_other // 2
        
        allocation = {}
        for regime in MarketRegime:
            if regime == current_regime:
                allocation[regime] = n_current
            else:
                allocation[regime] = n_per_other
        
        return allocation
    
    def _inverse_frequency_allocation(self, batch_size: int) -> Dict[MarketRegime, int]:
        """Oversample rare regimes."""
        counts = {r: len(self.buffers[r]) for r in MarketRegime}
        total = sum(counts.values())
        
        if total == 0:
            return self._balanced_allocation(batch_size)
        
        # Inverse weights
        weights = {}
        for regime in MarketRegime:
            if counts[regime] > 0:
                weights[regime] = total / counts[regime]
            else:
                weights[regime] = 0
        
        # Normalize
        total_weight = sum(weights.values())
        if total_weight == 0:
            return self._balanced_allocation(batch_size)
        
        allocation = {}
        allocated = 0
        for regime in MarketRegime:
            n = int(batch_size * weights[regime] / total_weight)
            allocation[regime] = n
            allocated += n
        
        # Handle remainder
        remainder = batch_size - allocated
        if remainder > 0:
            # Give to rarest regime with samples
            rarest = min(
                [r for r in MarketRegime if counts[r] > 0],
                key=lambda r: counts[r],
                default=MarketRegime.SIDEWAYS
            )
            allocation[rarest] += remainder
        
        return allocation
    
    def _adaptive_allocation(self, batch_size: int) -> Dict[MarketRegime, int]:
        """Allocate based on TD errors."""
        mean_errors = {}
        for regime in MarketRegime:
            if self.td_errors[regime]:
                mean_errors[regime] = np.mean(self.td_errors[regime])
            else:
                mean_errors[regime] = 1.0
        
        total_error = sum(mean_errors.values())
        if total_error == 0:
            return self._balanced_allocation(batch_size)
        
        allocation = {}
        allocated = 0
        for regime in MarketRegime:
            n = int(batch_size * mean_errors[regime] / total_error)
            allocation[regime] = n
            allocated += n
        
        # Handle remainder
        remainder = batch_size - allocated
        if remainder > 0:
            highest_error = max(mean_errors, key=mean_errors.get)
            allocation[highest_error] += remainder
        
        return allocation
    
    def get_regime_counts(self) -> Dict[MarketRegime, int]:
        """Get count of experiences per regime."""
        return {r: len(self.buffers[r]) for r in MarketRegime}
    
    def __len__(self):
        return sum(len(b) for b in self.buffers.values())
    
    def is_full(self):
        return len(self) >= self.buffer_size


__all__ = [
    'MarketRegime',
    'SamplingStrategy',
    'Experience',
    'RegimeExperience',
    'StandardReplayBuffer',
    'RegimeAwareReplayBuffer',
]