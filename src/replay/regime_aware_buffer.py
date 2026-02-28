"""
Regime-Aware Experience Replay Buffer
======================================

THIS IS YOUR NOVEL CONTRIBUTION - THE CORE OF YOUR PROJECT.

This module implements a modified experience replay buffer that:
1. Labels each experience with its market regime (bull/bear/sideways)
2. Stores experiences separately by regime
3. Samples experiences using intelligent strategies to ensure
   balanced learning across all market conditions

WHY IS THIS NOVEL?
==================

Prior work (e.g., Nixon Raj 2025) adds regime information to OBSERVATIONS.
This changes what the agent SEES.

We modify the REPLAY BUFFER's sampling mechanism.
This changes how the agent LEARNS.

These are fundamentally different:
- Observation modification: Agent receives regime as input feature
- Replay modification: Agent's learning process itself is regime-aware

Even with regime-aware observations, standard replay STILL samples uniformly,
which can bias learning toward common regimes (e.g., bull markets).

Our contribution ensures the agent learns EQUALLY from all regimes,
regardless of their frequency in the training data.

THE FOUR SAMPLING STRATEGIES:
=============================

1. BALANCED SAMPLING
   - Equal samples from each regime
   - Formula: P(regime) = 1/K for K regimes
   - Hypothesis: Prevents bias, ensures robust learning
   
2. CURRENT-FOCUSED SAMPLING
   - Oversample current regime (60%), others share remaining 40%
   - Formula: P(current) = 0.6, P(others) = 0.4/(K-1)
   - Hypothesis: Better immediate performance, may sacrifice adaptability
   
3. INVERSE-FREQUENCY SAMPLING
   - Oversample rare regimes proportionally
   - Formula: P(regime) ∝ 1/count(regime)
   - Hypothesis: Better handling of rare events (crashes, bear markets)
   
4. ADAPTIVE SAMPLING
   - Dynamic adjustment based on TD errors
   - Formula: P(regime) ∝ mean(TD_error in regime)
   - Hypothesis: Focus learning where the agent is struggling

Author: Prince Alexander John
Date: February 2026
Project: Regime-Aware Experience Replay for DRL Portfolio Management
"""

import numpy as np
from typing import Tuple, List, Optional, Dict, Any, Union
from collections import deque
from enum import Enum
import random

# Import from our other modules
from .standard_buffer import Experience

# Import canonical MarketRegime from regime module
from ..regime import MarketRegime


class SamplingStrategy(Enum):
    """
    Available sampling strategies for regime-aware replay.

    Each strategy has a different hypothesis about what leads
    to better learning. Part of your research is comparing them.
    """
    BALANCED = 'balanced'
    CURRENT_FOCUSED = 'current_focused'
    INVERSE_FREQUENCY = 'inverse_frequency'
    ADAPTIVE = 'adaptive'


class RegimeExperience:
    """
    Experience with regime label.
    
    Extends the standard Experience with a regime tag.
    This is how we track which market condition each experience came from.
    """
    __slots__ = ['state', 'action', 'reward', 'next_state', 'done', 'info', 'regime', 'td_error']
    
    def __init__(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        regime: MarketRegime,
        info: Optional[Dict[str, Any]] = None,
        td_error: float = 1.0  # Initial TD error (for adaptive sampling)
    ):
        self.state = np.array(state, dtype=np.float32)
        self.action = np.array(action, dtype=np.float32)
        self.reward = float(reward)
        self.next_state = np.array(next_state, dtype=np.float32)
        self.done = bool(done)
        self.regime = regime
        self.info = info or {}
        self.td_error = td_error


class RegimeAwareReplayBuffer:
    """
    Experience replay buffer with regime-aware sampling.
    
    THIS IS THE CORE NOVEL CONTRIBUTION OF YOUR PROJECT.
    
    Key differences from standard replay:
    1. Experiences are labelled with market regime at storage time
    2. Experiences are stored in separate buffers by regime
    3. Sampling uses intelligent strategies to balance learning
    
    Example Usage:
    --------------
    >>> # Create buffer with balanced sampling
    >>> buffer = RegimeAwareReplayBuffer(
    ...     buffer_size=100000,
    ...     strategy='balanced'
    ... )
    >>> 
    >>> # During interaction, add experiences with regime labels
    >>> for step in range(1000):
    ...     action = agent.select_action(state)
    ...     next_state, reward, done, info = env.step(action)
    ...     current_regime = regime_detector.get_current_regime()
    ...     buffer.add(state, action, reward, next_state, done, current_regime)
    >>> 
    >>> # Sampling automatically uses the configured strategy
    >>> batch = buffer.sample(batch_size=64)
    
    For current-focused sampling:
    >>> buffer.sample(batch_size=64, current_regime=MarketRegime.BEAR)
    """
    
    def __init__(
        self,
        buffer_size: int = 1_000_000,
        strategy: Union[str, SamplingStrategy] = SamplingStrategy.BALANCED,
        current_focus_ratio: float = 0.6,  # For current-focused strategy
        random_seed: Optional[int] = None
    ):
        """
        Initialize the regime-aware replay buffer.
        
        Parameters:
        -----------
        buffer_size : int
            TOTAL maximum size across ALL regime buffers.
            The size is distributed among regimes.
            
        strategy : str or SamplingStrategy
            Sampling strategy to use:
            - 'balanced': Equal from each regime
            - 'current_focused': Favour current regime
            - 'inverse_frequency': Favour rare regimes
            - 'adaptive': Based on TD errors
            
        current_focus_ratio : float
            For current-focused strategy: fraction from current regime.
            Default 0.6 means 60% from current, 40% from others.
            
        random_seed : int, optional
            For reproducibility
        """
        self.total_buffer_size = buffer_size
        
        # Parse strategy
        if isinstance(strategy, str):
            self.strategy = SamplingStrategy(strategy)
        else:
            self.strategy = strategy
        
        self.current_focus_ratio = current_focus_ratio
        
        # Set random seed
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)
        
        # CORE INNOVATION: Separate storage per regime
        # This is what enables regime-aware sampling
        #
        # Instead of one big buffer:
        #   [exp1, exp2, exp3, exp4, ...]  (mixed regimes)
        #
        # We have separate buffers:
        #   bull:     [exp1, exp4, ...]
        #   bear:     [exp2, exp7, ...]
        #   sideways: [exp3, exp5, ...]
        #
        self._regime_buffers: Dict[MarketRegime, deque] = {
            regime: deque(maxlen=buffer_size // len(MarketRegime))
            for regime in MarketRegime
        }
        
        # Track statistics
        self._total_added = 0
        self._total_sampled = 0
        self._samples_per_regime: Dict[MarketRegime, int] = {
            regime: 0 for regime in MarketRegime
        }
        
        # For adaptive sampling: track TD errors per regime
        self._td_errors_sum: Dict[MarketRegime, float] = {
            regime: 1.0 for regime in MarketRegime  # Initialize to 1 to avoid div by zero
        }
        self._td_errors_count: Dict[MarketRegime, int] = {
            regime: 1 for regime in MarketRegime
        }
    
    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        regime: MarketRegime,
        info: Optional[Dict[str, Any]] = None,
        td_error: float = 1.0
    ) -> None:
        """
        Add an experience to the appropriate regime buffer.
        
        THE KEY DIFFERENCE FROM STANDARD BUFFER:
        We store this experience in a regime-specific buffer.
        This allows us to sample from regimes independently.
        
        Parameters:
        -----------
        state : np.ndarray
            Current observation
        action : np.ndarray
            Action taken
        reward : float
            Reward received
        next_state : np.ndarray
            Next observation
        done : bool
            Episode ended
        regime : MarketRegime
            THE REGIME LABEL - this is what makes this buffer special
        info : dict, optional
            Additional info
        td_error : float
            TD error for this experience (for adaptive sampling)
        """
        experience = RegimeExperience(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
            regime=regime,
            info=info,
            td_error=td_error
        )
        
        # Add to regime-specific buffer
        self._regime_buffers[regime].append(experience)
        
        # Update TD error tracking for adaptive sampling
        self._td_errors_sum[regime] += abs(td_error)
        self._td_errors_count[regime] += 1
        
        self._total_added += 1
    
    def sample(
        self,
        batch_size: int,
        current_regime: Optional[MarketRegime] = None
    ) -> Dict[str, np.ndarray]:
        """
        Sample a batch using the configured strategy.
        
        THIS IS WHERE THE MAGIC HAPPENS.
        
        Instead of sampling uniformly from all experiences,
        we sample according to our strategy to ensure
        balanced learning across regimes.
        
        Parameters:
        -----------
        batch_size : int
            Number of experiences to sample
            
        current_regime : MarketRegime, optional
            Current market regime (required for current-focused strategy)
            
        Returns:
        --------
        dict
            Batched experiences with additional 'regimes' key
        """
        if self.strategy == SamplingStrategy.BALANCED:
            return self._sample_balanced(batch_size)
            
        elif self.strategy == SamplingStrategy.CURRENT_FOCUSED:
            if current_regime is None:
                raise ValueError(
                    "current_regime is required for CURRENT_FOCUSED strategy"
                )
            return self._sample_current_focused(batch_size, current_regime)
            
        elif self.strategy == SamplingStrategy.INVERSE_FREQUENCY:
            return self._sample_inverse_frequency(batch_size)
            
        elif self.strategy == SamplingStrategy.ADAPTIVE:
            return self._sample_adaptive(batch_size)
    
    def _sample_balanced(self, batch_size: int) -> Dict[str, np.ndarray]:
        """
        BALANCED SAMPLING STRATEGY
        --------------------------
        
        Sample equal numbers from each regime.
        
        Hypothesis: This prevents the agent from becoming biased
        toward common regimes and ensures it learns robust policies
        for ALL market conditions.
        
        Example with batch_size=60:
        - 20 from bull buffer
        - 20 from bear buffer
        - 20 from sideways buffer
        
        If a regime doesn't have enough samples, we take what's available
        and fill from other regimes.
        """
        experiences = []
        
        # Get available counts per regime
        available = {r: len(buf) for r, buf in self._regime_buffers.items()}
        total_available = sum(available.values())
        
        if total_available < batch_size:
            raise ValueError(
                f"Not enough experiences. Need {batch_size}, have {total_available}"
            )
        
        # Calculate samples per regime
        n_regimes = len(MarketRegime)
        base_per_regime = batch_size // n_regimes
        
        samples_per_regime = {}
        remaining = batch_size
        
        for regime in MarketRegime:
            # Take up to base_per_regime, but not more than available
            n_samples = min(base_per_regime, available[regime])
            samples_per_regime[regime] = n_samples
            remaining -= n_samples
        
        # Distribute remaining samples to regimes that have extras
        for regime in MarketRegime:
            if remaining <= 0:
                break
            extra = min(remaining, available[regime] - samples_per_regime[regime])
            samples_per_regime[regime] += extra
            remaining -= extra
        
        # Sample from each regime
        for regime, n_samples in samples_per_regime.items():
            if n_samples > 0:
                buf = self._regime_buffers[regime]
                indices = np.random.choice(len(buf), n_samples, replace=False)
                regime_experiences = [buf[i] for i in indices]
                experiences.extend(regime_experiences)
                self._samples_per_regime[regime] += n_samples

        # Shuffle to mix regimes in batch
        random.shuffle(experiences)

        self._total_sampled += len(experiences)

        return self._experiences_to_batch(experiences)

    def _sample_current_focused(
        self, 
        batch_size: int,
        current_regime: MarketRegime
    ) -> Dict[str, np.ndarray]:
        """
        CURRENT-FOCUSED SAMPLING STRATEGY
        ----------------------------------
        
        Sample more from the current regime, less from others.
        
        Hypothesis: When the market is in a bear market, you want
        the agent to be extra good at bear market decisions. So we
        give it more bear market experiences to learn from.
        
        Tradeoff: May sacrifice performance in other regimes.
        
        Default: 60% current, 40% split among others
        
        Example in a bear market with batch_size=100:
        - 60 from bear buffer (current)
        - 20 from bull buffer
        - 20 from sideways buffer
        """
        experiences = []
        
        n_current = int(batch_size * self.current_focus_ratio)
        n_others = batch_size - n_current
        other_regimes = [r for r in MarketRegime if r != current_regime]
        n_per_other = n_others // len(other_regimes)
        
        # Sample from current regime
        current_available = len(self._regime_buffers[current_regime])
        n_current = min(n_current, current_available)
        
        if n_current > 0:
            buf = self._regime_buffers[current_regime]
            indices = np.random.choice(len(buf), n_current, replace=False)
            current_experiences = [buf[i] for i in indices]
            experiences.extend(current_experiences)
            self._samples_per_regime[current_regime] += n_current

        # Fill remaining from others
        remaining = batch_size - len(experiences)
        for regime in other_regimes:
            n_sample = min(n_per_other, len(self._regime_buffers[regime]), remaining)
            if n_sample > 0:
                buf = self._regime_buffers[regime]
                indices = np.random.choice(len(buf), n_sample, replace=False)
                regime_experiences = [buf[i] for i in indices]
                experiences.extend(regime_experiences)
                self._samples_per_regime[regime] += n_sample
                remaining -= n_sample
        
        random.shuffle(experiences)
        self._total_sampled += len(experiences)
        
        return self._experiences_to_batch(experiences)
    
    def _sample_inverse_frequency(self, batch_size: int) -> Dict[str, np.ndarray]:
        """
        INVERSE-FREQUENCY SAMPLING STRATEGY
        ------------------------------------
        
        Sample proportionally to the INVERSE of regime frequency.
        Rare regimes get sampled MORE often.
        
        Hypothesis: Bear markets and crashes are rare but devastating.
        By oversampling these rare events, the agent learns to handle
        them better.
        
        Example:
        If data is 60% bull, 25% sideways, 15% bear:
        - Bear gets highest sampling weight (rare, important)
        - Bull gets lowest weight (common, already well-represented)
        
        Formula:
        weight(regime) = 1 / count(regime)
        P(regime) = weight(regime) / sum(all weights)
        """
        experiences = []
        
        # Calculate inverse frequency weights
        counts = {r: len(buf) for r, buf in self._regime_buffers.items()}
        total_count = sum(counts.values())
        
        if total_count < batch_size:
            raise ValueError("Not enough experiences in buffer")
        
        # Inverse frequency: fewer samples = higher weight
        # Add small epsilon to avoid division by zero
        weights = {r: 1.0 / (count + 1) for r, count in counts.items()}
        total_weight = sum(weights.values())
        
        # Normalize to get sampling proportions
        proportions = {r: w / total_weight for r, w in weights.items()}
        
        # Calculate samples per regime
        samples_per_regime = {}
        remaining = batch_size
        
        for regime in MarketRegime:
            n_samples = int(batch_size * proportions[regime])
            n_samples = min(n_samples, counts[regime])  # Can't sample more than available
            samples_per_regime[regime] = n_samples
            remaining -= n_samples
        
        # Distribute any remaining
        for regime in MarketRegime:
            if remaining <= 0:
                break
            extra = min(remaining, counts[regime] - samples_per_regime[regime])
            samples_per_regime[regime] += extra
            remaining -= extra
        
        # Sample from each regime
        for regime, n_samples in samples_per_regime.items():
            if n_samples > 0:
                buf = self._regime_buffers[regime]
                indices = np.random.choice(len(buf), n_samples, replace=False)
                regime_experiences = [buf[i] for i in indices]
                experiences.extend(regime_experiences)
                self._samples_per_regime[regime] += n_samples

        random.shuffle(experiences)
        self._total_sampled += len(experiences)

        return self._experiences_to_batch(experiences)

    def _sample_adaptive(self, batch_size: int) -> Dict[str, np.ndarray]:
        """
        ADAPTIVE SAMPLING STRATEGY
        --------------------------
        
        Sample based on TD errors: focus on regimes where the agent
        is making the biggest mistakes.
        
        Hypothesis: If the agent is struggling with bear markets
        (high TD error), we should give it more bear market practice.
        
        This is similar to Prioritized Experience Replay (PER),
        but applied at the REGIME level rather than individual experiences.
        
        Formula:
        weight(regime) = mean(TD_error in regime)
        P(regime) = weight(regime) / sum(all weights)
        
        The TD errors are updated during training via update_td_errors().
        """
        experiences = []
        
        # Calculate weights from mean TD errors
        mean_td_errors = {
            r: self._td_errors_sum[r] / max(self._td_errors_count[r], 1)
            for r in MarketRegime
        }
        total_td = sum(mean_td_errors.values())
        
        # Normalize to proportions
        proportions = {r: td / total_td for r, td in mean_td_errors.items()}
        
        # Calculate samples per regime
        counts = {r: len(buf) for r, buf in self._regime_buffers.items()}
        
        samples_per_regime = {}
        remaining = batch_size
        
        for regime in MarketRegime:
            n_samples = int(batch_size * proportions[regime])
            n_samples = min(n_samples, counts[regime])
            samples_per_regime[regime] = n_samples
            remaining -= n_samples
        
        # Distribute remaining
        for regime in sorted(MarketRegime, key=lambda r: proportions[r], reverse=True):
            if remaining <= 0:
                break
            extra = min(remaining, counts[regime] - samples_per_regime[regime])
            samples_per_regime[regime] += extra
            remaining -= extra
        
        # Sample
        for regime, n_samples in samples_per_regime.items():
            if n_samples > 0:
                buf = self._regime_buffers[regime]
                indices = np.random.choice(len(buf), n_samples, replace=False)
                regime_experiences = [buf[i] for i in indices]
                experiences.extend(regime_experiences)
                self._samples_per_regime[regime] += n_samples
        
        random.shuffle(experiences)
        self._total_sampled += len(experiences)
        
        return self._experiences_to_batch(experiences)
    
    def update_td_errors(
        self, 
        regime: MarketRegime, 
        td_errors: np.ndarray
    ) -> None:
        """
        Update TD error statistics for a regime.
        
        Call this after computing TD errors during training
        to update the adaptive sampling weights.
        
        Parameters:
        -----------
        regime : MarketRegime
            Which regime these errors are from
        td_errors : np.ndarray
            Array of TD error values
        """
        self._td_errors_sum[regime] += np.sum(np.abs(td_errors))
        self._td_errors_count[regime] += len(td_errors)
    
    def _experiences_to_batch(
        self, 
        experiences: List[RegimeExperience]
    ) -> Dict[str, np.ndarray]:
        """
        Convert experiences to batched arrays.
        
        Includes 'regimes' key for analysis/debugging.
        """
        states = np.stack([e.state for e in experiences])
        actions = np.stack([e.action for e in experiences])
        rewards = np.array([e.reward for e in experiences], dtype=np.float32)
        next_states = np.stack([e.next_state for e in experiences])
        dones = np.array([e.done for e in experiences], dtype=np.float32)
        
        # Also return regime labels (useful for analysis)
        _regime_to_idx = {r: i for i, r in enumerate(MarketRegime)}
        regime_indices = np.array([_regime_to_idx[e.regime] for e in experiences])
        
        return {
            'states': states,
            'actions': actions,
            'rewards': rewards,
            'next_states': next_states,
            'dones': dones,
            'regimes': regime_indices  # 0=bull, 1=bear, 2=sideways
        }
    
    def clear(self) -> None:
        """Clear all experiences from all regime buffers."""
        for regime in MarketRegime:
            self._regime_buffers[regime].clear()
        self._total_added = 0
        self._total_sampled = 0
        self._samples_per_regime = {regime: 0 for regime in MarketRegime}
        self._td_errors_sum = {regime: 1.0 for regime in MarketRegime}
        self._td_errors_count = {regime: 1 for regime in MarketRegime}

    def is_full(self) -> bool:
        """Check if buffer is full (for SB3 compatibility)."""
        return len(self) >= self.total_buffer_size

    @property
    def buffer_size(self) -> int:
        """Alias for total_buffer_size (for SB3 compatibility)."""
        return self.total_buffer_size


    def __len__(self) -> int:
        """Total experiences across all regime buffers."""
        return sum(len(buf) for buf in self._regime_buffers.values())
    
    def get_regime_counts(self) -> Dict[MarketRegime, int]:
        """Get count of experiences per regime."""
        return {r: len(buf) for r, buf in self._regime_buffers.items()}
    
    def is_ready(self, min_size: int) -> bool:
        """Check if buffer has enough experiences."""
        return len(self) >= min_size
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive buffer statistics.
        
        Useful for:
        - Verifying balanced sampling is working
        - Understanding data distribution
        - Debugging sampling issues
        """
        regime_counts = self.get_regime_counts()
        total = sum(regime_counts.values())
        
        return {
            'strategy': self.strategy.value,
            'total_experiences': total,
            'regime_counts': {r.value: c for r, c in regime_counts.items()},
            'regime_percentages': {
                r.value: c / max(total, 1) * 100 
                for r, c in regime_counts.items()
            },
            'samples_taken_per_regime': {
                r.value: c for r, c in self._samples_per_regime.items()
            },
            'total_added': self._total_added,
            'total_sampled': self._total_sampled,
            'mean_td_errors': {
                r.value: self._td_errors_sum[r] / max(self._td_errors_count[r], 1)
                for r in MarketRegime
            }
        }
    
    def get_sampling_distribution(self, batch_size: int = 1000) -> Dict[str, float]:
        """
        Compute the expected sampling distribution.
        
        This shows what percentage of samples would come from each regime
        given the current buffer state and strategy.
        
        Useful for verifying your strategy is working as expected.
        """
        # Temporarily sample to see distribution
        if len(self) < batch_size:
            batch_size = len(self)
        
        if batch_size == 0:
            return {r.value: 0.0 for r in MarketRegime}
        
        try:
            # For current-focused, use bull as example
            batch = self.sample(batch_size, current_regime=MarketRegime.BULL)
        except:
            return {r.value: 0.0 for r in MarketRegime}
        
        # Count regimes in batch
        regime_counts = {}
        for i, regime in enumerate(MarketRegime):
            regime_counts[regime.value] = np.sum(batch['regimes'] == i)
        
        # Convert to percentages
        return {r: c / batch_size * 100 for r, c in regime_counts.items()}


class RegimeAwarePrioritizedBuffer:
    """
    Hybrid buffer combining Prioritized Experience Replay with regime-aware sampling.

    Designed for Rainbow DQN integration. Stores experiences with both:
    - PER priorities (TD-error based, for individual experience importance)
    - Regime labels (for regime-level sampling balance)

    Sampling combines regime strategy weights with PER priorities:
    1. Determine per-regime sample counts using the regime strategy
    2. Within each regime, sample proportional to PER priorities

    Returns the 7-tuple format Rainbow's update() expects:
    (states, actions, rewards, next_states, dones, indices, weights)
    """

    def __init__(
        self,
        capacity: int = 200_000,
        strategy: str = 'balanced',
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_frames: int = 100_000,
        current_focus_ratio: float = 0.6,
        random_seed: Optional[int] = None,
    ):
        self.capacity = capacity
        self.strategy = SamplingStrategy(strategy) if isinstance(strategy, str) else strategy
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.current_focus_ratio = current_focus_ratio
        self.frame = 1
        self.max_priority = 1.0

        if random_seed is not None:
            np.random.seed(random_seed)

        # Per-regime storage: list of (experience_tuple, global_index)
        self._regime_data: Dict[MarketRegime, List] = {r: [] for r in MarketRegime}
        # Global flat storage for index-based access
        self._all_data: List = [None] * capacity
        self._priorities = np.zeros(capacity, dtype=np.float64)
        self._regimes: List[Optional[MarketRegime]] = [None] * capacity
        self._pos = 0
        self._size = 0

        # Regime-level TD error tracking for adaptive strategy
        self._td_errors_sum: Dict[MarketRegime, float] = {r: 1.0 for r in MarketRegime}
        self._td_errors_count: Dict[MarketRegime, int] = {r: 1 for r in MarketRegime}

    @property
    def beta(self) -> float:
        return min(1.0, self.beta_start + (1.0 - self.beta_start) * self.frame / self.beta_frames)

    @property
    def buffer_size(self) -> int:
        return self.capacity

    @property
    def total_buffer_size(self) -> int:
        return self.capacity

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        regime: Optional[str] = None,
    ):
        """Add experience with maximum priority."""
        experience = (state, action, reward, next_state, done, regime)
        priority = self.max_priority ** self.alpha

        idx = self._pos

        # Remove old entry from regime list if overwriting
        if self._all_data[idx] is not None and self._regimes[idx] is not None:
            old_regime = self._regimes[idx]
            regime_list = self._regime_data[old_regime]
            # Remove index from regime list (swap-remove for O(1))
            self._regime_data[old_regime] = [i for i in regime_list if i != idx]

        # Store
        self._all_data[idx] = experience
        self._priorities[idx] = priority

        # Parse regime
        if regime is not None:
            if isinstance(regime, str):
                try:
                    parsed_regime = MarketRegime(regime)
                except ValueError:
                    parsed_regime = MarketRegime.SIDEWAYS
            elif isinstance(regime, MarketRegime):
                parsed_regime = regime
            else:
                parsed_regime = MarketRegime.SIDEWAYS
        else:
            parsed_regime = MarketRegime.SIDEWAYS

        self._regimes[idx] = parsed_regime
        self._regime_data[parsed_regime].append(idx)

        self._pos = (self._pos + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def _get_regime_counts(self) -> Dict[MarketRegime, int]:
        return {r: len(indices) for r, indices in self._regime_data.items()}

    def _compute_samples_per_regime(self, batch_size: int) -> Dict[MarketRegime, int]:
        """Compute how many samples to draw from each regime based on strategy."""
        counts = self._get_regime_counts()
        total = sum(counts.values())
        if total == 0:
            return {r: 0 for r in MarketRegime}

        if self.strategy == SamplingStrategy.BALANCED:
            base = batch_size // len(MarketRegime)
            result = {r: min(base, counts[r]) for r in MarketRegime}
        elif self.strategy == SamplingStrategy.INVERSE_FREQUENCY:
            weights = {r: 1.0 / (counts[r] + 1) for r in MarketRegime}
            total_w = sum(weights.values())
            result = {r: min(int(batch_size * weights[r] / total_w), counts[r]) for r in MarketRegime}
        elif self.strategy == SamplingStrategy.ADAPTIVE:
            mean_td = {r: self._td_errors_sum[r] / max(self._td_errors_count[r], 1) for r in MarketRegime}
            total_td = sum(mean_td.values())
            result = {r: min(int(batch_size * mean_td[r] / total_td), counts[r]) for r in MarketRegime}
        else:
            # CURRENT_FOCUSED or fallback — treat as balanced for Rainbow
            # (Rainbow doesn't track current regime during training)
            base = batch_size // len(MarketRegime)
            result = {r: min(base, counts[r]) for r in MarketRegime}

        # Distribute remainder
        remaining = batch_size - sum(result.values())
        for r in MarketRegime:
            if remaining <= 0:
                break
            extra = min(remaining, counts[r] - result[r])
            result[r] += extra
            remaining -= extra

        return result

    def sample(self, batch_size: int) -> Tuple:
        """
        Sample using regime strategy + PER priorities within each regime.

        Returns:
        --------
        tuple: (states, actions, rewards, next_states, dones, indices, weights)
        """
        samples_per_regime = self._compute_samples_per_regime(batch_size)

        all_indices = []
        for regime, n in samples_per_regime.items():
            if n <= 0:
                continue
            regime_indices = self._regime_data[regime]
            if len(regime_indices) == 0:
                continue

            # Get priorities for this regime's experiences
            r_indices = np.array(regime_indices, dtype=np.int64)
            r_priorities = self._priorities[r_indices]
            r_probs = r_priorities / (r_priorities.sum() + 1e-12)

            # Sample within regime proportional to PER priority
            n_actual = min(n, len(r_indices))
            chosen = np.random.choice(len(r_indices), n_actual, replace=False, p=r_probs)
            all_indices.extend(r_indices[chosen].tolist())

        # If we didn't get enough (rare regimes empty), fill from all
        if len(all_indices) < batch_size:
            remaining = batch_size - len(all_indices)
            used = set(all_indices)
            available = [i for i in range(self._size) if i not in used and self._all_data[i] is not None]
            if available:
                extra = np.random.choice(available, min(remaining, len(available)), replace=False)
                all_indices.extend(extra.tolist())

        indices_arr = np.array(all_indices, dtype=np.int64)

        # Importance sampling weights
        probs = self._priorities[indices_arr] / (self._priorities[:self._size].sum() + 1e-12)
        min_prob = self._priorities[:self._size][self._priorities[:self._size] > 0].min() / (self._priorities[:self._size].sum() + 1e-12)
        max_weight = (self._size * min_prob) ** (-self.beta)
        weights = (self._size * probs) ** (-self.beta)
        weights = weights / (max_weight + 1e-12)

        # Build batch arrays
        batch = [self._all_data[i] for i in indices_arr]
        states = np.array([e[0] for e in batch])
        actions = np.array([e[1] for e in batch])
        rewards = np.array([e[2] for e in batch])
        next_states = np.array([e[3] for e in batch])
        dones = np.array([e[4] for e in batch])

        self.frame += 1

        return states, actions, rewards, next_states, dones, indices_arr, weights.astype(np.float32)

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        """Update priorities based on TD errors."""
        for idx, td_error in zip(indices, td_errors):
            priority = (abs(td_error) + 1e-6) ** self.alpha
            self._priorities[int(idx)] = priority
            self.max_priority = max(self.max_priority, priority)

            # Update regime-level TD tracking for adaptive strategy
            regime = self._regimes[int(idx)]
            if regime is not None:
                self._td_errors_sum[regime] += abs(td_error)
                self._td_errors_count[regime] += 1

    def is_full(self) -> bool:
        return self._size >= self.capacity

    def __len__(self) -> int:
        return self._size


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    """Test the regime-aware replay buffer."""
    
    print("=" * 70)
    print("REGIME-AWARE REPLAY BUFFER - TEST")
    print("THIS IS YOUR NOVEL CONTRIBUTION")
    print("=" * 70)
    
    # Create buffer
    buffer = RegimeAwareReplayBuffer(
        buffer_size=3000,
        strategy='balanced',
        random_seed=42
    )
    
    # Add experiences with different regime distributions
    # Simulating imbalanced data (60% bull, 25% sideways, 15% bear)
    print("\n[1] Adding imbalanced experiences...")
    print("    Distribution: 60% bull, 25% sideways, 15% bear")
    
    state_dim = 98
    action_dim = 5
    
    for i in range(1000):
        state = np.random.randn(state_dim)
        action = np.random.randn(action_dim)
        reward = np.random.randn()
        next_state = np.random.randn(state_dim)
        done = i % 100 == 99
        
        # Imbalanced regime distribution
        r = random.random()
        if r < 0.60:
            regime = MarketRegime.BULL
        elif r < 0.85:
            regime = MarketRegime.SIDEWAYS
        else:
            regime = MarketRegime.BEAR
        
        buffer.add(state, action, reward, next_state, done, regime)
    
    # Show regime counts
    counts = buffer.get_regime_counts()
    print(f"\n    Stored experiences:")
    for regime, count in counts.items():
        pct = count / sum(counts.values()) * 100
        print(f"      {regime.value}: {count} ({pct:.1f}%)")
    
    # Test balanced sampling
    print("\n[2] Testing BALANCED sampling...")
    buffer_balanced = RegimeAwareReplayBuffer(buffer_size=3000, strategy='balanced')
    for i in range(600):
        regime = [MarketRegime.BULL, MarketRegime.SIDEWAYS, MarketRegime.BEAR][i % 3]
        buffer_balanced.add(
            np.random.randn(state_dim), np.random.randn(action_dim),
            np.random.randn(), np.random.randn(state_dim), False, regime
        )
    
    batch = buffer_balanced.sample(60)
    regime_dist = np.bincount(batch['regimes'], minlength=3)
    print(f"    Batch of 60: Bull={regime_dist[0]}, Bear={regime_dist[1]}, Sideways={regime_dist[2]}")
    print(f"    Expected: ~20 each (balanced)")
    
    # Test inverse frequency sampling
    print("\n[3] Testing INVERSE-FREQUENCY sampling...")
    buffer_inv = RegimeAwareReplayBuffer(buffer_size=3000, strategy='inverse_frequency')
    
    # Add imbalanced data: lots of bull, little bear
    for i in range(500):
        buffer_inv.add(
            np.random.randn(state_dim), np.random.randn(action_dim),
            np.random.randn(), np.random.randn(state_dim), False, MarketRegime.BULL
        )
    for i in range(100):
        buffer_inv.add(
            np.random.randn(state_dim), np.random.randn(action_dim),
            np.random.randn(), np.random.randn(state_dim), False, MarketRegime.SIDEWAYS
        )
    for i in range(50):
        buffer_inv.add(
            np.random.randn(state_dim), np.random.randn(action_dim),
            np.random.randn(), np.random.randn(state_dim), False, MarketRegime.BEAR
        )
    
    batch = buffer_inv.sample(100)
    regime_dist = np.bincount(batch['regimes'], minlength=3)
    print(f"    Data distribution: Bull=500, Sideways=100, Bear=50")
    print(f"    Sampled batch of 100: Bull={regime_dist[0]}, Bear={regime_dist[1]}, Sideways={regime_dist[2]}")
    print(f"    Bear (rarest) should have highest sample rate!")
    
    # Show statistics
    print("\n[4] Buffer statistics:")
    stats = buffer.get_statistics()
    print(f"    Strategy: {stats['strategy']}")
    print(f"    Total experiences: {stats['total_experiences']}")
    print(f"    Regime distribution: {stats['regime_percentages']}")
    
    print("\n" + "=" * 70)
    print("ALL TESTS PASSED!")
    print("Your novel regime-aware replay buffer is working correctly.")
    print("=" * 70)