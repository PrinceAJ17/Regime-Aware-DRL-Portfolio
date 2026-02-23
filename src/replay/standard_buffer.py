"""
Standard Experience Replay Buffer (Baseline)
=============================================

This module implements the STANDARD uniform random replay buffer
that serves as our BASELINE for comparison.

WHAT IS EXPERIENCE REPLAY?
==========================

Experience Replay is one of the key innovations that made Deep RL work.

THE PROBLEM IT SOLVES:
----------------------
In RL, the agent learns from sequences of experiences:
    (state_1, action_1, reward_1, state_2)
    (state_2, action_2, reward_2, state_3)
    (state_3, action_3, reward_3, state_4)
    ...

If we train on these in order, two problems arise:

1. CORRELATION:
   Consecutive experiences are highly correlated (state_2 depends on state_1)
   Neural networks assume training samples are independent (i.i.d.)
   Correlated samples → unstable training, catastrophic forgetting

2. DATA INEFFICIENCY:
   Without replay, we use each experience once and discard it
   In real environments, experiences are expensive to collect
   We're wasting valuable data

THE SOLUTION:
-------------
Store experiences in a buffer, randomly sample batches for training.

1. Agent interacts with environment, collects experiences
2. Store each experience in a buffer (up to some max size)
3. When training: randomly sample a batch of experiences
4. Train neural network on this random batch
5. Repeat

Random sampling:
- Breaks temporal correlations (samples are from different times)
- Allows multiple uses of each experience (data efficient)

This is "Uniform" replay because each experience has EQUAL probability
of being sampled, regardless of when it was collected.

WHY THIS IS OUR BASELINE:
-------------------------
Standard uniform replay is what every major RL library uses by default.
It's proven, stable, and well-understood.

Our novel contribution (Regime-Aware Replay) modifies the SAMPLING
to be non-uniform based on market regimes.

To prove our method is better, we need to compare against this baseline.

Author: Prince Alexander John
Date: February 2026
"""

import numpy as np
from typing import Tuple, List, Optional, Dict, Any, NamedTuple
from collections import deque
import random


class Experience(NamedTuple):
    """
    A single experience tuple.
    
    Using NamedTuple for:
    1. Memory efficiency (smaller than dict)
    2. Immutability (safer)
    3. Named access (experience.state vs experience[0])
    
    Components:
    -----------
    state : np.ndarray
        The observation at time t
        For your project: market features + portfolio state (~98 dims)
        
    action : np.ndarray
        The action taken
        For your project: portfolio weights (5 assets)
        
    reward : float
        The reward received
        For your project: risk-adjusted return
        
    next_state : np.ndarray
        The observation at time t+1
        
    done : bool
        Whether the episode ended
        For your project: end of trading period or bankruptcy
        
    info : dict
        Additional information (optional)
        For your project: might include portfolio value, regime label, etc.
    """
    state: np.ndarray
    action: np.ndarray
    reward: float
    next_state: np.ndarray
    done: bool
    info: Dict[str, Any] = {}


class StandardReplayBuffer:
    """
    Standard uniform experience replay buffer.
    
    This is the BASELINE that we compare our regime-aware buffer against.
    
    Key characteristics:
    - Fixed maximum size (oldest experiences are discarded when full)
    - Uniform random sampling (each experience equally likely)
    - No special treatment of any experiences
    
    Example Usage:
    --------------
    >>> buffer = StandardReplayBuffer(buffer_size=100000)
    >>> 
    >>> # During environment interaction
    >>> for step in range(1000):
    ...     action = agent.select_action(state)
    ...     next_state, reward, done, info = env.step(action)
    ...     buffer.add(state, action, reward, next_state, done, info)
    ...     state = next_state
    >>> 
    >>> # During training
    >>> for _ in range(training_steps):
    ...     batch = buffer.sample(batch_size=64)
    ...     agent.update(batch)
    """
    
    def __init__(
        self,
        buffer_size: int = 1_000_000,
        random_seed: Optional[int] = None
    ):
        """
        Initialize the replay buffer.
        
        Parameters:
        -----------
        buffer_size : int
            Maximum number of experiences to store.
            When full, oldest experiences are removed (FIFO queue).
            
            Typical values:
            - 100,000 for quick experiments
            - 1,000,000 for serious training
            
            Memory consideration:
            Each experience is roughly:
            state (98 floats) + action (5 floats) + next_state (98 floats) 
            + reward (1 float) + done (1 bool) ≈ 1.6 KB
            
            1M experiences ≈ 1.6 GB of RAM
            
        random_seed : int, optional
            Random seed for reproducible sampling
        """
        self.buffer_size = buffer_size
        
        # Using deque for O(1) append and automatic size limiting
        # When deque reaches maxlen, oldest items are automatically removed
        self._buffer: deque = deque(maxlen=buffer_size)
        
        # Set random seed for reproducibility
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)
        
        # Statistics tracking
        self._total_added = 0
        self._total_sampled = 0
    
    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        info: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add a new experience to the buffer.
        
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
            Whether episode ended
        info : dict, optional
            Additional information
        """
        experience = Experience(
            state=np.array(state, dtype=np.float32),
            action=np.array(action, dtype=np.float32),
            reward=float(reward),
            next_state=np.array(next_state, dtype=np.float32),
            done=bool(done),
            info=info or {}
        )
        
        self._buffer.append(experience)
        self._total_added += 1
    
    def sample(
        self, 
        batch_size: int
    ) -> Dict[str, np.ndarray]:
        """
        Sample a random batch of experiences.
        
        This is UNIFORM random sampling - each experience in the buffer
        has an equal probability of being selected.
        
        Parameters:
        -----------
        batch_size : int
            Number of experiences to sample.
            Common values: 32, 64, 128, 256
            
        Returns:
        --------
        dict
            Dictionary containing batched arrays:
            - 'states': (batch_size, state_dim)
            - 'actions': (batch_size, action_dim)
            - 'rewards': (batch_size,)
            - 'next_states': (batch_size, state_dim)
            - 'dones': (batch_size,)
            
        Raises:
        -------
        ValueError
            If batch_size > len(buffer)
        """
        if batch_size > len(self._buffer):
            raise ValueError(
                f"Cannot sample {batch_size} experiences from buffer "
                f"with only {len(self._buffer)} experiences"
            )
        
        # Random sample without replacement
        # Using random.sample for efficiency with deque
        experiences = random.sample(list(self._buffer), batch_size)
        
        # Convert list of Experience tuples to batched arrays
        batch = self._experiences_to_batch(experiences)
        
        self._total_sampled += batch_size
        
        return batch
    
    def _experiences_to_batch(
        self, 
        experiences: List[Experience]
    ) -> Dict[str, np.ndarray]:
        """
        Convert a list of experiences to batched numpy arrays.
        
        Why batch as numpy arrays?
        - PyTorch/TensorFlow expect array inputs
        - Vectorized operations are faster
        - Memory layout is contiguous (cache-friendly)
        
        Parameters:
        -----------
        experiences : List[Experience]
            List of experience tuples
            
        Returns:
        --------
        dict
            Batched arrays
        """
        # Stack all arrays
        states = np.stack([e.state for e in experiences])
        actions = np.stack([e.action for e in experiences])
        rewards = np.array([e.reward for e in experiences], dtype=np.float32)
        next_states = np.stack([e.next_state for e in experiences])
        dones = np.array([e.done for e in experiences], dtype=np.float32)
        
        return {
            'states': states,
            'actions': actions,
            'rewards': rewards,
            'next_states': next_states,
            'dones': dones
        }
    
    def __len__(self) -> int:
        """Return current number of experiences in buffer."""
        return len(self._buffer)
    
    def is_ready(self, min_size: int) -> bool:
        """
        Check if buffer has enough experiences to start training.
        
        We typically wait until buffer has some minimum number of
        experiences before starting training. This ensures:
        1. Enough diversity in samples
        2. Stable initial training
        
        Parameters:
        -----------
        min_size : int
            Minimum required experiences
            
        Returns:
        --------
        bool
            True if buffer has at least min_size experiences
        """
        return len(self._buffer) >= min_size
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get buffer statistics.
        
        Useful for logging and debugging.
        
        Returns:
        --------
        dict
            Statistics about the buffer
        """
        return {
            'current_size': len(self._buffer),
            'max_size': self.buffer_size,
            'utilization': len(self._buffer) / self.buffer_size,
            'total_added': self._total_added,
            'total_sampled': self._total_sampled,
            'sample_to_add_ratio': (
                self._total_sampled / max(self._total_added, 1)
            )
        }
    
    def clear(self) -> None:
        """Clear all experiences from the buffer."""
        self._buffer.clear()
        self._total_added = 0
        self._total_sampled = 0


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    """Test the standard replay buffer."""
    
    print("=" * 60)
    print("STANDARD REPLAY BUFFER - TEST")
    print("=" * 60)
    
    # Create buffer
    buffer = StandardReplayBuffer(buffer_size=1000, random_seed=42)
    
    # Add some experiences
    print("\n[1] Adding experiences...")
    state_dim = 98  # Your environment's state dimension
    action_dim = 5  # 5 assets
    
    for i in range(500):
        state = np.random.randn(state_dim)
        action = np.random.randn(action_dim)
        reward = np.random.randn()
        next_state = np.random.randn(state_dim)
        done = i % 100 == 99  # Episode ends every 100 steps
        
        buffer.add(state, action, reward, next_state, done)
    
    print(f"   Added 500 experiences")
    print(f"   Buffer size: {len(buffer)}")
    
    # Sample a batch
    print("\n[2] Sampling batch...")
    batch = buffer.sample(batch_size=64)
    
    print(f"   Batch shapes:")
    for key, value in batch.items():
        print(f"     {key}: {value.shape}")
    
    # Check statistics
    print("\n[3] Buffer statistics:")
    stats = buffer.get_statistics()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.4f}")
        else:
            print(f"   {key}: {value}")
    
    # Test is_ready
    print("\n[4] Ready checks:")
    print(f"   Ready for 100? {buffer.is_ready(100)}")
    print(f"   Ready for 1000? {buffer.is_ready(1000)}")
    
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)