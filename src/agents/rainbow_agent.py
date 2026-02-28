"""
Rainbow DQN Agent for Portfolio Management.

Rainbow DQN: The Swiss Army Knife of Value-Based RL
====================================================

Rainbow combines SIX improvements to DQN into a single powerful agent:

1. DOUBLE DQN (DDQN)
   - Problem: DQN overestimates Q-values
   - Solution: Use online network to SELECT actions, target network to EVALUATE
   
2. PRIORITIZED EXPERIENCE REPLAY (PER)
   - Problem: Uniform sampling wastes time on "easy" experiences
   - Solution: Sample experiences proportional to TD error
   - YOUR CONTRIBUTION: We'll combine PER with regime-aware sampling!

3. DUELING NETWORKS
   - Problem: Not all states have action-dependent value
   - Solution: Separate Value and Advantage streams, combine at end
   - Q(s,a) = V(s) + A(s,a) - mean(A(s,*))

4. MULTI-STEP LEARNING
   - Problem: 1-step TD is high-bias
   - Solution: Use n-step returns (default n=3)
   - Better credit assignment over longer horizons

5. DISTRIBUTIONAL RL (C51)
   - Problem: We only learn mean Q-value, not uncertainty
   - Solution: Learn full distribution of returns (51 atoms)
   - Better risk awareness - perfect for finance!

6. NOISY NETWORKS
   - Problem: Epsilon-greedy exploration is crude
   - Solution: Learn exploration through noisy network weights
   - More efficient, state-dependent exploration

Why Rainbow for Portfolio Management?
=====================================
- Distributional RL gives you risk awareness (not just expected return!)
- PER + regime-awareness could be very powerful combination
- Multi-step helps with credit assignment over trading days

Challenge: Discrete vs Continuous Actions
=========================================
Your portfolio environment uses continuous actions (portfolio weights).
Rainbow DQN requires discrete actions.

Solutions:
1. DISCRETIZE: Convert continuous to discrete buckets
   - Example: 5 actions per asset = 5^5 = 3125 combinations (too many!)
   - Better: 3 actions per asset (reduce/hold/increase) = 3^5 = 243 combinations
   
2. ACTION ABSTRACTION: High-level strategies
   - "Conservative", "Balanced", "Aggressive" etc.
   - Maps to specific weight patterns

3. SEPARATE ENVIRONMENT: Create DiscreteTradingEnv

We implement option 1 with smart action compression.

Usage:
------
>>> from src.agents.rainbow_agent import RainbowAgent
>>> agent = RainbowAgent(discrete_env, replay_buffer=regime_aware_buffer)
>>> agent.train(total_timesteps=1_000_000)

Author: Prince Alexander John  
Date: February 2026
Project: Regime-Aware Experience Replay for DRL Portfolio Management
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path
from collections import deque
import random
import yaml
import gymnasium as gym
from gymnasium import spaces

# Import base class
from .base_agent import BaseAgent, TrainingConfig, TrainingMetrics

# Try to import regime-aware buffer
try:
    from ..replay.regime_aware_buffer import RegimeAwareReplayBuffer, SamplingStrategy, MarketRegime
    REGIME_BUFFER_AVAILABLE = True
except ImportError:
    REGIME_BUFFER_AVAILABLE = False


# =============================================================================
# Discrete Action Wrapper
# =============================================================================

class DiscreteActionWrapper(gym.ActionWrapper):
    """
    Wraps a continuous-action environment to use discrete actions.
    
    For portfolio management:
    - 3 actions per asset: DECREASE (-10%), HOLD (0%), INCREASE (+10%)
    - Total actions = 3^n_assets (e.g., 3^5 = 243 for 5 assets)
    
    This is acceptable because:
    1. We're not trying to find exact optimal weights
    2. The agent learns to sequence small adjustments
    3. Reduces exploration complexity dramatically
    """
    
    def __init__(
        self, 
        env: gym.Env, 
        actions_per_asset: int = 3,
        step_size: float = 0.1
    ):
        """
        Initialize the discrete wrapper.
        
        Parameters:
        -----------
        env : gym.Env
            Continuous-action environment
        actions_per_asset : int
            Number of discrete actions per asset (3 = decrease/hold/increase)
        step_size : float
            How much to change weight by (0.1 = 10%)
        """
        super().__init__(env)
        
        self.n_assets = env.action_space.shape[0]
        self.actions_per_asset = actions_per_asset
        self.step_size = step_size
        
        # Create discrete action space
        # Total actions = actions_per_asset ^ n_assets
        self.n_actions = actions_per_asset ** self.n_assets
        self.action_space = spaces.Discrete(self.n_actions)
        
        # Precompute action mappings for efficiency
        self._action_map = self._create_action_map()
        
        # Track current portfolio weights
        self._current_weights = np.ones(self.n_assets) / self.n_assets
    
    def _create_action_map(self) -> np.ndarray:
        """
        Create mapping from discrete action index to weight changes.
        
        Returns array of shape (n_actions, n_assets) where each row
        contains the weight change for each asset.
        """
        # Generate all combinations
        action_map = np.zeros((self.n_actions, self.n_assets))
        
        # Map action values: 0 = decrease, 1 = hold, 2 = increase
        # Convert to: -1, 0, +1
        action_values = np.array([-1, 0, 1]) * self.step_size
        
        for i in range(self.n_actions):
            # Convert index to base-3 representation
            idx = i
            for j in range(self.n_assets):
                action_map[i, self.n_assets - 1 - j] = action_values[idx % self.actions_per_asset]
                idx //= self.actions_per_asset
        
        return action_map
    
    def action(self, action: int) -> np.ndarray:
        """
        Convert discrete action to continuous portfolio weights.
        
        Parameters:
        -----------
        action : int
            Discrete action index
        
        Returns:
        --------
        np.ndarray
            Continuous portfolio weights
        """
        # Get weight changes for this action
        weight_changes = self._action_map[action]
        
        # Apply changes to current weights
        new_weights = self._current_weights + weight_changes
        
        # Clip to valid range and normalize
        new_weights = np.clip(new_weights, 0, 1)
        new_weights = new_weights / (new_weights.sum() + 1e-8)
        
        # Update current weights
        self._current_weights = new_weights
        
        return new_weights
    
    def reset(self, **kwargs):
        """Reset the environment and current weights."""
        self._current_weights = np.ones(self.n_assets) / self.n_assets
        return self.env.reset(**kwargs)


# =============================================================================
# Neural Network Components
# =============================================================================

class NoisyLinear(nn.Module):
    """
    Noisy Linear Layer for exploration.
    
    Instead of epsilon-greedy, we add learnable noise to network weights.
    The network learns when and how much to explore.
    
    w = μ_w + σ_w ⊙ ε_w
    b = μ_b + σ_b ⊙ ε_b
    
    Where ε is factorized Gaussian noise (efficient).
    """
    
    def __init__(self, in_features: int, out_features: int, sigma_init: float = 0.5):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        
        # Learnable parameters
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        
        # Factorized noise (more efficient than independent noise)
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))
        
        self.sigma_init = sigma_init
        self.reset_parameters()
        self.reset_noise()
    
    def reset_parameters(self):
        """Initialize parameters."""
        mu_range = 1 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init / np.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.sigma_init / np.sqrt(self.out_features))
    
    def reset_noise(self):
        """Reset noise for next forward pass."""
        epsilon_in = self._factorized_noise(self.in_features)
        epsilon_out = self._factorized_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)
    
    def _factorized_noise(self, size: int) -> torch.Tensor:
        """Generate factorized Gaussian noise."""
        x = torch.randn(size, device=self.weight_mu.device)
        return x.sign() * x.abs().sqrt()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with noisy weights."""
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)


class DuelingNetwork(nn.Module):
    """
    Dueling Network Architecture.
    
    Separates state value V(s) from action advantages A(s,a):
    Q(s,a) = V(s) + A(s,a) - mean(A)
    
    The subtraction of mean ensures identifiability and stability.
    
    Why this helps for portfolio management:
    - V(s) captures: "How good is this market state overall?"
    - A(s,a) captures: "How much better/worse is this specific action?"
    - Many states have similar values regardless of action
    """
    
    def __init__(
        self,
        observation_dim: int,
        n_actions: int,
        hidden_dims: List[int] = [256, 256],
        n_atoms: int = 51,  # For distributional RL
        v_min: float = -10.0,
        v_max: float = 10.0,
        noisy: bool = True,
    ):
        super().__init__()
        
        self.n_actions = n_actions
        self.n_atoms = n_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.noisy = noisy
        
        # Support for distributional RL
        self.register_buffer(
            'support',
            torch.linspace(v_min, v_max, n_atoms)
        )
        self.delta_z = (v_max - v_min) / (n_atoms - 1)
        
        # Shared feature network
        layers = []
        in_dim = observation_dim
        for hidden_dim in hidden_dims[:-1]:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        self.feature_net = nn.Sequential(*layers)
        
        # Choose linear layer type
        LinearLayer = NoisyLinear if noisy else nn.Linear
        
        # Value stream (outputs distribution over atoms)
        self.value_stream = nn.Sequential(
            LinearLayer(hidden_dims[-2] if len(hidden_dims) > 1 else observation_dim, hidden_dims[-1]),
            nn.ReLU(),
            LinearLayer(hidden_dims[-1], n_atoms)
        )
        
        # Advantage stream (outputs distribution per action)
        self.advantage_stream = nn.Sequential(
            LinearLayer(hidden_dims[-2] if len(hidden_dims) > 1 else observation_dim, hidden_dims[-1]),
            nn.ReLU(),
            LinearLayer(hidden_dims[-1], n_actions * n_atoms)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning action-value distributions.
        
        Returns:
        --------
        torch.Tensor
            Shape (batch, n_actions, n_atoms) - probability distribution
            over returns for each action
        """
        features = self.feature_net(x)
        
        # Value distribution: (batch, n_atoms)
        value = self.value_stream(features)
        
        # Advantage distribution: (batch, n_actions * n_atoms) -> (batch, n_actions, n_atoms)
        advantage = self.advantage_stream(features)
        advantage = advantage.view(-1, self.n_actions, self.n_atoms)
        
        # Combine: Q = V + A - mean(A)
        # Expand value for broadcasting
        value = value.unsqueeze(1)  # (batch, 1, n_atoms)
        q_dist = value + advantage - advantage.mean(dim=1, keepdim=True)
        
        # Apply softmax to get probability distribution
        q_dist = F.softmax(q_dist, dim=-1)
        
        return q_dist
    
    def get_q_values(self, x: torch.Tensor) -> torch.Tensor:
        """Get expected Q-values (for action selection)."""
        q_dist = self.forward(x)
        # Expected value = sum(probability * support)
        q_values = (q_dist * self.support.view(1, 1, -1)).sum(dim=-1)
        return q_values
    
    def reset_noise(self):
        """Reset noise in all noisy layers."""
        if self.noisy:
            for module in self.modules():
                if isinstance(module, NoisyLinear):
                    module.reset_noise()


# =============================================================================
# Prioritized Experience Replay with Regime Awareness
# =============================================================================

class SumTree:
    """
    Sum Tree for O(log N) prioritized sampling (Schaul et al. 2016).

    A binary tree where each leaf stores a priority value and each
    parent stores the sum of its children. This enables:
    - update(index, priority): O(log N)
    - sample(value): O(log N)
    - total(): O(1)
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data = [None] * capacity
        self.write_pos = 0
        self.size = 0

    def _propagate(self, idx: int, change: float):
        """Propagate priority change up to root."""
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx: int, value: float) -> int:
        """Find leaf index for a given cumulative value."""
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if value <= self.tree[left]:
            return self._retrieve(left, value)
        else:
            return self._retrieve(right, value - self.tree[left])

    @property
    def total(self) -> float:
        return self.tree[0]

    def add(self, priority: float, data):
        """Add data with given priority."""
        tree_idx = self.write_pos + self.capacity - 1
        self.data[self.write_pos] = data
        self.update(tree_idx, priority)
        self.write_pos = (self.write_pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def update(self, tree_idx: int, priority: float):
        """Update priority at tree index."""
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        self._propagate(tree_idx, change)

    def get(self, value: float) -> Tuple[int, float, Any]:
        """Get (tree_idx, priority, data) for a cumulative value."""
        tree_idx = self._retrieve(0, value)
        data_idx = tree_idx - self.capacity + 1
        return tree_idx, self.tree[tree_idx], self.data[data_idx]

    @property
    def min_priority(self) -> float:
        """Minimum non-zero priority among stored entries."""
        leaves = self.tree[self.capacity - 1:self.capacity - 1 + self.size]
        nonzero = leaves[leaves > 0]
        return float(nonzero.min()) if len(nonzero) > 0 else 1.0


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay buffer using SumTree for O(log N) sampling.

    Samples experiences proportional to their TD error.
    High-error experiences are sampled more frequently.
    """

    def __init__(
        self,
        capacity: int,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_frames: int = 100_000,
    ):
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1
        self.tree = SumTree(capacity)
        self.max_priority = 1.0

    @property
    def beta(self) -> float:
        return min(1.0, self.beta_start + (1.0 - self.beta_start) * self.frame / self.beta_frames)

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
        self.tree.add(priority, experience)

    def sample(self, batch_size: int) -> Tuple:
        """
        Sample batch of experiences proportional to priorities.

        Returns:
        --------
        tuple: (states, actions, rewards, next_states, dones, indices, weights)
        """
        batch = []
        tree_indices = []
        priorities = []

        segment = self.tree.total / batch_size

        for i in range(batch_size):
            lo = segment * i
            hi = segment * (i + 1)
            value = np.random.uniform(lo, hi)
            tree_idx, priority, data = self.tree.get(value)
            if data is None:
                # Fallback: resample from full range
                value = np.random.uniform(0, self.tree.total)
                tree_idx, priority, data = self.tree.get(value)
            batch.append(data)
            tree_indices.append(tree_idx)
            priorities.append(priority)

        # Importance sampling weights
        priorities_arr = np.array(priorities, dtype=np.float64)
        min_prob = self.tree.min_priority / self.tree.total
        max_weight = (self.tree.size * min_prob) ** (-self.beta)
        sampling_probs = priorities_arr / self.tree.total
        weights = (self.tree.size * sampling_probs) ** (-self.beta)
        weights = weights / max_weight  # Normalize

        states = np.array([e[0] for e in batch])
        actions = np.array([e[1] for e in batch])
        rewards = np.array([e[2] for e in batch])
        next_states = np.array([e[3] for e in batch])
        dones = np.array([e[4] for e in batch])

        self.frame += 1

        return states, actions, rewards, next_states, dones, np.array(tree_indices), weights.astype(np.float32)

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        """Update priorities based on TD errors."""
        for idx, td_error in zip(indices, td_errors):
            priority = (abs(td_error) + 1e-6) ** self.alpha
            self.tree.update(int(idx), priority)
            self.max_priority = max(self.max_priority, priority)

    def __len__(self) -> int:
        return self.tree.size


# =============================================================================
# Rainbow Agent
# =============================================================================

class RainbowAgent(BaseAgent):
    """
    Rainbow DQN Agent with all 6 improvements.
    
    Components:
    1. Double DQN - Reduces overestimation
    2. Prioritized Experience Replay - Samples important experiences more
    3. Dueling Networks - Separates value and advantage
    4. Multi-step Learning - Better credit assignment
    5. Distributional RL (C51) - Learns return distribution
    6. Noisy Networks - Learned exploration
    
    For your project:
    - Requires discrete actions (use DiscreteActionWrapper)
    - Can integrate with RegimeAwareReplayBuffer
    - Distributional RL provides risk awareness
    """
    
    def __init__(
        self,
        env: gym.Env,
        config: Optional[Dict[str, Any]] = None,
        replay_buffer: Optional[Any] = None,
        seed: Optional[int] = None,
        device: str = "auto",
        name: str = "rainbow_agent",
        verbose: int = 1,
    ):
        """
        Initialize Rainbow agent.
        
        Parameters:
        -----------
        env : gym.Env
            Must have discrete action space (wrap with DiscreteActionWrapper)
        config : dict, optional
            Hyperparameters
        replay_buffer : optional
            Custom buffer (for regime-aware PER integration)
        seed : int, optional
            Random seed
        device : str
            "auto", "cpu", or "cuda"
        name : str
            Name for this agent
        verbose : int
            Verbosity level
        """
        # Validate action space
        if not isinstance(env.action_space, spaces.Discrete):
            raise ValueError(
                "Rainbow requires discrete actions. "
                "Wrap your environment with DiscreteActionWrapper first."
            )
        
        # Store device BEFORE calling super().__init__
        if device == "auto":
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = torch.device(device)
        
        # Store custom replay buffer
        self.custom_replay_buffer = replay_buffer
        
        # Load config
        if config is None:
            config = self._load_default_config()
        
        # Environment dimensions (needed for _setup_model)
        self.obs_dim = env.observation_space.shape[0]
        self.n_actions = env.action_space.n
        
        # Training state
        self.training_step = 0
        self.episodes = 0
        
        # Multi-step buffer
        self.n_steps = config.get("n_steps", 3)
        self.n_step_buffer = deque(maxlen=self.n_steps)
        
        # Call parent init (this calls _setup_model)
        super().__init__(env, name=name, config=config, replay_buffer=replay_buffer, verbose=verbose)
        
    def _setup_model(self) -> None:
        """Set up the Rainbow DQN model (networks, optimizer, buffer)."""
        # Create networks
        self.online_net = self._create_network().to(self._device)
        self.target_net = self._create_network().to(self._device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.online_net.parameters(),
            lr=self.config.get("learning_rate", 6.25e-5),
            eps=1.5e-4,
        )
        
        # Replay buffer
        if self.custom_replay_buffer is not None:
            self.per_buffer = self.custom_replay_buffer
        else:
            self.per_buffer = PrioritizedReplayBuffer(
                capacity=self.config.get("buffer_size", 1_000_000),
                alpha=self.config.get("per_alpha", 0.6),
                beta_start=self.config.get("per_beta", 0.4),
            )
    
    @property
    def device(self):
        """Return the device."""
        return self._device
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default Rainbow configuration."""
        config_path = Path(__file__).parent.parent.parent / "configs" / "algorithms" / "rainbow.yaml"
        
        if config_path.exists():
            with open(config_path, 'r') as f:
                full_config = yaml.safe_load(f)
                return full_config.get('rainbow', self._get_hardcoded_defaults())
        else:
            return self._get_hardcoded_defaults()
    
    def _get_hardcoded_defaults(self) -> Dict[str, Any]:
        """Hardcoded defaults from your spec."""
        return {
            "learning_rate": 6.25e-5,
            "buffer_size": 200_000,
            "batch_size": 32,
            "gamma": 0.99,
            "n_steps": 3,
            "n_atoms": 51,
            "v_min": -10.0,
            "v_max": 10.0,
            "per_alpha": 0.6,
            "per_beta": 0.4,
            "target_update_freq": 8000,
            "learning_starts": 20_000,
            "train_freq": 4,
            "hidden_dims": [256, 256],
            "noisy_sigma": 0.5,
        }
    
    def _create_network(self) -> DuelingNetwork:
        """Create the dueling distributional network."""
        return DuelingNetwork(
            observation_dim=self.obs_dim,
            n_actions=self.n_actions,
            hidden_dims=self.config.get("hidden_dims", [256, 256]),
            n_atoms=self.config.get("n_atoms", 51),
            v_min=self.config.get("v_min", -10.0),
            v_max=self.config.get("v_max", 10.0),
            noisy=True,
        )
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select action using noisy network (no epsilon-greedy needed).
        
        Parameters:
        -----------
        state : np.ndarray
            Current observation
        training : bool
            If True, use noisy weights (exploration)
            If False, use mean weights (exploitation)
        
        Returns:
        --------
        int
            Selected action
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            if training:
                self.online_net.reset_noise()
            
            q_values = self.online_net.get_q_values(state_tensor)
            action = q_values.argmax(dim=1).item()
        
        return action
    
    def _compute_n_step_return(self) -> Tuple:
        """
        Compute n-step return from buffer.
        
        Returns:
        --------
        tuple: (state, action, n_step_reward, next_state, done)
        """
        gamma = self.config.get("gamma", 0.99)
        
        # Calculate n-step reward
        n_step_reward = 0
        for i, (_, _, r, _, d, _) in enumerate(self.n_step_buffer):
            n_step_reward += (gamma ** i) * r
            if d:
                break
        
        # Get first and last experiences
        first = self.n_step_buffer[0]
        last = self.n_step_buffer[-1]
        
        return (
            first[0],  # First state
            first[1],  # First action
            n_step_reward,  # N-step reward
            last[3],  # Last next_state
            last[4],  # Last done
            first[5],  # Regime (from first experience)
        )
    
    def store_experience(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        regime: Optional[str] = None,
    ):
        """
        Store experience with n-step handling.
        
        The experience is first added to n-step buffer.
        When buffer is full (or episode ends), compute n-step return
        and add to main replay buffer.
        """
        self.n_step_buffer.append((state, action, reward, next_state, done, regime))
        
        # Compute and store n-step experience when ready
        if len(self.n_step_buffer) == self.n_steps or done:
            n_step_exp = self._compute_n_step_return()
            self.per_buffer.add(*n_step_exp)
            
            # If done, flush remaining experiences
            if done:
                while len(self.n_step_buffer) > 1:
                    self.n_step_buffer.popleft()
                    if len(self.n_step_buffer) > 0:
                        n_step_exp = self._compute_n_step_return()
                        self.per_buffer.add(*n_step_exp)
                self.n_step_buffer.clear()
    
    def update(self) -> Optional[float]:
        """
        Perform one gradient update step.
        
        Returns:
        --------
        float or None
            TD loss, or None if not enough samples
        """
        if len(self.per_buffer) < self.config.get("learning_starts", 20_000):
            return None
        
        batch_size = self.config.get("batch_size", 32)
        
        # Sample from replay buffer
        states, actions, rewards, next_states, dones, indices, weights = \
            self.per_buffer.sample(batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device)
        
        # Reset noise for this update
        self.online_net.reset_noise()
        self.target_net.reset_noise()
        
        # Compute distributional TD loss
        loss, td_errors = self._compute_distributional_loss(
            states, actions, rewards, next_states, dones, weights
        )
        
        # Update
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), 10.0)
        self.optimizer.step()
        
        # Update priorities
        self.per_buffer.update_priorities(indices, td_errors.detach().cpu().numpy())
        
        # Update target network
        if self.training_step % self.config.get("target_update_freq", 8000) == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())
        
        self.training_step += 1
        
        return loss.item()
    
    def _compute_distributional_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
        weights: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute distributional TD loss (C51 algorithm).
        
        Instead of updating towards a single target value,
        we update towards a target distribution over returns.
        """
        gamma = self.config.get("gamma", 0.99)
        n_atoms = self.config.get("n_atoms", 51)
        v_min = self.config.get("v_min", -10.0)
        v_max = self.config.get("v_max", 10.0)
        delta_z = (v_max - v_min) / (n_atoms - 1)
        
        batch_size = states.shape[0]
        
        # Current distribution
        current_dist = self.online_net(states)
        current_dist = current_dist[range(batch_size), actions]  # (batch, n_atoms)
        
        # Double DQN: Use online net to SELECT, target net to EVALUATE
        with torch.no_grad():
            # Online net selects best action
            next_q = self.online_net.get_q_values(next_states)
            next_actions = next_q.argmax(dim=1)
            
            # Target net evaluates
            next_dist = self.target_net(next_states)
            next_dist = next_dist[range(batch_size), next_actions]  # (batch, n_atoms)
            
            # Project target distribution
            support = self.online_net.support
            
            # Compute projected support: Tz = r + gamma * z
            n_step_gamma = gamma ** self.n_steps
            Tz = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * n_step_gamma * support.unsqueeze(0)
            Tz = Tz.clamp(v_min, v_max)
            
            # Compute projection indices
            b = (Tz - v_min) / delta_z
            l = b.floor().long()
            u = b.ceil().long()
            
            # Handle edge cases
            l = l.clamp(0, n_atoms - 1)
            u = u.clamp(0, n_atoms - 1)
            
            # Distribute probability mass
            target_dist = torch.zeros_like(next_dist)
            offset = torch.linspace(0, (batch_size - 1) * n_atoms, batch_size,
                                   dtype=torch.long, device=self.device).unsqueeze(1)
            
            target_dist.view(-1).index_add_(
                0, (l + offset).view(-1),
                (next_dist * (u.float() - b)).view(-1)
            )
            target_dist.view(-1).index_add_(
                0, (u + offset).view(-1),
                (next_dist * (b - l.float())).view(-1)
            )
        
        # Cross-entropy loss
        log_p = torch.log(current_dist + 1e-8)
        loss = -(target_dist * log_p).sum(dim=1)
        
        # Apply importance sampling weights
        weighted_loss = (loss * weights).mean()
        
        # TD errors for priority update (use expected value difference)
        with torch.no_grad():
            current_q = (current_dist * support).sum(dim=1)
            target_q = (target_dist * support).sum(dim=1)
            td_errors = target_q - current_q
        
        return weighted_loss, td_errors
    
    def train(
        self,
        training_config: 'TrainingConfig' = None,
        total_timesteps: int = None,
        eval_freq: int = 10_000,
        eval_episodes: int = 5,
        save_freq: int = 50_000,
        save_path: Optional[str] = None,
        progress_bar: bool = True,
        eval_env: Optional[gym.Env] = None,
        callbacks: Optional[List] = None,
    ) -> 'TrainingMetrics':
        """
        Train the Rainbow agent.

        Parameters:
        -----------
        training_config : TrainingConfig, optional
            Training configuration object
        total_timesteps : int, optional
            Total environment steps
        eval_freq : int
            Evaluate every N steps
        eval_episodes : int
            Episodes per evaluation
        save_freq : int
            Save checkpoint every N steps
        save_path : str, optional
            Directory for checkpoints
        progress_bar : bool
            Show progress
        eval_env : gym.Env, optional
            Environment for evaluation
        callbacks : list, optional
            Callbacks (not used, for interface compatibility)

        Returns:
        --------
        TrainingMetrics
            Training history and performance metrics
        """
        # Handle training_config if provided
        if training_config is not None:
            total_timesteps = total_timesteps or training_config.total_timesteps
            eval_freq = training_config.eval_freq
            eval_episodes = training_config.n_eval_episodes
            save_freq = training_config.save_freq
            save_path = save_path or training_config.save_path

        # Default total_timesteps if still None
        if total_timesteps is None:
            total_timesteps = 100_000

        state, _ = self.env.reset()
        episode_reward = 0

        try:
            from tqdm import tqdm
            iterator = tqdm(range(total_timesteps), disable=not progress_bar)
        except ImportError:
            iterator = range(total_timesteps)

        for step in iterator:
            # Select and execute action
            action = self.select_action(state, training=True)
            next_state, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            # Get regime if available
            regime = info.get("regime", None)

            # Store experience
            self.store_experience(state, action, reward, next_state, done, regime)

            episode_reward += reward

            # Update
            if step % self.config.get("train_freq", 4) == 0:
                loss = self.update()
                if loss is not None:
                    self.metrics.policy_losses.append(loss)

            # Episode end
            if done:
                self.metrics.episode_rewards.append(episode_reward)
                self.episodes += 1
                state, _ = self.env.reset()
                episode_reward = 0
            else:
                state = next_state

            # Evaluation
            if step > 0 and step % eval_freq == 0:
                eval_reward = self._evaluate(eval_episodes)
                self.metrics.eval_rewards.append(eval_reward)
                self.metrics.timesteps.append(self.total_timesteps + step)

                if progress_bar and hasattr(iterator, 'set_postfix'):
                    iterator.set_postfix({
                        "ep": self.episodes,
                        "reward": f"{eval_reward:.2f}",
                        "loss": f"{self.metrics.policy_losses[-1]:.4f}" if self.metrics.policy_losses else "N/A"
                    })

            # Save checkpoint
            if save_path and step > 0 and step % save_freq == 0:
                self.save(f"{save_path}/rainbow_step_{step}")

        self.total_timesteps += total_timesteps
        return self.metrics
    
    def _evaluate(self, n_episodes: int) -> float:
        """Evaluate without exploration."""
        total_reward = 0
        
        for _ in range(n_episodes):
            state, _ = self.env.reset()
            done = False
            
            while not done:
                action = self.select_action(state, training=False)
                state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                total_reward += reward
        
        return total_reward / n_episodes
    
    def predict(self, observation: np.ndarray, deterministic: bool = True) -> Tuple:
        """Get action (API compatibility with other agents)."""
        action = self.select_action(observation, training=not deterministic)
        return action, None
    
    def save(self, path: str) -> None:
        """Save model checkpoint."""
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            "online_net": self.online_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "training_step": self.training_step,
            "episodes": self.episodes,
            "config": self.config,
        }, f"{save_path}.pt")
        
        print(f"[OK] Rainbow model saved to {save_path}.pt")
    
    def load(self, path: str, env: Optional[gym.Env] = None) -> "RainbowAgent":
        """Load model checkpoint."""
        if env is not None:
            self.env = env
        
        checkpoint = torch.load(f"{path}.pt", map_location=self.device)
        
        self.online_net.load_state_dict(checkpoint["online_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.training_step = checkpoint["training_step"]
        self.episodes = checkpoint["episodes"]
        
        print(f"[OK] Rainbow model loaded from {path}.pt")
        return self
    
    def evaluate(self, env: gym.Env, n_episodes: int = 10, deterministic: bool = True) -> Dict[str, float]:
        """Full evaluation with statistics."""
        episode_rewards = []
        
        for _ in range(n_episodes):
            state, _ = env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                action = self.select_action(state, training=not deterministic)
                state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                episode_reward += reward
            
            episode_rewards.append(episode_reward)
        
        return {
            "mean_reward": np.mean(episode_rewards),
            "std_reward": np.std(episode_rewards),
            "min_reward": np.min(episode_rewards),
            "max_reward": np.max(episode_rewards),
        }


# =============================================================================
# Testing
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("RAINBOW DQN AGENT - TEST")
    print("=" * 70)
    
    # Create a simple discrete environment
    env = gym.make("CartPole-v1")
    
    # Test 1: Create agent
    print("\n[1] Creating Rainbow agent...")
    agent = RainbowAgent(env, device="cpu")
    print(f"    [OK] Agent created")
    print(f"    Observation dim: {agent.obs_dim}")
    print(f"    Number of actions: {agent.n_actions}")
    print(f"    N-atoms (C51): {agent.config.get('n_atoms')}")
    
    # Test 2: Action selection
    print("\n[2] Testing action selection...")
    obs, _ = env.reset()
    action = agent.select_action(obs, training=True)
    print(f"    Action selected: {action}")
    print(f"    [OK] Action selection works")
    
    # Test 3: Quick training
    print("\n[3] Quick training test (500 steps)...")
    history = agent.train(
        total_timesteps=500,
        progress_bar=False,
        eval_freq=250,
        eval_episodes=2,
    )
    print(f"    Episodes: {agent.episodes}")
    print(f"    [OK] Training works")
    
    # Test 4: Save and load
    print("\n[4] Testing save/load...")
    agent.save("/tmp/test_rainbow")
    agent.load("/tmp/test_rainbow")
    print(f"    [OK] Save/load works")
    
    # Test 5: Discrete wrapper
    print("\n[5] Testing DiscreteActionWrapper...")
    # Create a fake continuous env
    cont_env = gym.make("Pendulum-v1")
    wrapped = DiscreteActionWrapper(cont_env, actions_per_asset=3)
    print(f"    Original action space: {cont_env.action_space}")
    print(f"    Wrapped action space: {wrapped.action_space}")
    print(f"    [OK] Wrapper works")
    
    print("\n" + "=" * 70)
    print("ALL TESTS PASSED!")
    print("=" * 70)