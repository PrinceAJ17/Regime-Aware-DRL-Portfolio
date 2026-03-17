"""
Base Agent Class
================

This module defines the abstract base class for all RL agents in the project.

WHY DO WE NEED A BASE CLASS?
============================

You're implementing 4 different algorithms: PPO, SAC, TD3, Rainbow DQN.
Without a base class, you'd have inconsistent interfaces:

    # Without base class (messy)
    ppo_agent.do_training(env, steps=1000)
    sac_agent.train_model(environment=env, num_steps=1000)
    td3_agent.learn(e=env, s=1000)

    # With base class (clean)
    for agent in [ppo_agent, sac_agent, td3_agent]:
        agent.train(env, total_timesteps=1000)

A base class ensures:
1. All agents have the same methods (train, predict, save, load)
2. Your experiment code can treat all agents the same way
3. Easier to add new algorithms later
4. Cleaner code and fewer bugs

WHAT THIS BASE CLASS PROVIDES:
==============================

1. Common interface for training and evaluation
2. Checkpointing (save/load models)
3. Logging infrastructure
4. Metrics tracking
5. Integration points for your regime-aware replay buffer

Author: Prince Alexander John
Date: February 2026
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional, Union, Tuple, List
from dataclasses import dataclass, field
from datetime import datetime
import json
import numpy as np
import pandas as pd
import gymnasium as gym


@dataclass
class TrainingConfig:
    """
    Configuration for training an agent.
    
    This dataclass holds all the settings needed for training.
    Using a dataclass makes it easy to:
    - Save/load configurations
    - Pass settings between functions
    - Document what settings are available
    """
    # Core training settings
    total_timesteps: int = 1_000_000
    eval_freq: int = 10_000  # Evaluate every N steps
    n_eval_episodes: int = 10  # Episodes per evaluation
    
    # Checkpointing
    save_freq: int = 50_000  # Save model every N steps
    save_path: str = "outputs/models"
    
    # Early stopping
    early_stopping: bool = True
    patience: int = 10  # Stop if no improvement for N evaluations
    min_improvement: float = 0.01  # Minimum improvement to count as progress
    
    # Logging
    log_interval: int = 1000  # Log metrics every N steps
    tensorboard_log: Optional[str] = "outputs/tensorboard"
    
    # Reproducibility
    seed: int = 42
    
    # Device
    device: str = "auto"  # "auto", "cpu", or "cuda"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for saving."""
        return {
            'total_timesteps': self.total_timesteps,
            'eval_freq': self.eval_freq,
            'n_eval_episodes': self.n_eval_episodes,
            'save_freq': self.save_freq,
            'save_path': self.save_path,
            'early_stopping': self.early_stopping,
            'patience': self.patience,
            'min_improvement': self.min_improvement,
            'log_interval': self.log_interval,
            'tensorboard_log': self.tensorboard_log,
            'seed': self.seed,
            'device': self.device
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'TrainingConfig':
        """Create config from dictionary."""
        return cls(**d)


@dataclass 
class TrainingMetrics:
    """
    Tracks training progress and metrics.
    
    This is what you'll use to:
    - Plot learning curves
    - Compare algorithms
    - Detect overfitting
    - Write your report
    """
    # Training progress
    timesteps: List[int] = field(default_factory=list)
    episodes: List[int] = field(default_factory=list)
    
    # Rewards
    episode_rewards: List[float] = field(default_factory=list)
    eval_rewards: List[float] = field(default_factory=list)
    
    # Portfolio-specific metrics
    portfolio_values: List[float] = field(default_factory=list)
    sharpe_ratios: List[float] = field(default_factory=list)
    max_drawdowns: List[float] = field(default_factory=list)
    
    # Loss values (for debugging)
    policy_losses: List[float] = field(default_factory=list)
    value_losses: List[float] = field(default_factory=list)
    entropy_losses: List[float] = field(default_factory=list)
    
    # Timing
    wall_time: List[float] = field(default_factory=list)
    
    # Regime-specific (YOUR NOVEL CONTRIBUTION!)
    regime_rewards: Dict[str, List[float]] = field(default_factory=lambda: {
        'bull': [], 'bear': [], 'sideways': []
    })
    
    def add_eval_result(
        self,
        timestep: int,
        mean_reward: float,
        sharpe: Optional[float] = None,
        max_dd: Optional[float] = None,
        regime_rewards: Optional[Dict[str, float]] = None
    ):
        """Add evaluation results."""
        self.timesteps.append(timestep)
        self.eval_rewards.append(mean_reward)
        
        if sharpe is not None:
            self.sharpe_ratios.append(sharpe)
        if max_dd is not None:
            self.max_drawdowns.append(max_dd)
        if regime_rewards is not None:
            for regime, reward in regime_rewards.items():
                self.regime_rewards[regime].append(reward)
    
    def get_best_eval_reward(self) -> Tuple[int, float]:
        """Get timestep and value of best evaluation."""
        if not self.eval_rewards:
            return 0, float('-inf')
        best_idx = np.argmax(self.eval_rewards)
        return self.timesteps[best_idx], self.eval_rewards[best_idx]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for saving."""
        return {
            'timesteps': self.timesteps,
            'episodes': self.episodes,
            'episode_rewards': self.episode_rewards,
            'eval_rewards': self.eval_rewards,
            'portfolio_values': self.portfolio_values,
            'sharpe_ratios': self.sharpe_ratios,
            'max_drawdowns': self.max_drawdowns,
            'policy_losses': self.policy_losses,
            'value_losses': self.value_losses,
            'entropy_losses': self.entropy_losses,
            'wall_time': self.wall_time,
            'regime_rewards': self.regime_rewards
        }
    
    def save(self, path: Union[str, Path]):
        """Save metrics to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'TrainingMetrics':
        """Load metrics from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        metrics = cls()
        for key, value in data.items():
            setattr(metrics, key, value)
        return metrics


class BaseAgent(ABC):
    """
    Abstract base class for all RL agents.
    
    All agents (PPO, SAC, TD3, Rainbow) inherit from this class.
    This ensures they all have the same interface.
    
    WHAT EACH AGENT MUST IMPLEMENT:
    ================================
    
    1. _setup_model() - Create the neural network and optimizer
    2. train() - Run the training loop
    3. predict() - Get action from observation
    4. save() / load() - Checkpoint the model
    
    WHAT THIS CLASS PROVIDES FOR FREE:
    ===================================
    
    1. Configuration handling
    2. Metrics tracking
    3. Logging setup
    4. Evaluation loop
    5. Checkpointing infrastructure
    
    Example Usage:
    --------------
    >>> class PPOAgent(BaseAgent):
    ...     def _setup_model(self):
    ...         self.model = PPO("MlpPolicy", self.env)
    ...     
    ...     def train(self, config):
    ...         self.model.learn(total_timesteps=config.total_timesteps)
    ...     
    ...     def predict(self, observation):
    ...         return self.model.predict(observation)
    
    >>> agent = PPOAgent(env, name="ppo_experiment_1")
    >>> agent.train(TrainingConfig(total_timesteps=100000))
    >>> agent.save("models/ppo_best")
    """
    
    def __init__(
        self,
        env: gym.Env,
        name: str = "agent",
        config: Optional[Dict[str, Any]] = None,
        regime_detector: Optional[Any] = None,  # Your regime detector
        replay_buffer: Optional[Any] = None,     # Your regime-aware buffer
        verbose: int = 1
    ):
        """
        Initialize the base agent.
        
        Parameters:
        -----------
        env : gym.Env
            The trading environment
            
        name : str
            Name for this agent (used in logging and saving)
            
        config : dict, optional
            Algorithm-specific configuration (learning rate, etc.)
            
        regime_detector : RegimeDetector, optional
            Your regime detector for regime-aware training
            
        replay_buffer : ReplayBuffer, optional
            Custom replay buffer (standard or regime-aware)
            
        verbose : int
            Verbosity level (0=silent, 1=info, 2=debug)
        """
        self.env = env
        self.name = name
        self.config = config or {}
        self.regime_detector = regime_detector
        self.replay_buffer = replay_buffer
        self.verbose = verbose
        
        # Metrics tracking
        self.metrics = TrainingMetrics()
        
        # Training state
        self.total_timesteps = 0
        self.total_episodes = 0
        self.best_eval_reward = float('-inf')
        self.no_improvement_count = 0
        
        # Timestamps
        self.created_at = datetime.now()
        self.last_save_time = None
        
        # Model (to be set by subclass)
        self.model = None
        
        # Setup the model
        self._setup_model()
    
    @abstractmethod
    def _setup_model(self) -> None:
        """
        Set up the model (neural network, optimizer, etc.).
        
        This is called in __init__ and must be implemented by each agent.
        
        Example:
        --------
        def _setup_model(self):
            self.model = PPO(
                "MlpPolicy",
                self.env,
                learning_rate=self.config.get('learning_rate', 3e-4),
                ...
            )
        """
        pass
    
    @abstractmethod
    def train(
        self,
        training_config: TrainingConfig,
        eval_env: Optional[gym.Env] = None,
        callbacks: Optional[List] = None
    ) -> TrainingMetrics:
        """
        Train the agent.
        
        Parameters:
        -----------
        training_config : TrainingConfig
            Training settings (timesteps, eval frequency, etc.)
            
        eval_env : gym.Env, optional
            Separate environment for evaluation (recommended)
            
        callbacks : list, optional
            Custom callbacks for logging, early stopping, etc.
            
        Returns:
        --------
        TrainingMetrics
            Training history and performance metrics
        """
        pass
    
    @abstractmethod
    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = True
    ) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
        """
        Get action from observation.
        
        Parameters:
        -----------
        observation : np.ndarray
            Current state/observation from environment
            
        deterministic : bool
            If True, use deterministic policy (for evaluation)
            If False, sample from policy (for training)
            
        Returns:
        --------
        action : np.ndarray
            The action to take
            
        info : dict, optional
            Additional info (action probabilities, value estimate, etc.)
        """
        pass
    
    @abstractmethod
    def save(self, path: Union[str, Path]) -> None:
        """
        Save the model to disk.
        
        Should save:
        - Model weights
        - Optimizer state
        - Configuration
        - Training metrics
        
        Parameters:
        -----------
        path : str or Path
            Directory to save to
        """
        pass
    
    @abstractmethod
    def load(self, path: Union[str, Path]) -> None:
        """
        Load model from disk.
        
        Parameters:
        -----------
        path : str or Path
            Directory to load from
        """
        pass
    
    # =========================================================================
    # HELPER METHODS (provided for free)
    # =========================================================================
    
    def evaluate(
        self,
        eval_env: gym.Env,
        n_episodes: int = 10,
        deterministic: bool = True
    ) -> Dict[str, float]:
        """
        Evaluate the agent on an environment.
        
        Parameters:
        -----------
        eval_env : gym.Env
            Environment to evaluate on
            
        n_episodes : int
            Number of episodes to run
            
        deterministic : bool
            Use deterministic policy
            
        Returns:
        --------
        dict
            Evaluation metrics (mean_reward, std_reward, etc.)
        """
        episode_rewards = []
        episode_lengths = []
        
        for _ in range(n_episodes):
            obs, info = eval_env.reset()
            done = False
            episode_reward = 0
            episode_length = 0
            
            while not done:
                action, _ = self.predict(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, info = eval_env.step(action)
                done = terminated or truncated
                episode_reward += reward
                episode_length += 1
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
        
        results = {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'min_reward': np.min(episode_rewards),
            'max_reward': np.max(episode_rewards),
            'mean_length': np.mean(episode_lengths),
            'n_episodes': n_episodes
        }
        
        # Try to get portfolio-specific metrics from env
        if hasattr(eval_env, 'get_metrics'):
            portfolio_metrics = eval_env.get_metrics()
            results.update(portfolio_metrics)
        
        return results
    
    def evaluate_by_regime(
        self,
        eval_env: gym.Env,
        n_episodes: int = 10
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate agent performance broken down by market regime.
        
        THIS IS KEY FOR YOUR RESEARCH!
        
        Compares how well the agent performs in:
        - Bull markets
        - Bear markets  
        - Sideways markets
        
        Parameters:
        -----------
        eval_env : gym.Env
            Environment to evaluate on
            
        n_episodes : int
            Number of episodes to run
            
        Returns:
        --------
        dict
            Metrics per regime
        """
        if self.regime_detector is None:
            raise ValueError(
                "No regime detector set. Cannot evaluate by regime. "
                "Pass regime_detector to agent constructor."
            )

        # Pre-compute regime cache from price data
        _regime_cache = None
        env = eval_env
        if hasattr(env, 'envs'):
            env = env.envs[0]
        if hasattr(env, 'unwrapped'):
            env = env.unwrapped
        for attr in ('price_data', '_price_data', 'data', 'df'):
            if hasattr(env, attr):
                try:
                    _regime_cache = self.regime_detector.detect(getattr(env, attr))
                except Exception:
                    pass
                break

        regime_rewards = {'bull': [], 'bear': [], 'sideways': []}

        for _ in range(n_episodes):
            obs, info = eval_env.reset()
            done = False

            # Track rewards per regime within episode
            episode_regime_rewards = {'bull': 0, 'bear': 0, 'sideways': 0}
            episode_regime_steps = {'bull': 0, 'bear': 0, 'sideways': 0}

            while not done:
                # Get current regime from cached series
                regime = 'sideways'  # Default
                if _regime_cache is not None:
                    # Get current step from env
                    inner_env = eval_env
                    if hasattr(inner_env, 'envs'):
                        inner_env = inner_env.envs[0]
                    if hasattr(inner_env, 'unwrapped'):
                        inner_env = inner_env.unwrapped
                    current_step = getattr(inner_env, 'current_step', None)
                    if current_step is not None and 0 <= current_step < len(_regime_cache):
                        regime_val = _regime_cache.iloc[current_step]
                        regime = regime_val.value if hasattr(regime_val, 'value') else str(regime_val)

                action, _ = self.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = eval_env.step(action)
                done = terminated or truncated

                # Track reward by regime
                episode_regime_rewards[regime] += reward
                episode_regime_steps[regime] += 1

            # Calculate mean reward per regime for this episode
            for regime in regime_rewards:
                if episode_regime_steps[regime] > 0:
                    mean_reward = episode_regime_rewards[regime] / episode_regime_steps[regime]
                    regime_rewards[regime].append(mean_reward)
        
        # Aggregate across episodes
        results = {}
        for regime, rewards in regime_rewards.items():
            if rewards:
                results[regime] = {
                    'mean_reward': np.mean(rewards),
                    'std_reward': np.std(rewards),
                    'n_episodes': len(rewards)
                }
            else:
                results[regime] = {
                    'mean_reward': None,
                    'std_reward': None,
                    'n_episodes': 0
                }
        
        return results
    
    def get_info(self) -> Dict[str, Any]:
        """Get information about this agent."""
        return {
            'name': self.name,
            'algorithm': self.__class__.__name__,
            'total_timesteps': self.total_timesteps,
            'total_episodes': self.total_episodes,
            'best_eval_reward': self.best_eval_reward,
            'created_at': self.created_at.isoformat(),
            'config': self.config,
            'has_regime_detector': self.regime_detector is not None,
            'has_custom_replay': self.replay_buffer is not None
        }
    
    def _log(self, message: str, level: int = 1):
        """Log a message if verbosity is high enough."""
        if self.verbose >= level:
            print(f"[{self.name}] {message}")


# Import BaseCallback here to avoid circular imports
try:
    from stable_baselines3.common.callbacks import BaseCallback
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    BaseCallback = object  # Fallback for type hints


class MetricsTrackingCallback(BaseCallback if SB3_AVAILABLE else object):
    """
    Callback to track training metrics at regular intervals.

    This callback evaluates the agent periodically during training and
    stores results in the TrainingMetrics object. This is essential for
    E6 (sample efficiency) analysis where we need metrics at multiple
    checkpoints during training.

    Usage:
    ------
    >>> metrics_cb = MetricsTrackingCallback(
    ...     eval_env=eval_env,
    ...     metrics=agent.metrics,
    ...     eval_freq=10000,
    ...     n_eval_episodes=5
    ... )
    >>> model.learn(total_timesteps=100000, callback=metrics_cb)
    """

    def __init__(
        self,
        eval_env: gym.Env,
        metrics: 'TrainingMetrics',
        eval_freq: int = 10000,
        n_eval_episodes: int = 5,
        verbose: int = 0
    ):
        """
        Initialize the callback.

        Parameters:
        -----------
        eval_env : gym.Env
            Environment for evaluation
        metrics : TrainingMetrics
            Metrics object to store results in
        eval_freq : int
            Evaluate every N timesteps
        n_eval_episodes : int
            Number of episodes per evaluation
        verbose : int
            Verbosity level
        """
        if SB3_AVAILABLE:
            super().__init__(verbose)
        self.eval_env = eval_env
        self.metrics = metrics
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self._last_eval_step = 0

    def _on_step(self) -> bool:
        """Called at each step during training."""
        # Check if it's time to evaluate
        if self.num_timesteps - self._last_eval_step >= self.eval_freq:
            self._last_eval_step = self.num_timesteps
            self._evaluate_and_record()
        return True

    def _evaluate_and_record(self):
        """Run evaluation and record metrics."""
        episode_rewards = []
        episode_lengths = []

        for _ in range(self.n_eval_episodes):
            obs, _ = self.eval_env.reset()
            done = False
            truncated = False
            episode_reward = 0.0
            episode_length = 0

            while not (done or truncated):
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = self.eval_env.step(action)
                episode_reward += reward
                episode_length += 1

            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)

        mean_reward = np.mean(episode_rewards)

        # Try to get financial metrics from the environment
        sharpe = None
        max_dd = None
        if hasattr(self.eval_env, 'get_metrics'):
            try:
                env_metrics = self.eval_env.get_metrics()
                sharpe = env_metrics.get('sharpe_ratio')
                max_dd = env_metrics.get('max_drawdown')
            except Exception:
                pass

        # Add to metrics
        self.metrics.add_eval_result(
            timestep=self.num_timesteps,
            mean_reward=mean_reward,
            sharpe=sharpe,
            max_dd=max_dd
        )

        if self.verbose > 0:
            print(f"[Metrics] Step {self.num_timesteps}: reward={mean_reward:.2f}")


class ReplayBufferWrapper:
    """
    Wrapper to make RegimeAwareReplayBuffer compatible with Stable Baselines 3.

    SB3 expects replay buffers to have specific methods and return formats.
    This wrapper adapts our RegimeAwareReplayBuffer to work with SB3's
    off-policy algorithms (SAC, TD3).

    WHY A WRAPPER?
    ==============

    SB3 expects methods like:
        - buffer.add(obs, action, reward, next_obs, done, infos)
        - buffer.sample(batch_size) -> ReplayBufferSamples (named tuple)

    Our RegimeAwareReplayBuffer has:
        - buffer.add(state, action, reward, next_state, done, regime)
        - buffer.sample(batch_size, current_regime) -> dict

    This wrapper translates between these interfaces.

    Usage:
    ------
    >>> from src.replay.regime_aware_buffer import RegimeAwareReplayBuffer
    >>> regime_buffer = RegimeAwareReplayBuffer(buffer_size=100000, strategy='balanced')
    >>> wrapped = ReplayBufferWrapper(regime_buffer, env.observation_space, env.action_space)
    >>> # Now use wrapped as SB3's replay buffer
    """

    def __init__(
        self,
        regime_buffer,
        observation_space,
        action_space,
        device: str = "cpu",
        n_envs: int = 1,
        regime_detector=None,
        price_data=None,
        env=None,
    ):
        """
        Initialize the wrapper.

        Parameters:
        -----------
        regime_buffer : RegimeAwareReplayBuffer
            Your custom regime-aware replay buffer
        observation_space : spaces.Space
            Environment's observation space
        action_space : spaces.Space
            Environment's action space
        device : str
            Device for tensors ("cpu" or "cuda")
        n_envs : int
            Number of parallel environments (usually 1)
        regime_detector : optional
            Regime detector for labelling experiences
        price_data : optional
            Price data for regime detection caching
        env : optional
            Environment reference for step lookups
        """
        self.regime_buffer = regime_buffer
        self.observation_space = observation_space
        self.action_space = action_space
        self.device = device
        self.n_envs = n_envs
        self.regime_detector = regime_detector
        self._env = env

        # SB3 expects these attributes
        # Handle both possible attribute names (buffer_size or total_buffer_size)
        if hasattr(regime_buffer, 'buffer_size'):
            self.buffer_size = regime_buffer.buffer_size
        elif hasattr(regime_buffer, 'total_buffer_size'):
            self.buffer_size = regime_buffer.total_buffer_size
        else:
            self.buffer_size = 1_000_000  # Default fallback
        self.pos = 0
        self.full = False
        self.optimize_memory_usage = False

        # Track current regime for sampling
        self._current_regime = None

        # Cache regime series for date-based lookups (not index-based!)
        # Using index-based lookup is BUGGY because regime array may have
        # different length than env dates due to NaN warmup drops.
        self._regime_cache = None
        self._env_dates = None
        if price_data is not None and regime_detector is not None:
            try:
                self._regime_cache = regime_detector.detect(price_data)
            except Exception:
                self._regime_cache = None

        # Get environment dates for proper alignment
        if env is not None:
            inner_env = env
            if hasattr(inner_env, 'envs'):
                inner_env = inner_env.envs[0]
            if hasattr(inner_env, 'unwrapped'):
                inner_env = inner_env.unwrapped
            if hasattr(inner_env, 'dates'):
                self._env_dates = inner_env.dates
    
    def _get_env_step(self) -> Optional[int]:
        """Get the current step index from the environment."""
        env = self._env
        if env is None:
            return None
        if hasattr(env, 'current_step'):
            return env.current_step
        if hasattr(env, 'envs') and hasattr(env.envs[0], 'current_step'):
            return env.envs[0].current_step
        if hasattr(env, 'unwrapped') and hasattr(env.unwrapped, 'current_step'):
            return env.unwrapped.current_step
        return None

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: list,
    ) -> None:
        """
        Add experience to buffer (SB3 format).

        SB3 calls this method during training. We translate to our format
        and add to the regime-aware buffer with proper regime detection.
        """
        from src.replay.regime_aware_buffer import MarketRegime

        regime_str = 'sideways'  # Default

        # Method 1: Check if environment provides regime in info
        if infos and len(infos) > 0 and isinstance(infos[0], dict) and 'regime' in infos[0]:
            regime_str = infos[0]['regime']

        # Method 2: Use cached regime series + env dates (date-based lookup)
        # This avoids index misalignment issues when regime cache has different
        # length due to NaN warmup drops in feature calculation
        elif self._regime_cache is not None and self._env_dates is not None:
            current_step = self._get_env_step()
            if current_step is not None and 0 <= current_step < len(self._env_dates):
                current_date = self._env_dates[current_step]
                # Convert numpy datetime64 to pandas-compatible format if needed
                if hasattr(current_date, 'astype'):
                    current_date = pd.Timestamp(current_date)
                if current_date in self._regime_cache.index:
                    regime_val = self._regime_cache.loc[current_date]
                    regime_str = regime_val.value if hasattr(regime_val, 'value') else str(regime_val)

        # Method 3: Use stored current regime from callback
        elif self._current_regime is not None:
            if hasattr(self._current_regime, 'value'):
                regime_str = self._current_regime.value
            else:
                regime_str = str(self._current_regime)

        # Convert string to enum
        regime = MarketRegime(regime_str)

        # Handle batched inputs (n_envs > 1)
        if len(obs.shape) > 1:
            for i in range(obs.shape[0]):
                self.regime_buffer.add(
                    state=obs[i],
                    action=action[i],
                    reward=float(reward[i]),
                    next_state=next_obs[i],
                    done=bool(done[i]),
                    regime=regime,
                )
        else:
            self.regime_buffer.add(
                state=obs,
                action=action,
                reward=float(reward),
                next_state=next_obs,
                done=bool(done),
                regime=regime,
            )

        # Update position tracking for SB3
        self.pos = len(self.regime_buffer) % self.buffer_size
        self.full = len(self.regime_buffer) >= self.buffer_size
    
    def sample(self, batch_size: int, env=None):
        """
        Sample a batch of experiences (SB3 format).

        Returns a ReplayBufferSamples named tuple that SB3 expects.
        Also stores regime info for TD error updates.
        """
        import torch
        from stable_baselines3.common.buffers import ReplayBufferSamples

        # Sample from regime-aware buffer
        batch = self.regime_buffer.sample(
            batch_size=batch_size,
            current_regime=self._current_regime,
        )

        # Store regime indices for TD error update (for adaptive sampling)
        self._last_batch_regimes = batch.get('regimes', None)

        # Convert to tensors
        observations = torch.as_tensor(batch['states'], device=self.device).float()
        actions = torch.as_tensor(batch['actions'], device=self.device).float()
        next_observations = torch.as_tensor(batch['next_states'], device=self.device).float()
        dones = torch.as_tensor(batch['dones'], device=self.device).float().unsqueeze(1)
        rewards = torch.as_tensor(batch['rewards'], device=self.device).float().unsqueeze(1)

        return ReplayBufferSamples(
            observations=observations,
            actions=actions,
            next_observations=next_observations,
            dones=dones,
            rewards=rewards,
        )

    def update_td_errors(self, td_errors) -> None:
        """
        Update TD error statistics for adaptive sampling.

        Called after training step with TD errors for the last sampled batch.
        Groups TD errors by regime and updates the underlying buffer.

        Parameters:
        -----------
        td_errors : np.ndarray or torch.Tensor
            TD errors for each sample in the last batch
        """
        import numpy as np
        from src.replay.regime_aware_buffer import MarketRegime

        if self._last_batch_regimes is None:
            return

        # Convert to numpy if tensor
        if hasattr(td_errors, 'cpu'):
            td_errors = td_errors.detach().cpu().numpy()
        td_errors = np.asarray(td_errors).flatten()

        regimes = self._last_batch_regimes
        if len(td_errors) != len(regimes):
            return

        # Group TD errors by regime and update buffer
        for regime in MarketRegime:
            regime_idx = list(MarketRegime).index(regime)
            mask = (regimes == regime_idx)
            if mask.any():
                regime_td_errors = td_errors[mask]
                self.regime_buffer.update_td_errors(regime, regime_td_errors)
    
    def set_current_regime(self, regime) -> None:
        """Set the current market regime for sampling."""
        from src.replay.regime_aware_buffer import MarketRegime
        # Convert string to enum if needed
        if isinstance(regime, str):
            self._current_regime = MarketRegime(regime)
        elif isinstance(regime, MarketRegime):
            self._current_regime = regime
        else:
            self._current_regime = MarketRegime.SIDEWAYS
    
    def size(self) -> int:
        """Return current buffer size."""
        return len(self.regime_buffer)
    
    def __len__(self) -> int:
        """Return current buffer size."""
        return len(self.regime_buffer)
    
    def reset(self) -> None:
        """Reset the buffer."""
        if hasattr(self.regime_buffer, 'clear'):
            self.regime_buffer.clear()
        self.pos = 0
        self.full = False

# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("BASE AGENT MODULE - TEST")
    print("=" * 70)
    
    # Test TrainingConfig
    print("\n[1] Testing TrainingConfig...")
    config = TrainingConfig(
        total_timesteps=100_000,
        eval_freq=5000,
        seed=123
    )
    print(f"    Total timesteps: {config.total_timesteps}")
    print(f"    Eval frequency: {config.eval_freq}")
    print(f"    Seed: {config.seed}")
    
    # Test serialization
    config_dict = config.to_dict()
    config_loaded = TrainingConfig.from_dict(config_dict)
    assert config.total_timesteps == config_loaded.total_timesteps
    print("    ✓ Config serialization works")
    
    # Test TrainingMetrics
    print("\n[2] Testing TrainingMetrics...")
    metrics = TrainingMetrics()
    metrics.add_eval_result(1000, 0.5, sharpe=1.2, max_dd=0.1)
    metrics.add_eval_result(2000, 0.7, sharpe=1.5, max_dd=0.08)
    metrics.add_eval_result(3000, 0.6, sharpe=1.3, max_dd=0.12)
    
    best_step, best_reward = metrics.get_best_eval_reward()
    print(f"    Best eval reward: {best_reward} at step {best_step}")
    assert best_step == 2000
    assert best_reward == 0.7
    print("    ✓ Metrics tracking works")
    
    # Test that BaseAgent is abstract
    print("\n[3] Testing BaseAgent is abstract...")
    try:
        # This should fail because BaseAgent is abstract
        import gymnasium as gym
        env = gym.make('CartPole-v1')
        agent = BaseAgent(env)  # Should raise TypeError
        print("    ✗ Should have raised TypeError")
    except TypeError as e:
        print(f"    ✓ Correctly raised TypeError (abstract class)")
    
    print("\n" + "=" * 70)
    print("All base agent tests passed!")
    print("=" * 70)