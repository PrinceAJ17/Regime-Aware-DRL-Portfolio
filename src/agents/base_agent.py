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

Author: [Your Name]
Date: February 2026
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional, Union, Tuple, List
from dataclasses import dataclass, field
from datetime import datetime
import json
import numpy as np
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
        
        regime_rewards = {'bull': [], 'bear': [], 'sideways': []}
        
        for _ in range(n_episodes):
            obs, info = eval_env.reset()
            done = False
            
            # Track rewards per regime within episode
            episode_regime_rewards = {'bull': 0, 'bear': 0, 'sideways': 0}
            episode_regime_steps = {'bull': 0, 'bear': 0, 'sideways': 0}
            
            while not done:
                # Get current regime
                if hasattr(eval_env, 'get_current_data'):
                    current_data = eval_env.get_current_data()
                    regime = self.regime_detector.detect_current(current_data)
                else:
                    regime = 'sideways'  # Default if can't detect
                
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