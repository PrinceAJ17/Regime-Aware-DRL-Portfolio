"""
PPO Agent with Regime-Aware Training
====================================

Proximal Policy Optimization (PPO) implementation with regime-aware modifications.

WHY PPO?
========

PPO is one of the most widely used RL algorithms because:

1. **Stable Training**: Uses clipped objectives to prevent large policy updates
   - Easier to tune than other policy gradient methods
   - Less sensitive to hyperparameters

2. **Sample Efficient** (for on-policy): Uses multiple epochs over collected data
   - Gets more learning signal from each batch
   - Better than simple policy gradient

3. **Simplicity**: Relatively easy to implement and understand
   - No replay buffer (simpler than SAC/TD3)
   - Single objective function

PPO AND YOUR REGIME-AWARE CONTRIBUTION:
========================================

PPO is ON-POLICY, meaning:
- It uses recent experiences only (not a replay buffer)
- Experiences are discarded after each update

This means we CAN'T use RegimeAwareReplayBuffer directly.

INSTEAD, we implement regime-awareness differently:

1. **Regime-Conditioned Policy**: Add regime to observation
2. **Regime-Balanced Rollouts**: Ensure each rollout has diverse regimes
3. **Regime Weighting**: Weight gradients by regime importance
4. **Regime-Specific Value Functions**: Learn different values per regime

For this implementation, we focus on:
- Option 1: Adding regime features to observation
- Tracking regime-specific performance for analysis

The key insight is that even without a replay buffer, we can make PPO
regime-aware through observation augmentation and analysis.

References:
-----------
[10] Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). 
     Proximal policy optimization algorithms. arXiv:1707.06347.

Author: [Your Name]
Date: February 2026
"""

import numpy as np
import torch
import gymnasium as gym
from pathlib import Path
from typing import Dict, Any, Optional, Union, Tuple, List
from dataclasses import dataclass
import json
from datetime import datetime

# Stable-Baselines3 imports
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
    from stable_baselines3.common.vec_env import DummyVecEnv
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    print("Warning: stable-baselines3 not installed. Install with: pip install stable-baselines3")

# Import base class and configs
from .base_agent import BaseAgent, TrainingConfig, TrainingMetrics


@dataclass
class PPOConfig:
    """
    PPO-specific hyperparameters.
    
    These are the settings from your project specification (Section 5.3.1).
    """
    # Learning
    learning_rate: float = 3e-4
    batch_size: int = 64
    n_steps: int = 2048  # Steps to collect before update
    n_epochs: int = 10  # Epochs per update
    
    # PPO specifics
    gamma: float = 0.99
    gae_lambda: float = 0.95  # GAE parameter
    clip_range: float = 0.2  # PPO clip parameter
    
    # Coefficients
    vf_coef: float = 0.5  # Value function coefficient
    ent_coef: float = 0.01  # Entropy coefficient
    max_grad_norm: float = 0.5  # Gradient clipping
    
    # Network architecture
    net_arch: List[int] = None  # Default: [256, 256]
    
    # Regime-aware settings
    add_regime_to_obs: bool = True  # Add regime one-hot to observation
    track_regime_metrics: bool = True  # Track performance by regime
    
    def __post_init__(self):
        if self.net_arch is None:
            self.net_arch = [256, 256]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'n_steps': self.n_steps,
            'n_epochs': self.n_epochs,
            'gamma': self.gamma,
            'gae_lambda': self.gae_lambda,
            'clip_range': self.clip_range,
            'vf_coef': self.vf_coef,
            'ent_coef': self.ent_coef,
            'max_grad_norm': self.max_grad_norm,
            'net_arch': self.net_arch,
            'add_regime_to_obs': self.add_regime_to_obs,
            'track_regime_metrics': self.track_regime_metrics
        }


class RegimeObservationWrapper(gym.ObservationWrapper):
    """
    Wrapper that adds regime information to observations.
    
    Since PPO is on-policy and can't use a replay buffer,
    we make it regime-aware by AUGMENTING THE OBSERVATION.
    
    The agent receives:
    - Original observation (e.g., 98 features)
    - Regime one-hot encoding (3 features: [bull, bear, sideways])
    
    Total: 101 features
    
    This lets the policy learn regime-conditional behavior:
    - "If we're in a bull market, do X"
    - "If we're in a bear market, do Y"
    """
    
    def __init__(self, env: gym.Env, regime_detector):
        """
        Initialize wrapper.
        
        Parameters:
        -----------
        env : gym.Env
            The trading environment
            
        regime_detector : UnifiedRegimeDetector
            For detecting current market regime
        """
        super().__init__(env)
        self.regime_detector = regime_detector
        
        # Modify observation space to include regime
        original_shape = env.observation_space.shape[0]
        new_shape = original_shape + 3  # Add 3 for regime one-hot
        
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(new_shape,),
            dtype=np.float32
        )
        
        # Track current regime
        self.current_regime = 'sideways'
    
    def observation(self, obs: np.ndarray) -> np.ndarray:
        """Augment observation with regime one-hot."""
        # Get current regime
        if hasattr(self.env, 'get_current_data'):
            current_data = self.env.get_current_data()
            self.current_regime = self.regime_detector.detect_current(current_data)
        
        # Create one-hot encoding
        regime_onehot = np.zeros(3, dtype=np.float32)
        regime_idx = {'bull': 0, 'bear': 1, 'sideways': 2}
        regime_onehot[regime_idx.get(self.current_regime, 2)] = 1.0
        
        # Concatenate
        augmented_obs = np.concatenate([obs, regime_onehot])
        return augmented_obs


class RegimeMetricsCallback(BaseCallback):
    """
    Callback to track regime-specific metrics during PPO training.
    
    Since PPO is on-policy, we track metrics during rollout collection:
    - Rewards earned in each regime
    - Steps spent in each regime
    - Transitions between regimes
    """
    
    def __init__(
        self,
        regime_detector,
        env,
        verbose: int = 0
    ):
        super().__init__(verbose)
        self.regime_detector = regime_detector
        self.env = env
        
        # Tracking
        self.regime_rewards = {'bull': [], 'bear': [], 'sideways': []}
        self.regime_steps = {'bull': 0, 'bear': 0, 'sideways': 0}
        self.current_regime = 'sideways'
        
        # Episode tracking
        self.episode_regime_rewards = {'bull': 0, 'bear': 0, 'sideways': 0}
    
    def _on_step(self) -> bool:
        """Called after each environment step."""
        # Get current regime
        if hasattr(self.env, 'envs'):
            # VecEnv
            base_env = self.env.envs[0]
        else:
            base_env = self.env
        
        if hasattr(base_env, 'current_regime'):
            self.current_regime = base_env.current_regime
        
        # Get reward
        reward = self.locals.get('rewards', [0])[0]
        
        # Track by regime
        self.episode_regime_rewards[self.current_regime] += reward
        self.regime_steps[self.current_regime] += 1
        
        # Check if episode ended
        dones = self.locals.get('dones', [False])
        if dones[0]:
            # Store episode rewards
            for regime, reward in self.episode_regime_rewards.items():
                if reward != 0 or self.regime_steps[regime] > 0:
                    self.regime_rewards[regime].append(reward)
            
            # Reset episode tracking
            self.episode_regime_rewards = {'bull': 0, 'bear': 0, 'sideways': 0}
        
        return True
    
    def _on_training_end(self):
        """Called at end of training."""
        print("\nRegime-specific training statistics:")
        print("-" * 50)
        
        total_steps = sum(self.regime_steps.values())
        
        for regime in ['bull', 'bear', 'sideways']:
            steps = self.regime_steps[regime]
            pct = (steps / total_steps * 100) if total_steps > 0 else 0
            
            rewards = self.regime_rewards[regime]
            mean_reward = np.mean(rewards) if rewards else 0
            
            print(f"  {regime.capitalize():10s}: {steps:6d} steps ({pct:5.1f}%), "
                  f"mean reward: {mean_reward:7.4f}")
    
    def get_regime_metrics(self) -> Dict[str, Dict[str, float]]:
        """Get summary of regime metrics."""
        results = {}
        for regime in ['bull', 'bear', 'sideways']:
            rewards = self.regime_rewards[regime]
            results[regime] = {
                'total_steps': self.regime_steps[regime],
                'n_episodes': len(rewards),
                'mean_reward': np.mean(rewards) if rewards else 0,
                'std_reward': np.std(rewards) if rewards else 0
            }
        return results


class PPOAgent(BaseAgent):
    """
    PPO Agent with regime-aware training enhancements.
    
    While PPO can't use a replay buffer (it's on-policy), we make it
    regime-aware through:
    
    1. Observation augmentation (add regime one-hot)
    2. Regime-specific metrics tracking
    3. Analysis of performance across regimes
    
    Example Usage:
    --------------
    >>> from src.environment.trading_env import TradingEnv
    >>> from src.regime import UnifiedRegimeDetector
    >>>
    >>> # Create components
    >>> env = TradingEnv(...)
    >>> detector = UnifiedRegimeDetector(method='rule_based')
    >>>
    >>> # Create agent with regime-augmented observations
    >>> agent = PPOAgent(
    ...     env=env,
    ...     name="ppo_regime_aware",
    ...     regime_detector=detector,
    ...     ppo_config=PPOConfig(add_regime_to_obs=True)
    ... )
    >>>
    >>> # Train
    >>> metrics = agent.train(TrainingConfig(total_timesteps=100000))
    """
    
    def __init__(
        self,
        env: gym.Env,
        name: str = "ppo_agent",
        ppo_config: Optional[PPOConfig] = None,
        regime_detector = None,
        replay_buffer = None,  # Not used for PPO, but kept for API consistency
        verbose: int = 1
    ):
        """
        Initialize PPO agent.
        
        Parameters:
        -----------
        env : gym.Env
            Trading environment
            
        name : str
            Agent name for logging
            
        ppo_config : PPOConfig
            PPO hyperparameters
            
        regime_detector : UnifiedRegimeDetector
            For regime detection
            
        verbose : int
            Verbosity level
        """
        self.ppo_config = ppo_config or PPOConfig()
        
        # Wrap environment if adding regime to obs
        self.original_env = env
        if self.ppo_config.add_regime_to_obs and regime_detector is not None:
            self._log_message = f"Wrapping environment with regime observations"
            env = RegimeObservationWrapper(env, regime_detector)
        
        # Store for base class
        config = self.ppo_config.to_dict()
        
        super().__init__(
            env=env,
            name=name,
            config=config,
            regime_detector=regime_detector,
            replay_buffer=None,  # PPO doesn't use replay buffer
            verbose=verbose
        )
        
        # Regime metrics callback (created during training)
        self.regime_callback = None
    
    def _log(self, message: str, level: int = 1):
        """Override to handle pre-init logging."""
        if hasattr(self, 'verbose') and self.verbose >= level:
            print(f"[{getattr(self, 'name', 'ppo')}] {message}")
    
    def _setup_model(self) -> None:
        """Set up the PPO model."""
        if not SB3_AVAILABLE:
            raise ImportError(
                "stable-baselines3 required. Install with: pip install stable-baselines3"
            )
        
        # Build policy kwargs
        policy_kwargs = {
            "net_arch": {
                "pi": self.ppo_config.net_arch,  # Policy network
                "vf": self.ppo_config.net_arch   # Value network
            }
        }
        
        # Create PPO model
        self.model = PPO(
            policy="MlpPolicy",
            env=self.env,
            learning_rate=self.ppo_config.learning_rate,
            n_steps=self.ppo_config.n_steps,
            batch_size=self.ppo_config.batch_size,
            n_epochs=self.ppo_config.n_epochs,
            gamma=self.ppo_config.gamma,
            gae_lambda=self.ppo_config.gae_lambda,
            clip_range=self.ppo_config.clip_range,
            vf_coef=self.ppo_config.vf_coef,
            ent_coef=self.ppo_config.ent_coef,
            max_grad_norm=self.ppo_config.max_grad_norm,
            policy_kwargs=policy_kwargs,
            verbose=self.verbose
        )
        
        if hasattr(self, '_log_message'):
            self._log(self._log_message)
        
        if self.ppo_config.add_regime_to_obs:
            self._log("PPO is using regime-augmented observations")
        else:
            self._log("PPO is using standard observations")
    
    def train(
        self,
        training_config: TrainingConfig,
        eval_env: Optional[gym.Env] = None,
        total_timesteps: int = None,  # ADD THIS
        callbacks: Optional[List] = None,
        progress_bar: bool = True,  # ADD THIS
    ) -> TrainingMetrics:
        """
        Train the PPO agent.
        
        Parameters:
        -----------
        training_config : TrainingConfig
            Training settings
            
        eval_env : gym.Env, optional
            Separate environment for evaluation
            
        callbacks : list, optional
            Additional callbacks
            
        Returns:
        --------
        TrainingMetrics
            Training history
        """
        self._log(f"Starting training for {training_config.total_timesteps} timesteps...")
        
        # Set up callbacks
        callback_list = callbacks or []
        
        # Handle direct total_timesteps argument
        if training_config is None and total_timesteps is not None:
            training_config = TrainingConfig(total_timesteps=total_timesteps)
        elif training_config is None:
            training_config = TrainingConfig()

        # Add regime metrics callback if tracking
        if self.ppo_config.track_regime_metrics and self.regime_detector is not None:
            self.regime_callback = RegimeMetricsCallback(
                regime_detector=self.regime_detector,
                env=self.original_env,
                verbose=self.verbose
            )
            callback_list.append(self.regime_callback)
        
        # Add evaluation callback
        if eval_env is not None:
            # Wrap eval env if needed
            if self.ppo_config.add_regime_to_obs and self.regime_detector is not None:
                eval_env = RegimeObservationWrapper(eval_env, self.regime_detector)
            
            eval_callback = EvalCallback(
                eval_env,
                best_model_save_path=training_config.save_path,
                log_path=training_config.save_path,
                eval_freq=training_config.eval_freq,
                n_eval_episodes=training_config.n_eval_episodes,
                deterministic=True,
                verbose=self.verbose
            )
            callback_list.append(eval_callback)
        
        # Train!
        start_time = datetime.now()
        
        self.model.learn(
            total_timesteps=training_config.total_timesteps,
            callback=callback_list if callback_list else None,
            log_interval=training_config.log_interval,
            tb_log_name=self.name,
            progress_bar=self.verbose > 0
        )
        
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Update metrics
        self.total_timesteps += training_config.total_timesteps
        self.metrics.wall_time.append(training_time)
        
        self._log(f"Training completed in {training_time:.1f} seconds")
        
        # Get regime metrics if tracked
        if self.regime_callback is not None:
            regime_metrics = self.regime_callback.get_regime_metrics()
            self._log("Regime performance summary:")
            for regime, metrics in regime_metrics.items():
                self._log(f"  {regime}: mean_reward={metrics['mean_reward']:.4f}")
        
        # Final evaluation
        if eval_env is not None:
            self._log("Running final evaluation...")
            
            # Need to wrap for evaluation
            eval_results = self._evaluate_wrapped(
                eval_env, 
                n_episodes=training_config.n_eval_episodes
            )
            self._log(f"Final mean reward: {eval_results['mean_reward']:.4f}")
            
            self.metrics.add_eval_result(
                timestep=self.total_timesteps,
                mean_reward=eval_results['mean_reward'],
                sharpe=eval_results.get('sharpe_ratio'),
                max_dd=eval_results.get('max_drawdown')
            )
        
        return self.metrics
    
    def _evaluate_wrapped(
        self,
        eval_env: gym.Env,
        n_episodes: int = 10
    ) -> Dict[str, float]:
        """Evaluate with properly wrapped environment."""
        episode_rewards = []
        episode_lengths = []
        
        for _ in range(n_episodes):
            obs, info = eval_env.reset()
            done = False
            episode_reward = 0
            episode_length = 0
            
            while not done:
                action, _ = self.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = eval_env.step(action)
                done = terminated or truncated
                episode_reward += reward
                episode_length += 1
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
        
        return {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'min_reward': np.min(episode_rewards),
            'max_reward': np.max(episode_rewards),
            'mean_length': np.mean(episode_lengths),
            'n_episodes': n_episodes
        }
    
    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = True
    ) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
        """Get action from observation."""
        action, _states = self.model.predict(observation, deterministic=deterministic)
        return action, None
    
    def save(self, path: Union[str, Path]) -> None:
        """Save the model and all associated data."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save SB3 model
        self.model.save(path / "model")
        
        # Save config
        with open(path / "config.json", 'w') as f:
            json.dump({
                'name': self.name,
                'ppo_config': self.ppo_config.to_dict(),
                'total_timesteps': self.total_timesteps,
                'total_episodes': self.total_episodes
            }, f, indent=2)
        
        # Save metrics
        self.metrics.save(path / "metrics.json")
        
        # Save regime metrics if available
        if self.regime_callback is not None:
            regime_metrics = self.regime_callback.get_regime_metrics()
            with open(path / "regime_metrics.json", 'w') as f:
                json.dump(regime_metrics, f, indent=2)
        
        self._log(f"Model saved to {path}")
        self.last_save_time = datetime.now()
    
    def load(self, path: Union[str, Path]) -> None:
        """Load model from disk."""
        path = Path(path)
        
        # Load SB3 model
        self.model = PPO.load(path / "model", env=self.env)
        
        # Load config
        with open(path / "config.json", 'r') as f:
            config = json.load(f)
        
        self.name = config['name']
        self.ppo_config = PPOConfig(**config['ppo_config'])
        self.total_timesteps = config['total_timesteps']
        self.total_episodes = config['total_episodes']
        
        # Load metrics
        if (path / "metrics.json").exists():
            self.metrics = TrainingMetrics.load(path / "metrics.json")
        
        self._log(f"Model loaded from {path}")


def create_ppo_agent(
    env: gym.Env,
    name: str = "ppo",
    add_regime_to_obs: bool = True,
    regime_detector = None,
    **kwargs
) -> PPOAgent:
    """
    Factory function to create PPO agent with common configurations.
    
    Parameters:
    -----------
    env : gym.Env
        Trading environment
        
    name : str
        Agent name
        
    add_regime_to_obs : bool
        Whether to augment observations with regime one-hot
        
    regime_detector : optional
        Regime detector instance
        
    **kwargs : dict
        Additional PPO config options
        
    Returns:
    --------
    PPOAgent
        Configured PPO agent
    """
    # Create config
    ppo_config = PPOConfig(
        add_regime_to_obs=add_regime_to_obs,
        **kwargs
    )
    
    return PPOAgent(
        env=env,
        name=name,
        ppo_config=ppo_config,
        regime_detector=regime_detector
    )


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("PPO AGENT WITH REGIME-AWARE TRAINING - TEST")
    print("=" * 70)
    
    if not SB3_AVAILABLE:
        print("stable-baselines3 not installed. Skipping tests.")
        exit(0)
    
    # Test PPOConfig
    print("\n[1] Testing PPOConfig...")
    config = PPOConfig(
        learning_rate=1e-3,
        n_steps=1024,
        add_regime_to_obs=True
    )
    print(f"    Learning rate: {config.learning_rate}")
    print(f"    N steps: {config.n_steps}")
    print(f"    Add regime to obs: {config.add_regime_to_obs}")
    print("    ✓ PPOConfig works")
    
    # Test with simple environment
    print("\n[2] Testing PPOAgent creation...")
    import gymnasium as gym
    
    # Use a simple continuous environment for testing
    env = gym.make('Pendulum-v1')
    
    # Create agent WITHOUT regime features (simpler test)
    agent = PPOAgent(
        env=env,
        name="test_ppo",
        ppo_config=PPOConfig(add_regime_to_obs=False),
        verbose=0
    )
    print(f"    Agent name: {agent.name}")
    print(f"    Algorithm: {agent.__class__.__name__}")
    print("    ✓ PPOAgent created successfully")
    
    # Test prediction
    print("\n[3] Testing prediction...")
    obs, _ = env.reset()
    action, info = agent.predict(obs)
    print(f"    Observation shape: {obs.shape}")
    print(f"    Action shape: {action.shape}")
    print("    ✓ Prediction works")
    
    # Test short training
    print("\n[4] Testing short training (1000 steps)...")
    metrics = agent.train(
        TrainingConfig(total_timesteps=1000)
    )
    print(f"    Total timesteps: {agent.total_timesteps}")
    print("    ✓ Training works")
    
    # Clean up
    env.close()
    
    print("\n" + "=" * 70)
    print("All PPO agent tests passed!")
    print("=" * 70)