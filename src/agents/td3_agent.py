"""
TD3 (Twin Delayed Deep Deterministic Policy Gradient) Agent for Portfolio Management.

TD3 Key Features:
=================
1. TWIN CRITICS: Uses two Q-networks and takes the minimum value to reduce
   overestimation bias (a common problem in value-based RL).

2. DELAYED POLICY UPDATES: Only updates the policy network every N critic updates.
   This reduces variance in policy gradients and stabilizes training.

3. TARGET POLICY SMOOTHING: Adds noise to target actions when computing
   target Q-values. This prevents the policy from exploiting Q-function errors.

Why TD3 for Portfolio Management?
=================================
- Off-policy: Can use your RegimeAwareReplayBuffer!
- Continuous actions: Natural fit for portfolio weights
- Stable: The twin critics + delayed updates reduce training instability
- Sample efficient: Reuses experience from replay buffer

Comparison with SAC:
- TD3: Deterministic policy + twin critics + delayed updates
- SAC: Stochastic policy + entropy regularization
- Both use replay buffers (good for your regime-aware sampling!)
- SAC often explores better, TD3 often converges faster

Usage:
------
>>> from src.agents.td3_agent import TD3Agent
>>> agent = TD3Agent(env, replay_buffer=regime_aware_buffer)
>>> agent.train(total_timesteps=1_000_000)
>>> agent.save("models/td3_regime_aware")

Author: [Your Name]
Date: February 2026
Project: Regime-Aware Experience Replay for DRL Portfolio Management
"""

import numpy as np
import torch
from typing import Optional, Dict, Any, Union, Callable
from pathlib import Path
import yaml

from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.buffers import ReplayBuffer
import gymnasium as gym

# Import base class
from .base_agent import BaseAgent, ReplayBufferWrapper

# Try to import regime-aware buffer (optional dependency)
try:
    from ..replay.regime_aware_buffer import RegimeAwareReplayBuffer, SamplingStrategy
    REGIME_BUFFER_AVAILABLE = True
except ImportError:
    REGIME_BUFFER_AVAILABLE = False


class TD3Agent(BaseAgent):
    """
    TD3 Agent wrapper with support for regime-aware replay.
    
    TD3 (Twin Delayed DDPG) is an off-policy actor-critic algorithm that
    addresses function approximation errors in actor-critic methods through:
    
    1. Clipped Double Q-Learning: Uses two Q-networks, takes the minimum
       to prevent overestimation
    
    2. Delayed Policy Updates: Updates policy less frequently than Q-functions
    
    3. Target Policy Smoothing: Adds noise to target actions for regularization
    
    Key Hyperparameters:
    --------------------
    - learning_rate: Learning rate for both actor and critic
    - buffer_size: Replay buffer capacity
    - batch_size: Minibatch size for gradient updates
    - tau: Soft update coefficient for target networks
    - gamma: Discount factor
    - policy_delay: Update policy every N critic updates (default: 2)
    - target_policy_noise: Noise added to target actions
    - target_noise_clip: Range to clip target noise
    - train_freq: How often to update (steps between updates)
    
    Attributes:
    -----------
    model : TD3
        The underlying Stable Baselines 3 TD3 model
    env : gym.Env
        The training environment
    config : dict
        Configuration dictionary with hyperparameters
    replay_buffer : ReplayBuffer or RegimeAwareReplayBuffer
        Experience replay buffer (can be regime-aware)
    """
    
    def __init__(
        self,
        env: gym.Env,
        config: Optional[Dict[str, Any]] = None,
        replay_buffer: Optional[Any] = None,
        action_noise: Optional[str] = "normal",
        noise_std: float = 0.1,
        tensorboard_log: Optional[str] = None,
        verbose: int = 1,
        seed: Optional[int] = None,
        device: str = "auto",
        name: str = "td3_agent",
    ):
        """
        Initialize TD3 agent.
        
        Parameters:
        -----------
        env : gym.Env
            Gymnasium-compatible environment (your TradingEnv)
        config : dict, optional
            Hyperparameters. If None, uses defaults from config file.
        replay_buffer : ReplayBuffer or RegimeAwareReplayBuffer, optional
            Custom replay buffer. If None, uses standard SB3 buffer.
            Pass your RegimeAwareReplayBuffer here for the novel contribution!
        action_noise : str
            Type of exploration noise: "normal", "ou", or None
        noise_std : float
            Standard deviation for action noise
        tensorboard_log : str, optional
            Path for TensorBoard logs
        verbose : int
            Verbosity level (0=none, 1=info, 2=debug)
        seed : int, optional
            Random seed for reproducibility
        device : str
            Device to use ("auto", "cpu", "cuda")
        name : str
            Name for this agent
        """
        # Store parameters BEFORE calling super().__init__
        # because super().__init__ calls _setup_model()
        self.tensorboard_log = tensorboard_log
        self._verbose = verbose
        self.device = device
        self.custom_replay_buffer = replay_buffer
        self.action_noise_type = action_noise
        self.noise_std = noise_std
        
        # Load config if not provided
        if config is None:
            config = self._load_default_config()
        
        # Track training info
        self.training_info = {
            "total_timesteps": 0,
            "episodes": 0,
            "best_reward": float("-inf"),
        }
        
        # Call parent init (this calls _setup_model)
        super().__init__(env, name=name, config=config, replay_buffer=replay_buffer, verbose=verbose)
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default TD3 configuration from YAML file."""
        config_path = Path(__file__).parent.parent.parent / "configs" / "algorithms" / "td3.yaml"
        
        if config_path.exists():
            with open(config_path, 'r') as f:
                full_config = yaml.safe_load(f)
                return full_config.get('td3', self._get_hardcoded_defaults())
        else:
            return self._get_hardcoded_defaults()
    
    def _get_hardcoded_defaults(self) -> Dict[str, Any]:
        """Hardcoded defaults matching your project specification."""
        return {
            # Learning rates
            "learning_rate": 3e-4,
            
            # Replay buffer
            "buffer_size": 1_000_000,
            "batch_size": 256,
            
            # Discount and soft updates
            "gamma": 0.99,
            "tau": 0.005,
            
            # TD3-specific
            "policy_delay": 2,  # Update policy every 2 critic updates
            "target_policy_noise": 0.2,  # Noise added to target actions
            "target_noise_clip": 0.5,  # Clip target noise
            
            # Training frequency
            "train_freq": 1,  # Update every step
            "gradient_steps": 1,  # Gradient steps per update
            "learning_starts": 10_000,  # Steps before training begins
            
            # Network architecture
            "policy_kwargs": {
                "net_arch": [256, 256],
            },
        }
    
    def _create_action_noise(
        self, 
        noise_type: Optional[str], 
        noise_std: float
    ):
        """
        Create action noise for exploration.
        
        TD3 is deterministic, so we need explicit exploration noise.
        
        Parameters:
        -----------
        noise_type : str or None
            "normal": Gaussian noise (simple, often works well)
            "ou": Ornstein-Uhlenbeck noise (correlated, good for momentum)
            None: No noise (pure exploitation)
        noise_std : float
            Standard deviation of the noise
        
        Returns:
        --------
        ActionNoise or None
        """
        if noise_type is None:
            return None
        
        n_actions = self.env.action_space.shape[0]
        
        if noise_type == "normal":
            return NormalActionNoise(
                mean=np.zeros(n_actions),
                sigma=noise_std * np.ones(n_actions)
            )
        elif noise_type == "ou":
            return OrnsteinUhlenbeckActionNoise(
                mean=np.zeros(n_actions),
                sigma=noise_std * np.ones(n_actions)
            )
        else:
            raise ValueError(f"Unknown noise type: {noise_type}. Use 'normal', 'ou', or None.")
    
    def _setup_model(self) -> None:
        """Create the TD3 model with configured hyperparameters."""
        
        # Create action noise for exploration
        self.action_noise = self._create_action_noise(self.action_noise_type, self.noise_std)
        
        # Extract policy kwargs
        policy_kwargs = self.config.get("policy_kwargs", {})
        
        # Handle activation function
        if "activation_fn" in policy_kwargs:
            activation_name = policy_kwargs["activation_fn"]
            if activation_name == "relu":
                policy_kwargs["activation_fn"] = torch.nn.ReLU
            elif activation_name == "tanh":
                policy_kwargs["activation_fn"] = torch.nn.Tanh
        
        # Create model
        self.model = TD3(
            policy="MlpPolicy",
            env=self.env,
            learning_rate=self.config.get("learning_rate", 3e-4),
            buffer_size=self.config.get("buffer_size", 1_000_000),
            learning_starts=self.config.get("learning_starts", 10_000),
            batch_size=self.config.get("batch_size", 256),
            tau=self.config.get("tau", 0.005),
            gamma=self.config.get("gamma", 0.99),
            train_freq=self.config.get("train_freq", 1),
            gradient_steps=self.config.get("gradient_steps", 1),
            policy_delay=self.config.get("policy_delay", 2),
            target_policy_noise=self.config.get("target_policy_noise", 0.2),
            target_noise_clip=self.config.get("target_noise_clip", 0.5),
            action_noise=self.action_noise,
            policy_kwargs=policy_kwargs if policy_kwargs else None,
            tensorboard_log=self.tensorboard_log,
            verbose=self._verbose,
            seed=self.config.get("seed"),
            device=self.device,
        )
        
        # Replace replay buffer if custom one provided
        if self.custom_replay_buffer is not None:
            self._integrate_custom_buffer(self.model)
    
    def _integrate_custom_buffer(self, model: TD3) -> None:
        """
        Integrate custom replay buffer (including RegimeAwareReplayBuffer).
        
        This is where your novel contribution connects to TD3!
        
        For regime-aware replay:
        - We wrap the RegimeAwareReplayBuffer to be compatible with SB3
        - The sampling strategy (balanced, inverse-frequency, etc.) is applied
          automatically when the model samples experiences
        """
        if REGIME_BUFFER_AVAILABLE and isinstance(self.replay_buffer, RegimeAwareReplayBuffer):
            # Wrap for SB3 compatibility
            wrapped_buffer = ReplayBufferWrapper(
                self.replay_buffer,
                observation_space=self.env.observation_space,
                action_space=self.env.action_space
            )
            model.replay_buffer = wrapped_buffer
            print(f"✓ TD3 using RegimeAwareReplayBuffer with {self.replay_buffer.strategy.value} sampling")
        elif isinstance(self.replay_buffer, ReplayBuffer):
            # Standard SB3 buffer
            model.replay_buffer = self.replay_buffer
            print("✓ TD3 using custom standard ReplayBuffer")
        else:
            # Try to use as-is
            model.replay_buffer = self.replay_buffer
            print("✓ TD3 using custom buffer (type: {})".format(type(self.replay_buffer).__name__))
    
    def train(
        self,
        total_timesteps: int = None,
        training_config = None,
        callback: Optional[BaseCallback] = None,
        log_interval: int = 100,
        eval_env: Optional[gym.Env] = None,
        eval_freq: int = 10_000,
        n_eval_episodes: int = 10,
        save_freq: int = 50_000,
        save_path: Optional[str] = None,
        reset_num_timesteps: bool = False,
        progress_bar: bool = True,
    ) -> "TD3Agent":
        """
        Train the TD3 agent.
        
        Parameters:
        -----------
        total_timesteps : int
            Total number of environment steps to train for.
            Your spec says 1,000,000 for full training.
        training_config : TrainingConfig, optional
            Training configuration object (alternative to individual params)
        callback : BaseCallback, optional
            Custom callback for monitoring/early stopping
        log_interval : int
            How often to log training info
        eval_env : gym.Env, optional
            Separate environment for evaluation (recommended)
        eval_freq : int
            Evaluate every N steps
        n_eval_episodes : int
            Number of episodes per evaluation
        save_freq : int
            Save model checkpoint every N steps
        save_path : str, optional
            Directory to save checkpoints
        reset_num_timesteps : bool
            Whether to reset the timestep counter
        progress_bar : bool
            Show training progress bar
        
        Returns:
        --------
        self : TD3Agent
            Returns self for method chaining
        """
        # Handle training_config if provided
        if training_config is not None:
            total_timesteps = total_timesteps or training_config.total_timesteps
            eval_freq = training_config.eval_freq
            n_eval_episodes = training_config.n_eval_episodes
            save_freq = training_config.save_freq
            save_path = save_path or training_config.save_path
        
        # Default total_timesteps if still None
        if total_timesteps is None:
            total_timesteps = 100_000
        # Build callback list
        callbacks = []
        
        # Add user callback
        if callback is not None:
            callbacks.append(callback)
        
        # Add evaluation callback
        if eval_env is not None:
            eval_callback = EvalCallback(
                eval_env,
                best_model_save_path=save_path,
                log_path=save_path,
                eval_freq=eval_freq,
                n_eval_episodes=n_eval_episodes,
                deterministic=True,
                render=False,
                verbose=self.verbose,
            )
            callbacks.append(eval_callback)
        
        # Add checkpoint callback
        if save_path is not None:
            checkpoint_callback = CheckpointCallback(
                save_freq=save_freq,
                save_path=save_path,
                name_prefix="td3",
                verbose=self.verbose,
            )
            callbacks.append(checkpoint_callback)
        
        # Combine callbacks
        from stable_baselines3.common.callbacks import CallbackList
        combined_callback = CallbackList(callbacks) if callbacks else None
        
        # Train!
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=combined_callback,
            log_interval=log_interval,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )
        
        # Update training info
        self.training_info["total_timesteps"] += total_timesteps
        
        return self
    
    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = True,
    ) -> tuple:
        """
        Get action from the trained policy.
        
        Parameters:
        -----------
        observation : np.ndarray
            Current observation from environment
        deterministic : bool
            If True, no noise is added (pure exploitation)
            If False, action noise is applied (exploration)
        
        Returns:
        --------
        action : np.ndarray
            The action to take
        state : None
            Hidden state (None for TD3, included for API consistency)
        """
        action, state = self.model.predict(observation, deterministic=deterministic)
        return action, state
    
    def save(self, path: str) -> None:
        """
        Save the model to disk.
        
        Parameters:
        -----------
        path : str
            Path to save the model (without extension)
        """
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(save_path))
        print(f"✓ TD3 model saved to {save_path}")
    
    def load(self, path: str, env: Optional[gym.Env] = None) -> "TD3Agent":
        """
        Load a saved model.
        
        Parameters:
        -----------
        path : str
            Path to the saved model
        env : gym.Env, optional
            Environment to use. If None, uses the original environment.
        
        Returns:
        --------
        self : TD3Agent
            Returns self for method chaining
        """
        if env is not None:
            self.env = env
        
        self.model = TD3.load(path, env=self.env)
        print(f"✓ TD3 model loaded from {path}")
        return self
    
    def evaluate(
        self,
        env: gym.Env,
        n_episodes: int = 10,
        deterministic: bool = True,
    ) -> Dict[str, float]:
        """
        Evaluate the agent's performance.
        
        Parameters:
        -----------
        env : gym.Env
            Environment to evaluate on
        n_episodes : int
            Number of episodes to run
        deterministic : bool
            Use deterministic actions (no exploration)
        
        Returns:
        --------
        dict
            Dictionary with evaluation metrics
        """
        episode_rewards = []
        episode_lengths = []
        
        for _ in range(n_episodes):
            obs, info = env.reset()
            done = False
            episode_reward = 0
            episode_length = 0
            
            while not done:
                action, _ = self.predict(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                episode_reward += reward
                episode_length += 1
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
        
        return {
            "mean_reward": np.mean(episode_rewards),
            "std_reward": np.std(episode_rewards),
            "min_reward": np.min(episode_rewards),
            "max_reward": np.max(episode_rewards),
            "mean_length": np.mean(episode_lengths),
        }
    
    def get_policy(self):
        """Get the policy network."""
        return self.model.policy
    
    def get_critic(self):
        """Get the critic networks (twin Q-networks)."""
        return self.model.critic, self.model.critic_target


# =============================================================================
# Testing
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("TD3 AGENT - TEST")
    print("=" * 70)
    
    # Create a simple test environment
    import gymnasium as gym
    env = gym.make("Pendulum-v1")
    
    # Test 1: Create agent with default config
    print("\n[1] Creating TD3 agent with default config...")
    agent = TD3Agent(env, verbose=0)
    print(f"    ✓ Agent created")
    print(f"    Learning rate: {agent.config.get('learning_rate')}")
    print(f"    Policy delay: {agent.config.get('policy_delay')}")
    print(f"    Target noise: {agent.config.get('target_policy_noise')}")
    
    # Test 2: Quick training
    print("\n[2] Quick training test (1000 steps)...")
    agent.train(total_timesteps=1000, progress_bar=False)
    print(f"    ✓ Training completed")
    
    # Test 3: Prediction
    print("\n[3] Testing prediction...")
    obs, _ = env.reset()
    action, _ = agent.predict(obs, deterministic=True)
    print(f"    Observation shape: {obs.shape}")
    print(f"    Action shape: {action.shape}")
    print(f"    ✓ Prediction works")
    
    # Test 4: Evaluation
    print("\n[4] Running evaluation (3 episodes)...")
    eval_results = agent.evaluate(env, n_episodes=3)
    print(f"    Mean reward: {eval_results['mean_reward']:.2f}")
    print(f"    ✓ Evaluation works")
    
    # Test 5: Save and load
    print("\n[5] Testing save/load...")
    agent.save("/tmp/test_td3")
    agent.load("/tmp/test_td3")
    print(f"    ✓ Save/load works")
    
    print("\n" + "=" * 70)
    print("ALL TESTS PASSED!")
    print("=" * 70)