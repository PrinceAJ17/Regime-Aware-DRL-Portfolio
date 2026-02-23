"""
SAC Agent with Regime-Aware Replay
==================================

Soft Actor-Critic (SAC) implementation with your novel regime-aware replay buffer.

WHY SAC?
========

SAC is one of the most popular off-policy algorithms because:

1. **Maximum Entropy**: Balances exploration and exploitation automatically
   - Adds entropy bonus to reward
   - Agent seeks high-reward states while maintaining diverse behavior
   - Reduces risk of getting stuck in local optima

2. **Off-Policy**: Can reuse old experiences (perfect for your project!)
   - Uses a replay buffer (where your contribution comes in)
   - More sample efficient than on-policy methods
   - Can learn from any experience, not just recent ones

3. **Stable Training**: Uses several tricks for stability
   - Twin critics (reduces overestimation)
   - Automatic entropy tuning
   - Soft updates to target networks

SAC + YOUR REGIME-AWARE REPLAY:
===============================

This is the perfect combination because:

1. SAC already uses a replay buffer for learning
2. We swap the standard buffer with your RegimeAwareReplayBuffer
3. Now SAC samples experiences intelligently by regime
4. Better learning across ALL market conditions

HOW IT WORKS:
=============

Standard SAC:
    buffer.sample(batch_size)  # Uniform random
    
Your SAC:
    buffer.sample(batch_size, strategy='balanced')  # Equal from each regime
    buffer.sample(batch_size, strategy='inverse_frequency')  # More from rare regimes
    buffer.sample(batch_size, current_regime='bear')  # Focus on current regime

References:
-----------
[11] Haarnoja, T., Zhou, A., Abbeel, P., & Levine, S. (2018). 
     Soft actor-critic: Off-policy maximum entropy deep reinforcement learning.
     ICML 2018.

Author: Prince Alexander John
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
    from stable_baselines3 import SAC
    from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
    from stable_baselines3.common.buffers import ReplayBuffer
    from stable_baselines3.common.type_aliases import ReplayBufferSamples
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    print("Warning: stable-baselines3 not installed. Install with: pip install stable-baselines3")

# Import base class and configs
from .base_agent import BaseAgent, TrainingConfig, TrainingMetrics


@dataclass
class SACConfig:
    """
    SAC-specific hyperparameters.
    
    These are the settings from your project specification (Section 5.3.2).
    """
    # Learning
    learning_rate: float = 3e-4
    buffer_size: int = 1_000_000
    batch_size: int = 256
    gamma: float = 0.99
    tau: float = 0.005  # Soft update coefficient
    
    # Entropy
    ent_coef: str = "auto"  # Automatic entropy tuning
    target_entropy: str = "auto"
    
    # Training frequency
    train_freq: int = 1  # Update every step
    gradient_steps: int = 1  # Gradient steps per update
    learning_starts: int = 10000  # Start learning after N steps
    
    # Network architecture
    net_arch: List[int] = None  # Default: [256, 256]
    
    # Replay buffer settings (YOUR CONTRIBUTION!)
    use_regime_aware_replay: bool = True
    replay_strategy: str = "balanced"  # balanced, current_focused, inverse_frequency, adaptive
    
    def __post_init__(self):
        if self.net_arch is None:
            self.net_arch = [256, 256]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'learning_rate': self.learning_rate,
            'buffer_size': self.buffer_size,
            'batch_size': self.batch_size,
            'gamma': self.gamma,
            'tau': self.tau,
            'ent_coef': self.ent_coef,
            'target_entropy': self.target_entropy,
            'train_freq': self.train_freq,
            'gradient_steps': self.gradient_steps,
            'learning_starts': self.learning_starts,
            'net_arch': self.net_arch,
            'use_regime_aware_replay': self.use_regime_aware_replay,
            'replay_strategy': self.replay_strategy
        }


class RegimeAwareReplayBufferWrapper:
    """
    Wrapper to make RegimeAwareReplayBuffer compatible with SB3.
    
    SB3 expects a specific interface for replay buffers.
    This wrapper adapts your buffer to match that interface.
    
    WHY A WRAPPER?
    ==============
    
    SB3's SAC expects methods like:
    - buffer.add(obs, action, reward, next_obs, done, infos)
    - buffer.sample(batch_size) -> ReplayBufferSamples
    
    Your buffer has:
    - buffer.add(state, action, reward, next_state, done, regime)
    - buffer.sample(batch_size, strategy) -> dict
    
    The wrapper translates between these interfaces.
    """
    
    def __init__(
        self,
        regime_aware_buffer,
        regime_detector,
        env: gym.Env,
        price_data=None,
        strategy: str = "balanced"
    ):
        """
        Initialize the wrapper.

        Parameters:
        -----------
        regime_aware_buffer : RegimeAwareReplayBuffer
            Your custom regime-aware buffer

        regime_detector : UnifiedRegimeDetector
            For detecting current market regime

        env : gym.Env
            The environment (needed for observation/action spaces)

        price_data : pd.DataFrame or Dict, optional
            Price data for regime detection

        strategy : str
            Sampling strategy to use
        """
        self.buffer = regime_aware_buffer
        self.regime_detector = regime_detector
        self.env = env
        self.price_data = price_data
        self.strategy = strategy

        # Track current regime (updated during training)
        self.current_regime = None

        # Cache regime series to avoid recomputing on every step
        self._regime_cache = None
        if price_data is not None and regime_detector is not None:
            try:
                self._regime_cache = regime_detector.detect(price_data)
            except Exception:
                self._regime_cache = None
        
        # For SB3 compatibility
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.n_envs = 1
        self.optimize_memory_usage = False
        
        # Buffer properties
        # Handle both possible attribute names
        if hasattr(regime_aware_buffer, 'buffer_size'):
            self.buffer_size = regime_aware_buffer.buffer_size
        elif hasattr(regime_aware_buffer, 'total_buffer_size'):
            self.buffer_size = regime_aware_buffer.total_buffer_size
        else:
            self.buffer_size = 1_000_000  # Default fallback
        self.pos = 0
        self.full = False
    
    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]]
    ):
        """
        Add experience to buffer (SB3 interface).

        This is called by SB3 after each environment step.
        We detect the current regime and pass it to your buffer.
        """
        from ..replay.regime_aware_buffer import MarketRegime

        # Determine regime - try multiple methods
        regime_str = 'sideways'  # Default fallback

        # Method 1: Check if environment provides regime in info
        if infos and len(infos) > 0 and 'regime' in infos[0]:
            regime_str = infos[0]['regime']

        # Method 2: Use cached regime series + env step index
        elif self._regime_cache is not None:
            try:
                current_step = self._get_env_step()
                if current_step is not None and 0 <= current_step < len(self._regime_cache):
                    regime_val = self._regime_cache.iloc[current_step]
                    regime_str = regime_val.value if hasattr(regime_val, 'value') else str(regime_val)
                else:
                    # Step out of range, use last known regime
                    regime_val = self._regime_cache.iloc[-1]
                    regime_str = regime_val.value if hasattr(regime_val, 'value') else str(regime_val)
            except Exception:
                if self.current_regime is not None:
                    regime_str = self.current_regime

        # Method 3: Use cached regime from callback
        elif self.current_regime is not None:
            regime_str = self.current_regime

        # Convert string to enum
        regime = MarketRegime(regime_str)

        # Add to your buffer
        self.buffer.add(
            state=obs.flatten() if obs.ndim > 1 else obs,
            action=action.flatten() if action.ndim > 1 else action,
            reward=float(reward),
            next_state=next_obs.flatten() if next_obs.ndim > 1 else next_obs,
            done=bool(done),
            regime=regime
        )

        # Update position tracking (for SB3 compatibility)
        self.pos = len(self.buffer)
        self.full = self.buffer.is_full()

    def _get_env_step(self) -> Optional[int]:
        """Get the current step index from the environment."""
        env = self.env
        # Try direct attribute
        if hasattr(env, 'current_step'):
            return env.current_step
        # VecEnv wrapped
        if hasattr(env, 'envs') and hasattr(env.envs[0], 'current_step'):
            return env.envs[0].current_step
        # Unwrapped
        if hasattr(env, 'unwrapped') and hasattr(env.unwrapped, 'current_step'):
            return env.unwrapped.current_step
        return None
    
    def sample(self, batch_size: int, env=None) -> 'ReplayBufferSamples':
        """
        Sample from buffer (SB3 interface).
        
        This is where YOUR CONTRIBUTION happens!
        Instead of uniform random sampling, we use your regime-aware strategy.
        """
        # Get current regime for current-focused strategy
        current_regime = None
        if self.strategy == 'current_focused' and self.current_regime is not None:
            from ..replay.regime_aware_buffer import MarketRegime
            current_regime = MarketRegime(self.current_regime)
        
        # Sample using your regime-aware buffer!
        batch = self.buffer.sample(
            batch_size=batch_size,
            current_regime=current_regime
        )
        
        # Convert to SB3 format
        return self._to_sb3_samples(batch)
    
    def _to_sb3_samples(self, batch: Dict[str, np.ndarray]) -> 'ReplayBufferSamples':
        """Convert your batch format to SB3's ReplayBufferSamples."""
        if not SB3_AVAILABLE:
            raise ImportError("stable-baselines3 required")
        
        # Get device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Convert to tensors
        observations = torch.as_tensor(batch['states'], device=device).float()
        actions = torch.as_tensor(batch['actions'], device=device).float()
        next_observations = torch.as_tensor(batch['next_states'], device=device).float()
        rewards = torch.as_tensor(batch['rewards'], device=device).float().unsqueeze(1)
        dones = torch.as_tensor(batch['dones'], device=device).float().unsqueeze(1)
        
        return ReplayBufferSamples(
            observations=observations,
            actions=actions,
            next_observations=next_observations,
            dones=dones,
            rewards=rewards
        )
    
    def set_current_regime(self, regime: str):
        """Update current regime (called during training)."""
        self.current_regime = regime
    
    def size(self) -> int:
        """Return current buffer size."""
        return len(self.buffer)
    
    def __len__(self) -> int:
        return len(self.buffer)


class RegimeTrackingCallback(BaseCallback):
    """
    Callback to track regime during training and update buffer.
    
    This callback:
    1. Detects current market regime at each step
    2. Updates the replay buffer wrapper with current regime
    3. Logs regime distribution statistics
    """
    
    def __init__(
        self,
        replay_wrapper: RegimeAwareReplayBufferWrapper,
        regime_detector,
        env,
        price_data=None,
        verbose: int = 0
    ):
        super().__init__(verbose)
        self.replay_wrapper = replay_wrapper
        self.regime_detector = regime_detector
        self.env = env
        self.price_data = price_data

        # Cache regime series to avoid recomputing on every step
        self._regime_cache = None
        if price_data is not None and regime_detector is not None:
            try:
                self._regime_cache = regime_detector.detect(price_data)
            except Exception:
                self._regime_cache = None

        # Tracking
        self.regime_counts = {'bull': 0, 'bear': 0, 'sideways': 0}

    def _on_step(self) -> bool:
        """Called after each environment step."""
        regime = 'sideways'  # Default

        try:
            # Unwrap environment to access current_step
            env = self.env
            if hasattr(env, 'envs'):
                env = env.envs[0]
            if hasattr(env, 'unwrapped'):
                env = env.unwrapped

            current_step = getattr(env, 'current_step', None)

            # Use cached regime series if available
            if self._regime_cache is not None and current_step is not None:
                if 0 <= current_step < len(self._regime_cache):
                    regime_val = self._regime_cache.iloc[current_step]
                    regime = regime_val.value if hasattr(regime_val, 'value') else str(regime_val)

            # Fallback: try to get price data from env and detect
            elif self.regime_detector is not None:
                price_data = getattr(env, 'price_data', None) or getattr(env, '_price_data', None)
                if price_data is not None and current_step is not None:
                    regime = self.regime_detector.detect_at_index(price_data, current_step)
                elif price_data is not None:
                    regime = self.regime_detector.detect_current(price_data)

        except Exception:
            pass  # Keep default 'sideways'

        # Update wrapper
        self.replay_wrapper.set_current_regime(regime)

        # Track counts
        self.regime_counts[regime] += 1

        return True
    
    def _on_training_end(self):
        """Called at end of training."""
        total = sum(self.regime_counts.values())
        if total > 0:
            print("\nRegime distribution during training:")
            for regime, count in self.regime_counts.items():
                pct = count / total * 100
                print(f"  {regime}: {count} steps ({pct:.1f}%)")


class SACAgent(BaseAgent):
    """
    SAC Agent with regime-aware replay buffer integration.
    
    This is your main SAC implementation that uses your novel contribution.
    
    Example Usage:
    --------------
    >>> from src.environment.trading_env import TradingEnv
    >>> from src.regime import UnifiedRegimeDetector
    >>> from src.replay import RegimeAwareReplayBuffer
    >>>
    >>> # Create components
    >>> env = TradingEnv(...)
    >>> detector = UnifiedRegimeDetector(method='rule_based')
    >>> buffer = RegimeAwareReplayBuffer(buffer_size=100000, strategy='balanced')
    >>>
    >>> # Create agent
    >>> agent = SACAgent(
    ...     env=env,
    ...     name="sac_balanced",
    ...     regime_detector=detector,
    ...     replay_buffer=buffer,
    ...     sac_config=SACConfig(replay_strategy='balanced')
    ... )
    >>>
    >>> # Train
    >>> metrics = agent.train(TrainingConfig(total_timesteps=100000))
    >>>
    >>> # Evaluate by regime
    >>> regime_results = agent.evaluate_by_regime(eval_env, n_episodes=10)
    """
    
    def __init__(
        self,
        env: gym.Env,
        name: str = "sac_agent",
        sac_config: Optional[SACConfig] = None,
        regime_detector = None,
        replay_buffer = None,
        verbose: int = 1
    ):
        """
        Initialize SAC agent.
        
        Parameters:
        -----------
        env : gym.Env
            Trading environment
            
        name : str
            Agent name for logging
            
        sac_config : SACConfig
            SAC hyperparameters
            
        regime_detector : UnifiedRegimeDetector
            For regime detection (required for regime-aware training)
            
        replay_buffer : RegimeAwareReplayBuffer
            Your custom replay buffer
            
        verbose : int
            Verbosity level
        """
        self.sac_config = sac_config or SACConfig()
        
        # Store for base class
        config = self.sac_config.to_dict()
        
        super().__init__(
            env=env,
            name=name,
            config=config,
            regime_detector=regime_detector,
            replay_buffer=replay_buffer,
            verbose=verbose
        )
    
    def _setup_model(self) -> None:
        """Set up the SAC model."""
        if not SB3_AVAILABLE:
            raise ImportError(
                "stable-baselines3 required. Install with: pip install stable-baselines3"
            )
        
        # Build policy kwargs
        policy_kwargs = {
            "net_arch": self.sac_config.net_arch
        }
        
        # Create SAC model
        self.model = SAC(
            policy="MlpPolicy",
            env=self.env,
            learning_rate=self.sac_config.learning_rate,
            buffer_size=self.sac_config.buffer_size,
            batch_size=self.sac_config.batch_size,
            gamma=self.sac_config.gamma,
            tau=self.sac_config.tau,
            ent_coef=self.sac_config.ent_coef,
            train_freq=self.sac_config.train_freq,
            gradient_steps=self.sac_config.gradient_steps,
            learning_starts=self.sac_config.learning_starts,
            policy_kwargs=policy_kwargs,
            verbose=self.verbose
        )
        
        # Replace replay buffer if using regime-aware version
        if self.sac_config.use_regime_aware_replay and self.replay_buffer is not None:
            self._log("Setting up regime-aware replay buffer...")
            
            if self.regime_detector is None:
                raise ValueError(
                    "regime_detector required when use_regime_aware_replay=True. "
                    "Either pass a regime_detector or set use_regime_aware_replay=False."
                )
            
            # Wrap your buffer for SB3 compatibility
            self.replay_wrapper = RegimeAwareReplayBufferWrapper(
                regime_aware_buffer=self.replay_buffer,
                regime_detector=self.regime_detector,
                env=self.env,
                price_data=self._get_price_data_from_env(),
                strategy=self.sac_config.replay_strategy
            )
            
            # Replace SB3's buffer with our wrapper
            self.model.replay_buffer = self.replay_wrapper
            
            self._log(f"Using regime-aware replay with '{self.sac_config.replay_strategy}' strategy")
        else:
            self.replay_wrapper = None
            self._log("Using standard uniform replay buffer")

    def _get_price_data_from_env(self):
        """Extract price data from environment for regime detection."""
        env = self.env

        # Unwrap if needed
        if hasattr(env, 'envs'):
            env = env.envs[0]
        if hasattr(env, 'unwrapped'):
            env = env.unwrapped

        # Try different attribute names
        for attr in ('price_data', '_price_data', 'data', 'df'):
            if hasattr(env, attr):
                return getattr(env, attr)

        return None

    def train(
        self,
        training_config: TrainingConfig,
        total_timesteps: int = None,  # ADD THIS
        eval_env: Optional[gym.Env] = None,
        callbacks: Optional[List] = None,
        progress_bar: bool = True,  # ADD THIS
    ) -> TrainingMetrics:
        """
        Train the SAC agent.
        
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
        # Add regime tracking callback if using regime-aware replay
        if self.replay_wrapper is not None:
            regime_callback = RegimeTrackingCallback(
                replay_wrapper=self.replay_wrapper,
                regime_detector=self.regime_detector,
                env=self.env,
                price_data=self._get_price_data_from_env(),
                verbose=self.verbose
            )
            callback_list.append(regime_callback)
        
        # Add evaluation callback
        if eval_env is not None:
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
        
        # Final evaluation
        if eval_env is not None:
            self._log("Running final evaluation...")
            eval_results = self.evaluate(eval_env, n_episodes=training_config.n_eval_episodes)
            self._log(f"Final mean reward: {eval_results['mean_reward']:.4f}")
            
            self.metrics.add_eval_result(
                timestep=self.total_timesteps,
                mean_reward=eval_results['mean_reward'],
                sharpe=eval_results.get('sharpe_ratio'),
                max_dd=eval_results.get('max_drawdown')
            )
            
            # Regime-specific evaluation
            if self.regime_detector is not None:
                self._log("Evaluating by regime...")
                regime_results = self.evaluate_by_regime(eval_env, n_episodes=5)
                for regime, results in regime_results.items():
                    if results['mean_reward'] is not None:
                        self._log(f"  {regime}: {results['mean_reward']:.4f}")
        
        return self.metrics
    
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
                'sac_config': self.sac_config.to_dict(),
                'total_timesteps': self.total_timesteps,
                'total_episodes': self.total_episodes
            }, f, indent=2)
        
        # Save metrics
        self.metrics.save(path / "metrics.json")
        
        self._log(f"Model saved to {path}")
        self.last_save_time = datetime.now()
    
    def load(self, path: Union[str, Path]) -> None:
        """Load model from disk."""
        path = Path(path)
        
        # Load SB3 model
        self.model = SAC.load(path / "model", env=self.env)
        
        # Load config
        with open(path / "config.json", 'r') as f:
            config = json.load(f)
        
        self.name = config['name']
        self.sac_config = SACConfig(**config['sac_config'])
        self.total_timesteps = config['total_timesteps']
        self.total_episodes = config['total_episodes']
        
        # Load metrics
        if (path / "metrics.json").exists():
            self.metrics = TrainingMetrics.load(path / "metrics.json")
        
        self._log(f"Model loaded from {path}")


def create_sac_agent(
    env: gym.Env,
    name: str = "sac",
    use_regime_aware: bool = True,
    replay_strategy: str = "balanced",
    regime_detector = None,
    **kwargs
) -> SACAgent:
    """
    Factory function to create SAC agent with common configurations.
    
    Parameters:
    -----------
    env : gym.Env
        Trading environment
        
    name : str
        Agent name
        
    use_regime_aware : bool
        Whether to use regime-aware replay (your contribution!)
        
    replay_strategy : str
        Sampling strategy: 'balanced', 'current_focused', 'inverse_frequency', 'adaptive'
        
    regime_detector : optional
        Regime detector instance
        
    **kwargs : dict
        Additional SAC config options
        
    Returns:
    --------
    SACAgent
        Configured SAC agent
    """
    # Create config
    sac_config = SACConfig(
        use_regime_aware_replay=use_regime_aware,
        replay_strategy=replay_strategy,
        **kwargs
    )
    
    # Create replay buffer if using regime-aware
    replay_buffer = None
    if use_regime_aware:
        from ..replay.regime_aware_buffer import RegimeAwareReplayBuffer
        replay_buffer = RegimeAwareReplayBuffer(
            buffer_size=sac_config.buffer_size,
            strategy=replay_strategy
        )
    
    return SACAgent(
        env=env,
        name=name,
        sac_config=sac_config,
        regime_detector=regime_detector,
        replay_buffer=replay_buffer
    )


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("SAC AGENT WITH REGIME-AWARE REPLAY - TEST")
    print("=" * 70)
    
    if not SB3_AVAILABLE:
        print("stable-baselines3 not installed. Skipping tests.")
        exit(0)
    
    # Test SACConfig
    print("\n[1] Testing SACConfig...")
    config = SACConfig(
        learning_rate=1e-3,
        buffer_size=50000,
        use_regime_aware_replay=True,
        replay_strategy='balanced'
    )
    print(f"    Learning rate: {config.learning_rate}")
    print(f"    Buffer size: {config.buffer_size}")
    print(f"    Regime-aware: {config.use_regime_aware_replay}")
    print(f"    Strategy: {config.replay_strategy}")
    print("    ✓ SACConfig works")
    
    # Test with simple environment
    print("\n[2] Testing SACAgent creation...")
    import gymnasium as gym
    
    # Use a simple continuous environment for testing
    env = gym.make('Pendulum-v1')
    
    # Create agent WITHOUT regime-aware (simpler test)
    agent = SACAgent(
        env=env,
        name="test_sac",
        sac_config=SACConfig(
            use_regime_aware_replay=False,
            buffer_size=10000
        ),
        verbose=0
    )
    print(f"    Agent name: {agent.name}")
    print(f"    Algorithm: {agent.__class__.__name__}")
    print("    ✓ SACAgent created successfully")
    
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
    print("All SAC agent tests passed!")
    print("=" * 70)