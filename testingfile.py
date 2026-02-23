"""
Integration Test Script - Week 5.5
==================================

This script verifies that all your components work together:
1. Data loading
2. Feature calculation
3. Trading environment
4. All 4 RL algorithms (PPO, SAC, TD3, Rainbow)
5. Regime-aware replay buffer

Run this BEFORE starting Week 6 experiments to catch any integration issues.

Usage:
------
python integration_test.py

Expected time: ~10-15 minutes (CPU)

Author: [Your Name]
Date: February 2026
"""

import sys
import time
import traceback
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import numpy as np
import warnings
warnings.filterwarnings('ignore')


def print_header(title: str):
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def print_step(step: str, status: str = ""):
    """Print a step with optional status."""
    if status:
        print(f"  {step}: {status}")
    else:
        print(f"  {step}...")


def test_data_loading():
    """Test 1: Load market data."""
    print_header("TEST 1: Data Loading")
    
    try:
        from src.data.downloader import download_ohlcv_data
        
        print_step("Downloading data for SPY, QQQ, TLT, GLD, BIL (2020-2023)")
        
        # Use a shorter period for quick testing
        data = download_ohlcv_data(
            tickers=['SPY', 'QQQ', 'TLT', 'GLD', 'BIL'],
            start_date='2020-01-01',
            end_date='2023-12-31'
        )
        
        print_step("Data loaded", f"✓ {len(data)} tickers")
        
        for ticker, df in data.items():
            print_step(f"  {ticker}", f"{len(df)} days, columns: {list(df.columns)[:5]}...")
        
        print("\n✓ TEST 1 PASSED: Data loading works!")
        return data
        
    except Exception as e:
        print(f"\n✗ TEST 1 FAILED: {e}")
        traceback.print_exc()
        return None


def test_feature_calculation(price_data):
    """Test 2: Calculate features."""
    print_header("TEST 2: Feature Calculation")
    
    if price_data is None:
        print("  Skipping - no price data available")
        return None
    
    try:
        from src.data.features import calculate_all_features
        
        print_step("Calculating technical features")
        features = calculate_all_features(price_data)
        
        print_step("Features calculated", f"✓ {len(features)} tickers")
        
        for ticker, df in features.items():
            valid_rows = (~df.isna().any(axis=1)).sum()
            print_step(f"  {ticker}", f"{len(df.columns)} features, {valid_rows} valid rows")
        
        print("\n✓ TEST 2 PASSED: Feature calculation works!")
        return features
        
    except Exception as e:
        print(f"\n✗ TEST 2 FAILED: {e}")
        traceback.print_exc()
        return None


def test_trading_environment(price_data, feature_data):
    """Test 3: Create and test trading environment."""
    print_header("TEST 3: Trading Environment")
    
    if price_data is None or feature_data is None:
        print("  Skipping - missing data")
        return None
    
    try:
        from src.environment.trading_env import TradingEnvironment
        
        print_step("Creating TradingEnvironment")
        env = TradingEnvironment(
            price_data=price_data,
            feature_data=feature_data,
            episode_length=252,
            random_start=True
        )
        
        print_step("Environment created", "✓")
        print_step(f"  Observation space", f"{env.observation_space.shape}")
        print_step(f"  Action space", f"{env.action_space.shape}")
        print_step(f"  Data length", f"{env.n_steps} days")
        
        # Test reset
        print_step("Testing reset")
        obs, info = env.reset()
        print_step("  Reset", f"✓ obs shape: {obs.shape}")
        
        # Test a few steps
        print_step("Testing 10 random steps")
        total_reward = 0
        for i in range(10):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        
        print_step("  Steps completed", f"✓ total reward: {total_reward:.4f}")
        
        env.close()
        print("\n✓ TEST 3 PASSED: Trading environment works!")
        return env
        
    except Exception as e:
        print(f"\n✗ TEST 3 FAILED: {e}")
        traceback.print_exc()
        return None


def test_ppo_training(price_data, feature_data):
    """Test 4: Train PPO agent."""
    print_header("TEST 4: PPO Training")
    
    if price_data is None or feature_data is None:
        print("  Skipping - missing data")
        return False
    
    try:
        from src.environment.trading_env import TradingEnvironment
        from src.agents.ppo_agent import PPOAgent
        
        print_step("Creating environment and agent")
        env = TradingEnvironment(
            price_data=price_data,
            feature_data=feature_data,
            episode_length=50,  # Short episodes for testing
            random_start=True
        )
        
        agent = PPOAgent(env, verbose=0)
        print_step("Agent created", "✓")
        
        print_step("Training for 1,000 steps")
        start_time = time.time()
        agent.train(total_timesteps=1000, progress_bar=False)
        elapsed = time.time() - start_time
        
        print_step("Training completed", f"✓ ({elapsed:.1f}s)")
        
        # Test prediction
        obs, _ = env.reset()
        action, _ = agent.predict(obs, deterministic=True)
        print_step("Prediction", f"✓ action shape: {action.shape}")
        
        env.close()
        print("\n✓ TEST 4 PASSED: PPO training works!")
        return True
        
    except Exception as e:
        print(f"\n✗ TEST 4 FAILED: {e}")
        traceback.print_exc()
        return False


def test_sac_training(price_data, feature_data):
    """Test 5: Train SAC agent."""
    print_header("TEST 5: SAC Training")
    
    if price_data is None or feature_data is None:
        print("  Skipping - missing data")
        return False
    
    try:
        from src.environment.trading_env import TradingEnvironment
        from src.agents.sac_agent import SACAgent
        
        print_step("Creating environment and agent")
        env = TradingEnvironment(
            price_data=price_data,
            feature_data=feature_data,
            episode_length=50,
            random_start=True
        )
        
        agent = SACAgent(env, verbose=0)
        print_step("Agent created", "✓")
        
        print_step("Training for 1,000 steps")
        start_time = time.time()
        agent.train(total_timesteps=1000, progress_bar=False)
        elapsed = time.time() - start_time
        
        print_step("Training completed", f"✓ ({elapsed:.1f}s)")
        
        env.close()
        print("\n✓ TEST 5 PASSED: SAC training works!")
        return True
        
    except Exception as e:
        print(f"\n✗ TEST 5 FAILED: {e}")
        traceback.print_exc()
        return False


def test_td3_training(price_data, feature_data):
    """Test 6: Train TD3 agent."""
    print_header("TEST 6: TD3 Training")
    
    if price_data is None or feature_data is None:
        print("  Skipping - missing data")
        return False
    
    try:
        from src.environment.trading_env import TradingEnvironment
        from src.agents.td3_agent import TD3Agent
        
        print_step("Creating environment and agent")
        env = TradingEnvironment(
            price_data=price_data,
            feature_data=feature_data,
            episode_length=50,
            random_start=True
        )
        
        agent = TD3Agent(env, verbose=0)
        print_step("Agent created", "✓")
        
        print_step("Training for 1,000 steps")
        start_time = time.time()
        agent.train(total_timesteps=1000, progress_bar=False)
        elapsed = time.time() - start_time
        
        print_step("Training completed", f"✓ ({elapsed:.1f}s)")
        
        env.close()
        print("\n✓ TEST 6 PASSED: TD3 training works!")
        return True
        
    except Exception as e:
        print(f"\n✗ TEST 6 FAILED: {e}")
        traceback.print_exc()
        return False


def test_rainbow_training(price_data, feature_data):
    """Test 7: Train Rainbow agent (with DiscreteActionWrapper)."""
    print_header("TEST 7: Rainbow Training")
    
    if price_data is None or feature_data is None:
        print("  Skipping - missing data")
        return False
    
    try:
        from src.environment.trading_env import TradingEnvironment
        from src.agents.rainbow_agent import RainbowAgent, DiscreteActionWrapper
        
        print_step("Creating environment")
        env = TradingEnvironment(
            price_data=price_data,
            feature_data=feature_data,
            episode_length=50,
            random_start=True
        )
        
        # Rainbow needs discrete actions
        print_step("Wrapping with DiscreteActionWrapper")
        wrapped_env = DiscreteActionWrapper(env, actions_per_asset=3, step_size=0.1)
        print_step("  Discrete actions", f"{wrapped_env.n_actions} total")
        
        agent = RainbowAgent(wrapped_env, device="cpu", verbose=0)
        print_step("Agent created", "✓")
        
        print_step("Training for 500 steps")
        start_time = time.time()
        agent.train(total_timesteps=500, progress_bar=False)
        elapsed = time.time() - start_time
        
        print_step("Training completed", f"✓ ({elapsed:.1f}s, {agent.episodes} episodes)")
        
        wrapped_env.close()
        print("\n✓ TEST 7 PASSED: Rainbow training works!")
        return True
        
    except Exception as e:
        print(f"\n✗ TEST 7 FAILED: {e}")
        traceback.print_exc()
        return False


def test_regime_aware_training(price_data, feature_data):
    """Test 8: Train with regime-aware replay buffer."""
    print_header("TEST 8: Regime-Aware Replay (SAC)")
    
    if price_data is None or feature_data is None:
        print("  Skipping - missing data")
        return False
    
    try:
        from src.environment.trading_env import TradingEnvironment
        from src.agents.sac_agent import SACAgent
        from src.replay.regime_aware_buffer import RegimeAwareReplayBuffer, SamplingStrategy
        
        print_step("Creating environment")
        env = TradingEnvironment(
            price_data=price_data,
            feature_data=feature_data,
            episode_length=50,
            random_start=True
        )
        
        print_step("Creating RegimeAwareReplayBuffer")
        buffer = RegimeAwareReplayBuffer(
            buffer_size=10000,
            strategy=SamplingStrategy.BALANCED
        )
        
        print_step("Creating SAC with regime-aware replay")
        agent = SACAgent(env, replay_buffer=buffer, verbose=0)
        
        print_step("Training for 1,000 steps")
        start_time = time.time()
        agent.train(total_timesteps=1000, progress_bar=False)
        elapsed = time.time() - start_time
        
        print_step("Training completed", f"✓ ({elapsed:.1f}s)")
        
        # Check buffer stats
        print_step("Buffer stats")
        stats = buffer.get_statistics()
        for regime, count in stats.get('experiences_per_regime', {}).items():
            print_step(f"  {regime}", f"{count} experiences")
        
        env.close()
        print("\n✓ TEST 8 PASSED: Regime-aware replay works!")
        return True
        
    except Exception as e:
        print(f"\n✗ TEST 8 FAILED: {e}")
        traceback.print_exc()
        return False


def estimate_training_time():
    """Estimate how long full training will take."""
    print_header("TRAINING TIME ESTIMATE")
    
    print("""
Based on your CPU-only setup:

| Timesteps | Estimated Time (per run) |
|-----------|--------------------------|
| 10,000    | ~2-5 minutes             |
| 100,000   | ~20-50 minutes           |
| 500,000   | ~2-4 hours               |
| 1,000,000 | ~4-8 hours               |

For Week 6 experiments (E1: Algorithm x Replay comparison):
- 4 algorithms × 5 replay types × 5 seeds = 100 runs
- At 100,000 steps each: ~33-83 hours total
- Recommendation: Start with 50,000 steps, run overnight

To speed up:
1. Use shorter episodes (50-100 days vs 252)
2. Run fewer seeds initially (3 instead of 5)
3. Use Google Colab for GPU access (free tier)
""")


def main():
    """Run all integration tests."""
    print("\n" + "=" * 70)
    print(" INTEGRATION TEST - Verifying All Components")
    print("=" * 70)
    print("\nThis script tests that your entire pipeline works together.")
    print("Estimated time: 10-15 minutes\n")
    
    results = {}
    
    # Test 1: Data loading
    price_data = test_data_loading()
    results['Data Loading'] = price_data is not None
    
    # Test 2: Feature calculation
    feature_data = test_feature_calculation(price_data)
    results['Feature Calculation'] = feature_data is not None
    
    # Test 3: Trading environment
    env_result = test_trading_environment(price_data, feature_data)
    results['Trading Environment'] = env_result is not None
    
    # Test 4-7: Algorithm training
    results['PPO Training'] = test_ppo_training(price_data, feature_data)
    results['SAC Training'] = test_sac_training(price_data, feature_data)
    results['TD3 Training'] = test_td3_training(price_data, feature_data)
    results['Rainbow Training'] = test_rainbow_training(price_data, feature_data)
    
    # Test 8: Regime-aware replay
    results['Regime-Aware Replay'] = test_regime_aware_training(price_data, feature_data)
    
    # Training time estimate
    estimate_training_time()
    
    # Summary
    print_header("INTEGRATION TEST SUMMARY")
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {test_name}")
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n" + "=" * 70)
        print(" 🎉 ALL INTEGRATION TESTS PASSED!")
        print("=" * 70)
        print("""
Your entire pipeline is working! You're ready for Week 6 experiments.

Recommended next steps:
1. Run a longer training session (10,000+ steps) to verify stability
2. Set up experiment tracking (TensorBoard or W&B)
3. Create the experiment runner script for systematic comparisons
""")
    else:
        print("\n" + "=" * 70)
        print(" ⚠️ SOME TESTS FAILED")
        print("=" * 70)
        print("\nFix the failing components before starting experiments.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)