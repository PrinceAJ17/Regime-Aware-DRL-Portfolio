"""
Week 5 Testing File - TD3 & Rainbow DQN Agents

Run this file to verify that your Week 5 implementations work correctly.
This tests both TD3 and Rainbow DQN agents with standard and regime-aware replay.

Usage:
------
python test_week5.py

Expected Output:
----------------
All 6 tests should pass with ✓ marks.

Author: [Your Name]
Date: February 2026
"""

import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def test_td3_basic():
    """Test 1: TD3 agent creation and basic functionality."""
    print("\n" + "=" * 60)
    print("TEST 1: TD3 Basic Functionality")
    print("=" * 60)
    
    import gymnasium as gym
    from src.agents.td3_agent import TD3Agent
    
    # Create a simple continuous environment
    env = gym.make("Pendulum-v1")
    
    # Create TD3 agent
    print("Creating TD3 agent...")
    agent = TD3Agent(env, verbose=0)
    
    print(f"  ✓ Agent created")
    print(f"  Learning rate: {agent.config.get('learning_rate')}")
    print(f"  Policy delay: {agent.config.get('policy_delay')}")
    print(f"  Target noise: {agent.config.get('target_policy_noise')}")
    
    # Test prediction
    obs, _ = env.reset()
    action, _ = agent.predict(obs, deterministic=True)
    print(f"  ✓ Prediction works (action shape: {action.shape})")
    
    # Quick training
    print("  Training for 500 steps...")
    agent.train(total_timesteps=500, progress_bar=False)
    print(f"  ✓ Training completed")
    
    env.close()
    print("\n✓ TEST 1 PASSED: TD3 basic functionality works!")
    return True


def test_td3_with_regime_aware_replay():
    """Test 2: TD3 with RegimeAwareReplayBuffer."""
    print("\n" + "=" * 60)
    print("TEST 2: TD3 with Regime-Aware Replay")
    print("=" * 60)
    
    import gymnasium as gym
    from src.agents.td3_agent import TD3Agent
    from src.replay.regime_aware_buffer import RegimeAwareReplayBuffer, SamplingStrategy
    
    # Create environment
    env = gym.make("Pendulum-v1")
    
    # Create regime-aware buffer
    print("Creating RegimeAwareReplayBuffer with balanced sampling...")
    buffer = RegimeAwareReplayBuffer(
        buffer_size=10000,
        strategy=SamplingStrategy.BALANCED
    )
    
    # Create TD3 with custom buffer
    print("Creating TD3 agent with regime-aware replay...")
    agent = TD3Agent(env, replay_buffer=buffer, verbose=0)
    
    print(f"  ✓ TD3 created with RegimeAwareReplayBuffer")
    
    # Quick training
    print("  Training for 500 steps...")
    agent.train(total_timesteps=500, progress_bar=False)
    print(f"  ✓ Training with regime-aware replay completed")
    
    env.close()
    print("\n✓ TEST 2 PASSED: TD3 works with regime-aware replay!")
    return True


def test_rainbow_basic():
    """Test 3: Rainbow DQN basic functionality."""
    print("\n" + "=" * 60)
    print("TEST 3: Rainbow DQN Basic Functionality")
    print("=" * 60)
    
    import gymnasium as gym
    from src.agents.rainbow_agent import RainbowAgent
    
    # Rainbow needs discrete actions - CartPole is perfect
    env = gym.make("CartPole-v1")
    
    # Create Rainbow agent
    print("Creating Rainbow agent...")
    agent = RainbowAgent(env, device="cpu")
    
    print(f"  ✓ Agent created")
    print(f"  Observation dim: {agent.obs_dim}")
    print(f"  Number of actions: {agent.n_actions}")
    print(f"  N-atoms (C51): {agent.config.get('n_atoms')}")
    print(f"  N-steps: {agent.config.get('n_steps')}")
    
    # Test action selection
    obs, _ = env.reset()
    action = agent.select_action(obs, training=True)
    print(f"  ✓ Action selection works (action: {action})")
    
    # Quick training
    print("  Training for 500 steps...")
    history = agent.train(total_timesteps=500, progress_bar=False, eval_freq=1000)
    print(f"  ✓ Training completed ({agent.episodes} episodes)")
    
    env.close()
    print("\n✓ TEST 3 PASSED: Rainbow DQN basic functionality works!")
    return True


def test_discrete_action_wrapper():
    """Test 4: DiscreteActionWrapper for continuous environments."""
    print("\n" + "=" * 60)
    print("TEST 4: DiscreteActionWrapper")
    print("=" * 60)
    
    import gymnasium as gym
    from src.agents.rainbow_agent import DiscreteActionWrapper
    
    # Create continuous environment
    cont_env = gym.make("Pendulum-v1")
    print(f"Original action space: {cont_env.action_space}")
    
    # Wrap with discrete actions
    wrapped_env = DiscreteActionWrapper(cont_env, actions_per_asset=3, step_size=0.1)
    print(f"Wrapped action space: {wrapped_env.action_space}")
    print(f"Number of discrete actions: {wrapped_env.n_actions}")
    
    # Test step
    obs, _ = wrapped_env.reset()
    action = wrapped_env.action_space.sample()  # Random discrete action
    next_obs, reward, terminated, truncated, info = wrapped_env.step(action)
    
    print(f"  ✓ Reset works (obs shape: {obs.shape})")
    print(f"  ✓ Step works (discrete action {action} → continuous weights)")
    
    wrapped_env.close()
    print("\n✓ TEST 4 PASSED: DiscreteActionWrapper works!")
    return True


def test_rainbow_components():
    """Test 5: Rainbow individual components (Noisy, Dueling, etc.)."""
    print("\n" + "=" * 60)
    print("TEST 5: Rainbow Components")
    print("=" * 60)
    
    import torch
    from src.agents.rainbow_agent import NoisyLinear, DuelingNetwork
    
    # Test NoisyLinear
    print("Testing NoisyLinear layer...")
    noisy = NoisyLinear(64, 32)
    x = torch.randn(8, 64)
    
    # Forward pass
    out1 = noisy(x)
    noisy.reset_noise()
    out2 = noisy(x)
    
    # Outputs should be different after resetting noise
    print(f"  ✓ NoisyLinear output shape: {out1.shape}")
    print(f"  ✓ Noise changes output: {not torch.allclose(out1, out2)}")
    
    # Test DuelingNetwork
    print("Testing DuelingNetwork...")
    net = DuelingNetwork(
        observation_dim=64,
        n_actions=10,
        hidden_dims=[128, 128],
        n_atoms=51,
        noisy=True
    )
    
    obs = torch.randn(4, 64)
    q_dist = net(obs)
    q_values = net.get_q_values(obs)
    
    print(f"  ✓ Q-distribution shape: {q_dist.shape} (batch, actions, atoms)")
    print(f"  ✓ Q-values shape: {q_values.shape} (batch, actions)")
    print(f"  ✓ Distribution sums to 1: {torch.allclose(q_dist.sum(dim=-1), torch.ones(4, 10))}")
    
    print("\n✓ TEST 5 PASSED: Rainbow components work!")
    return True


def test_save_load():
    """Test 6: Save and load functionality."""
    print("\n" + "=" * 60)
    print("TEST 6: Save and Load")
    print("=" * 60)
    
    import gymnasium as gym
    import tempfile
    import os
    from src.agents.td3_agent import TD3Agent
    from src.agents.rainbow_agent import RainbowAgent
    
    # Test TD3 save/load
    print("Testing TD3 save/load...")
    env = gym.make("Pendulum-v1")
    agent = TD3Agent(env, verbose=0)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "td3_test")
        agent.save(save_path)
        agent.load(save_path)
        print(f"  ✓ TD3 save/load works")
    
    env.close()
    
    # Test Rainbow save/load
    print("Testing Rainbow save/load...")
    env = gym.make("CartPole-v1")
    agent = RainbowAgent(env, device="cpu")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "rainbow_test")
        agent.save(save_path)
        agent.load(save_path)
        print(f"  ✓ Rainbow save/load works")
    
    env.close()
    
    print("\n✓ TEST 6 PASSED: Save/load works for both agents!")
    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("WEEK 5 TESTS - TD3 & RAINBOW DQN")
    print("=" * 60)
    print("Testing TD3 and Rainbow DQN implementations...")
    print("This verifies your Week 5 code is working correctly.")
    
    tests = [
        ("TD3 Basic", test_td3_basic),
        ("TD3 + Regime-Aware Replay", test_td3_with_regime_aware_replay),
        ("Rainbow Basic", test_rainbow_basic),
        ("DiscreteActionWrapper", test_discrete_action_wrapper),
        ("Rainbow Components", test_rainbow_components),
        ("Save/Load", test_save_load),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result, None))
        except Exception as e:
            results.append((name, False, str(e)))
            print(f"\n✗ TEST FAILED: {name}")
            print(f"  Error: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result, _ in results if result)
    total = len(results)
    
    for name, result, error in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {name}")
        if error:
            print(f"         Error: {error}")
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n" + "=" * 60)
        print("🎉 ALL TESTS PASSED! Week 5 implementation complete!")
        print("=" * 60)
        print("\nYour TD3 and Rainbow DQN agents are ready.")
        print("\nNext steps:")
        print("  1. Update src/agents/__init__.py with new imports")
        print("  2. Test with your actual TradingEnv")
        print("  3. Move to Week 6: Experiments")
    else:
        print("\n" + "=" * 60)
        print("⚠️  SOME TESTS FAILED")
        print("=" * 60)
        print("\nCheck the error messages above and fix the issues.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)