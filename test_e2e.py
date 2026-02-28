"""
End-to-End Integration Test
============================

Verifies the full pipeline: data -> features -> env -> regime -> buffer -> agent -> metrics.
Uses real cached training data and a short SAC training run.

Run: python test_e2e.py

Author: Prince Alexander John
Date: February 2026
"""

import sys
import os
import time
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# Test helpers (same pattern as test_all_fixes.py)
# ============================================================================

PASSED = 0
FAILED = 0
ERRORS = []


def print_header(title: str):
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def test(name: str, condition: bool, detail: str = ""):
    global PASSED, FAILED, ERRORS
    if condition:
        PASSED += 1
        print(f"  PASS: {name}")
    else:
        FAILED += 1
        ERRORS.append(name)
        print(f"  FAIL: {name}")
    if detail:
        print(f"        {detail}")


# ============================================================================
# Phase 1: Data Loading
# ============================================================================

def test_data_loading():
    print_header("Phase 1: Data Loading")

    from src.data.downloader import download_ohlcv_data

    tickers = ['SPY', 'QQQ', 'TLT', 'GLD', 'BIL']
    price_data = download_ohlcv_data(
        tickers=tickers,
        start_date='2010-01-01',
        end_date='2019-12-31',
    )

    test("Returns a dict", isinstance(price_data, dict))
    test("All 5 tickers present", len(price_data) == 5,
         f"Got {len(price_data)} tickers: {list(price_data.keys())}")

    all_have_rows = all(len(df) > 2000 for df in price_data.values())
    min_rows = min(len(df) for df in price_data.values())
    test("Each ticker has >2000 rows", all_have_rows,
         f"Min rows: {min_rows}")

    spy_cols = price_data['SPY'].columns.tolist()
    has_close = any(c in spy_cols for c in ['Close', 'Adj Close', 'close'])
    test("Price data has Close column", has_close,
         f"Columns: {spy_cols[:6]}")

    return price_data


# ============================================================================
# Phase 2: Feature Engineering
# ============================================================================

def test_feature_engineering(price_data):
    print_header("Phase 2: Feature Engineering")

    from src.data.features import calculate_all_features

    feature_data = calculate_all_features(price_data)

    test("Returns a dict", isinstance(feature_data, dict))
    test("All 5 tickers present", len(feature_data) == 5,
         f"Got {len(feature_data)} tickers")

    # Check feature count
    qqq_cols = feature_data['QQQ'].shape[1] if len(feature_data) > 0 else 0
    test("15+ features per ticker", qqq_cols >= 15,
         f"QQQ has {qqq_cols} features")

    # Check no infinities in cleaned data
    has_inf = False
    for ticker, df in feature_data.items():
        cleaned = df.dropna()
        if len(cleaned) > 0 and np.isinf(cleaned.values).any():
            has_inf = True
            break
    test("No infinite values", not has_inf)

    return feature_data


# ============================================================================
# Phase 3: Environment
# ============================================================================

def test_environment(price_data, feature_data):
    print_header("Phase 3: Environment")

    from src.environment.trading_env import TradingEnvironment

    env = TradingEnvironment(
        price_data=price_data,
        feature_data=feature_data,
        episode_length=50,
        random_start=True,
    )

    # Test reset
    obs, info = env.reset(seed=42)
    test("reset() returns observation", isinstance(obs, np.ndarray),
         f"Shape: {obs.shape}")
    test("reset() returns info dict", isinstance(info, dict))
    test("Observation matches obs space", obs.shape == env.observation_space.shape,
         f"obs={obs.shape}, space={env.observation_space.shape}")

    # Test step
    action = env.action_space.sample()
    result = env.step(action)
    test("step() returns 5-tuple", len(result) == 5)

    next_obs, reward, terminated, truncated, step_info = result
    test("step() reward is float", isinstance(reward, (float, np.floating)),
         f"reward={reward:.6f}")
    test("step() terminated is bool", isinstance(terminated, (bool, np.bool_)))
    test("step() truncated is bool", isinstance(truncated, (bool, np.bool_)))

    # Run a full episode
    obs, _ = env.reset(seed=42)
    total_reward = 0
    for _ in range(50):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break
    test("Full episode completes", True,
         f"Total reward: {total_reward:.4f}")

    return env


# ============================================================================
# Phase 4: Regime Detection
# ============================================================================

def test_regime_detection(price_data):
    print_header("Phase 4: Regime Detection")

    from src.regime import UnifiedRegimeDetector, MarketRegime

    detector = UnifiedRegimeDetector(method='rule_based')

    # Detect regimes
    regimes = detector.detect(price_data)
    test("detect() returns Series", isinstance(regimes, pd.Series),
         f"Length: {len(regimes)}")

    # Check diversity
    unique_regimes = set()
    for r in regimes:
        if hasattr(r, 'value'):
            unique_regimes.add(r.value)
        else:
            unique_regimes.add(str(r))
    test("At least 2 distinct regimes", len(unique_regimes) >= 2,
         f"Found: {unique_regimes}")

    # Test detect_current
    current = detector.detect_current(price_data)
    test("detect_current() returns string", isinstance(current, str),
         f"Current regime: {current}")
    test("Current regime is valid", current in ('bull', 'bear', 'sideways'))

    # Test statistics
    stats = detector.get_statistics(price_data)
    test("get_statistics() returns dict", isinstance(stats, dict))
    test("Stats has counts", 'counts' in stats,
         f"Counts: {stats.get('counts', 'N/A')}")

    return detector, regimes


# ============================================================================
# Phase 5: Replay Buffer
# ============================================================================

def test_replay_buffer(regimes, obs_dim, action_dim):
    print_header("Phase 5: Replay Buffer")

    from src.replay import RegimeAwareReplayBuffer, MarketRegime

    buffer = RegimeAwareReplayBuffer(
        buffer_size=10000,
        strategy='balanced',
        random_seed=42,
    )

    # Add 200 experiences with regime labels from detection
    # Start from index 250 to skip the SMA warm-up period (first ~200 days are all SIDEWAYS)
    start_idx = min(250, max(0, len(regimes) - 200))
    n_add = min(200, len(regimes) - start_idx)
    for i in range(n_add):
        state = np.random.randn(obs_dim).astype(np.float32)
        action = np.random.randn(action_dim).astype(np.float32)
        reward = np.random.randn()
        next_state = np.random.randn(obs_dim).astype(np.float32)
        done = (i % 50 == 49)
        regime = regimes.iloc[start_idx + i]
        buffer.add(state, action, reward, next_state, done, regime)

    test("Buffer has correct size", len(buffer) == n_add,
         f"Size: {len(buffer)}")

    # Check regime distribution
    regime_counts = buffer.get_regime_counts()
    non_empty = sum(1 for c in regime_counts.values() if c > 0)
    test("Multiple regimes stored", non_empty >= 2,
         f"Regime counts: {regime_counts}")

    # Sample a batch
    batch = buffer.sample(batch_size=30)
    test("Sample returns dict", isinstance(batch, dict))

    expected_keys = {'states', 'actions', 'rewards', 'next_states', 'dones'}
    actual_keys = set(batch.keys())
    test("Sample has required keys", expected_keys.issubset(actual_keys),
         f"Keys: {actual_keys}")

    test("Sample batch size correct", len(batch['states']) == 30,
         f"Got {len(batch['states'])} samples")

    return buffer


# ============================================================================
# Phase 6: SAC Training
# ============================================================================

def test_sac_training(price_data, feature_data, detector):
    print_header("Phase 6: SAC Training")

    from src.environment.trading_env import TradingEnvironment
    from src.agents.sac_agent import SACAgent, SACConfig
    from src.agents.base_agent import TrainingConfig, TrainingMetrics
    from src.replay import RegimeAwareReplayBuffer

    # Create fresh env and buffer for training
    train_env = TradingEnvironment(
        price_data=price_data,
        feature_data=feature_data,
        episode_length=50,
        random_start=True,
    )

    buffer = RegimeAwareReplayBuffer(
        buffer_size=10000,
        strategy='balanced',
        random_seed=42,
    )

    sac_config = SACConfig(
        learning_rate=3e-4,
        buffer_size=10000,
        batch_size=64,
        learning_starts=500,
        use_regime_aware_replay=True,
        replay_strategy='balanced',
    )

    # Create agent
    agent = SACAgent(
        env=train_env,
        name="e2e_test_sac",
        sac_config=sac_config,
        regime_detector=detector,
        replay_buffer=buffer,
        verbose=0,
    )
    test("SACAgent created", agent is not None)

    # Train
    training_config = TrainingConfig(
        total_timesteps=5000,
        tensorboard_log=None,
        early_stopping=False,
    )

    print("  Training SAC for 5000 timesteps (this may take 30-120s)...")
    start_time = time.time()

    metrics = agent.train(training_config=training_config)

    elapsed = time.time() - start_time
    test("Training completed", True, f"Took {elapsed:.1f}s")
    test("Training under 5 minutes", elapsed < 300,
         f"Elapsed: {elapsed:.1f}s")

    # Verify training happened
    test("Metrics returned", isinstance(metrics, TrainingMetrics))
    test("Timesteps recorded", agent.total_timesteps == 5000,
         f"Timesteps: {agent.total_timesteps}")

    # Test prediction
    obs, _ = train_env.reset(seed=99)
    action, info = agent.predict(obs, deterministic=True)
    test("predict() returns action", action is not None)
    test("Action shape is (5,)", action.shape == (5,),
         f"Shape: {action.shape}")
    test("Action values finite", np.all(np.isfinite(action)),
         f"Action: {action}")

    return agent, train_env


# ============================================================================
# Phase 7: Evaluation & Metrics
# ============================================================================

def test_evaluation(agent, env):
    print_header("Phase 7: Evaluation & Metrics")

    from src.utils.metrics import calculate_all_metrics

    # Run a manual episode collecting returns
    obs, _ = env.reset(seed=123)
    episode_rewards = []
    portfolio_values = []

    for step in range(50):
        action, _ = agent.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        episode_rewards.append(reward)
        if 'portfolio_value' in info:
            portfolio_values.append(info['portfolio_value'])
        if terminated or truncated:
            break

    test("Episode produced rewards", len(episode_rewards) > 0,
         f"Steps: {len(episode_rewards)}")

    total_reward = sum(episode_rewards)
    test("Total reward is finite", np.isfinite(total_reward),
         f"Total reward: {total_reward:.6f}")

    # Calculate metrics on episode returns
    returns = np.array(episode_rewards)
    if len(returns) > 1:
        metrics = calculate_all_metrics(returns)
        test("Metrics computed", isinstance(metrics, dict),
             f"Keys: {list(metrics.keys())[:5]}")

        if 'sharpe_ratio' in metrics:
            sr = metrics['sharpe_ratio']
            test("Sharpe ratio is finite", np.isfinite(sr),
                 f"Sharpe: {sr:.4f}")

        if 'max_drawdown' in metrics:
            mdd = metrics['max_drawdown']
            test("Max drawdown is finite", np.isfinite(mdd),
                 f"MDD: {mdd:.4f}")
    else:
        test("Metrics computed", False, "Not enough returns for metrics")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print(" END-TO-END INTEGRATION TEST")
    print(" Adaptive Portfolio RL - Full Pipeline Verification")
    print("=" * 70)

    overall_start = time.time()
    skip = False

    # Phase 1: Data Loading
    try:
        price_data = test_data_loading()
        if price_data is None or len(price_data) == 0:
            skip = True
            print("\n  CRITICAL: Data loading failed. Skipping remaining phases.")
    except Exception as e:
        price_data = None
        skip = True
        test("Data loading", False, f"Exception: {e}")

    # Phase 2: Feature Engineering
    feature_data = None
    if not skip:
        try:
            feature_data = test_feature_engineering(price_data)
            if feature_data is None or len(feature_data) == 0:
                skip = True
                print("\n  CRITICAL: Feature engineering failed. Skipping remaining phases.")
        except Exception as e:
            skip = True
            test("Feature engineering", False, f"Exception: {e}")

    # Phase 3: Environment
    env = None
    if not skip:
        try:
            env = test_environment(price_data, feature_data)
        except Exception as e:
            skip = True
            test("Environment creation", False, f"Exception: {e}")

    # Phase 4: Regime Detection
    detector = None
    regimes = None
    if not skip:
        try:
            detector, regimes = test_regime_detection(price_data)
        except Exception as e:
            test("Regime detection", False, f"Exception: {e}")

    # Phase 5: Replay Buffer
    if not skip and regimes is not None and env is not None:
        try:
            obs_dim = env.observation_space.shape[0]
            action_dim = env.action_space.shape[0]
            test_replay_buffer(regimes, obs_dim, action_dim)
        except Exception as e:
            test("Replay buffer", False, f"Exception: {e}")

    # Phase 6: SAC Training
    agent = None
    train_env = None
    if not skip and detector is not None:
        try:
            agent, train_env = test_sac_training(price_data, feature_data, detector)
        except Exception as e:
            test("SAC training", False, f"Exception: {e}")
            import traceback
            traceback.print_exc()

    # Phase 7: Evaluation
    if agent is not None and train_env is not None:
        try:
            test_evaluation(agent, train_env)
        except Exception as e:
            test("Evaluation", False, f"Exception: {e}")

    # Summary
    total_time = time.time() - overall_start
    print("\n" + "=" * 70)
    print(" SUMMARY")
    print("=" * 70)
    print(f"\n  Results: {PASSED}/{PASSED + FAILED} tests passed")
    print(f"  Time: {total_time:.1f} seconds")

    if ERRORS:
        print(f"\n  Failed tests:")
        for err in ERRORS:
            print(f"    - {err}")

    if FAILED == 0:
        print(f"\n  ALL TESTS PASSED!")
    else:
        print(f"\n  {FAILED} test(s) FAILED")

    print("=" * 70 + "\n")
    sys.exit(0 if FAILED == 0 else 1)
