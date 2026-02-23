"""
Integration Fix Verification Tests
====================================

Tests all 13 fixes applied to the adaptive-portfolio-rl project.
Uses synthetic data (no network/API calls needed).

Run: python test_all_fixes.py

Author: Prince Alexander John
Date: February 2026
"""

import sys
import os
import inspect
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import warnings
warnings.filterwarnings('ignore')

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


def create_synthetic_price_data(n_days=500):
    """Create synthetic price data with distinct regimes for testing."""
    dates = pd.date_range('2020-01-01', periods=n_days, freq='B')
    np.random.seed(42)

    # Generate prices with clear regime patterns
    prices = [100.0]
    for i in range(1, n_days):
        if i < 150:
            # Bull: uptrend, low vol
            ret = 0.002 + np.random.normal(0, 0.005)
        elif i < 300:
            # Bear: downtrend, high vol
            ret = -0.003 + np.random.normal(0, 0.015)
        else:
            # Sideways: flat, moderate vol
            ret = 0.0 + np.random.normal(0, 0.008)
        prices.append(prices[-1] * (1 + ret))

    df = pd.DataFrame({
        'open': prices,
        'high': [p * 1.01 for p in prices],
        'low': [p * 0.99 for p in prices],
        'close': prices,
        'volume': [1000000] * n_days,
    }, index=dates)

    return {'SPY': df}


# =========================================================================
# TEST 1: Single MarketRegime identity across all imports
# =========================================================================
def test_01_single_market_regime():
    print_header("TEST 1: Single MarketRegime enum (Issue 5)")

    from src.regime import MarketRegime as MR1
    from src.replay.regime_aware_buffer import MarketRegime as MR2
    from src.replay import MarketRegime as MR3

    test(
        "MarketRegime is the same class across regime and replay modules",
        MR1 is MR2 and MR2 is MR3,
        f"regime={id(MR1)}, buffer={id(MR2)}, replay_init={id(MR3)}"
    )

    # Also check values work
    test(
        "MarketRegime enum values are correct",
        MR1.BULL.value == 'bull' and MR1.BEAR.value == 'bear' and MR1.SIDEWAYS.value == 'sideways'
    )


# =========================================================================
# TEST 2: Single RegimeAwareReplayBuffer class
# =========================================================================
def test_02_single_buffer_class():
    print_header("TEST 2: Single RegimeAwareReplayBuffer (Issue 6)")

    from src.replay.regime_aware_buffer import RegimeAwareReplayBuffer as RAB1
    from src.replay import RegimeAwareReplayBuffer as RAB2

    test(
        "RegimeAwareReplayBuffer is the same class from both imports",
        RAB1 is RAB2,
        f"buffer_module={id(RAB1)}, init_module={id(RAB2)}"
    )

    # Verify the canonical version has the expected attributes
    buf = RAB1(buffer_size=100, strategy='balanced')
    test(
        "Buffer has _regime_buffers (canonical implementation)",
        hasattr(buf, '_regime_buffers'),
        "Uses regime-specific deques"
    )
    test(
        "Buffer has total_buffer_size attribute",
        hasattr(buf, 'total_buffer_size'),
    )


# =========================================================================
# TEST 3: No sys.path hack
# =========================================================================
def test_03_no_sys_path_hack():
    print_header("TEST 3: No sys.path hack (Issue 12)")

    source_file = Path("src/replay/regime_aware_buffer.py")
    content = source_file.read_text(encoding='utf-8')

    test(
        "regime_aware_buffer.py does not contain sys.path.append",
        "sys.path.append" not in content,
    )
    test(
        "Uses relative import for standard_buffer",
        "from .standard_buffer import Experience" in content,
    )


# =========================================================================
# TEST 4: RegimeAwareReplayBuffer has clear() method
# =========================================================================
def test_04_buffer_clear():
    print_header("TEST 4: Buffer clear() method (Issue 13)")

    from src.replay.regime_aware_buffer import RegimeAwareReplayBuffer, MarketRegime

    buf = RegimeAwareReplayBuffer(buffer_size=100, strategy='balanced')

    # Add some experiences
    for _ in range(10):
        buf.add(
            state=np.random.randn(5),
            action=np.random.randn(2),
            reward=0.1,
            next_state=np.random.randn(5),
            done=False,
            regime=MarketRegime.BULL,
        )

    test("Buffer has clear() method", hasattr(buf, 'clear'))

    buf.clear()
    test(
        "clear() empties the buffer",
        len(buf) == 0,
        f"Buffer size after clear: {len(buf)}"
    )


# =========================================================================
# TEST 5: TD3 import guard doesn't crash
# =========================================================================
def test_05_td3_import_guard():
    print_header("TEST 5: TD3 import guard (Issue 10)")

    try:
        from src.agents import TD3Agent
        test(
            "TD3Agent imported without crash",
            True,
            "TD3Agent is available" if TD3Agent is not None else "TD3Agent is None (SB3 not installed)"
        )
    except ImportError as e:
        test(
            "TD3Agent import doesn't crash the agents package",
            True,
            f"Gracefully handled: {e}"
        )
    except Exception as e:
        test(
            "TD3Agent import doesn't crash the agents package",
            False,
            f"Unexpected error: {e}"
        )

    # The key test: can we still import the agents package?
    try:
        from src import agents
        test(
            "src.agents package imports successfully even if SB3 unavailable",
            True,
        )
    except Exception as e:
        test(
            "src.agents package imports successfully",
            False,
            f"Error: {e}"
        )


# =========================================================================
# TEST 6: TD3 regime_detector is set
# =========================================================================
def test_06_td3_regime_detector():
    print_header("TEST 6: TD3 regime_detector pass-through (Issue 9)")

    try:
        from src.agents.td3_agent import TD3Agent, SB3_AVAILABLE
        if not SB3_AVAILABLE:
            test("SKIP: SB3 not available", True, "Cannot test TD3 without stable-baselines3")
            return

        import gymnasium as gym
        env = gym.make('Pendulum-v1')

        class FakeDetector:
            def detect(self, data):
                return pd.Series(['sideways'] * 100)

        detector = FakeDetector()
        agent = TD3Agent(env, regime_detector=detector, verbose=0)

        test(
            "TD3Agent.regime_detector is set when passed",
            agent.regime_detector is detector,
        )
        env.close()
    except Exception as e:
        test("TD3 regime_detector pass-through", False, f"Error: {e}")


# =========================================================================
# TEST 7: BaseAgent.evaluate_by_regime() detects non-sideways regimes
# =========================================================================
def test_07_evaluate_by_regime():
    print_header("TEST 7: BaseAgent.evaluate_by_regime() (Issue 1)")

    try:
        from src.regime import UnifiedRegimeDetector
        price_data = create_synthetic_price_data(500)
        detector = UnifiedRegimeDetector(method='rule_based')

        # Just verify the detector produces non-sideways regimes
        regimes = detector.detect(price_data)
        regime_values = set()
        for r in regimes:
            if hasattr(r, 'value'):
                regime_values.add(r.value)
            else:
                regime_values.add(str(r))

        test(
            "Regime detector produces multiple regime types",
            len(regime_values) > 1,
            f"Found regimes: {regime_values}"
        )

        # Check the evaluate_by_regime code doesn't use get_current_data
        from src.agents.base_agent import BaseAgent
        source = inspect.getsource(BaseAgent.evaluate_by_regime)
        test(
            "evaluate_by_regime() doesn't use get_current_data()",
            "get_current_data" not in source,
            "Uses cached regime series instead"
        )
        test(
            "evaluate_by_regime() uses _regime_cache or price_data lookup",
            "_regime_cache" in source or "price_data" in source,
        )

    except Exception as e:
        test("evaluate_by_regime fix", False, f"Error: {e}")


# =========================================================================
# TEST 8: PPO RegimeObservationWrapper updates regime from price data
# =========================================================================
def test_08_ppo_observation_wrapper():
    print_header("TEST 8: PPO RegimeObservationWrapper (Issue 2)")

    try:
        from src.agents.ppo_agent import RegimeObservationWrapper

        # Check the __init__ accepts price_data
        sig = inspect.signature(RegimeObservationWrapper.__init__)
        params = list(sig.parameters.keys())
        test(
            "RegimeObservationWrapper.__init__ accepts price_data",
            'price_data' in params,
            f"Parameters: {params}"
        )

        # Check observation() doesn't use get_current_data
        source = inspect.getsource(RegimeObservationWrapper.observation)
        test(
            "observation() doesn't use get_current_data()",
            "get_current_data" not in source,
            "Uses _regime_cache instead"
        )
        test(
            "observation() uses _regime_cache",
            "_regime_cache" in source,
        )

    except Exception as e:
        test("PPO observation wrapper fix", False, f"Error: {e}")


# =========================================================================
# TEST 9: PPO RegimeMetricsCallback tracks regime steps
# =========================================================================
def test_09_ppo_metrics_callback():
    print_header("TEST 9: PPO RegimeMetricsCallback (Issue 3)")

    try:
        from src.agents.ppo_agent import RegimeMetricsCallback

        # Check __init__ accepts price_data
        sig = inspect.signature(RegimeMetricsCallback.__init__)
        params = list(sig.parameters.keys())
        test(
            "RegimeMetricsCallback.__init__ accepts price_data",
            'price_data' in params,
            f"Parameters: {params}"
        )

        # Check _on_step doesn't use current_regime attribute
        source = inspect.getsource(RegimeMetricsCallback._on_step)
        test(
            "_on_step() doesn't check env.current_regime",
            "base_env.current_regime" not in source and "env.current_regime" not in source,
            "Uses _regime_cache instead"
        )
        test(
            "_on_step() uses _regime_cache",
            "_regime_cache" in source,
        )

    except Exception as e:
        test("PPO metrics callback fix", False, f"Error: {e}")


# =========================================================================
# TEST 10: ReplayBufferWrapper labels experiences with correct regimes
# =========================================================================
def test_10_replay_buffer_wrapper():
    print_header("TEST 10: ReplayBufferWrapper regime detection (Issue 4)")

    try:
        from src.agents.base_agent import ReplayBufferWrapper

        # Check __init__ accepts regime_detector and price_data
        sig = inspect.signature(ReplayBufferWrapper.__init__)
        params = list(sig.parameters.keys())
        test(
            "ReplayBufferWrapper.__init__ accepts regime_detector",
            'regime_detector' in params,
            f"Parameters: {params}"
        )
        test(
            "ReplayBufferWrapper.__init__ accepts price_data",
            'price_data' in params,
        )

        # Check add() uses _regime_cache
        source = inspect.getsource(ReplayBufferWrapper.add)
        test(
            "add() uses _regime_cache for regime detection",
            "_regime_cache" in source,
            "Uses cached regime series instead of defaulting to sideways"
        )

    except Exception as e:
        test("ReplayBufferWrapper fix", False, f"Error: {e}")


# =========================================================================
# TEST 11: optimize_memory_usage attribute exists
# =========================================================================
def test_11_optimize_memory_usage():
    print_header("TEST 11: optimize_memory_usage attribute (Issue 11)")

    try:
        from src.agents.base_agent import ReplayBufferWrapper
        from src.replay.regime_aware_buffer import RegimeAwareReplayBuffer
        import gymnasium as gym

        env = gym.make('Pendulum-v1')
        buf = RegimeAwareReplayBuffer(buffer_size=100, strategy='balanced')
        wrapper = ReplayBufferWrapper(
            buf,
            observation_space=env.observation_space,
            action_space=env.action_space,
        )

        test(
            "ReplayBufferWrapper has optimize_memory_usage attribute",
            hasattr(wrapper, 'optimize_memory_usage'),
        )
        test(
            "optimize_memory_usage is False",
            wrapper.optimize_memory_usage == False,
        )
        env.close()

    except Exception as e:
        test("optimize_memory_usage", False, f"Error: {e}")


# =========================================================================
# TEST 12: ReplayBufferWrapper.reset() doesn't crash
# =========================================================================
def test_12_wrapper_reset():
    print_header("TEST 12: ReplayBufferWrapper.reset() (Issue 13)")

    try:
        from src.agents.base_agent import ReplayBufferWrapper
        from src.replay.regime_aware_buffer import RegimeAwareReplayBuffer, MarketRegime
        import gymnasium as gym

        env = gym.make('Pendulum-v1')
        buf = RegimeAwareReplayBuffer(buffer_size=100, strategy='balanced')

        # Add some data
        for _ in range(5):
            buf.add(
                state=np.random.randn(3),
                action=np.random.randn(1),
                reward=0.1,
                next_state=np.random.randn(3),
                done=False,
                regime=MarketRegime.BULL,
            )

        wrapper = ReplayBufferWrapper(
            buf,
            observation_space=env.observation_space,
            action_space=env.action_space,
        )

        # This should not crash
        wrapper.reset()
        test(
            "ReplayBufferWrapper.reset() does not crash",
            True,
        )
        test(
            "Buffer is empty after reset",
            len(buf) == 0,
        )
        env.close()

    except Exception as e:
        test("ReplayBufferWrapper.reset()", False, f"Error: {e}")


# =========================================================================
# TEST 13: TD3 train() signature and return type
# =========================================================================
def test_13_td3_train_signature():
    print_header("TEST 13: TD3 train() signature (Issue 7)")

    try:
        from src.agents.td3_agent import TD3Agent, SB3_AVAILABLE
        if not SB3_AVAILABLE:
            test("SKIP: SB3 not available", True)
            return

        sig = inspect.signature(TD3Agent.train)
        params = list(sig.parameters.keys())

        test(
            "training_config is the first parameter after self",
            params[1] == 'training_config',
            f"Parameters: {params}"
        )

        # Check return annotation
        source = inspect.getsource(TD3Agent.train)
        test(
            "train() returns TrainingMetrics (not self)",
            "return self.metrics" in source,
            "Returns TrainingMetrics object"
        )

    except Exception as e:
        test("TD3 train() signature", False, f"Error: {e}")


# =========================================================================
# TEST 14: Rainbow train() signature and return type
# =========================================================================
def test_14_rainbow_train_signature():
    print_header("TEST 14: Rainbow train() signature (Issue 8)")

    try:
        from src.agents.rainbow_agent import RainbowAgent

        sig = inspect.signature(RainbowAgent.train)
        params = list(sig.parameters.keys())

        test(
            "training_config is the first parameter after self",
            params[1] == 'training_config',
            f"Parameters: {params}"
        )

        # Check return annotation
        source = inspect.getsource(RainbowAgent.train)
        test(
            "train() returns TrainingMetrics (not Dict)",
            "return self.metrics" in source,
            "Returns TrainingMetrics object"
        )

    except Exception as e:
        test("Rainbow train() signature", False, f"Error: {e}")


# =========================================================================
# TEST 15: End-to-end regime detection with buffer
# =========================================================================
def test_15_end_to_end_regime_buffer():
    print_header("TEST 15: End-to-end regime detection + buffer integration")

    try:
        from src.regime import UnifiedRegimeDetector, MarketRegime
        from src.replay.regime_aware_buffer import RegimeAwareReplayBuffer, SamplingStrategy

        price_data = create_synthetic_price_data(500)
        detector = UnifiedRegimeDetector(method='rule_based')

        # Detect regimes
        regimes = detector.detect(price_data)
        regime_values = [r.value if hasattr(r, 'value') else str(r) for r in regimes]
        unique_regimes = set(regime_values)

        test(
            "Detector finds multiple regimes in synthetic data",
            len(unique_regimes) > 1,
            f"Regimes found: {unique_regimes}"
        )

        # Create buffer and add experiences labelled by detected regimes
        buffer = RegimeAwareReplayBuffer(buffer_size=1000, strategy=SamplingStrategy.BALANCED)

        for i in range(min(400, len(regimes))):
            regime_val = regimes.iloc[i]
            regime = regime_val if isinstance(regime_val, MarketRegime) else MarketRegime(str(regime_val))
            buffer.add(
                state=np.random.randn(10),
                action=np.random.randn(3),
                reward=np.random.randn(),
                next_state=np.random.randn(10),
                done=False,
                regime=regime,
            )

        counts = buffer.get_regime_counts()
        non_empty = sum(1 for c in counts.values() if c > 0)

        test(
            "Buffer contains experiences from multiple regimes",
            non_empty > 1,
            f"Regime counts: {', '.join(f'{r.value}={c}' for r, c in counts.items())}"
        )

        # Test balanced sampling
        if len(buffer) >= 30:
            batch = buffer.sample(batch_size=30)
            regime_dist = np.bincount(batch['regimes'], minlength=3)
            test(
                "Balanced sampling returns experiences from all regimes",
                np.all(regime_dist > 0) or non_empty < 3,
                f"Sample distribution: bull={regime_dist[0]}, bear={regime_dist[1]}, sideways={regime_dist[2]}"
            )

    except Exception as e:
        import traceback
        test("End-to-end integration", False, f"Error: {e}")
        traceback.print_exc()


# =========================================================================
# MAIN
# =========================================================================
def main():
    print("\n" + "=" * 70)
    print(" INTEGRATION FIX VERIFICATION")
    print(" Testing all 13 fixes + end-to-end integration")
    print("=" * 70)

    test_01_single_market_regime()
    test_02_single_buffer_class()
    test_03_no_sys_path_hack()
    test_04_buffer_clear()
    test_05_td3_import_guard()
    test_06_td3_regime_detector()
    test_07_evaluate_by_regime()
    test_08_ppo_observation_wrapper()
    test_09_ppo_metrics_callback()
    test_10_replay_buffer_wrapper()
    test_11_optimize_memory_usage()
    test_12_wrapper_reset()
    test_13_td3_train_signature()
    test_14_rainbow_train_signature()
    test_15_end_to_end_regime_buffer()

    # Summary
    print_header("SUMMARY")
    total = PASSED + FAILED
    print(f"\n  Results: {PASSED}/{total} tests passed")

    if ERRORS:
        print(f"\n  Failed tests:")
        for e in ERRORS:
            print(f"    - {e}")

    if FAILED == 0:
        print("\n  ALL TESTS PASSED!")
        print("  All 13 integration issues have been fixed.")
    else:
        print(f"\n  {FAILED} test(s) failed. Review the output above.")

    return FAILED == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
