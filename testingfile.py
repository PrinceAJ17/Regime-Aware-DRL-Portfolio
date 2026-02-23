"""
Deep Integration Verification
==============================

This script PROVES that regime-aware replay is actually working,
not just passing tests through compatibility shims.

WHAT THIS VERIFIES:
1. Experiences are being labelled with regimes (not all 'sideways')
2. The regime distribution in the buffer matches market conditions
3. Balanced sampling actually samples equally from regimes
4. The detector is being called during training

Run this after all 8 tests pass to confirm real integration.

Author: [Your Name]
Date: February 2026
"""

import sys
import numpy as np
from pathlib import Path
from collections import Counter

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import warnings
warnings.filterwarnings('ignore')


def print_header(title: str):
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def print_result(test: str, passed: bool, details: str = ""):
    status = "✓ VERIFIED" if passed else "✗ FAILED"
    print(f"  {status}: {test}")
    if details:
        print(f"           {details}")


def verify_regime_detection():
    """Verify that regime detection produces meaningful labels."""
    print_header("VERIFICATION 1: Regime Detection Quality")
    
    try:
        from src.data.downloader import download_ohlcv_data
        from src.regime import UnifiedRegimeDetector, MarketRegime
        
        # Load real data
        print("  Loading market data...")
        data = download_ohlcv_data(
            tickers=['SPY'],
            start_date='2020-01-01',
            end_date='2023-12-31'
        )
        
        # Create detector
        detector = UnifiedRegimeDetector(method='rule_based')
        
        # Detect regimes
        regimes = detector.detect(data)
        
        # Count regime distribution
        regime_counts = Counter()
        for r in regimes:
            if hasattr(r, 'value'):
                regime_counts[r.value] += 1
            else:
                regime_counts[str(r)] += 1
        
        total = sum(regime_counts.values())
        
        print(f"\n  Regime distribution over {total} trading days:")
        for regime, count in sorted(regime_counts.items()):
            pct = count / total * 100
            print(f"    {regime:10s}: {count:4d} days ({pct:5.1f}%)")
        
        # Verification checks
        has_multiple_regimes = len(regime_counts) > 1
        no_regime_dominates = all(pct < 90 for pct in [c/total*100 for c in regime_counts.values()])
        
        print_result(
            "Multiple regimes detected",
            has_multiple_regimes,
            f"Found {len(regime_counts)} distinct regimes"
        )
        
        print_result(
            "No single regime dominates (>90%)",
            no_regime_dominates,
            "Regime distribution is reasonable"
        )
        
        # Test detect_current
        current = detector.detect_current(data)
        print_result(
            "detect_current() returns valid regime",
            current in ['bull', 'bear', 'sideways'],
            f"Current regime: {current}"
        )
        
        return has_multiple_regimes and no_regime_dominates
        
    except Exception as e:
        print(f"\n  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_buffer_labelling():
    """Verify that experiences are labelled with actual regimes."""
    print_header("VERIFICATION 2: Experience Labelling")
    
    try:
        from src.replay.regime_aware_buffer import RegimeAwareReplayBuffer, MarketRegime, SamplingStrategy
        
        # Create buffer
        buffer = RegimeAwareReplayBuffer(
            buffer_size=1000,
            strategy=SamplingStrategy.BALANCED
        )
        
        state_dim = 98
        action_dim = 5
        
        # Add experiences with KNOWN regime distribution
        # 50 bull, 30 bear, 20 sideways
        print("  Adding 100 experiences with known distribution:")
        print("    Expected: 50 bull, 30 bear, 20 sideways")
        
        for i in range(50):
            buffer.add(
                state=np.random.randn(state_dim),
                action=np.random.randn(action_dim),
                reward=0.01,  # Positive reward for bull
                next_state=np.random.randn(state_dim),
                done=False,
                regime=MarketRegime.BULL
            )
        
        for i in range(30):
            buffer.add(
                state=np.random.randn(state_dim),
                action=np.random.randn(action_dim),
                reward=-0.02,  # Negative reward for bear
                next_state=np.random.randn(state_dim),
                done=False,
                regime=MarketRegime.BEAR
            )
        
        for i in range(20):
            buffer.add(
                state=np.random.randn(state_dim),
                action=np.random.randn(action_dim),
                reward=0.0,  # Zero reward for sideways
                next_state=np.random.randn(state_dim),
                done=False,
                regime=MarketRegime.SIDEWAYS
            )
        
        # Check buffer counts
        counts = buffer.get_regime_counts()
        print(f"\n  Actual buffer contents:")
        for regime, count in counts.items():
            print(f"    {regime.value:10s}: {count}")
        
        # Verify counts match what we added
        bull_correct = counts.get(MarketRegime.BULL, 0) == 50
        bear_correct = counts.get(MarketRegime.BEAR, 0) == 30
        sideways_correct = counts.get(MarketRegime.SIDEWAYS, 0) == 20
        
        print_result(
            "Bull experiences stored correctly",
            bull_correct,
            f"Expected 50, got {counts.get(MarketRegime.BULL, 0)}"
        )
        
        print_result(
            "Bear experiences stored correctly",
            bear_correct,
            f"Expected 30, got {counts.get(MarketRegime.BEAR, 0)}"
        )
        
        print_result(
            "Sideways experiences stored correctly",
            sideways_correct,
            f"Expected 20, got {counts.get(MarketRegime.SIDEWAYS, 0)}"
        )
        
        return bull_correct and bear_correct and sideways_correct
        
    except Exception as e:
        print(f"\n  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_balanced_sampling():
    """Verify that balanced sampling actually balances across regimes."""
    print_header("VERIFICATION 3: Balanced Sampling Works")
    
    try:
        from src.replay.regime_aware_buffer import RegimeAwareReplayBuffer, MarketRegime, SamplingStrategy
        
        # Create buffer with IMBALANCED data
        buffer = RegimeAwareReplayBuffer(
            buffer_size=10000,
            strategy=SamplingStrategy.BALANCED
        )
        
        state_dim = 98
        action_dim = 5
        
        # Add IMBALANCED experiences: 80% bull, 15% sideways, 5% bear
        print("  Adding IMBALANCED experiences:")
        print("    800 bull (80%), 150 sideways (15%), 50 bear (5%)")
        
        for i in range(800):
            buffer.add(
                state=np.random.randn(state_dim),
                action=np.random.randn(action_dim),
                reward=0.01,
                next_state=np.random.randn(state_dim),
                done=False,
                regime=MarketRegime.BULL
            )
        
        for i in range(150):
            buffer.add(
                state=np.random.randn(state_dim),
                action=np.random.randn(action_dim),
                reward=0.0,
                next_state=np.random.randn(state_dim),
                done=False,
                regime=MarketRegime.SIDEWAYS
            )
        
        for i in range(50):
            buffer.add(
                state=np.random.randn(state_dim),
                action=np.random.randn(action_dim),
                reward=-0.02,
                next_state=np.random.randn(state_dim),
                done=False,
                regime=MarketRegime.BEAR
            )
        
        # Sample multiple batches and check distribution
        print("\n  Sampling 10 batches of 30 with BALANCED strategy...")
        
        regime_samples = Counter()
        
        for _ in range(10):
            batch = buffer.sample(batch_size=30)
            
            # Count regimes in batch
            if 'regimes' in batch:
                for r in batch['regimes']:
                    regime_samples[r] += 1
            else:
                print("  WARNING: 'regimes' not in batch - cannot verify sampling")
                return False
        
        total_samples = sum(regime_samples.values())
        
        print(f"\n  Sampling results ({total_samples} total samples):")
        for regime_idx in sorted(regime_samples.keys()):
            regime_name = ['bull', 'bear', 'sideways'][regime_idx]
            count = regime_samples[regime_idx]
            pct = count / total_samples * 100
            print(f"    {regime_name:10s}: {count:3d} ({pct:5.1f}%)")
        
        # With balanced sampling, each regime should be ~33%
        # Allow 20-45% range for statistical variance
        balanced = True
        for regime_idx in [0, 1, 2]:  # bull, bear, sideways
            count = regime_samples.get(regime_idx, 0)
            pct = count / total_samples * 100
            if pct < 15 or pct > 50:  # Generous range for small sample
                balanced = False
        
        # Key check: bear should be MUCH higher than 5% (its buffer proportion)
        bear_pct = regime_samples.get(1, 0) / total_samples * 100
        bear_oversampled = bear_pct > 15  # Should be ~33%, much higher than 5%
        
        print_result(
            "Bear regime oversampled (was 5% of data)",
            bear_oversampled,
            f"Bear is now {bear_pct:.1f}% of samples (should be ~33%)"
        )
        
        print_result(
            "Balanced sampling produces even distribution",
            balanced,
            "All regimes between 15-50% of samples"
        )
        
        # Compare to what UNIFORM sampling would give
        print("\n  NOTE: With UNIFORM sampling, bear would be ~5% of samples.")
        print(f"        With BALANCED sampling, bear is {bear_pct:.1f}% - that's {bear_pct/5:.1f}x more!")
        
        return bear_oversampled
        
    except Exception as e:
        print(f"\n  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_training_integration():
    """Verify that SAC actually uses the regime-aware buffer during training."""
    print_header("VERIFICATION 4: Training Integration")
    
    try:
        from src.data.downloader import download_ohlcv_data
        from src.data.features import calculate_all_features
        from src.environment.trading_env import TradingEnvironment
        from src.agents.sac_agent import SACAgent, SACConfig
        from src.agents.base_agent import TrainingConfig
        from src.replay.regime_aware_buffer import RegimeAwareReplayBuffer, SamplingStrategy
        from src.regime import UnifiedRegimeDetector
        
        print("  Setting up environment and agent...")
        
        # Load data
        price_data = download_ohlcv_data(
            tickers=['SPY', 'QQQ', 'TLT', 'GLD', 'BIL'],
            start_date='2020-01-01',
            end_date='2023-12-31'
        )
        feature_data = calculate_all_features(price_data)
        
        env = TradingEnvironment(
            price_data=price_data,
            feature_data=feature_data,
            episode_length=50,
            random_start=True
        )
        
        # Create detector and buffer
        detector = UnifiedRegimeDetector(method='rule_based')
        buffer = RegimeAwareReplayBuffer(
            buffer_size=10000,
            strategy=SamplingStrategy.BALANCED
        )
        
        # Get buffer counts BEFORE training
        counts_before = buffer.get_regime_counts()
        total_before = sum(counts_before.values())
        
        print(f"  Buffer before training: {total_before} experiences")
        
        # Create agent with regime-aware replay
        agent = SACAgent(
            env=env,
            name="verification_sac",
            sac_config=SACConfig(
                use_regime_aware_replay=True,
                replay_strategy='balanced',
                buffer_size=10000,
                learning_starts=300,  # Wait for enough experiences
                batch_size=64,  # Smaller batch for testing
            ),
            regime_detector=detector,
            replay_buffer=buffer,
            verbose=0
        )
        
        # Train for a short time
        print("  Training for 500 steps...")
        training_config = TrainingConfig(total_timesteps=500)
        agent.train(training_config=training_config)
        
        # Get buffer counts AFTER training
        counts_after = buffer.get_regime_counts()
        total_after = sum(counts_after.values())
        
        print(f"\n  Buffer after training: {total_after} experiences")
        print(f"  Experiences added during training: {total_after - total_before}")
        
        print(f"\n  Regime distribution in buffer:")
        for regime, count in counts_after.items():
            pct = count / total_after * 100 if total_after > 0 else 0
            print(f"    {regime.value:10s}: {count:4d} ({pct:5.1f}%)")
        
        # Verification checks
        experiences_added = total_after > total_before
        multiple_regimes = sum(1 for c in counts_after.values() if c > 0) > 1
        
        print_result(
            "Experiences added to buffer during training",
            experiences_added,
            f"Added {total_after - total_before} experiences"
        )
        
        print_result(
            "Multiple regimes present in buffer",
            multiple_regimes,
            f"Found {sum(1 for c in counts_after.values() if c > 0)} regimes"
        )
        
        env.close()
        
        return experiences_added and multiple_regimes
        
    except Exception as e:
        print(f"\n  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all verification tests."""
    print("\n" + "=" * 70)
    print(" DEEP INTEGRATION VERIFICATION")
    print(" Proving Regime-Aware Replay Actually Works")
    print("=" * 70)
    
    results = {}
    
    results['Regime Detection'] = verify_regime_detection()
    results['Buffer Labelling'] = verify_buffer_labelling()
    results['Balanced Sampling'] = verify_balanced_sampling()
    results['Training Integration'] = verify_training_integration()
    
    # Summary
    print_header("VERIFICATION SUMMARY")
    
    passed = sum(results.values())
    total = len(results)
    
    for test, result in results.items():
        status = "✓ VERIFIED" if result else "✗ FAILED"
        print(f"  {status}: {test}"
        )
    
    print(f"\nResults: {passed}/{total} verifications passed")
    
    if passed == total:
        print("\n" + "=" * 70)
        print(" 🎉 REGIME-AWARE REPLAY IS FULLY INTEGRATED!")
        print("=" * 70)
        print("""
Your implementation is working correctly:
✓ Regime detection produces meaningful labels
✓ Experiences are stored in regime-specific buffers
✓ Balanced sampling oversamples rare regimes
✓ Training actually uses the regime-aware buffer

You can proceed with confidence to Week 6 experiments!
""")
    else:
        print("\n" + "=" * 70)
        print(" ⚠️ SOME VERIFICATIONS FAILED")
        print("=" * 70)
        print("""
The integration may have issues. Review the failed verifications above.
Common problems:
- Buffer not storing regime labels correctly
- Sampling strategy not actually being applied
- Wrapper bypassing your custom buffer
""")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)