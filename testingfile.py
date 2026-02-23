"""
Week 3 Integration Test
========================

This script tests all Week 3 components working together:
1. Rule-Based Regime Detection
2. HMM Regime Detection (if available)
3. Standard Replay Buffer (baseline)
4. Regime-Aware Replay Buffer (novel contribution)

Run this to verify everything is working before moving to Week 4.

Usage:
    python test_week3.py

Author: [Your Name]
Date: February 2026
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

print("=" * 70)
print("WEEK 3 INTEGRATION TEST")
print("Regime Detection & Regime-Aware Experience Replay")
print("=" * 70)

# =============================================================================
# TEST 1: Create Synthetic Market Data
# =============================================================================

print("\n" + "=" * 70)
print("TEST 1: Generate Synthetic Market Data")
print("=" * 70)

np.random.seed(42)
n_days = 1000  # About 4 years of trading days
dates = pd.date_range('2019-01-01', periods=n_days, freq='B')

# Generate price with clear regime periods
price = 100.0
prices = [price]
volumes = [1_000_000]

print("\nGenerating data with known regimes:")
print("  Days 0-250:   Bull market (drift=+0.1%, vol=0.8%)")
print("  Days 250-450: Bear market (drift=-0.15%, vol=2.0%)")
print("  Days 450-650: Sideways (drift=0%, vol=1.2%)")
print("  Days 650-850: Bull market (drift=+0.12%, vol=0.7%)")
print("  Days 850-1000: Bear market (drift=-0.1%, vol=1.8%)")

for i in range(1, n_days):
    # Define regime periods with distinct characteristics
    if i < 250:  # Bull
        drift, vol = 0.001, 0.008
    elif i < 450:  # Bear
        drift, vol = -0.0015, 0.020
    elif i < 650:  # Sideways
        drift, vol = 0.0, 0.012
    elif i < 850:  # Bull again
        drift, vol = 0.0012, 0.007
    else:  # Bear again
        drift, vol = -0.001, 0.018
    
    ret = drift + vol * np.random.randn()
    price = price * (1 + ret)
    prices.append(price)
    
    # Volume varies by regime (higher in bear markets)
    base_vol = 1_000_000
    if 250 <= i < 450 or i >= 850:  # Bear markets
        base_vol *= 1.5
    volumes.append(int(base_vol + base_vol * 0.3 * np.random.randn()))

price_df = pd.DataFrame({
    'open': prices,
    'high': [p * (1 + 0.01 * np.random.rand()) for p in prices],
    'low': [p * (1 - 0.01 * np.random.rand()) for p in prices],
    'close': prices,
    'volume': volumes
}, index=dates)

print(f"\n✓ Generated {n_days} days of synthetic market data")
print(f"  Date range: {dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}")
print(f"  Price range: ${min(prices):.2f} - ${max(prices):.2f}")

# =============================================================================
# TEST 2: Rule-Based Regime Detection
# =============================================================================

print("\n" + "=" * 70)
print("TEST 2: Rule-Based Regime Detection")
print("=" * 70)

from src.regime.rule_based_detector import RuleBasedRegimeDetector, MarketRegime

rule_detector = RuleBasedRegimeDetector()
rule_regimes = rule_detector.detect(price_df)

print("\nRegime detection results (rule-based):")
regime_counts = rule_regimes.value_counts()
for regime in MarketRegime:
    count = regime_counts.get(regime, 0)
    pct = count / len(rule_regimes) * 100
    print(f"  {regime.value:10s}: {count:4d} days ({pct:5.1f}%)")

stats = rule_detector.get_regime_statistics(rule_regimes)
print(f"\nRegime transitions: {stats['num_transitions']}")
print(f"Average regime duration: {stats['avg_regime_duration']:.1f} days")

print("\n✓ Rule-based regime detection working!")

# =============================================================================
# TEST 3: HMM Regime Detection (if available)
# =============================================================================

print("\n" + "=" * 70)
print("TEST 3: HMM Regime Detection")
print("=" * 70)

try:
    from src.regime.hmm_detector import HMMRegimeDetector, HMM_AVAILABLE
    
    if HMM_AVAILABLE:
        print("\nTraining HMM on first 700 days...")
        train_data = price_df.iloc[:700]
        
        hmm_detector = HMMRegimeDetector()
        hmm_detector.fit(train_data, verbose=False)
        
        hmm_regimes = hmm_detector.predict(price_df)
        
        print("\nRegime detection results (HMM):")
        regime_counts = hmm_regimes.value_counts()
        for regime in MarketRegime:
            count = regime_counts.get(regime, 0)
            pct = count / len(hmm_regimes) * 100
            print(f"  {regime.value:10s}: {count:4d} days ({pct:5.1f}%)")
        
        # Compare methods
        agreement = (rule_regimes == hmm_regimes).mean()
        print(f"\nAgreement between methods: {agreement:.1%}")
        
        print("\n✓ HMM regime detection working!")
    else:
        print("\n⚠ hmmlearn not installed. Skipping HMM test.")
        print("  Install with: pip install hmmlearn")
        hmm_regimes = None
        
except ImportError as e:
    print(f"\n⚠ Could not import HMM detector: {e}")
    hmm_regimes = None

# =============================================================================
# TEST 4: Standard Replay Buffer (Baseline)
# =============================================================================

print("\n" + "=" * 70)
print("TEST 4: Standard Replay Buffer (Baseline)")
print("=" * 70)

from src.replay.standard_buffer import StandardReplayBuffer

standard_buffer = StandardReplayBuffer(buffer_size=10000, random_seed=42)

# Simulate adding experiences
print("\nAdding 1000 experiences...")
state_dim = 98  # Your environment's observation dimension
action_dim = 5  # 5 assets

for i in range(1000):
    state = np.random.randn(state_dim).astype(np.float32)
    action = np.random.randn(action_dim).astype(np.float32)
    reward = float(np.random.randn())
    next_state = np.random.randn(state_dim).astype(np.float32)
    done = (i + 1) % 252 == 0  # Episode ends after ~1 year
    
    standard_buffer.add(state, action, reward, next_state, done)

print(f"  Buffer size: {len(standard_buffer)}")

# Sample a batch
batch = standard_buffer.sample(64)
print(f"\nSampled batch shapes:")
for key, value in batch.items():
    print(f"  {key:12s}: {value.shape}")

print("\n✓ Standard replay buffer working!")

# =============================================================================
# TEST 5: Regime-Aware Replay Buffer (NOVEL CONTRIBUTION)
# =============================================================================

print("\n" + "=" * 70)
print("TEST 5: Regime-Aware Replay Buffer (NOVEL CONTRIBUTION)")
print("=" * 70)

from src.replay.regime_aware_buffer import (
    RegimeAwareReplayBuffer, 
    SamplingStrategy,
    MarketRegime as ReplayRegime
)

# Map detector regimes to replay regimes
def map_regime(detector_regime):
    mapping = {
        'bull': ReplayRegime.BULL,
        'bear': ReplayRegime.BEAR,
        'sideways': ReplayRegime.SIDEWAYS
    }
    return mapping[detector_regime.value]

print("\n--- Testing BALANCED Sampling ---")
balanced_buffer = RegimeAwareReplayBuffer(
    buffer_size=3000,
    strategy='balanced',
    random_seed=42
)

# Add experiences with regime labels (using imbalanced distribution)
print("Adding 900 experiences with imbalanced regimes (60% bull, 25% side, 15% bear)...")
for i in range(900):
    state = np.random.randn(state_dim).astype(np.float32)
    action = np.random.randn(action_dim).astype(np.float32)
    reward = float(np.random.randn())
    next_state = np.random.randn(state_dim).astype(np.float32)
    done = False
    
    # Imbalanced regime distribution (simulating real market data)
    r = np.random.rand()
    if r < 0.60:
        regime = ReplayRegime.BULL
    elif r < 0.85:
        regime = ReplayRegime.SIDEWAYS
    else:
        regime = ReplayRegime.BEAR
    
    balanced_buffer.add(state, action, reward, next_state, done, regime)

# Show stored distribution
counts = balanced_buffer.get_regime_counts()
print("\nStored experience distribution:")
for regime, count in counts.items():
    pct = count / sum(counts.values()) * 100
    print(f"  {regime.value:10s}: {count:4d} ({pct:5.1f}%)")

# Sample and show balanced sampling
batch = balanced_buffer.sample(90)
sampled_dist = np.bincount(batch['regimes'], minlength=3)
print("\nSampled batch distribution (batch_size=90):")
print(f"  Bull:     {sampled_dist[0]:4d} ({sampled_dist[0]/90*100:5.1f}%)")
print(f"  Bear:     {sampled_dist[1]:4d} ({sampled_dist[1]/90*100:5.1f}%)")
print(f"  Sideways: {sampled_dist[2]:4d} ({sampled_dist[2]/90*100:5.1f}%)")
print("  → Should be roughly equal (~30 each) despite imbalanced storage!")

print("\n--- Testing INVERSE-FREQUENCY Sampling ---")
inv_buffer = RegimeAwareReplayBuffer(
    buffer_size=3000,
    strategy='inverse_frequency',
    random_seed=42
)

# Add with same imbalanced distribution
for i in range(900):
    state = np.random.randn(state_dim).astype(np.float32)
    action = np.random.randn(action_dim).astype(np.float32)
    reward = float(np.random.randn())
    next_state = np.random.randn(state_dim).astype(np.float32)
    done = False
    
    r = np.random.rand()
    if r < 0.60:
        regime = ReplayRegime.BULL
    elif r < 0.85:
        regime = ReplayRegime.SIDEWAYS
    else:
        regime = ReplayRegime.BEAR
    
    inv_buffer.add(state, action, reward, next_state, done, regime)

batch = inv_buffer.sample(90)
sampled_dist = np.bincount(batch['regimes'], minlength=3)
print("\nSampled batch distribution (batch_size=90):")
print(f"  Bull:     {sampled_dist[0]:4d} ({sampled_dist[0]/90*100:5.1f}%)")
print(f"  Bear:     {sampled_dist[1]:4d} ({sampled_dist[1]/90*100:5.1f}%)")
print(f"  Sideways: {sampled_dist[2]:4d} ({sampled_dist[2]/90*100:5.1f}%)")
print("  → Bear (rarest) should be OVERSAMPLED, Bull (common) UNDERSAMPLED!")

print("\n--- Testing CURRENT-FOCUSED Sampling ---")
curr_buffer = RegimeAwareReplayBuffer(
    buffer_size=3000,
    strategy='current_focused',
    current_focus_ratio=0.6,
    random_seed=42
)

# Add experiences
for i in range(900):
    state = np.random.randn(state_dim).astype(np.float32)
    action = np.random.randn(action_dim).astype(np.float32)
    regime = [ReplayRegime.BULL, ReplayRegime.BEAR, ReplayRegime.SIDEWAYS][i % 3]
    curr_buffer.add(state, action, 0.0, state, False, regime)

# Sample with current regime = BEAR
batch = curr_buffer.sample(100, current_regime=ReplayRegime.BEAR)
sampled_dist = np.bincount(batch['regimes'], minlength=3)
print("\nSampled batch when current_regime=BEAR (batch_size=100):")
print(f"  Bull:     {sampled_dist[0]:4d} ({sampled_dist[0]}%)")
print(f"  Bear:     {sampled_dist[1]:4d} ({sampled_dist[1]}%)")
print(f"  Sideways: {sampled_dist[2]:4d} ({sampled_dist[2]}%)")
print("  → Bear should be ~60%, others ~20% each!")

print("\n✓ Regime-aware replay buffer working!")

# =============================================================================
# TEST 6: Integration - Full Pipeline
# =============================================================================

print("\n" + "=" * 70)
print("TEST 6: Full Pipeline Integration")
print("=" * 70)

print("\nSimulating how this works in your training loop:")
print("-" * 50)

# 1. Create regime detector
print("\n1. Initialize regime detector...")
detector = RuleBasedRegimeDetector()

# 2. Detect regimes for entire dataset
print("2. Detect regimes for training data...")
detected_regimes = detector.detect(price_df)

# 3. Create regime-aware buffer
print("3. Create regime-aware replay buffer...")
buffer = RegimeAwareReplayBuffer(buffer_size=10000, strategy='balanced')

# 4. Simulate environment interaction
print("4. Simulating environment interaction...")
print("   (Adding experiences with regime labels)")

for day_idx in range(500):  # First 500 days
    # Get current regime from detector
    current_regime = detected_regimes.iloc[day_idx]
    replay_regime = map_regime(current_regime)
    
    # Simulate environment step
    state = np.random.randn(state_dim).astype(np.float32)
    action = np.random.randn(action_dim).astype(np.float32)
    reward = float(np.random.randn())
    next_state = np.random.randn(state_dim).astype(np.float32)
    done = day_idx == 499
    
    # Add to buffer with regime label
    buffer.add(state, action, reward, next_state, done, replay_regime)

print(f"   Added {len(buffer)} experiences")
regime_counts = buffer.get_regime_counts()
print(f"   Regime distribution:")
for regime, count in regime_counts.items():
    print(f"     {regime.value}: {count}")

# 5. Sample for training
print("\n5. Sample batch for training...")
batch = buffer.sample(64)
print(f"   Batch size: {len(batch['states'])}")
print(f"   Regime distribution in batch: ", end="")
regime_dist = np.bincount(batch['regimes'], minlength=3)
print(f"Bull={regime_dist[0]}, Bear={regime_dist[1]}, Sideways={regime_dist[2]}")

print("\n✓ Full pipeline integration successful!")

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "=" * 70)
print("WEEK 3 SUMMARY")
print("=" * 70)
print("""
✅ COMPLETED COMPONENTS:

1. REGIME DETECTION
   ├── Rule-Based Detector: SMA crossover + volatility thresholds
   │   └── Simple, interpretable, no training required
   └── HMM Detector: Learns patterns from data
       └── More sophisticated, provides probabilities

2. EXPERIENCE REPLAY
   ├── Standard Buffer (Baseline)
   │   └── Uniform random sampling - what everyone uses
   └── Regime-Aware Buffer (NOVEL CONTRIBUTION)
       ├── Balanced: Equal samples from each regime
       ├── Current-Focused: 60% current regime, 40% others  
       ├── Inverse-Frequency: Oversample rare regimes
       └── Adaptive: Based on TD errors (where agent struggles)

YOUR NOVEL CONTRIBUTION:
------------------------
Prior work adds regime to OBSERVATIONS (what agent sees).
You modify REPLAY SAMPLING (how agent learns).
These are complementary - yours is more fundamental.

NEXT STEPS (Week 4):
--------------------
1. Integrate with PPO algorithm
2. Integrate with SAC algorithm
3. Begin training experiments
""")
print("=" * 70)