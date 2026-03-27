# Regime-Aware Experience Replay for Deep Reinforcement Learning Portfolio Optimization

![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![Stable-Baselines3](https://img.shields.io/badge/Stable--Baselines3-2.0+-green.svg)

## Overview

This project investigates **regime-aware experience replay** as a mechanism for
improving deep reinforcement learning agents in portfolio management. Unlike
prior work by Nixon Raj (2025), which augments agent *observations* with regime
information, our approach modifies the replay buffer's *sampling strategy* ---
changing what the agent remembers, not what it sees. We evaluate four RL
algorithms (PPO, SAC, TD3, Rainbow DQN) across five replay strategies on a
five-asset portfolio (SPY, QQQ, TLT, GLD, BIL) during the 2022--2023 bear
market. The best configuration, TD3 with standard replay, achieves a test Sharpe
ratio of +0.632 compared to SPY buy-and-hold at -0.006. 

## Key Results

### Top 5 Algorithm-Strategy Configurations (E1, 100 runs)

| Rank | Algorithm | Strategy | Test Sharpe | Val Sharpe | Val-Test Gap |
|------|-----------|----------|-------------|------------|--------------|
| 1 | TD3 | standard | **+0.632** | 1.537 | 0.905 |
| 2 | TD3 | current_focused | +0.620 | 1.725 | 1.105 |
| 3 | TD3 | adaptive | +0.501 | 1.444 | 0.943 |
| 4 | SAC | standard | +0.458 | 1.702 | 1.244 |
| 5 | TD3 | balanced | +0.438 | 1.415 | 0.977 |

### Passive Baseline Comparison (Test Period 2022--2023)

| Strategy | Sharpe | Return | Max Drawdown |
|----------|--------|--------|--------------|
| SPY Buy-and-Hold | -0.006 | -0.2% | 25.3% |
| 60/40 SPY/TLT | -0.468 | -13.2% | 27.6% |
| Equal-Weight | -0.133 | -2.9% | 20.6% |
| Risk Parity | -0.055 | -0.1% | 0.9% |

All trained RL agents outperform all passive baselines on the test period.

### Key Findings

1. **TD3 standard is the best overall configuration** (+0.632 Sharpe, beating SPY by +0.64)
2. **TD3 is the bear market champion** (+1.33 regime Sharpe in bear markets, statistically significant at p < 0.0001)
3. **SAC and regime-aware replay are incompatible** --- non-uniform sampling biases SAC's entropy estimates
4. **All RL agents beat passive baselines** in the 2022--2023 bear market test period
5. **400k training timesteps is optimal** (E7: best test Sharpe +0.815, lowest val-test gap 0.607)
6. **Reduced features reduce overfitting** (E4: smaller val-test gap without hurting performance)
7. **Detection method does not matter** (E5: HMM and rule-based produce equivalent results with standard replay)
8. **Adaptive sampling is less effective than expected** --- after bug fixes, TD3 adaptive (+0.501) underperforms standard (+0.632)

## Research Questions and Experiments

| Exp | Name | Question | Runs |
|-----|------|----------|------|
| E1 | Replay Comparison | Which algorithm + strategy combination works best? | 100 |
| E2 | Regime Performance | How does each configuration perform in bull/bear/sideways regimes? | --- |
| E3 | Robustness | Is the approach robust to noisy regime detection? | 36 |
| E4 | Feature Ablation | Which input features matter most? | 20 |
| E5 | Regime Detection | Does the detection method (rule-based vs HMM) matter? | 12 |
| E6 | Sample Efficiency | How sample-efficient is each algorithm? | 30 |
| E7 | Timestep Ablation | What is the optimal training duration to avoid overfitting? | 30 |

**Total: 228 runs** across all experiments. E2 is an analysis-only experiment
derived from E1 data.

**Statistical methodology:** 5 random seeds per configuration, Cohen's d effect
sizes, Bonferroni-corrected p-values, bootstrap confidence intervals, and
per-regime breakdowns.

## Project Architecture

```
src/
├── agents/                  # RL algorithm wrappers (PPO, SAC, TD3, Rainbow)
├── analysis/                # Deep analysis scripts for E1--E7
├── data/                    # Data download and feature engineering
├── environment/             # Gymnasium trading environment
├── regime/                  # Market regime detection (rule-based + HMM)
├── replay/                  # Replay buffers (standard + regime-aware)
├── training/                # Experiment runner, run configs, agent factory
└── utils/                   # Metrics calculation (Sharpe, Sortino, MDD)

scripts/
├── launch_experiment.py     # Main entry point for running experiments
├── run_experiments.py       # Experiment orchestration CLI
├── compute_baselines.py     # Passive baseline computation
├── generate_tier2_plots.py  # Generate thesis figures
├── download_data.py         # Download market data via yfinance
├── generate_docs.py         # Generate codebase documentation
└── verify_metrics.py        # Independent metric verification

configs/
├── data_config.yaml         # Assets, date ranges, features, indicators
├── env_config.yaml          # Environment parameters, reward function
├── training_config.yaml     # Algorithm hyperparameters (PPO, SAC, TD3, Rainbow)
└── experiment_config.yaml   # Experiment definitions and hypotheses

results/
├── experiments/             # Raw experiment outputs (manifest + per-run results)
├── figures/                 # 15 generated plots for the thesis
├── tables/                  # Statistical test CSVs and regime analysis
├── baselines.json           # Passive baseline metrics
└── e*_analysis_summary.csv  # Per-experiment summary tables

tests/
└── test_e2e.py              # End-to-end integration test (70 assertions)
```

The novel contribution resides in `src/replay/regime_aware_buffer.py`.

## Replay Strategies

| Strategy | Description | Role |
|----------|-------------|------|
| **standard** | Uniform random sampling (no regime awareness) | Baseline |
| **balanced** | Equal samples drawn from each regime | Regime-aware |
| **current_focused** | 60% from current regime, 40% shared among others | Regime-aware |
| **inverse_frequency** | Oversample rare regimes (e.g., bear markets) proportionally | Regime-aware |
| **adaptive** | Weight sampling by per-regime TD errors with EMA decay | Novel contribution |

Standard replay serves as the control. The adaptive strategy, which uses
exponential moving average-decayed TD errors to prioritise regimes where the
agent is learning most, is the primary novel contribution.

## Algorithms

| Algorithm | Type | Library | Regime-Awareness Mechanism |
|-----------|------|---------|---------------------------|
| **PPO** | On-policy | SB3 `PPO` | Regime features added to observations via wrapper |
| **SAC** | Off-policy | SB3 `SAC` | Regime-aware replay buffer sampling |
| **TD3** | Off-policy | SB3 `TD3` | Regime-aware replay buffer sampling |
| **Rainbow DQN** | Off-policy | SB3 `DQN` | Regime-aware replay buffer sampling + discrete action wrapper |

PPO, as an on-policy algorithm, does not use a replay buffer. Regime information
is instead provided through a `RegimeObservationWrapper` that appends regime
features to the observation vector. The three off-policy algorithms (SAC, TD3,
Rainbow) use the `RegimeAwareReplayBuffer` with configurable sampling strategies.

## Data

**Assets:**

| Ticker | Role | Description |
|--------|------|-------------|
| SPY | Equity | S&P 500 broad market |
| QQQ | Equity (Tech) | Nasdaq-100 technology-heavy |
| TLT | Fixed Income | 20+ year Treasury bonds |
| GLD | Commodity | Gold (inflation/crisis hedge) |
| BIL | Cash Proxy | 1-3 month Treasury bills (risk-free proxy) |

**Data splits:**

| Split | Period | Purpose |
|-------|--------|---------|
| Train | 2010-01-01 to 2019-12-31 | Policy learning |
| Validation | 2020-01-01 to 2021-12-31 | Hyperparameter selection |
| Test | 2022-01-01 to 2023-12-31 | Final evaluation |

The validation period (2020--2021) is predominantly bullish, while the test
period (2022--2023) includes a significant bear market. This intentional
distribution shift tests generalisation under regime change.

**Source:** Yahoo Finance via `yfinance`. Features include technical indicators
(RSI, MACD, Bollinger Bands, momentum, volatility) in both full and reduced
sets, compared in E4.

## Installation and Setup

```bash
# Clone the repository
git clone <repository-url>
cd pxx276

# Create and activate a virtual environment
python -m venv DRL
source DRL/Scripts/activate      # Windows
# source DRL/bin/activate        # Linux/macOS

# Install dependencies
pip install -r requirements.txt

# Download market data
python scripts/download_data.py

# Verify the installation with the end-to-end test
python test_e2e.py
```

**Requirements:**
- Python 3.12+
- CUDA-capable GPU is optional but recommended (this project used an NVIDIA RTX 5090 with 24 GB VRAM)
- All Python dependencies are listed in `requirements.txt`

## Usage

### Running Experiments

```bash
# Run a specific experiment with 4 parallel workers
python scripts/launch_experiment.py --experiments replay_comparison --workers 4

# Check experiment status
python scripts/launch_experiment.py --status

# Retry failed runs
python scripts/launch_experiment.py --retry-failed

# Run with custom timesteps (useful for smoke tests)
python scripts/launch_experiment.py --experiments robustness --timesteps 10000

# Dry run (list runs without executing)
python scripts/run_experiments.py --dry-run
```

Valid experiment names: `replay_comparison`, `regime_performance`, `robustness`,
`feature_ablation`, `regime_detection`, `sample_efficiency`, `timestep_ablation`.

### Computing Baselines

```bash
python scripts/compute_baselines.py
```

Outputs `results/baselines.json` with Sharpe, return, MDD, and Sortino for SPY,
60/40, equal-weight, and risk parity portfolios.

### Running Analysis

```bash
python src/analysis/analyze_e1_deep.py       # E1: Replay strategy comparison
python src/analysis/analyze_e2_regime.py      # E2: Per-regime performance
python src/analysis/analyze_e3_deep.py        # E3: Robustness to noise
python src/analysis/analyze_e4_features.py    # E4: Feature ablation
python src/analysis/analyze_e5_detection.py   # E5: Detection method comparison
python src/analysis/analyze_e6_efficiency.py  # E6: Sample efficiency
python src/analysis/analyze_e7_timesteps.py   # E7: Timestep ablation
```

Each script reads from `results/experiments/`, computes metrics, runs
statistical tests, and writes summary CSVs to `results/` and figures to
`results/figures/`.

### Generating Thesis Figures

```bash
python scripts/generate_tier2_plots.py
```

Produces equity curves, portfolio weight breakdowns, regime timelines, and
strategy comparison bar charts in `results/figures/`.

## Results Structure

```
results/
├── baselines.json                  # Passive baseline metrics
├── e1_analysis_summary.csv         # E1 (20 configs, mean across seeds)
├── e3_analysis_summary.csv         # E3 (12 configs: noise x strategy)
├── e4_analysis_summary.csv         # E4 (4 configs: algo x feature set)
├── e5_analysis_summary.csv         # E5 (4 configs: method x strategy)
├── e6_analysis_summary.csv         # E6 (6 configs: algo x strategy)
├── e7_analysis_summary.csv         # E7 (10 configs: timesteps x strategy)
├── experiments/                    # Raw per-run results and manifest
│   └── manifest.json               # Run tracking (status, timing, errors)
├── figures/                        # 15 generated plots
│   ├── baseline_comparison.png
│   ├── e1_learning_curves.png
│   ├── e1_sharpe_boxplots.png
│   ├── e2_bear_market_rankings.png
│   ├── e2_regime_heatmap.png
│   ├── e2_regime_sharpe_boxplots.png
│   ├── e3_robustness_analysis.png
│   ├── e4_feature_ablation.png
│   ├── e5_detection_comparison.png
│   ├── e6_sample_efficiency.png
│   ├── e7_timestep_analysis.png
│   ├── equity_curves_top5.png
│   ├── regime_timeline.png
│   ├── sharpe_by_strategy_bars.png
│   └── weights_by_regime.png
└── tables/                         # Statistical analysis
    ├── e1_replay_comparison.csv
    ├── e2_*.csv                    # 9 regime analysis tables
    ├── statistical_tests_sharpe.csv
    └── win_rates.csv
```

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **Sharpe Ratio** | Primary metric. Risk-adjusted return (r_f = 0). |
| **Sortino Ratio** | Downside-risk-adjusted return. |
| **Maximum Drawdown** | Largest peak-to-trough decline. |
| **Cumulative Return** | Total return over the evaluation period. |

Metrics are calculated **per episode** and then **averaged across seeds**.
Cumulative return is computed from portfolio values (start to end), not from
summing step returns. Transaction costs are already deducted in portfolio value
updates and are not subtracted again from step returns.

<details>
<summary><strong>Bug Fixes (11 total)</strong></summary>

The following bugs were identified and fixed during development. They are
documented here for reproducibility and academic integrity.

| # | Description | File(s) | Status |
|---|-------------|---------|--------|
| 1 | Episode length mismatch for val/test evaluation | `experiment_runner.py` | Fixed |
| 2 | Rainbow agent missing `regime_detector` parameter | `rainbow_agent.py` | Fixed |
| 3 | Agent factory not passing `regime_detector` | `agent_factory.py` | Fixed |
| 4 | E3 incorrectly used SAC instead of TD3 | `run_config.py` | Fixed |
| 5 | Transaction cost double-counted in reward | `trading_env.py` | Fixed |
| 6 | Cumulative return compounding error | `experiment_runner.py` | Fixed |
| 7 | Test/validation date mismatch | `experiment_runner.py`, `compute_baselines.py` | Fixed |
| 8 | `N_EVAL_EPISODES` mismatch between training and evaluation | `experiment_runner.py` | Fixed |
| 9 | Adaptive TD error accumulation without decay (partial) | `regime_aware_buffer.py` | Fixed |
| 10 | Bug #9 incomplete: `add()` still accumulated TD errors without EMA | `regime_aware_buffer.py` | Fixed |
| 11 | Risk parity baseline look-ahead bias (`iloc[-60:]` to `iloc[:60]`) | `compute_baselines.py` | Fixed |

Bugs #10 and #11 were fixed on March 22, 2026, and all affected experiments
(E1 adaptive, E7 adaptive, baselines) were re-run with corrected code.

</details>

## Citation and Academic Context

**Key references:**

- Nixon Raj (2025). *Adaptive and Regime-Aware RL for Portfolio Optimization*. arXiv:2509.14385.
- Pippas, Turkay & Phelps (2025). *Enhancing Portfolio Management with DRL and HMM*.
- Raffin et al. (2021). *Stable-Baselines3: Reliable RL Implementations*. JMLR.
- Jiang, Xu & Liang (2017). *Deep Reinforcement Learning for Portfolio Management*.
- Hamilton (1989). *A New Approach to the Economic Analysis of Nonstationary Time Series and the Business Cycle*.
- Schaul et al. (2016). *Prioritized Experience Replay*. ICLR.

## Author

**Student ID:** pxx276
**Programme:** Computer Science
**Institution:** University of Birmingham
**Academic Year:** 2025/26
