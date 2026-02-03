# Regime-Aware Experience Replay for Deep Reinforcement Learning in Adaptive Portfolio Management

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Project Overview

This research project develops a **novel experience replay mechanism** for deep reinforcement learning (DRL) applied to portfolio management. The core contribution is **Regime-Aware Experience Replay**, which labels and samples training experiences based on detected market regimes (bull, bear, sideways), addressing the fundamental challenge of non-stationarity in financial markets.

### Key Innovation

While prior work has explored adding regime information to agent observations, we address a distinct problem: **the learning process itself**. Standard experience replay samples uniformly regardless of the market regime when experiences were collected. This biases learning toward common regimes and hinders adaptation to rare but critical market conditions.

Our regime-aware experience replay:
- Labels experiences by detected market regime
- Implements intelligent sampling strategies to ensure balanced learning
- Compares four sampling strategies: balanced, current-focused, inverse-frequency, and adaptive

### Algorithms Compared

| Algorithm | Type | Action Space |
|-----------|------|--------------|
| PPO (Proximal Policy Optimization) | On-policy | Continuous |
| SAC (Soft Actor-Critic) | Off-policy | Continuous |
| TD3 (Twin Delayed DDPG) | Off-policy | Continuous |
| Rainbow DQN | Off-policy | Discrete |

## Project Structure

```
adaptive-portfolio-rl/
├── configs/                    # Configuration files (YAML)
├── data/                       # Data storage (not in git)
├── src/                        # Source code
│   ├── data/                   # Data downloading and processing
│   ├── environment/            # Trading environment (Gymnasium)
│   ├── regime/                 # Regime detection methods
│   ├── replay/                 # Experience replay (core contribution)
│   ├── agents/                 # RL algorithm wrappers
│   ├── training/               # Training orchestration
│   └── utils/                  # Utility functions
├── tests/                      # Unit tests
├── notebooks/                  # Jupyter notebooks
├── scripts/                    # Executable scripts
├── dashboard/                  # Streamlit dashboard
├── results/                    # Experiment results (not in git)
└── docs/                       # Documentation
```

## Installation

### Prerequisites

- Python 3.12 or higher
- Git

### Setup

1. **Clone the repository:**
   ```bash
   git clone https://git.cs.bham.ac.uk/projects-2025-26/pxx276.git
   cd pxx276
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv DRL
   ```

3. **Activate the environment:**
   
   On Windows (PowerShell):
   ```powershell
   .\DRL\Scripts\Activate.ps1
   ```
   
   On Windows (Command Prompt):
   ```cmd
   DRL\Scripts\activate.bat
   ```
   
   On macOS/Linux:
   ```bash
   source DRL/bin/activate
   ```

4. **Install dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

5. **Verify installation:**
   ```bash
   python verify_setup.py
   ```

### Optional: FRED API Key

To download economic indicators, get a free API key from [FRED](https://fred.stlouisfed.org/docs/api/api_key.html) and set it as an environment variable:

```bash
export FRED_API_KEY="your_api_key_here"
```

## Quick Start

### 1. Download Data

```bash
python scripts/download_data.py
```

### 2. Train an Agent

```bash
python scripts/train.py --algorithm sac --replay balanced
```

### 3. Run the Dashboard

```bash
streamlit run dashboard/app.py
```

### 4. Run Experiments

```bash
python scripts/run_experiment.py --experiment replay_comparison
```

## Research Questions

1. **Algorithm Comparison**: How do PPO, SAC, TD3, and Rainbow DQN compare for portfolio management?
2. **Replay Strategies**: Does regime-aware replay improve upon standard uniform replay?
3. **Regime Detection**: How does the regime detection method affect learning?
4. **Robustness**: Is regime-aware replay robust to noise in regime labels?
5. **Sample Efficiency**: Does regime-aware replay improve learning speed?

## Asset Universe

| Asset | Type | Description |
|-------|------|-------------|
| SPY | Equity | S&P 500 ETF (benchmark) |
| QQQ | Equity | Nasdaq 100 ETF |
| TLT | Bond | 20+ Year Treasury Bond ETF |
| GLD | Commodity | Gold ETF |
| BIL | Cash | 1-3 Month T-Bill ETF |

## Data Splits

| Period | Date Range | Purpose |
|--------|------------|---------|
| Training | 2010-01-01 to 2019-12-31 | Agent learning |
| Validation | 2020-01-01 to 2021-12-31 | Hyperparameter tuning |
| Testing | 2022-01-01 to 2023-12-31 | Final evaluation |

## Evaluation Metrics

- **Cumulative Return**: Total portfolio growth
- **Sharpe Ratio**: Risk-adjusted return
- **Sortino Ratio**: Downside risk-adjusted return
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Calmar Ratio**: Return relative to maximum drawdown

## Project Timeline

| Week | Focus |
|------|-------|
| 1 | Foundation, data infrastructure |
| 2 | Trading environment |
| 3 | Regime detection & replay buffer (core contribution) |
| 4 | PPO & SAC implementation |
| 5 | TD3 & Rainbow DQN |
| 6 | Core experiments |
| 7 | Analysis & visualization |
| 8 | Report writing |
| 9 | Polish & submission |

## Repository Guidelines

- Commit regularly with meaningful messages
- Keep large files out of git (use `.gitignore`)
- Document your code with docstrings
- Write tests for critical components

## References

Key papers this work builds upon:

1. Jiang et al. (2017) - Deep Reinforcement Learning for Portfolio Management
2. Liu et al. (2020) - FinRL: A Deep Reinforcement Learning Library
3. Schaul et al. (2016) - Prioritized Experience Replay
4. Hamilton (1989) - Markov-switching models for regime detection

## Author

**Student ID**: pxx276  
**Programme**: Computer Science  
**Institution**: University of Birmingham  
**Academic Year**: 2025/26

## License

This project is submitted as part of the Final Year Project requirements. All rights reserved by the author and the University of Birmingham.
