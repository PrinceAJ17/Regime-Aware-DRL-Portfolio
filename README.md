# Regime-Aware Experience Replay for Deep Reinforcement Learning in Adaptive Portfolio Management

**Final Year Project - University of Birmingham**  
**Academic Year: 2025/26**

---

## Overview

This project explores Deep Reinforcement Learning (DRL) for portfolio management, with a focus on handling the non-stationary nature of financial markets through regime-aware experience replay mechanisms.

Unlike traditional portfolio optimization methods (Markowitz, Black-Litterman) that assume normal distributions and stationary conditions, this approach learns optimal capital allocation strategies directly from market data.

---

## Core Research Focus

**The Challenge**: Financial markets shift between different regimes (bull markets, bear markets, sideways markets) with fundamentally different characteristics. Standard RL methods struggle with this non-stationarity.

**The Approach**: Develop and evaluate experience replay mechanisms that are aware of market regimes, comparing performance across multiple RL algorithms (PPO, SAC, TD3, Rainbow DQN).

**Key Research Questions**:
- How do different RL algorithms perform on portfolio management tasks?
- Can regime-aware methods improve performance compared to standard approaches?
- What impact does feature engineering have versus end-to-end learning?
- How robust are these methods to market regime changes?

---

## Project Structure
```
adaptive-portfolio-rl/
├── configs/              # Configuration files
├── src/                  # Source code
│   ├── data/            # Data downloading and preprocessing
│   ├── features/        # Feature engineering
│   ├── environment/     # Trading environment
│   ├── regime/          # Market regime detection
│   ├── replay/          # Experience replay mechanisms
│   ├── algorithms/      # RL algorithm implementations
│   └── utils/           # Helper utilities
├── notebooks/           # Analysis and visualization notebooks
├── scripts/             # Executable scripts
├── tests/               # Unit tests
├── data/                # Datasets (not tracked in git)
├── outputs/             # Results and trained models
└── logs/                # Training logs
```

---

### Setup
```bash
# Clone the repository
git clone https://git.cs.bham.ac.uk/projects-2025-26/YOUR_USERNAME.git
cd YOUR_USERNAME

# Create virtual environment
conda create -n portfolio-rl python=3.10
conda activate portfolio-rl

# Install dependencies
pip install -r requirements.txt
```

---

## Data

**Assets**: SPY (S&P 500), QQQ (Nasdaq), TLT (Treasury Bonds), GLD (Gold), BIL (Cash)

**Time Periods**:
- Training: 2010-2019
- Validation: 2020-2021  
- Testing: 2022-2023

**Features**:
- OHLCV data (Open, High, Low, Close, Volume)
- Technical indicators (RSI, MACD, Bollinger Bands, etc.)
- Relative strength measures
- Volatility metrics
- Sentiment data (optional)

---

## Algorithms

Comparing four state-of-the-art RL algorithms:

1. **PPO** (Proximal Policy Optimization) - Stable policy gradient method
2. **SAC** (Soft Actor-Critic) - Maximum entropy RL
3. **TD3** (Twin Delayed DDPG) - Improved actor-critic
4. **Rainbow DQN** - Comprehensive DQN improvements

Each tested with both standard and regime-aware experience replay.

---

## Timeline

**Duration**: 9 weeks (Semester 2)

- **Weeks 1-2**: Data pipeline and trading environment
- **Week 3**: Regime detection and experience replay mechanisms
- **Weeks 4-5**: Algorithm implementation and integration
- **Week 6**: Experimental evaluation
- **Week 7**: Analysis and visualization
- **Weeks 8-9**: Report writing and final polish

**Key Deadlines**:
- Week 10: Report and GitLab submission
- Weeks 11-12: Project inspection

---

## Current Progress

- [x] Repository setup
- [ ] Data pipeline implementation
- [ ] Trading environment development
- [ ] Regime detection methods
- [ ] Experience replay mechanisms
- [ ] Algorithm implementation
- [ ] Experimental evaluation
- [ ] Results analysis
- [ ] Final report

---

## Running the Project
```bash
# Download data
python scripts/download_data.py

# Train an agent
python scripts/train.py --algo ppo --config configs/ppo.yaml

# Evaluate trained model
python scripts/evaluate.py --model outputs/models/ppo_model.zip

# Launch dashboard
streamlit run dashboard/app.py
```

---

## Testing
```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_environment.py
```

---

## References

Key papers and resources:
- Jiang et al. (2017) - Deep RL for Portfolio Management
- Liu et al. (2020) - FinRL Library
- Schulman et al. (2017) - PPO Algorithm
- Haarnoja et al. (2018) - SAC Algorithm

Full bibliography available in project report.

---

## Contact

**Student**: Prince Alexander John  
**Supervisor**: Tao Chen  
**Email**: t.chen@bham.ac.uk

---

**Last Updated**: January 2026  
**Status**: In Development - Week 1