"""
Independent verification of financial metrics for one completed run.
Compares:
  A. Gross return from raw portfolio_value trajectory (no cost deduction)
  B. Net return from info['step_return'] (costs deducted — what our metrics use)
  C. Manual Sharpe calculation vs stored Sharpe
  D. Agent performance vs SPY buy-and-hold benchmark
"""
import sys, json, pathlib, numpy as np
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, ".")

from src.data.downloader import download_ohlcv_data
from src.data.features import calculate_all_features
from src.environment.trading_env import TradingEnvironment
from src.agents.ppo_agent import PPOAgent
from src.utils.metrics import calculate_all_metrics

TICKERS = ["SPY", "QQQ", "TLT", "GLD", "BIL"]
test_price = download_ohlcv_data(TICKERS, "2022-01-01", "2023-12-31", cache_dir="data/raw/testing")
test_feat  = calculate_all_features(test_price)
test_env   = TradingEnvironment(price_data=test_price, feature_data=test_feat,
                                episode_length=252, random_start=False, start_idx=0)

rdir  = pathlib.Path("results/experiments/replay_comparison/ppo/standard/replay_comparison__ppo__standard__seed42")
agent = PPOAgent(test_env)
agent.load(str(rdir / "model"))

print("=" * 60)
print("MANUAL VERIFICATION: PPO/standard/seed42 — TEST (2022-23)")
print("=" * 60)

obs, _ = test_env.reset()
initial_capital = test_env.initial_capital
print(f"Initial capital : ${initial_capital:,.0f}")

pv_trace, sr_trace, tc_trace, reward_trace = [], [], [], []
done = False
while not done:
    action, _ = agent.predict(obs, deterministic=True)
    obs, reward, term, trunc, info = test_env.step(action)
    pv_trace.append(info["portfolio_value"])
    sr_trace.append(info["step_return"])
    tc_trace.append(info["transaction_cost"])
    reward_trace.append(reward)
    done = term or trunc

pv  = np.array(pv_trace)
sr  = np.array(sr_trace)
tc  = np.array(tc_trace)
rw  = np.array(reward_trace)

print(f"Steps           : {len(sr)}")
print(f"Total tx costs  : ${tc.sum():,.2f}  ({tc.sum()/initial_capital*100:.3f}% of capital)")
print()

# --- Return comparison ---
gross_returns = np.diff(np.concatenate([[initial_capital], pv])) / np.concatenate([[initial_capital], pv[:-1]])
gross_cum = (np.prod(1 + gross_returns) - 1) * 100
net_cum   = (np.prod(1 + sr) - 1) * 100
rew_cum   = (np.prod(1 + rw) - 1) * 100

print(f"Cumulative return comparison:")
print(f"  A. Gross (portfolio_value, no cost deduction) : {gross_cum:+.3f}%")
print(f"  B. Net   (step_return, costs deducted)        : {net_cum:+.3f}%  <- what metrics use")
print(f"  C. Reward-based (old, distorted)              : {rew_cum:+.3f}%  <- old buggy approach")
print(f"  Cost drag A -> B                              : {gross_cum - net_cum:.4f}%")
print()

# --- Sharpe comparison ---
def manual_sharpe(returns, rf_annual=0.02):
    ann_ret = (1 + returns.mean()) ** 252 - 1
    ann_vol = returns.std() * np.sqrt(252)
    return (ann_ret - rf_annual) / ann_vol if ann_vol > 0 else 0.0

print(f"Sharpe ratio comparison:")
print(f"  Manual from step_return  : {manual_sharpe(sr):+.4f}")
print(f"  Stored (from our func)   : {json.loads((rdir/'test_results.json').read_text())['sharpe_ratio']:+.4f}")
print(f"  Manual from reward       : {manual_sharpe(rw):+.4f}  (old distorted)")
diff = abs(manual_sharpe(sr) - json.loads((rdir/"test_results.json").read_text())["sharpe_ratio"])
print(f"  Discrepancy (step vs stored) : {diff:.6f}  {'[OK <0.01]' if diff < 0.01 else '[MISMATCH]'}")
print()

# --- MDD comparison ---
def manual_mdd(returns):
    cum = np.cumprod(1 + returns)
    peak = np.maximum.accumulate(cum)
    dd = (peak - cum) / peak
    return dd.max()

stored = json.loads((rdir / "test_results.json").read_text())
print(f"Max drawdown comparison:")
print(f"  Manual from step_return  : {manual_mdd(sr)*100:.2f}%")
print(f"  Stored (from our func)   : {stored['max_drawdown']*100:.2f}%")
print()

# --- Benchmark sanity ---
spy_prices = test_price["SPY"]["Close"].values
spy_ret = (spy_prices[-1] / spy_prices[0] - 1) * 100
bil_prices = test_price["BIL"]["Close"].values
bil_ret = (bil_prices[-1] / bil_prices[0] - 1) * 100

print(f"Benchmark sanity (same 2022-23 window):")
print(f"  SPY buy-and-hold : {spy_ret:+.2f}%  (S&P 500; -19% in 2022, +26% in 2023)")
print(f"  BIL buy-and-hold : {bil_ret:+.2f}%  (cash/T-bills; near risk-free rate)")
print(f"  Agent net return : {net_cum:+.2f}%  (1 episode, 252 days)")
print()
print("VERDICT: If agent net return is between BIL and SPY, metrics are plausible.")
print("=" * 60)
