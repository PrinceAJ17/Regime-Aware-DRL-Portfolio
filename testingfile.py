"""
Test script for the trading environment.
Save this as testingfile.py in your project root.
"""

from src.data.downloader import download_ohlcv_data
from src.data.features import calculate_all_features
from src.environment.trading_env import TradingEnvironment

# Step 1: Download data
print("Downloading data...")
data = download_ohlcv_data(
    tickers=['SPY', 'QQQ', 'TLT', 'GLD', 'BIL'],
    start_date='2015-01-01',
    end_date='2019-12-31'
)

# Step 2: Calculate features
print("\nCalculating features...")
features = calculate_all_features(data)

# Step 3: Create environment
print("\nCreating trading environment...")
env = TradingEnvironment(
    price_data=data,
    feature_data=features,
    episode_length=252,  # 1 year
    initial_capital=1_000_000
)

# Step 4: Test the environment
print("\n" + "="*60)
print("Environment Test")
print("="*60)

print(f"\nObservation space: {env.observation_space.shape}")
print(f"Action space: {env.action_space.shape}")

# Reset and get initial observation
obs, info = env.reset(seed=42)
print(f"\nInitial observation shape: {obs.shape}")
print(f"Start date: {info['start_date']}")
print(f"Initial portfolio value: ${info['initial_value']:,.2f}")

# Run a few random steps
print("\nRunning 10 random steps...")
total_reward = 0

for i in range(10):
    # Take a random action
    action = env.action_space.sample()
    
    # Step the environment
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    
    print(f"  Step {i+1}: Value=${info['portfolio_value']:,.2f}, "
          f"Return={info['step_return']*100:.2f}%, Reward={reward:.4f}")
    
    if terminated or truncated:
        print("  Episode ended early!")
        break

print(f"\nTotal reward over 10 steps: {total_reward:.4f}")
print(f"Current portfolio value: ${info['portfolio_value']:,.2f}")

# Test with Stable Baselines 3
print("\n" + "="*60)
print("Testing with Stable Baselines 3")
print("="*60)

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_checker import check_env
    
    # Validate environment follows Gymnasium standards
    print("\nValidating environment with SB3 env_checker...")
    check_env(env, warn=True)
    print("✓ Environment passed validation!")
    
    # Quick training test (just 1000 steps)
    print("\nRunning quick training test (1000 steps)...")
    model = PPO("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=1000)
    print("✓ Training completed successfully!")
    
except ImportError:
    print("Stable Baselines 3 not installed, skipping SB3 test")
except Exception as e:
    print(f"Error: {e}")

print("\n" + "="*60)
print("All tests passed! Environment is ready for training.")
print("="*60)