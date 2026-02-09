from src.data.downloader import download_ohlcv_data
from src.data.features import calculate_all_features

# Step 1: Download raw data
data = download_ohlcv_data(
    tickers=['SPY', 'QQQ', 'TLT', 'GLD', 'BIL'],
    start_date='2020-01-01',
    end_date='2023-12-31'
)

# Step 2: Calculate features
features = calculate_all_features(data)

# Step 3: Access features for a specific ticker
spy_features = features['SPY']
print(spy_features.head())
print(spy_features.columns.tolist())