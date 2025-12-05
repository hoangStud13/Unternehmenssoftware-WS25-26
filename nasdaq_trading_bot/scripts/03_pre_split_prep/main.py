
import os
import pandas as pd

import sys
# Add current directory to path to allow imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from features import FeatureBuilder
from targets import TargetBuilder

# Start processing from a given offset within the symbol list (useful for chunking runs).
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
data_dir = os.path.join(project_root, 'data')
price_file = os.path.join(data_dir, 'nasdaq100_index_1m.parquet')
df = pd.read_parquet(price_file)

# Define output paths
features_parquet = os.path.join(data_dir, 'nasdaq100_index_1m_features.parquet')
features_csv = os.path.join(data_dir, 'nasdaq100_index_1m_features.csv')

if os.path.exists(features_parquet):
    print(f"Loading features from cache: {features_parquet}")
    features_df = pd.read_parquet(features_parquet)
else:
    print("Cache not found. Generating features...")
    # Instantiate the FeatureBuilder and build features
    fb = FeatureBuilder(df, ema_windows=[5, 20], return_windows=[1, 5, 15])
    features_df = fb.build_features_before_split()

    # Calculate Targets
    tb = TargetBuilder(features_df)
    features_df = tb.calculate_vwap_targets(windows=[1, 3, 4, 5, 10, 15])

    # Filter rows without news
    if 'news_id' in features_df.columns:
        initial_count = len(features_df)
        features_df = features_df.dropna(subset=['news_id'])
        dropped_count = initial_count - len(features_df)
        print(f"Dropped {dropped_count} rows without news. Remaining: {len(features_df)}")
        
        print(f"Saving filtered features and targets to: {features_parquet}")
        features_df.to_parquet(features_parquet, index=True)
        features_df.to_csv(features_csv, index=True)

print(features_df.head())

