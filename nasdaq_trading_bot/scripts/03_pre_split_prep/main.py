import yaml
import os
import pandas as pd

from FeatureBuilder import FeatureBuilder

# Start processing from a given offset within the symbol list (useful for chunking runs).
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
data_dir = os.path.join(project_root, 'data')
price_file = os.path.join(data_dir, 'nasdaq100_index_1m.parquet')
df = pd.read_parquet(price_file)

# Instantiate the FeatureBuilder and build features
fb = FeatureBuilder(df, ema_windows=[5, 20], return_windows=[1, 5, 15])
features_df = fb.build_features_before_split()
print(features_df.head())
