import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt

def calculate_correlations():
    # ----------------------------------------
    # Setup paths
    # ----------------------------------------
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
    data_dir = os.path.join(project_root, 'data')
    
    data_file = os.path.join(data_dir, 'nasdaq100_index_1m_features.parquet')
    output_csv = os.path.join(data_dir, 'feature_target_correlations.csv')
    output_plot = os.path.join(project_root, 'images', 'feature_target_correlation_matrix.png')

    if not os.path.exists(data_file):
        print(f"Error: Data file not found at {data_file}")
        return

    # ----------------------------------------
    # Load data
    # ----------------------------------------
    print(f"Loading data from {data_file}...")
    df = pd.read_parquet(data_file)

    # ----------------------------------------
    # Identify feature and target columns
    # ----------------------------------------
    # Using the same columns as in analyze_features_and_targets.py
    target_cols = [
        'target_return_1m', 'target_return_3m', 'target_return_5m', 'target_return_10m', 'target_return_15m'
    ]

    feature_cols = [
        'simple_return_1m', 'simple_return_5m', 'simple_return_15m',
        'ema_5', 'ema_20', 'ema_diff',
        'volume', 'volume_zscore_30m',
        'realized_volatility',
        'avg_volume_per_trade',
        'hl_span',
        'last_news_sentiment', 'news_age_minutes', 'effective_sentiment_t'
    ]
    
    # Filter columns that actually exist in the dataframe
    target_cols = [c for c in target_cols if c in df.columns]
    feature_cols = [c for c in feature_cols if c in df.columns]

    if not target_cols or not feature_cols:
        print("Error: Missing target or feature columns in the dataset.")
        return

    print(f"Calculating correlations between {len(feature_cols)} features and {len(target_cols)} targets...")

    # ----------------------------------------
    # Calculate Correlation
    # ----------------------------------------
    # We want rows = features, columns = targets
    # So we correlate all, then slice
    
    combined_cols = feature_cols + target_cols
    corr_matrix = df[combined_cols].corr()
    
    # Slice the matrix: Rows are features, Columns are targets
    feature_target_corr = corr_matrix.loc[feature_cols, target_cols]
    
    # Sort by the first target column (usually target_return_1m) in descending order
    if target_cols:
        primary_target = target_cols[0]
        print(f"\nSorting features by correlation with {primary_target} (descending)...")
        feature_target_corr = feature_target_corr.sort_values(by=primary_target, ascending=False)
    
    print("\nFeature-Target Correlations (Sorted):")
    print(feature_target_corr)
    
    # ----------------------------------------
    # Save to CSV
    # ----------------------------------------
    print(f"\nSaving correlations to {output_csv}...")
    feature_target_corr.to_csv(output_csv)
    
if __name__ == "__main__":
    calculate_correlations()
