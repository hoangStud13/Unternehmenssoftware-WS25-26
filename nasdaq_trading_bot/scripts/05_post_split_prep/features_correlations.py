import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt

def calculate_feature_correlations():
    # ----------------------------------------
    # Setup paths
    # ----------------------------------------
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
    data_dir = os.path.join(project_root, 'data')
    images_dir = os.path.join(project_root, 'images')
    
    data_file = os.path.join(data_dir, 'nasdaq100_index_1m_features.parquet')
    output_csv = os.path.join(data_dir, 'feature_correlations.csv')
    output_plot = os.path.join(images_dir, 'feature_correlation_matrix.png')

    if not os.path.exists(data_file):
        print(f"Error: Data file not found at {data_file}")
        return

    # ----------------------------------------
    # Load data
    # ----------------------------------------
    print(f"Loading data from {data_file}...")
    df = pd.read_parquet(data_file)

    # ----------------------------------------
    # Identify feature columns
    # ----------------------------------------
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
    feature_cols = [c for c in feature_cols if c in df.columns]

    if not feature_cols:
        print("Error: Missing feature columns in the dataset.")
        return

    print(f"Calculating correlations between {len(feature_cols)} features...")

    # ----------------------------------------
    # Calculate Correlation
    # ----------------------------------------
    corr_matrix = df[feature_cols].corr()
    
    print("\nFeature-Feature Correlations:")
    print(corr_matrix)
    
    # ----------------------------------------
    # Save to CSV
    # ----------------------------------------
    print(f"\nSaving correlations to {output_csv}...")
    corr_matrix.to_csv(output_csv)

    # ----------------------------------------
    # Plot Heatmap
    # ----------------------------------------
    print(f"Saving correlation heatmap to {output_plot}...")
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig(output_plot)
    plt.close()
    
if __name__ == "__main__":
    calculate_feature_correlations()
