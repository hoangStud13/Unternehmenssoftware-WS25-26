import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


def analyze_data():
    # ----------------------------------------
    # Setup paths
    # ----------------------------------------
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
    data_dir = os.path.join(project_root, 'data')
    plots_dir = os.path.join(project_root, 'images')
    os.makedirs(plots_dir, exist_ok=True)

    data_file = os.path.join(data_dir, 'nasdaq100_index_1m_features.parquet')

    if not os.path.exists(data_file):
        print(f"Error: Data file not found at {data_file}")
        return

    # Plot style
    plt.style.use("seaborn-v0_8")
    plt.rcParams["figure.dpi"] = 120
    plt.rcParams["savefig.dpi"] = 120

    # ----------------------------------------
    # Load data
    # ----------------------------------------
    print(f"Loading data from {data_file}...")
    df = pd.read_parquet(data_file)

    print("Shape:", df.shape)
    print("Columns:", list(df.columns))

    # ----------------------------------------
    # Identify feature and target columns
    # ----------------------------------------
    target_cols = [
        'target_return_1m', 'target_return_3m', 'target_return_4m',
        'target_return_5m', 'target_return_10m', 'target_return_15m'
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

    # ----------------------------------------
    # Descriptive Statistics
    # ----------------------------------------
    print("\nCalculating descriptive statistics...")

    feature_stats = df[feature_cols].describe().transpose()
    target_stats = df[target_cols].describe().transpose()

    print("\nFeature stats sample:")
    print(feature_stats.head())

    print("\nTarget stats:")
    print(target_stats)

    feature_stats.to_csv(os.path.join(data_dir, "feature_descriptive_statistics.csv"))
    target_stats.to_csv(os.path.join(data_dir, "target_descriptive_statistics.csv"))

if __name__ == "__main__":
    analyze_data()
