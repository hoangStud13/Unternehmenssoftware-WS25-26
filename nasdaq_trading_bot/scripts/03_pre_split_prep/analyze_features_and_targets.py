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
        'target_vwap_1m', 'target_vwap_3m', 'target_vwap_4m',
        'target_vwap_5m', 'target_vwap_10m', 'target_vwap_15m'
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
    # 1. Descriptive Statistics
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



# ----------------------------------------
    # 2. Correlation matrix 
    # ----------------------------------------
    print("\nComputing feature correlation matrix...")

    # Wir nehmen NUR die Features, keine Targets
    corr_cols = feature_cols 
    
    # Dropna wichtig, falls durch Rolling Windows noch NaNs da sind
    clean_df = df[corr_cols].dropna()
    corr_df = clean_df.corr()

    # --- A) Plotting (Nur Features) ---
    plt.figure(figsize=(12, 10)) # Etwas kleiner, da weniger Spalten
    sns.heatmap(corr_df, cmap="coolwarm", annot=False, center=0, vmin=-1, vmax=1)
    plt.title("Feature Correlation Matrix (Check for Multicollinearity)")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "03_feature_correlation_matrix.png"))
    plt.close()

    print("\n--- REDUNDANCY CHECK REPORT (Features only) ---")
    print("Suche nach Feature-Paaren mit Korrelation > 0.85...")
    
    threshold = 0.85
    redundant_pairs = []

    # Iteration durch das obere Dreieck der Matrix
    for i in range(len(corr_df.columns)):
        for j in range(i+1, len(corr_df.columns)):
            col1 = corr_df.columns[i]
            col2 = corr_df.columns[j]
            val = corr_df.iloc[i, j]

            if abs(val) > threshold:
                redundant_pairs.append((col1, col2, val))

    if not redundant_pairs:
        print("✅ Alles sauber! Keine redundanten Features gefunden.")
    else:
        print(f"⚠️ ACHTUNG: {len(redundant_pairs)} redundante Paare gefunden!")
        print(f"-> Diese Features sind fast identisch. Lösche jeweils das schwächere:")
        redundant_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        for c1, c2, v in redundant_pairs:
            print(f"   • {c1} <--> {c2} : {v:.4f}")

    print("\nAnalysis complete.")


if __name__ == "__main__":
    analyze_data()
