"""
LSTM Backtest v5 with AlphaVantage News for QQQ
================================================

Goal:
- Validate model predictions vs true future returns (1/3/5/10/15m)
- NEWS ENABLED: fetches real news from AlphaVantage and calculates sentiment
- Uses exact 14 features from training (features_clean.txt)

Outputs:
- scaler_y shape diagnostics
- news feature statistics
- distribution stats for TRUE and PRED (mean/std/p99/maxabs)
- metrics: MAE/RMSE/Directional Accuracy
- optional CSV export

Usage:
  python lstm_backtest.py --days 7 --sample-prints 3
  python lstm_backtest.py --days 10 --save-csv
"""

from __future__ import annotations

import os
import sys
import argparse
from typing import List, Tuple, Dict, Optional
from datetime import datetime

import numpy as np
import pandas as pd
import yaml
import joblib
import importlib.util

import yfinance as yf
import torch
from torch import nn
from dotenv import load_dotenv
import matplotlib.pyplot as plt

# ======================================================
# Paths
# ======================================================
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

FEATURES_PY_PATH = os.path.join(PROJECT_ROOT, "scripts", "03_pre_split_prep", "features.py")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models", "lstm")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

# MUST match training
FEATURE_LIST_PATH = os.path.join(MODELS_DIR, "features_clean.txt")
MODEL_PATH = os.path.join(MODELS_DIR, "best_lstm_model.pth")
SCALER_Y_PATH = os.path.join(DATA_DIR, "scaler_y.joblib")
SCALER_X_PATH = os.path.join(DATA_DIR, "scaler_X.joblib")

os.makedirs(RESULTS_DIR, exist_ok=True)

# Load News Module
NEWS_FEATURES_PATH = os.path.join(THIS_DIR, "news_features.py")
spec_news = importlib.util.spec_from_file_location("news_features_module", NEWS_FEATURES_PATH)
news_features_module = importlib.util.module_from_spec(spec_news)
spec_news.loader.exec_module(news_features_module)  # type: ignore
NewsFeatureProvider = news_features_module.NewsFeatureProvider

# Load environment
load_dotenv()

# Load Config
with open(os.path.join(PROJECT_ROOT, "conf", "params.yaml"), "r") as f:
    params = yaml.safe_load(f)

# Load FeatureBuilder
spec = importlib.util.spec_from_file_location("features_module", FEATURES_PY_PATH)
features_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(features_module)  # type: ignore
FeatureBuilder = features_module.FeatureBuilder

# ======================================================
# Constants (MUST MATCH TRAINING)
# ======================================================
TICKER = "QQQ"
SEQ_LEN = 50
INPUT_SIZE = 14  # 11 Technical + 3 News
HIDDEN_SIZE = 384
NUM_LAYERS = 2
OUTPUT_SIZE = 5
DROPOUT = 0.2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NEWS_FEATURES = [
    "last_news_sentiment",
    "news_age_minutes",
    "effective_sentiment_t",
]

# ======================================================
# Model
# ======================================================
class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=INPUT_SIZE,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            batch_first=True,
            dropout=DROPOUT if NUM_LAYERS > 1 else 0.0,
        )
        self.fc = nn.Linear(HIDDEN_SIZE, OUTPUT_SIZE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1])

# ======================================================
# Helpers
# ======================================================
def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [str(c[0]) for c in df.columns]
    else:
        df.columns = [str(c[0]) if isinstance(c, tuple) else str(c) for c in df.columns]
    return df

def load_feature_list() -> List[str]:
    if not os.path.exists(FEATURE_LIST_PATH):
        raise FileNotFoundError(f"Feature list not found: {FEATURE_LIST_PATH}")
    with open(FEATURE_LIST_PATH, "r") as f:
        return [line.strip() for line in f if line.strip()]

def download_data(days: int) -> pd.DataFrame:
    print(f"[DATA] Downloading {days} days of data for {TICKER}...")
    df = yf.download(
        TICKER,
        period=f"{days}d",
        interval="1m",
        auto_adjust=True,
        prepost=False,
        progress=False,
        group_by="column",
    )
    if df is None or df.empty:
        raise RuntimeError("No data downloaded from yfinance.")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")

    return df

def build_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()
    df = df.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
    df["timestamp"] = df.index
    df["vwap"] = (df["high"] + df["low"] + df["close"]) / 3.0

    builder = FeatureBuilder(
        df=df,
        ema_windows=params["DATA_PREP"]["EMA_PERIODS"],
        return_windows=params["DATA_PREP"]["SLOPE_PERIODS"],
        price_col="vwap",
        timestamp_col="timestamp",
    )
    df_feat = builder.build_features_before_split()
    df_feat = flatten_columns(df_feat)
    
    if "avg_volume_per_trade" not in df_feat.columns:
        df_feat["avg_volume_per_trade"] = df_feat["volume"] / 100.0
        # print("[INFO] avg_volume_per_trade approximated as volume/100")
    
    df_feat = df_feat.replace([np.inf, -np.inf], np.nan).dropna()

    if df_feat.empty:
        raise RuntimeError("All features NaN after rolling windows.")

    return df_feat

def add_news_features(df_feat: pd.DataFrame, news_provider: NewsFeatureProvider) -> pd.DataFrame:
    print("[NEWS] Fetching news data once for backtest period...")
    # Fetch once
    try:
        df_news = news_provider.fetch_news_df_once(tickers=["QQQ"])
    except Exception as e:
        print(f"[NEWS WARN] Failed to fetch news: {e}")
        df_news = pd.DataFrame()
    
    if df_news.empty:
        print("[NEWS WARNING] No news available, using neutral features")
        df_feat["last_news_sentiment"] = 0.0
        df_feat["news_age_minutes"] = 0.0
        df_feat["effective_sentiment_t"] = 0.0
        return df_feat
    
    print(f"[NEWS] Aligning {len(df_news)} articles to {len(df_feat)} bars...")
    
    decay_lambda = news_provider.decay_lambda
    
    # Using pandas asof merge is much faster/cleaner
    df_news_indexed = df_news.set_index("timestamp").sort_index()
    # Ensure TZ awareness matches
    if df_feat.index.tz is None:
        df_feat.index = df_feat.index.tz_localize("UTC")
    
    # Include timestamp as a column in df_news_indexed
    df_news_indexed["news_ts"] = df_news_indexed.index
    
    merged_full = pd.merge_asof(
        df_feat.sort_index(),
        df_news_indexed[["sentiment_score", "news_ts"]],
        left_index=True,
        right_index=True,
        direction='backward'
    )
    
    # Fill NaNs (no news yet)
    merged_full["sentiment_score"] = merged_full["sentiment_score"].fillna(0.0)
    
    # Calculate Age
    bar_times = merged_full.index
    news_times = merged_full["news_ts"]
    
    # Age in minutes
    age_series = (bar_times - news_times).dt.total_seconds() / 60.0
    age_series = age_series.fillna(0.0)
    
    # Effective sentiment
    eff_series = merged_full["sentiment_score"] * np.exp(-decay_lambda * age_series)
    
    df_feat["last_news_sentiment"] = merged_full["sentiment_score"]
    df_feat["news_age_minutes"] = age_series
    df_feat["effective_sentiment_t"] = eff_series
    
    print("[NEWS] News alignment complete.")
    return df_feat

def build_X(df_feat: pd.DataFrame, feat_list: List[str]) -> np.ndarray:
    df_feat = flatten_columns(df_feat)
    cols = []
    for f in feat_list:
        if f in df_feat.columns:
            cols.append(df_feat[f].values)
        else:
            print(f"[ERROR] Missing feature '{f}' in dataframe!")
            print(f"Available: {list(df_feat.columns)[:20]}...")
            raise KeyError(f"Feature '{f}' missing")

    X = np.column_stack(cols).astype(np.float32)
    return X

def true_return(close: pd.Series, i: int, horizon: int) -> float | None:
    if i + horizon >= len(close):
        return None
    # Returns in DECIMAL (0.01 = 1%)
    return float(close.iloc[i + horizon] / close.iloc[i] - 1.0)

# ======================================================
# Main Backtest
# ======================================================
def run_backtest(days: int, sample_prints: int, save_csv: bool) -> None:
    print("=" * 70)
    print(f"LSTM BACKTEST v5 (Optimized) | Ticker: {TICKER} | Days: {days}")
    print("=" * 70)

    # 1. Load Model & Scalers
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    model = LSTMModel().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    scaler_y = joblib.load(SCALER_Y_PATH)
    scaler_X = joblib.load(SCALER_X_PATH)

    feat_list = load_feature_list()
    print(f"[CONFIG] Features: {len(feat_list)} (Expect {INPUT_SIZE})")
    
    if len(feat_list) != INPUT_SIZE:
        raise ValueError(f"Feature list length ({len(feat_list)}) != INPUT_SIZE ({INPUT_SIZE})")

    # 2. Get Data
    df_raw = download_data(days=max(days, 5))
    
    # 3. Build Features
    df_feat = build_features(df_raw)
    
    # 4. Add News (Robust Merge)
    news_provider = NewsFeatureProvider()
    df_feat = add_news_features(df_feat, news_provider)
    
    # 5. Build X
    X_raw = build_X(df_feat, feat_list)
    print(f"[DATA] X_raw shape: {X_raw.shape}")
    
    # 6. Scale X
    X = scaler_X.transform(X_raw)
    print(f"[DATA] X scaled mean: {X.mean():.4f}, std: {X.std():.4f}")

    # 7. Close Prices for Ground Truth
    close_prices = df_raw.loc[df_feat.index, "Close"].astype(float)

    # 8. Loop & Predict
    results = []
    # Stop -15 to allow true_return calculation
    limit = len(df_feat) - 15
    
    print(f"[BACKTEST] Running inference on {limit - SEQ_LEN} bars...")
    
    with torch.no_grad():
        for i in range(SEQ_LEN, limit):
            # Sequence: [i-50 : i]
            x_seq = X[i - SEQ_LEN : i] # shape (50, 14)
            x_batch = torch.from_numpy(x_seq).unsqueeze(0).to(DEVICE) # shape (1, 50, 14)
            
            out_scaled = model(x_batch).cpu().numpy()[0] # shape (5,)
            out_inv = scaler_y.inverse_transform([out_scaled])[0] # shape (5,)
            
            row = {
                "timestamp": df_feat.index[i],
                "pred_1m": out_inv[0],
                "pred_3m": out_inv[1],
                "pred_5m": out_inv[2],
                "pred_10m": out_inv[3],
                "pred_15m": out_inv[4],
                "true_1m": true_return(close_prices, i, 1),
                "true_3m": true_return(close_prices, i, 3),
                "true_5m": true_return(close_prices, i, 5),
                "true_10m": true_return(close_prices, i, 10),
                "true_15m": true_return(close_prices, i, 15),
            }
            results.append(row)

            if sample_prints > 0 and i % 500 == 0:
                print(f"  [{row['timestamp']}] Pred 5m: {row['pred_5m']*100:.3f}% | True 5m: {row['true_5m']*100:.3f}%")

    df_res = pd.DataFrame(results).dropna()
    print(f"[BACKTEST] Usable results: {len(df_res)}")
    
    # 9. Metrics
    print("\nMETRICS (Decimals converted to %):")
    for h in ["1m", "3m", "5m"]:
        p = df_res[f"pred_{h}"].values
        t = df_res[f"true_{h}"].values
        mae = np.mean(np.abs(p - t))
        rmse = np.sqrt(np.mean((p - t)**2))
        da = np.mean(np.sign(p) == np.sign(t))
        print(f"  {h}: MAE={mae*100:.4f}% | RMSE={rmse*100:.4f}% | DirAcc={da*100:.1f}%")

    # 10. Save & Plot
    if save_csv:
        path = os.path.join(RESULTS_DIR, "backtest_results.csv")
        df_res.to_csv(path, index=False)
        print(f"[SAVE] Results saved to {path}")

    # Plot
    plt.figure(figsize=(15, 6))
    subset = df_res.iloc[-300:]
    plt.plot(subset["timestamp"], subset["true_5m"], label="True 5m", alpha=0.5)
    plt.plot(subset["timestamp"], subset["pred_5m"], label="Pred 5m", alpha=0.8)
    plt.axhline(0, color="k", linestyle="--", alpha=0.3)
    plt.legend()
    plt.title("LSTM Backtest: Predicted vs True Returns (5m Horizon) - Last 300 pts")
    
    plot_path = os.path.join(RESULTS_DIR, "backtest_plot.png")
    plt.savefig(plot_path)
    print(f"[PLOT] Saved to {plot_path}")
    # plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--sample-prints", type=int, default=3)
    parser.add_argument("--save-csv", action="store_true")
    args = parser.parse_args()
    
    run_backtest(args.days, args.sample_prints, args.save_csv)

if __name__ == "__main__":
    main()
