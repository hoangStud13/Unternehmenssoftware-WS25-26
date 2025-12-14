"""
Feed Forward Backtest with Trading Logic (Entry/Exit Simulation)
=================================================================
Simulates trades based on FF predictions, using the same logic as deployment.
"""

import os
import sys
import argparse
import joblib
import torch
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import timedelta, timezone
import importlib.util

from torch import nn

# Add project root to path
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

# Import NewsFeatureProvider
NEWS_SCRIPT_PATH = os.path.join(PROJECT_ROOT, "scripts", "07_deployment", "news_features.py")
spec_news = importlib.util.spec_from_file_location("news_features", NEWS_SCRIPT_PATH)
news_module = importlib.util.module_from_spec(spec_news)
spec_news.loader.exec_module(news_module)
NewsFeatureProvider = news_module.NewsFeatureProvider

# Import FeatureBuilder
FEATURES_PY_PATH = os.path.join(PROJECT_ROOT, "scripts", "03_pre_split_prep", "features.py")
spec = importlib.util.spec_from_file_location("features_module", FEATURES_PY_PATH)
features_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(features_module)
FeatureBuilder = features_module.FeatureBuilder


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
TICKER = "QQQ"
MODELS_DIR = os.path.join(PROJECT_ROOT, "models", "feed_forward")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODEL_DIR_LSTM = os.path.join(PROJECT_ROOT, "models", "lstm") # For features list

SCALER_X_PATH = os.path.join(DATA_DIR, "scaler_X.joblib")
SCALER_Y_PATH = os.path.join(DATA_DIR, "scaler_y.joblib")
FEATURE_LIST_PATH = os.path.join(MODEL_DIR_LSTM, "features_clean.txt")

# Model Params (Must match training)
INPUT_SIZE = 14
OUTPUT_SIZE = 5
DROPOUT = 0.2
# Hidden layers from training script
HIDDEN1 = 1024
HIDDEN2 = 1024
HIDDEN3 = 512
HIDDEN4 = 512
HIDDEN5 = 256

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Trading Params
ENTRY_THRESHOLD = 0.0001
STOP_LOSS_PCT = -0.004
TAKE_PROFIT_PCT = 0.007
MIN_HOLD_MINUTES = 8
MAX_HOLD_MINUTES = 15
POSITION_SIZE_CASH = 10000.0  # Fixed cash per trade for simplicity

# -----------------------------------------------------------------------------
# Model Class
# -----------------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, in_dim, h1, h2, h3, h4, h5, out_dim, dropout_p):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, h1),
            nn.LeakyReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(h1, h2),
            nn.LeakyReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(h2, h3),
            nn.LeakyReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(h3, h4),
            nn.LeakyReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(h4, h5),
            nn.LeakyReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(h5, out_dim)
        )
    def forward(self, x):
        return self.net(x)

# -----------------------------------------------------------------------------
# Data & Features
# -----------------------------------------------------------------------------
def load_feature_list():
    with open(FEATURE_LIST_PATH, "r") as f:
        return [line.strip() for line in f if line.strip()]

def download_data(days=5):
    print(f"[DATA] Downloading {days} days of data for {TICKER}...")
    df = yf.download(TICKER, period=f"{days}d", interval="1m", progress=False)
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    return df

def build_features(df_raw):
    df = df_raw.copy()
    df = df.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
    df["timestamp"] = df.index
    df["vwap"] = (df["high"] + df["low"] + df["close"]) / 3.0

    # Dummy config matching params.yaml (assumed)
    ema_periods = [9, 21, 50]
    slope_periods = [1, 3, 5, 10, 15]

    builder = FeatureBuilder(
        df=df,
        ema_windows=ema_periods,
        return_windows=slope_periods,
        price_col="vwap",
        timestamp_col="timestamp",
    )
    df_feat = builder.build_features_before_split()

    if "avg_volume_per_trade" not in df_feat.columns:
        df_feat["avg_volume_per_trade"] = df_feat["volume"] / 100.0
    
    pd.set_option('future.no_silent_downcasting', True)
    df_feat = df_feat.replace([np.inf, -np.inf], np.nan).dropna()
    return df_feat

def add_news_features(df_feat, provider):
    print("[NEWS] Aligning historical news...")
    try:
        df_news = provider.fetch_news_df_once(tickers=[TICKER])
        if df_news.empty:
            raise ValueError("No news found")
        
        df_news = df_news.sort_values("timestamp")
        df_news_idx = df_news.set_index("timestamp").sort_index()

        # Merge asof
        merged = pd.merge_asof(
            df_feat.sort_index(), 
            df_news_idx[["sentiment_score"]], 
            left_index=True, 
            right_index=True, 
            direction='backward'
        )
        merged["sentiment_score"] = merged["sentiment_score"].fillna(0.0)

        # Age calculation
        df_news_idx["pts"] = df_news_idx.index
        merged_ts = pd.merge_asof(
            df_feat.sort_index(),
            df_news_idx[["pts"]],
            left_index=True,
            right_index=True,
            direction='backward'
        )
        
        bar_ts = merged_ts.index
        news_ts = merged_ts["pts"]
        
        # If no news before bar, age is infinite/undefined -> handle fill
        age_s = (bar_ts - news_ts).dt.total_seconds() / 60.0
        age_s = age_s.fillna(999999.0)

        sent_s = merged["sentiment_score"]
        eff_s = sent_s * np.exp(-provider.decay_lambda * age_s)

        df_feat["last_news_sentiment"] = sent_s
        df_feat["news_age_minutes"] = age_s
        df_feat["effective_sentiment_t"] = eff_s
        
    except Exception as e:
        print(f"[NEWS] Setup failed ({e}). Using neutral news.")
        df_feat["last_news_sentiment"] = 0.0
        df_feat["news_age_minutes"] = 0.0
        df_feat["effective_sentiment_t"] = 0.0
        
    return df_feat

# -----------------------------------------------------------------------------
# Checkers
# -----------------------------------------------------------------------------
def check_signals(pred):
    # Pred: [1m, 3m, 5m, 10m, 15m]
    r3 = pred[1] # 3m
    r5 = pred[2] # 5m
    s = 0.6 * r3 + 0.4 * r5
    return s, r3

# -----------------------------------------------------------------------------
# Simulation Class
# -----------------------------------------------------------------------------
class FeatureAwareSimulator:
    def __init__(self, initial_cash=100000.0):
        self.cash = initial_cash
        self.equity = initial_cash
        self.position = None # {qty, entry_price, entry_time, sl, tp}
        self.trades = []
        self.equity_curve = []
    
    def update_equity(self, current_price):
        val = self.cash
        if self.position:
            val += self.position['qty'] * current_price
        self.equity = val
        return val

    def try_enter(self, time, price, signal, r3):
        # Already in position?
        if self.position:
            return False

        # Entry logic
        if signal > ENTRY_THRESHOLD and r3 > 0:
            qty = int(POSITION_SIZE_CASH / price)
            if qty <= 0: return False
            
            sl = price * (1 + STOP_LOSS_PCT)
            tp = price * (1 + TAKE_PROFIT_PCT)
            
            self.position = {
                'entry_time': time,
                'entry_price': price,
                'qty': qty,
                'sl': sl,
                'tp': tp
            }
            self.cash -= qty * price
            # print(f"[BUY] @ {time} {price:.2f} | s={signal:.5f}")
            return True
        return False

    def try_exit(self, time, price, signal, r3):
        if not self.position:
            return False
        
        p = self.position
        entry_age = (time - p['entry_time']).total_seconds() / 60.0
        
        reason = None
        
        # 1. Bracket
        if price <= p['sl']: reason = "StopLoss"
        elif price >= p['tp']: reason = "TakeProfit"
        
        # 2. Time
        elif entry_age >= MAX_HOLD_MINUTES: reason = "MaxHold"
        
        # 3. Strategy (only if min hold passed)
        elif entry_age >= MIN_HOLD_MINUTES:
            if signal < 0: reason = "SignalFlip"
            elif r3 < 0: reason = "3mFlip"
            
        if reason:
            pnl = (price - p['entry_price']) * p['qty']
            self.cash += p['qty'] * price
            
            self.trades.append({
                'entry_time': p['entry_time'],
                'exit_time': time,
                'entry_price': p['entry_price'],
                'exit_price': price,
                'qty': p['qty'],
                'pnl': pnl,
                'reason': reason
            })
            # print(f"[SELL] @ {time} {price:.2f} PnL={pnl:.2f} ({reason})")
            self.position = None
            return True
        
        return False

# -----------------------------------------------------------------------------
# Main Backtest Loop
# -----------------------------------------------------------------------------
def run_simulation(days):
    # 1. Load Model & Scalers
    model = MLP(INPUT_SIZE, HIDDEN1, HIDDEN2, HIDDEN3, HIDDEN4, HIDDEN5, OUTPUT_SIZE, DROPOUT).to(DEVICE)
    model.load_state_dict(torch.load(os.path.join(MODELS_DIR, "best_model_feed_forward.pt"), map_location=DEVICE))
    model.eval()
    
    scaler_x = joblib.load(SCALER_X_PATH)
    scaler_y = joblib.load(SCALER_Y_PATH)
    feature_list = load_feature_list()

    # 2. Prepare Data
    df_raw = download_data(days)
    if df_raw.empty: return

    df_feat = build_features(df_raw)
    
    # News
    news_provider = NewsFeatureProvider()
    df_feat = add_news_features(df_feat, news_provider)
    
    X_raw = []
    found_cols = []
    missing_cols = []
    
    # We must iterate feature_list to ensure correct order and count
    for col in feature_list:
        if col in df_feat.columns:
            X_raw.append(df_feat[col].values)
            found_cols.append(col)
        else:
            missing_cols.append(col)
            X_raw.append(np.zeros(len(df_feat), dtype=np.float32))
            
    print(f"[DEBUG] Found cols: {len(found_cols)}")
    if missing_cols:
        print(f"[DEBUG] Missing cols: {missing_cols}")
        
    X_raw = np.column_stack(X_raw).astype(np.float32)
    
    # Scale X
    X_df = pd.DataFrame(X_raw, columns=feature_list)
    X_scaled = scaler_x.transform(X_df)
    
    # Prices for simulation
    close_prices = df_raw["Close"].reindex(df_feat.index).astype(float)
    open_prices = df_raw["Open"].reindex(df_feat.index).astype(float)
    
    sim = FeatureAwareSimulator()
    
    print(f"[BACKTEST] Running on {len(df_feat)} bars...")
    
    # Loop
    for i in range(0, len(df_feat) - 1):
        # 1. Predict at time t (index i)
        x_vec = X_scaled[i] # Shape (14,)
        
        with torch.no_grad():
            x_tensor = torch.from_numpy(x_vec).float().unsqueeze(0).to(DEVICE) # (1, 14)
            pred_scaled = model(x_tensor).cpu().numpy()[0]
        
        pred = scaler_y.inverse_transform([pred_scaled])[0]
        pred = pred / 100.0 # Convert to decimal
        
        s, r3 = check_signals(pred)
        
        # 2. Execution Logic
        # Decision made at Close[i].
        # Actions occur at Open[i+1].
        
        t_next = df_feat.index[i+1]
        price_next_open = open_prices.iloc[i+1]
        
        # Check exits
        if sim.position:
            sim.try_exit(t_next, price_next_open, s, r3)
            
        # Try entry
        sim.try_enter(t_next, price_next_open, s, r3)
        
        # Update equity curve
        sim.equity_curve.append({"time": t_next, "equity": sim.update_equity(price_next_open)})

    # Summary
    trades = pd.DataFrame(sim.trades)
    if trades.empty:
        print("No trades executed.")
    else:
        wins = trades[trades['pnl'] > 0]
        win_rate = len(wins) / len(trades)
        total_pnl = trades['pnl'].sum()
        
        print("\n" + "="*40)
        print(f"RESULTS ({days} days)")
        print("="*40)
        print(f"Total Trades: {len(trades)}")
        print(f"Win Rate:     {win_rate:.1%}")
        print(f"Total PnL:    ${total_pnl:.2f}")
        print(f"Final Equity: ${sim.equity:.2f}")
        print("-" * 40)
        print(trades[["entry_time", "qty", "pnl", "reason"]].tail(10).to_string())

    # Plot
    if sim.equity_curve:
        ec_df = pd.DataFrame(sim.equity_curve).set_index("time")
        plt.figure(figsize=(10, 6))
        plt.plot(ec_df.index, ec_df["equity"], label="Equity")
        plt.title(f"FF Backtest Equity Curve ({dataset_name(days)})")
        plt.legend()
        plt.grid(True)
        out_path = os.path.join(PROJECT_ROOT, "images", "08_ff_trade_backtest.png")
        plt.savefig(out_path)
        print(f"\n[PLOT] Saved to {out_path}")

def dataset_name(days):
    return f"Last {days} Days"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=7)
    args = parser.parse_args()
    
    run_simulation(args.days)
