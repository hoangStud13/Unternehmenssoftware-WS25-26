"""
LSTM Deployment Script with Selectable Strategies
===================================================
Main entry point for trading. Allows selection of different strategies
and automatically uses the correct broker API (Alpaca or OANDA).

Usage:
    python lstm_deploy.py --list                    # List all strategies
    python lstm_deploy.py --strategy conservative_long --dry-run
    python lstm_deploy.py --strategy cfd_2x_leverage --loop
"""

from __future__ import annotations

import os
import sys
import time
import argparse
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import yaml
import pytz
import joblib
import importlib.util

import yfinance as yf

import torch
from torch import nn

# -----------------------------
# Paths / Imports
# -----------------------------
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, THIS_DIR)

# Import strategies module
from strategies.strategy_config import load_all_strategies, list_available_strategies, StrategyConfig
from strategies.strategies import LongOnlyMomentumStrategy, ShortOnlyStrategy, CFDLeveragedStrategy
from strategies.broker_adapters import create_broker, AlpacaBroker, OANDABroker

# Import feature builder
FEATURES_PY_PATH = os.path.join(PROJECT_ROOT, "scripts", "03_pre_split_prep", "features.py")
spec = importlib.util.spec_from_file_location("features_module", FEATURES_PY_PATH)
features_module = importlib.util.module_from_spec(spec) if spec else None
if spec and spec.loader:
    spec.loader.exec_module(features_module)
else:
    raise RuntimeError(f"Could not load features.py from {FEATURES_PY_PATH}")

FeatureBuilder = getattr(features_module, "FeatureBuilder")

# Import news features
from news_features import NewsFeatureProvider

CONF_DIR = os.path.join(PROJECT_ROOT, "conf")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models", "lstm")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
SCALER_X_PATH = os.path.join(DATA_DIR, "scaler_X.joblib")

# -----------------------------
# Load configs
# -----------------------------
with open(os.path.join(CONF_DIR, "params.yaml"), "r") as f:
    params = yaml.safe_load(f)

with open(os.path.join(CONF_DIR, "keys.yaml"), "r") as f:
    keys = yaml.safe_load(f)

# LSTM params (must match training)
SEQUENCE_LENGTH = 50
INPUT_SIZE = 14
HIDDEN_SIZE = 384
NUM_LAYERS = 2
OUTPUT_SIZE = 5
DROPOUT = 0.2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EASTERN = pytz.timezone("US/Eastern")

# Feature list path
FEATURE_LIST_PATH = os.path.join(MODELS_DIR, "features_clean.txt")

# Cooldown tracking
last_trade_time: Dict[str, datetime] = {}


# -----------------------------
# Model
# -----------------------------
class LSTMModel(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        output_size: int,
        bidirectional: bool = False,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        self.fc = nn.Linear(hidden_size * self.num_directions, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, (h_n, c_n) = self.lstm(x)
        last_layer_h = h_n[-self.num_directions :, :, :]
        last_layer_h = last_layer_h.transpose(0, 1).reshape(x.size(0), -1)
        return self.fc(last_layer_h)


def create_last_sequence(X: np.ndarray, seq_len: int) -> np.ndarray:
    if len(X) < seq_len:
        return np.array([])
    return np.array([X[-seq_len:]])


# -----------------------------
# Data
# -----------------------------
def download_market_data(ticker: str, days: int = 5) -> pd.DataFrame:
    """Download market data from yfinance"""
    print(f"[DATA] Downloading {days}d of 1m for {ticker} via yfinance...")
    df = yf.download(ticker, period=f"{days}d", interval="1m", auto_adjust=True, prepost=False, progress=False)
    if df is None or df.empty:
        return pd.DataFrame()

    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        
    return df


def download_oanda_data(broker: OANDABroker, ticker: str) -> pd.DataFrame:
    """Download market data from OANDA"""
    print(f"[DATA] Downloading 1m candles for {ticker} via OANDA...")
    try:
        candles = broker.get_candles(ticker, granularity="M1", count=500)
        
        data = []
        for c in candles:
            if c.get("complete"):
                mid = c.get("mid", {})
                data.append({
                    "timestamp": c.get("time"),
                    "Open": float(mid.get("o", 0)),
                    "High": float(mid.get("h", 0)),
                    "Low": float(mid.get("l", 0)),
                    "Close": float(mid.get("c", 0)),
                    "Volume": int(c.get("volume", 0)),
                })
        
        df = pd.DataFrame(data)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.set_index("timestamp", inplace=True)
        df.index = df.index.tz_convert("UTC")
        return df
    except Exception as e:
        print(f"[ERROR] OANDA data download failed: {e}")
        return pd.DataFrame()


# -----------------------------
# Features
# -----------------------------
def load_feature_list(path: str) -> List[str]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing feature list: {path}")
    feats = []
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if s:
                feats.append(s)
    return feats


def build_features_no_news(df_raw: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Timestamp]:
    df = df_raw.copy()
    df = df.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
    df["timestamp"] = df.index
    df["vwap"] = (df["high"] + df["low"] + df["close"]) / 3.0

    ema_periods = params["DATA_PREP"]["EMA_PERIODS"]
    slope_periods = params["DATA_PREP"]["SLOPE_PERIODS"]

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

    if df_feat.empty:
        raise RuntimeError("All features NaN after rolling windows.")

    last_ts = df_feat.index[-1]
    return df_feat, last_ts


# -----------------------------
# Model loading
# -----------------------------
def load_lstm_model() -> Tuple[LSTMModel, object, object]:
    model_path = os.path.join(MODELS_DIR, "best_lstm_model.pth")
    scaler_y_path = os.path.join(DATA_DIR, "scaler_y.joblib")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not os.path.exists(scaler_y_path):
        raise FileNotFoundError(f"Scaler Y not found: {scaler_y_path}")
    if not os.path.exists(SCALER_X_PATH):
        raise FileNotFoundError(f"Scaler X not found: {SCALER_X_PATH}")

    scaler_y = joblib.load(scaler_y_path)
    scaler_X = joblib.load(SCALER_X_PATH)

    model = LSTMModel(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        output_size=OUTPUT_SIZE,
        bidirectional=False,
        dropout=DROPOUT,
    ).to(DEVICE)

    state_dict = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    return model, scaler_y, scaler_X


# -----------------------------
# Strategy helpers
# -----------------------------
def get_strategy_class(strategy_type: str):
    """Get the strategy class based on strategy_type"""
    mapping = {
        "long_only": LongOnlyMomentumStrategy,
        "short_only": ShortOnlyStrategy,
        "cfd_leveraged": CFDLeveragedStrategy,
        "long_short": CFDLeveragedStrategy,  # Uses same class with different params
    }
    return mapping.get(strategy_type, LongOnlyMomentumStrategy)


def calc_signal(pred: np.ndarray, config: StrategyConfig) -> Tuple[float, float]:
    """Calculate trading signal from predictions"""
    r3 = float(pred[1])
    r5 = float(pred[2])
    s = 0.6 * r3 + 0.4 * r5
    return s, r3


def can_enter_position(symbol: str, signal: float, r3: float, config: StrategyConfig, broker) -> bool:
    """Check if entry conditions are met"""
    if signal <= config.entry_threshold:
        return False
    if r3 <= 0:
        return False

    if symbol in last_trade_time:
        dt = datetime.now(timezone.utc) - last_trade_time[symbol]
        if dt < timedelta(minutes=config.cooldown_minutes):
            mins_left = config.cooldown_minutes - dt.total_seconds() / 60
            print(f"[COOLDOWN] {symbol}: {mins_left:.1f} min left")
            return False

    positions = broker.get_positions()
    if len(positions) >= config.max_positions:
        print(f"[LIMIT] max positions reached ({config.max_positions})")
        return False

    return True


def should_exit_position(symbol: str, signal: float, r3: float, config: StrategyConfig, broker) -> Tuple[bool, str]:
    """Check if exit conditions are met"""
    now = datetime.now(timezone.utc)
    entry_time = broker.get_last_fill_time(symbol, 'buy')
    if entry_time is None:
        entry_age = 999.0
    else:
        entry_age = (now - entry_time).total_seconds() / 60.0

    if entry_age >= config.max_hold_minutes:
        return True, f"Max hold reached ({entry_age:.1f}m)"

    if entry_age < config.min_hold_minutes:
        return False, f"Min hold not reached ({entry_age:.1f}m)"

    if signal < config.exit_threshold:
        return True, f"Signal below threshold (s={signal:.6f})"
    if r3 < 0:
        return True, f"3m negative (r3={r3:.6f})"

    return False, "Hold"


# -----------------------------
# RTH helpers for Alpaca
# -----------------------------
def build_calendar_map(broker: AlpacaBroker, start_dt: datetime, end_dt: datetime) -> Dict:
    days = broker.get_calendar(start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d"))
    cal_map = {}
    for d in days:
        date_str = d.get("date")
        open_str = d.get("open")
        close_str = d.get("close")
        if not date_str or not open_str or not close_str:
            continue
        y, m, dd = map(int, date_str.split("-"))
        oh, om = map(int, open_str.split(":"))
        ch, cm = map(int, close_str.split(":"))
        open_dt = EASTERN.localize(datetime(y, m, dd, oh, om))
        close_dt = EASTERN.localize(datetime(y, m, dd, ch, cm))
        cal_map[open_dt.date()] = (open_dt, close_dt)
    return cal_map


def is_rth(ts: pd.Timestamp, cal_map: Dict) -> bool:
    if ts.tzinfo is None:
        ts_eastern = ts.tz_localize("UTC").astimezone(EASTERN)
    else:
        try:
            ts_eastern = ts.tz_convert(EASTERN)
        except Exception:
            ts_eastern = ts.tz_localize("UTC").astimezone(EASTERN)

    d = ts_eastern.date()
    if d not in cal_map:
        return False
    open_dt, close_dt = cal_map[d]
    return open_dt <= ts_eastern < close_dt


# -----------------------------
# Main run
# -----------------------------
def run_once(config: StrategyConfig, broker, dry_run: bool = False, use_news: bool = True):
    """Run one trading iteration with the given strategy"""
    
    print("=" * 70)
    print(f"LSTM Trading Bot - Strategy: {config.name}")
    print(f"API: {config.api_type.upper()} | Ticker: {config.ticker} | Leverage: {config.leverage}x")
    print("=" * 70)

    # Get account info
    acct = broker.get_account_info()
    equity = float(acct.get("equity", 0))
    cash = float(acct.get("cash", acct.get("balance", 0)))
    print(f"[ACCOUNT] Equity=${equity:,.2f} Cash=${cash:,.2f}")

    # Load model
    model, scaler_y, scaler_X = load_lstm_model()
    feat_list = load_feature_list(FEATURE_LIST_PATH)

    print(f"[MODEL] Input={INPUT_SIZE} Features={len(feat_list)}")
    
    if len(feat_list) != INPUT_SIZE:
        print(f"[ERROR] Feature list mismatch! Found {len(feat_list)} but model expects {INPUT_SIZE}")
        return

    # News provider
    news_provider = None
    if use_news:
        try:
            news_provider = NewsFeatureProvider(decay_lambda=0.001, cache_minutes=5)
            print("[NEWS] Real-time Alpha Vantage news enabled")
        except ValueError as e:
            print(f"[NEWS WARNING] {e}")
            use_news = False

    # Get market data
    if config.api_type == "alpaca":
        df_raw = download_market_data(config.ticker, days=5)
    else:
        df_raw = download_oanda_data(broker, config.ticker)
    
    if df_raw.empty:
        print("[ERROR] No market data available.")
        return

    # RTH filter for Alpaca
    if config.api_type == "alpaca":
        end_dt = datetime.now(tz=EASTERN)
        start_dt = end_dt - timedelta(days=10)
        cal_map = build_calendar_map(broker, start_dt, end_dt)
        df_rth = df_raw[df_raw.index.to_series().map(lambda ts: is_rth(ts, cal_map))]
    else:
        df_rth = df_raw  # OANDA doesn't need RTH filter for CFDs

    if df_rth.empty:
        print("[WARN] No trading data available.")
        return

    if len(df_rth) < SEQUENCE_LENGTH + 2:
        print("[ERROR] Not enough bars.")
        return

    # Last completed bar
    bar_time = df_rth.index[-2]
    val = df_rth["Close"].iloc[-2]
    last_completed_price = float(val.item() if hasattr(val, "item") else val)

    # Build features
    df_feat, last_ts = build_features_no_news(df_rth)
    
    if len(df_feat.columns) > 0 and isinstance(df_feat.columns[0], tuple):
        df_feat.columns = [col[0] if isinstance(col, tuple) else col for col in df_feat.columns]
    
    # News features
    news_val = {"last_news_sentiment": 0.0, "news_age_minutes": 0.0, "effective_sentiment_t": 0.0}
    
    if use_news and news_provider is not None:
        try:
            current_time = bar_time.to_pydatetime()
            # Use QQQ for news even if trading CFD (same underlying)
            news_val = news_provider.get_news_features_dict(current_time, tickers=["QQQ"])
            print(f"[NEWS] S={news_val['last_news_sentiment']:.4f} Eff={news_val['effective_sentiment_t']:.4f}")
        except Exception as e:
            print(f"[NEWS FAIL] {e}")

    df_feat["last_news_sentiment"] = news_val["last_news_sentiment"]
    df_feat["news_age_minutes"] = news_val["news_age_minutes"]
    df_feat["effective_sentiment_t"] = news_val["effective_sentiment_t"]
    
    # Build feature vector
    X_list = []
    for feat in feat_list:
        if feat in df_feat.columns:
            X_list.append(df_feat[feat].values.astype(np.float32))
        else:
            raise ValueError(f"Feature '{feat}' missing!")
            
    X_raw = np.column_stack(X_list).astype(np.float32)
    
    X_df_raw = pd.DataFrame(X_raw, columns=feat_list)
    X = scaler_X.transform(X_df_raw)
    
    X_seq = create_last_sequence(X, SEQUENCE_LENGTH)
    if X_seq.size == 0:
        print("[ERROR] Not enough data for sequence")
        return
        
    X_tensor = torch.from_numpy(X_seq).float().to(DEVICE)

    # Predict
    with torch.no_grad():
        pred_scaled = model(X_tensor).cpu().numpy()[0]
    pred = scaler_y.inverse_transform([pred_scaled])[0]
    pred = pred / 100.0

    s, r3 = calc_signal(pred, config)
    print(
        f"[PRED] 1m={pred[0]*100:.3f}% 3m={pred[1]*100:.3f}% 5m={pred[2]*100:.3f}% "
        f"10m={pred[3]*100:.3f}% 15m={pred[4]*100:.3f}%"
    )
    print(f"[SIGNAL] s={s:.6f} (θ={config.entry_threshold:.6f}) r3={r3:.6f} @ {bar_time}")

    # Trading Logic
    ticker = config.ticker
    pos = broker.get_position(ticker)

    if pos is None:
        if can_enter_position(ticker, s, r3, config, broker):
            acct = broker.get_account_info()
            equity = float(acct.get("equity", 0))
            target_value = equity * config.position_size_pct
            qty = int(target_value / last_completed_price)
            
            if config.leverage > 1:
                qty = int(qty * config.leverage)
            
            if qty <= 0:
                print("[WARN] qty=0 (equity too low or price too high)")
                return

            sl = last_completed_price * (1 + config.stop_loss_pct)
            tp = last_completed_price * (1 + config.take_profit_pct)

            print(f"[ENTRY] BUY {ticker} qty={qty} ref_price={last_completed_price:.2f} SL={sl:.2f} TP={tp:.2f}")
            
            if not dry_run:
                if isinstance(broker, AlpacaBroker):
                    od = broker.submit_bracket_order(ticker, qty, "buy", sl, tp)
                else:
                    od = broker.submit_order({
                        "symbol": ticker,
                        "qty": qty,
                        "side": "buy",
                        "stop_loss": {"price": sl},
                        "take_profit": {"price": tp},
                    })
                if od:
                    last_trade_time[ticker] = datetime.now(timezone.utc)
            else:
                print("[DRY RUN] not submitting order.")
        else:
            print("[NO ENTRY] conditions not met.")
    else:
        exit_now, reason = should_exit_position(ticker, s, r3, config, broker)
        if exit_now:
            print(f"[EXIT] {ticker}: {reason}")
            if not dry_run:
                broker.close_position(ticker)
            else:
                print("[DRY RUN] not closing position.")
        else:
            qty = pos.get("qty")
            entry_price = float(pos.get("avg_entry_price", 0))
            print(f"[HOLD] {ticker} qty={qty} entry={entry_price:.2f} | {reason}")


# -----------------------------
# CLI
# -----------------------------
def main():
    ap = argparse.ArgumentParser(
        description="LSTM Trading Bot with Selectable Strategies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python lstm_deploy.py --list
  python lstm_deploy.py --strategy conservative_long --dry-run
  python lstm_deploy.py --strategy cfd_2x_leverage --loop --interval 300
        """
    )
    ap.add_argument("--list", action="store_true", help="List all available strategies")
    ap.add_argument("--strategy", "-s", type=str, help="Strategy ID to use (e.g., conservative_long)")
    ap.add_argument("--dry-run", action="store_true", help="Run without executing trades")
    ap.add_argument("--loop", action="store_true", help="Run continuously")
    ap.add_argument("--interval", type=int, default=300, help="Loop interval in seconds (default: 300)")
    ap.add_argument("--skip-market-hours", action="store_true", help="Skip market hours check")
    ap.add_argument("--no-news", action="store_true", help="Disable news features")

    args = ap.parse_args()

    # List strategies
    if args.list:
        list_available_strategies()
        return

    # Require strategy
    if not args.strategy:
        print("[ERROR] No strategy specified. Use --strategy <id> or --list to see options.")
        list_available_strategies()
        return

    # Load strategy
    all_strategies = load_all_strategies()
    if args.strategy not in all_strategies:
        print(f"[ERROR] Unknown strategy: {args.strategy}")
        list_available_strategies()
        return

    config = all_strategies[args.strategy]
    print(f"\n[STRATEGY] Loaded: {config.name}")
    print(f"[STRATEGY] Type: {config.strategy_type} | API: {config.api_type} | Ticker: {config.ticker}")

    # Create broker
    try:
        broker = create_broker(config.api_type, keys.get("KEYS", {}))
    except ValueError as e:
        print(f"[ERROR] {e}")
        return

    # Run
    if args.loop:
        i = 0
        try:
            while True:
                i += 1
                now = datetime.now(EASTERN)

                is_weekday = now.weekday() < 5
                market_open = now.time() >= datetime.strptime("09:30", "%H:%M").time()
                market_close = now.time() <= datetime.strptime("16:00", "%H:%M").time()
                is_market_hours = is_weekday and market_open and market_close

                print(f"\n--- RUN #{i} {now.strftime('%Y-%m-%d %H:%M:%S %Z')} ---")

                # CFDs trade 24/5, so skip market hours check for OANDA
                if config.api_type == "oanda" or args.skip_market_hours or is_market_hours:
                    if args.skip_market_hours and not is_market_hours and config.api_type != "oanda":
                        print("[WARNING] Market CLOSED but running anyway (--skip-market-hours)")
                    run_once(config, broker, dry_run=args.dry_run, use_news=not args.no_news)
                else:
                    print("[SKIP] Market is CLOSED - Waiting for market hours (9:30-16:00 ET Mon-Fri)")

                time.sleep(args.interval)
        except KeyboardInterrupt:
            print(f"\n[STOPPED] By user (Ctrl+C) after {i} runs")
    else:
        run_once(config, broker, dry_run=args.dry_run, use_news=not args.no_news)


if __name__ == "__main__":
    main()