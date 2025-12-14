"""
Improved LSTM Deployment Script for QQQ with Real-Time News + Replay Backtest
============================================================================
- Live: uses last COMPLETED minute bar time for news + prediction (no leakage).
- Live: uses last COMPLETED minute bar time for news + prediction (no leakage).
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
import requests
import joblib
import importlib.util

import yfinance as yf

import torch
from torch import nn

from news_features import NewsFeatureProvider

# -----------------------------
# Paths / Imports
# -----------------------------
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

FEATURES_PY_PATH = os.path.join(PROJECT_ROOT, "scripts", "03_pre_split_prep", "features.py")
spec = importlib.util.spec_from_file_location("features_module", FEATURES_PY_PATH)
features_module = importlib.util.module_from_spec(spec) if spec else None
if spec and spec.loader:
    spec.loader.exec_module(features_module)  # type: ignore[attr-defined]
else:
    raise RuntimeError(f"Could not load features.py from {FEATURES_PY_PATH}")

FeatureBuilder = getattr(features_module, "FeatureBuilder")

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

# -----------------------------
# Trading params
# -----------------------------
TICKER = "QQQ"

ENTRY_THRESHOLD = 0.0001  # 0.1%
MAX_POSITIONS = 5
POSITION_SIZE_PCT = 0.01
COOLDOWN_MINUTES = 10

STOP_LOSS_PCT = -0.004
TAKE_PROFIT_PCT = 0.007

MIN_HOLD_MINUTES = 8
MAX_HOLD_MINUTES = 15

# LSTM params (must match training)
SEQUENCE_LENGTH = 50
INPUT_SIZE = 14  # MUST match training (14 features)
HIDDEN_SIZE = 384
NUM_LAYERS = 2
OUTPUT_SIZE = 5
DROPOUT = 0.2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EASTERN = pytz.timezone("US/Eastern")

# Alpaca (live only)
ALPACA_KEY_ID = os.getenv("ALPACA_KEY_ID", keys["KEYS"].get("APCA-API-KEY-ID-Paper"))
ALPACA_SECRET = os.getenv("ALPACA_SECRET", keys["KEYS"].get("APCA-API-SECRET-KEY-Paper"))
ALPACA_BASE = os.getenv("ALPACA_BASE", "https://paper-api.alpaca.markets")

# Feature list (ordered!) - must match the model's training input schema
FEATURE_LIST_PATH = os.path.join(MODELS_DIR, "features_clean.txt")

# cooldown tracking (in-memory; good enough for prototyping)
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
        last_layer_h = h_n[-self.num_directions :, :, :]  # (dir, batch, hidden)
        last_layer_h = last_layer_h.transpose(0, 1).reshape(x.size(0), -1)  # (batch, hidden*dir)
        return self.fc(last_layer_h)


def create_last_sequence(X: np.ndarray, seq_len: int) -> np.ndarray:
    if len(X) < seq_len:
        return np.array([])
    return np.array([X[-seq_len:]])


# -----------------------------
# Alpaca helpers (live only)
# -----------------------------
def alpaca_headers() -> Dict[str, str]:
    if not ALPACA_KEY_ID or not ALPACA_SECRET:
        raise RuntimeError("Missing Alpaca keys. Set env vars or conf/keys.yaml.")
    return {
        "APCA-API-KEY-ID": ALPACA_KEY_ID,
        "APCA-API-SECRET-KEY": ALPACA_SECRET,
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


def get_account_info() -> dict:
    r = requests.get(f"{ALPACA_BASE}/v2/account", headers=alpaca_headers(), timeout=30)
    r.raise_for_status()
    return r.json()


def get_positions() -> List[dict]:
    r = requests.get(f"{ALPACA_BASE}/v2/positions", headers=alpaca_headers(), timeout=30)
    if r.status_code == 404:
        return []
    r.raise_for_status()
    return r.json()


def get_position(symbol: str) -> Optional[dict]:
    r = requests.get(f"{ALPACA_BASE}/v2/positions/{symbol}", headers=alpaca_headers(), timeout=30)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    return r.json()


def submit_bracket_market(symbol: str, qty: int, sl_price: float, tp_price: float) -> Optional[dict]:
    payload = {
        "symbol": symbol,
        "qty": qty,
        "side": "buy",
        "type": "market",
        "time_in_force": "day",
        "order_class": "bracket",
        "take_profit": {"limit_price": f"{tp_price:.2f}"},
        "stop_loss": {"stop_price": f"{sl_price:.2f}"},
    }
    try:
        r = requests.post(f"{ALPACA_BASE}/v2/orders", headers=alpaca_headers(), json=payload, timeout=30)
        r.raise_for_status()
        od = r.json()
        print(f"[ORDER] BRACKET BUY {qty} {symbol} | SL={sl_price:.2f} TP={tp_price:.2f} | id={od.get('id')}")
        return od
    except Exception as e:
        print(f"[ERROR] submit_bracket_market failed: {e}")
        return None


def close_position(symbol: str) -> bool:
    try:
        r = requests.delete(f"{ALPACA_BASE}/v2/positions/{symbol}", headers=alpaca_headers(), timeout=30)
        r.raise_for_status()
        print(f"[CLOSE] Closed {symbol}")
        return True
    except Exception as e:
        print(f"[ERROR] close_position failed for {symbol}: {e}")
        return False


def get_recent_filled_orders(symbol: str, limit: int = 100) -> List[dict]:
    params_q = {"status": "closed", "limit": str(limit), "direction": "desc", "nested": "false"}
    r = requests.get(f"{ALPACA_BASE}/v2/orders", headers=alpaca_headers(), params=params_q, timeout=30)
    r.raise_for_status()
    orders = r.json()
    out = []
    for o in orders:
        if str(o.get("status", "")).lower() != "filled":
            continue
        if str(o.get("symbol", "")).upper() != symbol.upper():
            continue
        out.append(o)
    return out


def get_last_buy_fill_time(symbol: str) -> Optional[datetime]:
    try:
        orders = get_recent_filled_orders(symbol, limit=200)
    except Exception as e:
        print(f"[WARN] cannot fetch orders for fill-time: {e}")
        return None

    last_dt: Optional[datetime] = None
    for o in orders:
        if str(o.get("side", "")).lower() != "buy":
            continue
        filled_at = o.get("filled_at")
        if not filled_at:
            continue
        try:
            dt = datetime.fromisoformat(str(filled_at).replace("Z", "+00:00"))
        except Exception:
            continue
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
        if last_dt is None or dt > last_dt:
            last_dt = dt
    return last_dt


def build_calendar_map(start_dt: datetime, end_dt: datetime) -> Dict[datetime.date, Tuple[datetime, datetime]]:
    params_q = {"start": start_dt.strftime("%Y-%m-%d"), "end": end_dt.strftime("%Y-%m-%d")}
    r = requests.get(f"{ALPACA_BASE}/v2/calendar", headers=alpaca_headers(), params=params_q, timeout=30)
    r.raise_for_status()
    days = r.json()
    cal_map: Dict[datetime.date, Tuple[datetime, datetime]] = {}
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


def is_rth(ts: pd.Timestamp, cal_map: Dict[datetime.date, Tuple[datetime, datetime]]) -> bool:
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
# Data
# -----------------------------
def download_qqq_data(days: int = 5) -> pd.DataFrame:
    print(f"[DATA] Downloading {days}d of 1m for {TICKER} via yfinance...")
    df = yf.download(TICKER, period=f"{days}d", interval="1m", auto_adjust=True, prepost=False, progress=False)
    if df is None or df.empty:
        return pd.DataFrame()

    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    
    # Flatten multi-index columns if present (yfinance behavior)
    if isinstance(df.columns, pd.MultiIndex):
        # We want the FIRST level (Price type: Open, Close, etc.), not the ticker
        df.columns = df.columns.get_level_values(0)
        
    return df



# -----------------------------
# Features (base, no news columns added here)
# -----------------------------
def load_feature_list(path: str) -> List[str]:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing feature list: {path}\n"
            "Create it from training (ordered features)."
        )
    feats: List[str] = []
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
    
    # Approx trade volume if missing
    if "avg_volume_per_trade" not in df_feat.columns:
        df_feat["avg_volume_per_trade"] = df_feat["volume"] / 100.0

    pd.set_option('future.no_silent_downcasting', True)
    df_feat = df_feat.replace([np.inf, -np.inf], np.nan).dropna()

    if df_feat.empty:
        raise RuntimeError("All features NaN after rolling windows (insufficient history?).")

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
# Strategy
# -----------------------------
def calc_signal(pred: np.ndarray) -> Tuple[float, float]:
    r3 = float(pred[1])
    r5 = float(pred[2])
    # Simple weighted average of 3m and 5m predicted returns
    s = 0.6 * r3 + 0.4 * r5
    return s, r3


def can_enter(symbol: str, signal: float, r3: float) -> bool:
    if signal <= ENTRY_THRESHOLD:
        return False
    if r3 <= 0:
        return False

    if symbol in last_trade_time:
        dt = datetime.now(timezone.utc) - last_trade_time[symbol]
        if dt < timedelta(minutes=COOLDOWN_MINUTES):
            mins_left = COOLDOWN_MINUTES - dt.total_seconds() / 60
            print(f"[COOLDOWN] {symbol}: {mins_left:.1f} min left")
            return False

    pos = get_positions()
    if len(pos) >= MAX_POSITIONS:
        print(f"[LIMIT] max positions reached ({MAX_POSITIONS})")
        return False

    return True


def should_exit(symbol: str, signal: float, r3: float) -> Tuple[bool, str]:
    now = datetime.now(timezone.utc)
    entry_time = get_last_buy_fill_time(symbol)
    if entry_time is None:
        entry_age = 999.0
    else:
        entry_age = (now - entry_time).total_seconds() / 60.0

    if entry_age >= MAX_HOLD_MINUTES:
        return True, f"Max hold reached ({entry_age:.1f}m)"

    if entry_age < MIN_HOLD_MINUTES:
        return False, f"Min hold not reached ({entry_age:.1f}m)"

    if signal < 0:
        return True, f"Signal negative (s={signal:.6f})"
    if r3 < 0:
        return True, f"3m negative (r3={r3:.6f})"

    return False, "Hold"




# -----------------------------
# Run once (live)
# -----------------------------
def run_once(dry_run: bool = False, test_data: bool = False, use_news: bool = True):
    print("=" * 70)
    print("LSTM QQQ Paper Bot (with Alpha Vantage News)" if use_news else "LSTM QQQ Paper Bot (News disabled)")
    print("=" * 70)

    acct = get_account_info()
    equity = float(acct.get("equity", 0))
    cash = float(acct.get("cash", 0))
    print(f"[ACCOUNT] Equity=${equity:,.2f} Cash=${cash:,.2f}")

    model, scaler_y, scaler_X = load_lstm_model()
    feat_list = load_feature_list(FEATURE_LIST_PATH)

    print(f"[MODEL] Input={INPUT_SIZE} Features={len(feat_list)}")
    
    if len(feat_list) != INPUT_SIZE:
        print(f"[ERROR] Feature list mismatch! Found {len(feat_list)} but model expects {INPUT_SIZE}")
        return

    news_provider = None
    if use_news:
        try:
            news_provider = NewsFeatureProvider(decay_lambda=0.001, cache_minutes=5)
            print("[NEWS] Real-time Alpha Vantage news enabled")
        except ValueError as e:
            print(f"[NEWS WARNING] {e}")
            print("[NEWS] Falling back to neutral news features (0)")
            use_news = False

    df_raw = download_qqq_data(days=5)
    if df_raw.empty:
        print("[ERROR] No yfinance data.")
        return

    # RTH filter (Alpaca calendar)
    end_dt = datetime.now(tz=EASTERN)
    start_dt = end_dt - timedelta(days=10)
    cal_map = build_calendar_map(start_dt, end_dt)

    df_rth = df_raw[df_raw.index.to_series().map(lambda ts: is_rth(ts, cal_map))]
    if df_rth.empty:
        print("[WARN] No RTH bars.")
        return

    if len(df_rth) < SEQUENCE_LENGTH + 2:
        print("[ERROR] Not enough bars.")
        return

    # Last completed minute bar (THIS is the correct "now" for features/news)
    bar_time = df_rth.index[-2] # -1 is incomplete current bar, -2 is last full
    val = df_rth["Close"].iloc[-2]
    last_completed_price = float(val.item() if hasattr(val, "item") else val)



    df_feat, last_ts = build_features_no_news(df_rth)
    
    if len(df_feat.columns) > 0 and isinstance(df_feat.columns[0], tuple):
        df_feat.columns = [col[0] if isinstance(col, tuple) else col for col in df_feat.columns]
    
    # News calc
    news_val = {
        "last_news_sentiment": 0.0,
        "news_age_minutes": 0.0,
        "effective_sentiment_t": 0.0
    }
    
    if use_news and news_provider is not None:
        try:
            current_time = bar_time.to_pydatetime()
            news_val = news_provider.get_news_features_dict(current_time, tickers=["QQQ"])
            print(f"[NEWS] S={news_val['last_news_sentiment']:.4f} Age={news_val['news_age_minutes']:.1f} Eff={news_val['effective_sentiment_t']:.4f}")
        except Exception as e:
            print(f"[NEWS DATA FAIL] {e}")

    # Build Feature Vector
    # We need the LAST sequence [t-(SEQ-1) ... t]
    # But df_feat has all history.
    
    # First: add news cols to scalar df
    # NOTE: df_feat is full history. For LIVE, we only strictly need the last 50 rows.
    # But we need to handle "past" news for the last 50 rows?
    # Actually, the model input assumes "effective sentiment" is known at each step.
    # For simplicity in LIVE run_once (low latency):
    # We assume historical effective sentiment was "close enough" to current or we re-fetch.
    # But re-fetching history for 50 bars from API per minute is expensive/impossible.
    # SOLUTION: For the live 'sequence', we assume the news state hasn't wildly changed 
    # OR we just fill the 'current' news state across the sequence if we lack history? 
    # Better: We only fetch current.
    # We'll fill the whole sequence with the CURRENT news features (approx).
    # This is a slight inaccuracy but acceptable for live deployment vs complex cached state.
    
    # 3. Add to DF
    df_feat["last_news_sentiment"] = news_val["last_news_sentiment"]
    df_feat["news_age_minutes"] = news_val["news_age_minutes"]
    df_feat["effective_sentiment_t"] = news_val["effective_sentiment_t"]
    
    # 4. Select features in order
    X_list = []
    for feat in feat_list:
        if feat in df_feat.columns:
            X_list.append(df_feat[feat].values.astype(np.float32))
        else:
            raise ValueError(f"Feature '{feat}' missing from live DF!")
            
    X_raw = np.column_stack(X_list).astype(np.float32)
    
    # 5. Scale
    # Reconstruct DF to suppress UserWarning for feature names
    X_df_raw = pd.DataFrame(X_raw, columns=feat_list)
    X = scaler_X.transform(X_df_raw)

    
    # 6. Seq
    X_seq = create_last_sequence(X, SEQUENCE_LENGTH)
    if X_seq.size == 0:
        print("[ERROR] not enough data for seq")
        return
        
    X_tensor = torch.from_numpy(X_seq).float().to(DEVICE) # (1, 50, 14)

    with torch.no_grad():
        pred_scaled = model(X_tensor).cpu().numpy()[0]
    pred = scaler_y.inverse_transform([pred_scaled])[0]
    pred = pred / 100.0  # <<< FIX: convert percent-units to decimal returns
    print("[DEBUG] pred (dec):", pred)
    print("[DEBUG] pred (pct):", pred * 100)

    s, r3 = calc_signal(pred)
    print(
        f"[PRED] 1m={pred[0]*100:.3f}% 3m={pred[1]*100:.3f}% 5m={pred[2]*100:.3f}% "
        f"10m={pred[3]*100:.3f}% 15m={pred[4]*100:.3f}%"
    )
    print(f"[SIGNAL] s={s:.6f} (θ={ENTRY_THRESHOLD:.6f}) r3={r3:.6f} @ {bar_time}")

    # Trading Logic
    pos = get_position(TICKER)

    if pos is None:
        if can_enter(TICKER, s, r3):
            acct = get_account_info()
            equity = float(acct.get("equity", 0))
            target_value = equity * POSITION_SIZE_PCT
            qty = int(target_value / last_completed_price)
            if qty <= 0:
                print("[WARN] qty=0 (equity too low or price too high)")
                return

            sl = last_completed_price * (1 + STOP_LOSS_PCT)
            tp = last_completed_price * (1 + TAKE_PROFIT_PCT)

            print(f"[ENTRY] BUY {TICKER} qty={qty} ref_price={last_completed_price:.2f} SL={sl:.2f} TP={tp:.2f}")
            if not dry_run:
                od = submit_bracket_market(TICKER, qty, sl, tp)
                if od:
                    last_trade_time[TICKER] = datetime.now(timezone.utc)
            else:
                print("[DRY RUN] not submitting order.")
        else:
            print("[NO ENTRY] conditions not met.")
    else:
        exit_now, reason = should_exit(TICKER, s, r3)
        if exit_now:
            print(f"[EXIT] {TICKER}: {reason}")
            if not dry_run:
                close_position(TICKER)
            else:
                print("[DRY RUN] not closing position.")
        else:
            qty = pos.get("qty")
            entry_price = float(pos.get("avg_entry_price", 0))
            cur_price = float(pos.get("current_price", 0))
            uplpc = float(pos.get("unrealized_plpc", 0))
            print(f"[HOLD] {TICKER} qty={qty} entry={entry_price:.2f} cur={cur_price:.2f} upl={uplpc*100:.2f}% | {reason}")


# -----------------------------
# CLI
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--test-data", action="store_true")
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--interval", type=int, default=300)
    ap.add_argument("--skip-market-hours", action="store_true", help="Skip market hours check (for testing)")
    ap.add_argument("--no-news", action="store_true", help="Disable Alpha Vantage news (use neutral features)")


    args = ap.parse_args()

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

                if args.skip_market_hours or is_market_hours:
                    if args.skip_market_hours and not is_market_hours:
                        print("[WARNING] Market CLOSED but running anyway (--skip-market-hours)")
                    run_once(dry_run=args.dry_run, test_data=args.test_data, use_news=not args.no_news)
                else:
                    print("[SKIP] Market is CLOSED - Waiting for market hours (9:30-16:00 ET Mon-Fri)")

                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\n[STOPPED] By user (Ctrl+C)")
            print(f"Total runs: {i}")
    else:
        run_once(dry_run=args.dry_run, test_data=args.test_data, use_news=not args.no_news)


if __name__ == "__main__":
    main()
