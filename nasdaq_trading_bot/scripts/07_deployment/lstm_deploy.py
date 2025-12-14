"""
Improved LSTM Deployment Script for QQQ with Real-Time News + Replay Backtest
============================================================================
- Live: uses last COMPLETED minute bar time for news + prediction (no leakage).
- Replay: runs a proper "as-of timestamp" backtest with simulated fills + PnL
  even when the market is closed.

Replay fill rule (conservative):
- Signal computed at minute close t
- Entry/Exit filled at next minute open t+1
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

ENTRY_THRESHOLD = 0.001  # 0.1%
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
# Replay simulator state (in-memory)
# -----------------------------
sim_state = {
    "cash": 100000.0,
    "qty": 0,
    "entry_price": None,
    "entry_time": None,
    "sl": None,
    "tp": None,
    "trades": [],  # list of dicts
}

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
# Replay simulator helpers
# -----------------------------
def sim_in_position() -> bool:
    return sim_state["qty"] > 0


def sim_buy(qty: int, price: float, now: datetime, sl: float, tp: float) -> bool:
    cost = qty * price
    if cost > sim_state["cash"]:
        print(f"[SIM] Not enough cash for qty={qty} cost={cost:.2f}")
        return False

    sim_state["cash"] -= cost
    sim_state["qty"] = qty
    sim_state["entry_price"] = price
    sim_state["entry_time"] = now
    sim_state["sl"] = sl
    sim_state["tp"] = tp
    sim_state["trades"].append({"type": "BUY", "time": now, "price": price, "qty": qty})
    print(f"[SIM BUY] {now} qty={qty} fill={price:.2f} SL={sl:.2f} TP={tp:.2f}")
    return True


def sim_close(price: float, now: datetime, reason: str):
    qty = sim_state["qty"]
    if qty <= 0:
        return

    entry_price = float(sim_state["entry_price"])
    pnl = (price - entry_price) * qty
    sim_state["cash"] += qty * price

    sim_state["trades"].append({"type": "SELL", "time": now, "price": price, "qty": qty, "pnl": pnl, "reason": reason})
    print(f"[SIM SELL] {now} qty={qty} fill={price:.2f} PnL={pnl:.2f} | {reason}")

    sim_state["qty"] = 0
    sim_state["entry_price"] = None
    sim_state["entry_time"] = None
    sim_state["sl"] = None
    sim_state["tp"] = None


def sim_check_bracket(price: float, now: datetime) -> bool:
    if not sim_in_position():
        return False
    sl = sim_state["sl"]
    tp = sim_state["tp"]
    if sl is not None and price <= sl:
        sim_close(price, now, "Stop loss hit")
        return True
    if tp is not None and price >= tp:
        sim_close(price, now, "Take profit hit")
        return True
    return False


def sim_entry_age_minutes(now: datetime) -> float:
    et = sim_state["entry_time"]
    if et is None:
        return 999.0
    return (now - et).total_seconds() / 60.0


def should_exit_sim(signal: float, r3: float, now: datetime) -> Tuple[bool, str]:
    age = sim_entry_age_minutes(now)

    if age >= MAX_HOLD_MINUTES:
        return True, f"Max hold reached ({age:.1f}m)"

    if age < MIN_HOLD_MINUTES:
        return False, f"Min hold not reached ({age:.1f}m)"

    if signal < 0:
        return True, f"Signal negative (s={signal:.6f})"
    if r3 < 0:
        return True, f"3m negative (r3={r3:.6f})"

    return False, "Hold"


# -----------------------------
# Replay news helper
# -----------------------------
def news_features_at_time_from_df(df_news: pd.DataFrame, t: pd.Timestamp, decay_lambda: float) -> dict:
    if df_news is None or df_news.empty:
        return {"last_news_sentiment": 0.0, "news_age_minutes": 0.0, "effective_sentiment_t": 0.0}

    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    else:
        t = t.tz_convert("UTC")

    past = df_news[df_news["timestamp"] <= t]
    if past.empty:
        return {"last_news_sentiment": 0.0, "news_age_minutes": 0.0, "effective_sentiment_t": 0.0}

    # df_news -> assumes sorted timestamp (but pandas asof is better normally, here we do simple filter)
    # We want MOST RECENT one
    most_recent = past.iloc[-1]
    
    sentiment = float(most_recent["sentiment_score"])
    age = (t - most_recent["timestamp"]).total_seconds() / 60.0
    eff = sentiment * float(np.exp(-decay_lambda * age))

    return {"last_news_sentiment": sentiment, "news_age_minutes": float(age), "effective_sentiment_t": float(eff)}


# -----------------------------
# Replay backtest (trades + PnL)
# -----------------------------
def run_replay(days: int = 5):
    print("=" * 70)
    print(f"REPLAY MODE (days={days}) - Trades + PnL (market can be closed)")
    print("=" * 70)

    model, scaler_y, scaler_X = load_lstm_model()
    feat_list = load_feature_list(FEATURE_LIST_PATH)
    
    print(f"[REPLAY] Loaded feature list ({len(feat_list)}): {feat_list}")

    df_raw = download_qqq_data(days=days)
    if df_raw.empty:
        print("[ERROR] No yfinance data.")
        return

    # RTH filter via Alpaca calendar (matches typical training on RTH)
    end_dt = df_raw.index[-1].tz_convert(EASTERN).to_pydatetime()
    start_dt = (df_raw.index[0].tz_convert(EASTERN) - pd.Timedelta(days=3)).to_pydatetime()
    try:
        cal_map = build_calendar_map(start_dt, end_dt)
        df_rth = df_raw[df_raw.index.to_series().map(lambda ts: is_rth(ts, cal_map))]
    except Exception as e:
        print(f"[WARN] Calendar fetch failed ({e}). Using full data (no RTH filter).")
        df_rth = df_raw
    
    if df_rth.empty or len(df_rth) < SEQUENCE_LENGTH + 50:
        print("[ERROR] Not enough RTH bars for replay.")
        return

    # Build base features once
    df_feat, _ = build_features_no_news(df_rth)

    # Flatten multi-index if needed
    if len(df_feat.columns) > 0 and isinstance(df_feat.columns[0], tuple):
        df_feat.columns = [c[0] if isinstance(c, tuple) else c for c in df_feat.columns]

    # Fetch news once + align per bar timestamp
    provider = NewsFeatureProvider(decay_lambda=0.001, cache_minutes=999999)
    try:
        df_news = provider.fetch_news_df_once(tickers=["QQQ"], topics=None)
        if df_news.empty:
            print("[REPLAY] No news returned.")
    except Exception as e:
        print(f"[REPLAY] News fetch failed: {e}")
        df_news = pd.DataFrame()

    # Sort news
    if not df_news.empty:
        df_news = df_news.sort_values("timestamp", ascending=True)

    # Enhance DataFrame with news features
    print("[REPLAY] Adding news features to all bars...")
    
    news_s, news_age, news_eff = [], [], []
    
    # We can do a faster merge_asof here like in backtest, but for Replay simplicity let's do robust loop or asof
    if not df_news.empty:
        df_news_idx = df_news.set_index("timestamp").sort_index()
        # Create asof merge
        merged = pd.merge_asof(
             df_feat.sort_index(), df_news_idx[["sentiment_score"]], left_index=True, right_index=True, direction='backward'
        )
        merged["sentiment_score"] = merged["sentiment_score"].fillna(0.0)
        
        # Calculate Age
        # We need news timestamp for age.
        df_news_idx["pts"] = df_news_idx.index
        merged_ts = pd.merge_asof(
             df_feat.sort_index(), df_news_idx[["pts"]], left_index=True, right_index=True, direction='backward'
        )
        # Age
        bar_ts = merged_ts.index
        news_ts = merged_ts["pts"]
        age_s = (bar_ts - news_ts).dt.total_seconds() / 60.0
        age_s = age_s.fillna(0.0)
        
        sent_s = merged["sentiment_score"]
        eff_s = sent_s * np.exp(-provider.decay_lambda * age_s)
        
        df_feat["last_news_sentiment"] = sent_s
        df_feat["news_age_minutes"] = age_s
        df_feat["effective_sentiment_t"] = eff_s
    else:
        df_feat["last_news_sentiment"] = 0.0
        df_feat["news_age_minutes"] = 0.0
        df_feat["effective_sentiment_t"] = 0.0

    # Build X matrix exactly in training schema order
    X_list = []
    
    # Ensure all features exist
    for feat in feat_list:
        if feat not in df_feat.columns:
            print(f"[ERROR] Required feature {feat} not in columns: {list(df_feat.columns)}")
            return
        X_list.append(df_feat[feat].values.astype(np.float32))
        
    X_raw = np.column_stack(X_list).astype(np.float32)
    # Scale X
    X = scaler_X.transform(X_raw)

    # Prices (yfinance raw df columns)
    closes = df_rth["Close"].copy()
    opens = df_rth["Open"].copy()

    # reset sim
    sim_state["cash"] = 100000.0
    sim_state["qty"] = 0
    sim_state["entry_price"] = None
    sim_state["entry_time"] = None
    sim_state["sl"] = None
    sim_state["tp"] = None
    sim_state["trades"] = []

    start_i = SEQUENCE_LENGTH
    # We can go until the end, but we need t+1 for valid fill simulation
    end_i = len(df_feat) - 2 

    print(f"[REPLAY] Simulating {end_i - start_i} steps...")

    for i in range(start_i, end_i):
        t = df_feat.index[i]
        t_next = df_feat.index[i + 1]

        price_t = float(closes.asof(t))
        price_next_open = float(opens.asof(t_next))

        now_dt = t.to_pydatetime()
        if now_dt.tzinfo is None:
            now_dt = now_dt.replace(tzinfo=timezone.utc)

        # bracket check @ close(t)
        sim_check_bracket(price_t, now_dt)

        # predict at t
        X_seq = X[i - SEQUENCE_LENGTH + 1 : i + 1]
        X_seq = np.expand_dims(X_seq, axis=0) # (1, 50, 14)

        with torch.no_grad():
            pred_scaled = model(torch.from_numpy(X_seq).float().to(DEVICE)).cpu().numpy()[0]
        pred = scaler_y.inverse_transform([pred_scaled])[0]

        s, r3 = calc_signal(pred)

        # ENTRY fill @ open(t+1)
        if not sim_in_position():
            if (s > ENTRY_THRESHOLD) and (r3 > 0):
                equity = sim_state["cash"]
                target_value = equity * POSITION_SIZE_PCT
                qty = int(target_value / price_next_open)
                if qty > 0:
                    sl = price_next_open * (1 + STOP_LOSS_PCT)
                    tp = price_next_open * (1 + TAKE_PROFIT_PCT)
                    sim_buy(qty, price_next_open, t_next.to_pydatetime(), sl, tp)

        # EXIT fill @ open(t+1)
        else:
            exit_now, reason = should_exit_sim(s, r3, now_dt)
            if exit_now:
                sim_close(price_next_open, t_next.to_pydatetime(), reason)

    sells = [x for x in sim_state["trades"] if x["type"] == "SELL"]
    total_pnl = sum(x.get("pnl", 0.0) for x in sells)
    wins = sum(1 for x in sells if x.get("pnl", 0.0) > 0)
    n = len(sells)

    print("=" * 70)
    print(f"[REPLAY RESULT] Trades closed: {n} | Wins: {wins} | Winrate: {(wins/n*100 if n else 0):.1f}%")
    print(f"[REPLAY RESULT] Total PnL: {total_pnl:.2f} | End cash: {sim_state['cash']:.2f}")
    print("=" * 70)


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
    last_completed_price = float(df_rth["Close"].iloc[-2])

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
    X = scaler_X.transform(X_raw)
    
    # 6. Seq
    X_seq = create_last_sequence(X, SEQUENCE_LENGTH)
    if X_seq.size == 0:
        print("[ERROR] not enough data for seq")
        return
        
    X_tensor = torch.from_numpy(X_seq).float().to(DEVICE) # (1, 50, 14)

    with torch.no_grad():
        pred_scaled = model(X_tensor).cpu().numpy()[0]
    pred = scaler_y.inverse_transform([pred_scaled])[0]

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

    # Replay args
    ap.add_argument("--replay", action="store_true", help="Replay backtest on recent minute bars")
    ap.add_argument("--replay-days", type=int, default=5, help="How many days of 1m data to replay")

    args = ap.parse_args()

    if args.replay:
        run_replay(days=args.replay_days)
        return

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
