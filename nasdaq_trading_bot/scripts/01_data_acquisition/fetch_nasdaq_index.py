import os
from dotenv import load_dotenv

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import Adjustment

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetCalendarRequest

import pandas as pd
from datetime import datetime, timedelta
import pytz

# ---------------------------------------------------
# 1) API-Keys aus .env laden
# ---------------------------------------------------
load_dotenv()
API_KEY = os.getenv('APCA_API_KEY_ID')
SECRET_KEY = os.getenv('APCA_API_SECRET_KEY')

if not API_KEY or not SECRET_KEY:
    raise ValueError("API keys not found in .env file")

# ---------------------------------------------------
# 2) Clients initialisieren (Data + Trading für Kalender)
# ---------------------------------------------------
data_client = StockHistoricalDataClient(api_key=API_KEY, secret_key=SECRET_KEY)
trading_client = TradingClient(API_KEY, SECRET_KEY)

# ---------------------------------------------------
# 3) Konfiguration
# ---------------------------------------------------
SYMBOL = 'QQQ'

# Static end date (damit der Download reproduzierbar bleibt)
eastern = pytz.timezone('US/Eastern')
END_DATE = eastern.localize(datetime(2025, 11, 20))
START_DATE = END_DATE - timedelta(days=5 * 365)  # ca. 5 Jahre

print(f"Fetching 1m bars for {SYMBOL} from {START_DATE} to {END_DATE} (RTH only)...")

# ---------------------------------------------------
# 4) US-Markt-Kalender holen (Feiertage + Öffnungszeiten)
# ---------------------------------------------------
# Für den Kalender reichen Datumsangaben (ohne Zeit)
cal_request = GetCalendarRequest(start=START_DATE.date(), end=END_DATE.date())
calendar = trading_client.get_calendar(cal_request)

# Map: date -> (open_dt, close_dt) in US/Eastern
cal_map = {}
for c in calendar:
    # c.open und c.close sind naive datetime in Eastern
    open_dt = eastern.localize(c.open)
    close_dt = eastern.localize(c.close)
    cal_map[c.date] = (open_dt, close_dt)

def is_regular_trading(ts: pd.Timestamp) -> bool:
    """
    Prüft, ob ein Timestamp während der regulären Handelszeiten liegt
    (unter Berücksichtigung von Wochenende & Feiertagen).
    Erwartet ts als UTC oder mit TZ-Info.
    """
    # Falls noch keine TZ: als UTC interpretieren
    if ts.tzinfo is None:
        ts_utc = ts.tz_localize("UTC")
    else:
        ts_utc = ts

    # In US/Eastern konvertieren
    ts_eastern = ts_utc.astimezone(eastern)
    d = ts_eastern.date()

    # Kein Eintrag im Kalender → Markt geschlossen (Wochenende/Feiertag)
    if d not in cal_map:
        return False

    open_dt, close_dt = cal_map[d]
    return open_dt <= ts_eastern < close_dt

# ---------------------------------------------------
# 5) Daten von Alpaca holen
# ---------------------------------------------------
request = StockBarsRequest(
    symbol_or_symbols=SYMBOL,
    timeframe=TimeFrame.Minute,
    adjustment=Adjustment.ALL,
    start=START_DATE,
    end=END_DATE
)

try:
    bars = data_client.get_stock_bars(request)

    # In DataFrame umwandeln
    df = bars.df
    df.reset_index(inplace=True)

    # 'symbol'-Spalte entfernen, wenn vorhanden
    if 'symbol' in df.columns:
        df.drop(columns=['symbol'], inplace=True)

    # ---------------------------------------------------
    # 6) Nur Regular Trading Hours behalten
    # ---------------------------------------------------
    if 'timestamp' not in df.columns:
        raise ValueError("Column 'timestamp' not found in dataframe")

    df['is_rth'] = df['timestamp'].map(is_regular_trading)
    df = df[df['is_rth']].drop(columns=['is_rth'])

    # ---------------------------------------------------
    # 7) Speichern
    # ---------------------------------------------------
    data_dir = 'data'
    os.makedirs(data_dir, exist_ok=True)

    output_parquet = os.path.join(data_dir, 'nasdaq100_index_1m.parquet')
    output_csv = os.path.join(data_dir, 'nasdaq100_index_1m.csv')

    df.to_parquet(output_parquet, index=False)
    df.to_csv(output_csv, index=False)

    print(f"Successfully saved {len(df)} RTH rows to:\n- {output_parquet}\n- {output_csv}")

except Exception as e:
    print(f"Error fetching data: {e}")
    if "forbidden" in str(e).lower() or "subscription" in str(e).lower():
        print("NOTE: If 'NDX' failed due to subscription limits, try 'QQQ' as a proxy.")
