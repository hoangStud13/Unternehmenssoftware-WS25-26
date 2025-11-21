import os
from dotenv import load_dotenv
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import Adjustment
import pandas as pd
from datetime import datetime, timedelta
import pytz

# Load environment variables
load_dotenv()
API_KEY = os.getenv('APCA_API_KEY_ID')
SECRET_KEY = os.getenv('APCA_API_SECRET_KEY')

if not API_KEY or not SECRET_KEY:
    raise ValueError("API keys not found in .env file")

# Initialize Alpaca client
client = StockHistoricalDataClient(api_key=API_KEY, secret_key=SECRET_KEY)

# Configuration
SYMBOL = 'QQQ'
# Static end date to match retrieved data (2025-11-20)
# This ensures we don't fetch new data if run again
tz = pytz.timezone('US/Eastern')
END_DATE = tz.localize(datetime(2025, 11, 20))
START_DATE = END_DATE - timedelta(days=5*365) # Approx 5 years

print(f"Fetching 1m bars for {SYMBOL} from {START_DATE} to {END_DATE}...")

# Create request
request = StockBarsRequest(
    symbol_or_symbols=SYMBOL,
    timeframe=TimeFrame.Minute,
    adjustment=Adjustment.ALL,
    start=START_DATE,
    end=END_DATE
)

try:
    # Fetch data
    bars = client.get_stock_bars(request)
    
    # Convert to DataFrame
    df = bars.df
    df.reset_index(inplace=True)
    
    # Basic cleanup
    if 'symbol' in df.columns:
        df.drop(columns=['symbol'], inplace=True)
    
    # Create data directory if it doesn't exist
    data_dir = 'data'
    os.makedirs(data_dir, exist_ok=True)

    # Save to Parquet
    output_parquet = os.path.join(data_dir, 'nasdaq100_index_1m.parquet')
    df.to_parquet(output_parquet, index=False)
    
    # Save to CSV
    output_csv = os.path.join(data_dir, 'nasdaq100_index_1m.csv')
    df.to_csv(output_csv, index=False)
    
    print(f"Successfully saved {len(df)} rows to:\n- {output_parquet}\n- {output_csv}")
    
except Exception as e:
    print(f"Error fetching data: {e}")
    # Fallback suggestion if NDX fails
    if "forbidden" in str(e).lower() or "subscription" in str(e).lower():
        print("NOTE: If 'NDX' failed due to subscription limits, try 'QQQ' as a proxy.")
