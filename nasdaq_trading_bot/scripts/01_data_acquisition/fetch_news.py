import os
import requests
import pandas as pd
from dotenv import load_dotenv
from datetime import datetime, timedelta
import time
import pytz

# Load environment variables
load_dotenv()
API_KEY = os.getenv('APCA_API_KEY_ID')
SECRET_KEY = os.getenv('APCA_API_SECRET_KEY')
BASE_URL = "https://data.alpaca.markets/v1beta1"

if not API_KEY or not SECRET_KEY:
    raise ValueError("API keys not found in .env file")

# Configuration
SYMBOLS = "QQQ,NDX"
tz = pytz.utc
END_DATE = datetime(2025, 11, 20, tzinfo=tz)
START_DATE = END_DATE - timedelta(days=5*365) # 5 Years back

# Output
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
data_dir = os.path.join(project_root, 'data')
os.makedirs(data_dir, exist_ok=True)
output_file = os.path.join(data_dir, 'nasdaq_news_5y.csv')

def fetch_news_raw():
    print(f"Starting full 5-year news fetch for {SYMBOLS} (Raw API)")
    
    start_str = START_DATE.strftime("%Y-%m-%dT%H:%M:%SZ")
    end_str = END_DATE.strftime("%Y-%m-%dT%H:%M:%SZ")
    
    print(f"Range: {start_str} to {end_str}")
    
    headers = {
        "APCA-API-KEY-ID": API_KEY,
        "APCA-API-SECRET-KEY": SECRET_KEY
    }
    
    # Initialize CSV with headers if it doesn't exist
    if not os.path.exists(output_file):
        pd.DataFrame(columns=["id", "headline", "summary", "author", "created_at", "updated_at", "url", "symbols", "source"]).to_csv(output_file, index=False)
    
    page_token = None
    total_fetched = 0
    
    while True:
        params = {
            "start": start_str,
            "end": end_str,
            "symbols": SYMBOLS,
            "limit": 50,
            "sort": "DESC", # Newest first
            "include_content": "false" # We don't need full HTML content
        }
        
        if page_token:
            params["page_token"] = page_token
            
        try:
            response = requests.get(f"{BASE_URL}/news", headers=headers, params=params)
            response.raise_for_status()
            data = response.json()
            
            news_items = data.get("news", [])
            
            if not news_items:
                print("No more items found.")
                break
                
            # Process and Save Chunk
            df_chunk = pd.DataFrame(news_items)
            
            # Ensure columns match our desired schema (handle missing cols gracefully)
            desired_cols = ["id", "headline", "summary", "author", "created_at", "updated_at", "url", "symbols", "source"]
            for col in desired_cols:
                if col not in df_chunk.columns:
                    df_chunk[col] = None
            
            # Select and reorder
            df_chunk = df_chunk[desired_cols]
            
            # Append to CSV
            df_chunk.to_csv(output_file, mode='a', header=False, index=False)
            
            total_fetched += len(news_items)
            print(f"Fetched {len(news_items)} items. Total: {total_fetched}", flush=True)
            
            page_token = data.get("next_page_token")
            if not page_token:
                break
                
            # Respect rate limits
            time.sleep(0.2)
            
        except Exception as e:
            print(f"Error fetching data: {e}")
            if 'response' in locals() and response.text:
                print(f"Response: {response.text}")
            # Wait a bit and retry or break? Let's break to avoid infinite loops on auth error
            break

    print(f"\nSUCCESS: Finished fetching. Total items: {total_fetched}")
    print(f"Saved to {output_file}")

if __name__ == "__main__":
    fetch_news_raw()
