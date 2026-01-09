import os
import sys
import time
import yaml
from datetime import datetime, timezone

# Add project root to path
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, THIS_DIR)

from strategies.broker_adapters import AlpacaBroker

def main():
    # Load keys
    keys_path = os.path.join(PROJECT_ROOT, "conf", "keys.yaml")
    with open(keys_path, "r") as f:
        keys = yaml.safe_load(f)
    
    api_key = keys["KEYS"]["APCA-API-KEY-ID-Paper"]
    secret_key = keys["KEYS"]["APCA-API-SECRET-KEY-Paper"]
    
    print("--- STARTING ALPACA-PY TEST ---")
    broker = AlpacaBroker(api_key, secret_key, paper=True)
    
    # 1. Get Account Info
    print("\n[STEP 1] Getting Account Info...")
    acct = broker.get_account_info()
    print(f"Equity: ${acct['equity']:.2f}")
    print(f"Cash:   ${acct['cash']:.2f}")
    
    SYMBOL = "QQQ"
    
    # Clean up existing positions first
    print(f"\n[CLEANUP] Closing existing positions for {SYMBOL}...")
    broker.close_position(SYMBOL)
    time.sleep(2)
    
    # 2. Test LONG Order
    print(f"\n[STEP 2] Testing LONG Order (BUY 1 {SYMBOL})...")
    order = broker.submit_order({
        "symbol": SYMBOL,
        "qty": 1,
        "side": "buy"
    })
    if order:
        print(f"Order submitted: ID={order['id']}")
        time.sleep(5) # Wait for fill
        pos = broker.get_position(SYMBOL)
        if pos:
            print(f"Position active: {pos['qty']} shares of {pos['symbol']} @ {pos['avg_entry_price']}")
        else:
            print("Position not found yet.")
    
    # 3. Close Long Position
    print(f"\n[STEP 3] Closing LONG position...")
    if broker.close_position(SYMBOL):
        print("Close command sent.")
    time.sleep(5)
    
    # 4. Test SHORT Order
    # Note: Shorting QQQ might require specific margin settings, 
    # but we can try a SELL order if there's no position.
    print(f"\n[STEP 4] Testing SHORT Order (SELL 1 {SYMBOL})...")
    order = broker.submit_order({
        "symbol": SYMBOL,
        "qty": 1,
        "side": "sell"
    })
    if order:
        print(f"Order submitted: ID={order['id']}")
        time.sleep(5) # Wait for fill
        pos = broker.get_position(SYMBOL)
        if pos:
            print(f"Position active: {pos['qty']} shares of {pos['symbol']} (Negative indicates short in some brokers)")
        else:
            print("Position not found yet (Shorting might not be possible on this account).")

    # 5. Final Cleanup
    print(f"\n[STEP 5] Final Cleanup...")
    broker.close_position(SYMBOL)
    print("Done.")

if __name__ == "__main__":
    main()
