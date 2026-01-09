"""
Broker Adapters for Trading System
===================================
Provides unified interface for different brokers (Alpaca for stocks/ETFs, OANDA for CFDs).
"""

from abc import ABC, abstractmethod
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple
import os
import requests

from .strategy_config import BrokerInterface


from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    MarketOrderRequest, 
    LimitOrderRequest, 
    TakeProfitRequest, 
    StopLossRequest, 
    GetOrdersRequest, 
    GetCalendarRequest,
    ClosePositionRequest
)
from alpaca.trading.enums import OrderSide, TimeInForce, OrderType, OrderClass, QueryOrderStatus

from .strategy_config import BrokerInterface


class AlpacaBroker(BrokerInterface):
    """Alpaca broker adapter for stocks and ETFs (Paper + Live trading) using alpaca-py"""
    
    def __init__(self, api_key: str, secret_key: str, base_url: str = None, paper: bool = True):
        self.api_key = api_key
        self.secret_key = secret_key
        self.paper = paper
        
        # Initialize the alpaca-py TradingClient
        # Note: base_url is handled by the client if we don't provide it, 
        # but we can pass it if needed for custom endpoints.
        self.client = TradingClient(api_key, secret_key, paper=paper)
        print(f"[ALPACA] Initialized {'Paper' if paper else 'Live'} trading via alpaca-py")
    
    def get_account_info(self) -> dict:
        account = self.client.get_account()
        # Map to common format expected by the system
        return {
            "equity": float(account.equity),
            "cash": float(account.cash),
            "buying_power": float(account.buying_power),
            "status": account.status
        }
    
    def get_positions(self) -> List[dict]:
        positions = self.client.get_all_positions()
        result = []
        for p in positions:
            result.append({
                "symbol": p.symbol,
                "qty": int(p.qty),
                "side": p.side,
                "avg_entry_price": float(p.avg_entry_price),
                "unrealized_pl": float(p.unrealized_pl)
            })
        return result
    
    def get_position(self, symbol: str) -> Optional[dict]:
        try:
            p = self.client.get_open_position(symbol)
            return {
                "symbol": p.symbol,
                "qty": int(p.qty),
                "side": p.side,
                "avg_entry_price": float(p.avg_entry_price),
                "unrealized_pl": float(p.unrealized_pl)
            }
        except Exception:
            return None
    
    def submit_order(self, order_params: dict) -> Optional[dict]:
        """Submit a simple order using alpaca-py"""
        try:
            # Check if it's a bracket order payload or a simple one
            if order_params.get("order_class") == "bracket":
                return self.submit_bracket_order(
                    symbol=order_params["symbol"],
                    qty=int(order_params["qty"]),
                    side=order_params["side"],
                    sl_price=float(order_params["stop_loss"]["stop_price"]),
                    tp_price=float(order_params["take_profit"]["limit_price"])
                )

            side = OrderSide.BUY if order_params.get("side") == "buy" else OrderSide.SELL
            
            request = MarketOrderRequest(
                symbol=order_params["symbol"],
                qty=int(order_params["qty"]),
                side=side,
                time_in_force=TimeInForce.DAY
            )
            order = self.client.submit_order(order_data=request)
            return {"id": str(order.id), "status": order.status}
        except Exception as e:
            print(f"[ALPACA ERROR] submit_order failed: {e}")
            return None
    
    def submit_bracket_order(self, symbol: str, qty: int, side: str, 
                            sl_price: float, tp_price: float) -> Optional[dict]:
        """Submit a bracket order using alpaca-py"""
        try:
            order_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL
            
            # Submitting bracket order
            request = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=order_side,
                time_in_force=TimeInForce.DAY,
                order_class=OrderClass.BRACKET,
                take_profit=TakeProfitRequest(limit_price=round(tp_price, 2)),
                stop_loss=StopLossRequest(stop_price=round(sl_price, 2))
            )
            
            order = self.client.submit_order(order_data=request)
            print(f"[ALPACA ORDER] {side.upper()} {qty} {symbol} | SL={sl_price:.2f} TP={tp_price:.2f}")
            return {"id": str(order.id), "status": order.status}
        except Exception as e:
            print(f"[ALPACA ERROR] submit_bracket_order failed: {e}")
            return None
    
    def close_position(self, symbol: str) -> bool:
        try:
            self.client.close_position(symbol)
            print(f"[ALPACA] Closed position: {symbol}")
            return True
        except Exception as e:
            print(f"[ALPACA ERROR] close_position failed: {e}")
            return False
    
    def get_last_fill_time(self, symbol: str, side: str) -> Optional[datetime]:
        try:
            order_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL
            request = GetOrdersRequest(
                status=QueryOrderStatus.CLOSED,
                limit=50,
                side=order_side,
                symbols=[symbol]
            )
            orders = self.client.get_orders(filter=request)
            
            last_dt = None
            for o in orders:
                if o.status.value != "filled":
                    continue
                
                if o.filled_at:
                    dt = o.filled_at.astimezone(timezone.utc)
                    if last_dt is None or dt > last_dt:
                        last_dt = dt
            
            return last_dt
        except Exception as e:
            print(f"[ALPACA WARN] get_last_fill_time failed: {e}")
            return None
    
    def get_calendar(self, start_date: str, end_date: str) -> List[dict]:
        try:
            request = GetCalendarRequest(start=start_date, end=end_date)
            calendar = self.client.get_calendar(filters=request)
            
            result = []
            for c in calendar:
                result.append({
                    "date": str(c.date),
                    "open": str(c.open),
                    "close": str(c.close)
                })
            return result
        except Exception as e:
            print(f"[ALPACA ERROR] get_calendar failed: {e}")
            return []


class OANDABroker(BrokerInterface):
    """OANDA broker adapter for CFD trading (Practice + Live)"""
    
    def __init__(self, api_key: str, account_id: str, practice: bool = True):
        self.api_key = api_key
        self.account_id = account_id
        
        if practice:
            self.base_url = "https://api-fxpractice.oanda.com"
        else:
            self.base_url = "https://api-fxtrade.oanda.com"
        
        self.practice = practice
        print(f"[OANDA] Initialized {'Practice' if practice else 'Live'} trading")
        print(f"[OANDA] Account ID: {account_id}")
    
    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
    
    def get_account_info(self) -> dict:
        r = requests.get(f"{self.base_url}/v3/accounts/{self.account_id}", 
                        headers=self._headers(), timeout=30)
        r.raise_for_status()
        data = r.json()
        account = data.get("account", {})
        
        # Map to common format
        return {
            "equity": float(account.get("NAV", 0)),
            "cash": float(account.get("balance", 0)),
            "margin_available": float(account.get("marginAvailable", 0)),
            "margin_used": float(account.get("marginUsed", 0)),
            "unrealized_pl": float(account.get("unrealizedPL", 0)),
        }
    
    def get_positions(self) -> List[dict]:
        r = requests.get(f"{self.base_url}/v3/accounts/{self.account_id}/openPositions", 
                        headers=self._headers(), timeout=30)
        r.raise_for_status()
        data = r.json()
        
        positions = []
        for pos in data.get("positions", []):
            # OANDA has long and short separately
            long_units = int(pos.get("long", {}).get("units", 0))
            short_units = int(pos.get("short", {}).get("units", 0))
            
            if long_units != 0:
                positions.append({
                    "symbol": pos.get("instrument"),
                    "qty": long_units,
                    "side": "long",
                    "avg_entry_price": float(pos.get("long", {}).get("averagePrice", 0)),
                    "unrealized_pl": float(pos.get("long", {}).get("unrealizedPL", 0)),
                })
            if short_units != 0:
                positions.append({
                    "symbol": pos.get("instrument"),
                    "qty": abs(short_units),
                    "side": "short",
                    "avg_entry_price": float(pos.get("short", {}).get("averagePrice", 0)),
                    "unrealized_pl": float(pos.get("short", {}).get("unrealizedPL", 0)),
                })
        
        return positions
    
    def get_position(self, symbol: str) -> Optional[dict]:
        positions = self.get_positions()
        for pos in positions:
            if pos.get("symbol") == symbol:
                return pos
        return None
    
    def submit_order(self, order_params: dict) -> Optional[dict]:
        """Submit an order to OANDA"""
        symbol = order_params.get("symbol")
        units = order_params.get("qty", order_params.get("units", 0))
        side = order_params.get("side", "buy")
        
        # OANDA uses negative units for sell/short
        if side in ("sell", "short"):
            units = -abs(units)
        
        oanda_order = {
            "order": {
                "type": "MARKET",
                "instrument": symbol,
                "units": str(units),
                "timeInForce": "FOK",  # Fill or Kill
            }
        }
        
        # Add stop-loss if provided
        if "stop_loss" in order_params:
            sl = order_params["stop_loss"]
            oanda_order["order"]["stopLossOnFill"] = {
                "price": str(sl.get("stop_price", sl.get("price")))
            }
        
        # Add take-profit if provided
        if "take_profit" in order_params:
            tp = order_params["take_profit"]
            oanda_order["order"]["takeProfitOnFill"] = {
                "price": str(tp.get("limit_price", tp.get("price")))
            }
        
        try:
            r = requests.post(f"{self.base_url}/v3/accounts/{self.account_id}/orders",
                            headers=self._headers(), json=oanda_order, timeout=30)
            r.raise_for_status()
            result = r.json()
            print(f"[OANDA ORDER] {side.upper()} {abs(units)} {symbol}")
            return result
        except Exception as e:
            print(f"[OANDA ERROR] submit_order failed: {e}")
            return None
    
    def close_position(self, symbol: str) -> bool:
        """Close all positions for a symbol"""
        try:
            # Close long positions
            r = requests.put(
                f"{self.base_url}/v3/accounts/{self.account_id}/positions/{symbol}/close",
                headers=self._headers(),
                json={"longUnits": "ALL"},
                timeout=30
            )
            
            # Close short positions
            r2 = requests.put(
                f"{self.base_url}/v3/accounts/{self.account_id}/positions/{symbol}/close",
                headers=self._headers(),
                json={"shortUnits": "ALL"},
                timeout=30
            )
            
            print(f"[OANDA] Closed position: {symbol}")
            return True
        except Exception as e:
            print(f"[OANDA ERROR] close_position failed: {e}")
            return False
    
    def get_last_fill_time(self, symbol: str, side: str) -> Optional[datetime]:
        """Get the last fill time for a symbol and side"""
        try:
            r = requests.get(
                f"{self.base_url}/v3/accounts/{self.account_id}/trades",
                headers=self._headers(),
                params={"instrument": symbol, "state": "ALL", "count": 50},
                timeout=30
            )
            r.raise_for_status()
            trades = r.json().get("trades", [])
            
            for trade in trades:
                trade_side = "buy" if int(trade.get("currentUnits", 0)) > 0 else "sell"
                if trade_side == side:
                    open_time = trade.get("openTime")
                    if open_time:
                        return datetime.fromisoformat(open_time.replace("Z", "+00:00"))
            
            return None
        except Exception as e:
            print(f"[OANDA WARN] get_last_fill_time failed: {e}")
            return None
    
    def get_candles(self, symbol: str, granularity: str = "M1", count: int = 500) -> List[dict]:
        """Get candlestick data from OANDA"""
        r = requests.get(
            f"{self.base_url}/v3/instruments/{symbol}/candles",
            headers=self._headers(),
            params={"granularity": granularity, "count": count, "price": "M"},
            timeout=30
        )
        r.raise_for_status()
        return r.json().get("candles", [])


def create_broker(api_type: str, keys: dict) -> BrokerInterface:
    """
    Factory function to create the appropriate broker based on api_type.
    
    Args:
        api_type: "alpaca" or "oanda"
        keys: Dictionary containing API keys
        
    Returns:
        BrokerInterface instance
    """
    if api_type == "alpaca":
        api_key = keys.get("APCA-API-KEY-ID-Paper") or keys.get("ALPACA_KEY_ID")
        secret = keys.get("APCA-API-SECRET-KEY-Paper") or keys.get("ALPACA_SECRET")
        base_url = keys.get("ALPACA_BASE", "https://paper-api.alpaca.markets")
        
        if not api_key or not secret:
            raise ValueError("Missing Alpaca API keys. Set in conf/keys.yaml")
        
        return AlpacaBroker(api_key, secret, base_url, paper=True)
    
    elif api_type == "oanda":
        api_key = keys.get("OANDA_API_KEY")
        account_id = keys.get("OANDA_ACCOUNT_ID")
        
        if not api_key or not account_id:
            raise ValueError(
                "Missing OANDA API keys. Add to conf/keys.yaml:\n"
                "  OANDA_API_KEY: your-api-key\n"
                "  OANDA_ACCOUNT_ID: your-account-id\n\n"
                "Get your Practice account at: https://www.oanda.com/demo-account/"
            )
        
        return OANDABroker(api_key, account_id, practice=True)
    
    else:
        raise ValueError(f"Unknown api_type: {api_type}. Supported: alpaca, oanda")
