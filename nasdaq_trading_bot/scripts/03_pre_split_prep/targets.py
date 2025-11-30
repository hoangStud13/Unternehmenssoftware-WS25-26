import pandas as pd
import numpy as np
from typing import List

class TargetBuilder:
    """
    Calculates targets for the ML model, specifically future VWAP.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        # Ensure index is datetime and sorted
        if not isinstance(self.df.index, pd.DatetimeIndex):
            if 'timestamp' in self.df.columns:
                self.df['timestamp'] = pd.to_datetime(self.df['timestamp'], utc=True)
                self.df = self.df.set_index('timestamp')
            else:
                self.df.index = pd.to_datetime(self.df.index, utc=True)
        self.df = self.df.sort_index()

    def calculate_vwap_targets(self, windows: List[int] = [1, 3, 4, 5, 10, 15]) -> pd.DataFrame:
        """
        Calculates future VWAP for the specified windows.
        Formula: Sum(Price * Volume) / Sum(Volume) over the next N minutes.
        """
        print("Calculating VWAP targets...")
        
        # Pre-calculate Price * Volume
        # Use existing VWAP if available for better accuracy, otherwise fallback to open
        price_col = 'vwap' if 'vwap' in self.df.columns else 'open'
        self.df['pv'] = self.df[price_col] * self.df['volume']
        
        # We need to look forward. 
        # Rolling window in pandas looks backward. 
        # So we reverse the dataframe, calculate rolling, and reverse back.
        
        df_reversed = self.df.iloc[::-1]
        
        for window in windows:
            # Rolling sum of PV and Volume over the window
            # We use 'min_periods=1' to get values even if we are near the end (though those might be less reliable)
            # But typically for targets we want full windows. Let's stick to default min_periods=window?
            # Actually, for targets, if we don't have data, we should probably have NaN.
            
            # Note: rolling(window) includes the current row. 
            # If we want "next N minutes", does it include current minute?
            # Usually "target 5m" means what happens in the next 5 minutes.
            # If we are at t, we want data from t+1 to t+window.
            # Reversing and rolling(window) at index t (which was original index T-t) 
            # will give sum from T-t to T-t-(window-1).
            # Let's verify logic.
            
            # Example: [0, 1, 2, 3, 4, 5]
            # Reverse: [5, 4, 3, 2, 1, 0]
            # Rolling(2) at 1 (orig 4): sum(1, 0) -> sum of orig 4 and 5.
            # This includes current row.
            # If we want strictly future, we should shift.
            
            # Let's calculate rolling sum including current, then shift.
            # If we want target for time t, based on t+1...t+k.
            
            roll_pv = df_reversed['pv'].rolling(window=window).sum()
            roll_vol = df_reversed['volume'].rolling(window=window).sum()
            
            # Reverse back
            roll_pv = roll_pv.iloc[::-1]
            roll_vol = roll_vol.iloc[::-1]
            
            # Calculate VWAP (Absolute)
            vwap_future = roll_pv / roll_vol
            
            # Shift backwards to align: 
            # Currently vwap at t represents sum(t...t+window-1).
            # If we want target to be strictly future (t+1...t+window), we shift -1.
            vwap_future = vwap_future.shift(-1)
            
            # Calculate Percentage Return Target
            # Target = (Future_VWAP - Current_Price) / Current_Price
            # We use the same price_col as current price reference
            current_price = self.df[price_col]
            
            target_col = f'target_return_{window}m'
            self.df[target_col] = (vwap_future - current_price) / current_price * 100
            
        # Drop temporary column
        self.df.drop(columns=['pv'], inplace=True)
        
        return self.df
