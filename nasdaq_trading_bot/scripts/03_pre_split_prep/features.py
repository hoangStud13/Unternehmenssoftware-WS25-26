from __future__ import annotations

import os
import pandas as pd
import numpy as np
from datetime import datetime
import pytz
from typing import List

class FeatureBuilder:
    """
    Baut alle Features, die später ins ML-Modell gehen.
    Erwartet einen DataFrame mit mindestens einer 'price'-Spalte
    und einer Datetime-Index oder einer 'timestamp'-Spalte.
    """

    # Configuration
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
    data_dir = os.path.join(project_root, 'data')

    def __init__(
        self,
        df: pd.DataFrame,
        ema_windows: List[int],
        return_windows: List[int],
        price_col: str = 'open',
        timestamp_col: str = 'timestamp',
        news_col: str = 'sentiment'
    ) -> None:
        self.df = df.copy()  # DataFrame intern speichern
        self.ema_windows = ema_windows
        self.return_windows = return_windows
        self.price_col = price_col
        self.timestamp_col = timestamp_col
        self.news_col = news_col

        # optional: sicherstellen, dass Timestamp korrekt als Index gesetzt ist
        if self.timestamp_col in self.df.columns:
            self.df[self.timestamp_col] = pd.to_datetime(self.df[self.timestamp_col], utc=True)
            self.df = self.df.set_index(self.timestamp_col)
        
        # Ensure index is datetime and sorted
        if not isinstance(self.df.index, pd.DatetimeIndex):
            # If index is not datetime, try to convert it (assuming it was set above or passed as index)
            self.df.index = pd.to_datetime(self.df.index, utc=True)
            
        self.df = self.df.sort_index()

    # Beispiel: einfache Returns berechnen
    def _add_simple_return(self):
        for window in self.return_windows:
            self.df[f'simple_return_{window}m'] = self.df[self.price_col].pct_change(window)


    # EMA
    def _calculate_ema(self, span: int = 10):
        for window in self.ema_windows:
            self.df[f'ema_{window}'] = self.df[self.price_col].ewm(span=window,adjust=False).mean()

    # EMA(5) - EMA(20)
    def _calculate_ema_diff(self):
        # Check if columns exist first
        if 'ema_5' in self.df.columns and 'ema_20' in self.df.columns:
            self.df['ema_diff'] = self.df['ema_5'] - self.df['ema_20']
        else:
            # Calculate if missing
            self.df['ema_5'] = self.df[self.price_col].ewm(span=5, adjust=False).mean()
            self.df['ema_20'] = self.df[self.price_col].ewm(span=20, adjust=False).mean()
            self.df['ema_diff'] = self.df['ema_5'] - self.df['ema_20']

    # Z-Score (Volume based)
    def _calculate_z_score(self, window: int = 30):
        # Rolling Mean und Std mit shift(1) um nur vergangene Daten zu nutzen
        col = 'volume'
        if col not in self.df.columns:
            print(f"Warning: Column {col} missing for Z-score calculation.")
            return

        rolling_mean = self.df[col].rolling(window=window).mean().shift(1)
        rolling_std = self.df[col].rolling(window=window).std().shift(1)
        self.df[f'volume_zscore_{window}m'] = (self.df[col] - rolling_mean) / rolling_std

    # Handelsvolumen Feature
    def _calculate_trade_volume(self, col: str = 'volume', trades_col: str = 'trade_count'):
        # Durchschnittliches Volumen pro Trade = Gesamtvolumen / Anzahl der Trades
        if col in self.df.columns and trades_col in self.df.columns:
            self.df['avg_volume_per_trade'] = self.df[col] / self.df[trades_col]
        else:
            print(f"Warning: Columns {col} or {trades_col} missing for trade volume calculation.")


    # Realisierte Volatilität berechnen
    def _calculate_realized_volatility(self):
        self.df['log_return'] = np.log(self.df[self.price_col]) / np.log(self.df[self.price_col].shift(1))
        self.df['realized_volatility'] = self.df['log_return'].rolling(window=20).std()

    # High Low Spannweite
    def _calculate_hl_span(self):
        if 'high' in self.df.columns and 'low' in self.df.columns:
            self.df['hl_span'] = self.df['high'] - self.df['low']

    # News Sentiment 
    def _align_news_with_price(self):
        """
        Align news sentiment with price data and apply exponential decay.
        """
        print("Aligning news data...")

        news_file = os.path.join(FeatureBuilder.data_dir, 'nasdaq_news_5y_with_sentiment.csv')
        
        if not os.path.exists(news_file):
            print(f"Warning: News file not found at {news_file}. Skipping news alignment.")
            return

        # Decay parameter (λ)
        LAMBDA = 0.001 
        
        # Load news data
        news_df = pd.read_csv(news_file)
        news_df['created_at'] = pd.to_datetime(news_df['created_at'], utc=True)
        news_df = news_df.sort_values('created_at').reset_index(drop=True)
        
        # Initialize columns for aligned data
        self.df['last_news_sentiment'] = 0.0
        self.df['news_age_minutes'] = np.nan
        self.df['effective_sentiment_t'] = 0.0
        self.df['news_id'] = None
        self.df['news_headline'] = None
        
        # For each price bar, find the most recent news
        news_idx = 0
        total_rows = len(self.df)
        
        # Iterate over the index (timestamps)
        for i, (timestamp, row) in enumerate(self.df.iterrows()):
            current_time = timestamp
            
            # Find the most recent news before or at current_time
            # Move news_idx forward as long as news is before current_time
            while news_idx < len(news_df) - 1 and news_df.iloc[news_idx + 1]['created_at'] <= current_time:
                news_idx += 1
            
            # Check if we have any news before this time
            if news_df.iloc[news_idx]['created_at'] <= current_time:
                news_row = news_df.iloc[news_idx]
                
                # Calculate news age in minutes
                news_age = (current_time - news_row['created_at']).total_seconds() / 60.0
                
                # Get sentiment
                sentiment = news_row['sentiment_score']
                
                # Apply exponential decay
                effective_sentiment = sentiment * np.exp(-LAMBDA * news_age)
                
                # Store values
                self.df.at[timestamp, 'last_news_sentiment'] = sentiment
                self.df.at[timestamp, 'news_age_minutes'] = news_age
                self.df.at[timestamp, 'effective_sentiment_t'] = effective_sentiment
                self.df.at[timestamp, 'news_id'] = news_row['id']
                self.df.at[timestamp, 'news_headline'] = news_row['headline']
            
            # Progress indicator
            if (i + 1) % 100000 == 0:
                print(f"Processed {i + 1:,} / {total_rows:,} bars ({(i+1)/total_rows*100:.1f}%)")
        
        print("News alignment complete.")


    # Funktion, um alle Features zu bauen
    def build_features_before_split(self):
        self._add_simple_return()
        self._calculate_ema()
        self._calculate_z_score()
        self._calculate_trade_volume()
        self._calculate_realized_volatility()
        self._calculate_ema_diff()
        self._calculate_hl_span()
        self._align_news_with_price()


        cols_return = [c for c in self.df.columns if "simple_return_" in c]
        cols_ema = [c for c in self.df.columns if "ema_" in c]
        
        # Manually specify other columns we calculated
        other_cols = ["timestamp", "vwap", "volume", "volume_zscore_30m", "realized_volatility", "avg_volume_per_trade", "hl_span",
                      "last_news_sentiment", "news_age_minutes", "effective_sentiment_t", "news_id"]
        
        # Filter to ensure they exist
        other_cols = [c for c in other_cols if c in self.df.columns]

        # Alle zusammenführen
        columns_to_keep = cols_return + cols_ema + other_cols
        df_features = self.df[columns_to_keep]
        output_parquet = os.path.join(FeatureBuilder.data_dir, 'nasdaq100_index_1m_features.parquet')
        output_csv = os.path.join(FeatureBuilder.data_dir, 'nasdaq100_index_1m_features.csv')
        df_features.to_csv(output_csv, index=True)   
        df_features.to_parquet(output_parquet, index=True)
        return df_features
        
