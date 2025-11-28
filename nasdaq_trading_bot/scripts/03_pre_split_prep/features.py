from __future__ import annotations

import numpy as np
import pandas as pd
from typing import List

class FeatureBuilder:
    """
    Baut alle Features, die später ins ML-Modell gehen.
    Erwartet einen DataFrame mit mindestens einer 'price'-Spalte
    und einer Datetime-Index oder einer 'timestamp'-Spalte.
    """

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
            self.df[self.timestamp_col] = pd.to_datetime(self.df[self.timestamp_col])
            self.df = self.df.set_index(self.timestamp_col)

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
        self.df['ema_diff'] = self.df['ema_5'] - self.df['ema_20']

    # Z-Score
    def _calculate_z_score(self, window: int = 20):
        return 0

    # Handelsvolumen Feature
    def _calculate_trade_volume(self, col: str = 'volume'):
        return 0


    # Realisierte Volatilität berechnen
    def _calculate_realized_volatility(self):
        self.df['log_return'] = np.log(self.df[self.price_col]) / np.log(self.df[self.price_col].shift(1))
        self.df['realized_volatility'] = self.df['log_return'].rolling(window=20).std()

    # High Low Spannweite
    def _calculate_hl_span(self):
        self.df['hl_span'] = self.df['high'] - self.df['low']


    # Funktion, um alle Features zu bauen
    def build_features(self):
        self._add_simple_return()
        self._calculate_ema()
        self._calculate_z_score()
        self._calculate_trade_volume()
        self._calculate_realized_volatility()
        self._calculate_ema_diff()
        self._calculate_hl_span()

        columns_to_keep = ['simple_return_{window}m']

        return self.df  # gebe den DataFrame zurück
