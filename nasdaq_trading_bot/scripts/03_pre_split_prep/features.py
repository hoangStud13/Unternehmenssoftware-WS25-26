
from __future__ import annotations

import pandas as pd

class FeatureBuilder:
    """
        Baut alle Features, die später ins ML-Modell gehen.
        Erwartet einen DataFrame mit mindestens einer 'price'-Spalte
        und einer Datetime-Index oder einer 'timestamp'-Spalte.
    """
    def __init__(
            self,
            return_windows: [],
            price_col: str = 'open',
            timestamp_col: str = 'timestamp',
            news_col: str = 'sentiment'
            ):

            self.return_windows = return_windows
            self.price_col = price_col
            self.timestamp_col = timestamp_col
            self.news_col = news_col

    def _add_simple_return(self,df: pd.DataFrame):

        return 0

    def _calculate_ema(self,df: pd.DataFrame):
        return 0

    def _calculate_z_score(self,df: pd.DataFrame):
        # Deine implementierung hier
        return 0

    def _calculate_trade_volume(self,df: pd.DataFrame):
        # Deine implementierung hier
        return 0