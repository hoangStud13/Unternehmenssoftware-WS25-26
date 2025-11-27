"""
Feature engineering utilities for pre-split data preparation.

This module derives a set of normalized price/volume features from minute bars to be
used for model training. It computes:
- Z-normalized VWAP and Volume
- Exponential moving averages (EMAs) over configured horizons
- Discrete slopes over multiple horizons for base series (VWAP, Volume) and EMAs
- Second-order slopes (slope-of-slope with t=1) for all slope features
- Z-normalization across all newly created features

Inputs/assumptions:
- Input DataFrame must contain at least the columns: 'vwap' and 'volume'.
- Index is treated positionally; timestamps are not required for the computations here.
- EMA and slope horizons are positive integers; z_norm_window should be large enough to provide
  a stable rolling mean/std.

Outputs:
- A copy of the original DataFrame with all engineered features appended as new columns.
- A list of the names of all added feature columns (for logging/metadata).

Note: The implementation keeps NaN rows created by rolling/slope operations until the end,
so callers can drop them after concatenation as needed (see main.py in the same folder).
"""

from __future__ import annotations

import pandas as pd

# Small epsilon to avoid division-by-zero in rolling standard deviation normalization.
eps = 1e-12


def generate_features(
    df: pd.DataFrame,
    ema_periods: list[int],
    slope_periods: list[int],
    z_norm_window: int = 1200,
) -> tuple[pd.DataFrame, list[str]]:
    """Create normalized TA features from bar data.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing at least 'vwap' and 'volume'.
    ema_periods : list[int]
        Periods (in bars) for which EMAs on 'vwap' are computed (e.g., [5, 20, 60]).
    slope_periods : list[int]
        Horizons (in bars) over which discrete slopes are computed for base series
        and EMAs (e.g., [1, 3, 5]).
    z_norm_window : int, default 1200
        Rolling window length used for z-normalization.

    Returns
    -------
    (df_with_features, features_added) : tuple[pd.DataFrame, list[str]]
        The augmented DataFrame and a list of added feature column names.

    Notes
    -----
    - This function does not modify the input DataFrame in-place; it returns a new copy with features.
    - NaNs introduced by rolling/slope operations are preserved. Downstream code can drop them once
      all features and targets are assembled to keep alignment.
    """
    # Work on a defensive copy to avoid side effects on the caller's DataFrame.
    df = df.copy()

    # Lightweight helpers kept local to constrain their scope.
    def ema(series: pd.Series, span: int) -> pd.Series:
        """Exponentially weighted moving average."""
        return series.ewm(span=span, adjust=False).mean()

    def slope(series: pd.Series, t: int = 1) -> pd.Series:
        """Difference per t bars (approximate slope)."""
        prev = series.shift(t)
        return (series - prev) / t

    def z_norm(series: pd.Series, window: int) -> pd.Series:
        """Rolling z-score."""
        mean = series.rolling(window).mean()
        std = series.rolling(window).std(ddof=0)
        return (series - mean) / (std + eps)

    # --- Prepare new feature container ---
    feats = pd.DataFrame(index=df.index)

    # --- Add z-normalized VWAP + Volume ---
    feats['VWAP_norm'] = z_norm(df['vwap'], z_norm_window)
    feats['Volume_norm'] = z_norm(df['volume'], z_norm_window)

    # --- EMAs ---
    horizons = ema_periods
    for h in horizons:
        feats[f'EMA_{h}'] = ema(df['vwap'], h)

    # --- Slopes for VWAP, Volume, EMAs ---
    base_cols = ['vwap', 'volume'] + [f'EMA_{h}' for h in horizons]

    for col in base_cols:
        # Source series may reside in the original df (for base cols) or in feats (for EMAs).
        source = df[col] if col in df.columns else feats[col]
        for t in slope_periods:
            feats[f'Slope_{col}_{t}'] = slope(source, t)

    # --- Slope of Slope (t=1 only) ---
    # Generate second-order slopes for all first-order slope features to capture acceleration.
    slope_cols = [c for c in feats.columns if c.startswith('Slope_')]
    slope_of_slope = pd.DataFrame(
        {f'Slope_{col}_1': slope(feats[col], 1) for col in slope_cols},
        index=df.index
    )

    # Add them in one go
    feats = pd.concat([feats, slope_of_slope], axis=1)

    # --- Z-normalization ---
    # Normalize all features except the already normalized VWAP/Volume to comparable scales.
    to_norm = [c for c in feats.columns if c not in ['VWAP_norm', 'Volume_norm']]
    feats[to_norm] = feats[to_norm].apply(lambda s: z_norm(s, z_norm_window))

    # --- Concatenate ---
    df = pd.concat([df, feats], axis=1)

    # Return also a list of new feature columns for traceability/metadata.
    features_added = feats.columns.tolist()

    return df, features_added
