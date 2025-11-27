import os
from datetime import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ---------------------------------------------------------------------
# Paths & setup
# ---------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
IMAGES_DIR = os.path.join(PROJECT_ROOT, "images")

os.makedirs(IMAGES_DIR, exist_ok=True)

sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)


# ---------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------
def load_news_data() -> pd.DataFrame:
    path = os.path.join(DATA_DIR, "nasdaq_news_5y_with_sentiment.csv")
    print(f"[INFO] Loading news data from {path}")
    df = pd.read_csv(path)

    # Timestamps in UTC
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True)

    # Sentiment-Kategorien
    conds = [
        df["sentiment_score"] < -0.2,
        df["sentiment_score"] > 0.2,
    ]
    df["sentiment_category"] = np.select(
        conds, ["Negative", "Positive"], default="Neutral"
    )
    return df


def load_index_data() -> pd.DataFrame:
    path = os.path.join(DATA_DIR, "nasdaq100_index_1m.csv")
    print(f"[INFO] Loading index data from {path}")
    df = pd.read_csv(path)

    # Alpaca liefert i.d.R. UTC – wir erzwingen das hier
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    # Zusatzspalte in US/Eastern für Trading-Hours-Filter
    df["timestamp_et"] = df["timestamp"].dt.tz_convert("US/Eastern")

    # Nur Regular Trading Hours (09:30–16:00)
    mask_rth = df["timestamp_et"].dt.time.between(time(9, 30), time(16, 0))
    df = df.loc[mask_rth].copy()

    df = df.sort_values("timestamp").reset_index(drop=True)
    print(f"  [INFO] Index rows after RTH filter: {len(df):,}")
    return df


# ---------------------------------------------------------------------
# Sentiment Distribution Plot
# ---------------------------------------------------------------------
def plot_sentiment_distribution(news_df: pd.DataFrame) -> None:
    print("[INFO] Plotting sentiment distribution ...")

    order = ["Negative", "Neutral", "Positive"]
    counts = news_df["sentiment_category"].value_counts().reindex(order).fillna(0)
    percentages = counts / counts.sum() * 100

    plt.figure(figsize=(8, 5))
    colors = ["red", "gray", "green"]
    bars = plt.bar(counts.index, counts.values, color=colors, alpha=0.7)

    for bar, count, pct in zip(bars, counts.values, percentages.values):
        h = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            h,
            f"{int(count)}\n({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    plt.title("News Sentiment Distribution")
    plt.xlabel("Sentiment Category")
    plt.ylabel("Number of Articles")
    plt.ylim(top=counts.max() * 1.20)
    plt.tight_layout()
    out = os.path.join(IMAGES_DIR, "sentiment_distribution.png")
    plt.savefig(out)
    plt.close()
    print(f"  [OK] Saved: {out}")


# ---------------------------------------------------------------------
# Hilfsfunktionen für Event-Study
# ---------------------------------------------------------------------
def _filter_events_to_intraday_window(
    news_subset: pd.DataFrame,
    window_before: int,
    window_after: int,
) -> pd.DataFrame:
    """
    Behalte nur News, bei denen das komplette Fenster [-before, +after]
    innerhalb der Regular Trading Hours liegt.

    RTH = 09:30–16:00 US/Eastern
    Daraus folgt für ±60 min:
        Event-Zeitfenster ungefähr 10:30–15:00 ET
    """
    if news_subset.empty:
        return news_subset

    # Wir brauchen die Event-Zeit in US/Eastern
    created_et = news_subset["created_at"].dt.tz_convert("US/Eastern")
    news_subset = news_subset.copy()
    news_subset["created_at_et"] = created_et

    start_allowed = time(9, 30 + window_before // 60)  # grobe Untergrenze
    end_allowed = time(16 - window_after // 60, 0)     # grobe Obergrenze

    mask = news_subset["created_at_et"].dt.time.between(start_allowed, end_allowed)
    filtered = news_subset.loc[mask].copy()
    print(
        f"    [INFO] Intraday event filter: {len(filtered)}/{len(news_subset)} "
        f"events kept (window {window_before}m before, {window_after}m after)"
    )
    return filtered


def compute_price_path(
    news_subset: pd.DataFrame,
    index_df: pd.DataFrame,
    window_before: int = 60,
    window_after: int = 60,
    max_events: int = 300,
) -> np.ndarray:
    """
    Preis-Pfad: relative Änderung in % gegenüber Preis bei t=0.
    """
    target_len = window_before + window_after + 1
    if news_subset.empty:
        return np.zeros(target_len)

    # Nur Events, bei denen das Fenster vollständig im Handel liegt
    news_subset = _filter_events_to_intraday_window(
        news_subset, window_before, window_after
    )
    if news_subset.empty:
        return np.zeros(target_len)

    n_sample = min(max_events, len(news_subset))
    events = news_subset.sample(n=n_sample, random_state=42)

    paths = []

    for _, ev in events.iterrows():
        event_time = ev["created_at"].floor("min")  # auf Minute nach unten
        start_time = event_time - pd.Timedelta(minutes=window_before)
        end_time = event_time + pd.Timedelta(minutes=window_after)

        mask = (index_df["timestamp"] >= start_time) & (
            index_df["timestamp"] <= end_time
        )
        window = index_df.loc[mask, ["timestamp", "close"]].copy()
        if window.empty:
            continue

        # 1-Minuten-Raster
        window = (
            window.set_index("timestamp")
            .resample("1min")
            .last()
        )

        # Vollständiges Zeitraster
        full_index = pd.date_range(start=start_time, end=end_time, freq="1min")
        window = window.reindex(full_index).ffill()

        if window["close"].isna().any():
            continue

        values = window["close"].values
        if len(values) != target_len:
            continue

        base = values[window_before]
        if base <= 0 or np.isnan(base):
            continue

        rel = (values - base) / base * 100.0
        paths.append(rel)

    if not paths:
        print("    [WARN] No valid price windows, returning zeros.")
        return np.zeros(target_len)

    paths_arr = np.vstack(paths)
    print(f"    [INFO] Price: using {paths_arr.shape[0]} valid events")
    return paths_arr.mean(axis=0)


def compute_liquidity_path(
    news_subset: pd.DataFrame,
    index_df: pd.DataFrame,
    value_col: str,          # 'volume' oder 'trade_count'
    window_before: int = 60,
    window_after: int = 60,
    baseline_end_offset: int = 5,  # letzte 5 Minuten vor Event NICHT im Baseline
    max_events: int = 300,
) -> np.ndarray:
    """
    Volume/Trade-Event-Study:
    relative Abweichung vom Durchschnittslevel vor der News (Baseline).

    Baseline = mittlerer Wert von [-window_before, -baseline_end_offset] Minuten.
    """
    target_len = window_before + window_after + 1
    if news_subset.empty:
        return np.zeros(target_len)

    news_subset = _filter_events_to_intraday_window(
        news_subset, window_before, window_after
    )
    if news_subset.empty:
        return np.zeros(target_len)

    n_sample = min(max_events, len(news_subset))
    events = news_subset.sample(n=n_sample, random_state=42)

    paths = []

    for _, ev in events.iterrows():
        event_time = ev["created_at"].floor("min")
        start_time = event_time - pd.Timedelta(minutes=window_before)
        end_time = event_time + pd.Timedelta(minutes=window_after)

        mask = (index_df["timestamp"] >= start_time) & (
            index_df["timestamp"] <= end_time
        )
        window = index_df.loc[mask, ["timestamp", value_col]].copy()
        if window.empty:
            continue

        window = (
            window.set_index("timestamp")
            .resample("1min")
            .sum()     # Summe pro Minute
        )

        full_index = pd.date_range(start=start_time, end=end_time, freq="1min")
        window = window.reindex(full_index).fillna(0.0)

        values = window[value_col].values
        if len(values) != target_len:
            continue

        # Baseline vor der News: [-window_before, -baseline_end_offset]
        baseline_slice = values[: window_before - baseline_end_offset]
        baseline_mean = baseline_slice.mean()

        if baseline_mean <= 0 or np.isnan(baseline_mean):
            continue

        rel = (values / baseline_mean - 1.0) * 100.0
        paths.append(rel)

    if not paths:
        print(f"    [WARN] No valid {value_col} windows, returning zeros.")
        return np.zeros(target_len)

    paths_arr = np.vstack(paths)
    print(
        f"    [INFO] {value_col}: using {paths_arr.shape[0]} valid events, "
        f"baseline = pre-news average"
    )
    return paths_arr.mean(axis=0)


# ---------------------------------------------------------------------
# Plot-Funktionen
# ---------------------------------------------------------------------
def plot_price_event_study(news_df: pd.DataFrame, index_df: pd.DataFrame) -> None:
    print("[INFO] Price Event Study (-60 to +60 min) ...")

    pos = news_df[news_df["sentiment_category"] == "Positive"]
    neg = news_df[news_df["sentiment_category"] == "Negative"]
    neu = news_df[news_df["sentiment_category"] == "Neutral"]

    win_before, win_after = 60, 60

    pos_path = compute_price_path(pos, index_df, win_before, win_after)
    neg_path = compute_price_path(neg, index_df, win_before, win_after)
    neu_path = compute_price_path(neu, index_df, win_before, win_after)

    minutes = np.arange(-win_before, win_after + 1)

    plt.figure(figsize=(12, 6))
    plt.plot(minutes, pos_path, color="green", label="Positive News (> 0.2)", linewidth=2)
    plt.plot(minutes, neg_path, color="red", label="Negative News (< -0.2)", linewidth=2)
    plt.plot(minutes, neu_path, color="gray", linestyle="--", label="Neutral News")

    plt.axvline(0, color="black", linestyle="--", linewidth=1.5, label="News Published")
    plt.axhline(0, color="black", linewidth=0.5)
    plt.title("Price Event Study: -60 to +60 Minutes Around News")
    plt.xlabel("Minutes Relative to News Publication")
    plt.ylabel("Average Price Change (%)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out = os.path.join(IMAGES_DIR, "price_event_study.png")
    plt.savefig(out)
    plt.close()
    print(f"  [OK] Saved: {out}")


def plot_volume_event_study(news_df: pd.DataFrame, index_df: pd.DataFrame) -> None:
    print("[INFO] Volume Event Study (-60 to +60 min) ...")

    pos = news_df[news_df["sentiment_category"] == "Positive"]
    neg = news_df[news_df["sentiment_category"] == "Negative"]
    neu = news_df[news_df["sentiment_category"] == "Neutral"]

    win_before, win_after = 60, 60

    pos_path = compute_liquidity_path(pos, index_df, "volume", win_before, win_after)
    neg_path = compute_liquidity_path(neg, index_df, "volume", win_before, win_after)
    neu_path = compute_liquidity_path(neu, index_df, "volume", win_before, win_after)

    minutes = np.arange(-win_before, win_after + 1)

    plt.figure(figsize=(12, 6))
    plt.plot(minutes, pos_path, color="green", label="Positive News (> 0.2)", linewidth=2)
    plt.plot(minutes, neg_path, color="red", label="Negative News (< -0.2)", linewidth=2)
    plt.plot(minutes, neu_path, color="gray", linestyle="--", label="Neutral News")

    plt.axvline(0, color="black", linestyle="--", linewidth=1.5, label="News Published")
    plt.axhline(0, color="black", linewidth=0.5)
    plt.title("Volume Event Study: -60 to +60 Minutes Around News")
    plt.xlabel("Minutes Relative to News Publication")
    plt.ylabel("Average Volume Change vs Pre-News Baseline (%)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out = os.path.join(IMAGES_DIR, "volume_event_study.png")
    plt.savefig(out)
    plt.close()
    print(f"  [OK] Saved: {out}")


def plot_trade_count_event_study(news_df: pd.DataFrame, index_df: pd.DataFrame) -> None:
    print("[INFO] Trade Count Event Study (-60 to +60 min) ...")

    pos = news_df[news_df["sentiment_category"] == "Positive"]
    neg = news_df[news_df["sentiment_category"] == "Negative"]
    neu = news_df[news_df["sentiment_category"] == "Neutral"]

    win_before, win_after = 60, 60

    pos_path = compute_liquidity_path(
        pos, index_df, "trade_count", win_before, win_after
    )
    neg_path = compute_liquidity_path(
        neg, index_df, "trade_count", win_before, win_after
    )
    neu_path = compute_liquidity_path(
        neu, index_df, "trade_count", win_before, win_after
    )

    minutes = np.arange(-win_before, win_after + 1)

    plt.figure(figsize=(12, 6))
    plt.plot(minutes, pos_path, color="green", label="Positive News (> 0.2)", linewidth=2)
    plt.plot(minutes, neg_path, color="red", label="Negative News (< -0.2)", linewidth=2)
    plt.plot(minutes, neu_path, color="gray", linestyle="--", label="Neutral News")

    plt.axvline(0, color="black", linestyle="--", linewidth=1.5, label="News Published")
    plt.axhline(0, color="black", linewidth=0.5)
    plt.title("Trade Count Event Study: -60 to +60 Minutes Around News")
    plt.xlabel("Minutes Relative to News Publication")
    plt.ylabel("Average Trade Count Change vs Pre-News Baseline (%)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out = os.path.join(IMAGES_DIR, "trade_count_event_study.png")
    plt.savefig(out)
    plt.close()
    print(f"  [OK] Saved: {out}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    print("\n" + "=" * 60)
    print("CLEAN EVENT STUDY ANALYSIS (raw news + index, RTH only)")
    print("=" * 60 + "\n")

    news_df = load_news_data()
    index_df = load_index_data()

    plot_sentiment_distribution(news_df)
    plot_price_event_study(news_df, index_df)
    plot_volume_event_study(news_df, index_df)
    plot_trade_count_event_study(news_df, index_df)

    print("\n" + "=" * 60)
    print("Analysis complete. Check the 'images' directory.")
    print("Generated:")
    print("  - sentiment_distribution.png")
    print("  - price_event_study.png")
    print("  - volume_event_study.png")
    print("  - trade_count_event_study.png")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
