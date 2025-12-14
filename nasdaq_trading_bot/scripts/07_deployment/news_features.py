"""
Real-time News Features Module for LSTM Deployment
===================================================
Fetches latest news from Alpha Vantage and calculates the 3 news features:
- last_news_sentiment
- news_age_minutes
- effective_sentiment_t

This module is designed for deployment/live trading AND replay backtests.
"""

import os
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from typing import Optional, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from dotenv import load_dotenv

# Load environment
load_dotenv()

class NewsFeatureProvider:
    """
    Provides real-time news features for deployment.
    Fetches from Alpha Vantage and calculates sentiment using cached FinBERT model.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        decay_lambda: float = 0.001,
        cache_minutes: int = 5,
    ):
        self.api_key = api_key or os.getenv("ALPHA_VANTAGE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "ALPHA_VANTAGE_API_KEY not found. "
                "Set it in .env file or pass it to constructor."
            )

        self.decay_lambda = decay_lambda
        self.cache_minutes = cache_minutes

        # Cache
        self.cached_news: Optional[pd.DataFrame] = None
        self.cache_timestamp: Optional[datetime] = None

        # FinBERT model (loaded on first use)
        self.tokenizer = None
        self.model = None
        self.device = None

        print("[NEWS] NewsFeatureProvider initialized")

    def _load_sentiment_model(self):
        if self.model is not None:
            return

        print("[NEWS] Loading FinBERT model for sentiment analysis...")
        MODEL_NAME = "ProsusAI/finbert"

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

        print(f"[NEWS] FinBERT loaded on {self.device}")

    def _calculate_sentiment(self, text: str) -> float:
        if not text or pd.isna(text):
            return 0.0

        self._load_sentiment_model()

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

        # FinBERT outputs: [negative, neutral, positive]
        probs = predictions.cpu().numpy()[0]
        sentiment_score = probs[2] - probs[0]
        return float(sentiment_score)

    def _fetch_latest_news(self, tickers: list = None, topics: list = None) -> pd.DataFrame:
        if tickers is None and topics is None:
            topics = ["technology", "financial_markets"]

        print(f"[NEWS] Fetching latest news (topics: {topics}, tickers: {tickers})...")

        params = {
            "function": "NEWS_SENTIMENT",
            "apikey": self.api_key,
            "limit": 50,
            "sort": "LATEST",
        }

        if tickers and not topics:
            params["tickers"] = ",".join(tickers)

        if topics:
            params["topics"] = ",".join(topics)

        try:
            response = requests.get("https://www.alphavantage.co/query", params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            print(f"[NEWS DEBUG] API response keys: {list(data.keys())}")

            if "Error Message" in data:
                print(f"[NEWS ERROR] {data['Error Message']}")
                return pd.DataFrame()

            if "Note" in data:
                print(f"[NEWS] API rate limit note: {data['Note']}")
                return pd.DataFrame()

            if "Information" in data:
                print(f"[NEWS] API Information: {data['Information']}")
                return pd.DataFrame()

            feed = data.get("feed", [])
            if not feed:
                print(f"[NEWS] No news items found. Full response: {data}")
                return pd.DataFrame()

            news_list = []
            for item in feed:
                news_list.append(
                    {
                        "timestamp": item.get("time_published", ""),
                        "headline": item.get("title", ""),
                        "summary": item.get("summary", ""),
                    }
                )

            df = pd.DataFrame(news_list)

            df["timestamp"] = pd.to_datetime(
                df["timestamp"],
                format="%Y%m%dT%H%M%S",
                errors="coerce",
            )

            df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")

            print(f"[NEWS] Analyzing sentiment for {len(df)} articles...")
            df["sentiment_score"] = df.apply(
                lambda row: self._calculate_sentiment(f"{row['headline']}. {row['summary']}"),
                axis=1,
            )

            print(f"[NEWS] Fetched {len(df)} articles with sentiment")
            print(
                f"[NEWS] Sentiment stats: mean={df['sentiment_score'].mean():.3f}, "
                f"std={df['sentiment_score'].std():.3f}"
            )

            df = df.sort_values("timestamp", ascending=False).reset_index(drop=True)
            return df

        except Exception as e:
            print(f"[NEWS ERROR] Failed to fetch news: {e}")
            return pd.DataFrame()

    def _get_cached_or_fetch(self, tickers: list = None, topics: list = None) -> pd.DataFrame:
        now = datetime.now(timezone.utc)

        if self.cached_news is not None and self.cache_timestamp is not None:
            age_minutes = (now - self.cache_timestamp).total_seconds() / 60.0
            if age_minutes < self.cache_minutes:
                print(f"[NEWS] Using cached news (age: {age_minutes:.1f} min)")
                return self.cached_news

        df = self._fetch_latest_news(tickers, topics)

        if not df.empty:
            self.cached_news = df
            self.cache_timestamp = now

        return df

    # >>> ADDED FOR REPLAY <<<
    def fetch_news_df_once(self, tickers: list = None, topics: list = None) -> pd.DataFrame:
        """
        Fetch news ONCE for replay usage (no caching by wall-clock time).
        """
        return self._fetch_latest_news(tickers=tickers, topics=topics)

    def get_news_features(
        self,
        current_time: datetime,
        tickers: list = None,
        topics: list = None,
    ) -> Tuple[float, float, float]:
        if current_time.tzinfo is None:
            current_time = current_time.replace(tzinfo=timezone.utc)
        else:
            current_time = current_time.astimezone(timezone.utc)

        df_news = self._get_cached_or_fetch(tickers, topics)

        if df_news.empty:
            print("[NEWS] No news available, returning neutral features")
            return (0.0, 0.0, 0.0)  # sentiment, age, effective (use 0.0 instead of NaN)

        past_news = df_news[df_news["timestamp"] <= current_time]
        if past_news.empty:
            print("[NEWS] No news before current time, returning neutral features")
            return (0.0, 0.0, 0.0)  # sentiment, age, effective (use 0.0 instead of NaN)

        most_recent = past_news.iloc[0]

        sentiment = float(most_recent["sentiment_score"])
        news_age = (current_time - most_recent["timestamp"]).total_seconds() / 60.0
        effective_sentiment = sentiment * np.exp(-self.decay_lambda * news_age)

        print(f"[NEWS] Most recent: '{most_recent['headline'][:60]}...'")
        print(f"[NEWS] Published: {most_recent['timestamp']}, Age: {news_age:.1f} miSSSn")
        print(f"[NEWS] Sentiment: {sentiment:.4f}, Effective: {effective_sentiment:.4f}")

        return (sentiment, news_age, float(effective_sentiment))

    def get_news_features_dict(
        self,
        current_time: datetime,
        tickers: list = None,
        topics: list = None,
    ) -> dict:
        sentiment, age, effective = self.get_news_features(current_time, tickers, topics)
        return {
            "last_news_sentiment": float(sentiment),
            "news_age_minutes": float(age) if age == age else 0.0,  # Use 0.0 instead of np.nan
            "effective_sentiment_t": float(effective),
        }

def get_realtime_news_features(
    current_time: datetime,
    api_key: Optional[str] = None,
    decay_lambda: float = 0.001,
) -> Tuple[float, float, float]:
    provider = NewsFeatureProvider(api_key=api_key, decay_lambda=decay_lambda)
    return provider.get_news_features(current_time)


if __name__ == "__main__":
    print("=" * 60)
    print("Testing News Feature Provider")
    print("=" * 60)

    provider = NewsFeatureProvider()
    now = datetime.now(timezone.utc)
    features = provider.get_news_features_dict(now)

    print("\nNews Features:")
    for key, value in features.items():
        print(f"  {key}: {value}")

    print("\n" + "=" * 60)
    print("Test complete!")
    print("=" * 60)
