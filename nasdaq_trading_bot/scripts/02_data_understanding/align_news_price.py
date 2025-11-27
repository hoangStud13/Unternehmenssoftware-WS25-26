import os
import pandas as pd
import numpy as np
from datetime import datetime
import pytz

# Configuration
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
data_dir = os.path.join(project_root, 'data')

news_file = os.path.join(data_dir, 'nasdaq_news_5y_with_sentiment.csv')
price_file = os.path.join(data_dir, 'nasdaq100_index_1m.csv')
output_file = os.path.join(data_dir, 'nasdaq_aligned_with_news.csv')

# Decay parameter (λ)
# Higher λ = faster decay
# For example: λ = 0.001 means half-life of ~693 minutes (~11.5 hours)
#              λ = 0.01 means half-life of ~69 minutes
LAMBDA = 0.001  # Adjust based on your strategy

def align_news_with_price():
    """
    Align news sentiment with price data and apply exponential decay.
    
    For each price bar (1-minute), we:
    1. Find the most recent news before that time
    2. Calculate news_age_minutes
    3. Apply decay: effective_sentiment = sentiment * exp(-λ * news_age_minutes)
    """
    print("Loading data...")
    
    # Load news data
    news_df = pd.read_csv(news_file)
    news_df['created_at'] = pd.to_datetime(news_df['created_at'], utc=True)
    news_df = news_df.sort_values('created_at').reset_index(drop=True)
    
    print(f"Loaded {len(news_df)} news items")
    print(f"Date range: {news_df['created_at'].min()} to {news_df['created_at'].max()}")
    
    # Load price data
    price_df = pd.read_csv(price_file)
    price_df['timestamp'] = pd.to_datetime(price_df['timestamp'], utc=True)
    price_df = price_df.sort_values('timestamp').reset_index(drop=True)
    
    print(f"Loaded {len(price_df)} price bars")
    print(f"Date range: {price_df['timestamp'].min()} to {price_df['timestamp'].max()}")
    
    # Initialize columns for aligned data
    price_df['last_news_sentiment'] = 0.0
    price_df['news_age_minutes'] = np.nan
    price_df['effective_sentiment_t'] = 0.0
    price_df['news_id'] = None
    price_df['news_headline'] = None
    
    print("\nAligning news with price bars...")
    
    # For each price bar, find the most recent news
    news_idx = 0
    
    for i, row in price_df.iterrows():
        current_time = row['timestamp']
        
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
            price_df.at[i, 'last_news_sentiment'] = sentiment
            price_df.at[i, 'news_age_minutes'] = news_age
            price_df.at[i, 'effective_sentiment_t'] = effective_sentiment
            price_df.at[i, 'news_id'] = news_row['id']
            price_df.at[i, 'news_headline'] = news_row['headline']
        
        # Progress indicator
        if (i + 1) % 100000 == 0:
            print(f"Processed {i + 1:,} / {len(price_df):,} bars ({(i+1)/len(price_df)*100:.1f}%)")
    
    print("\nAlignment complete!")
    
    # Save results
    price_df.to_csv(output_file, index=False)
    print(f"Saved aligned data to {output_file}")
    
    # Statistics
    print("\n" + "="*60)
    print("ALIGNMENT STATISTICS")
    print("="*60)
    
    has_news = price_df['news_age_minutes'].notna()
    print(f"\nPrice bars with news: {has_news.sum():,} ({has_news.sum()/len(price_df)*100:.1f}%)")
    print(f"Price bars without news: {(~has_news).sum():,} ({(~has_news).sum()/len(price_df)*100:.1f}%)")
    
    if has_news.any():
        print(f"\nNews Age Statistics (minutes):")
        print(f"  Mean: {price_df.loc[has_news, 'news_age_minutes'].mean():.1f}")
        print(f"  Median: {price_df.loc[has_news, 'news_age_minutes'].median():.1f}")
        print(f"  Max: {price_df.loc[has_news, 'news_age_minutes'].max():.1f}")
        
        print(f"\nEffective Sentiment Statistics:")
        print(f"  Mean: {price_df.loc[has_news, 'effective_sentiment_t'].mean():.4f}")
        print(f"  Std: {price_df.loc[has_news, 'effective_sentiment_t'].std():.4f}")
        print(f"  Min: {price_df.loc[has_news, 'effective_sentiment_t'].min():.4f}")
        print(f"  Max: {price_df.loc[has_news, 'effective_sentiment_t'].max():.4f}")
        
        # Decay analysis
        print(f"\nDecay Analysis (λ = {LAMBDA}):")
        print(f"  Half-life: {np.log(2)/LAMBDA:.1f} minutes ({np.log(2)/LAMBDA/60:.1f} hours)")
        
        # Show how many bars have significant sentiment (>0.1 effective)
        significant = (price_df['effective_sentiment_t'].abs() > 0.1).sum()
        print(f"  Bars with |effective_sentiment| > 0.1: {significant:,} ({significant/len(price_df)*100:.1f}%)")

if __name__ == "__main__":
    align_news_with_price()
