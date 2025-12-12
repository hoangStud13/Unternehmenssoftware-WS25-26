import pandas as pd
import os

def count_sentiment_zeros():
    # Robust path handling
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    file_path = os.path.join(base_dir, 'data', 'nasdaq100_index_1m_features.csv')
    
    try:
        print(f"Loading {file_path}...")
        df = pd.read_csv(file_path)
        
        # Check for relevant columns
        cols = df.columns.tolist()
        print(f"Columns found: {len(cols)}")
        
        target_col = 'news_age_minutes'
        if target_col not in cols:
            print(f"Warning: '{target_col}' not found. Available columns: {cols}")
            return

        # Count news with age < 1 minute (news that first appear fresh)
        threshold = 1.0  # 1 minute
        fresh_mask = df[target_col] < threshold
        
        # Count UNIQUE news_id where age < 1 minute
        unique_news_fresh = df.loc[fresh_mask, 'news_id'].nunique()
        total_unique_news = df['news_id'].nunique()
        
        print(f"\nAnalysis for '{target_col}':")
        print(f"Total rows: {len(df)}")
        print(f"Total unique news used: {total_unique_news}")
        print(f"Rows where news_age < {threshold} min: {fresh_mask.sum()}")
        print(f"UNIQUE NEWS where news_age < {threshold} min: {unique_news_fresh}")
        print(f"Percentage of unique news appearing fresh: {unique_news_fresh / total_unique_news * 100:.2f}%")
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    count_sentiment_zeros()
