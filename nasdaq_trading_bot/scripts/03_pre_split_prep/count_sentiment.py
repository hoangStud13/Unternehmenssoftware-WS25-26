import pandas as pd
import numpy as np
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

        # Count "around 0" for news_age_minutes
        # Using a small epsilon
        epsilon = 1e-6
        zero_mask = df[target_col].abs() < epsilon
        count_zero = zero_mask.sum()
        total = len(df)
        
        print(f"\nAnalysis for '{target_col}':")
        print(f"Total rows: {total}")
        print(f"Count where abs(value) < {epsilon}: {count_zero}")
        print(f"Percentage: {count_zero / total * 100:.2f}%")
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    count_sentiment_zeros()
