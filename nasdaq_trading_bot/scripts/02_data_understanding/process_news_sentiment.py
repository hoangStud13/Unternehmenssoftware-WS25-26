import os

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Configuration
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
data_dir = os.path.join(project_root, 'data')

input_file = os.path.join(data_dir, 'nasdaq_news_5y.csv')
output_file = os.path.join(data_dir, 'nasdaq_news_5y_with_sentiment.csv')

# FinBERT model
MODEL_NAME = "ProsusAI/finbert"

def load_finbert():
    """Load FinBERT model and tokenizer."""
    print("Loading FinBERT model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    
    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    print(f"Model loaded on {device}")
    return tokenizer, model, device

def get_sentiment(text, tokenizer, model, device):
    """
    Calculate sentiment score for a given text.
    Returns a score between -1 (negative) and +1 (positive).
    """
    if pd.isna(text) or text == "":
        return 0.0  # Neutral for missing text
    
    # Tokenize
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Get predictions
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    
    # FinBERT outputs: [negative, neutral, positive]
    probs = predictions.cpu().numpy()[0]
    
    # Calculate weighted sentiment score: -1 (negative) to +1 (positive)
    sentiment_score = probs[2] - probs[0]  # positive - negative
    
    return float(sentiment_score)

def process_news_sentiment():
    """Process all news items and add sentiment scores."""
    print(f"Reading news from {input_file}...")
    df = pd.read_csv(input_file)
    
    print(f"Found {len(df)} news items")
    
    # Load model
    tokenizer, model, device = load_finbert()
    
    # Combine headline and summary for better context
    df['text_for_sentiment'] = df.apply(
        lambda row: f"{row['headline']}. {row['summary']}" if pd.notna(row['summary']) else row['headline'],
        axis=1
    )
    
    # Calculate sentiment for each item
    print("Calculating sentiment scores...")
    sentiments = []
    
    for text in tqdm(df['text_for_sentiment'], desc="Processing"):
        sentiment = get_sentiment(text, tokenizer, model, device)
        sentiments.append(sentiment)
    
    df['sentiment_score'] = sentiments
    
    # Drop the temporary column
    df.drop(columns=['text_for_sentiment'], inplace=True)
    
    # Save results
    df.to_csv(output_file, index=False)
    print(f"\nSentiment analysis complete!")
    print(f"Results saved to {output_file}")
    print(f"\nSentiment Statistics:")
    print(f"  Mean: {df['sentiment_score'].mean():.3f}")
    print(f"  Std:  {df['sentiment_score'].std():.3f}")
    print(f"  Min:  {df['sentiment_score'].min():.3f}")
    print(f"  Max:  {df['sentiment_score'].max():.3f}")
    
    # Distribution
    positive = (df['sentiment_score'] > 0.1).sum()
    neutral = ((df['sentiment_score'] >= -0.1) & (df['sentiment_score'] <= 0.1)).sum()
    negative = (df['sentiment_score'] < -0.1).sum()
    
    print(f"\nSentiment Distribution:")
    print(f"  Positive: {positive} ({positive/len(df)*100:.1f}%)")
    print(f"  Neutral:  {neutral} ({neutral/len(df)*100:.1f}%)")
    print(f"  Negative: {negative} ({negative/len(df)*100:.1f}%)")

if __name__ == "__main__":
    process_news_sentiment()
