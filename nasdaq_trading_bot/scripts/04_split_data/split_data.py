# split_data.py
import os
import pandas as pd

def split_data(features_file, data_dir, train_ratio=0.7, validation_ratio=0.15):
   
    os.makedirs(data_dir, exist_ok=True)
    
    # Load features
    df = pd.read_parquet(features_file)
    df_length = len(df)
    
    train_end = int(train_ratio * df_length)
    validation_end = int((train_ratio + validation_ratio) * df_length)
    
    # Split
    train = df.iloc[:train_end]
    validation = df.iloc[train_end:validation_end]
    test = df.iloc[validation_end:]
    
    # Save
    train.to_parquet(os.path.join(data_dir, "nasdaq_train.parquet"), index=True)
    train.to_csv(os.path.join(data_dir, "nasdaq_train.csv"), index=True)
    
    validation.to_parquet(os.path.join(data_dir, "nasdaq_validation.parquet"), index=True)
    validation.to_csv(os.path.join(data_dir, "nasdaq_validation.csv"), index=True)
    
    test.to_parquet(os.path.join(data_dir, "nasdaq_test.parquet"), index=True)
    test.to_csv(os.path.join(data_dir, "nasdaq_test.csv"), index=True)
    
    print(f"Train: {len(train)}, Validation: {len(validation)}, Test: {len(test)}")
    print(f"Splits saved in {data_dir}")





def main():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    data_dir = os.path.join(project_root, 'data')
    features_file = os.path.join(data_dir, 'nasdaq100_index_1m_features.parquet')

    split_data(features_file, data_dir)

if __name__ == "__main__":
    main()