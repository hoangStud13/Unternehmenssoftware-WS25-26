"""
Step 5: Post-Split Preparation
----------------------------------------
- Build X/y
- Train-only StandardScaler
- Save scaled/unscaled splits + scaler
- Save example scaling plot (TEMP)
"""

import os
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
data_dir = os.path.join(project_root, 'data')
image_dir = os.path.join(project_root, 'images')

train_file = os.path.join(data_dir, 'nasdaq_train.csv')
test_file =  os.path.join(data_dir, 'nasdaq_test.csv')
validation_file =  os.path.join(data_dir, 'nasdaq_validation.csv')


train_df = pd.read_csv(train_file, index_col=0, parse_dates=True)
validation_df = pd.read_csv(validation_file, index_col=0, parse_dates=True)
test_df = pd.read_csv(test_file, index_col=0, parse_dates=True)

target = ["target_return_1m", "target_return_3m", "target_return_4m", "target_return_5m","target_return_10m","target_return_15m"]
exclude_cols = ["target_return_1m", "target_return_3m", "target_return_4m", "target_return_5m","target_return_10m","target_return_15m","news_id","timestamp"]
feature_cols = [c for c in train_df.columns if c not in exclude_cols]

# Separate features (X) and target (y) for each split
X_train, y_train = train_df[feature_cols], train_df[target]
X_val, y_val = validation_df[feature_cols], validation_df[target]
X_test, y_test = test_df[feature_cols], test_df[target]

# Initialize a StandardScaler and fit only on the training data to avoid data leakage
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Keep only numeric columns in y (Targets must be numeric)
y_train = y_train.select_dtypes(include='number')
y_val   = y_val.select_dtypes(include='number')
y_test  = y_test.select_dtypes(include='number')


# Save unscaled feature and target splits to CSV files
X_train.to_csv(os.path.join(data_dir, "X_train.csv"))
y_train.to_csv(os.path.join(data_dir, "y_train.csv"))
X_val.to_csv(os.path.join(data_dir, "X_val.csv"))
y_val.to_csv(os.path.join(data_dir, "y_val.csv"))
X_test.to_csv(os.path.join(data_dir, "X_test.csv"))
y_test.to_csv(os.path.join(data_dir, "y_test.csv"))

# Save scaled feature splits to CSV files, preserving the index and column names
pd.DataFrame(X_train_scaled, index=X_train.index, columns=feature_cols).to_csv(os.path.join(data_dir, "X_train_scaled.csv"))
pd.DataFrame(X_val_scaled, index=X_val.index, columns=feature_cols).to_csv(os.path.join(data_dir, "X_val_scaled.csv"))
pd.DataFrame(X_test_scaled, index=X_test.index, columns=feature_cols).to_csv(os.path.join(data_dir, "X_test_scaled.csv"))
# Save the fitted scaler object for later use (e.g., during inference)
joblib.dump(scaler, os.path.join(data_dir, "scaler.joblib"))

