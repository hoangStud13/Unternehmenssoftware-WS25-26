"""
Step 5: Post-Split Preparation
----------------------------------------
- Build X/y
- Train-only StandardScaler (X + y)
- Save scaled/unscaled splits + scaler
"""

import os
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
data_dir = os.path.join(project_root, 'data')
image_dir = os.path.join(project_root, 'images')

train_file      = os.path.join(data_dir, 'nasdaq_train.csv')
test_file       = os.path.join(data_dir, 'nasdaq_test.csv')
validation_file = os.path.join(data_dir, 'nasdaq_validation.csv')

train_df      = pd.read_csv(train_file,      index_col=0, parse_dates=True)
validation_df = pd.read_csv(validation_file, index_col=0, parse_dates=True)
test_df       = pd.read_csv(test_file,       index_col=0, parse_dates=True)

target = [
    "target_return_1m", "target_return_3m",
    "target_return_5m", "target_return_10m", "target_return_15m"
]
exclude_cols = target + ["news_id", "timestamp"]
feature_cols = [c for c in train_df.columns if c not in exclude_cols]

# ------------------ X / y trennen ------------------ #
X_train = train_df[feature_cols]
X_val   = validation_df[feature_cols]
X_test  = test_df[feature_cols]

y_train = train_df[target].select_dtypes(include='number')
y_val   = validation_df[target].select_dtypes(include='number')
y_test  = test_df[target].select_dtypes(include='number')

# ------------------ Scaler für X ------------------ #
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_val_scaled   = scaler_X.transform(X_val)
X_test_scaled  = scaler_X.transform(X_test)

# ------------------ Scaler für y (Targets) ------------------ #
scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train.values)
y_val_scaled   = scaler_y.transform(y_val.values)
y_test_scaled  = scaler_y.transform(y_test.values)

# ------------------ CSVs speichern ------------------ #
# unskaliert
X_train.to_csv(os.path.join(data_dir, "X_train.csv"))
y_train.to_csv(os.path.join(data_dir, "y_train.csv"))
X_val.to_csv(os.path.join(data_dir, "X_val.csv"))
y_val.to_csv(os.path.join(data_dir, "y_val.csv"))
X_test.to_csv(os.path.join(data_dir, "X_test.csv"))
y_test.to_csv(os.path.join(data_dir, "y_test.csv"))

# skaliert (X)
pd.DataFrame(X_train_scaled, index=X_train.index, columns=feature_cols) \
  .to_csv(os.path.join(data_dir, "X_train_scaled.csv"))
pd.DataFrame(X_val_scaled, index=X_val.index, columns=feature_cols) \
  .to_csv(os.path.join(data_dir, "X_val_scaled.csv"))
pd.DataFrame(X_test_scaled, index=X_test.index, columns=feature_cols) \
  .to_csv(os.path.join(data_dir, "X_test_scaled.csv"))

# skaliert (y)
pd.DataFrame(y_train_scaled, index=y_train.index, columns=target) \
  .to_csv(os.path.join(data_dir, "y_train_scaled.csv"))
pd.DataFrame(y_val_scaled, index=y_val.index, columns=target) \
  .to_csv(os.path.join(data_dir, "y_val_scaled.csv"))
pd.DataFrame(y_test_scaled, index=y_test.index, columns=target) \
  .to_csv(os.path.join(data_dir, "y_test_scaled.csv"))

# ------------------ Scaler speichern ------------------ #
joblib.dump(scaler_X, os.path.join(data_dir, "scaler_X.joblib"))
joblib.dump(scaler_y, os.path.join(data_dir, "scaler_y.joblib"))

print("Step 5 abgeschlossen: X/y getrennt, skaliert und gespeichert.")
