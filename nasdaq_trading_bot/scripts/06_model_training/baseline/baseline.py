"""
Baseline: Linear Regression
===========================
Simple linear regression baseline for model comparison.
LINEAR REGRESSION SHOULD NOT USE SEQUENCES - just direct regression on features.
"""

import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
data_dir = os.path.join(project_root, 'data')
images_dir = os.path.join(project_root, "images")
os.makedirs(images_dir, exist_ok=True)


# Load data - NO SEQUENCES for Linear Regression!
print("Loading data...")
X_train_scaled = pd.read_csv(os.path.join(data_dir, 'X_train_scaled.csv'), index_col=0).values
y_train_scaled = pd.read_csv(os.path.join(data_dir, 'y_train_scaled.csv'), index_col=0).values
X_val_scaled = pd.read_csv(os.path.join(data_dir, 'X_val_scaled.csv'), index_col=0).values
y_val_scaled = pd.read_csv(os.path.join(data_dir, 'y_val_scaled.csv'), index_col=0).values
X_test_scaled = pd.read_csv(os.path.join(data_dir, 'X_test_scaled.csv'), index_col=0).values
y_test_scaled = pd.read_csv(os.path.join(data_dir, 'y_test_scaled.csv'), index_col=0).values
scaler_y = joblib.load(os.path.join(data_dir, "scaler_y.joblib"))

# Handle NaNs
X_train_scaled = np.nan_to_num(X_train_scaled)
y_train_scaled = np.nan_to_num(y_train_scaled)
X_val_scaled = np.nan_to_num(X_val_scaled)
y_val_scaled = np.nan_to_num(y_val_scaled)
X_test_scaled = np.nan_to_num(X_test_scaled)
y_test_scaled = np.nan_to_num(y_test_scaled)

print(f"Train: X={X_train_scaled.shape}, y={y_train_scaled.shape}")
print(f"Val:   X={X_val_scaled.shape}, y={y_val_scaled.shape}")
print(f"Test:  X={X_test_scaled.shape}, y={y_test_scaled.shape}")

# NO sequence creation - use data directly like Feed Forward does!
# This is the correct approach for linear regression baseline

# Train Linear Regression
print("\nTraining Linear Regression (without sequences)...")
model = LinearRegression()
model.fit(X_train_scaled, y_train_scaled)

# Predict
y_pred_train = model.predict(X_train_scaled)
y_pred_val = model.predict(X_val_scaled)
y_pred_test = model.predict(X_test_scaled)

# Calculate metrics
train_mse = mean_squared_error(y_train_scaled, y_pred_train)
val_mse = mean_squared_error(y_val_scaled, y_pred_val)
test_mse = mean_squared_error(y_test_scaled, y_pred_test)
train_mae = mean_absolute_error(y_train_scaled, y_pred_train)
val_mae = mean_absolute_error(y_val_scaled, y_pred_val)
test_mae = mean_absolute_error(y_test_scaled, y_pred_test)

print("\n" + "="*50)
print("BASELINE: LINEAR REGRESSION RESULTS")
print("="*50)
print(f"Train MSE: {train_mse:.6f}")
print(f"Val MSE:   {val_mse:.6f}")
print(f"Test MSE:  {test_mse:.6f}")
print(f"Train MAE: {train_mae:.6f}")
print(f"Val MAE:   {val_mae:.6f}")
print(f"Test MAE:  {test_mae:.6f}")
print("="*50)

# Comparison with other models (using Val MSE for fair comparison)
print("\n>>> VALIDATION MSE COMPARISON:")
print(f"    Linear Regression Val MSE: {val_mse:.4f}")
print(f"    RNN Val MSE:               0.50")
print(f"    LSTM Val MSE:              0.33")
print(f"    Feed Forward Val MSE:      0.20")
print("-"*50)

# Inverse transform for plotting
y_test_inv = scaler_y.inverse_transform(y_test_scaled)
y_pred_inv = scaler_y.inverse_transform(y_pred_test)

# Plot
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle(f'Linear Regression Baseline (Val MSE: {val_mse:.4f} | Test MSE: {test_mse:.4f})', fontsize=14, fontweight='bold')

target_names = ['1min', '3min', '5min', '10min', '15min']
n_samples = 100

for i, name in enumerate(target_names):
    ax = axes[i // 3, i % 3]
    ax.plot(y_test_inv[:n_samples, i], label='Actual', linewidth=2, color='black')
    ax.plot(y_pred_inv[:n_samples, i], label='Predicted', linewidth=1.5, alpha=0.7, color='blue')
    ax.set_title(f'Target: {name}')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Return (%)')
    ax.legend()
    ax.grid(True, alpha=0.3)

# Bar chart comparison in last subplot
ax = axes[1, 2]
models = ['Linear\nBaseline', 'RNN', 'LSTM', 'Feed Forward']
mses = [val_mse, 0.50, 0.33, 0.20]  # Using Val MSE for comparison
colors = ['orange', 'blue', 'green', 'purple']
bars = ax.bar(models, mses, color=colors, alpha=0.8, edgecolor='black')
ax.set_ylabel('Validation MSE')
ax.set_title('Model Comparison (Val MSE)')
ax.grid(axis='y', alpha=0.3)
for bar, val in zip(bars, mses):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
            f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plot_path = os.path.join(images_dir, "06_model_comparison_final.png")
fig.savefig(plot_path, dpi=200)
print(f"\nPlot saved: {plot_path}")
