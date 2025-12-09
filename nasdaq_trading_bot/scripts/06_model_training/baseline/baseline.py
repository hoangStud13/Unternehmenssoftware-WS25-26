import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import joblib
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

def create_sequences(X, y, sequence_length=10):
    """
    Erstellt Sequenzen (identisch zu rnn.py für fairen Vergleich)
    X: Features (samples, features)
    y: Targets (samples, outputs)
    sequence_length: Anzahl der Zeitschritte
    """
    X_seq, y_seq = [], []
    for i in range(len(X) - sequence_length):
        X_seq.append(X[i:i + sequence_length])
        y_seq.append(y[i + sequence_length])
    return np.array(X_seq), np.array(y_seq)

def flatten_sequences(X_seq):
    """
    Flacht Sequenzen ab für Lineare Regression
    (samples, time_steps, features) -> (samples, time_steps * features)
    """
    nsamples, nsteps, nfeatures = X_seq.shape
    return X_seq.reshape(nsamples, nsteps * nfeatures)

# Pfade definieren
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
data_dir = os.path.join(project_root, 'data')
images_dir = os.path.join(project_root, "images")

os.makedirs(images_dir, exist_ok=True)

# Datendateien (wir verwenden die Scaled Test Data)
required_files = [
    'X_train_scaled.csv', 'y_train_scaled.csv',
    'X_test_scaled.csv',  'y_test_scaled.csv',
    'scaler_y.joblib'
]

files_exist = all(os.path.exists(os.path.join(data_dir, f)) for f in required_files)
if not files_exist:
    raise FileNotFoundError("Benötigte Dateien aus Step 5 fehlen. Bitte Step 5 zuerst ausführen.")

print("Lade Daten...")

X_train_scaled = pd.read_csv(os.path.join(data_dir, 'X_train_scaled.csv'), index_col=0).values
y_train_scaled = pd.read_csv(os.path.join(data_dir, 'y_train_scaled.csv'), index_col=0).values
X_test_scaled = pd.read_csv(os.path.join(data_dir, 'X_test_scaled.csv'), index_col=0).values
y_test_scaled = pd.read_csv(os.path.join(data_dir, 'y_test_scaled.csv'), index_col=0).values

# y-Scaler laden
scaler_y = joblib.load(os.path.join(data_dir, "scaler_y.joblib"))

# NaNs prüfen und behandeln
if np.isnan(X_train_scaled).any() or np.isnan(y_train_scaled).any():
    print("Warnung: NaNs in Trainingsdaten gefunden. Werden mit 0 gefüllt.")
    X_train_scaled = np.nan_to_num(X_train_scaled)
    y_train_scaled = np.nan_to_num(y_train_scaled)

if np.isnan(X_test_scaled).any() or np.isnan(y_test_scaled).any():
    print("Warnung: NaNs in Testdaten gefunden. Werden mit 0 gefüllt.")
    X_test_scaled = np.nan_to_num(X_test_scaled)
    y_test_scaled = np.nan_to_num(y_test_scaled)

# Sequenzen erstellen (um den gleichen Input wie RNN zu simulieren)
sequence_length = 10
X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_scaled, sequence_length)
X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test_scaled, sequence_length)

# Flatten Inputs für Scikit-Learn (nur Baseline Linear Regressor braucht das)
X_train_flat = flatten_sequences(X_train_seq)
X_test_flat = flatten_sequences(X_test_seq)

print(f"Train Shape Flat: {X_train_flat.shape}")
print(f"Test Shape Flat:  {X_test_flat.shape}")

# ---------------------------------------------------------
# 1. Baseline: Dummy Regressor (Mean)
# ---------------------------------------------------------
print("\n--- Baseline 1: Dummy Regressor (Mean) ---")
dummy_regr = DummyRegressor(strategy="mean")
dummy_regr.fit(X_train_flat, y_train_seq)
y_pred_dummy = dummy_regr.predict(X_test_flat)

mse_dummy = mean_squared_error(y_test_seq, y_pred_dummy)
mae_dummy = mean_absolute_error(y_test_seq, y_pred_dummy)

print(f"Dummy MSE: {mse_dummy:.6f}")
print(f"Dummy MAE: {mae_dummy:.6f}")


# ---------------------------------------------------------
# 2. Baseline: Linear Regression
# ---------------------------------------------------------
print("\n--- Baseline 2: Linear Regression ---")
lin_reg = LinearRegression()
lin_reg.fit(X_train_flat, y_train_seq)
y_pred_lin = lin_reg.predict(X_test_flat)

mse_lin = mean_squared_error(y_test_seq, y_pred_lin)
mae_lin = mean_absolute_error(y_test_seq, y_pred_lin)

print(f"Linear MSE: {mse_lin:.6f}")
print(f"Linear MAE: {mae_lin:.6f}")


# ---------------------------------------------------------
# Visualisierung (Linear Regression vs Actual)
# ---------------------------------------------------------
# Rücktransformation für Plotting
try:
    y_test_inv = scaler_y.inverse_transform(y_test_seq)
    y_pred_lin_inv = scaler_y.inverse_transform(y_pred_lin)
    y_pred_dummy_inv = scaler_y.inverse_transform(y_pred_dummy)
except Exception as e:
    print(f"Warnung bei inverse_transform: {e}")
    y_test_inv = y_test_seq
    y_pred_lin_inv = y_pred_lin
    y_pred_dummy_inv = y_pred_dummy

fig, axes = plt.subplots(3, 2, figsize=(15, 12))
fig.suptitle(f'Baseline (Linear Reg) vs Actual Values\nMSE Linear: {mse_lin:.6f}, MSE Dummy: {mse_dummy:.6f}')

n_outputs_to_plot = min(6, y_test_inv.shape[1])
for i in range(n_outputs_to_plot):
    ax = axes[i // 2, i % 2]
    # Plot Actual
    ax.plot(y_test_inv[:100, i], label='Actual', linewidth=2, color='black')
    # Plot Linear Prediction
    ax.plot(y_pred_lin_inv[:100, i], label='Linear Reg', linewidth=2, alpha=0.7, color='blue')
    # Plot Dummy Prediction (als Referenzlinie)
    ax.plot(y_pred_dummy_inv[:100, i], label='Dummy Mean', linewidth=1, linestyle='--', color='red', alpha=0.5)
    
    ax.set_title(f'Output {i + 1}')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Value')
    ax.legend()
    ax.grid(True)

# Leere Subplots entfernen
if n_outputs_to_plot < 6:
     fig.delaxes(axes[2, 1])

plt.tight_layout()
plot_path = os.path.join(images_dir, "06_baseline_results.png")
fig.savefig(plot_path, dpi=200)

print("\nFertig.")

# Metriken speichern
metrics_path = os.path.join(os.path.dirname(__file__), "metrics.txt")
with open(metrics_path, "w") as f:
    f.write(f"Dummy MSE: {mse_dummy:.6f}\n")
    f.write(f"Dummy MAE: {mae_dummy:.6f}\n")
    f.write(f"Linear MSE: {mse_lin:.6f}\n")
    f.write(f"Linear MAE: {mae_lin:.6f}\n")
print(f"Metriken gespeichert unter: {metrics_path}")

