import os
import matplotlib.pyplot as plt
import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim
import joblib 

class BasicRNN(nn.Module):
    def __init__(self, input_size=15, hidden_size=64, num_layers=1, output_size=6):
        super(BasicRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = out[:, -1, :]  # Letzter Zeitschritt
        return self.fc(out)


def create_sequences(X, y, sequence_length=10):
    """
    Erstellt Sequenzen für RNN Training
    X: Features (samples, features)
    y: Targets (samples, outputs)
    sequence_length: Anzahl der Zeitschritte
    """
    X_seq, y_seq = [], []
    for i in range(len(X) - sequence_length):
        X_seq.append(X[i:i + sequence_length])
        y_seq.append(y[i + sequence_length])
    return np.array(X_seq), np.array(y_seq)


# Pfade definieren
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
data_dir = os.path.join(project_root, 'data')
models_dir = os.path.join(project_root, "models", "rnn")
images_dir = os.path.join(project_root, "images")

os.makedirs(models_dir, exist_ok=True)
os.makedirs(images_dir, exist_ok=True)

# Prüfen ob Dateien existieren
csv_files = ['X_train_scaled.csv', 'y_train.csv', 'X_test_scaled.csv', 'y_test.csv']
files_exist = all(os.path.exists(os.path.join(data_dir, f)) for f in csv_files)

# Wir erwarten jetzt diese Dateien aus Step 5:
required_files = [
    'X_train_scaled.csv', 'y_train_scaled.csv',
    'X_val_scaled.csv',   'y_val_scaled.csv',
    'X_test_scaled.csv',  'y_test_scaled.csv',
    'scaler_y.joblib'
]
files_exist = all(os.path.exists(os.path.join(data_dir, f)) for f in required_files)

if not files_exist:
    raise FileNotFoundError("Benötigte Dateien aus Step 5 fehlen. Bitte Step 5 zuerst ausführen.")

print("Lade Step-5-Daten...")

X_train_scaled = pd.read_csv(
    os.path.join(data_dir, 'X_train_scaled.csv'),
    index_col=0
).values

y_train_scaled = pd.read_csv(
    os.path.join(data_dir, 'y_train_scaled.csv'),
    index_col=0
).values

X_val_scaled = pd.read_csv(
    os.path.join(data_dir, 'X_val_scaled.csv'),
    index_col=0
).values

y_val_scaled = pd.read_csv(
    os.path.join(data_dir, 'y_val_scaled.csv'),
    index_col=0
).values

X_test_scaled = pd.read_csv(
    os.path.join(data_dir, 'X_test_scaled.csv'),
    index_col=0
).values

y_test_scaled = pd.read_csv(
    os.path.join(data_dir, 'y_test_scaled.csv'),
    index_col=0
).values

print(f"Shapes:")
print(f"  X_train_scaled: {X_train_scaled.shape}, y_train_scaled: {y_train_scaled.shape}")
print(f"  X_val_scaled:   {X_val_scaled.shape},   y_val_scaled:   {y_val_scaled.shape}")
print(f"  X_test_scaled:  {X_test_scaled.shape},  y_test_scaled:  {y_test_scaled.shape}")

# y-Scaler laden für späteres inverse_transform
scaler_y = joblib.load(os.path.join(data_dir, "scaler_y.joblib"))

# Sequenzen erstellen
sequence_length = 10
X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_scaled, sequence_length)
X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test_scaled, sequence_length)

print(f"\nSequenz-Shapes:")
print(f"X_train_seq shape: {X_train_seq.shape}")  # (samples, seq_length, features)
print(f"y_train_seq shape: {y_train_seq.shape}")  # (samples, outputs)

# Zu PyTorch Tensoren konvertieren
X_train_tensor = torch.FloatTensor(X_train_seq)
y_train_tensor = torch.FloatTensor(y_train_seq)
X_test_tensor = torch.FloatTensor(X_test_seq)
y_test_tensor = torch.FloatTensor(y_test_seq)

# Model initialisieren
n_features = X_train_scaled.shape[1]
n_outputs = y_train_scaled.shape[1]

model = BasicRNN(input_size=n_features, hidden_size=64, num_layers=2, output_size=n_outputs)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

EPOCHS = 100
losses = []

print("\nTraining startet...")
for epoch in range(EPOCHS):
    model.train()

    # Forward pass
    output = model(X_train_tensor)
    loss = criterion(output, y_train_tensor)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    losses.append(loss.item())

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/{EPOCHS}], Loss: {loss.item():.4f}")

# Evaluation
model.eval()
with torch.no_grad():
    predictions = model(X_test_tensor).numpy()

# Rücktransformation
try:
    y_test_inv = scaler_y.inverse_transform(y_test_seq)
    predictions_inv = scaler_y.inverse_transform(predictions)
except Exception as e:
    print(f"Warnung bei inverse_transform: {e}")
    y_test_inv = y_test_seq
    predictions_inv = predictions

# Visualisierung
fig, axes = plt.subplots(3, 2, figsize=(15, 12))
fig.suptitle('RNN Predictions vs Actual Values')

# Plotte alle Outputs
n_outputs_to_plot = min(6, y_test_inv.shape[1])
for i in range(n_outputs_to_plot):
    ax = axes[i // 2, i % 2]
    ax.plot(y_test_inv[:100, i], label='Actual', linewidth=2)
    ax.plot(predictions_inv[:100, i], label='Predicted', linewidth=2, alpha=0.7)
    ax.set_title(f'Output {i + 1}')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Value')
    ax.legend()
    ax.grid(True)

# Loss Plot
if n_outputs_to_plot < 6:
    axes[2, 1].plot(losses)
    axes[2, 1].set_title('Training Loss')
    axes[2, 1].set_xlabel('Epoch')
    axes[2, 1].set_ylabel('MSE Loss')
    axes[2, 1].grid(True)
else:
    # Falls wir 6 Outputs haben, entferne leere Subplots
    fig.delaxes(axes[2, 1])

plt.tight_layout()
plt.show()
plot_path = os.path.join(images_dir, "06_rnn_results.png")
fig.savefig(plot_path, dpi=200)
print(f"Plot gespeichert unter: {plot_path}")

print("\nTraining abgeschlossen!")
print(f"Finale Loss: {losses[-1]:.4f}")
print(f"Test MSE: {criterion(torch.FloatTensor(predictions), torch.FloatTensor(y_test_seq)).item():.4f}")

# -------------------------------
# Modell speichern
# -------------------------------

model_path = os.path.join(models_dir, "best_rnn_model.pth")
torch.save(model.state_dict(), model_path)

print(f"\nModell gespeichert unter: {model_path}")