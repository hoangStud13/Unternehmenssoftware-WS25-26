import os
import matplotlib.pyplot as plt
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import joblib

# Reproduzierbarkeit
torch.manual_seed(42)
np.random.seed(42)
torch.set_num_threads(4)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Verwende Device: {device}")


# -------------------------------
# LSTM-Modell
# -------------------------------

class LSTMModel(nn.Module):
    def __init__(
        self,
        input_size=15,
        hidden_size=64,
        num_layers=1,
        output_size=6,
        bidirectional=False,
        dropout=0.0
    ):
        super(LSTMModel, self).__init__()

        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional
        )

        self.fc = nn.Linear(hidden_size * self.num_directions, output_size)

    def forward(self, x):
        out, (h_n, c_n) = self.lstm(x)
        last_layer_h = h_n[-self.num_directions:, :, :]
        last_layer_h = last_layer_h.transpose(0, 1).reshape(x.size(0), -1)
        return self.fc(last_layer_h)


# -------------------------------
# Dataset & Sequenz-Erzeugung
# -------------------------------

class SequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def create_sequences(X, y, sequence_length=20):
    X_seq, y_seq = [], []
    for i in range(len(X) - sequence_length):
        X_seq.append(X[i:i + sequence_length])
        y_seq.append(y[i + sequence_length])
    return np.array(X_seq), np.array(y_seq)


# -------------------------------
# Daten laden (aus Step 5)
# -------------------------------

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
data_dir = os.path.join(project_root, 'data')
models_dir = os.path.join(project_root, "models", "lstm")
images_dir = os.path.join(project_root, "images")

os.makedirs(models_dir, exist_ok=True)
os.makedirs(images_dir, exist_ok=True)

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

# -------------------------------
# Sequenzen pro Split erzeugen
# -------------------------------

sequence_length = 20

X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_scaled, sequence_length)
X_val_seq,   y_val_seq   = create_sequences(X_val_scaled,   y_val_scaled,   sequence_length)
X_test_seq,  y_test_seq  = create_sequences(X_test_scaled,  y_test_scaled,  sequence_length)

print("\nSequenz-Shapes:")
print(f"  X_train_seq: {X_train_seq.shape}, y_train_seq: {y_train_seq.shape}")
print(f"  X_val_seq:   {X_val_seq.shape},   y_val_seq:   {y_val_seq.shape}")
print(f"  X_test_seq:  {X_test_seq.shape},  y_test_seq:  {y_test_seq.shape}")

train_dataset = SequenceDataset(X_train_seq, y_train_seq)
val_dataset   = SequenceDataset(X_val_seq,   y_val_seq)
test_dataset  = SequenceDataset(X_test_seq,  y_test_seq)

batch_size = 64

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=0)
test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=0)

# -------------------------------
# Modell, Loss, Optimizer
# -------------------------------

n_features = X_train_seq.shape[2]
n_outputs  = y_train_seq.shape[1]

model = LSTMModel(
    input_size=n_features,
    hidden_size=256,
    num_layers=2,
    output_size=n_outputs,
    bidirectional=False,
    dropout=0.2
).to(device)

criterion = nn.MSELoss()  # oder nn.SmoothL1Loss(beta=1.0)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    factor=0.5,
    patience=3,   # wenn Val-Loss 3 Epochen nicht besser wird: LR halbieren
)

EPOCHS = 200
losses_train = []
losses_val = []

patience = 10
min_delta = 1e-5
best_val_loss = float('inf')
patience_counter = 0
best_model_state = None



# -------------------------------
# Training mit Early Stopping
# -------------------------------

print("\nTraining startet...")
for epoch in range(EPOCHS):
    # ----- Training -----
    model.train()
    epoch_train_loss = 0.0

    for batch_idx, (xb, yb) in enumerate(train_loader):
        xb = xb.to(device)
        yb = yb.to(device)

        optimizer.zero_grad()
        output = model(xb)
        loss = criterion(output, yb)
        loss.backward()
        optimizer.step()

        epoch_train_loss += loss.item() * xb.size(0)

        if (batch_idx + 1) % 500 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} - Batch {batch_idx+1}/{len(train_loader)} - Train Loss: {loss.item():.6f}")

    epoch_train_loss /= len(train_dataset)
    losses_train.append(epoch_train_loss)

    # ----- Validation -----
    model.eval()
    epoch_val_loss = 0.0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            out = model(xb)
            val_loss = criterion(out, yb)
            epoch_val_loss += val_loss.item() * xb.size(0)

    epoch_val_loss /= len(val_dataset)
    losses_val.append(epoch_val_loss)

    print(f"==> Epoch [{epoch + 1}/{EPOCHS}], Train Loss: {epoch_train_loss:.6f}, Val Loss: {epoch_val_loss:.6f}")
    # LR-Scheduler updaten
    scheduler.step(epoch_val_loss)
    
    # Early Stopping
    if epoch_val_loss + min_delta < best_val_loss:
        best_val_loss = epoch_val_loss
        patience_counter = 0
        best_model_state = model.state_dict()
    else:
        patience_counter += 1
        print(f"   Keine Verbesserung. Patience: {patience_counter}/{patience}")
        if patience_counter >= patience:
            print("\n>>> Early Stopping ausgelöst! <<<")
            break

if best_model_state is not None:
    model.load_state_dict(best_model_state)
    print(f"\nBestes Modell mit Val Loss: {best_val_loss:.6f} geladen.")


# -------------------------------
# Evaluation auf Testdaten
# -------------------------------

model.eval()
all_preds = []
all_targets = []

with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(device)
        out = model(xb)
        all_preds.append(out.cpu().numpy())
        all_targets.append(yb.cpu().numpy())

predictions_scaled = np.concatenate(all_preds, axis=0)
y_test_scaled_seq  = np.concatenate(all_targets, axis=0)

# Rücktransformation in Original-Skala
y_test_inv   = scaler_y.inverse_transform(y_test_scaled_seq)
predictions_inv = scaler_y.inverse_transform(predictions_scaled)

# -------------------------------
# Visualisierung
# -------------------------------

fig, axes = plt.subplots(3, 2, figsize=(15, 12))
fig.suptitle('LSTM Predictions vs Actual Values & Loss')

axes_flat = axes.flatten()  # [ax0, ax1, ax2, ax3, ax4, ax5]

# bis zu 5 Outputs anzeigen
n_outputs_to_plot = min(5, y_test_inv.shape[1])

# ---- Outputs 1..5 ----
for i in range(n_outputs_to_plot):
    ax = axes_flat[i]
    ax.plot(y_test_inv[:200, i], label='Actual', linewidth=2)
    ax.plot(predictions_inv[:200, i], label='Predicted', linewidth=2, alpha=0.7)
    ax.set_title(f'Output {i + 1}')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Value')
    ax.legend()
    ax.grid(True)

# evtl. Zwischenslots löschen (z.B. wenn <5 Outputs)
for j in range(n_outputs_to_plot, len(axes_flat) - 1):
    fig.delaxes(axes_flat[j])

# ---- Loss unten rechts ----
loss_ax = axes_flat[-1]
epochs_range = range(1, len(losses_train) + 1)
loss_ax.plot(epochs_range, losses_train, label='Train Loss')
loss_ax.plot(epochs_range, losses_val,   label='Val Loss')
loss_ax.set_title('Training & Validation Loss')
loss_ax.set_xlabel('Epoch')
loss_ax.set_ylabel('MSE Loss')
loss_ax.legend()
loss_ax.grid(True)

plt.tight_layout()
plt.show()
plot_path = os.path.join(images_dir, "06_lstm_results.png")
fig.savefig(plot_path, dpi=200)
print(f"Plot gespeichert unter: {plot_path}")

# -------------------------------
# Metriken (im Original-Space)
# -------------------------------

test_mse = np.mean((predictions_inv - y_test_inv) ** 2)
print("\nTraining abgeschlossen!")
print(f"Finaler Train Loss (scaled): {losses_train[-1]:.6f}")
print(f"Finaler Val   Loss (scaled): {losses_val[-1]:.6f}")
print(f"Test MSE (original scale):   {test_mse:.6f}")

# -------------------------------
# Modell speichern
# -------------------------------

model_path = os.path.join(models_dir, "best_lstm_model.pth")
torch.save(best_model_state, model_path)

print(f"\nModell gespeichert unter: {model_path}")

