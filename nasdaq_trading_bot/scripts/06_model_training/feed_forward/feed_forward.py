import os
import pandas as pd
import torch
import numpy as np
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -----------------------------
# Pfade
# -----------------------------
data_dir = "nasdaq_trading_bot/data"
img_dir  = "nasdaq_trading_bot/images"
model_dir = "nasdaq_trading_bot/models/feed_forward"
os.makedirs(img_dir, exist_ok=True)

# CSV-Dateien
X_train_file = os.path.join(data_dir, "X_train_scaled.csv")
X_val_file   = os.path.join(data_dir, "X_val_scaled.csv")
X_test_file  = os.path.join(data_dir, "X_test_scaled.csv")

y_train_file = os.path.join(data_dir, "y_train_scaled.csv")
y_val_file   = os.path.join(data_dir, "y_val_scaled.csv")
y_test_file  = os.path.join(data_dir, "y_test_scaled.csv")

# -----------------------------
# Daten einlesen und vorbereiten
# -----------------------------
# Features: erste und letzte Spalte droppen + direkt Tensor
X_train = torch.tensor(pd.read_csv(X_train_file).iloc[:, 1:].values, dtype=torch.float32)
X_val   = torch.tensor(pd.read_csv(X_val_file).iloc[:, 1:].values, dtype=torch.float32)
X_test  = torch.tensor(pd.read_csv(X_test_file).iloc[:, 1:].values, dtype=torch.float32)

# Targets: 5 Zeithorizonte (1,3,5,10,15 min)
y_train = torch.tensor(pd.read_csv(y_train_file).iloc[:, 1:].values, dtype=torch.float32)
y_val   = torch.tensor(pd.read_csv(y_val_file).iloc[:, 1:].values, dtype=torch.float32)
y_test  = torch.tensor(pd.read_csv(y_test_file).iloc[:, 1:].values, dtype=torch.float32)

# -----------------------------
# DataLoader erstellen
# -----------------------------
batch_size = 2048

train_ds = TensorDataset(X_train, y_train)
val_ds   = TensorDataset(X_val, y_val)
test_ds  = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

# -----------------------------
# Modell definieren
# -----------------------------
in_dim = X_train.shape[1]
out_dim = y_train.shape[1]  # 5 Targets
hidden1 = 1024
hidden2 = 1024
hidden3 = 512
hidden4 = 512
hidden5 = 256
dropout_p = 0.1

class MLP(nn.Module):
    def __init__(self, in_dim, h1, h2, h3, h4, h5, out_dim, dropout_p):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, h1),
            nn.LeakyReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(h1, h2),
            nn.LeakyReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(h2, h3),
            nn.LeakyReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(h3, h4),
            nn.LeakyReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(h4, h5),
            nn.LeakyReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(h5, out_dim)
        )
    def forward(self, x):
        return self.net(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MLP(in_dim, hidden1, hidden2, hidden3, hidden4, hidden5, out_dim, dropout_p).to(device)

# -----------------------------
# Loss & Optimizer
# -----------------------------
criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

# -----------------------------
# Training Loop mit Early Stopping
# -----------------------------
epochs = 100
patience = 5
best_val_loss = float('inf')
no_improve = 0

train_loss_hist = []
val_loss_hist   = []
test_loss_hist  = []

for epoch in range(1, epochs+1):
    # Training
    model.train()
    epoch_train_loss = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        epoch_train_loss += loss.item() * xb.size(0)
    epoch_train_loss /= len(train_loader.dataset)
    train_loss_hist.append(epoch_train_loss)

    # Validation
    model.eval()
    epoch_val_loss = 0.0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            epoch_val_loss += criterion(out, yb).item() * xb.size(0)
    epoch_val_loss /= len(val_loader.dataset)
    val_loss_hist.append(epoch_val_loss)

    # Test
    model.eval()
    epoch_test_loss = 0.0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            epoch_test_loss += criterion(out, yb).item() * xb.size(0)
    epoch_test_loss /= len(test_loader.dataset)
    test_loss_hist.append(epoch_test_loss)

    print(f"Epoch {epoch} | Train: {epoch_train_loss:.6f} | Val: {epoch_val_loss:.6f} | Test: {epoch_test_loss:.6f}")

    # Early Stopping
    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        torch.save(model.state_dict(), os.path.join(model_dir, "best_model.pt"))
        no_improve = 0
    else:
        no_improve += 1
        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

# -----------------------------
# Plot Loss: Train, Val, Test
# -----------------------------
plt.figure(figsize=(12,6))
plt.plot(train_loss_hist, label="Train Loss", color='tab:blue')
plt.plot(val_loss_hist, label="Val Loss", color='tab:orange')
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Train / Validation Loss")
plt.grid(True, linestyle=':', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(img_dir, "06_feed_forward_loss_curves_all_data.png"))
plt.close()
print("Loss-Kurven gespeichert in:", img_dir)

# -----------------------------
# Plot Actual vs Predicted Linien (2x3 Grid)
# -----------------------------
model.eval()
actuals = []
preds_list = []

with torch.no_grad():
    for xb, yb in test_loader:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb)
        actuals.append(yb.cpu().numpy())
        preds_list.append(pred.cpu().numpy())

actuals = np.vstack(actuals)
preds_list = np.vstack(preds_list)

fig, axes = plt.subplots(2, 3, figsize=(18,10))
axes = axes.flatten()
targets = [1, 3, 5, 10, 15]  # Minuten

for i, t in enumerate(targets):
    ax = axes[i]
    ax.plot(actuals[:, i], label="Actual", color='tab:blue')
    ax.plot(preds_list[:, i], label="Predicted", color='tab:orange', linestyle='--')
    ax.set_title(f"Target: {t} min")
    ax.set_xlabel("Sample")
    ax.set_ylabel("Value")
    ax.legend()
    ax.grid(True, linestyle=':')

# Validation Loss im 6. Subplot
axes[5].plot(val_loss_hist, color='tab:red')
axes[5].set_title("Validation Loss")
axes[5].set_xlabel("Epoch")
axes[5].set_ylabel("MSE Loss")
axes[5].grid(True, linestyle=':')

plt.suptitle("Feed-Forward Model: Actual vs Predicted Curves & Validation Loss", fontsize=16)
plt.tight_layout(rect=[0,0,1,0.96])
plt.savefig(os.path.join(img_dir, "06_feed_forward_actual_vs_predicted_curves._all_data.png"))
plt.close()
print("Actual vs Predicted Linienplots gespeichert in:", img_dir)
