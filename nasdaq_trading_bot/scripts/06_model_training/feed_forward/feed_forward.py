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
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
data_dir = os.path.join(project_root, "data")
img_dir  = os.path.join(project_root, "images")
model_dir = os.path.join(project_root, "models", "feed_forward")
os.makedirs(model_dir, exist_ok=True)

# CSV-Dateien definieren
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
in_dim = X_train.shape[1] #15
out_dim = y_train.shape[1]  # 5 Targets
hidden1 = 1024
hidden2 = 1024
hidden3 = 512
hidden4 = 512
hidden5 = 256
dropout_p = 0.2

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

# gpu verwenden falls verfügbar
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MLP(in_dim, hidden1, hidden2, hidden3, hidden4, hidden5, out_dim, dropout_p).to(device)

# -----------------------------
# Loss & Optimizer
# -----------------------------
criterion = nn.MSELoss() #mean squared error
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

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
        xb, yb = xb.to(device), yb.to(device) #x_batch, y_batch
        optimizer.zero_grad() #gradient zurücksetzen
        out = model(xb) #forward pass -> xb durch das modell jagen
        loss = criterion(out, yb) 
        loss.backward() #backpropagation -> in welche Richtung muss der Gradient die Gewichtung anpassen damit der loss kleiner wird
        optimizer.step() #gewichte anpassen
        epoch_train_loss += loss.item() * xb.size(0) #zuerst summieren wir den epoch train loss über alle batches
    epoch_train_loss /= len(train_loader.dataset)   # dann teilen wir das summierte durch die gesamte anzahl der Trainingssamples
    train_loss_hist.append(epoch_train_loss) # speichern des epoch train loss

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
        torch.save(model.state_dict(), os.path.join(model_dir, "best_model_feed_forward.pt"))
        no_improve = 0
    else:
        no_improve += 1
        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

# -----------------------------
# Plot Loss: Train, Val
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
plt.savefig(os.path.join(img_dir, "06_feed_forward_loss.png"))
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

targets = [1, 3, 5, 10, 15]
limit = 200

# ============================================
# 1) PLOT: ALL DATA
# ============================================
fig, axes = plt.subplots(2, 3, figsize=(18,10))
axes = axes.flatten()


for i, t in enumerate(targets):
    ax = axes[i]
    ax.plot(actuals[:, i], label="Actual", color='tab:blue')
    ax.plot(preds_list[:, i], label="Predicted", color='tab:orange', linestyle='--')
    ax.set_title(f"Target: {t} min (ALL DATA)")
    ax.set_xlabel("Sample")
    ax.set_ylabel("Value")
    ax.legend()
    ax.grid(True, linestyle=':')

# Validation Loss
axes[5].plot(val_loss_hist, color='tab:red')
axes[5].set_title("Validation Loss")
axes[5].set_xlabel("Epoch")
axes[5].set_ylabel("mean MSE Loss")
axes[5].grid(True, linestyle=':')

plt.suptitle("Feed-Forward Model: Actual vs Predicted – ALL DATA", fontsize=16)
plt.tight_layout(rect=[0,0,1,0.96])
plt.savefig(os.path.join(img_dir, "06_feed_forward_predictions_all.png"))
plt.close()
print("Plot ALL DATA gespeichert.")


# ============================================
# 2) PLOT: FIRST 200
# ============================================
fig, axes = plt.subplots(2, 3, figsize=(18,10))
axes = axes.flatten()

for i, t in enumerate(targets):
    ax = axes[i]
    ax.plot(actuals[:limit, i], label="Actual")
    ax.plot(preds_list[:limit, i], label="Predicted", linestyle='--')
    ax.set_title(f"Target: {t} min (FIRST 200)")
    ax.set_xlabel("Sample")
    ax.set_ylabel("Value")
    ax.legend()
    ax.grid(True, linestyle=':')

# Validation Loss wieder bei Plot 6
axes[5].plot(val_loss_hist, color='tab:red')
axes[5].set_title("Validation Loss")
axes[5].set_xlabel("Epoch")
axes[5].set_ylabel("mean MSE Loss")
axes[5].grid(True, linestyle=':')

plt.suptitle("Feed-Forward Predictions – FIRST 200 Points", fontsize=16)
plt.tight_layout(rect=[0,0,1,0.96])
plt.savefig(os.path.join(img_dir, "06_feed_forward_predictions_first_200.png"))
plt.close()
print("Plot FIRST 200 gespeichert.")

# -----------------------------
# Test Accuracy nach Training (Approximation für Regression)
# -----------------------------
# Lade das beste Modell
model.load_state_dict(torch.load(os.path.join(model_dir, "best_model_feed_forward.pt")))
model.eval()

tolerance = 0.1  # z.B. ±10% Toleranz
correct = 0
total = 0

with torch.no_grad():
    for xb, yb in test_loader:
        xb, yb = xb.to(device), yb.to(device)
        out = model(xb)
        rel_error = torch.abs(out - yb) / (yb + 1e-8)  # relative Fehler
        correct += torch.sum(rel_error <= tolerance).item()
        total += yb.numel()

accuracy_percent = 100 * correct / total
print(f"Approx. Accuracy within ±{tolerance*100:.0f}%: {accuracy_percent:.2f}%")
