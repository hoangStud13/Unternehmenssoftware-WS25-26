import os
import pandas as pd
import torch
import numpy as np
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
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
    # Daten einlesen
    # -----------------------------
    X_train = torch.tensor(pd.read_csv(X_train_file).iloc[:, 1:].values, dtype=torch.float32)
    X_val   = torch.tensor(pd.read_csv(X_val_file).iloc[:, 1:].values, dtype=torch.float32)
    X_test  = torch.tensor(pd.read_csv(X_test_file).iloc[:, 1:].values, dtype=torch.float32)

    y_train = torch.tensor(pd.read_csv(y_train_file).iloc[:, 1:].values, dtype=torch.float32)
    y_val   = torch.tensor(pd.read_csv(y_val_file).iloc[:, 1:].values, dtype=torch.float32)
    y_test  = torch.tensor(pd.read_csv(y_test_file).iloc[:, 1:].values, dtype=torch.float32)

    # -----------------------------
    # DataLoader
    # -----------------------------
    batch_size = 2048

    train_ds = TensorDataset(X_train, y_train)
    val_ds   = TensorDataset(X_val, y_val)
    test_ds  = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # -----------------------------
    # Modell
    # -----------------------------
    in_dim = X_train.shape[1]
    out_dim = y_train.shape[1]

    hidden1, hidden2 = 1024, 1024
    hidden3, hidden4, hidden5 = 512, 512, 256
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP(
        in_dim, hidden1, hidden2, hidden3, hidden4, hidden5, out_dim, dropout_p
    ).to(device)

    # -----------------------------
    # Loss & Optimizer
    # -----------------------------
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

    # -----------------------------
    # Training
    # -----------------------------
    epochs = 100
    patience = 5
    best_val_loss = float('inf')
    no_improve = 0

    train_loss_hist = []
    val_loss_hist   = []
    test_loss_hist  = []

    for epoch in range(1, epochs + 1):
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

        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                epoch_val_loss += criterion(model(xb), yb).item() * xb.size(0)

        epoch_val_loss /= len(val_loader.dataset)
        val_loss_hist.append(epoch_val_loss)

        epoch_test_loss = 0.0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(device), yb.to(device)
                epoch_test_loss += criterion(model(xb), yb).item() * xb.size(0)

        epoch_test_loss /= len(test_loader.dataset)
        test_loss_hist.append(epoch_test_loss)

        print(f"Epoch {epoch} | Train: {epoch_train_loss:.6f} | Val: {epoch_val_loss:.6f} | Test: {epoch_test_loss:.6f}")

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
    # Plots
    # -----------------------------
    plt.figure(figsize=(12,6))
    plt.plot(train_loss_hist, label="Train Loss")
    plt.plot(val_loss_hist, label="Val Loss")
    plt.legend()
    plt.grid(True, linestyle=':')
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, "06_feed_forward_loss.png"))
    plt.close()

    model.load_state_dict(torch.load(os.path.join(model_dir, "best_model_feed_forward.pt")))
    model.eval()

    actuals, preds = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            preds.append(model(xb).cpu().numpy())
            actuals.append(yb.numpy())

    actuals = np.vstack(actuals)
    preds = np.vstack(preds)

    tolerance = 0.1
    correct = np.sum(np.abs(preds - actuals) / (actuals + 1e-8) <= tolerance)
    accuracy = 100 * correct / actuals.size

    print(f"Approx. Accuracy within ±{tolerance*100:.0f}%: {accuracy:.2f}%")


if __name__ == "__main__":
    main()
