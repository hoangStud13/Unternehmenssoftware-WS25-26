import os
import matplotlib.pyplot as plt
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import joblib


def main():
    # Reproduzierbarkeit
    torch.manual_seed(42)
    np.random.seed(42)
    torch.set_num_threads(4)

    # Device (Nur GPU)
    if not torch.cuda.is_available():
        raise RuntimeError("GPU-Verarbeitung angefordert, aber keine GPU gefunden (CUDA nicht verfügbar).")

    device = torch.device("cuda")
    print("Verwende Device: gpu")

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

    def create_sequences(X, y, sequence_length=50):
        X_seq, y_seq = [], []
        for i in range(len(X) - sequence_length):
            X_seq.append(X[i:i + sequence_length])
            y_seq.append(y[i + sequence_length])
        return np.array(X_seq), np.array(y_seq)

    # -------------------------------
    # Daten laden
    # -------------------------------
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
    data_dir = os.path.join(project_root, 'data')
    models_dir = os.path.join(project_root, "models", "lstm")
    images_dir = os.path.join(project_root, "images")

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)

    required_files = [
        'X_train_scaled.csv', 'y_train_scaled.csv',
        'X_val_scaled.csv',   'y_val_scaled.csv',
        'X_test_scaled.csv',  'y_test_scaled.csv',
        'scaler_y.joblib'
    ]

    if not all(os.path.exists(os.path.join(data_dir, f)) for f in required_files):
        raise FileNotFoundError("Benötigte Dateien aus Step 5 fehlen. Bitte Step 5 zuerst ausführen.")

    print("Lade Step-5-Daten...")

    X_train_scaled = pd.read_csv(os.path.join(data_dir, 'X_train_scaled.csv'), index_col=0).values
    y_train_scaled = pd.read_csv(os.path.join(data_dir, 'y_train_scaled.csv'), index_col=0).values
    X_val_scaled   = pd.read_csv(os.path.join(data_dir, 'X_val_scaled.csv'),   index_col=0).values
    y_val_scaled   = pd.read_csv(os.path.join(data_dir, 'y_val_scaled.csv'),   index_col=0).values
    X_test_scaled  = pd.read_csv(os.path.join(data_dir, 'X_test_scaled.csv'),  index_col=0).values
    y_test_scaled  = pd.read_csv(os.path.join(data_dir, 'y_test_scaled.csv'),  index_col=0).values

    scaler_y = joblib.load(os.path.join(data_dir, "scaler_y.joblib"))

    # -------------------------------
    # Sequenzen
    # -------------------------------
    sequence_length = 50
    X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_scaled, sequence_length)
    X_val_seq,   y_val_seq   = create_sequences(X_val_scaled,   y_val_scaled,   sequence_length)
    X_test_seq,  y_test_seq  = create_sequences(X_test_scaled,  y_test_scaled,  sequence_length)

    train_loader = DataLoader(SequenceDataset(X_train_seq, y_train_seq), batch_size=64, shuffle=True)
    val_loader   = DataLoader(SequenceDataset(X_val_seq,   y_val_seq),   batch_size=64)
    test_loader  = DataLoader(SequenceDataset(X_test_seq,  y_test_seq),  batch_size=64)

    # -------------------------------
    # Modell, Loss, Optimizer
    # -------------------------------
    model = LSTMModel(
        input_size=X_train_seq.shape[2],
        hidden_size=384,
        num_layers=2,
        output_size=y_train_seq.shape[1],
        dropout=0.2
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3)

    # -------------------------------
    # Training
    # -------------------------------
    EPOCHS = 200
    patience = 10
    min_delta = 1e-5

    losses_train, losses_val = [], []
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    print("\nTraining startet...")
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)

        train_loss /= len(train_loader.dataset)
        losses_train.append(train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                val_loss += criterion(model(xb), yb).item() * xb.size(0)

        val_loss /= len(val_loader.dataset)
        losses_val.append(val_loss)
        scheduler.step(val_loss)

        print(f"Epoch {epoch+1}/{EPOCHS} | Train {train_loss:.6f} | Val {val_loss:.6f}")

        if val_loss + min_delta < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(">>> Early Stopping <<<")
                break

    model.load_state_dict(best_model_state)

    # -------------------------------
    # Modell speichern
    # -------------------------------
    model_path = os.path.join(models_dir, "best_lstm_model.pth")
    torch.save(best_model_state, model_path)
    print(f"Modell gespeichert unter: {model_path}")


if __name__ == "__main__":
    main()
