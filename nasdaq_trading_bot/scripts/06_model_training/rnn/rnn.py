import os
import matplotlib.pyplot as plt
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.optim as optim
import joblib


def main():
    class BasicRNN(nn.Module):
        def __init__(self, input_size=15, hidden_size=64, num_layers=1, output_size=6):
            super(BasicRNN, self).__init__()
            self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_size, output_size)

        def forward(self, x):
            out, _ = self.rnn(x)
            out = out[:, -1, :]  # letzter Zeitschritt
            return self.fc(out)

    def create_sequences(X, y, sequence_length=10):
        X_seq, y_seq = [], []
        for i in range(len(X) - sequence_length):
            X_seq.append(X[i:i + sequence_length])
            y_seq.append(y[i + sequence_length])
        return np.array(X_seq), np.array(y_seq)

    # -------------------------------
    # Pfade
    # -------------------------------
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
    data_dir = os.path.join(project_root, 'data')
    models_dir = os.path.join(project_root, "models", "rnn")
    images_dir = os.path.join(project_root, "images")

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)

    # -------------------------------
    # Dateien prüfen
    # -------------------------------
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

    print(f"Shapes:")
    print(f"  X_train_scaled: {X_train_scaled.shape}, y_train_scaled: {y_train_scaled.shape}")
    print(f"  X_val_scaled:   {X_val_scaled.shape},   y_val_scaled:   {y_val_scaled.shape}")
    print(f"  X_test_scaled:  {X_test_scaled.shape},  y_test_scaled:  {y_test_scaled.shape}")

    scaler_y = joblib.load(os.path.join(data_dir, "scaler_y.joblib"))

    # -------------------------------
    # Sequenzen
    # -------------------------------
    sequence_length = 10
    X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_scaled, sequence_length)
    X_val_seq,   y_val_seq   = create_sequences(X_val_scaled,   y_val_scaled,   sequence_length)
    X_test_seq,  y_test_seq  = create_sequences(X_test_scaled,  y_test_scaled,  sequence_length)

    print("\nSequenz-Shapes:")
    print(f"X_train_seq: {X_train_seq.shape}")
    print(f"y_train_seq: {y_train_seq.shape}")
    print(f"X_val_seq:   {X_val_seq.shape}")
    print(f"y_val_seq:   {y_val_seq.shape}")
    print(f"X_test_seq:  {X_test_seq.shape}")
    print(f"y_test_seq:  {y_test_seq.shape}")

    # -------------------------------
    # Tensoren
    # -------------------------------
    X_train_tensor = torch.FloatTensor(X_train_seq)
    y_train_tensor = torch.FloatTensor(y_train_seq)
    X_val_tensor   = torch.FloatTensor(X_val_seq)
    y_val_tensor   = torch.FloatTensor(y_val_seq)
    X_test_tensor  = torch.FloatTensor(X_test_seq)
    y_test_tensor  = torch.FloatTensor(y_test_seq)

    # -------------------------------
    # Modell
    # -------------------------------
    n_features = X_train_scaled.shape[1]
    n_outputs  = y_train_scaled.shape[1]

    model = BasicRNN(
        input_size=n_features,
        hidden_size=64,
        num_layers=2,
        output_size=n_outputs
    )

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    EPOCHS = 100
    patience = 10

    losses_train = []
    losses_val = []

    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    # -------------------------------
    # Training
    # -------------------------------
    print("\nTraining startet...")
    for epoch in range(EPOCHS):
        model.train()
        output = model(X_train_tensor)
        loss = criterion(output, y_train_tensor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses_train.append(loss.item())

        model.eval()
        with torch.no_grad():
            val_output = model(X_val_tensor)
            val_loss = criterion(val_output, y_val_tensor)

        losses_val.append(val_loss.item())

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{EPOCHS}] | Train: {loss.item():.4f} | Val: {val_loss.item():.4f}")

        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n>>> Early Stopping bei Epoch {epoch + 1} <<<")
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\nBestes Modell mit Val Loss: {best_val_loss:.4f} geladen.")

    # -------------------------------
    # Evaluation
    # -------------------------------
    model.eval()
    with torch.no_grad():
        predictions = model(X_test_tensor).numpy()

    try:
        y_test_inv = scaler_y.inverse_transform(y_test_seq)
        predictions_inv = scaler_y.inverse_transform(predictions)
    except Exception as e:
        print(f"Inverse transform fehlgeschlagen: {e}")
        y_test_inv = y_test_seq
        predictions_inv = predictions

    # -------------------------------
    # Plot
    # -------------------------------
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle('RNN Predictions vs Actual Values')

    n_outputs_to_plot = min(6, y_test_inv.shape[1])
    for i in range(n_outputs_to_plot):
        ax = axes[i // 2, i % 2]
        ax.plot(y_test_inv[:100, i], label='Actual')
        ax.plot(predictions_inv[:100, i], label='Predicted', linestyle='--')
        ax.set_title(f'Output {i + 1}')
        ax.legend()
        ax.grid(True)

    epochs_range = range(1, len(losses_train) + 1)
    axes[2, 1].plot(epochs_range, losses_train, label='Train Loss')
    axes[2, 1].plot(epochs_range, losses_val, label='Val Loss')
    axes[2, 1].legend()
    axes[2, 1].grid(True)

    plt.tight_layout()
    plot_path = os.path.join(images_dir, "06_rnn_results.png")
    fig.savefig(plot_path, dpi=200)
    print(f"Plot gespeichert unter: {plot_path}")

    # -------------------------------
    # Metriken
    # -------------------------------
    test_mse_scaled = criterion(
        torch.FloatTensor(predictions),
        torch.FloatTensor(y_test_seq)
    ).item()

    test_mse_original = np.mean((predictions_inv - y_test_inv) ** 2)

    print("\n" + "=" * 50)
    print("Training abgeschlossen!")
    print("=" * 50)
    print(f"Finale Train Loss (scaled): {losses_train[-1]:.4f}")
    print(f"Finale Val Loss (scaled):   {best_val_loss:.4f}")
    print(f"Test MSE (scaled):          {test_mse_scaled:.4f}")
    print(f"Test MSE (original scale):  {test_mse_original:.4f}")
    print("=" * 50)

    # -------------------------------
    # Modell speichern
    # -------------------------------
    model_path = os.path.join(models_dir, "best_rnn_model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"\nModell gespeichert unter: {model_path}")


if __name__ == "__main__":
    main()
