import os
import matplotlib.pyplot as plt
import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim


class BasicRNN(nn.Module):
    def __init__(self, input_size=15, hidden_size=64, num_layers=1, output_size=6):
        super(BasicRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)  # 5 Outputs

    def forward(self, x):
        out, _ = self.rnn(x)
        out = out[:, -1, :]  # Letzter Zeitschritt
        return self.fc(out)


# Funktion zur Erstellung von Sequenzen für RNN
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
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
data_dir = os.path.join(project_root, 'data')

# Prüfen ob Dateien existieren
csv_files = ['X_train_scaled.csv', 'y_train.csv', 'X_test_scaled.csv', 'y_test.csv']
files_exist = all(os.path.exists(os.path.join(data_dir, f)) for f in csv_files)

if not files_exist:
    print("WARNUNG: CSV-Dateien nicht gefunden. Erstelle Beispieldaten...")

    # Beispieldaten erstellen (ersetze dies mit deinen echten Daten)
    np.random.seed(42)
    n_samples = 1000
    n_features = 15
    n_outputs = 6

    X = np.random.randn(n_samples, n_features)
    y = np.random.randn(n_samples, n_outputs)

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Skalierung
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    y_train_scaled = scaler_y.fit_transform(y_train)
    y_test_scaled = scaler_y.transform(y_test)

else:
    print("Lade existierende CSV-Dateien...")
    X_train_scaled = pd.read_csv(os.path.join(data_dir, 'X_train_scaled.csv')).values
    y_train_scaled = pd.read_csv(os.path.join(data_dir, 'y_train.csv')).values
    X_test_scaled = pd.read_csv(os.path.join(data_dir, 'X_test_scaled.csv')).values
    y_test_scaled = pd.read_csv(os.path.join(data_dir, 'y_test.csv')).values

    # Scaler für Rücktransformation (musst du anpassen)
    scaler_y = MinMaxScaler()
    scaler_y.fit(y_train_scaled)

# Sequenzen erstellen (wichtig für RNN!)
sequence_length = 10
X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_scaled, sequence_length)
X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test_scaled, sequence_length)

print(f"X_train_seq shape: {X_train_seq.shape}")  # Sollte (samples, seq_length, features) sein
print(f"y_train_seq shape: {y_train_seq.shape}")  # Sollte (samples, outputs) sein

# Zu PyTorch Tensoren konvertieren
X_train_tensor = torch.FloatTensor(X_train_seq)
y_train_tensor = torch.FloatTensor(y_train_seq)
X_test_tensor = torch.FloatTensor(X_test_seq)
y_test_tensor = torch.FloatTensor(y_test_seq)

# Model initialisieren
model = BasicRNN(input_size=15, hidden_size=64, num_layers=2, output_size=6)
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

# Rücktransformation (wenn Daten skaliert waren)
try:
    y_test_inv = scaler_y.inverse_transform(y_test_seq)
    predictions_inv = scaler_y.inverse_transform(predictions)
except:
    y_test_inv = y_test_seq
    predictions_inv = predictions

# Visualisierung
fig, axes = plt.subplots(3, 2, figsize=(15, 12))
fig.suptitle('RNN Predictions vs Actual Values')

# Plotte erste 5 Outputs
for i in range(6):
    ax = axes[i // 2, i % 2]
    ax.plot(y_test_inv[:100, i], label='Actual', linewidth=2)
    ax.plot(predictions_inv[:100, i], label='Predicted', linewidth=2, alpha=0.7)
    ax.set_title(f'Output {i + 1}')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Value')
    ax.legend()
    ax.grid(True)

# Loss Plot
axes[2, 1].plot(losses)
axes[2, 1].set_title('Training Loss')
axes[2, 1].set_xlabel('Epoch')
axes[2, 1].set_ylabel('MSE Loss')
axes[2, 1].grid(True)

plt.tight_layout()
plt.show()

print("\nTraining abgeschlossen!")
print(f"Finale Loss: {losses[-1]:.4f}")
print(f"Test MSE: {criterion(torch.FloatTensor(predictions), torch.FloatTensor(y_test_seq)).item():.4f}")