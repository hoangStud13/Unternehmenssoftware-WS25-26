import matplotlib
matplotlib.use("Agg")  # Für Umgebungen ohne Display
import matplotlib.pyplot as plt
import pandas as pd
import os

# Pfad zu deinen gespeicherten Splits
data_dir = "nasdaq_trading_bot/data"
img_dir= "nasdaq_trading_bot/images"

train_file = os.path.join(data_dir, "nasdaq_train.parquet")
val_file   = os.path.join(data_dir, "nasdaq_validation.parquet")
test_file  = os.path.join(data_dir, "nasdaq_test.parquet")

# Einlesen
train_len = len(pd.read_parquet(train_file))
val_len   = len(pd.read_parquet(val_file))
test_len  = len(pd.read_parquet(test_file))

# Für den Plot
splits = ["Train", "Validation", "Test"]
values = [train_len, val_len, test_len]

colors = ["#2ecc71", "#f39c12", "#e74c3c"]  # Grün / Orange / Rot

plt.figure(figsize=(10, 5))
plt.bar(splits, values, color=colors)

plt.title("Distribution of Samples Across Splits")
plt.ylabel("Number of Samples")
plt.grid(axis='y', linestyle=':', alpha=0.7)

for i, v in enumerate(values):
    plt.text(i, v, f"{v:,}", ha='center', va='bottom', fontsize=10, fontweight="bold")

plt.tight_layout()
plt.savefig(os.path.join(img_dir, "04_split_distribution.png"))
plt.close()

print("Saved 04_split_distribution.png in:", img_dir)
