"""
Train a scikit-learn DecisionTree on embeddings from the trained MLP (best_acc_model_temp.pt),
loading as much training data as possible within available memory.

Steps:
- Load the trained MLP checkpoint and feature order from checkpoint if present.
- Stream through training parquet shards in batches.
- Extract 2nd hidden layer embeddings (64D by default) for each batch on-the-fly.
- Accumulate as much data as fits into a memory budget (based on psutil.available memory).
- Train sklearn.tree.DecisionTreeClassifier on the accumulated embeddings.
- Save the trained tree to disk and print a brief summary.
"""

import os
import glob
import pickle
import math
import yaml
import psutil
import numpy as np
import pandas as pd
import torch
from torch import nn
import pyarrow.parquet as pq
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score

# -----------------------------
# Config and paths
# -----------------------------
params = yaml.safe_load(open("../../conf/params.yaml"))
processed_path = f"{params['DATA_PREP']['PROCESSED_PATH']}_shuffled"
target = params['MODELING']['TARGET']
feature_path = params['DATA_PREP'].get('FEATURE_PATH')
model_path = params['MODELING']['MODEL_PATH']
os.makedirs(model_path, exist_ok=True)

# Read features (fallback); may be overridden by checkpoint
features = []
with open(feature_path, "r") as f:
    for line in f:
        feat = line.strip()
        if feat:
            features.append(feat)

# -----------------------------
# Define the same MLP and load weights
# -----------------------------
hidden1 = params['MODELING'].get('HIDDEN1', 128)
hidden2 = params['MODELING'].get('HIDDEN2', 64)
dropout_p = params['MODELING'].get('DROPOUT', 0.1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MLP(nn.Module):
    """MLP architecture matching the training script (Sequential under `net`).

    net.0: Linear(in_dim -> h1)
    net.1: ReLU
    net.2: Dropout
    net.3: Linear(h1 -> h2)
    net.4: Dropout
    net.5: Linear(h2 -> 1)

    The embed() method returns the representation after net.4 (i.e., output of second hidden layer), shape (N, h2).
    """
    def __init__(self, in_dim, h1, h2, dropout_p):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, h1),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(h1, h2),
            nn.Dropout(dropout_p),
            nn.Linear(h2, 1)
        )

    def forward(self, x):
        return self.net(x)

    def embed(self, x):
        x = self.net[0](x)
        x = self.net[1](x)
        x = self.net[2](x)
        x = self.net[3](x)
        x = self.net[4](x)
        return x

# Load checkpoint and align config/feature order
checkpoint_path = os.path.join(model_path, "best_acc_model_temp.pt")
checkpoint = torch.load(checkpoint_path, map_location=device)

ckpt_features = checkpoint.get("feature_cols")
if ckpt_features and isinstance(ckpt_features, (list, tuple)):
    if list(ckpt_features) != list(features):
        print("[WARN] Using feature list from checkpoint to match trained model.")
    features = list(ckpt_features)

ckpt_cfg = checkpoint.get("config", {})
hidden1 = ckpt_cfg.get('hidden1', hidden1)
hidden2 = ckpt_cfg.get('hidden2', hidden2)
dropout_p = ckpt_cfg.get('dropout', dropout_p)

in_dim = len(features)
model = MLP(in_dim, hidden1, hidden2, dropout_p).to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# -----------------------------
# Memory budget and helpers
# -----------------------------
init_avail = psutil.virtual_memory().available
# Use up to 70% of currently available memory, but cap at 16 GB for safety
limit_bytes = int(min(16 * (1024**3), init_avail * 0.70))
print(f"[MEM] Available: {init_avail/1e9:.2f} GB, Budget: {limit_bytes/1e9:.2f} GB")

X_chunks = []  # list of np.ndarray (float32)
y_chunks = []  # list of np.ndarray (int8)
bytes_used = 0

# -----------------------------
# Stream training shards and accumulate until budget is reached
# -----------------------------
train_files = sorted(glob.glob(os.path.join(processed_path, "train_shard_*.parquet")))
num_files = len(train_files)
overall_rows_seen = 0

for file_idx, file in enumerate(train_files, start=1):
    print(f"[File {file_idx}/{num_files}] {os.path.basename(file)}")
    pq_file = pq.ParquetFile(file)

    for batch_idx, batch in enumerate(pq_file.iter_batches(batch_size=2048, columns=features + [target]), start=1):
        df = batch.to_pandas()
        if df.empty:
            continue
        X_tensor = torch.tensor(df[features].values, dtype=torch.float32, device=device)
        with torch.no_grad():
            emb = model.embed(X_tensor).detach().cpu().numpy().astype(np.float32, copy=False)
        y = (df[target].values >= 0).astype(np.int8, copy=False)

        batch_bytes = emb.nbytes + y.nbytes
        # If adding this batch would exceed budget, stop loading more
        if bytes_used + batch_bytes > limit_bytes:
            print(f"[MEM] Budget reached. Stopping accumulation at file {file_idx}, batch {batch_idx}.")
            stop = True
        else:
            stop = False

        if not stop:
            X_chunks.append(emb)
            y_chunks.append(y)
            bytes_used += batch_bytes
            overall_rows_seen += len(df)
        else:
            # Try to add a fraction of the batch that still fits (optional fine-grained fill)
            remaining = max(0, limit_bytes - bytes_used)
            if remaining >= (y.itemsize + emb.itemsize):
                # estimate per-row cost
                per_row_bytes = emb.shape[1] * emb.itemsize + y.itemsize
                n_can_add = int(remaining // per_row_bytes)
                if n_can_add > 0:
                    X_chunks.append(emb[:n_can_add])
                    y_chunks.append(y[:n_can_add])
                    bytes_used += n_can_add * per_row_bytes
                    overall_rows_seen += n_can_add
                    print(f"[MEM] Added partial batch rows={n_can_add} to fill budget.")
            break
    else:
        # only executed if the inner loop didn't break
        continue
    # inner loop broke -> break outer
    break

print(f"[LOAD] Accumulated rows: {overall_rows_seen} | Estimated bytes: {bytes_used/1e9:.2f} GB")

if overall_rows_seen == 0:
    raise RuntimeError("No training samples could be loaded within the memory budget.")

# Concatenate all chunks
X = np.concatenate(X_chunks, axis=0)
y = np.concatenate(y_chunks, axis=0)
print(f"[SHAPE] X={X.shape} (float32), y={y.shape} (int8)")

# -----------------------------
# Train scikit-learn Decision Tree
# -----------------------------
# Sensible defaults; can be tuned via params.yaml in the future if needed
max_depth = params['MODELING'].get('DT_MAX_DEPTH', 8)
#min_samples_split = params['MODELING'].get('DT_MIN_SAMPLES_SPLIT', 2)
#min_samples_leaf = params['MODELING'].get('DT_MIN_SAMPLES_LEAF', 1)
#class_weight = params['MODELING'].get('DT_CLASS_WEIGHT')  # e.g., 'balanced' or dict
#random_state = params['MODELING'].get('RANDOM_STATE', 42)

clf = DecisionTreeClassifier(
    max_depth=max_depth,
    #min_samples_split=min_samples_split,
    #min_samples_leaf=min_samples_leaf,
    #class_weight=class_weight,
    #random_state=random_state,
)
print("[TRAIN] Fitting DecisionTreeClassifier ...")
clf.fit(X, y)
print("[TRAIN] Done.")

# -----------------------------
# Save the trained model
# -----------------------------
out_path = os.path.join(model_path, "decision_tree_scikit.pkl")
with open(out_path, "wb") as f:
    pickle.dump({
        "model": clf,
        "feature_cols": features,
        "embed_dim": X.shape[1],
        "meta": {
            "rows": int(X.shape[0]),
            "bytes_used": int(bytes_used),
            "budget": int(limit_bytes),
        }
    }, f)
print(f"[SAVE] Saved DecisionTreeClassifier to: {out_path}")

# Quick fit summary
train_pred = clf.predict(X)
train_acc = accuracy_score(y, train_pred)
print(f"[METRICS] Train accuracy on loaded subset: {train_acc:.4f}")
print(f"[TREE] Depth={clf.get_depth()}, Leaves={clf.get_n_leaves()}")

# Export a compact textual representation of the tree
try:
    feature_names = [f"emb_{i}" for i in range(X.shape[1])]
    rules_txt = export_text(clf, feature_names=feature_names, max_depth=4)
    rules_path = os.path.join(model_path, "decision_tree_scikit_rules.txt")
    with open(rules_path, "w", encoding="utf-8") as f:
        f.write(rules_txt)
    print(f"[SAVE] Exported tree rules (depth<=4) to: {rules_path}")
except Exception as e:
    print(f"[WARN] Could not export tree rules: {e}")

