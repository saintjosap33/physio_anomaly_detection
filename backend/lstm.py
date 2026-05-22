# ============================================================
# NORMAL-ONLY TRAINING PIPELINE (CORRECT UNSUPERVISED SETUP)
# ============================================================

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
import joblib
import os
import time
import json

# ===============================
# PATH SETUP
# ===============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "saved_models")
os.makedirs(MODEL_DIR, exist_ok=True)

# ===============================
# CONFIG
# ===============================
BATCH_SIZE = 64
EPOCHS_AE = 60
EPOCHS_LSTM = 80
SEQ_LEN = 20
LR = 0.0005

# ===============================
# LOAD DATA (NORMAL ONLY)
# ===============================
X = np.load(os.path.join(BASE_DIR, "features.npy"))
print("Feature Shape:", X.shape)

# ===============================
# CLEAN + SCALE
# ===============================
X = np.nan_to_num(X)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train = X_scaled
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)

# ===============================
# LOGGER (NO ANOMALY ASSUMPTION)
# ===============================
def log_model(name, scores, start_time):
    duration = time.time() - start_time

    mean = float(np.mean(scores))
    std = float(np.std(scores))
    min_v = float(np.min(scores))
    max_v = float(np.max(scores))

    print("\n" + "="*60)
    print(f"📊 MODEL: {name}")
    print("="*60)
    print(f"⏱ Time: {duration:.2f}s")
    print(f"Mean Score: {mean:.6f}")
    print(f"Std Dev   : {std:.6f}")
    print(f"Min       : {min_v:.6f}")
    print(f"Max       : {max_v:.6f}")

    return {
        "Model": name,
        "Mean": mean,
        "Std": std,
        "Min": min_v,
        "Max": max_v,
        "Time": float(duration)
    }
logs = []

# ============================================================
# 1. ISOLATION FOREST (learn normal density)
# ============================================================
start = time.time()

iso = IsolationForest(contamination="auto", random_state=42)
iso.fit(X_train)

iso_scores = -iso.decision_function(X_train)
logs.append(log_model("Isolation Forest", iso_scores, start))

# ============================================================
# 2. ONE-CLASS SVM
# ============================================================
start = time.time()

svm = OneClassSVM(kernel='rbf', nu=0.05)
svm.fit(X_train)

svm_scores = -svm.decision_function(X_train)
logs.append(log_model("One-Class SVM", svm_scores, start))

# ============================================================
# 3. LOF
# ============================================================
start = time.time()

lof = LocalOutlierFactor(n_neighbors=20, novelty=True)
lof.fit(X_train)

lof_scores = -lof.decision_function(X_train)
logs.append(log_model("LOF", lof_scores, start))

# ============================================================
# 4. PCA (reconstruction learning)
# ============================================================
start = time.time()

pca = PCA(n_components=0.90)
X_pca = pca.fit_transform(X_train)
X_recon = pca.inverse_transform(X_pca)

pca_scores = np.mean((X_train - X_recon)**2, axis=1)

print("\n📌 PCA Explained Variance:", np.sum(pca.explained_variance_ratio_))
logs.append(log_model("PCA", pca_scores, start))

# ============================================================
# 5. AUTOENCODER
# ============================================================
class AE(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 16)
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, dim)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

print("\nTraining Autoencoder...")
start = time.time()

ae = AE(X.shape[1])
optimizer = torch.optim.Adam(ae.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

loader = DataLoader(TensorDataset(X_train_tensor, X_train_tensor),
                    batch_size=BATCH_SIZE, shuffle=True)

ae_losses = []

for epoch in range(EPOCHS_AE):
    total_loss = 0
    for x, _ in loader:
        optimizer.zero_grad()
        out = ae(x)
        loss = loss_fn(out, x)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    ae_losses.append(avg_loss)

    print(f"AE Epoch {epoch+1}: {avg_loss:.6f}")

with torch.no_grad():
    recon = ae(X_train_tensor)
    ae_scores = torch.mean((X_train_tensor - recon)**2, dim=1).numpy()

logs.append(log_model("Autoencoder", ae_scores, start))

# ============================================================
# 6. BiLSTM (temporal learning)
# ============================================================
def create_sequences(data, seq_len):
    return np.array([data[i:i+seq_len] for i in range(len(data)-seq_len)])

X_seq = create_sequences(X_train, SEQ_LEN)

class LSTM_AE(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.encoder = nn.LSTM(dim, 64, batch_first=True)
        self.decoder = nn.LSTM(64, dim, batch_first=True)

    def forward(self, x):
        _, (h, _) = self.encoder(x)
        h = h.repeat(x.size(1), 1, 1).permute(1, 0, 2)
        out, _ = self.decoder(h)
        return out

print("\nTraining BiLSTM...")
start = time.time()

lstm = LSTM_AE(X.shape[1])
optimizer = torch.optim.Adam(lstm.parameters(), lr=LR)

loader = DataLoader(torch.tensor(X_seq, dtype=torch.float32), batch_size=32)

lstm_losses = []

for epoch in range(EPOCHS_LSTM):
    total_loss = 0
    for x in loader:
        optimizer.zero_grad()
        out = lstm(x)
        loss = loss_fn(out, x)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    lstm_losses.append(avg_loss)

    print(f"LSTM Epoch {epoch+1}: {avg_loss:.6f}")

with torch.no_grad():
    recon = lstm(torch.tensor(X_seq, dtype=torch.float32))
    lstm_scores = torch.mean((torch.tensor(X_seq, dtype=torch.float32) - recon)**2, dim=(1,2)).numpy()

logs.append(log_model("BiLSTM", lstm_scores, start))

# ============================================================
# SAVE MODELS
# ============================================================
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
joblib.dump(pca, os.path.join(MODEL_DIR, "pca.pkl"))
joblib.dump(iso, os.path.join(MODEL_DIR, "iso.pkl"))
joblib.dump(svm, os.path.join(MODEL_DIR, "svm.pkl"))
joblib.dump(lof, os.path.join(MODEL_DIR, "lof.pkl"))

torch.save(ae.state_dict(), os.path.join(MODEL_DIR, "ae.pth"))
torch.save(lstm.state_dict(), os.path.join(MODEL_DIR, "lstm.pth"))

# ============================================================
# SAVE NORMAL DISTRIBUTION STATS (CRITICAL)
# ============================================================
np.save(os.path.join(BASE_DIR, "error_stats.npy"), {
    "pca_mean": float(np.mean(pca_scores)),
    "pca_std": float(np.std(pca_scores)),
    "ae_mean": float(np.mean(ae_scores)),
    "ae_std": float(np.std(ae_scores)),
    "lstm_mean": float(np.mean(lstm_scores)),
    "lstm_std": float(np.std(lstm_scores))
})

# ============================================================
# SAVE LOGS
# ============================================================
df = pd.DataFrame(logs)
df.to_csv(os.path.join(BASE_DIR, "model_comparison.csv"), index=False)

with open(os.path.join(BASE_DIR, "training_logs.json"), "w") as f:
    json.dump(logs, f, indent=4)

# ============================================================
# PLOTS
# ============================================================
plt.figure()
plt.plot(ae_losses, label="AE Loss")
plt.plot(lstm_losses, label="LSTM Loss")
plt.legend()
plt.title("Training Loss")
plt.show()

print("\n✅ NORMAL PATTERN TRAINING COMPLETE")