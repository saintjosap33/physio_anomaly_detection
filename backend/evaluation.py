# ============================================================
# LOAD PRETRAINED MODELS + EVALUATE ON SYNTHETIC ANOMALIES
# ============================================================

import numpy as np
import torch
import torch.nn as nn

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)

import pandas as pd
import joblib
import os

# ============================================================
# PATH SETUP
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.path.join(BASE_DIR, "saved_models")

# ============================================================
# CONFIG
# ============================================================

SEQ_LEN = 20

# ============================================================
# LOAD FEATURES
# ============================================================

X = np.load(os.path.join(BASE_DIR, "features_upgraded.npy"))

X = np.nan_to_num(X)

print("Feature Shape:", X.shape)

# ============================================================
# LOAD SAVED MODELS
# ============================================================

print("\nLoading saved models...")

scaler = joblib.load(
    os.path.join(MODEL_DIR, "scaler.pkl")
)

pca = joblib.load(
    os.path.join(MODEL_DIR, "pca.pkl")
)

iso = joblib.load(
    os.path.join(MODEL_DIR, "iso.pkl")
)

svm = joblib.load(
    os.path.join(MODEL_DIR, "svm.pkl")
)

lof = joblib.load(
    os.path.join(MODEL_DIR, "lof.pkl")
)

print("✅ Sklearn models loaded")

# ============================================================
# SCALE DATA
# ============================================================

X_scaled = scaler.transform(X)

# ============================================================
# AUTOENCODER MODEL
# ============================================================

class AE(nn.Module):

    def __init__(self, dim):

        super().__init__()

        self.encoder = nn.Sequential(

            nn.Linear(dim, 128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, 16)
        )

        self.decoder = nn.Sequential(

            nn.Linear(16, 64),
            nn.ReLU(),

            nn.Linear(64, 128),
            nn.ReLU(),

            nn.Linear(128, dim)
        )

    def forward(self, x):

        encoded = self.encoder(x)

        decoded = self.decoder(encoded)

        return decoded

# ============================================================
# LOAD AUTOENCODER
# ============================================================

ae = AE(X.shape[1])

ae.load_state_dict(
    torch.load(
        os.path.join(MODEL_DIR, "ae.pth")
    )
)

ae.eval()

print("✅ Autoencoder loaded")

# ============================================================
# LSTM AUTOENCODER MODEL
# ============================================================

class LSTM_AE(nn.Module):

    def __init__(self, dim):

        super().__init__()

        self.encoder = nn.LSTM(
            dim,
            64,
            batch_first=True
        )

        self.decoder = nn.LSTM(
            64,
            dim,
            batch_first=True
        )

    def forward(self, x):

        _, (h, _) = self.encoder(x)

        h = h.repeat(
            x.size(1),
            1,
            1
        ).permute(1,0,2)

        out, _ = self.decoder(h)

        return out

# ============================================================
# LOAD LSTM MODEL
# ============================================================

lstm = LSTM_AE(X.shape[1])

lstm.load_state_dict(
    torch.load(
        os.path.join(MODEL_DIR, "lstm.pth")
    )
)

lstm.eval()

print("✅ BiLSTM loaded")

# ============================================================
# CREATE SEQUENCES
# ============================================================

def create_sequences(data, seq_len):

    return np.array([
        data[i:i+seq_len]
        for i in range(len(data)-seq_len)
    ])

# ============================================================
# SYNTHETIC ANOMALY GENERATION
# ============================================================

def generate_synthetic_anomalies(
    X_normal,
    anomaly_percent=10
):

    X_normal = X_normal.copy()

    total_samples = len(X_normal)

    n_anomaly = int(
        (anomaly_percent / 100) * total_samples
    )

    anomaly_indices = np.random.choice(
        total_samples,
        n_anomaly,
        replace=False
    )

    X_anomaly = X_normal[anomaly_indices].copy()

    for i in range(len(X_anomaly)):

        anomaly_type = np.random.choice([
            "spike",
            "drift",
            "noise",
            "drop",
            "flatline"
        ])

        n_features = X_anomaly.shape[1]

        # ====================================================
        # SPIKE
        # ====================================================

        if anomaly_type == "spike":

            idx = np.random.randint(0, n_features)

            X_anomaly[i, idx] += np.random.uniform(5, 10)

        # ====================================================
        # DRIFT
        # ====================================================

        elif anomaly_type == "drift":

            X_anomaly[i] += np.random.uniform(
                2,
                5,
                size=n_features
            )

        # ====================================================
        # NOISE
        # ====================================================

        elif anomaly_type == "noise":

            X_anomaly[i] += np.random.normal(
                0,
                3,
                size=n_features
            )

        # ====================================================
        # DROP
        # ====================================================

        elif anomaly_type == "drop":

            idx = np.random.randint(0, n_features)

            X_anomaly[i, idx] -= np.random.uniform(5, 10)

        # ====================================================
        # FLATLINE
        # ====================================================

        elif anomaly_type == "flatline":

            value = np.mean(X_anomaly[i])

            X_anomaly[i] = value

    # ========================================================
    # COMBINE
    # ========================================================

    X_test = np.vstack([
        X_normal,
        X_anomaly
    ])

    y_normal = np.zeros(len(X_normal))

    y_anomaly = np.ones(len(X_anomaly))

    y_test = np.hstack([
        y_normal,
        y_anomaly
    ])

    idx = np.random.permutation(len(X_test))

    X_test = X_test[idx]

    y_test = y_test[idx]

    return X_test, y_test

# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate_model(name, y_true, scores):

    threshold = (
        np.mean(scores[y_true == 0])
        +
        3 * np.std(scores[y_true == 0])
    )

    y_pred = (scores > threshold).astype(int)

    accuracy = accuracy_score(y_true, y_pred)

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_true,
        scores
    )

    pr_auc = average_precision_score(
        y_true,
        scores
    )

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    print("\n" + "="*70)

    print(f"📊 {name}")

    print("="*70)

    print(f"Accuracy : {accuracy:.4f}")

    print(f"Precision: {precision:.4f}")

    print(f"Recall   : {recall:.4f}")

    print(f"F1 Score : {f1:.4f}")

    print(f"ROC-AUC  : {roc_auc:.4f}")

    print(f"PR-AUC   : {pr_auc:.4f}")

    print("\nConfusion Matrix:")

    print(cm)

    return {
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "ROC_AUC": roc_auc,
        "PR_AUC": pr_auc
    }

# ============================================================
# TESTING
# ============================================================

TEST_LEVELS = [10, 15, 20]

all_results = []

for percent in TEST_LEVELS:

    print("\n" + "#"*80)

    print(f"TESTING WITH {percent}% SYNTHETIC ANOMALIES")

    print("#"*80)

    # ========================================================
    # CREATE TEST DATA
    # ========================================================

    X_test, y_test = generate_synthetic_anomalies(
        X_scaled,
        anomaly_percent=percent
    )

    X_test_tensor = torch.tensor(
        X_test,
        dtype=torch.float32
    )

    # ========================================================
    # ISOLATION FOREST
    # ========================================================

    iso_scores = -iso.decision_function(X_test)

    all_results.append(
        evaluate_model(
            f"Isolation Forest ({percent}%)",
            y_test,
            iso_scores
        )
    )

    # ========================================================
    # OCSVM
    # ========================================================

    svm_scores = -svm.decision_function(X_test)

    all_results.append(
        evaluate_model(
            f"One-Class SVM ({percent}%)",
            y_test,
            svm_scores
        )
    )

    # ========================================================
    # LOF
    # ========================================================

    lof_scores = -lof.decision_function(X_test)

    all_results.append(
        evaluate_model(
            f"LOF ({percent}%)",
            y_test,
            lof_scores
        )
    )

    # ========================================================
    # PCA
    # ========================================================

    X_test_pca = pca.transform(X_test)

    X_test_recon = pca.inverse_transform(
        X_test_pca
    )

    pca_scores = np.mean(
        (X_test - X_test_recon)**2,
        axis=1
    )

    all_results.append(
        evaluate_model(
            f"PCA ({percent}%)",
            y_test,
            pca_scores
        )
    )

    # ========================================================
    # AUTOENCODER
    # ========================================================

    with torch.no_grad():

        recon = ae(X_test_tensor)

        ae_scores = torch.mean(
            (X_test_tensor - recon)**2,
            dim=1
        ).numpy()

    all_results.append(
        evaluate_model(
            f"Autoencoder ({percent}%)",
            y_test,
            ae_scores
        )
    )

    # ========================================================
    # BiLSTM
    # ========================================================

    X_test_seq = create_sequences(
        X_test,
        SEQ_LEN
    )

    y_test_seq = y_test[SEQ_LEN:]

    X_test_seq_tensor = torch.tensor(
        X_test_seq,
        dtype=torch.float32
    )

    with torch.no_grad():

        recon = lstm(X_test_seq_tensor)

        lstm_scores = torch.mean(
            (X_test_seq_tensor - recon)**2,
            dim=(1,2)
        ).numpy()

    all_results.append(
        evaluate_model(
            f"BiLSTM ({percent}%)",
            y_test_seq,
            lstm_scores
        )
    )

# ============================================================
# SAVE FINAL RESULTS
# ============================================================

results_df = pd.DataFrame(all_results)

results_df.to_csv(
    os.path.join(BASE_DIR, "research_metrics.csv"),
    index=False
)

print("\n✅ RESEARCH EVALUATION COMPLETE")

print("\nFINAL RESULTS:\n")

print(results_df)