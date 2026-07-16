# ============================================================
# PRATA-NET V6
# Advanced Physiological Anomaly Detection
#
# V6 FIXES OVER V5:
#   ✅ Adversarial anomaly training
#       - Anomalies generated DURING training
#       - Model explicitly penalized for reconstructing them
#       - Directly targets bimodal anomaly score problem
#       - Subtle anomalies (slow drift, autonomic imbalance)
#         forced to score high, not just easy ones
#   ✅ Anomaly reconstruction penalty loss
#       - Normal: minimize reconstruction error (same as before)
#       - Anomaly: MAXIMIZE reconstruction error
#       - Margin-based: only penalize if anomaly score < margin
#   ✅ Latent separation loss
#       - Normal latent vectors: pulled toward latent center
#       - Anomaly latent vectors: pushed away from center
#       - Directly shapes latent space geometry
#   ✅ All V5 features retained
#
# TARGET:
#   ROC-AUC > 0.90
#   PR-AUC  > 0.90
#
# ============================================================

import os
import numpy as np
import pandas as pd
import joblib

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)

# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.path.join(BASE_DIR, "saved_models")

os.makedirs(MODEL_DIR, exist_ok=True)

# ============================================================
# LOAD FEATURES
# ============================================================

X = np.load(
    os.path.join(BASE_DIR, "features_upgraded.npy")
)

subjects = np.load(
    os.path.join(BASE_DIR, "subjects.npy")
)

X = np.nan_to_num(X).astype(np.float32)

print("=" * 80)
print("RAW DATA")
print("=" * 80)

print("Features Shape :", X.shape)

INPUT_DIM = X.shape[1]

# ============================================================
# NORMALIZATION
# ============================================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

joblib.dump(
    scaler,
    os.path.join(MODEL_DIR, "prata_scaler_v6.pkl")
)

# ============================================================
# TEMPORAL WINDOWING
# ============================================================

WINDOW_SIZE = 32
STRIDE = 4

windows = []

for start in range(0, len(X_scaled) - WINDOW_SIZE, STRIDE):

    end = start + WINDOW_SIZE

    windows.append(X_scaled[start:end])

windows = np.array(windows, dtype=np.float32)

print()
print("=" * 80)
print("WINDOWED DATA")
print("=" * 80)

print("Windows Shape :", windows.shape)

# ============================================================
# PURE NORMAL LEARNING
# ============================================================

X_normal = windows.copy()

split = int(0.8 * len(X_normal))

X_train = X_normal[:split]

X_test_normal = X_normal[split:]

print()
print("=" * 80)
print("PURE NORMAL PHYSIOLOGY")
print("=" * 80)

print("Train Windows :", len(X_train))
print("Test Windows  :", len(X_test_normal))

# ============================================================
# V4: REALISTIC PHYSIOLOGICAL ANOMALY GENERATION
#
# Key insight: real physiological anomalies are NOT random
# spikes or noise. They are:
#
#   1. Gradual autonomic imbalance
#      -> slow drift in signal variability
#      -> not a sudden jump
#
#   2. Cross-signal desynchronization
#      -> normally BVP, EDA, TEMP move together
#      -> anomaly: they decouple
#      -> modeled as lagged or phase-shifted coupling
#
#   3. HRV rhythm instability
#      -> natural HRV has rhythmic LF/HF oscillations
#      -> anomaly: those rhythms collapse or become chaotic
#
#   4. Slow autonomic drift
#      -> very gradual mean shift over entire window
#      -> harder to detect than spikes
#
# These are combined with V3 anomalies but weighted toward
# the harder realistic types.
# ============================================================

def compute_signal_stats(X):
    """
    Precompute per-feature statistics over the training set.
    Used to make realistic anomaly magnitudes data-relative.
    """
    flat = X.reshape(-1, X.shape[-1])
    return {
        "mean": flat.mean(axis=0),
        "std": flat.std(axis=0) + 1e-6,
        "max": flat.max(axis=0),
        "min": flat.min(axis=0),
    }


def generate_physiological_anomalies(
        X,
        signal_stats=None,
        anomaly_strength=3.5,
        hard_ratio=0.6):
    """
    Generate realistic physiological anomalies.

    hard_ratio: fraction of samples using harder realistic
                anomaly types (autonomic imbalance, desync,
                hrv instability). Remaining use classic types.
    """

    X = X.copy()

    N, T, F = X.shape

    for i in range(N):

        # --------------------------------------------------
        # Choose anomaly type
        # Bias toward harder realistic types (hard_ratio)
        # --------------------------------------------------

        if np.random.rand() < hard_ratio:

            anomaly_type = np.random.choice([

                "autonomic_imbalance",
                "cross_signal_desync",
                "hrv_instability",
                "slow_drift",
                "lagged_coupling_break",

            ])

        else:

            anomaly_type = np.random.choice([

                "spike",
                "drift",
                "noise",
                "flatline",
                "frequency_distortion",
                "signal_dropout",
                "inverse_pattern",

            ])

        # ==================================================
        # REALISTIC ANOMALY TYPES
        # ==================================================

        # --------------------------------------------------
        # 1. AUTONOMIC IMBALANCE (HARDER)
        #    -> V4 was too subtle, autoencoder reconstructed it
        #    -> V5: affects 40-70% of features (wider impact)
        #    -> V5: variability multiplier increased to 4-8x
        #    -> V5: also shifts the mean, not just variance
        # --------------------------------------------------

        if anomaly_type == "autonomic_imbalance":

            # Wider feature impact: 40-70% of features
            n_affected = max(1, int(F * np.random.uniform(0.4, 0.7)))
            features = np.random.choice(F, n_affected, replace=False)

            # Stronger variability curve: 4-8x instead of 2.5-5x
            variability_curve = np.linspace(
                1.0,
                np.random.uniform(4.0, 8.0),
                T
            )

            # Also add a mean shift to make it harder to reconstruct
            mean_shift = np.random.uniform(2.0, 4.0) * np.random.choice([-1, 1])
            shift_curve = np.linspace(0, mean_shift, T)

            for f in features:
                noise = np.random.randn(T)
                X[i, :, f] += noise * variability_curve + shift_curve

        # --------------------------------------------------
        # 2. CROSS-SIGNAL DESYNCHRONIZATION (HARDER)
        #    -> V4: only 2 features affected
        #    -> V5: affects 3-6 feature pairs simultaneously
        #    -> V5: adds correlated structured noise on top
        # --------------------------------------------------

        elif anomaly_type == "cross_signal_desync":

            if F >= 4:

                # Multiple pairs broken simultaneously
                n_pairs = np.random.randint(2, min(4, F // 2 + 1))

                for _ in range(n_pairs):

                    f1 = np.random.randint(0, F)
                    f2 = np.random.randint(0, F)
                    while f2 == f1:
                        f2 = np.random.randint(0, F)

                    orig_f1 = X[i, :, f1].copy()
                    orig_f2 = X[i, :, f2].copy()

                    lag = np.random.randint(3, 10)

                    X[i, lag:, f1] = -orig_f2[:T - lag]
                    X[i, lag:, f2] = -orig_f1[:T - lag]

                    std = signal_stats["std"][f1] if signal_stats else 1.0
                    X[i, :lag, f1] = np.random.randn(lag) * std * 2.0

                # Add structured correlated noise across all features
                # to break global inter-feature relationships
                shared_noise = np.random.randn(T) * anomaly_strength * 0.5
                n_noise_feats = max(1, int(F * 0.3))
                noise_feats = np.random.choice(F, n_noise_feats, replace=False)
                for f in noise_feats:
                    X[i, :, f] += shared_noise * np.random.choice([-1, 1])

        # --------------------------------------------------
        # 3. HRV RHYTHM INSTABILITY (HARDER)
        #    -> V4: 1-3 features, moderate strength
        #    -> V5: 3-8 features, stronger instability
        #    -> V5: combined chaos + resonance for worst case
        # --------------------------------------------------

        elif anomaly_type == "hrv_instability":

            # More features affected
            n_affected = np.random.randint(3, min(9, F + 1))
            features = np.random.choice(F, n_affected, replace=False)

            instability_type = np.random.choice([
                "chaos",
                "collapse",
                "resonance",
                "combined",    # V5 NEW: chaos + resonance together
            ])

            t_axis = np.linspace(0, 1, T)

            for f in features:

                if instability_type == "chaos":
                    X[i, :, f] += np.random.randn(T) * anomaly_strength * 2.5

                elif instability_type == "collapse":
                    mean_val = np.mean(X[i, :, f])
                    collapse_weight = np.linspace(0, 1, T)
                    X[i, :, f] = (
                        X[i, :, f] * (1 - collapse_weight) +
                        mean_val * collapse_weight
                    )
                    # Add jitter after collapse to make it weirder
                    X[i, T//2:, f] += np.random.randn(T - T//2) * 0.5

                elif instability_type == "resonance":
                    wrong_freq = np.random.uniform(4.0, 10.0)
                    X[i, :, f] += (
                        np.sin(2 * np.pi * wrong_freq * t_axis) *
                        anomaly_strength * 2.0
                    )

                elif instability_type == "combined":
                    # Chaos AND resonance simultaneously
                    wrong_freq = np.random.uniform(4.0, 10.0)
                    X[i, :, f] += (
                        np.sin(2 * np.pi * wrong_freq * t_axis) *
                        anomaly_strength * 1.5 +
                        np.random.randn(T) * anomaly_strength
                    )

        # --------------------------------------------------
        # 4. SLOW DRIFT (HARDER)
        #    -> V4: magnitude 0.8-2x anomaly_strength
        #    -> V5: magnitude 2-5x anomaly_strength
        #    -> V5: nonlinear drift (quadratic) harder to catch
        #    -> V5: affects 50-80% of features
        # --------------------------------------------------

        elif anomaly_type == "slow_drift":

            n_affected = max(1, int(F * np.random.uniform(0.5, 0.8)))
            features = np.random.choice(F, n_affected, replace=False)

            drift_magnitude = np.random.uniform(
                anomaly_strength * 2.0,
                anomaly_strength * 5.0
            )

            # Nonlinear (quadratic) drift — harder to reconstruct
            t_norm = np.linspace(0, 1, T)
            drift_curve = (t_norm ** 2) * drift_magnitude

            direction = np.random.choice([-1, 1])

            for f in features:
                X[i, :, f] += drift_curve * direction

        # --------------------------------------------------
        # 5. LAGGED COUPLING BREAK (HARDER)
        #    -> V4: reverse segments
        #    -> V5: reverse + add cross-feature leakage
        #    -> V5: more features, longer scramble segments
        # --------------------------------------------------

        elif anomaly_type == "lagged_coupling_break":

            n_affected = np.random.randint(2, min(5, F + 1))
            features = np.random.choice(F, n_affected, replace=False)

            for f in features:

                seg_len = np.random.randint(6, 12)
                for start in range(0, T - seg_len, seg_len):
                    end = start + seg_len
                    X[i, start:end, f] = X[i, start:end, f][::-1]

            # V5 NEW: cross-feature leakage after scramble
            # makes the inter-feature structure incoherent
            if len(features) >= 2:
                leak_strength = np.random.uniform(0.5, 1.5)
                X[i, :, features[0]] += X[i, :, features[1]] * leak_strength * -1

        # ==================================================
        # CLASSIC ANOMALY TYPES (V3, retained for coverage)
        # ==================================================

        elif anomaly_type == "spike":

            for _ in range(np.random.randint(1, 5)):
                t = np.random.randint(0, T)
                f = np.random.randint(0, F)
                X[i, t, f] += np.random.uniform(6, 12)

        elif anomaly_type == "drift":

            drift = np.linspace(0, anomaly_strength, T)
            X[i] += drift[:, None]

        elif anomaly_type == "noise":

            noise = np.random.normal(0, anomaly_strength, X[i].shape)
            X[i] += noise

        elif anomaly_type == "flatline":

            start = np.random.randint(5, 15)
            end = start + np.random.randint(5, 10)
            X[i, start:end] = np.mean(X[i])

        elif anomaly_type == "frequency_distortion":

            wave = np.sin(np.linspace(0, 20, T))
            X[i] += wave[:, None] * 4

        elif anomaly_type == "signal_dropout":

            f = np.random.randint(0, F)
            X[i, :, f] = 0

        elif anomaly_type == "inverse_pattern":

            X[i] = -X[i]

    return X


# ============================================================
# TEST SET
# ============================================================

signal_stats = compute_signal_stats(X_train)

X_fake_anomaly = generate_physiological_anomalies(
    X_test_normal.copy(),
    signal_stats=signal_stats
)

X_test = np.concatenate([
    X_test_normal,
    X_fake_anomaly
])

y_test = np.concatenate([
    np.zeros(len(X_test_normal)),
    np.ones(len(X_fake_anomaly))
])

idx = np.random.permutation(len(X_test))

X_test = X_test[idx]
y_test = y_test[idx]

print()
print("=" * 80)
print("TEST SET")
print("=" * 80)

print("Test Shape :", X_test.shape)

# ============================================================
# TENSORS
# ============================================================

train_tensor = torch.tensor(X_train, dtype=torch.float32)
test_tensor  = torch.tensor(X_test,  dtype=torch.float32)
test_labels  = torch.tensor(y_test,  dtype=torch.float32)

train_loader = DataLoader(
    TensorDataset(train_tensor),
    batch_size=128,
    shuffle=True
)

test_loader = DataLoader(
    TensorDataset(test_tensor, test_labels),
    batch_size=128,
    shuffle=False
)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"\nUsing Device : {device}")

# ============================================================
# FEATURE GATE
# (unchanged from V3 - proven effective)
# ============================================================

class FeatureGate(nn.Module):

    def __init__(self, dim):

        super().__init__()

        self.gate = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )

    def forward(self, x):

        weights = self.gate(x)
        return x * weights

# ============================================================
# MULTI SCALE CNN
# (unchanged from V3 - proven effective)
# ============================================================

class MultiScaleCNN(nn.Module):

    def __init__(self, input_dim, hidden):

        super().__init__()

        self.conv3 = nn.Conv1d(input_dim, hidden, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(input_dim, hidden, kernel_size=5, padding=2)
        self.conv7 = nn.Conv1d(input_dim, hidden, kernel_size=7, padding=3)

        self.bn = nn.BatchNorm1d(hidden * 3)

        self.residual = nn.Conv1d(input_dim, hidden * 3, kernel_size=1)

    def forward(self, x):

        x = x.permute(0, 2, 1)

        c3 = F.gelu(self.conv3(x))
        c5 = F.gelu(self.conv5(x))
        c7 = F.gelu(self.conv7(x))

        out = torch.cat([c3, c5, c7], dim=1)
        out = self.bn(out)

        res = self.residual(x)
        out = out + res

        out = out.permute(0, 2, 1)

        return out

# ============================================================
# TRANSFORMER BLOCK
# (unchanged from V3)
# ============================================================

class TransformerBlock(nn.Module):

    def __init__(self, dim, heads=4):

        super().__init__()

        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x):

        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)

        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)

        return x

# ============================================================
# V4 NEW: TEMPORAL TRANSFORMER ENCODER
#
# Replaces second BiLSTM layer.
# Motivation:
#   - BiLSTM is good at local temporal patterns
#   - Transformer is better at LONG-RANGE dependencies
#   - Physiological signals have long-range rhythms
#     (e.g. autonomic oscillations over 20-30s windows)
#   - Using Transformer AFTER BiLSTM gives best of both:
#     local (BiLSTM) + global (Transformer)
# ============================================================

class TemporalTransformerEncoder(nn.Module):
    """
    Stack of Transformer blocks applied along the time axis.
    Positional encoding added to preserve temporal order.
    """

    def __init__(self, dim, n_layers=3, heads=8):

        super().__init__()

        self.layers = nn.ModuleList([
            TransformerBlock(dim, heads)
            for _ in range(n_layers)
        ])

        # Learnable positional encoding
        self.pos_enc = nn.Parameter(
            torch.randn(1, WINDOW_SIZE, dim) * 0.02
        )

    def forward(self, x):

        # Add positional encoding
        x = x + self.pos_enc[:, :x.shape[1], :]

        for layer in self.layers:
            x = layer(x)

        return x
# (unchanged from V3)
# ============================================================

class FrequencyEncoder(nn.Module):

    def forward(self, x):

        fft = torch.fft.rfft(x, dim=1)
        magnitude = torch.abs(fft)
        return magnitude

# ============================================================
# V4 NEW: PROJECTION HEAD FOR CONTRASTIVE LEARNING
#
# Maps latent representation to a lower-dim space
# where contrastive loss is applied.
# (Standard practice from SimCLR / MoCo)
# ============================================================

class ProjectionHead(nn.Module):

    def __init__(self, latent_dim=64, proj_dim=32):

        super().__init__()

        self.proj = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, proj_dim),
            nn.LayerNorm(proj_dim)
        )

    def forward(self, x):

        # x: (batch, time, latent_dim)
        # Pool over time -> (batch, latent_dim)
        pooled = x.mean(dim=1)
        return self.proj(pooled)

# ============================================================
# PRATA-NET V4
# ============================================================

class PRATANetV4(nn.Module):

    def __init__(
            self,
            input_dim=189,
            hidden=128,
            latent_dim=64,
            proj_dim=32):

        super().__init__()

        self.feature_gate = FeatureGate(input_dim)

        self.multi_scale = MultiScaleCNN(input_dim, hidden)

        fused = hidden * 3  # 384

        # --- ENCODER ---

        # Two Transformer blocks (same as V3)
        self.transformer1 = TransformerBlock(fused)
        self.transformer2 = TransformerBlock(fused)

        # BiLSTM: local temporal patterns
        self.bi_lstm = nn.LSTM(
            fused,
            128,
            num_layers=1,           # reduced to 1 layer
            batch_first=True,
            bidirectional=True,
            dropout=0.0             # no dropout with 1 layer
        )

        # V4 NEW: Temporal Transformer after BiLSTM
        # for long-range physiological dynamics
        self.temporal_transformer = TemporalTransformerEncoder(
            dim=256,                # BiLSTM output: 128*2
            n_layers=3,
            heads=8
        )

        self.frequency_encoder = FrequencyEncoder()

        self.latent = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, latent_dim)
        )

        # V4 NEW: Projection head for contrastive learning
        self.proj_head = ProjectionHead(latent_dim, proj_dim)

        # --- DECODER ---

        self.decoder = nn.LSTM(
            latent_dim,
            128,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )

        self.output = nn.Sequential(
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, input_dim)
        )

    def forward(self, x):

        # Light noise augmentation (same as V3)
        noise = torch.randn_like(x) * 0.005
        x = x + noise

        # Feature gating
        x = self.feature_gate(x)

        # Multi-scale CNN
        x = self.multi_scale(x)

        # Transformer blocks
        x = self.transformer1(x)
        x = self.transformer2(x)

        # BiLSTM: local temporal
        x, _ = self.bi_lstm(x)  # (B, T, 256)

        # V4: Temporal Transformer: global temporal
        x = self.temporal_transformer(x)  # (B, T, 256)

        # Latent projection
        latent = self.latent(x)  # (B, T, latent_dim)

        # V4: Projection for contrastive loss
        proj = self.proj_head(latent)  # (B, proj_dim)

        # Decoder
        decoded, _ = self.decoder(latent)

        # Reconstruction
        reconstruction = self.output(decoded)

        return reconstruction, latent, proj

# ============================================================
# V4 NEW: NT-XENT CONTRASTIVE LOSS
#
# Motivation:
#   In latent space we want:
#   - Normal windows from same recording CLOSE together
#   - Windows with different dynamics FAR apart
#
# Implementation:
#   We treat two augmented views of the same window
#   as a positive pair (SimCLR style).
#   All other windows in the batch are negatives.
#
#   At test time: normal windows cluster tightly,
#   anomalous windows are far from the normal cluster.
#   This directly improves ROC-AUC.
# ============================================================

class NTXentLoss(nn.Module):

    def __init__(self, temperature=0.07):

        super().__init__()

        self.temperature = temperature

    def forward(self, z1, z2):
        """
        z1, z2: (batch, proj_dim) - two augmented views
        Both come from normal samples only.
        """

        B = z1.shape[0]

        # Normalize
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)

        # Concatenate: (2B, proj_dim)
        z = torch.cat([z1, z2], dim=0)

        # Similarity matrix: (2B, 2B)
        sim = torch.matmul(z, z.T) / self.temperature

        # Mask out self-similarity
        mask = torch.eye(2 * B, device=z.device).bool()
        sim.masked_fill_(mask, float('-inf'))

        # Positive pairs: (i, i+B) and (i+B, i)
        labels = torch.cat([
            torch.arange(B, 2 * B),
            torch.arange(0, B)
        ]).to(z.device)

        loss = F.cross_entropy(sim, labels)

        return loss

# ============================================================
# ADVANCED LOSS V4
# ============================================================

class AdvancedPhysioLossV4(nn.Module):

    def __init__(
            self,
            alpha=1.0,
            beta=0.15,
            gamma=0.10,
            delta=0.05,
            contrastive_weight=0.20):

        super().__init__()

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.contrastive_weight = contrastive_weight

        self.contrastive = NTXentLoss(temperature=0.07)

    def forward(
            self,
            original,
            reconstruction,
            latent,
            proj=None,
            proj_aug=None):

        # ====================================================
        # RECONSTRUCTION LOSS
        # ====================================================

        recon = F.mse_loss(reconstruction, original)

        # ====================================================
        # LATENT TEMPORAL SMOOTHNESS
        # ====================================================

        latent_loss = torch.mean(
            torch.abs(latent[:, 1:] - latent[:, :-1])
        )

        # ====================================================
        # TEMPORAL CONSISTENCY
        # ====================================================

        temporal = torch.mean(
            (reconstruction[:, 1:] - reconstruction[:, :-1]) ** 2
        )

        # ====================================================
        # FREQUENCY LOSS
        # ====================================================

        fft_orig  = torch.fft.rfft(original, dim=1)
        fft_recon = torch.fft.rfft(reconstruction, dim=1)

        freq_loss = F.mse_loss(
            torch.abs(fft_orig),
            torch.abs(fft_recon)
        )

        # ====================================================
        # V4: CONTRASTIVE LOSS
        # Only applied when augmented projections are provided
        # ====================================================

        contrastive_loss = torch.tensor(0.0, device=original.device)

        if proj is not None and proj_aug is not None:
            contrastive_loss = self.contrastive(proj, proj_aug)

        total = (
            self.alpha * recon +
            self.beta  * latent_loss +
            self.gamma * temporal +
            self.delta * freq_loss +
            self.contrastive_weight * contrastive_loss
        )

        return total

# ============================================================
# V6 NEW: ADVERSARIAL ANOMALY LOSS
#
# Core idea: during training, generate anomalies from the
# current batch and explicitly teach the model two things:
#
#   1. RECONSTRUCTION PENALTY
#      Normal: minimize reconstruction error (standard)
#      Anomaly: error should be ABOVE margin * normal error
#      Hinge loss — only fires when anomaly reconstructs too well
#
#   2. LATENT SEPARATION
#      Normal latent: pulled toward latent center
#      Anomaly latent: pushed away from center
#      Directly shapes geometry so Mahalanobis scoring
#      works on subtle anomalies, not just easy ones
#
# Why this fixes the bimodal problem:
#   Bottom 25% of anomalies score near-normal because the
#   autoencoder generalizes to reconstruct them. This loss
#   explicitly prevents that generalization during training.
# ============================================================

class AdversarialAnomalyLoss(nn.Module):

    def __init__(
            self,
            recon_margin=2.0,
            latent_sep_weight=0.3,
            recon_penalty_weight=0.5):

        super().__init__()

        self.recon_margin         = recon_margin
        self.latent_sep_weight    = latent_sep_weight
        self.recon_penalty_weight = recon_penalty_weight

    def forward(
            self,
            x_normal,
            recon_normal,
            latent_normal,
            x_anomaly,
            recon_anomaly,
            latent_anomaly,
            latent_center):

        # --------------------------------------------------
        # 1. RECONSTRUCTION PENALTY
        # --------------------------------------------------

        normal_recon_err = torch.mean(
            (x_normal - recon_normal) ** 2,
            dim=(1, 2)
        )

        anomaly_recon_err = torch.mean(
            (x_anomaly - recon_anomaly) ** 2,
            dim=(1, 2)
        )

        target = self.recon_margin * normal_recon_err.detach()

        recon_penalty = torch.mean(
            F.relu(target - anomaly_recon_err)
        )

        # --------------------------------------------------
        # 2. LATENT SEPARATION
        # --------------------------------------------------

        latent_normal_mean  = latent_normal.mean(dim=1)
        latent_anomaly_mean = latent_anomaly.mean(dim=1)

        pull_loss = torch.mean(
            (latent_normal_mean - latent_center.detach()) ** 2
        )

        anomaly_dist = torch.mean(
            (latent_anomaly_mean - latent_center.detach()) ** 2,
            dim=1
        )

        push_margin = 1.0
        push_loss = torch.mean(
            F.relu(push_margin - anomaly_dist)
        )

        latent_sep = pull_loss + push_loss

        total = (
            self.recon_penalty_weight * recon_penalty +
            self.latent_sep_weight    * latent_sep
        )

        return total, recon_penalty, latent_sep

# ============================================================
# MODEL
# ============================================================

model = PRATANetV4(input_dim=INPUT_DIM).to(device)

criterion     = AdvancedPhysioLossV4()
adv_criterion = AdversarialAnomalyLoss(
    recon_margin=2.0,
    latent_sep_weight=0.3,
    recon_penalty_weight=0.5
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=1e-5
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer,
    T_0=10
)

print()
print("=" * 80)
print("MODEL")
print("=" * 80)

params = sum(p.numel() for p in model.parameters())

print(f"Parameters : {params:,}")

# ============================================================
# CONTRASTIVE AUGMENTATION
#
# For NT-Xent we need two augmented views of each window.
# We use lightweight physiologically-plausible augmentations:
#   - Small Gaussian noise (SNR-preserving)
#   - Temporal jitter (±1 sample shift)
#   - Amplitude scaling (±10%)
#
# These do NOT change the physiological meaning of the signal,
# so the model learns: same physiology -> same latent region.
# ============================================================

def contrastive_augment(x):
    """
    Create an augmented view of batch x for contrastive learning.
    x: (batch, time, features) tensor
    Returns: augmented tensor of same shape
    """

    aug = x.clone()

    B, T, F = aug.shape

    # 1. Small Gaussian noise
    aug = aug + torch.randn_like(aug) * 0.05

    # 2. Amplitude scaling (per sample, per feature)
    scale = 1.0 + (torch.rand(B, 1, F, device=x.device) - 0.5) * 0.2
    aug = aug * scale

    # 3. Temporal jitter: shift by ±1 along time axis
    shift = np.random.choice([-1, 0, 1])
    if shift != 0:
        aug = torch.roll(aug, shifts=shift, dims=1)

    return aug

# ============================================================
# TRAINING
# ============================================================

EPOCHS = 100

best_loss = 999999

print()
print("=" * 80)
print("TRAINING")
print("=" * 80)

for epoch in range(EPOCHS):

    model.train()

    running      = 0
    running_adv  = 0

    # Running latent center updated each epoch
    # Used by adversarial loss to define "normal" in latent space
    latent_center_accum = []

    for batch in train_loader:

        x = batch[0].to(device)

        # --------------------------------------------------
        # Forward pass (original normal batch)
        # --------------------------------------------------
        reconstruction, latent, proj = model(x)

        # --------------------------------------------------
        # Forward pass (augmented view for contrastive)
        # --------------------------------------------------
        x_aug = contrastive_augment(x)
        _, _, proj_aug = model(x_aug)

        # --------------------------------------------------
        # Standard loss (reconstruction + contrastive)
        # --------------------------------------------------
        loss = criterion(
            x,
            reconstruction,
            latent,
            proj=proj,
            proj_aug=proj_aug
        )

        # --------------------------------------------------
        # V6: Adversarial anomaly loss
        # Generate anomalies from current batch on-the-fly
        # and penalize the model for reconstructing them well
        # --------------------------------------------------

        # Accumulate latent center from this batch
        with torch.no_grad():
            latent_center_batch = latent.mean(dim=1).mean(dim=0)
            latent_center_accum.append(latent_center_batch)

        # Use running center from previous batches if available
        if len(latent_center_accum) > 1:

            latent_center_running = torch.stack(
                latent_center_accum[:-1]
            ).mean(dim=0)

            # Generate anomalies (CPU, convert back to tensor)
            x_np = x.cpu().numpy()
            x_anom_np = generate_physiological_anomalies(
                x_np,
                signal_stats=signal_stats
            )
            x_anom = torch.tensor(
                x_anom_np,
                dtype=torch.float32,
                device=device
            )

            recon_anom, latent_anom, _ = model(x_anom)

            adv_loss, _, _ = adv_criterion(
                x,
                reconstruction,
                latent,
                x_anom,
                recon_anom,
                latent_anom,
                latent_center_running
            )

            total_loss = loss + adv_loss
            running_adv += adv_loss.item()

        else:
            total_loss = loss

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        running += loss.item()

    scheduler.step()

    avg     = running     / len(train_loader)
    avg_adv = running_adv / max(1, len(train_loader) - 1)

    print(
        f"Epoch [{epoch+1}/{EPOCHS}] "
        f"Loss : {avg:.6f} | "
        f"Adv : {avg_adv:.6f}"
    )

    if avg < best_loss:

        best_loss = avg

        torch.save(
            model.state_dict(),
            os.path.join(MODEL_DIR, "best_prata_v6.pth")
        )

# ============================================================
# LOAD BEST MODEL
# ============================================================

model.load_state_dict(
    torch.load(
        os.path.join(MODEL_DIR, "best_prata_v6.pth")
    )
)

print("\nBest model loaded.")

# ============================================================
# TRAIN SCORES + LATENT STATISTICS
#
# V5 KEY UPGRADE: Mahalanobis-style latent scoring
#
# The problem in V4: 36% of anomalies scored below normal mean.
# Root cause: subtle anomalies (slow drift, autonomic imbalance)
# reconstruct well because the autoencoder generalizes too much.
#
# Solution: measure distance in LATENT space from the normal
# distribution, not just reconstruction error.
#
# For each normal window we record its latent mean vector.
# At test time: anomalies that reconstruct well will still
# have latent vectors far from the normal latent cloud.
#
# Implementation:
#   1. Collect all latent means from training (normal only)
#   2. Compute mean and std of latent distribution
#   3. Score = z-score distance from normal latent center
#   4. Combine with reconstruction score (weighted sum)
# ============================================================

model.eval()

train_scores    = []
train_latents   = []   # collect for latent distribution stats

with torch.no_grad():

    for batch in train_loader:

        x = batch[0].to(device)

        reconstruction, latent, proj = model(x)

        # --------------------------------------------------
        # Standard scores (same as V4)
        # --------------------------------------------------

        recon = torch.mean(
            (x - reconstruction) ** 2,
            dim=(1, 2)
        )

        latent_score = torch.mean(
            torch.abs(latent[:, 1:] - latent[:, :-1]),
            dim=(1, 2)
        )

        fft_orig  = torch.fft.rfft(x, dim=1)
        fft_recon = torch.fft.rfft(reconstruction, dim=1)

        freq_score = torch.mean(
            (torch.abs(fft_orig) - torch.abs(fft_recon)) ** 2,
            dim=(1, 2)
        )

        proj_norm    = F.normalize(proj, dim=-1)
        batch_center = proj_norm.mean(dim=0, keepdim=True)
        contrastive_score = 1.0 - (proj_norm * batch_center).sum(dim=-1)

        score = (
            0.55 * recon +
            0.10 * latent_score +
            0.15 * freq_score +
            0.20 * contrastive_score
        )

        train_scores.extend(score.cpu().numpy())

        # --------------------------------------------------
        # V5: Collect latent means for Mahalanobis scoring
        # latent: (B, T, latent_dim) -> mean over time -> (B, latent_dim)
        # --------------------------------------------------
        latent_mean = latent.mean(dim=1)   # (B, latent_dim)
        train_latents.append(latent_mean.cpu())

train_scores  = np.array(train_scores)
train_latents = torch.cat(train_latents, dim=0)   # (N_train, latent_dim)

# --------------------------------------------------
# Compute normal latent distribution statistics
# --------------------------------------------------

latent_center = train_latents.mean(dim=0)          # (latent_dim,)
latent_std    = train_latents.std(dim=0) + 1e-6    # (latent_dim,)

# Save for inference
np.save(
    os.path.join(MODEL_DIR, "prata_v6_latent_center.npy"),
    latent_center.numpy()
)
np.save(
    os.path.join(MODEL_DIR, "prata_v6_latent_std.npy"),
    latent_std.numpy()
)

latent_center = latent_center.to(device)
latent_std    = latent_std.to(device)

# ============================================================
# DYNAMIC THRESHOLD
# ============================================================

threshold = np.percentile(train_scores, 97)

print(f"\nThreshold : {threshold:.6f}")

# ============================================================
# EVALUATION
# ============================================================

all_scores = []
all_true   = []

# Accumulate train projections for reference center
train_projs = []

# First pass: collect train projections
model.eval()

with torch.no_grad():
    for batch in train_loader:
        x = batch[0].to(device)
        _, _, proj = model(x)
        proj_norm = F.normalize(proj, dim=-1)
        train_projs.append(proj_norm.cpu())

# Normal manifold center in projection space
train_proj_center = torch.cat(train_projs, dim=0).mean(dim=0)
train_proj_center = F.normalize(train_proj_center, dim=-1).to(device)

with torch.no_grad():

    for x, y in test_loader:

        x = x.to(device)

        reconstruction, latent, proj = model(x)

        recon = torch.mean(
            (x - reconstruction) ** 2,
            dim=(1, 2)
        )

        latent_score = torch.mean(
            torch.abs(latent[:, 1:] - latent[:, :-1]),
            dim=(1, 2)
        )

        fft_orig  = torch.fft.rfft(x, dim=1)
        fft_recon = torch.fft.rfft(reconstruction, dim=1)

        freq_score = torch.mean(
            (torch.abs(fft_orig) - torch.abs(fft_recon)) ** 2,
            dim=(1, 2)
        )

        proj_norm = F.normalize(proj, dim=-1)
        contrastive_score = 1.0 - (proj_norm * train_proj_center).sum(dim=-1)

        # V5: Mahalanobis-style latent distance
        # Measures how far each sample's latent mean is
        # from the normal latent distribution center,
        # normalized by per-dimension std (z-score distance)
        latent_mean = latent.mean(dim=1)   # (B, latent_dim)
        mahal_score = torch.mean(
            ((latent_mean - latent_center) / latent_std) ** 2,
            dim=1
        )   # (B,) — high for anomalies, low for normal

        # V5 score: reconstruction + latent structure + mahalanobis
        score = (
            0.45 * recon +
            0.08 * latent_score +
            0.12 * freq_score +
            0.15 * contrastive_score +
            0.20 * mahal_score
        )

        all_scores.extend(score.cpu().numpy())
        all_true.extend(y.numpy())

all_scores = np.array(all_scores)
all_true   = np.array(all_true)

predictions = (all_scores > threshold).astype(int)

# ============================================================
# METRICS
# ============================================================

accuracy  = accuracy_score(all_true, predictions)
precision = precision_score(all_true, predictions)
recall    = recall_score(all_true, predictions)
f1        = f1_score(all_true, predictions)
roc_auc   = roc_auc_score(all_true, all_scores)
pr_auc    = average_precision_score(all_true, all_scores)

# ============================================================
# RESULTS
# ============================================================

results = pd.DataFrame({
    "Metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "ROC_AUC",
        "PR_AUC"
    ],
    "Value": [
        accuracy,
        precision,
        recall,
        f1,
        roc_auc,
        pr_auc
    ]
})

print()
print("=" * 80)
print("FINAL RESULTS")
print("=" * 80)

print(results)

print()
print(f"ROC-AUC : {roc_auc:.4f}")
print(f"PR-AUC  : {pr_auc:.4f}")

# ============================================================
# SAVE
# ============================================================

results.to_csv(
    os.path.join(BASE_DIR, "prata_v6_results.csv"),
    index=False
)

np.save(
    os.path.join(BASE_DIR, "prata_v6_scores.npy"),
    all_scores
)

np.save(
    os.path.join(BASE_DIR, "prata_v6_labels.npy"),
    all_true
)

# Save train projection center for inference
np.save(
    os.path.join(MODEL_DIR, "prata_v6_proj_center.npy"),
    train_proj_center.cpu().numpy()
)

print()
print("=" * 80)
print("FILES SAVED")
print("=" * 80)

print("best_prata_v6.pth")
print("prata_v6_results.csv")
print("prata_v6_scores.npy")
print("prata_v6_labels.npy")
print("prata_v6_proj_center.npy")
print("prata_v6_latent_center.npy")
print("prata_v6_latent_std.npy")