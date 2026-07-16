"""
PRATA v5 — Anomaly Detection Backend
Run: python backend_server.py
Deps: pip install fastapi uvicorn torch numpy python-multipart

Expects best_prata_v5.pth in the same folder.

CSV input format (27 rows × 7 columns):
    BVP, EDA, TEMP, ACC_x, ACC_y, ACC_z, HR
    (or any 7-channel signal your model was trained on)
"""

import os
import io
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import uvicorn

# ── Model architecture ────────────────────────────────────────────────────────

class FeatureGate(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(input_dim, input_dim), nn.LayerNorm(input_dim), nn.SiLU(),
            nn.Linear(input_dim, input_dim), nn.Sigmoid()
        )
    def forward(self, x): return x * self.gate(x)

class MultiScaleConv(nn.Module):
    def __init__(self, input_dim, out_channels=128):
        super().__init__()
        self.conv3    = nn.Conv1d(input_dim, out_channels, 3, padding=1)
        self.conv5    = nn.Conv1d(input_dim, out_channels, 5, padding=2)
        self.conv7    = nn.Conv1d(input_dim, out_channels, 7, padding=3)
        self.bn       = nn.BatchNorm1d(out_channels * 3)
        self.residual = nn.Conv1d(input_dim, out_channels * 3, 1)
    def forward(self, x):
        x_t = x.transpose(1, 2)
        out = torch.cat([self.conv3(x_t), self.conv5(x_t), self.conv7(x_t)], dim=1)
        return F.silu(self.bn(out) + self.residual(x_t)).transpose(1, 2)

class TransformerBlock(nn.Module):
    def __init__(self, d_model, nhead=8):
        super().__init__()
        self.attn  = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff    = nn.Sequential(nn.Linear(d_model, d_model*4), nn.GELU(), nn.Dropout(0.1), nn.Linear(d_model*4, d_model))
    def forward(self, x):
        a, _ = self.attn(x, x, x)
        x = self.norm1(x + a)
        return self.norm2(x + self.ff(x))

class TemporalTransformer(nn.Module):
    def __init__(self, d_model=256, seq_len=32, num_layers=3):
        super().__init__()
        self.pos_enc = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
        self.layers  = nn.ModuleList([TransformerBlock(d_model) for _ in range(num_layers)])
    def forward(self, x):
        x = x + self.pos_enc[:, :x.size(1)]
        for layer in self.layers: x = layer(x)
        return x

class ProjectionHead(nn.Module):
    def __init__(self, in_dim=64, proj_dim=32):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(in_dim, in_dim), nn.ReLU(), nn.Linear(in_dim, proj_dim), nn.LayerNorm(proj_dim))
    def forward(self, x): return self.proj(x)

class PRATAv5(nn.Module):
    def __init__(self, input_dim=189, seq_len=32):
        super().__init__()
        self.feature_gate         = FeatureGate(input_dim)
        self.multi_scale          = MultiScaleConv(input_dim, 128)
        self.transformer1         = TransformerBlock(384)
        self.transformer2         = TransformerBlock(384)
        self.bi_lstm              = nn.LSTM(384, 128, batch_first=True, bidirectional=True)
        self.temporal_transformer = TemporalTransformer(256, seq_len)
        self.latent    = nn.Sequential(nn.Linear(256,128), nn.GELU(), nn.Dropout(0.1), nn.Linear(128,64))
        self.proj_head = ProjectionHead(64, 32)
        self.decoder   = nn.LSTM(64, 128, num_layers=2, batch_first=True, bidirectional=True)
        self.output    = nn.Sequential(nn.Linear(256,256), nn.GELU(), nn.Dropout(0.1), nn.Linear(256,input_dim))
    def encode(self, x):
        x = self.feature_gate(x)
        x = self.multi_scale(x)
        x = self.transformer1(x)
        x = self.transformer2(x)
        x, _ = self.bi_lstm(x)
        return self.latent(self.temporal_transformer(x))
    def decode(self, z):
        out, _ = self.decoder(z)
        return self.output(out)
    def forward(self, x):
        z = self.encode(x)
        return self.decode(z), self.proj_head(z.mean(dim=1)), z

# ── Load model ────────────────────────────────────────────────────────────────

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best_prata_v5.pth")
DEVICE     = torch.device("cpu")

model = PRATAv5(input_dim=189, seq_len=32)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False))
model.eval()
print(f"✅ Model loaded — input: 27 timesteps × 7 channels = 189 features")

ANOMALY_THRESHOLD = 0.30   # update after running calibrate_threshold.py
TIMESTEPS         = 27
CHANNELS          = 7

# ── FastAPI ───────────────────────────────────────────────────────────────────

app = FastAPI(title="PRATA v5 Anomaly Detection")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

def run_inference(window: np.ndarray):
    """
    window: (27, 7) float32 array
    Returns dict with anomaly result.
    """
    # normalise each channel to [0,1] using its own min/max
    mn  = window.min(axis=0, keepdims=True)
    mx  = window.max(axis=0, keepdims=True)
    rng = np.where((mx - mn) < 1e-8, 1.0, mx - mn)
    w   = (window - mn) / rng                         # (27, 7)

    flat = w.flatten().astype(np.float32)              # (189,)
    x    = torch.tensor(flat.reshape(1, 1, 189))       # (1, 1, 189)

    with torch.no_grad():
        recon, _, _ = model(x)
        mse = F.mse_loss(recon, x).item()

    anomaly    = 1 if mse > ANOMALY_THRESHOLD else 0
    confidence = round(min(abs(mse - ANOMALY_THRESHOLD) / ANOMALY_THRESHOLD, 1.0), 4)

    return {
        "anomaly":    anomaly,
        "label":      "Anomaly" if anomaly else "Normal",
        "score":      round(mse, 6),
        "threshold":  ANOMALY_THRESHOLD,
        "confidence": confidence,
    }

# ── /predict/csv  (main endpoint) ────────────────────────────────────────────

@app.post("/predict/csv")
async def predict_csv(file: UploadFile = File(...)):
    """
    Upload a CSV file with 27 rows × 7 columns.
    Columns: BVP, EDA, TEMP, ACC_x, ACC_y, ACC_z, HR
    (header row optional — detected automatically)
    Returns anomaly result + per-timestep scores.
    """
    content = await file.read()
    text    = content.decode("utf-8", errors="ignore")

    try:
        # auto-detect header
        first_line = text.strip().split("\n")[0]
        has_header = not all(
            part.replace(".", "").replace("-", "").replace("e", "").replace("E","").replace("+","").isdigit()
            for part in first_line.split(",") if part.strip()
        )
        data = np.genfromtxt(
            io.StringIO(text), delimiter=",",
            skip_header=1 if has_header else 0,
            dtype=np.float32
        )
    except Exception as e:
        raise HTTPException(400, f"Could not parse CSV: {e}")

    if data.ndim == 1:
        data = data.reshape(1, -1)

    rows, cols = data.shape

    # ── accept various shapes ─────────────────────────────────────────────
    # shape A: (27, 7)  — one window, 7 channels
    # shape B: (N, 7)   — N timesteps, sliding window of 27
    # shape C: (27, K)  — one window, K channels (pad/trim to 7)
    # shape D: (1, 189) — already flattened

    if cols == 189 and rows == 1:
        # already flat
        window = data.reshape(TIMESTEPS, CHANNELS)
    elif cols == 189:
        window = data[0].reshape(TIMESTEPS, CHANNELS)
    elif rows == TIMESTEPS and cols >= CHANNELS:
        window = data[:, :CHANNELS]
    elif rows >= TIMESTEPS and cols >= CHANNELS:
        window = data[:TIMESTEPS, :CHANNELS]
    elif rows == TIMESTEPS and cols < CHANNELS:
        # pad missing channels with zeros
        pad    = np.zeros((rows, CHANNELS - cols), dtype=np.float32)
        window = np.hstack([data, pad])
    else:
        raise HTTPException(400,
            f"CSV shape ({rows}×{cols}) not usable. Need at least {TIMESTEPS} rows × {CHANNELS} cols, "
            f"or 1 row × 189 cols.")

    result = run_inference(window)

    # also compute per-timestep MSE for the sparkline chart
    per_step = []
    for i in range(TIMESTEPS):
        row_flat = window[i].astype(np.float32)
        row_flat = (row_flat - row_flat.min()) / max(row_flat.max() - row_flat.min(), 1e-8)
        padded   = np.zeros(189, dtype=np.float32)
        padded[:CHANNELS] = row_flat
        x = torch.tensor(padded.reshape(1, 1, 189))
        with torch.no_grad():
            recon, _, _ = model(x)
            mse = F.mse_loss(recon, x).item()
        per_step.append(round(mse, 6))

    result["per_step_scores"] = per_step
    result["timesteps"]       = TIMESTEPS
    result["channels"]        = CHANNELS
    result["rows_received"]   = int(rows)
    result["cols_received"]   = int(cols)
    return result

# ── /predict/window  (JSON array) ────────────────────────────────────────────

class WindowInput(BaseModel):
    window: List[List[float]]   # 27 × 7

@app.post("/predict/window")
def predict_window(body: WindowInput):
    """
    JSON: { "window": [[bvp,eda,temp,ax,ay,az,hr], ...] }  (27 rows)
    """
    arr = np.array(body.window, dtype=np.float32)
    if arr.shape != (TIMESTEPS, CHANNELS):
        raise HTTPException(400, f"Expected ({TIMESTEPS},{CHANNELS}), got {arr.shape}")
    return run_inference(arr)

# ── /health ───────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "model": "PRATA v5", "input": f"{TIMESTEPS}×{CHANNELS}=189", "threshold": ANOMALY_THRESHOLD}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=False)
