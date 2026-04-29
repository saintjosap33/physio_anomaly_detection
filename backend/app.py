from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware  # ⭐ IMPORTANT
import numpy as np
import joblib
import os
import pandas as pd
from feature_extractor import extract_features_from_csv

app = FastAPI()

# ===== CORS FIX =====
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all (for development)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== PATHS =====
MODEL_PATH = r"D:\somma\research\health-dashboard-clean\backend\saved_models"
THRESHOLD_PATH = r"D:\somma\research\health-dashboard-clean\backend\thresholds.npy"

# ===== LOAD MODELS =====
scaler = joblib.load(os.path.join(MODEL_PATH, "scaler.pkl"))
pca = joblib.load(os.path.join(MODEL_PATH, "pca.pkl"))
iso = joblib.load(os.path.join(MODEL_PATH, "isolation_forest.pkl"))

thresholds = np.load(THRESHOLD_PATH)
threshold_pca = thresholds[0]

print("✅ Models loaded")

@app.get("/")
def home():
    return {"status": "API running 🚀"}

# =========================
# CSV UPLOAD ENDPOINT
# =========================
@app.post("/predict-csv")
async def predict_csv(file: UploadFile = File(...)):
    try:
        df = pd.read_csv(file.file)

        data = df.values

        # ✅ Validate shape
        if data.shape != (60, 4):
            return {
                "status": "error",
                "message": "CSV must be 60 rows × 4 columns"
            }

        # ===== FEATURE EXTRACTION =====
        features = extract_features_from_csv(data)

        # ===== MODEL PIPELINE =====
        X_scaled = scaler.transform(features)

        X_pca = pca.transform(X_scaled)
        X_recon = pca.inverse_transform(X_pca)

        pca_error = np.mean((X_scaled - X_recon) ** 2)
        pca_anomaly = int(pca_error > threshold_pca)

        iso_pred = iso.predict(X_scaled)[0]
        iso_anomaly = 1 if iso_pred == -1 else 0

        final = 1 if (pca_anomaly or iso_anomaly) else 0

        return {
            "status": "success",
            "anomaly": int(final),  # 🔧 changed to int (frontend friendly)
            "pca": pca_anomaly,
            "iso": iso_anomaly,
            "pca_error": float(pca_error),
            "score": float(pca_error)
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }