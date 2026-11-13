from pathlib import Path
import sys
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from faultdiag.data import load_data
from faultdiag.features import build_features

MODEL_PATH = ROOT / "results" / "intellitwin_model.joblib"
DATA_PATH = ROOT / "data" / "pump_predictive_maintenance_dataset1.csv"
RESULTS_PATH = ROOT / "results" / "results.csv"

if not MODEL_PATH.exists():
    raise RuntimeError(f"Model bundle not found: {MODEL_PATH}")
if not DATA_PATH.exists():
    raise RuntimeError(f"Dataset not found: {DATA_PATH}")

bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
scaler = bundle["scaler"]
feature_columns = bundle["feature_columns"]
classes = bundle.get("classes", ["Normal", "Overload", "Pressure_drop", "Shutdown"])
best_model = bundle.get("best_model", "xgboost")
best_strategy = bundle.get("best_strategy", "smote")

# Use the exact training feature builder and saved column order.
raw_df = load_data(str(DATA_PATH))
processed_df, feature_frame = build_features(raw_df, columns=feature_columns)
scaled_features = scaler.transform(feature_frame)
latest_indices = processed_df.groupby("asset_id", sort=False).tail(1).index
latest_df = processed_df.loc[latest_indices].copy()
latest_df = latest_df.sort_values("asset_id").reset_index(drop=True)
latest_index_by_asset = {str(processed_df.loc[int(i), "asset_id"]): int(i) for i in latest_indices}

app = FastAPI(title="IntelliTwin API", version="1.0.0",
              description="Pump fault diagnosis using the saved IntelliTwin XGBoost model.")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def safe_float(value):
    try:
        value = float(value)
        return value if np.isfinite(value) else None
    except (TypeError, ValueError):
        return None

def row_payload(row):
    return {
        "asset_id": str(row["asset_id"]),
        "timestamp": pd.Timestamp(row["timestamp"]).isoformat(),
        "pump_type": str(row.get("pump_type", "Unknown")),
        "manufacturer": str(row.get("manufacturer", "Unknown")),
        "flow_rate_bpd": safe_float(row.get("flow_rate_bpd")),
        "pressure_psi": safe_float(row.get("pressure_psi")),
        "temperature_c": safe_float(row.get("temperature_c")),
        "vibration_mm_s": safe_float(row.get("vibration_mm_s")),
        "rotation_rpm": safe_float(row.get("rotation_rpm")),
        "power_kw": safe_float(row.get("power_kw")),
        "current_a": safe_float(row.get("current_a")),
        "voltage_v": safe_float(row.get("voltage_v")),
        "efficiency_pct": safe_float(row.get("efficiency_pct")),
        "operating_hours": safe_float(row.get("operating_hours")),
    }

def decode_fault_label(label):
    value = str(label)
    if value in classes:
        return value
    try:
        index = int(label)
        if 0 <= index < len(classes):
            return classes[index]
    except (TypeError, ValueError):
        pass
    return value

def prediction_for_index(index):
    x = scaled_features[index:index + 1]
    predicted_raw = model.predict(x)[0]
    probabilities = model.predict_proba(x)[0]
    predicted = decode_fault_label(predicted_raw)

    probability_map = {
        decode_fault_label(label): round(float(prob), 6)
        for label, prob in zip(model.classes_, probabilities)
    }

    confidence = float(max(probabilities))
    risk = "Low" if predicted == "Normal" else ("High" if confidence >= 0.80 else "Elevated")

    return {
        "predicted_fault": predicted,
        "confidence": round(confidence, 4),
        "confidence_percent": round(confidence * 100, 1),
        "risk_category": risk,
        "class_probabilities": probability_map,
        "model": best_model,
        "imbalance_strategy": best_strategy,
    }

@app.get("/api/health")
def health():
    return {"status": "ok", "model_loaded": True, "feature_count": len(feature_columns)}

@app.get("/api/overview")
def overview():
    counts = latest_df.shape[0]
    machine_payloads = []
    for _, row in latest_df.iterrows():
        idx = int(row.name) if row.name is not None else None
        machine_payloads.append(row_payload(row))
    # Predictions are calculated against the exact latest-row feature index.
    fault_counts = {c: 0 for c in classes}
    for idx in latest_indices:
        pred = prediction_for_index(int(idx))["predicted_fault"]
        fault_counts[pred] = fault_counts.get(pred, 0) + 1
    return {
        "machine_count": int(counts),
        "fault_counts": fault_counts,
        "latest_timestamp": pd.Timestamp(latest_df["timestamp"].max()).isoformat(),
        "model_name": best_model,
        "strategy": best_strategy,
        "feature_count": len(feature_columns),
        "data_mode": "Historical dataset",
    }

@app.get("/api/machines")
def machines():
    items = []
    for idx in latest_indices:
        row = processed_df.loc[int(idx)]
        item = row_payload(row)
        item.update(prediction_for_index(int(idx)))
        items.append(item)
    return {"machines": items}

@app.get("/api/machines/{asset_id}")
def machine_detail(asset_id: str):
    matches = latest_df[latest_df["asset_id"].astype(str) == str(asset_id)]
    if matches.empty:
        raise HTTPException(status_code=404, detail="Machine not found")
    idx = latest_index_by_asset[str(asset_id)]
    row = processed_df.loc[idx]
    result = row_payload(row)
    result.update(prediction_for_index(idx))
    return result

@app.get("/api/diagnose/{asset_id}")
def diagnose(asset_id: str):
    matches = latest_df[latest_df["asset_id"].astype(str) == str(asset_id)]
    if matches.empty:
        raise HTTPException(status_code=404, detail="Machine not found")
    idx = latest_index_by_asset[str(asset_id)]
    row = processed_df.loc[idx]
    result = row_payload(row)
    result.update(prediction_for_index(idx))
    result["diagnosis_basis"] = "Latest available historical reading for this pump"
    return result

@app.get("/api/model-results")
def model_results():
    if not RESULTS_PATH.exists():
        return {"rows": []}
    frame = pd.read_csv(RESULTS_PATH)
    rows = frame.replace({np.nan: None}).to_dict(orient="records")
    return {"rows": rows}
