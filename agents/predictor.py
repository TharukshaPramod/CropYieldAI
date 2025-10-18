# agents/predictor.py
import os, sys, re
from typing import Optional, Type, Dict, Any
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from pydantic import BaseModel
from crewai import Agent
from crewai.tools import BaseTool
from tools import PredictToolSchema
from utils.security import sanitize_input, encrypt, decrypt
from utils.db import get_all_decrypted_docs

MODEL_DIR = os.path.join(ROOT, "models")
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, "predictor.joblib")
ENC_PATH = os.path.join(MODEL_DIR, "soil_encoder.joblib")

DEFAULT_MODEL_TYPE = os.getenv("PREDICTOR_MODEL", "rf").lower()


def _build_dataframe_from_docs() -> pd.DataFrame:
    """Convert DB docs into a structured DataFrame for training."""
    docs = get_all_decrypted_docs()
    rows = []
    for d in docs:
        text = d.get("data", "")
        rv = re.search(r"Rainfall[:\s]*([0-9]+\.?[0-9]*)", text)
        tv = re.search(r"Temperature[:\s]*([0-9]+\.?[0-9]*)", text)
        yv = re.search(r"Yield[:\s]*([0-9]+\.?[0-9]*)", text)
        fv = re.search(r"Fertilizer[:\s]*(True|False)", text)
        iv = re.search(r"Irrigation[:\s]*(True|False)", text)
        sv = re.search(r"Soil[:\s]*([A-Za-z]+)", text)
        if rv and tv and yv and sv:
            rows.append({
                "Rainfall": float(rv.group(1)),
                "Temperature": float(tv.group(1)),
                "Fertilizer": 1 if fv and fv.group(1) == "True" else 0,
                "Irrigation": 1 if iv and iv.group(1) == "True" else 0,
                "Soil": sv.group(1),
                "Yield": float(yv.group(1))
            })
    return pd.DataFrame(rows)


def _train_and_persist(df: pd.DataFrame, model_type: Optional[str] = None) -> Dict[str, Any]:
    """Train model on df, run cross-validation, persist model and encoder. Returns metrics dict."""
    if model_type is None:
        model_type = DEFAULT_MODEL_TYPE

    # Encode soil
    le = LabelEncoder()
    df = df.copy()
    df["Soil_enc"] = le.fit_transform(df["Soil"].astype(str))

    X = df[["Rainfall", "Temperature", "Fertilizer", "Irrigation", "Soil_enc"]]
    y = df["Yield"]

    if model_type.startswith("rf"):
        model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    else:
        model = LinearRegression()

    # cross-validated predictions
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    try:
        preds = cross_val_predict(model, X, y, cv=cv, n_jobs=-1)
    except Exception:
        # fallback: simple train/test split prediction if cross_val fails
        model.fit(X, y)
        preds = model.predict(X)

    mae = float(mean_absolute_error(y, preds))
    mse = mean_squared_error(y, preds)
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y, preds))

    # fit final model on full data and persist
    model.fit(X, y)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(le, ENC_PATH)

    metrics = {"mae": mae, "rmse": rmse, "r2": r2, "n_samples": len(df), "model_type": model_type}
    return metrics


def retrain_model(model_type: Optional[str] = None) -> Dict[str, Any]:
    """Public function that trains model from DB and returns metrics."""
    df = _build_dataframe_from_docs()
    if df.empty:
        return {"error": "No training data found in DB."}
    return _train_and_persist(df, model_type=model_type)


class PredictTool(BaseTool):
    name: str = "PredictYield"
    description: str = "Predicts crop yield using weather/soil features (uses persisted model)"
    args_schema: Optional[Type[BaseModel]] = PredictToolSchema

    def _run(self, data: Optional[dict] = None) -> str:
        # Input validation
        if not data:
            return encrypt("Error: No input provided to predictor.")
        q = data.get("description") if isinstance(data, dict) else data
        if not isinstance(q, str) or not q.strip():
            return encrypt("Error: Missing description for prediction.")

        # decrypt if necessary
        if isinstance(q, str) and q.startswith("gAAAAA"):
            try:
                q = decrypt(q)
            except Exception:
                return encrypt("Error: Could not decrypt predictor input.")

        q = sanitize_input(q or "")

        # parse trivial fields
        crop = None
        location = None
        m = re.search(r"(\b[A-Za-z]+\b)\s+yield", q, re.I)
        if m:
            crop = m.group(1)
        m2 = re.search(r"in\s+([A-Za-z\s\-]+)", q, re.I)
        if m2:
            location = m2.group(1).strip()

        df = _build_dataframe_from_docs()
        if df.empty:
            return encrypt("Error: No training data available for prediction.")

        # ensure model exists and is valid
        if not (os.path.exists(MODEL_PATH) and os.path.exists(ENC_PATH)):
            _train_and_persist(df)

        # Try to load model with error handling
        try:
            model = joblib.load(MODEL_PATH)
            le = joblib.load(ENC_PATH)
        except Exception as e:
            print(f"[Predictor] Model loading failed: {e}. Retraining...")
            _train_and_persist(df)
            try:
                model = joblib.load(MODEL_PATH)
                le = joblib.load(ENC_PATH)
            except Exception as e2:
                return encrypt(f"Error: Could not load or retrain model: {e2}")


        # parse query features (fallbacks)
        q_r = re.search(r"Rainfall[:\s]*([0-9]+\.?[0-9]*)", q)
        q_t = re.search(r"Temperature[:\s]*([0-9]+\.?[0-9]*)", q)
        q_f = re.search(r"Fertilizer[:\s]*(True|False)", q)
        q_i = re.search(r"Irrigation[:\s]*(True|False)", q)
        q_s = re.search(r"Soil[:\s]*([A-Za-z]+)", q)

        rainfall = float(q_r.group(1)) if q_r else float(df["Rainfall"].mean())
        temperature = float(q_t.group(1)) if q_t else float(df["Temperature"].mean())
        fertilizer = 1 if (q_f and q_f.group(1) == "True") else 0
        irrigation = 1 if (q_i and q_i.group(1) == "True") else 0

        soil_val = q_s.group(1) if q_s else df["Soil"].mode().iloc[0]
        if soil_val in list(le.classes_):
            soil_enc = int(le.transform([soil_val])[0])
        else:
            try:
                mode_soil = df["Soil"].mode().iloc[0]
                soil_enc = int(le.transform([mode_soil])[0])
            except Exception:
                soil_enc = 0

        inp = pd.DataFrame([{
            "Rainfall": rainfall,
            "Temperature": temperature,
            "Fertilizer": fertilizer,
            "Irrigation": irrigation,
            "Soil_enc": soil_enc
        }])

        try:
            prediction = float(model.predict(inp)[0])
        except Exception:
            prediction = float(df["Yield"].mean())

        crop = crop or "unspecified crop"
        location = location or "unknown location"
        result = f"Predicted yield for {crop} in {location}: {prediction:.2f} tons/ha"
        print(f"[Predictor] {result}")
        return encrypt(result)


predictor_agent = Agent(
    role="Predictor Agent",
    goal="Predict crop yields",
    backstory="Uses a persisted ML model for predictions",
    tools=[PredictTool()]
)


if __name__ == "__main__":
    # quick local test
    sample = {"description": "wheat yield in East with Rainfall: 492 Temperature: 15 Soil: Clay Fertilizer: True Irrigation: True"}
    enc = predictor_agent.tools[0]._run(sample)
    from utils.security import decrypt
    print("Decrypted prediction:", decrypt(enc))
