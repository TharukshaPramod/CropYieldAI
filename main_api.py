import os
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException, Request, Depends
from pydantic import BaseModel
import jwt

from orchestrator import run_pipeline
from agents.predictor import retrain_model
from utils.db import get_all_decrypted_docs

JWT_SECRET = os.getenv("JWT_SECRET", "supersecretkey")

app = FastAPI(title="🌾 Crop Yield AI API")

# ------------------------
# Models
# ------------------------
class PredictRequest(BaseModel):
    query: str
    use_llm: bool = False
    model: str | None = None

# ------------------------
# Auth
# ------------------------
def verify_token(request: Request):
    auth = request.headers.get("Authorization")
    if not auth or not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
    token = auth.split(" ", 1)[1]
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
        return payload
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

# ------------------------
# Routes
# ------------------------
@app.post("/predict")
def predict(req: PredictRequest, payload=Depends(verify_token)):
    if not req.query:
        raise HTTPException(status_code=400, detail="Query required")
    if req.model:
        os.environ["PREDICTOR_MODEL"] = req.model
    res = run_pipeline(req.query, use_llm=req.use_llm)
    return {"result": res}

@app.post("/train")
def train(payload=Depends(verify_token)):
    metrics = retrain_model()
    return {"metrics": metrics}

@app.get("/stats")
def stats(payload=Depends(verify_token)):
    docs = get_all_decrypted_docs()
    if not docs:
        return {"n_records": 0}

    import re, statistics
    yields = []
    crops = set()
    locations = set()
    for d in docs:
        if d.get("crop"):
            crops.add(d.get("crop"))
        if d.get("location"):
            locations.add(d.get("location"))
        m = re.search(r"Yield[:\s]*([0-9]+\.?[0-9]*)", d.get("data", ""))
        if m:
            try:
                yields.append(float(m.group(1)))
            except Exception:
                pass

    return {
        "n_records": len(docs),
        "crops": list(crops),
        "locations": list(locations),
        "yield_mean": statistics.mean(yields) if yields else None,
        "yield_min": min(yields) if yields else None,
        "yield_max": max(yields) if yields else None,
    }

# ------------------------
# Health check
# ------------------------
@app.get("/health")
def health():
    return {"status": "ok"}
