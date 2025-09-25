# main_api.py
import os
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import jwt
import concurrent.futures
import traceback

from orchestrator import run_pipeline
from agents.predictor import retrain_model
from utils.db import get_all_decrypted_docs

JWT_SECRET = os.getenv("JWT_SECRET", "supersecretkey")
# how long to wait for the pipeline (seconds) before returning a timeout to caller
RUN_PIPELINE_TIMEOUT = int(os.getenv("RUN_PIPELINE_TIMEOUT", "25"))

app = FastAPI(title="🌾 Crop Yield AI API", version="0.2.0")

# CORS for local dev UI and other tools
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

    # Run the pipeline in a thread with timeout to avoid blocking the server
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        future = ex.submit(run_pipeline, req.query, req.use_llm)
        try:
            res = future.result(timeout=RUN_PIPELINE_TIMEOUT)
        except concurrent.futures.TimeoutError:
            # If it times out, cancel the future and return a 504 so UI can react.
            future.cancel()
            raise HTTPException(status_code=504, detail=f"Pipeline timed out after {RUN_PIPELINE_TIMEOUT}s")
        except Exception as e:
            # unexpected error while running pipeline
            tb = traceback.format_exc()
            raise HTTPException(status_code=500, detail=f"Pipeline failed: {e}\n{tb}")

    return {"result": res}


# ------------------------
# Async job endpoints (non-blocking)
# ------------------------
import uuid
from typing import Dict, Any

JOBS: Dict[str, Dict[str, Any]] = {}
_EXECUTOR: concurrent.futures.ThreadPoolExecutor | None = None


def _get_executor() -> concurrent.futures.ThreadPoolExecutor:
    global _EXECUTOR
    if _EXECUTOR is None:
        _EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=4)
    return _EXECUTOR


@app.post("/predict_async")
def predict_async(req: PredictRequest, payload=Depends(verify_token)):
    if not req.query:
        raise HTTPException(status_code=400, detail="Query required")
    if req.model:
        os.environ["PREDICTOR_MODEL"] = req.model

    job_id = str(uuid.uuid4())
    JOBS[job_id] = {"status": "queued", "result": None, "error": None}

    def _run_job():
        JOBS[job_id]["status"] = "running"
        try:
            res = run_pipeline(req.query, req.use_llm)
            JOBS[job_id]["result"] = res
            JOBS[job_id]["status"] = "completed"
        except Exception as e:
            JOBS[job_id]["error"] = str(e)
            JOBS[job_id]["status"] = "failed"

    ex = _get_executor()
    ex.submit(_run_job)
    return {"job_id": job_id, "status": JOBS[job_id]["status"]}


@app.get("/jobs/{job_id}")
def get_job(job_id: str, payload=Depends(verify_token)):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job

@app.post("/train")
def train(payload=Depends(verify_token)):
    try:
        metrics = retrain_model()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retrain failed: {e}")
    return {"metrics": metrics}

@app.get("/stats")
def stats(payload=Depends(verify_token)):
    try:
        docs = get_all_decrypted_docs()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB error: {e}")

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
