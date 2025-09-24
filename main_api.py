# main_api.py (replace existing file)

import os
from dotenv import load_dotenv
load_dotenv()
from fastapi import FastAPI, HTTPException, Request, Depends
from pydantic import BaseModel
import jwt
from orchestrator import run_pipeline

JWT_SECRET = os.getenv("JWT_SECRET", "supersecretkey")

app = FastAPI(title="Crop Yield AI API")

class PredictRequest(BaseModel):
    query: str

def verify_token(request: Request):
    auth = request.headers.get("Authorization")
    if not auth or not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    token = auth.split(" ", 1)[1]
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
        return payload
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid token")

@app.post("/predict")
def predict(req: PredictRequest, payload = Depends(verify_token)):
    if not req.query:
        raise HTTPException(status_code=400, detail="Query required")
    res = run_pipeline(req.query)
    return {"result": res}
