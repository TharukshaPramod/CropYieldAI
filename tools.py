# tools.py
from pydantic import BaseModel
from typing import Optional, Dict

class PreProcessToolSchema(BaseModel):
    raw_data: Optional[dict] = None

class RetrieveDataToolSchema(BaseModel):
    query: Optional[dict] = None

class PredictToolSchema(BaseModel):
    data: Optional[dict] = None

class InterpretToolSchema(BaseModel):
    prediction: Optional[dict] = None
    baseline: Optional[float] = None  # <-- ensure float or None
    units: Optional[str] = "tons/ha"
