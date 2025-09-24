# tools.py
from pydantic import BaseModel
from typing import Optional

class PreProcessToolSchema(BaseModel):
    raw_data: Optional[dict] = None

class RetrieveDataToolSchema(BaseModel):
    query: Optional[dict] = None

class PredictToolSchema(BaseModel):
    data: Optional[dict] = None

class InterpretToolSchema(BaseModel):
    prediction: Optional[dict] = None
