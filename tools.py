# tools.py
from pydantic import BaseModel
from typing import Optional, Dict, Union

class PreProcessToolSchema(BaseModel):
    # Accept dict, raw string, or None (crew can pass "None" as a string)
    raw_data: Optional[Union[dict, str]] = None

class RetrieveDataToolSchema(BaseModel):
    query: Optional[dict] = None



class InterpretToolSchema(BaseModel):
    # Accept dict, float, or string for prediction (LLM might pass raw numbers)
    prediction: Optional[Union[dict, float, str]] = None
    # Accept float, string "None", or None for baseline
    baseline: Optional[Union[float, str]] = None
    units: Optional[str] = "tons/ha"
