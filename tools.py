# tools.py
from pydantic import BaseModel, field_validator
from typing import Optional, Any

class PreProcessToolSchema(BaseModel):
    raw_data: Optional[Any] = None

class RetrieveDataToolSchema(BaseModel):
    query: Optional[Any] = None

class PredictToolSchema(BaseModel):
    data: Optional[Any] = None

class InterpretToolSchema(BaseModel):
    prediction: Optional[Any] = None
    baseline: Optional[float] = None
    units: Optional[str] = "tons/ha"

    @field_validator("baseline", mode="before")
    def cast_baseline(cls, v):
        """Accepts float, str, dict, or None"""
        if v is None:
            return None
        if isinstance(v, dict):
            # LLM sometimes sends {"description": "5.0", "type": "float"}
            val = v.get("description") or v.get("value")
            try:
                return float(val)
            except Exception:
                return None
        try:
            return float(v)
        except Exception:
            return None

    @field_validator("units", mode="before")
    def cast_units(cls, v):
        """Accepts str, dict, or None"""
        if v is None:
            return "tons/ha"
        if isinstance(v, dict):
            return v.get("description") or "tons/ha"
        return str(v)
