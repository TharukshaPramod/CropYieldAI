# agents/pre_processor.py
import os
import sys
import re
import json
from typing import Optional, Type

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# NLP fallback
try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
except Exception:
    nlp = None

from pydantic import BaseModel
from crewai import Agent
from crewai.tools import BaseTool
from tools import PreProcessToolSchema
from utils.security import sanitize_input, encrypt, decrypt

# small helper to call Ollama REST API
def _call_ollama(prompt: str, timeout: int = 10) -> Optional[str]:
    """
    Sends prompt to Ollama /api/generate and returns the model text (string)
    or None on failure.
    """
    try:
        import requests
    except Exception:
        return None

    base = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
    model = os.getenv("OLLAMA_MODEL", "llama3:latest")
    url = f"{base}/api/generate"

    payload = {
        "model": model,
        "prompt": prompt
    }
    headers = {"Content-Type": "application/json"}
    # Some Ollama deployments use an API key env; optional
    api_key = os.getenv("OLLAMA_API_KEY")
    if api_key:
        # Ollama local normally doesn't need auth; include as a header only if set.
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        r = requests.post(url, json=payload, headers=headers, timeout=timeout)
        if r.status_code == 200:
            # The API returns textual output; attempt to get the string
            text = r.text
            # Some responses may be raw or include streaming artifacts; try to parse simple JSON body if present
            try:
                # If Ollama returned JSON object itself, attempt to decode and find 'output' text
                body = r.json()
                # common pattern: body might be {"id":..., "result":[{"content":"..."}], ...}
                # fallback to returning r.text if none found
                # We handle varied shapes by picking any string-like member.
                def find_text(obj):
                    if isinstance(obj, str):
                        return obj
                    if isinstance(obj, dict):
                        for v in obj.values():
                            t = find_text(v)
                            if t:
                                return t
                    if isinstance(obj, list):
                        for v in obj:
                            t = find_text(v)
                            if t:
                                return t
                    return None
                found = find_text(body)
                if found:
                    return found if isinstance(found, str) else str(found)
                return text
            except Exception:
                return text
        else:
            return None
    except Exception:
        return None


def _extract_json_from_text(text: str) -> Optional[dict]:
    """
    Try to find JSON substring inside text and parse it.
    Returns dict or None.
    """
    if not text:
        return None
    # Try direct load
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # Try to find the first JSON object in the text using braces matching
    first = text.find("{")
    last = text.rfind("}")
    if first != -1 and last != -1 and last > first:
        snippet = text[first:last+1]
        try:
            return json.loads(snippet)
        except Exception:
            # as a last resort, try to fix common issues (single quotes -> double quotes)
            fixed = snippet.replace("'", '"')
            try:
                return json.loads(fixed)
            except Exception:
                return None
    return None


def _llm_extract_fields(user_text: str) -> Optional[dict]:
    """
    Uses Ollama to extract structured fields, returns dict or None.
    """
    # Build a strict instruction prompt
    prompt = (
        "You are a strict data extraction assistant. Given an input sentence "
        "containing agricultural field information, extract EXACTLY the following fields "
        "as a JSON object (no additional text):\n\n"
        "crop: string or null\n"
        "location: string or null\n"
        "rainfall: number (mm) or null\n"
        "temperature: number (°C) or null\n"
        "soil: string or null\n"
        "fertilizer: true/false or null\n"
        "irrigation: true/false or null\n\n"
        "Return VALID JSON ONLY. Example:\n"
        '{"crop":"wheat","location":"East","rainfall":492.0,"temperature":15.0,"soil":"Clay","fertilizer":true,"irrigation":true}\n\n'
        "If a field is missing, set it to null. Do not add commentary.\n\n"
        f"User text: \"{user_text}\"\n"
    )
    model_out = _call_ollama(prompt)
    if not model_out:
        return None
    parsed = _extract_json_from_text(model_out)
    return parsed


class PreProcessTool(BaseTool):
    name: str = "PreProcessData"
    description: str = "Pre-process raw crop queries into cleaned structured text (LLM extraction + fallback)."
    args_schema: Optional[Type[BaseModel]] = PreProcessToolSchema

    def _run(self, raw_data: Optional[dict] = None) -> str:
        # accept dict or string input
        q = None
        if isinstance(raw_data, dict):
            q = raw_data.get("description") or raw_data.get("raw") or str(raw_data)
        else:
            q = raw_data

        # Normalize common invalid forms: None, "None", empty strings
        if q is None or (isinstance(q, str) and q.strip().lower() in ("", "none", "null")):
            q = ""

        # decrypt if passed encrypted
        if isinstance(q, str) and q.startswith("gAAAAA"):
            try:
                q = decrypt(q)
            except Exception:
                pass

        cleaned = sanitize_input(q)
        if not cleaned:
            processed = "Cleaned:  (Extracted: Crop=unknown, Location=unknown)"
            print(f"[PreProcessor] {processed}")
            return encrypt(processed)
        extracted = None

        # If environment says to use LLM, attempt LLM extraction
        use_llm = os.getenv("USE_LLM", "false").lower() in ("1", "true", "yes")
        if use_llm:
            try:
                extracted = _llm_extract_fields(cleaned)
            except Exception:
                extracted = None

        # Fallback to regex + spaCy if LLM not used or failed
        if not extracted:
            crop = None
            loc = None
            rainfall = None
            temperature = None
            soil = None
            fertilizer = None
            irrigation = None

            m = re.search(r'(\b[A-Za-z]+\b)\s+yield', cleaned, re.I)
            if m:
                crop = m.group(1)

            m2 = re.search(r'in\s+([A-Za-z\s\-]+)', cleaned, re.I)
            if m2:
                loc = m2.group(1).strip()

            rv = re.search(r'Rainfall[:\s]*([0-9]+\.?[0-9]*)', cleaned, re.I)
            tv = re.search(r'Temperature[:\s]*([0-9]+\.?[0-9]*)', cleaned, re.I)
            sv = re.search(r'Soil[:\s]*([A-Za-z]+)', cleaned, re.I)
            fv = re.search(r'Fertilizer[:\s]*(True|False|true|false)', cleaned)
            iv = re.search(r'Irrigation[:\s]*(True|False|true|false)', cleaned)

            if rv:
                rainfall = float(rv.group(1))
            if tv:
                temperature = float(tv.group(1))
            if sv:
                soil = sv.group(1)
            if fv:
                fertilizer = True if fv.group(1).lower() == "true" else False
            if iv:
                irrigation = True if iv.group(1).lower() == "true" else False

            # try spaCy fallback to fill crop/location
            if nlp and (not crop or not loc):
                try:
                    doc = nlp(cleaned)
                    for ent in doc.ents:
                        if ent.label_ in ("PRODUCT", "NORP", "ORG") and not crop:
                            crop = ent.text
                        if ent.label_ in ("GPE", "LOC") and not loc:
                            loc = ent.text
                except Exception:
                    pass

            extracted = {
                "crop": crop,
                "location": loc,
                "rainfall": rainfall,
                "temperature": temperature,
                "soil": soil,
                "fertilizer": fertilizer,
                "irrigation": irrigation
            }

        # Normalize/format extracted values into readable processed string
        crop = extracted.get("crop") or "unknown"
        loc = extracted.get("location") or "unknown"
        # Build feature substrings for predictor to pick up via regex
        feature_parts = []
        if extracted.get("rainfall") is not None:
            feature_parts.append(f"Rainfall: {extracted['rainfall']}")
        if extracted.get("temperature") is not None:
            feature_parts.append(f"Temperature: {extracted['temperature']}")
        if extracted.get("soil"):
            feature_parts.append(f"Soil: {extracted['soil']}")
        if extracted.get("fertilizer") is not None:
            feature_parts.append(f"Fertilizer: {str(extracted['fertilizer'])}")
        if extracted.get("irrigation") is not None:
            feature_parts.append(f"Irrigation: {str(extracted['irrigation'])}")

        features_str = " ".join(feature_parts).strip()
        if features_str:
            processed = f"Cleaned: {cleaned} with {features_str} (Extracted: Crop={crop}, Location={loc})"
        else:
            processed = f"Cleaned: {cleaned} (Extracted: Crop={crop}, Location={loc})"

        print(f"[PreProcessor] {processed}")
        return encrypt(processed)


pre_processor_agent = Agent(
    role="Pre-Processor Agent",
    goal="Clean and prepare raw crop data for analysis",
    backstory=(
        "Specializes in data pre-processing with NLP and LLM-based extraction. "
        "Always use the tool named 'PreProcessData' with JSON input. "
        "Do not repeat the same input. If tool fails, provide a final answer instead of retrying."
    ),
    tools=[PreProcessTool()]
)


if __name__ == "__main__":
    from utils.security import decrypt
    sample = {"description": "Raw data: wheat! @East #Rainfall492mm Yield3.26 Temperature: 15 Soil:Clay Fertilizer: True Irrigation: True"}
    enc = pre_processor_agent.tools[0]._run(sample)
    print("Decrypted result:", decrypt(enc))
