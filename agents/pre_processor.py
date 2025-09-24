# agents/pre_processor.py
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import re
from typing import Optional, Type
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

class PreProcessTool(BaseTool):
    name: str = "PreProcessData"
    description: str = "Pre-process raw crop queries into cleaned structured text."
    args_schema: Optional[Type[BaseModel]] = PreProcessToolSchema

    def _run(self, raw_data: Optional[dict] = None) -> str:
        q = None
        if isinstance(raw_data, dict):
            q = raw_data.get("description") or raw_data.get("raw") or str(raw_data)
        else:
            q = raw_data
        if isinstance(q, str) and q.startswith("gAAAAA"):
            try:
                q = decrypt(q)
            except:
                pass
        cleaned = sanitize_input(q)
        crop = None
        loc = None
        m = re.search(r'(\b[A-Za-z]+\b)\s+yield', cleaned, re.I)
        if m:
            crop = m.group(1)
        m2 = re.search(r'in\s+([A-Za-z\s\-]+)', cleaned, re.I)
        if m2:
            loc = m2.group(1).strip()
        if nlp and (not crop or not loc):
            doc = nlp(cleaned)
            for ent in doc.ents:
                if ent.label_ in ("PRODUCT", "NORP", "ORG") and not crop:
                    crop = ent.text
                if ent.label_ in ("GPE", "LOC") and not loc:
                    loc = ent.text
        crop = crop or "unknown"
        loc = loc or "unknown"
        processed = f"Cleaned: {cleaned} (Extracted: Crop={crop}, Location={loc})"
        print(f"[PreProcessor] {processed}")
        return encrypt(processed)

pre_processor_agent = Agent(
    role="Pre-Processor Agent",
    goal="Clean and prepare raw crop data for analysis",
    backstory="Specializes in data pre-processing with NLP and security",
    tools=[PreProcessTool()]
)

if __name__ == "__main__":
    from utils.security import decrypt
    sample = {"description": "Raw data: wheat! @East #Rainfall492mm Yield3.26"}
    enc = pre_processor_agent.tools[0]._run(sample)
    print("Decrypted result:", decrypt(enc))
