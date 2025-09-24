# agents/predictor.py
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import re
from typing import Optional, Type
from pydantic import BaseModel
from crewai import Agent
from crewai.tools import BaseTool
from tools import PredictToolSchema
from utils.security import sanitize_input, encrypt, decrypt
from utils.db import get_all_decrypted_docs
import pandas as pd
from sklearn.linear_model import LinearRegression

class PredictTool(BaseTool):
    name: str = "PredictYield"
    description: str = "Predicts crop yield based on weather data"
    args_schema: Optional[Type[BaseModel]] = PredictToolSchema

    def _run(self, data: Optional[dict] = None) -> str:
        q = data.get("description") if isinstance(data, dict) else data
        if isinstance(q, str) and q.startswith("gAAAAA"):
            try:
                q = decrypt(q)
            except:
                pass
        q = sanitize_input(q)
        crop = None
        location = None
        m = re.search(r'(\b[A-Za-z]+\b)\s+yield', q, re.I)
        if m:
            crop = m.group(1)
        m2 = re.search(r'in\s+([A-Za-z\s\-]+)', q, re.I)
        if m2:
            location = m2.group(1).strip()

        docs = get_all_decrypted_docs()
        rows = []
        for d in docs:
            text = d['data']
            rv = re.search(r'Rainfall[:\s]*([0-9]+\.?[0-9]*)', text)
            tv = re.search(r'Temperature[:\s]*([0-9]+\.?[0-9]*)', text)
            yv = re.search(r'Yield[:\s]*([0-9]+\.?[0-9]*)', text)
            if rv and tv and yv:
                rows.append({
                    'Rainfall': float(rv.group(1)),
                    'Temperature': float(tv.group(1)),
                    'Yield': float(yv.group(1))
                })
        df = pd.DataFrame(rows)
        if not df.empty and len(df) >= 2:
            X = df[['Rainfall', 'Temperature']]
            y = df['Yield']
            model = LinearRegression().fit(X, y)
            q_r = re.search(r'Rainfall[:\s]*([0-9]+\.?[0-9]*)', q)
            q_t = re.search(r'Temperature[:\s]*([0-9]+\.?[0-9]*)', q)
            if q_r and q_t:
                inp = pd.DataFrame({'Rainfall': [float(q_r.group(1))], 'Temperature':[float(q_t.group(1))]})
            else:
                inp = pd.DataFrame({'Rainfall': [df['Rainfall'].mean()], 'Temperature':[df['Temperature'].mean()]})
            prediction = float(model.predict(inp)[0])
        else:
            prediction = float(df['Yield'].mean()) if not df.empty else 4.0

        crop = crop or "wheat"
        location = location or "unknown"
        result = f"Predicted yield for {crop} in {location}: {prediction:.2f} tons/ha"
        print(f"[Predictor] {result}")
        return encrypt(result)

predictor_agent = Agent(
    role="Predictor Agent",
    goal="Predict crop yields based on processed data",
    backstory="Specializes in yield prediction using NLP and basic models",
    tools=[PredictTool()]
)

if __name__ == "__main__":
    from utils.security import decrypt
    sample = {"description": "wheat yield in East"}
    enc = predictor_agent.tools[0]._run(sample)
    print("Decrypted prediction:", decrypt(enc))
