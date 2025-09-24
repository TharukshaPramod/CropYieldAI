import os, sys, re, pandas as pd
from typing import Optional, Type
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from pydantic import BaseModel
from crewai import Agent
from crewai.tools import BaseTool
from tools import PredictToolSchema
from utils.security import sanitize_input, encrypt, decrypt
from utils.db import get_all_decrypted_docs

class PredictTool(BaseTool):
    name: str = "PredictYield"
    description: str = "Predicts crop yield based on weather and soil data"
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
        soil_set = set()
        for d in docs:
            text = d['data']
            rv = re.search(r'Rainfall[:\s]*([0-9]+\.?[0-9]*)', text)
            tv = re.search(r'Temperature[:\s]*([0-9]+\.?[0-9]*)', text)
            fv = re.search(r'Fertilizer[:\s]*(True|False)', text)
            iv = re.search(r'Irrigation[:\s]*(True|False)', text)
            yv = re.search(r'Yield[:\s]*([0-9]+\.?[0-9]*)', text)
            sv = re.search(r'Soil[:\s]*([A-Za-z]+)', text)
            if rv and tv and yv and sv and fv and iv:
                soil_set.add(sv.group(1))
                rows.append({
                    'Rainfall': float(rv.group(1)),
                    'Temperature': float(tv.group(1)),
                    'Fertilizer': 1 if fv.group(1) == "True" else 0,
                    'Irrigation': 1 if iv.group(1) == "True" else 0,
                    'Soil': sv.group(1),
                    'Yield': float(yv.group(1))
                })

        df = pd.DataFrame(rows)
        if df.empty:
            prediction = 4.0
        else:
            # Encode Soil
            le = LabelEncoder()
            df['Soil_enc'] = le.fit_transform(df['Soil'])

            X = df[['Rainfall', 'Temperature', 'Fertilizer', 'Irrigation', 'Soil_enc']]
            y = df['Yield']
            model = LinearRegression().fit(X, y)

            # extract features from query
            q_r = re.search(r'Rainfall[:\s]*([0-9]+\.?[0-9]*)', q)
            q_t = re.search(r'Temperature[:\s]*([0-9]+\.?[0-9]*)', q)
            q_f = re.search(r'Fertilizer[:\s]*(True|False)', q)
            q_i = re.search(r'Irrigation[:\s]*(True|False)', q)
            q_s = re.search(r'Soil[:\s]*([A-Za-z]+)', q)

            inp = pd.DataFrame([{
                'Rainfall': float(q_r.group(1)) if q_r else df['Rainfall'].mean(),
                'Temperature': float(q_t.group(1)) if q_t else df['Temperature'].mean(),
                'Fertilizer': 1 if q_f and q_f.group(1) == "True" else 0,
                'Irrigation': 1 if q_i and q_i.group(1) == "True" else 0,
                'Soil_enc': le.transform([q_s.group(1)])[0] if q_s and q_s.group(1) in le.classes_ else df['Soil_enc'].mean()
            }])

            prediction = float(model.predict(inp)[0])

        crop = crop or "wheat"
        location = location or "unknown"
        result = f"Predicted yield for {crop} in {location}: {prediction:.2f} tons/ha"
        print(f"[Predictor] {result}")
        return encrypt(result)

predictor_agent = Agent(
    role="Predictor Agent",
    goal="Predict crop yields based on weather, soil, and irrigation/fertilizer data",
    backstory="Specializes in yield prediction using multiple features",
    tools=[PredictTool()]
)

if __name__ == "__main__":
    from utils.security import decrypt
    sample = {"description": "wheat yield in East with Rainfall: 492 Temperature: 15 Soil: Clay Fertilizer: True Irrigation: True"}
    enc = predictor_agent.tools[0]._run(sample)
    print("Decrypted prediction:", decrypt(enc))
