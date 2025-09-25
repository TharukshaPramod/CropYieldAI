# utils/evaluate.py
from utils.db import get_all_decrypted_docs
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib, os, math

def build_df():
    docs = get_all_decrypted_docs()
    rows = []
    import re
    for d in docs:
        text = d.get("data","")
        rv = re.search(r"Rainfall[:\s]*([0-9]+\.?[0-9]*)", text)
        tv = re.search(r"Temperature[:\s]*([0-9]+\.?[0-9]*)", text)
        yv = re.search(r"Yield[:\s]*([0-9]+\.?[0-9]*)", text)
        fv = re.search(r"Fertilizer[:\s]*(True|False)", text)
        iv = re.search(r"Irrigation[:\s]*(True|False)", text)
        sv = re.search(r"Soil[:\s]*([A-Za-z]+)", text)
        if rv and tv and yv and sv:
            rows.append({
                "Rainfall": float(rv.group(1)),
                "Temperature": float(tv.group(1)),
                "Fertilizer": 1 if fv and fv.group(1) == "True" else 0,
                "Irrigation": 1 if iv and iv.group(1) == "True" else 0,
                "Soil": sv.group(1),
                "Yield": float(yv.group(1))
            })
    return pd.DataFrame(rows)

if __name__ == "__main__":
    df = build_df()
    if df.empty:
        print("No data for evaluation")
        raise SystemExit(1)

    le = LabelEncoder()
    df['Soil_enc'] = le.fit_transform(df['Soil'])
    X = df[['Rainfall','Temperature','Fertilizer','Irrigation','Soil_enc']]
    y = df['Yield']
    Xtr, Xte, ytr, yte = train_test_split(X,y,test_size=0.2,random_state=42)
    for model_name, model in [("rf", RandomForestRegressor(n_estimators=200, random_state=42)), ("linear", LinearRegression())]:
        m = model.fit(Xtr,ytr)
        preds = m.predict(Xte)
        mae = mean_absolute_error(yte,preds)
        rmse = mean_squared_error(yte,preds, squared=False)
        r2 = r2_score(yte,preds)
        print(f"{model_name} -> MAE: {mae:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}")
