# utils/evaluate.py
from utils.db import get_all_decrypted_docs
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

def build_df_from_docs():
    docs = get_all_decrypted_docs()
    rows = []
    for d in docs:
        text = d['data']
        import re
        rv = re.search(r'Rainfall[:\s]*([0-9]+\.?[0-9]*)', text)
        tv = re.search(r'Temperature[:\s]*([0-9]+\.?[0-9]*)', text)
        fv = re.search(r'Fertilizer[:\s]*(True|False)', text)
        iv = re.search(r'Irrigation[:\s]*(True|False)', text)
        yv = re.search(r'Yield[:\s]*([0-9]+\.?[0-9]*)', text)
        sv = re.search(r'Soil[:\s]*([A-Za-z]+)', text)
        if rv and tv and yv and sv:
            rows.append({
                'Rainfall': float(rv.group(1)),
                'Temperature': float(tv.group(1)),
                'Fertilizer': 1 if fv and fv.group(1) == "True" else 0,
                'Irrigation': 1 if iv and iv.group(1) == "True" else 0,
                'Soil': sv.group(1),
                'Yield': float(yv.group(1))
            })
    return pd.DataFrame(rows)

if __name__ == "__main__":
    df = build_df_from_docs()
    if df.empty:
        print("No data found.")
        raise SystemExit(1)

    le = LabelEncoder()
    df['Soil_enc'] = le.fit_transform(df['Soil'])
    X = df[['Rainfall','Temperature','Fertilizer','Irrigation','Soil_enc']]
    y = df['Yield']

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
    model = LinearRegression().fit(X_train,y_train)
    preds = model.predict(X_test)

    print("MAE:", mean_absolute_error(y_test, preds))
    print("RMSE:", mean_squared_error(y_test, preds, squared=False))
