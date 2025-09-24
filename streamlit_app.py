# streamlit_app.py
import os
from dotenv import load_dotenv
load_dotenv()
import streamlit as st
import requests
import jwt

API_URL = os.getenv("API_URL", "http://localhost:8000/predict")
JWT_SECRET = os.getenv("JWT_SECRET", "supersecretkey")

st.set_page_config(page_title="Crop Yield Prediction", page_icon="🌾")
st.title("Crop Yield Prediction — Demo")

query = st.text_input("Enter query (e.g., wheat yield in East)", "wheat yield in East")
if st.button("Predict"):
    try:
        token = jwt.encode({"user": "demo"}, JWT_SECRET, algorithm="HS256")
        headers = {"Authorization": f"Bearer {token}"}
        r = requests.post(API_URL, json={"query": query}, headers=headers, timeout=20)
        if r.status_code == 200:
            st.success("Prediction + Interpretation:")
            st.write(r.json().get("result"))
        else:
            st.error(f"API error {r.status_code}: {r.text}")
    except Exception as e:
        st.error(f"Request failed: {e}")
