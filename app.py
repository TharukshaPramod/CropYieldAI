import os
from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import requests
import jwt
import pandas as pd

# ------------------------
# Config
# ------------------------
API_BASE = os.getenv("API_URL", "http://localhost:8000/predict")
API_URL = API_BASE
TRAIN_URL = API_BASE.replace("/predict", "/train")
STATS_URL = API_BASE.replace("/predict", "/stats")
JWT_SECRET = os.getenv("JWT_SECRET", "supersecretkey")

# ------------------------
# Streamlit setup
# ------------------------
st.set_page_config(page_title="Crop Yield Prediction", page_icon="🌾", layout="wide")
st.title("🌾 Crop Yield Prediction — Advanced UI")

# ------------------------
# Sidebar
# ------------------------
st.sidebar.header("Options")
use_llm = st.sidebar.checkbox("Use LLM (Crew / Ollama)", value=False)
model_choice = st.sidebar.selectbox("Predictor model (runtime override)", ["rf", "linear"])
st.sidebar.markdown("---")

# ------------------------
# Input form
# ------------------------
crop = st.text_input("Crop", "wheat")
location = st.text_input("Location / Region", "East")
rainfall = st.number_input("Rainfall (mm)", value=492.0, format="%.2f")
temperature = st.number_input("Temperature (°C)", value=15.0, format="%.2f")
soil = st.text_input("Soil type", "Clay")
fert = st.selectbox("Fertilizer used?", ["True", "False"])
irrig = st.selectbox("Irrigation used?", ["True", "False"])

use_structured = st.checkbox("Build query from structured fields", value=True)
if use_structured:
    query = (
        f"{crop} yield in {location} with "
        f"Rainfall: {rainfall} Temperature: {temperature} "
        f"Soil: {soil} Fertilizer: {fert} Irrigation: {irrig}"
    )
else:
    query = st.text_input("Free-form query", "wheat yield in East")

# ------------------------
# Prediction
# ------------------------
if st.button("Predict"):
    try:
        token = jwt.encode({"user": "demo"}, JWT_SECRET, algorithm="HS256")
        if isinstance(token, bytes):  # PyJWT may return bytes
            token = token.decode("utf-8")

        headers = {"Authorization": f"Bearer {token}"}
        payload = {"query": query, "use_llm": use_llm, "model": model_choice}

        r = requests.post(API_URL, json=payload, headers=headers, timeout=30)
        r.raise_for_status()
        res = r.json().get("result")

        st.subheader("Raw result")
        st.write(res)

        if isinstance(res, dict):
            if "prediction" in res:
                st.subheader("Prediction")
                st.write(res["prediction"])
            if "interpretation" in res:
                st.subheader("Interpretation")
                st.write(res["interpretation"])
    except Exception as e:
        st.error(f"❌ Request failed: {e}")

# ------------------------
# Sidebar actions
# ------------------------
st.sidebar.markdown("---")
if st.sidebar.button("🔄 Retrain model (train on DB)"):
    try:
        token = jwt.encode({"user": "demo"}, JWT_SECRET, algorithm="HS256")
        if isinstance(token, bytes):
            token = token.decode("utf-8")

        headers = {"Authorization": f"Bearer {token}"}
        with st.spinner("Retraining model..."):
            r = requests.post(TRAIN_URL, headers=headers, timeout=120)
            r.raise_for_status()
            st.success("✅ Retrain finished")
            st.json(r.json())
    except Exception as e:
        st.error(f"❌ Retrain failed: {e}")

if st.sidebar.button("📊 Load dataset stats"):
    try:
        token = jwt.encode({"user": "demo"}, JWT_SECRET, algorithm="HS256")
        if isinstance(token, bytes):
            token = token.decode("utf-8")

        headers = {"Authorization": f"Bearer {token}"}
        r = requests.get(STATS_URL, headers=headers, timeout=20)
        r.raise_for_status()
        st.sidebar.write(r.json())
    except Exception as e:
        st.sidebar.error(f"❌ Stats request failed: {e}")
