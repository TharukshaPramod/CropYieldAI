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
