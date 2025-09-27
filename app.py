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


