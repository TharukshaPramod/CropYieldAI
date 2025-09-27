import os
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException, Request, Depends
from pydantic import BaseModel
import jwt

from orchestrator import run_pipeline
from agents.predictor import retrain_model
from utils.db import get_all_decrypted_docs

JWT_SECRET = os.getenv("JWT_SECRET", "supersecretkey")

app = FastAPI(title="🌾 Crop Yield AI API")
