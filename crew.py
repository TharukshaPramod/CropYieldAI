# crew.py
import os
from dotenv import load_dotenv
load_dotenv()

from crewai import Crew, Task
from crewai.llm import LLM
from agents.pre_processor import pre_processor_agent
from agents.retriever import retriever_agent
from agents.predictor import predictor_agent
from agents.interpreter import interpreter_agent

# ----------------------------
# Ollama Configuration
# ----------------------------
OLLAMA_BASE = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_KEY = os.getenv("OLLAMA_API_KEY", "ollama")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3:latest")


