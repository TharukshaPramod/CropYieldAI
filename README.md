## Crop Yield AI — Advanced Multi‑Agent with Ollama Support

This project predicts crop yields using a robust deterministic pipeline or an LLM‑orchestrated pipeline (CrewAI + Ollama). It exposes a FastAPI backend and a Streamlit UI with async jobs, progress, and model retraining.

### Key Features
- Deterministic pipeline with pre‑processing, retrieval (FAISS/TF‑IDF), prediction (RF/Linear), and interpretation
- Optional CrewAI pipeline orchestrated by an Ollama local LLM
- FastAPI endpoints with timeouts, async job API, and CORS
- Streamlit UI with async polling, live status, and dataset stats
- SQLite by default, Postgres optional; data encrypted at rest

---

## 1) Prerequisites
- Python 3.10–3.12 recommended
- Windows 10/11 or Linux/macOS
- Git
- Ollama installed and running locally (optional but recommended)

Install Ollama and a model (example):

```bash
ollama pull llama3:latest
ollama serve
```

By default Ollama serves at `http://localhost:11434`.

---

## 2) Setup

Create a virtual environment and install dependencies:

```bash
python -m venv crop-ai-env
./crop-ai-env/Scripts/Activate.ps1  # Windows PowerShell
pip install -U pip
pip install -r requirements.txt
```

Create a `.env` in the project root with defaults:

```env
# API/UI
API_URL=http://localhost:8000/predict
JWT_SECRET=supersecretkey
RUN_PIPELINE_TIMEOUT=25

# Database
DATABASE_URL=sqlite:///crop.db
ENCRYPT_KEY=

# LLM / CrewAI
USE_LLM=false
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3:latest
OLLAMA_API_KEY=
```

Load sample data into the database (optional):

```bash
python -m utils.db --csv data/crop_yield_data_sampled.csv
```

---

## 3) Running

Start the API server:

```bash
uvicorn main_api:app --reload --host 0.0.0.0 --port 8000
```

Start the Streamlit UI (in a second terminal):

```bash
streamlit run app.py
```

Open the UI in your browser. Toggle "Async mode" for background jobs and "Use LLM" to enable the Ollama‑driven Crew pipeline (requires Ollama running).

---

## 4) Using the LLM (Ollama)
1. Ensure `ollama serve` is running and the model is available (e.g., `ollama pull llama3:latest`).
2. Set in `.env`:
   - `USE_LLM=true`
   - `OLLAMA_BASE_URL=http://localhost:11434`
   - `OLLAMA_MODEL=llama3:latest`
3. Restart API and UI.

If CrewAI cannot connect to Ollama, the system falls back to the deterministic pipeline.

---

## 5) Troubleshooting
- LLM timeout: ensure Ollama is running and reachable. Use async mode to avoid UI blocking.
- No data: run the sample data loader or update `DATABASE_URL`.
- Encryption: set `ENCRYPT_KEY` for stable decryption between runs.
- Vector store deps: FAISS/HF failover to TF‑IDF automatically.

---

## 6) Tests

```bash
pytest -q
```