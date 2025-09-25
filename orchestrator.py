# orchestrator.py
import os
import re
from dotenv import load_dotenv
load_dotenv()

from utils.security import decrypt, decrypt_embedded_tokens
from agents.pre_processor import pre_processor_agent
from agents.retriever import retriever_agent
from agents.predictor import predictor_agent
from agents.interpreter import interpreter_agent

# Optional LLM crew integration
USE_LLM_ENV = os.getenv("USE_LLM", "false").lower() in ("1", "true", "yes")
try:
    from crew import crew  # may be None if not configured
except Exception:
    crew = None

# ----------------------------
# Deterministic Pipeline
# ----------------------------
def run_det_pipeline(query: str) -> dict:
    """Run pipeline without LLM involvement (step-by-step with encryption/decryption)."""
    # Preprocess
    enc1 = pre_processor_agent.tools[0]._run({"description": query})
    pre = decrypt(enc1)

    # Retrieve
    enc2 = retriever_agent.tools[0]._run({"description": pre})
    retrieved = decrypt(enc2)

    # Predict
    enc3 = predictor_agent.tools[0]._run({"description": pre})
    prediction = decrypt(enc3)

    # Interpret
    enc4 = interpreter_agent.tools[0]._run({"description": prediction})
    interpretation = decrypt(enc4)

    return {
        "mode": "deterministic",
        "preprocessed": pre,
        "retrieved": retrieved,
        "prediction": prediction,
        "interpretation": interpretation
    }

# ----------------------------
# LLM-Orchestrated Pipeline
# ----------------------------
def run_llm_pipeline(query: str) -> dict:
    """Run pipeline where crew (Ollama LLM) coordinates the agents.

    This function attempts to:
      - run crew.kickoff
      - decrypt any embedded encrypted tokens
      - try to extract preprocessed, retrieved, predicted, interpreted pieces
      - if interpretation isn't present, run local interpreter on discovered prediction
    """
    if crew is None:
        return {"error": "Crew/LLM not configured on this machine."}

    try:
        raw = crew.kickoff(inputs={"query": query})
        text = str(raw)
        # Replace encrypted tokens (gAAAA...) with decrypted text if present
        text = decrypt_embedded_tokens(text)

        # Best-effort extraction of pieces
        preprocessed = None
        retrieved = None
        prediction_text = None
        interpretation = None

        # attempt to extract cleaned/preprocessed block
        m_pre = re.search(r"(Cleaned[:].*?)(?=(Retrieved|Predicted|Interpretation|$))", text, re.S | re.I)
        if m_pre:
            preprocessed = m_pre.group(1).strip()

        # attempt to extract retrieval block (FAISS/TFIDF)
        m_ret = re.search(r"(Retrieved\s*\(.*?\)\s*:\s*\[.*?\])", text, re.S | re.I)
        if not m_ret:
            # generic fallback: look for 'Retrieved' until Prediction or end
            m_ret2 = re.search(r"(Retrieved[:].*?)(?=(Predicted|Interpretation|$))", text, re.S | re.I)
            if m_ret2:
                retrieved = m_ret2.group(1).strip()
        else:
            retrieved = m_ret.group(1).strip()

        # attempt to find "Predicted yield..." lines
        m_pred = re.search(r"(Predicted yield .*?:\s*[0-9]+\.?[0-9]*\s*tons?\/ha)", text, re.I)
        if m_pred:
            prediction_text = m_pred.group(1).strip()

        # attempt to find explicit interpretation text
        m_interp = re.search(r"(Interpretation[:]?.*?)(?=$)", text, re.S | re.I)
        if m_interp:
            # keep only first reasonable chunk
            interpretation = m_interp.group(1).strip()

        # If no interpretation found but a prediction was found, call local interpreter to be safe
        if interpretation is None and prediction_text is not None:
            try:
                enc_local_interp = interpreter_agent.tools[0]._run({"description": prediction_text})
                interpretation = decrypt(enc_local_interp)
            except Exception as e:
                interpretation = f"[local-interpretation-failed: {e}]"

        out = {
            "mode": "llm",
            "llm_raw": text,
            "preprocessed": preprocessed,
            "retrieved": retrieved,
            "prediction": prediction_text,
            "interpretation": interpretation
        }
        return out
    except Exception as e:
        return {"error": f"LLM run failed: {e}"}

# ----------------------------
# Unified Entry
# ----------------------------
def run_pipeline(query: str, use_llm: bool = None) -> dict:
    """Main entry point: choose deterministic or LLM pipeline."""
    if use_llm is None:
        use_llm = USE_LLM_ENV

    if use_llm:
        return run_llm_pipeline(query)
    else:
        return run_det_pipeline(query)

if __name__ == "__main__":
    test_query = "wheat yield in East with Rainfall: 492 Temperature: 15 Soil: Clay Fertilizer: True Irrigation: True"
    print("=== Deterministic Test ===")
    print(run_pipeline(test_query, use_llm=False))

    print("\n=== LLM Test (if configured) ===")
    print(run_pipeline("wheat yield in East", use_llm=True))
