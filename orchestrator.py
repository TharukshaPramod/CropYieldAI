# orchestrator.py
import os
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
    from crew import crew
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
        "preprocessed": pre,
        "retrieved": retrieved,
        "prediction": prediction,
        "interpretation": interpretation
    }

# ----------------------------
# LLM-Orchestrated Pipeline
# ----------------------------
def run_llm_pipeline(query: str) -> dict:
    """Run pipeline where crew (Ollama LLM) coordinates the agents."""
    if crew is None:
        return {"error": "Crew/LLM not configured on this machine."}

    try:
        raw = crew.kickoff(inputs={"query": query})
        text = str(raw)
        text = decrypt_embedded_tokens(text)  # Replace encrypted tokens if present
        return {"llm_raw": text}
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
