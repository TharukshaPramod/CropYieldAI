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

# Build an LLM client only if the env says so / if it can be instantiated.
ollama_llm = None
crew = None

try:
    # Try to create the LLM object (may raise if crewai.llm.LLM not available or unreachable)
    ollama_llm = LLM(
        model=f"ollama/{OLLAMA_MODEL}",
        base_url=OLLAMA_BASE,
        api_key=OLLAMA_KEY
    )
    print("[crew] Ollama LLM client created.")
except Exception as e:
    ollama_llm = None
    print(f"[crew] Ollama LLM not configured or instantiation failed: {e}")

# Only assign the LLM to agents if we have one
if ollama_llm is not None:
    for agent in (pre_processor_agent, retriever_agent, predictor_agent, interpreter_agent):
        try:
            agent.llm = ollama_llm
        except Exception as e:
            print(f"[crew] Warning: could not attach llm to agent {agent}: {e}")

# ----------------------------
# Define Tasks
# ----------------------------
pre_task = Task(
    description="Pre-process the user query and clean it for downstream agents.",
    expected_output="Cleaned query string",
    agent=pre_processor_agent
)

ret_task = Task(
    description="Retrieve relevant data based on the cleaned query.",
    expected_output="Retrieved dataset or matching records",
    agent=retriever_agent,
    context=[pre_task]
)

pred_task = Task(
    description="Predict crop yield from retrieved and pre-processed inputs.",
    expected_output="Predicted yield with units (e.g., tons/ha)",
    agent=predictor_agent,
    context=[ret_task]
)

interp_task = Task(
    description="Interpret the prediction and provide insights (above/below average, risks, suggestions).",
    expected_output="Human-readable interpretation",
    agent=interpreter_agent,
    context=[pred_task]
)

# ----------------------------
# Initialize Crew (only if LLM available)
# ----------------------------
if ollama_llm is not None:
    try:
        # Limit iterations to prevent infinite looping
        max_iters = int(os.getenv("MAX_ITERATIONS", "8"))
        crew = Crew(
            agents=[pre_processor_agent, retriever_agent, predictor_agent, interpreter_agent],
            tasks=[pre_task, ret_task, pred_task, interp_task],
            verbose=True,
            max_iterations=max_iters
        )
        print("[crew] Crew initialized with Ollama LLM.")
    except Exception as e:
        crew = None
        print(f"[crew] Failed to initialize Crew: {e}")
else:
    crew = None
    print("[crew] Crew not initialized because no Ollama LLM is configured.")

if __name__ == "__main__":
    from orchestrator import run_pipeline

    print("=== Deterministic pipeline run (no LLM) ===")
    print(run_pipeline("wheat yield in East"))

    print("\n=== LLM-driven Crew kickoff (only if Ollama configured) ===")
    if crew:
        try:
            result = crew.kickoff(inputs={"query": "wheat yield in East"})
            print(result)
        except Exception as e:
            print("[crew] kickoff failed:", e)
    else:
        print("Crew not initialized (no LLM).")
