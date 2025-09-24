# orchestrator.py
from utils.security import decrypt
from agents.pre_processor import pre_processor_agent
from agents.retriever import retriever_agent
from agents.predictor import predictor_agent
from agents.interpreter import interpreter_agent

def run_pipeline(query: str) -> str:
    try:
        enc1 = pre_processor_agent.tools[0]._run({"description": query})
        enc2 = retriever_agent.tools[0]._run({"description": enc1})
        enc3 = predictor_agent.tools[0]._run({"description": enc2})
        enc4 = interpreter_agent.tools[0]._run({"description": enc3})
        return decrypt(enc4)
    except Exception as e:
        return f"Pipeline failed: {e}"

if __name__ == "__main__":
    print(run_pipeline("wheat yield in East"))
