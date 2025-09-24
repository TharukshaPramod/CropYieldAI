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

OLLAMA_BASE = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_KEY = os.getenv("OLLAMA_API_KEY", "ollama")

ollama_llm = LLM(model="ollama/llama3:latest", base_url=OLLAMA_BASE, api_key=OLLAMA_KEY)

pre_processor_agent.llm = ollama_llm
retriever_agent.llm = ollama_llm
predictor_agent.llm = ollama_llm
interpreter_agent.llm = ollama_llm

pre_task = Task(description="Pre-process the query", expected_output="Cleaned data", agent=pre_processor_agent)
ret_task = Task(description="Retrieve data", expected_output="Retrieved data", agent=retriever_agent, context=[pre_task])
pred_task = Task(description="Predict yield", expected_output="Prediction", agent=predictor_agent, context=[ret_task])
interp_task = Task(description="Interpretation", expected_output="Interpretation", agent=interpreter_agent, context=[pred_task])

crew = Crew(
    agents=[pre_processor_agent, retriever_agent, predictor_agent, interpreter_agent],
    tasks=[pre_task, ret_task, pred_task, interp_task],
    verbose=True
)

if __name__ == "__main__":
    from orchestrator import run_pipeline
    print("Deterministic pipeline run (no LLM):")
    print(run_pipeline("wheat yield in East"))
    # Optionally run the LLM-driven crew kickoff:
    # print("LLM-driven kickoff:")
    # print(crew.kickoff(inputs={"query": "wheat yield in East"}))
