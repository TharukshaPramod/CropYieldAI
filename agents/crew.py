from crewai import Crew, Task
from crewai.llm import LLM
from transformers import pipeline
import torch
from agents.retriever import retriever_agent
from agents.pre_processor import pre_processor_agent
from agents.predictor import predictor_agent
from agents.interpreter import interpreter_agent

class CustomHFLLM(LLM):
    def __init__(self):
        self.generator = pipeline("text-generation", model="distilgpt2", device=0 if torch.cuda.is_available() else -1)
        self.model = "distilgpt2"  # Dummy attribute to satisfy crewai

    def call(self, messages, **kwargs):
        prompt = messages[-1]['content'] if messages else "Generate a response."
        response = self.generator(prompt, max_length=len(prompt.split()) + 20, num_return_sequences=1, do_sample=True, temperature=0.7)
        return response[0]['generated_text']  # Return string directly

hf_llm = CustomHFLLM()

# Assign local LLM to agents
pre_processor_agent.llm = hf_llm
retriever_agent.llm = hf_llm
predictor_agent.llm = hf_llm
interpreter_agent.llm = hf_llm

pre_process_task = Task(description="Pre-process input query", expected_output="Cleaned data", agent=pre_processor_agent)
retrieve_task = Task(description="Retrieve data", expected_output="Retrieved summary", agent=retriever_agent, context=[pre_process_task])
predict_task = Task(description="Predict yield", expected_output="Prediction", agent=predictor_agent, context=[retrieve_task])
interpret_task = Task(description="Interpret results", expected_output="Final explanation", agent=interpreter_agent, context=[predict_task])

crop_crew = Crew(
    agents=[pre_processor_agent, retriever_agent, predictor_agent, interpreter_agent],
    tasks=[pre_process_task, retrieve_task, predict_task, interpret_task],
    verbose=True
)

if __name__ == "__main__":
    result = crop_crew.kickoff(inputs={"query": "wheat yield in East"})
    print(result)