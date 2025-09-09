from crewai import Crew, Task
from agents.retriever import retriever_agent
from agents.pre_processor import pre_processor_agent
from agents.predictor import predictor_agent
from agents.interpreter import interpreter_agent

pre_process_task = Task(description="Pre-process query", expected_output="Cleaned data", agent=pre_processor_agent)
retrieve_task = Task(description="Retrieve data", expected_output="Retrieved summary", agent=retriever_agent, context=[pre_process_task])
predict_task = Task(description="Predict yield", expected_output="Prediction with explanation", agent=predictor_agent, context=[retrieve_task])
interpret_task = Task(description="Interpret results", expected_output="Final explanation", agent=interpreter_agent, context=[predict_task])

crop_crew = Crew(
    agents=[pre_processor_agent, retriever_agent, predictor_agent, interpreter_agent],
    tasks=[pre_process_task, retrieve_task, predict_task, interpret_task],
    verbose=2
)