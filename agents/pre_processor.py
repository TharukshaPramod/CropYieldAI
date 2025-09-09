from crewai import Agent, Tool
import pandas as pd
from utils.security import sanitize_input

def pre_process_tool(raw_data: str) -> str:
    sanitized = sanitize_input(raw_data)
    df = pd.read_csv(pd.compat.StringIO(sanitized)) if 'csv' in sanitized else pd.DataFrame()
    return df.to_json()

pre_processor_agent = Agent(
    role="Pre-Processor Agent",
    goal="Clean and prepare data",
    backstory="Uses pandas for data normalization",
    tools=[Tool(name="PreProcess", func=pre_process_tool, description="Data cleaning tool")]
)