from crewai import Agent
from crewai.tools import BaseTool
from langchain.schema import Document
import spacy
from utils.security import sanitize_input, encrypt
from utils.db import Session, CropData

nlp = spacy.load('en_core_web_sm')

class PreProcessTool(BaseTool):
    name: str = "PreProcessData"
    description: str = "Pre-processes raw crop data for analysis"

    def _run(self, raw_data: str) -> str:
        cleaned_data = sanitize_input(raw_data)
        # Add pre-processing logic (e.g., tokenization or basic cleaning)
        processed = f"Cleaned: {cleaned_data}"
        print(f"Pre-processing log: {processed}")
        return encrypt(processed)

pre_processor_agent = Agent(
    role="Pre-Processor Agent",
    goal="Clean and prepare raw crop data for analysis",
    backstory="Specializes in data pre-processing with NLP and security",
    tools=[PreProcessTool()]
)

if __name__ == "__main__":
    from utils.security import decrypt
    sample_data = "Raw data: wheat! @California #yield5"
    print(decrypt(pre_processor_agent.tools[0]._run(sample_data)))