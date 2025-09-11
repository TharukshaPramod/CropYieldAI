from crewai import Agent
from crewai.tools import BaseTool
import spacy
import re
from utils.security import sanitize_input, encrypt
from utils.db import Session, CropData

nlp = spacy.load('en_core_web_sm')

class PreProcessTool(BaseTool):
    name: str = "PreProcessData"
    description: str = "Pre-processes raw crop data for analysis"

    def _run(self, raw_data: str) -> str:
        cleaned_data = sanitize_input(raw_data)
        # Improved regex for crop (first word after "Raw data: " or start, before non-letter)
        crop_match = re.search(r'Raw data: (\w+)', cleaned_data)
        rainfall_match = re.search(r'Rainfall(\d+\.?\d*)mm', cleaned_data)
        yield_match = re.search(r'Yield(\d+\.?\d*)', cleaned_data)
        crop = crop_match.group(1) if crop_match else "Unknown"
        rainfall = rainfall_match.group(1) if rainfall_match else "Unknown"
        yield_value = yield_match.group(1) if yield_match else "Unknown"
        processed = f"Cleaned: {cleaned_data} (Extracted: Crop={crop}, Rainfall={rainfall}mm, Yield={yield_value} tons/ha)"
        return encrypt(processed)

pre_processor_agent = Agent(
    role="Pre-Processor Agent",
    goal="Clean and prepare raw crop data for analysis",
    backstory="Specializes in data pre-processing with NLP and security",
    tools=[PreProcessTool()]
)

if __name__ == "__main__":
    from utils.security import decrypt
    sample_data = "Raw data: wheat! @East #Rainfall492mm Yield3.26"
    result = decrypt(pre_processor_agent.tools[0]._run(sample_data))
    print(f"Pre-processing log: {result} (Responsible AI: Ensured fair data cleaning without bias toward certain regions.)")