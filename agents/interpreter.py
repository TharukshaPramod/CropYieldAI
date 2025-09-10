from crewai import Agent
from crewai.tools import BaseTool
from langchain.schema import Document
import spacy
from utils.security import sanitize_input, encrypt
from utils.db import Session, CropData

nlp = spacy.load('en_core_web_sm')

class InterpretTool(BaseTool):
    name: str = "InterpretResult"
    description: str = "Interprets crop yield predictions"

    def _run(self, prediction: str) -> str:
        query = sanitize_input(prediction)
        doc = nlp(query)
        entities = {ent.label_: ent.text for ent in doc.ents}
        location = entities.get('GPE', 'unknown')
        yield_value = "5 tons/ha"  # Placeholder from prediction
        interpretation = f"Interpretation for {location}: Yield of {yield_value} is above average due to good rainfall."
        print(f"Interpretation log: {interpretation}")
        return encrypt(interpretation)

interpreter_agent = Agent(
    role="Interpreter Agent",
    goal="Interpret crop yield predictions for actionable insights",
    backstory="Specializes in explaining predictions with NLP and domain knowledge",
    tools=[InterpretTool()]
)

if __name__ == "__main__":
    from utils.security import decrypt
    sample_prediction = "Predicted yield for California: 5 tons/ha"
    print(decrypt(interpreter_agent.tools[0]._run(sample_prediction)))