from crewai import Agent
from crewai.tools import BaseTool
from langchain.schema import Document
import spacy
from utils.security import sanitize_input, encrypt
from utils.db import Session, CropData

nlp = spacy.load('en_core_web_sm')

class PredictTool(BaseTool):
    name: str = "PredictYield"
    description: str = "Predicts crop yield based on location and data"

    def _run(self, data: str) -> str:
        query = sanitize_input(data)
        doc = nlp(query)
        entities = {ent.label_: ent.text for ent in doc.ents}
        location = entities.get('GPE', 'unknown')
        # Simple prediction logic (placeholder)
        prediction = f"Predicted yield for {location}: 5 tons/ha"
        print(f"Prediction log: {prediction}")
        return encrypt(prediction)

predictor_agent = Agent(
    role="Predictor Agent",
    goal="Predict crop yields based on processed data",
    backstory="Specializes in yield prediction using NLP and basic models",
    tools=[PredictTool()]
)

if __name__ == "__main__":
    from utils.security import decrypt
    sample_data = "wheat yield in California"
    print(decrypt(predictor_agent.tools[0]._run(sample_data)))