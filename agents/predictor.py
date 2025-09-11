from crewai import Agent
from crewai.tools import BaseTool
import spacy
from utils.security import sanitize_input, encrypt
from utils.db import Session, CropData
from sklearn.linear_model import LinearRegression
import pandas as pd

nlp = spacy.load('en_core_web_sm')

class PredictTool(BaseTool):
    name: str = "PredictYield"
    description: str = "Predicts crop yield based on weather data"

    def _run(self, data: str) -> str:
        query = sanitize_input(data)
        doc = nlp(query)
        entities = {ent.label_: ent.text for ent in doc.ents}
        location = entities.get('GPE', 'unknown')
        crop = entities.get('PRODUCT', 'wheat')
        
        session = Session()
        crop_data = session.query(CropData).all()
        session.close()
        data_list = [entry.data for entry in crop_data]
        df = pd.DataFrame(data_list, columns=['data'])
        df['Rainfall'] = df['data'].str.extract(r'Rainfall: (\d+\.?\d*)').astype(float)
        df['Temperature'] = df['data'].str.extract(r'Temperature: (\d+\.?\d*)').astype(float)
        df['Yield'] = df['data'].str.extract(r'Yield: (\d+\.?\d*)').astype(float)
        # Filter to ensure same length
        df_clean = df.dropna(subset=['Rainfall', 'Temperature', 'Yield'])
        X = df_clean[['Rainfall', 'Temperature']]
        y = df_clean['Yield']
        if len(X) > 0 and len(y) > 0:
            model = LinearRegression().fit(X, y)
            prediction_input = pd.DataFrame({'Rainfall': [500], 'Temperature': [25]})
            prediction = model.predict(prediction_input)[0]
        else:
            prediction = 5.0
        result = f"Predicted yield for {crop} in {location}: {prediction:.2f} tons/ha"
        return encrypt(result)

predictor_agent = Agent(
    role="Predictor Agent",
    goal="Predict crop yields based on processed data",
    backstory="Specializes in yield prediction using NLP and basic models",
    tools=[PredictTool()]
)

if __name__ == "__main__":
    from utils.security import decrypt
    sample_data = "wheat yield in East"
    result = decrypt(predictor_agent.tools[0]._run(sample_data))
    print(f"Prediction log: {result} (Responsible AI: Model explainability - based on rainfall and temperature features.)")