from crewai import Agent, Tool
from sklearn.linear_model import LinearRegression
import pandas as pd
from utils.security import decrypt
import shap

def predict_tool(encrypted_data: str) -> dict:
    data = decrypt(encrypted_data)
    df = pd.read_json(data)
    X = df[['rain', 'ph']]
    y = df['yield']
    model = LinearRegression().fit(X, y)
    prediction = model.predict([[400, 6.5]])[0]

    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)
    explanation = str(shap_values)

    return {"prediction": prediction, "explanation": explanation}

predictor_agent = Agent(
    role="Predictor Agent",
    goal="Predict crop yields with ML",
    backstory="Uses scikit-learn and SHAP for explainable predictions",
    tools=[Tool(name="PredictYield", func=predict_tool, description="ML prediction tool")]
)