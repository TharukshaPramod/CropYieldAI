from crewai import Agent, Tool
from openai import OpenAI

client = OpenAI()

def interpret_tool(prediction_data: dict) -> str:
    prompt = f"Interpret yield prediction {prediction_data['prediction']}: {prediction_data['explanation']}. Make it user-friendly and check ethics."
    response = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}])
    return response.choices[0].message.content

interpreter_agent = Agent(
    role="Interpreter Agent",
    goal="Explain results in natural language",
    backstory="Uses LLM for clear, ethical interpretations",
    tools=[Tool(name="Interpret", func=interpret_tool, description="Result explanation tool")]
)