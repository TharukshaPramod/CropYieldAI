# agents/interpreter.py
import os, sys, re
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from typing import Optional, Type
from pydantic import BaseModel
from crewai import Agent
from crewai.tools import BaseTool
from tools import InterpretToolSchema
from utils.security import sanitize_input, encrypt, decrypt
from utils.db import get_all_decrypted_docs

class InterpretTool(BaseTool):
    name: str = "InterpretResult"
    description: str = "Interprets predicted yields and compares to baseline"
    args_schema: Optional[Type[BaseModel]] = InterpretToolSchema

    def _run(self, prediction: Optional[dict] = None) -> str:
        q = prediction.get("description") if isinstance(prediction, dict) else prediction
        if isinstance(q, str) and q.startswith("gAAAAA"):
            try:
                q = decrypt(q)
            except:
                pass
        q = sanitize_input(str(q or ""))
        m = re.search(r"([0-9]+\.?[0-9]*)\s*tons?\/ha", q)
        predicted = float(m.group(1)) if m else None

        docs = get_all_decrypted_docs()
        yields = []
        for d in docs:
            mm = re.search(r"Yield[:\s]*([0-9]+\.?[0-9]*)", d["data"])
            if mm:
                try:
                    yields.append(float(mm.group(1)))
                except:
                    pass
        avg = (sum(yields) / len(yields)) if yields else None

        if predicted is None:
            interpretation = f"Could not parse numeric prediction. Raw: {q}"
        else:
            if avg is None:
                interpretation = f"Predicted yield: {predicted:.2f} tons/ha. No dataset baseline available."
            else:
                if predicted > avg * 1.05:
                    status = "above average"
                elif predicted < avg * 0.95:
                    status = "below average"
                else:
                    status = "around average"
                interpretation = (
                    f"Interpretation: Predicted yield {predicted:.2f} tons/ha is {status} "
                    f"(dataset avg: {avg:.2f} tons/ha). Consider checking rainfall/temperature shifts."
                )
        print(f"[Interpreter] {interpretation}")
        return encrypt(interpretation)

interpreter_agent = Agent(
    role="Interpreter Agent",
    goal="Interpret predictions for actionable insights",
    backstory="Explains model output in natural language",
    tools=[InterpretTool()]
)

if __name__ == "__main__":
    from utils.security import decrypt
    sample = {"description": "Predicted yield for wheat in East: 4.33 tons/ha"}
    enc = interpreter_agent.tools[0]._run(sample)
    print("Interpretation:", decrypt(enc))
