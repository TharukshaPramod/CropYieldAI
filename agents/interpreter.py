# agents/interpreter.py
import os, sys, re
from typing import Optional, Type, Any

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

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

    def _run(self, prediction: Optional[Any] = None, baseline: Optional[float] = None, units: str = "tons/ha") -> str:
        if not prediction:
            return encrypt("Error: No prediction provided to interpreter.")

        predictions = {}
        # Case 1: dict of {crop: value}
        if isinstance(prediction, dict) and "description" not in prediction:
            try:
                predictions = {str(k): float(v) for k, v in prediction.items()}
            except Exception:
                return encrypt("Error: Invalid prediction dict for interpretation.")
        else:
            # Case 2: dict with description or raw string
            q = prediction.get("description") if isinstance(prediction, dict) else prediction
            if not isinstance(q, str):
                return encrypt("Error: Prediction format not understood.")
            if q.startswith("gAAAAA"):  # encrypted
                try:
                    q = decrypt(q)
                except Exception:
                    return encrypt("Error: Could not decrypt interpreter input.")
            q = sanitize_input(q or "")
            m = re.search(r"([0-9]+\.?[0-9]*)\s*tons?\/ha", q)
            if m:
                predictions = {"yield": float(m.group(1))}
            else:
                return encrypt("Error: No numeric yield found in prediction.")

        # Compute baseline
        if baseline is None:
            docs = get_all_decrypted_docs()
            yields = []
            for d in docs:
                mm = re.search(r"Yield[:\s]*([0-9]+\.?[0-9]*)", d.get("data", ""))
                if mm:
                    try:
                        yields.append(float(mm.group(1)))
                    except:
                        pass
            baseline_val = (sum(yields) / len(yields)) if yields else None
        else:
            baseline_val = baseline

        # Interpret
        interpretations = []
        for crop, val in predictions.items():
            val = float(val)
            if baseline_val is None:
                status_text = "No dataset baseline available."
            else:
                if val > baseline_val * 1.05:
                    status_text = "above average"
                elif val < baseline_val * 0.95:
                    status_text = "below average"
                else:
                    status_text = "around average"
            interpretations.append(f"{crop}: {val:.2f} {units}, {status_text}")

        final_output = " | ".join(interpretations)
        final_output += " Consider verifying rainfall, temperature, fertilizer and irrigation inputs."
        print(f"[Interpreter] {final_output}")
        return encrypt(final_output)

interpreter_agent = Agent(
    role="Interpreter Agent",
    goal="Interpret predictions for actionable insights",
    backstory="Explains model output in natural language, based strictly on ML predictions.",
    tools=[InterpretTool()]
)

if __name__ == "__main__":
    sample = {"wheat": 4.33, "corn": 5.12}
    enc = interpreter_agent.tools[0]._run(prediction=sample, baseline=4.65)
    print("Interpretation:", decrypt(enc))
