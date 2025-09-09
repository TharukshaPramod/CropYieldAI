from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import jwt
from agents.crew import crop_crew
from dotenv import load_dotenv
import os

load_dotenv()
app = FastAPI()
security = HTTPBearer()

class Query(BaseModel):
    query: str

@app.post("/predict")
async def predict(query: Query, credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        jwt.decode(credentials.credentials, os.getenv('JWT_SECRET'), algorithms=["HS256"])
    except:
        raise HTTPException(status_code=401, detail="Unauthorized")
    result = crop_crew.kickoff(inputs={"query": query.query})
    return {"result": result}