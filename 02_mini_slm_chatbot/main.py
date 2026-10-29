from fastapi import FastAPI
from pydantic import BaseModel
from chatbot import generate_response

app = FastAPI()

@app.get("/")
def home():
    return {"message": "First FastAPI Application"}

class ChatRequest(BaseModel):
    message : str
    

@app.post("/chat")
def  chat(request: ChatRequest):
    answer = generate_response(request.message)
    return {"response" : answer}


