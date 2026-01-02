from fastapi import FastAPI
import requests
import os

app = FastAPI()

VLLM_URL = os.getenv("VLLM_URL", "http://vllm:8000/generate")

@app.get("/")
def root():
    return {"message": "LLM API Running"}

@app.get("/chat")
def chat(q: str):
    payload = {
        "prompt": q,
        "max_tokens": 64,
    }
    r = requests.post(VLLM_URL, json=payload)
    return r.json()
