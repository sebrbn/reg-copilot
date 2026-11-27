from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from answer import answer

app = FastAPI(title="Reg Copilot")
INDEX = Path(__file__).parent / "static" / "index.html"


class Question(BaseModel):
    question: str


@app.get("/")
def home():
    return FileResponse(INDEX)


@app.post("/ask")
def ask(body: Question):
    q = body.question.strip()
    if not q:
        raise HTTPException(status_code=400, detail="Please enter a question.")
    if len(q) > 500:
        raise HTTPException(status_code=400, detail="Question is too long (max 500 characters).")
    try:
        text, chunks = answer(q)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    return {
        "answer": text,
        "sources": [{"page": m["page"], "text": d} for d, m in chunks],
    }