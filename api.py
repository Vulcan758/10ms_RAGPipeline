from fastapi import FastAPI
from pydantic import BaseModel
from BanglaRAG import BanglaRAG  # Your existing class

pdf_path = "book/HSC26-Bangla1st-Paper.pdf"
data_path = "data/"

app = FastAPI()
rag = BanglaRAG(the_pdf_path=pdf_path)  # Initialize once

class QueryRequest(BaseModel):
    text: str

@app.post("/ask")
async def ask_question(request: QueryRequest):
    """Endpoint for Bengali Q&A"""
    response = rag.rag_pipeline(request.text)
    return {"answer": response.strip()}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}