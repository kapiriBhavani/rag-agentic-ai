"""
FastAPI Application for the RAG Agentic AI Chatbot.

Exposes a REST API endpoint for querying the RAG pipeline:
- POST /chat   : Submit a query and receive an answer with context and confidence
- GET  /health : Health check endpoint
"""

import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables before importing project modules
load_dotenv()

from src.graph import build_rag_graph
from src.config import PINECONE_INDEX_NAME, print_config

# ---------------------------------------------------------------------------
# FastAPI Application
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Agentic AI RAG Chatbot API",
    description="A RAG-based chatbot that answers questions from the Agentic AI eBook.",
    version="1.0.0",
)


# ---------------------------------------------------------------------------
# Request / Response Models
# ---------------------------------------------------------------------------
class QueryRequest(BaseModel):
    """Request model for the /chat endpoint."""
    query: str

    class Config:
        json_schema_extra = {
            "example": {
                "query": "What is Agentic AI according to the eBook?"
            }
        }


class QueryResponse(BaseModel):
    """Response model for the /chat endpoint."""
    final_answer: str
    retrieved_context: list[str]
    confidence_score: float


# ---------------------------------------------------------------------------
# Initialize the RAG Graph on startup
# ---------------------------------------------------------------------------
print_config()
print("\nInitializing RAG Graph...")
graph = build_rag_graph(index_name=PINECONE_INDEX_NAME)
print("✅ RAG Graph initialized successfully!\n")


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "Agentic AI RAG Chatbot"}


@app.post("/chat", response_model=QueryResponse)
async def chat_endpoint(request: QueryRequest):
    """
    Process a user query through the RAG pipeline.

    Accepts a question, retrieves relevant context from the Agentic AI eBook,
    and generates a grounded answer with a confidence score.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        # Invoke the LangGraph RAG workflow
        initial_state = {
            "question": request.query,
            "context": [],
            "answer": "",
            "score": 0.0,
        }
        result = graph.invoke(initial_state)

        return QueryResponse(
            final_answer=result["answer"],
            retrieved_context=result["context"],
            confidence_score=result["score"],
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}",
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
