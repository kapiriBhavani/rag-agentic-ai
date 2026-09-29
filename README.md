# 🤖 RAG-based Agentic AI Chatbot

A **Retrieval-Augmented Generation (RAG)** chatbot built with **Python**, **LangGraph**, **Pinecone**, **FastAPI**, and **Streamlit**. The chatbot answers questions strictly grounded in the **Agentic AI eBook**, ensuring accurate, context-based responses.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Setup & Installation](#setup--installation)
- [Running the Ingestion Pipeline](#running-the-ingestion-pipeline)
- [Running the Application](#running-the-application)
- [API Documentation](#api-documentation)
- [Sample Test Queries](#sample-test-queries)
- [Configuration Options](#configuration-options)

---

## Overview

This project implements a RAG pipeline that:

1. **Ingests** the Agentic AI eBook PDF — parses, chunks, and stores embeddings in Pinecone
2. **Retrieves** the most relevant document chunks for any user query
3. **Generates** a grounded answer using an LLM, strictly based on the retrieved context
4. **Exposes** results via a FastAPI REST API and a Streamlit web UI

### Key Features

- **Strict Grounding**: The LLM only answers from the document — refuses out-of-scope questions
- **Confidence Scoring**: Each response includes a confidence score (0.0 to 1.0)
- **Dual Interface**: Both REST API (FastAPI) and Web UI (Streamlit)
- **Flexible Providers**: Supports OpenAI or HuggingFace embeddings, and OpenAI or Ollama LLMs

---

## Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────────┐
│           LangGraph Workflow            │
│                                         │
│  ┌──────────┐       ┌──────────────┐    │
│  │ RETRIEVE │──────▶│   GENERATE   │    │
│  │  Node    │       │    Node      │    │
│  └────┬─────┘       └──────┬───────┘    │
│       │                    │            │
│       ▼                    ▼            │
│  ┌──────────┐       ┌──────────────┐    │
│  │ Pinecone │       │  LLM (GPT/   │    │
│  │ Vector   │       │   Ollama)    │    │
│  │  Store   │       └──────────────┘    │
│  └──────────┘                           │
└─────────────────────────────────────────┘
    │
    ▼
┌──────────────────────┐
│  Response:           │
│  - Answer            │
│  - Context Chunks    │
│  - Confidence Score  │
└──────────────────────┘
```

**Graph Flow:** `START → retrieve → generate → END`

---

## Project Structure

```
rag-agentic-ai/
│
├── data/
│   └── Ebook-Agentic-AI.pdf        # Source document (60 pages)
│
├── src/
│   ├── __init__.py
│   ├── config.py                    # Environment setup & provider selection
│   ├── ingestion.py                 # PDF loading, chunking & Pinecone indexing
│   └── graph.py                     # LangGraph RAG workflow definition
│
├── app.py                           # FastAPI REST API application
├── streamlit_app.py                 # Streamlit web UI
├── tests_sample_queries.py          # Benchmark test queries (6 queries)
├── requirements.txt                 # Python dependencies
├── .env.example                     # Environment variable template
├── .gitignore                       # Git ignore rules
└── README.md                        # This file
```

---

## Prerequisites

- **Python** 3.10 or higher
- **Pinecone** free tier account ([sign up here](https://app.pinecone.io/))
- **OpenAI** API key ([get one here](https://platform.openai.com/api-keys)) — *or use free alternatives (see Configuration)*

---

## Setup & Installation

### 1. Clone the Repository

```bash
git clone https://github.com/kapiriBhavani/rag-agentic-ai.git
cd rag-agentic-ai
```

### 2. Create & Activate Virtual Environment

```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate         # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

```bash
cp .env.example .env
```

Edit the `.env` file and add your API keys:

```env
OPENAI_API_KEY=sk-your-openai-key-here
PINECONE_API_KEY=your-pinecone-key-here
PINECONE_INDEX_NAME=agentic-ai-index
```

> **Note:** If you don't have an OpenAI API key, set `EMBEDDING_PROVIDER=huggingface` and `LLM_PROVIDER=ollama` to use free alternatives.

---

## Running the Ingestion Pipeline

Before using the chatbot, you must ingest the PDF document into Pinecone:

```bash
python -m src.ingestion
```

This will:
1. Load the 60-page Agentic AI eBook PDF
2. Split it into ~120+ text chunks (1000 chars each, 200 overlap)
3. Generate embedding vectors for each chunk
4. Create a Pinecone index and upsert all vectors

Expected output:
```
==================================================
Starting Document Ingestion Pipeline
==================================================
[1/4] Loading PDF from: data/Ebook-Agentic-AI.pdf
       Loaded 60 pages from the PDF.
[2/4] Splitting document into chunks (size=1000, overlap=200)...
       Created 127 text chunks.
[3/4] Setting up Pinecone index: 'agentic-ai-index' (dimension=1536)...
[4/4] Generating embeddings and upserting to Pinecone...
       Successfully upserted 127 chunks to index 'agentic-ai-index'.

✅ Ingestion pipeline completed successfully!
```

---

## Running the Application

### Option A: FastAPI Backend

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at: `http://localhost:8000`  
Interactive docs at: `http://localhost:8000/docs`

### Option B: Streamlit Web UI

```bash
streamlit run streamlit_app.py
```

The web UI will open in your browser at: `http://localhost:8501`

---

## API Documentation

### `POST /chat`

Submit a query to the RAG chatbot.

**Request:**
```json
{
    "query": "What is Agentic AI according to the eBook?"
}
```

**Response:**
```json
{
    "final_answer": "According to the eBook, Agentic AI refers to...",
    "retrieved_context": [
        "Chapter 1: Agentic AI is defined as...",
        "Chapter 2: The key characteristics of Agentic AI include...",
        "Chapter 3: Unlike traditional systems, Agentic AI..."
    ],
    "confidence_score": 0.92
}
```

### `GET /health`

Health check endpoint.

```json
{
    "status": "healthy",
    "service": "Agentic AI RAG Chatbot"
}
```

---

## Sample Test Queries

Run the benchmark test suite:

```bash
python tests_sample_queries.py
```

| # | Query | Expected Behavior |
|---|-------|-------------------|
| 1 | "What is Agentic AI according to the eBook?" | Returns definition from the eBook |
| 2 | "How do AI agents differ from traditional automation systems?" | Highlights key differences |
| 3 | "What are the core components of an Agentic Architecture?" | Lists architectural components |
| 4 | "What role does memory play in Agentic AI workflows?" | Describes memory's role |
| 5 | "What are some real-world applications of Agentic AI?" | Lists use cases from the eBook |
| 6 | "Who won the 2022 FIFA World Cup?" | **Refuses** — out of document scope |

---

## Configuration Options

The system supports two provider modes, configured via `.env`:

| Setting | OpenAI (Default) | Free Alternative |
|---------|-----------------|------------------|
| `EMBEDDING_PROVIDER` | `openai` | `huggingface` |
| `LLM_PROVIDER` | `openai` | `ollama` |
| Embedding Model | `text-embedding-3-small` (1536d) | `all-MiniLM-L6-v2` (384d) |
| LLM Model | `gpt-4o-mini` | `llama3.2` (local) |
| Vector Dimension | 1536 | 384 |

> **Auto-detection:** If `OPENAI_API_KEY` is not set, the system automatically falls back to HuggingFace embeddings + Ollama LLM.

---

## License

This project is for educational purposes as part of the Appening AI assignment.
