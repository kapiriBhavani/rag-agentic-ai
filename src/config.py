"""
Configuration module for the RAG Agentic AI Chatbot.

Handles environment variable loading, API key validation,
and provider selection (OpenAI / Google Gemini / HuggingFace+Ollama).
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ---------------------------------------------------------------------------
# API Keys
# ---------------------------------------------------------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")

# ---------------------------------------------------------------------------
# Pinecone Configuration
# ---------------------------------------------------------------------------
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "agentic-ai-index")

# ---------------------------------------------------------------------------
# Provider Selection (auto-detect based on available API keys)
# ---------------------------------------------------------------------------
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "").lower()
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "").lower()

# Auto-detect: prioritize OpenAI > Google > free alternatives
if not EMBEDDING_PROVIDER:
    if OPENAI_API_KEY and OPENAI_API_KEY != "your_openai_api_key":
        EMBEDDING_PROVIDER = "openai"
    elif GOOGLE_API_KEY and GOOGLE_API_KEY != "your_google_api_key":
        EMBEDDING_PROVIDER = "google"
    else:
        EMBEDDING_PROVIDER = "huggingface"

if not LLM_PROVIDER:
    if OPENAI_API_KEY and OPENAI_API_KEY != "your_openai_api_key":
        LLM_PROVIDER = "openai"
    elif GOOGLE_API_KEY and GOOGLE_API_KEY != "your_google_api_key":
        LLM_PROVIDER = "google"
    else:
        LLM_PROVIDER = "ollama"

# ---------------------------------------------------------------------------
# Embedding Configuration
# ---------------------------------------------------------------------------
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
OPENAI_EMBEDDING_DIMENSION = 1536

GOOGLE_EMBEDDING_MODEL = "models/text-embedding-004"
GOOGLE_EMBEDDING_DIMENSION = 768

HUGGINGFACE_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
HUGGINGFACE_EMBEDDING_DIMENSION = 384

# Active embedding settings based on provider
if EMBEDDING_PROVIDER == "openai":
    EMBEDDING_DIMENSION = OPENAI_EMBEDDING_DIMENSION
elif EMBEDDING_PROVIDER == "google":
    EMBEDDING_DIMENSION = GOOGLE_EMBEDDING_DIMENSION
else:
    EMBEDDING_DIMENSION = HUGGINGFACE_EMBEDDING_DIMENSION

# ---------------------------------------------------------------------------
# LLM Configuration
# ---------------------------------------------------------------------------
OPENAI_LLM_MODEL = "gpt-4o-mini"
GOOGLE_LLM_MODEL = "gemini-2.0-flash"
OLLAMA_LLM_MODEL = "llama3.2"

# ---------------------------------------------------------------------------
# Document Processing
# ---------------------------------------------------------------------------
PDF_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "Ebook-Agentic-AI.pdf")
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
TOP_K_RESULTS = 3


def get_embeddings():
    """Return the appropriate embedding model based on the configured provider."""
    if EMBEDDING_PROVIDER == "openai":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(
            model=OPENAI_EMBEDDING_MODEL,
            openai_api_key=OPENAI_API_KEY,
        )
    elif EMBEDDING_PROVIDER == "google":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        return GoogleGenerativeAIEmbeddings(
            model=GOOGLE_EMBEDDING_MODEL,
            google_api_key=GOOGLE_API_KEY,
        )
    else:
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(
            model_name=HUGGINGFACE_EMBEDDING_MODEL,
        )


def get_llm():
    """Return the appropriate LLM based on the configured provider."""
    if LLM_PROVIDER == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=OPENAI_LLM_MODEL,
            temperature=0,
            openai_api_key=OPENAI_API_KEY,
        )
    elif LLM_PROVIDER == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=GOOGLE_LLM_MODEL,
            temperature=0,
            google_api_key=GOOGLE_API_KEY,
        )
    else:
        from langchain_community.llms import Ollama
        return Ollama(
            model=OLLAMA_LLM_MODEL,
            temperature=0,
        )


def print_config():
    """Print the current configuration for debugging."""
    print("=" * 50)
    print("RAG Agentic AI - Configuration")
    print("=" * 50)
    print(f"  Embedding Provider : {EMBEDDING_PROVIDER}")
    print(f"  Embedding Dimension: {EMBEDDING_DIMENSION}")
    print(f"  LLM Provider       : {LLM_PROVIDER}")
    print(f"  Pinecone Index     : {PINECONE_INDEX_NAME}")
    print(f"  PDF Path           : {PDF_PATH}")
    print(f"  Chunk Size         : {CHUNK_SIZE}")
    print(f"  Chunk Overlap      : {CHUNK_OVERLAP}")
    print(f"  Top-K Results      : {TOP_K_RESULTS}")
    print(f"  OpenAI Key Set     : {'Yes' if OPENAI_API_KEY and OPENAI_API_KEY != 'your_openai_api_key' else 'No'}")
    print(f"  Google Key Set     : {'Yes' if GOOGLE_API_KEY and GOOGLE_API_KEY != 'your_google_api_key' else 'No'}")
    print(f"  Pinecone Key Set   : {'Yes' if PINECONE_API_KEY and PINECONE_API_KEY != 'your_pinecone_api_key' else 'No'}")
    print("=" * 50)
