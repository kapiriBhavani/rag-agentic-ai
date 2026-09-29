"""
Document Ingestion Pipeline for the RAG Agentic AI Chatbot.

This module handles:
1. Loading the PDF document using PyPDFLoader
2. Splitting the document into manageable text chunks
3. Generating embedding vectors for each chunk
4. Creating/connecting to a Pinecone index and upserting the vectors
"""

import os
from pinecone import Pinecone, ServerlessSpec
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore

from src.config import (
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    EMBEDDING_DIMENSION,
    PDF_PATH,
    get_embeddings,
    print_config,
)


def load_and_split_pdf(pdf_path: str) -> list:
    """
    Load a PDF document and split it into text chunks.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        List of document chunks ready for embedding.
    """
    print(f"[1/4] Loading PDF from: {pdf_path}")
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found at: {pdf_path}")

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f"       Loaded {len(documents)} pages from the PDF.")

    print(f"[2/4] Splitting document into chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = text_splitter.split_documents(documents)
    print(f"       Created {len(chunks)} text chunks.")

    return chunks


def create_pinecone_index(index_name: str, dimension: int) -> None:
    """
    Create a Pinecone index if it does not already exist.

    Args:
        index_name: Name of the Pinecone index to create.
        dimension: Dimensionality of the embedding vectors.
    """
    print(f"[3/4] Setting up Pinecone index: '{index_name}' (dimension={dimension})...")

    pc = Pinecone(api_key=PINECONE_API_KEY)

    # Check if index already exists
    existing_indexes = [idx.name for idx in pc.list_indexes()]
    if index_name in existing_indexes:
        print(f"       Index '{index_name}' already exists. Skipping creation.")
        return

    # Create a new serverless index
    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1",
        ),
    )
    print(f"       Index '{index_name}' created successfully.")


def upsert_to_pinecone(chunks: list, index_name: str) -> PineconeVectorStore:
    """
    Convert document chunks to embeddings and upsert them into Pinecone.

    Args:
        chunks: List of document chunks to embed and store.
        index_name: Name of the Pinecone index.

    Returns:
        PineconeVectorStore instance for querying.
    """
    import time
    from langchain_pinecone import PineconeVectorStore
    print(f"[4/4] Generating embeddings and upserting to Pinecone in batches to respect rate limits...")

    embeddings = get_embeddings()
    
    # Initialize an empty vector store to add to
    vector_store = PineconeVectorStore(
        index_name=index_name,
        embedding=embeddings,
        pinecone_api_key=PINECONE_API_KEY
    )

    batch_size = 20
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        print(f"       Processing batch {i//batch_size + 1}/{len(chunks)//batch_size + 1}: chunks {i} to {i+len(batch)-1}")
        vector_store.add_documents(batch)
        if i + batch_size < len(chunks):
            print("       Waiting 15 seconds to respect rate limits...")
            time.sleep(15)

    print(f"       Successfully upserted {len(chunks)} chunks to index '{index_name}'.")
    return vector_store


def run_ingestion(pdf_path: str = None, index_name: str = None) -> PineconeVectorStore:
    """
    Execute the full ingestion pipeline:
    Load PDF -> Split into chunks -> Create index -> Upsert embeddings.

    Args:
        pdf_path: Path to the PDF file (defaults to config value).
        index_name: Pinecone index name (defaults to config value).

    Returns:
        PineconeVectorStore instance ready for retrieval.
    """
    pdf_path = pdf_path or PDF_PATH
    index_name = index_name or PINECONE_INDEX_NAME

    print("\n" + "=" * 50)
    print("Starting Document Ingestion Pipeline")
    print("=" * 50)
    print_config()

    # Step 1 & 2: Load and split
    chunks = load_and_split_pdf(pdf_path)

    # Step 3: Create Pinecone index
    create_pinecone_index(index_name, EMBEDDING_DIMENSION)

    # Step 4: Upsert embeddings
    vector_store = upsert_to_pinecone(chunks, index_name)

    print("\n✅ Ingestion pipeline completed successfully!")
    print("=" * 50 + "\n")

    return vector_store


if __name__ == "__main__":
    run_ingestion()
