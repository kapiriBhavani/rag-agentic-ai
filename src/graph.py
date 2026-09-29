"""
LangGraph RAG Workflow for the Agentic AI Chatbot.

This module defines a stateful execution graph using LangGraph that:
1. Retrieves relevant document chunks from Pinecone based on user query
2. Generates a grounded response using the LLM, strictly from retrieved context
3. Computes a confidence score reflecting retrieval quality

Graph Flow: START -> retrieve -> generate -> END
"""

from typing import List, TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_pinecone import PineconeVectorStore

from src.config import (
    PINECONE_INDEX_NAME,
    TOP_K_RESULTS,
    get_embeddings,
    get_llm,
)


class AgentState(TypedDict):
    """State schema for the RAG agent workflow."""
    question: str
    context: List[str]
    answer: str
    score: float


# ---------------------------------------------------------------------------
# System prompt enforcing strict grounding on retrieved context
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a knowledgeable assistant for the "Agentic AI eBook". \
You must answer the user's question relying ONLY on the context chunks provided below.

Rules:
1. Base your answer EXCLUSIVELY on the provided context.
2. If the context does not contain sufficient information to answer the question, \
respond with: "I cannot answer this question based on the provided document."
3. Do NOT use any outside knowledge or make assumptions beyond what is in the context.
4. Provide a clear, well-structured answer.
5. At the end of your response, on a new line, output a confidence score between \
0.0 and 1.0 in the format: [CONFIDENCE: X.XX]

Context:
{context}

Question: {question}"""


def build_rag_graph(index_name: str = None):
    """
    Build and compile the LangGraph RAG workflow.

    The graph consists of two nodes:
    - retrieve: Queries Pinecone for top-k relevant document chunks
    - generate: Uses the LLM to produce a grounded answer from the context

    Args:
        index_name: Pinecone index name (defaults to config value).

    Returns:
        Compiled LangGraph workflow ready for invocation.
    """
    index_name = index_name or PINECONE_INDEX_NAME

    # Initialize embedding model and vector store
    embeddings = get_embeddings()
    vectorstore = PineconeVectorStore(
        index_name=index_name,
        embedding=embeddings,
    )
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": TOP_K_RESULTS},
    )

    # Initialize LLM
    llm = get_llm()

    # ------------------------------------------------------------------
    # Node 1: Retrieve relevant chunks from Pinecone
    # ------------------------------------------------------------------
    def retrieve_node(state: AgentState) -> dict:
        """Query the vector store for relevant document chunks."""
        docs = retriever.invoke(state["question"])
        context_texts = [doc.page_content for doc in docs]
        return {"context": context_texts}

    # ------------------------------------------------------------------
    # Node 2: Generate a grounded answer using the LLM
    # ------------------------------------------------------------------
    def generate_node(state: AgentState) -> dict:
        """Generate an answer strictly grounded in the retrieved context."""
        context_str = "\n\n---\n\n".join(state["context"])

        prompt = SYSTEM_PROMPT.format(
            context=context_str,
            question=state["question"],
        )

        response = llm.invoke(prompt)

        # Extract the response content (handle both ChatModel and LLM outputs)
        if hasattr(response, "content"):
            if isinstance(response.content, list):
                # Handle Gemini 3.8 returning a list of content blocks
                texts = []
                for block in response.content:
                    if isinstance(block, dict) and "text" in block:
                        texts.append(block["text"])
                    elif isinstance(block, str):
                        texts.append(block)
                answer_text = " ".join(texts)
            else:
                answer_text = str(response.content)
        else:
            answer_text = str(response)

        # Parse confidence score from the response
        confidence = _parse_confidence(answer_text, state["context"])

        # Remove the confidence tag from the displayed answer
        clean_answer = answer_text.split("[CONFIDENCE:")[0].strip()

        return {"answer": clean_answer, "score": confidence}

    # ------------------------------------------------------------------
    # Build the StateGraph
    # ------------------------------------------------------------------
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node)

    # Define edges: START -> retrieve -> generate -> END
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)

    # Compile and return
    return workflow.compile()


def _parse_confidence(answer_text: str, context: list) -> float:
    """
    Extract confidence score from LLM response.
    Falls back to a heuristic based on context availability.

    Args:
        answer_text: The raw LLM response text.
        context: List of retrieved context chunks.

    Returns:
        Confidence score between 0.0 and 1.0.
    """
    try:
        if "[CONFIDENCE:" in answer_text:
            score_str = answer_text.split("[CONFIDENCE:")[1].split("]")[0].strip()
            score = float(score_str)
            return max(0.0, min(1.0, score))
    except (IndexError, ValueError):
        pass

    # Fallback heuristic: based on number of retrieved chunks
    if not context or len(context) == 0:
        return 0.0
    elif "cannot answer" in answer_text.lower() or "not enough information" in answer_text.lower():
        return 0.1
    elif len(context) >= 3:
        return 0.85
    else:
        return 0.6
