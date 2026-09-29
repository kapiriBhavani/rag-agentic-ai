"""
Streamlit UI for the RAG Agentic AI Chatbot.

Provides a lightweight web interface for interacting with the RAG pipeline:
- Chat input for user queries
- Side panel displaying retrieved context chunks and confidence score
- Persistent chat history within the session
"""

import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from src.graph import build_rag_graph
from src.config import PINECONE_INDEX_NAME, EMBEDDING_PROVIDER, LLM_PROVIDER

# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Agentic AI RAG Chatbot",
    page_icon="🤖",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Initialize RAG Graph (cached to avoid reloading on every interaction)
# ---------------------------------------------------------------------------
@st.cache_resource
def init_graph():
    """Initialize and cache the RAG graph."""
    return build_rag_graph(index_name=PINECONE_INDEX_NAME)


graph = init_graph()

# ---------------------------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_context" not in st.session_state:
    st.session_state.last_context = []

if "last_score" not in st.session_state:
    st.session_state.last_score = 0.0

# ---------------------------------------------------------------------------
# Sidebar: Configuration & Retrieved Context
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("⚙️ Configuration")
    st.markdown(f"**Embedding Provider:** `{EMBEDDING_PROVIDER}`")
    st.markdown(f"**LLM Provider:** `{LLM_PROVIDER}`")
    st.divider()

    st.title("📄 Retrieved Context")
    if st.session_state.last_context:
        st.metric("Confidence Score", f"{st.session_state.last_score:.2f}")
        st.divider()
        for i, chunk in enumerate(st.session_state.last_context, 1):
            with st.expander(f"Chunk {i}", expanded=(i == 1)):
                st.text(chunk)
    else:
        st.info("Ask a question to see retrieved context chunks here.")

# ---------------------------------------------------------------------------
# Main Chat Interface
# ---------------------------------------------------------------------------
st.title("🤖 Agentic AI RAG Chatbot")
st.caption("Ask questions about the Agentic AI eBook. Answers are strictly grounded in the document.")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if user_query := st.chat_input("Ask a question about Agentic AI..."):
    # Display user message
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Retrieving context and generating answer..."):
            try:
                initial_state = {
                    "question": user_query,
                    "context": [],
                    "answer": "",
                    "score": 0.0,
                }
                result = graph.invoke(initial_state)

                answer = result["answer"]
                context = result["context"]
                score = result["score"]

                # Update sidebar state
                st.session_state.last_context = context
                st.session_state.last_score = score

                # Display answer
                st.markdown(answer)
                st.caption(f"📊 Confidence Score: **{score:.2f}**")

                # Save to chat history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                })

            except Exception as e:
                error_msg = f"❌ Error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg,
                })

    # Rerun to update sidebar with new context
    st.rerun()
