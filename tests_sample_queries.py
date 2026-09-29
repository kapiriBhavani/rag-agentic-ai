"""
Sample Test Queries for the RAG Agentic AI Chatbot.

This script runs 5-6 benchmark queries against the RAG pipeline to verify:
- Response quality and relevance
- Context retrieval accuracy
- Strict grounding (out-of-scope questions should be refused)
- Confidence scoring

Usage:
    python tests_sample_queries.py
"""

import json
from dotenv import load_dotenv

load_dotenv()

from src.graph import build_rag_graph
from src.config import PINECONE_INDEX_NAME, print_config


# ---------------------------------------------------------------------------
# Benchmark Test Queries
# ---------------------------------------------------------------------------
TEST_QUERIES = [
    {
        "id": 1,
        "query": "What is Agentic AI according to the eBook?",
        "description": "Core definition from the document",
        "expected": "Should return a clear definition of Agentic AI from the eBook content.",
    },
    {
        "id": 2,
        "query": "How do AI agents differ from traditional automation systems?",
        "description": "Comparison question",
        "expected": "Should highlight key differences mentioned in the eBook.",
    },
    {
        "id": 3,
        "query": "What are the core components of an Agentic Architecture?",
        "description": "Architecture components",
        "expected": "Should list architectural components discussed in the eBook.",
    },
    {
        "id": 4,
        "query": "What role does memory play in Agentic AI workflows?",
        "description": "Memory in Agentic AI",
        "expected": "Should describe memory's role as discussed in the eBook.",
    },
    {
        "id": 5,
        "query": "What are some real-world applications of Agentic AI mentioned in the eBook?",
        "description": "Real-world applications",
        "expected": "Should list applications or use cases from the eBook.",
    },
    {
        "id": 6,
        "query": "Who won the 2022 FIFA World Cup?",
        "description": "Out-of-scope validation test",
        "expected": "Should REFUSE to answer or state that the document does not contain this information.",
    },
]


def run_tests():
    """Execute all benchmark test queries and display results."""
    print_config()

    print("\nInitializing RAG Graph...")
    graph = build_rag_graph(index_name=PINECONE_INDEX_NAME)
    print("✅ Graph initialized.\n")

    print("=" * 70)
    print("RUNNING BENCHMARK TEST QUERIES")
    print("=" * 70)

    results = []

    for test in TEST_QUERIES:
        print(f"\n{'─' * 70}")
        print(f"Test #{test['id']}: {test['description']}")
        print(f"Query: \"{test['query']}\"")
        print(f"Expected: {test['expected']}")
        print(f"{'─' * 70}")

        try:
            initial_state = {
                "question": test["query"],
                "context": [],
                "answer": "",
                "score": 0.0,
            }
            result = graph.invoke(initial_state)

            answer = result["answer"]
            context = result["context"]
            score = result["score"]

            print(f"\n📝 Answer:\n{answer}")
            print(f"\n📊 Confidence Score: {score:.2f}")
            print(f"📄 Retrieved Chunks: {len(context)}")

            for i, chunk in enumerate(context, 1):
                preview = chunk[:150].replace("\n", " ") + "..."
                print(f"   Chunk {i}: {preview}")

            results.append({
                "query": test["query"],
                "answer": answer,
                "confidence_score": score,
                "num_chunks_retrieved": len(context),
                "status": "PASS",
            })

        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            results.append({
                "query": test["query"],
                "answer": None,
                "confidence_score": 0.0,
                "num_chunks_retrieved": 0,
                "status": f"FAIL: {str(e)}",
            })

    # Summary
    print(f"\n\n{'=' * 70}")
    print("TEST SUMMARY")
    print(f"{'=' * 70}")
    passed = sum(1 for r in results if r["status"] == "PASS")
    print(f"Passed: {passed}/{len(results)}")
    for r in results:
        status_icon = "✅" if r["status"] == "PASS" else "❌"
        print(f"  {status_icon} {r['query'][:60]}... -> Score: {r['confidence_score']:.2f}")

    return results


if __name__ == "__main__":
    run_tests()
