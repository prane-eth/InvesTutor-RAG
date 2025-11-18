import os
from typing import Any, Dict, List

import cohere

from utils.model_utils import vectorstore

co = cohere.ClientV2()


RERANK_MODEL_NAME = os.getenv("RERANK_MODEL_NAME", "")
if not RERANK_MODEL_NAME:
    raise Exception("RERANK_MODEL_NAME is not set in the environment variables")


def search_documents(query: str, k: int = 5) -> List[Dict[str, Any]]:
    """Search documents using semantic similarity."""
    if vectorstore is None:
        return []

    docs = vectorstore.similarity_search_with_score(query, k=k)
    results = []
    for doc, score in docs:
        results.append(
            {"content": doc.page_content, "metadata": doc.metadata, "score": score}
        )
    return results


def rerank_results(
    query: str, results: List[Dict[str, Any]], top_k: int = 3
) -> List[Dict[str, Any]]:
    """Rerank results using Cohere rerank API."""
    if not results:
        return results

    try:
        # Call Cohere rerank API
        response = co.rerank(
            query=query,
            documents=[result["content"] for result in results],
            top_n=top_k,
            model=RERANK_MODEL_NAME,
        )

        # Reorder results based on rerank response
        reranked = []
        for rerank_result in response.results:
            original_result = results[rerank_result.index]
            original_result["rerank_score"] = rerank_result.relevance_score
            original_result["combined_score"] = (
                0.7 * original_result["score"] + 0.3 * rerank_result.relevance_score
            )
            reranked.append(original_result)

        return reranked

    except Exception as e:
        print(f"Cohere reranking failed: {e}, falling back to basic reranking")
        # Fallback to basic reranking
        reranked = sorted(
            results,
            key=lambda x: x["score"] * (1 + len(x["content"]) / 1000),
            reverse=True,
        )
        return reranked[:top_k]
