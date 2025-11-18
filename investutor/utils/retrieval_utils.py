import os
from typing import Any, Dict, List

import cohere
from dotenv import load_dotenv
from langchain_cohere import CohereEmbeddings
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone as PineconeClient

load_dotenv()

RERANK_MODEL_NAME = os.getenv("RERANK_MODEL_NAME", "")
if not RERANK_MODEL_NAME:
    raise Exception("RERANK_MODEL_NAME is not set in the environment variables")

embed_model_name = os.getenv("EMBED_MODEL_NAME", "")
if not embed_model_name:
    raise Exception("EMBED_MODEL_NAME is not set in the environment variables")

index_name = os.getenv("PINECONE_INDEX_NAME", "")
if not index_name:
    raise Exception("PINECONE_INDEX_NAME is not set in the environment variables")

embeddings = CohereEmbeddings(
    client=cohere.ClientV2(), async_client=None, model=embed_model_name
)
co = cohere.ClientV2()
pc = PineconeClient()


# RAG setup using modern LangChain LCEL


# Custom retriever with reranking
class RerankingRetriever(BaseRetriever):
    vectorstore: Any = None
    k: int = 5
    rerank_top_k: int = 3

    def __init__(self, vectorstore, k=5, rerank_top_k=3):
        super().__init__()
        self.vectorstore = vectorstore
        self.k = k
        self.rerank_top_k = rerank_top_k

    def _get_relevant_documents(
        self, query: str, *, run_manager=None
    ) -> List[Document]:
        # Get initial results
        results = search_documents(query, k=self.k)

        # Rerank results
        reranked = rerank_results(query, results, top_k=self.rerank_top_k)

        # Convert back to Document objects
        docs = []
        for result in reranked:
            doc = Document(page_content=result["content"], metadata=result["metadata"])
            docs.append(doc)

        return docs


# Initialize retriever
if index_name in [idx.name for idx in pc.list_indexes()]:
    vectorstore = PineconeVectorStore(pc.Index(index_name), embeddings)
else:
    raise Exception("Pinecone index not found or PINECONE_INDEX_NAME not set.")

retriever = RerankingRetriever(vectorstore, k=5, rerank_top_k=3)


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
