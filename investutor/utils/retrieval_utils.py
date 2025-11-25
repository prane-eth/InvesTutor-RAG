import os
from typing import Any, Dict, List

import cohere
from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_pinecone import PineconeVectorStore
import openai
from pinecone import Pinecone

load_dotenv()

RERANK_MODEL_NAME = os.getenv("RERANK_MODEL_NAME", "")
if not RERANK_MODEL_NAME:
    raise Exception("RERANK_MODEL_NAME is not set in the environment variables")

embed_model_name = os.getenv("EMBED_MODEL_NAME", "")
if not embed_model_name:
    raise Exception("EMBED_MODEL_NAME is not set in the environment variables")

embed_api_key = os.getenv("EMBEDDING_API_KEY", "")
if not embed_api_key:
    raise Exception("EMBEDDING_API_KEY is not set in the environment variables")

embed_base_url = os.getenv("EMBEDDING_BASE_URL", "")
if not embed_base_url:
    raise Exception("EMBEDDING_BASE_URL is not set in the environment variables")

index_name = os.getenv("PINECONE_INDEX_NAME", "")
if not index_name:
    raise Exception("PINECONE_INDEX_NAME is not set in the environment variables")

embed_client = openai.OpenAI(api_key=embed_api_key, base_url = embed_base_url)

cohere_client = cohere.ClientV2(api_key=embed_api_key,
                                base_url=embed_base_url.replace("/v1", ""))


class CustomEmbeddings(Embeddings):
    def embed_query(self, text: str) -> List[float]:
        response = embed_client.embeddings.create(input=[text], model=embed_model_name)
        return response.data[0].embedding

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        response = embed_client.embeddings.create(input=texts, model=embed_model_name)
        return [data.embedding for data in response.data]


embeddings = CustomEmbeddings()
vdb_client = Pinecone()



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


# Create index if it doesn't exist
if index_name not in [idx.name for idx in vdb_client.list_indexes()]:
    # Generate a sample embedding and find size
    print("Generating sample embedding to determine dimension...")
    sample_embedding = embeddings.embed_query("Sample text for dimension check")
    dimension = len(sample_embedding)

    print("Creating Pinecone index:", index_name, "with dimension:", dimension)
    vdb_client.create_index(
        name=index_name,
        dimension=dimension,
        metric="cosine",
        vector_type="dense",
        spec={"serverless": {"cloud": "aws", "region": "us-east-1"}},
    )

vectorstore = PineconeVectorStore(vdb_client.Index(index_name), embeddings)
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

    # Call Cohere rerank API
    response = cohere_client.rerank(
        query=query,
        documents=[result["content"] for result in results],
        top_n=top_k,
        model=RERANK_MODEL_NAME,
    )

    # Reorder results based on rerank response
    reranked = []
    for rerank_result in response.results:
        # print("original_result:")
        # print(results[rerank_result.index])
        # print("rerank_result:")
        # print(rerank_result)
        original_result = results[rerank_result.index]
        original_result["rerank_score"] = rerank_result.relevance_score
        original_result["combined_score"] = (
            0.7 * original_result["score"] + 0.3 * rerank_result.relevance_score
        )
        reranked.append(original_result)

    return reranked


if __name__ == "__main__":
    # Test the retriever
    query = "What Is an Investment?"
    docs = retriever._get_relevant_documents(query)
    for i, doc in enumerate(docs):
        print(f"Document {i+1}: {doc.page_content} | Metadata: {doc.metadata}")
