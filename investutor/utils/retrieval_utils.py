import os
from typing import Any, Dict, List

import cohere
from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_milvus import Milvus
from langchain_pinecone import PineconeVectorStore
import openai
from pinecone import Pinecone
from pymilvus import MilvusClient

load_dotenv()

EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "")
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY", "")
EMBEDDING_BASE_URL = os.getenv("EMBEDDING_BASE_URL", "")

RERANK_MODEL_NAME = os.getenv("RERANK_MODEL_NAME", "")
RERANK_API_KEY = os.getenv("RERANK_API_KEY", "")
RERANK_BASE_URL = os.getenv("RERANK_BASE_URL", "")

index_name = os.getenv("PINECONE_INDEX_NAME", "")

for val in [RERANK_MODEL_NAME, EMBED_MODEL_NAME, EMBEDDING_API_KEY, EMBEDDING_BASE_URL,
            RERANK_API_KEY, RERANK_BASE_URL, index_name]:
    if not val:
        raise Exception("One or more required environment variables are not set.")


milvus_url = os.getenv("MILVUS_URI", "")
if milvus_url:
    index_name = index_name.replace("-", "_")  # compatible with Milvus naming

embed_client = openai.OpenAI(api_key=EMBEDDING_API_KEY, base_url=EMBEDDING_BASE_URL)

cohere_client = cohere.ClientV2(api_key=RERANK_API_KEY, base_url=RERANK_BASE_URL)


class CustomEmbeddings(Embeddings):
    def embed_query(self, text: str) -> List[float]:
        response = embed_client.embeddings.create(input=[text], model=EMBED_MODEL_NAME)
        return response.data[0].embedding

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        response = embed_client.embeddings.create(input=texts, model=EMBED_MODEL_NAME)
        return [data.embedding for data in response.data]


embeddings = CustomEmbeddings()
pc_client = Pinecone()

milvus_client = None
if milvus_url:
    milvus_client = MilvusClient(uri=milvus_url, token="root:Milvus")


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
if milvus_client:
    # Create collection
    if index_name not in milvus_client.list_collections():  # type: ignore
        print("Generating sample embedding to determine dimension...")
        sample_embedding = embeddings.embed_query("Sample text for dimension check")
        dimension = len(sample_embedding)

        print("Creating Milvus collection:", index_name)
        milvus_client.create_collection(
            collection_name=index_name,
            dimension=dimension,
            metric_type="COSINE",
        )

    # # Create index
    # if index_name not in milvus_client.list_indexes(collection_name=index_name):
    #     milvus_client.create_index(
    #         collection_name=index_name,
    #         index_params=IndexParams([
    #             IndexParam(
    #                 field_name="embeddings",
    #                 index_type="HNSW", # "FLAT",
    #                 index_name=index_name,
    #             )
    #         ]),
    #         # field_name="embeddings",
    #     )

    vectorstore = Milvus(
        collection_name=index_name,
        embedding_function=embeddings,
        connection_args={"uri": milvus_url, "token": "root:Milvus", "db_name": index_name},
        index_params={"metric_type": "COSINE"},
        consistency_level="Strong",
        drop_old=False,
        auto_id=True,
    )
else:
    if index_name not in [idx.name for idx in pc_client.list_indexes()]:
        # Generate a sample embedding and find size
        print("Generating sample embedding to determine dimension...")
        sample_embedding = embeddings.embed_query("Sample text for dimension check")
        dimension = len(sample_embedding)

        print("Creating Pinecone index:", index_name, "with dimension:", dimension)
        pc_client.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            vector_type="dense",
            spec={"serverless": {"cloud": "aws", "region": "us-east-1"}},
        )

    vectorstore = PineconeVectorStore(pc_client.Index(index_name),
                                      embeddings, index_name=index_name)

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
