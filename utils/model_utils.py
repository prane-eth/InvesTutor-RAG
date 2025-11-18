# Copyright (c) Praneeth Vadlapati

import os
from typing import Any, List

import cohere
from dotenv import load_dotenv
from langchain_cohere import CohereEmbeddings
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone as PineconeClient

from utils.retrieval import rerank_results, search_documents

load_dotenv()

# Initialize Cohere client
co = cohere.ClientV2()
model = os.getenv("OPENAI_MODEL", "")
if not model:
    raise Exception("OPENAI_MODEL is not set in the environment variables")
print(f"Model: {model}")
model_name_short = model.split("/")[-1].lower()

EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "")
if not EMBED_MODEL_NAME:
    raise Exception("EMBED_MODEL_NAME is not set in the environment variables")


llm = ChatOpenAI(model=model, temperature=0.7)


# Embedding model setup using Cohere API

embeddings = CohereEmbeddings(client=co, async_client=None,
                              model=EMBED_MODEL_NAME)


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


pc = PineconeClient()
index_name = os.getenv("PINECONE_INDEX_NAME", "")

if index_name and index_name in [idx.name for idx in pc.list_indexes()]:
    vectorstore = PineconeVectorStore(pc.Index(index_name), embeddings)
    retriever = RerankingRetriever(vectorstore, k=5, rerank_top_k=3)
else:
    raise Exception("Pinecone index not found or PINECONE_INDEX_NAME not set.")

# Initialize retriever
retriever = RerankingRetriever(vectorstore, k=5, rerank_top_k=3)

# RAG prompt for investment tutoring
rag_prompt_template = """
You are an AI tutor specializing in investment education. Use the following pieces of context to answer the student's question.
If you don't know the answer based on the provided context, say "I don't have enough information from my knowledge base to answer this question accurately."

Context:
{context}

Question: {question}

Instructions:
- Provide clear, educational explanations
- Include relevant examples when possible
- ALWAYS cite sources for any factual claims, statistics, or specific investment advice
- If a claim cannot be supported by the provided context, do not make it
- Use phrases like "According to [source]" or "Based on [source]" when citing
- If you cannot find supporting evidence in the context, say "I don't have source-backed information for that"
- Encourage critical thinking about investments

Answer:"""

RAG_PROMPT = PromptTemplate(
    template=rag_prompt_template, input_variables=["context", "question"]
)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def validate_citations(response: str, context_docs: List[Document]) -> str:
    """Validate that claims in the response are backed by sources."""
    # Extract all source references from context
    context_sources = set()
    for doc in context_docs:
        source = doc.metadata.get("source", "")
        if source:
            # Extract domain or key identifier
            if "investopedia.com" in source:
                context_sources.add("Investopedia")
            elif "wikipedia.org" in source:
                context_sources.add("Wikipedia")
            else:
                context_sources.add(source.split("/")[-1] if "/" in source else source)

    # Check if response makes uncited claims
    claim_indicators = [
        "typically",
        "generally",
        "usually",
        "often",
        "research shows",
        "studies show",
        "experts say",
        "according to",
        "data shows",
        "statistics show",
        "market data",
        "historical data",
    ]

    response_lower = response.lower()
    has_uncited_claims = any(
        indicator in response_lower for indicator in claim_indicators
    )

    # If response makes claims but doesn't cite sources, add disclaimer
    if (
        has_uncited_claims
        and "according to" not in response_lower
        and "based on" not in response_lower
    ):
        disclaimer = f"\n\n*Note: This response is based on general investment principles. For specific claims or current market data, please consult the cited sources: {', '.join(context_sources)}.*"
        response += disclaimer

    return response


# Initialize RAG chain only if retriever exists
base_rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | RAG_PROMPT
    | llm
    | StrOutputParser()
)

def rag_chain(question: str) -> str:
    """RAG chain with citation validation."""
    if retriever is None:
        return "The RAG system is not properly initialized. Please check your Pinecone configuration and ensure documents have been ingested."

    # Get context documents
    context_docs = retriever._get_relevant_documents(question)

    # Generate response
    response = base_rag_chain.invoke(question)

    # Validate citations
    validated_response = validate_citations(response, context_docs)

    return validated_response
