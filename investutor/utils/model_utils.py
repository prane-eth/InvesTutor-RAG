# Copyright (c) Praneeth Vadlapati

import os
from typing import List

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

from investutor.utils.retrieval_utils import retriever

# Initialize Cohere client
model = os.getenv("OPENAI_MODEL", "")
if not model:
    raise Exception("OPENAI_MODEL is not set in the environment variables")
print(f"Model: {model}")
model_name_short = model.split("/")[-1].lower()

llm = ChatOpenAI(model=model, temperature=0.7)



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
