#!/usr/bin/env python3
"""
Test script for the AI Investment Tutor RAG system
"""

import os

from investutor.utils.ingestion import ingest_md_document
from investutor.utils.llm_utils import rag_chain
from investutor.utils.news_api_integration import fetch_financial_news
from investutor.utils.retrieval_utils import rerank_results, search_documents


def test_ingestion():
    """Document ingestion."""
    print("Testing document ingestion...")

    # Test with a sample PDF if available
    test_files = [
        "sample_investment_guide.pdf",  # Would need to have this file
        "https://www.investopedia.com/terms/i/investment.asp",  # Test HTML
    ]

    for file_path in test_files:
        if os.path.exists(file_path) or file_path.startswith("http"):
            try:
                ingest_md_document(file_path)
                print(f"✓ Successfully ingested: {file_path}")
            except Exception as e:
                print(f"✗ Failed to ingest {file_path}: {str(e)}")
        else:
            print(f"⚠ Test file not found: {file_path}")


def test_retrieval():
    """Test retrieval functionality."""
    print("\nTesting retrieval...")

    test_queries = [
        "What is diversification in investing?",
        "Explain compound interest",
        "What are bonds?",
    ]

    for query in test_queries:
        try:
            results = search_documents(query, k=3)
            reranked = rerank_results(query, results)

            print(f"\nQuery: {query}")
            print(f"Found {len(results)} results, reranked to {len(reranked)}")

            if reranked:
                top_result = reranked[0]
                print(f"Top result score: {top_result['score']:.3f}")
                print(f"Content preview: {top_result['content'][:100]}...")

        except Exception as e:
            print(f"✗ Retrieval failed for '{query}': {str(e)}")


def test_rag_chain():
    """Test the RAG chain."""
    print("\nTesting RAG chain...")

    test_questions = [
        "What is the difference between stocks and bonds?",
        "How does diversification reduce risk?",
    ]

    for question in test_questions:
        try:
            print(f"\nQuestion: {question}")
            response = rag_chain(question)
            print(f"Response: {response[:200]}...")

            # Check for source citations
            if "source" in response.lower() or "according to" in response.lower():
                print("✓ Response includes source references")
            else:
                print("⚠ Response may lack source citations")

        except Exception as e:
            print(f"✗ RAG chain failed for '{question}': {str(e)}")


def test_news_integration():
    """Test news integration."""
    print("\nTesting news integration...")

    try:
        news = fetch_financial_news()
        print(f"✓ Fetched {len(news)} news articles")

        if news:
            print(f"Sample article: {news[0]['title']}")

    except Exception as e:
        print(f"✗ News integration failed: {str(e)}")


def main():
    """Run all tests."""
    print("🚀 Starting AI Investment Tutor Tests")
    print("=" * 50)

    # Check environment variables
    required_env_vars = ["OPENAI_API_KEY", "PINECONE_API_KEY", "PINECONE_INDEX_NAME"]
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    if missing_vars:
        print(f"⚠ Missing environment variables: {', '.join(missing_vars)}")
        print("Some tests may fail without proper configuration.")

    # Run tests
    test_ingestion()
    print("=" * 30)
    test_retrieval()
    print("=" * 30)
    test_rag_chain()
    print("=" * 30)
    test_news_integration()

    print("\n" + "=" * 50)
    print("✅ Testing complete!")


if __name__ == "__main__":
    main()
