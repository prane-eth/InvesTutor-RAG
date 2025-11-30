import os
from datetime import datetime, timedelta
import tempfile
from typing import Any, Dict, List

import requests
from bs4 import BeautifulSoup
from newsapi import NewsApiClient

from investutor.utils.llm_utils import llm

# News API integration for financial news
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")
FINANCIAL_NEWS_SOURCES = [
    "bloomberg",
    "reuters",
    "financial-times",
    "the-wall-street-journal",
    "cnbc",
    "marketwatch",
    "investopedia",
    "seeking-alpha",
]


def fetch_financial_news(
    query: str = "investments OR stocks OR bonds OR markets", days_back: int = 1
) -> List[Dict[str, Any]]:
    """Fetch recent financial news articles."""
    if not NEWS_API_KEY:
        return []

    from_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")

    try:
        newsapi = NewsApiClient(api_key=NEWS_API_KEY)
        data = newsapi.get_everything(
            q=query,
            from_param=from_date,
            language="en",
            sort_by="relevancy",
            page_size=10,
        )

        articles = []
        for article in data.get("articles", []):
            articles.append(
                {
                    "title": article.get("title", ""),
                    "description": article.get("description", ""),
                    "url": article.get("url", ""),
                    "source": article.get("source", {}).get("name", ""),
                    "published_at": article.get("publishedAt", ""),
                    "content": (
                        article.get("content", "")[:500]
                        if article.get("content")
                        else ""
                    ),
                }
            )

        return articles
    except Exception as e:
        print(f"Error fetching news: {str(e)}")
        return []


def summarize_news_article(url: str) -> str:
    """Summarize a news article using LLM."""
    try:
        # Fetch article content
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, "html.parser")

        # Remove unwanted elements
        for element in soup(
            ["script", "style", "nav", "header", "footer", "aside", "noscript"]
        ):
            element.extract()

        # Try to find main content
        main_content = (
            soup.find("main")
            or soup.find("article")
            or soup.find("div", class_="content")
            or soup.find("div", id="content")
        )
        if main_content:
            text = main_content.get_text()
        else:
            text = soup.get_text()

        # Clean text
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        article_text = " ".join(chunk for chunk in chunks if chunk and len(chunk) > 20)[
            :2000
        ]  # Limit length

        if not article_text:
            return "Could not extract article content."

        # Use LLM to summarize
        summary_prompt = f"""
        Summarize this news article in 2-3 sentences, focusing on key financial/investment implications:
        
        {article_text}
        
        Summary:"""

        summary = llm.invoke(summary_prompt).content.strip()
        return summary

    except Exception as e:
        return f"Error summarizing article: {str(e)}"


def get_market_sentiment() -> Dict[str, Any]:
    """Get overall market sentiment based on recent news."""
    try:
        # Fetch recent news
        news = fetch_financial_news(days_back=1)

        if not news:
            return {"overall": "neutral", "confidence": 0.5, "key_themes": []}

        # Analyze sentiment using LLM
        news_text = "\n".join(
            [f"- {article['title']}: {article['description']}" for article in news[:5]]
        )

        sentiment_prompt = f"""
        Analyze the sentiment of these recent financial news headlines and provide:
        1. Overall market sentiment (bullish/bearish/neutral)
        2. Confidence level (0-1)
        3. Key themes emerging
        
        Headlines:
        {news_text}
        
        Analysis:"""

        analysis = llm.invoke(sentiment_prompt).content.strip()

        # Parse response (simplified)
        if "bullish" in analysis.lower():
            sentiment = "bullish"
        elif "bearish" in analysis.lower():
            sentiment = "bearish"
        else:
            sentiment = "neutral"

        return {
            "overall": sentiment,
            "confidence": 0.7,
            "key_themes": ["market analysis", "investment news"],
            "analysis": analysis,
        }

    except Exception as e:
        return {
            "overall": "neutral",
            "confidence": 0.5,
            "key_themes": [],
            "error": str(e),
        }


def ingest_news_to_vectorstore(
    query: str = "investments OR stocks OR bonds OR markets", days_back: int = 1
):
    """Fetch news and ingest summaries into vectorstore."""
    from investutor.utils.ingestion import ingest_md_document

    news = fetch_financial_news(query, days_back)

    for article in news:
        if article["url"] and article["title"]:
            # Create a summary document
            summary = summarize_news_article(article["url"])

            news_content = f"""
            Title: {article['title']}
            Source: {article['source']}
            Published: {article['published_at']}
            
            Summary: {summary}
            
            Original Description: {article['description']}
            """

            # Ingest as if it's a document
            metadata = {
                "source": article["url"],
                "doc_type": "news",
                "published_at": article["published_at"],
                "source_name": article["source"],
            }

            # For now, save as temporary file and ingest
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt", delete=False
            ) as f:
                f.write(news_content)
                temp_file = f.name

            try:
                ingest_md_document(temp_file, metadata)
                print(f"Ingested news: {article['title']}")
            except Exception as e:
                print(f"Failed to ingest news {article['title']}: {e}")
            finally:
                os.unlink(temp_file)


if __name__ == "__main__":
    # Test news fetching
    news = fetch_financial_news()
    print(f"Fetched {len(news)} articles")
    for article in news[:3]:
        print(f"- {article['title']}")
