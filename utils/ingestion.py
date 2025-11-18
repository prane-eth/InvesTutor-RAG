import os
from pathlib import Path
from typing import Any, Dict, List

import requests
from bs4 import BeautifulSoup
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone as PineconeClient
from pypdf import PdfReader
from sentence_transformers import CrossEncoder

from utils.model_utils import embed_model_name, embeddings

# Initialize Pinecone
pc = PineconeClient(api_key=os.getenv("PINECONE_API_KEY"))
index_name = os.getenv("PINECONE_INDEX_NAME", "")
if not index_name:
    raise Exception("PINECONE_INDEX_NAME is not set in the environment variables")

# Create index if it doesn't exist
if index_name not in [idx.name for idx in pc.list_indexes()]:
    # Determine dimension based on embedding model
    if "GGUF" in embed_model_name:
        dimension = 4096  # Adjust based on your GGUF model
    else:
        dimension = 384 if "MiniLM" in embed_model_name else 1536
    
    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric="cosine",
        spec={
            "serverless": {
                "cloud": "aws",
                "region": "us-east-1"
            }
        }
    )

# Use LangChain's Pinecone wrapper
vectorstore = PineconeVectorStore(pc.Index(index_name), embeddings)

def load_pdf(file_path: str) -> str:
    """Load text from PDF file."""
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def load_html(url: str) -> str:
    """Load text from HTML URL."""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Remove script, style, nav, header, footer elements
    for element in soup(["script", "style", "nav", "header", "footer", "aside", "noscript"]):
        element.extract()

    # Try to find main content
    main_content = soup.find('main') or soup.find('article') or soup.find('div', class_='content') or soup.find('div', id='content')
    if main_content:
        text = main_content.get_text()
    else:
        text = soup.get_text()

    # Clean up whitespace
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    text = ' '.join(chunk for chunk in chunks if chunk and len(chunk) > 10)  # Filter out very short chunks

    return text

def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    """Split text into chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    return text_splitter.split_text(text)

def ingest_document(file_path: str, metadata: Dict[str, Any] | None = None) -> None:
    """Ingest a single document (PDF or HTML URL)."""
    global vectorstore

    if file_path.startswith('http'):
        text = load_html(file_path)
        doc_type = "html"
    else:
        text = load_pdf(file_path)
        doc_type = "pdf"

    chunks = chunk_text(text)

    # Add metadata
    metadatas = []
    for i, chunk in enumerate(chunks):
        chunk_metadata = {
            "source": file_path,
            "chunk_id": i,
            "doc_type": doc_type,
            "text": chunk[:500]  # Store first 500 chars as preview
        }
        if metadata:
            chunk_metadata.update(metadata)
        metadatas.append(chunk_metadata)

    # Add to vectorstore
    vectorstore.add_texts(chunks, metadatas=metadatas)
    print(f"Ingested {len(chunks)} chunks from {file_path}")

def ingest_directory(directory_path: str) -> None:
    """Ingest all PDFs in a directory."""
    path = Path(directory_path)
    for pdf_file in path.glob("*.pdf"):
        ingest_document(str(pdf_file))

if __name__ == "__main__":
    # Example usage
    # ingest_document("path/to/document.pdf")
    # ingest_directory("path/to/pdf/folder")
    urls = [
        'https://en.wikipedia.org/wiki/Investment',
        'https://www.investopedia.com/terms/i/investment.asp',
        'https://www.investopedia.com/terms/s/stock.asp',
        'https://www.investopedia.com/terms/b/bond.asp',
        'https://www.investopedia.com/terms/d/diversification.asp'
    ]
    for url in urls:
        try:
            ingest_document(url)
            print(f'Ingested: {url}')
        except Exception as e:
            print(f'Failed: {url} - {e}')

    from utils.retrieval import search_documents
    results = search_documents('What is investment?')
    print('Results:', len(results))
    print(results[0]['content'][:200] if results else 'No results')
