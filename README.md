# InvesTutor-RAG - AI Investment Tutor

[![AI](https://img.shields.io/badge/AI-C21B00?style=for-the-badge&logo=openaigym&logoColor=white)]()
[![LLMs](https://img.shields.io/badge/LLMs-1A535C?style=for-the-badge&logo=openai&logoColor=white)]()
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=ffdd54)]()
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-darkgreen.svg?style=for-the-badge&logo=github&logoColor=white)](./LICENSE.md)

An AI-powered investment education platform with RAG (Retrieval-Augmented Generation) capabilities. Ingests financial documents, provides personalized tutoring, and offers real-time market insights.

## Features

- 📄 **Document Ingestion**: Upload and process PDF documents and web pages
- 🧠 **Intelligent Q&A**: Ask questions about investments with source-backed answers
- 📈 **Market News Integration**: Stay updated with latest financial news
- 🎯 **Personalized Learning**: Topic-based tutoring with progress tracking
- 🔍 **Semantic Search**: Advanced retrieval with reranking
- 📚 **Source Citations**: "No source → no claim" logic ensures reliable information

## Architecture

- **Backend**: FastAPI server with OpenAI GPT integration
- **Frontend**: Streamlit web interface
- **Vector Store**: Pinecone for embeddings and retrieval
- **Embeddings**: OpenAI text-embedding-ada-002 or custom models
- **Chunking**: Recursive text splitting with semantic overlap

## Setup

### Prerequisites

- Python 3.8+
- OpenAI API key
- Pinecone account and API key
- (Optional) NewsAPI key for news features

### Installation

1. **Clone and setup environment:**
```bash
git clone https://github.com/prane-eth/InvesTutor-RAG
cd InvesTutor-RAG
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Configure environment variables:**

Create a `.env` file based on `.env.example`.

4. **Initialize Pinecone index:**
```bash
python -c "from ingestion import pc; print('Pinecone ready')"
```

## Usage

### Starting the Application

1. **Run the model server:**
```bash
python host_models.py
```

2. **Run the frontend (in another terminal):**
```bash
streamlit run utils/chat_server.py
```

3. **Access the application:**
   - Chatbot page: http://localhost:8501
   - Model API: http://localhost:8001

### Ingesting Documents

```python
from ingestion import ingest_document

# Ingest a PDF
ingest_document("path/to/investment_guide.pdf")

# Ingest a webpage
ingest_document("https://www.investopedia.com/terms/i/investment.asp")
```

### Asking Questions

```python
from utils.common_utils import rag_chain

response = rag_chain.invoke("What is diversification?")
print(response)
```

## Testing

Run the comprehensive test suite:

```bash
python test_system.py
```

## API Endpoints

- `GET /v1/models` - List available models
- `POST /v1/chat/completions` - Chat completions with RAG

## Key Components

- **ingestion.py**: Document processing and vector storage
- **utils/server.py**: FastAPI backend server
- **utils/chat_server.py**: Streamlit frontend
- **utils/common_utils.py**: LLM and RAG chain setup
- **news_integration.py**: Financial news fetching
- **test_system.py**: System validation tests
