# Functions to ingest documents (PDFs and HTML) and chunk them for vectorstore storage.

from pathlib import Path
from typing import Any, Dict

# from langchain_text_splitters import RecursiveCharacterTextSplitter
from semantic_chunkers import ConsecutiveChunker
from semantic_router.encoders import OpenAIEncoder
import tempfile

from investutor.utils.converter_utils import convert_to_md_file, fetch_url
from investutor.utils.retrieval_utils import (
    vectorstore,
    embed_api_key,
    embed_base_url,
    embed_model_name,
)


# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=1000,  # characters, not tokens
#     chunk_overlap=200,
#     length_function=len,
#     separators=["\n###", "\n##", "\n#", "\n\n", "\n"],
#     # Separate by markdown-based headings first, then paragraphs, then lines
# )

class SemanticChunker:
    def __init__(self) -> None:
        self.encoder = OpenAIEncoder(
            name=None, openai_api_key=embed_api_key, openai_base_url=embed_base_url
        )
        self.encoder.name = embed_model_name
        self.chunker = ConsecutiveChunker(encoder=self.encoder)

    def split_text(self, text: str) -> list[str]:
        chunks = self.chunker(docs=[text])
        all_values: list[str] = []
        for chunk in chunks:
            all_values.extend([" ".join(item.splits) for item in chunk])
        return all_values

text_splitter = SemanticChunker()


def ingest_md_document(
    file_path: str | Path, metadata: Dict[str, Any] | None = None
) -> None:
    """Ingest a single document (PDF or HTML URL)."""
    with open(file_path, encoding="utf-8") as f:
        text = f.read()

    chunks = text_splitter.split_text(text)

    # Add metadata
    metadatas = []
    for i, chunk in enumerate(chunks):
        if not chunk.strip():
            continue
        chunk_metadata = {
            "source": str(file_path),
            "chunk_id": i,
            "doc_type": "text/markdown",
            "text": chunk[:500],  # Store first 500 chars as preview
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
        convert_to_md_file(pdf_file, "pdf")
    for html_file in path.glob("*.html"):
        convert_to_md_file(html_file, "html")
    for docx_file in path.glob("*.docx"):
        convert_to_md_file(docx_file, "docx")

    for md_file in path.glob("*.md"):
        ingest_md_document(md_file)


def ingest_urls(urls: list[str]) -> None:
    """Ingest a list of URLs (HTML pages)."""
    print("Ingesting URLs...")
    # download the HTML content and save to temp files

    for url in urls:
        try:
            response = fetch_url(url)
            if response is None:
                print(f"Error downloading {url}: too many failed attempts")
                continue

            suffix = ".html"
            content_type = response.headers.get("Content-Type", "")
            if "pdf" in content_type:
                suffix = ".pdf"
            elif "msword" in content_type or "officedocument" in content_type:
                suffix = ".docx"
            elif "markdown" in content_type or "md" in content_type:
                suffix = ".md"

            with tempfile.NamedTemporaryFile(
                delete=False, prefix=str(hash(url)), suffix=suffix
            ) as tmp_file:
                tmp_file.write(response.content)
                tmp_file_path = tmp_file.name
                if not tmp_file_path:
                    raise ValueError(f"Failed to create temporary file for URL {url}")
        except Exception as e:
            print(f"Error downloading {url}: {e}")
            continue

        tmp_file_path = convert_to_md_file(tmp_file_path, suffix)

        try:
            ingest_md_document(tmp_file_path, metadata={"url": url})
        except Exception as e:
            print(f"Error ingesting {url} - {tmp_file_path}: {e}")


if __name__ == "__main__":
    # Example usage
    # ingest_document("path/to/document.pdf")
    # ingest_directory("path/to/pdf/folder")
    from retrieval_utils import search_documents

    urls = [
        "https://en.wikipedia.org/wiki/Investment",
        "https://www.investopedia.com/terms/i/investment.asp",
        "https://www.investopedia.com/terms/s/stock.asp",
        "https://www.investopedia.com/terms/b/bond.asp",
        "https://www.investopedia.com/terms/d/diversification.asp",
    ]
    ingest_urls(urls)

    results = search_documents("What is investment?")
    print("Results:", len(results))
    print(results[0]["content"][:200] if results else "No results")
