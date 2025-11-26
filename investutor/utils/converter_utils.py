import logging
import tempfile
import time
from pathlib import Path

import mammoth
import requests
from markdownify import markdownify as mdify
from pypdf import PdfReader


def convert_pdf_to_md(pdf_path: str | Path) -> str:
    """Convert a PDF file to a markdown (`.md`) temporary file and return its path.

    Uses `pypdf` (already in project dependencies) to extract text. If `pypdf`
    isn't available, an ImportError will be raised.
    """
    text_parts: list[str] = []
    try:
        reader = PdfReader(pdf_path)
        for page in reader.pages:
            try:
                page_text = page.extract_text() or ""
            except Exception:
                page_text = ""
            if page_text:
                text_parts.append(page_text)
        return "\n\n".join(text_parts).strip()
    except Exception as e:
        raise RuntimeError(f"Failed to read PDF {pdf_path}: {e}")


def convert_html_to_md(html_path: str | Path) -> str:
    """Convert an HTML file to markdown (`.md`) and return the temp file path.

    Tries to use `markdownify`. If not installed, raises ImportError with guidance.
    """
    try:
        with open(html_path, encoding="utf-8", errors="ignore") as f:
            html = f.read()
        return mdify(html, heading_style="ATX")
    except Exception as e:
        raise RuntimeError(f"Failed to read HTML file {html_path}: {e}")


def convert_docx_to_md(docx_path: str | Path) -> str:
    """Convert a DOCX file to markdown (`.md`) and return the temp file path.

    Uses `mammoth` to convert DOCX -> HTML, then `markdownify` to convert HTML -> MD.
    """
    try:
        with open(docx_path, "rb") as docx_file:
            result = mammoth.convert_to_html(docx_file)
            html = result.value
        return mdify(html, heading_style="ATX")
    except Exception as e:
        raise RuntimeError(f"Failed to convert DOCX {docx_path} to HTML: {e}")


def preprocess_md_text(md_text: str) -> str:
    """Preprocess markdown text by removing extra whitespace and normalizing line breaks."""
    # Remove excessive blank lines
    lines = md_text.splitlines()
    cleaned_lines = []
    previous_line_blank = False
    for line in lines:
        stripped_line = line.rstrip()
        if not stripped_line:
            if not previous_line_blank:
                cleaned_lines.append("")
            previous_line_blank = True
        else:
            cleaned_lines.append(stripped_line)
            previous_line_blank = False
    return "\n".join(cleaned_lines).strip()


def convert_to_md_file(file_path: str | Path, format: str) -> Path:
    """Convert a file to markdown (`.md`) based on its format and return the temp file path.

    Supported formats: 'pdf', 'html', 'docx'.
    """
    format = format.lower().lstrip(".")
    if format == "md" or format == "markdown":
        with open(file_path, encoding="utf-8") as f:
            text = f.read()
        text = preprocess_md_text(text)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(text)
        return Path(file_path)

    if format == "pdf":
        text = convert_pdf_to_md(file_path)
    elif format == "html":
        text = convert_html_to_md(file_path)
    elif format == "docx" or format == "doc":
        text = convert_docx_to_md(file_path)
    else:
        raise ValueError(f"Unsupported format for conversion to markdown: {format}")

    text = preprocess_md_text(text)
    if not text:
        raise ValueError(f"No text extracted from {file_path} for conversion to markdown.")

    # Write to a temporary markdown file
    with tempfile.NamedTemporaryFile(
        delete=False, prefix=f"{format}_to_md_", suffix=".md",
        mode="w", encoding="utf-8"
    ) as tmp:
        tmp.write(text)
        return Path(tmp.name)


headers = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/117.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}

def fetch_url(u: str, max_retries: int = 3, timeout: int = 15):
    session = requests.Session()
    for attempt in range(1, max_retries + 1):
        try:
            resp = session.get(u, headers=headers, timeout=timeout, allow_redirects=True)
            resp.raise_for_status()
            return resp
        except requests.RequestException as e:
            if "403" in str(e) or "404" in str(e):
                logging.error("Request error for %s: %s", u, e)
                return None
            logging.warning("Request error for %s on attempt %s: %s", u, attempt, e)
        # backoff
        time.sleep(2 ** (attempt - 1))
    return None
