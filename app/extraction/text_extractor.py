from __future__ import annotations

import io

import fitz
from docx import Document


class UnsupportedFileTypeError(ValueError):
    pass


def extract_text_from_pdf(content: bytes) -> tuple[str, int]:
    with fitz.open(stream=content, filetype="pdf") as document:
        page_count = document.page_count
        pages = [page.get_text("text") for page in document]
    text = "\n".join(pages)
    return text, page_count


def extract_text_from_docx(content: bytes) -> tuple[str, int]:
    document = Document(io.BytesIO(content))
    paragraphs = [p.text for p in document.paragraphs if p.text.strip()]
    text = "\n".join(paragraphs)
    page_count = max(1, len(paragraphs) // 30)
    return text, page_count


def extract_text(filename: str, content: bytes) -> tuple[str, int]:
    lower = filename.lower()
    if lower.endswith(".pdf"):
        return extract_text_from_pdf(content)
    if lower.endswith(".docx"):
        return extract_text_from_docx(content)
    raise UnsupportedFileTypeError("Only PDF and DOCX are supported")
