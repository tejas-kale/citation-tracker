"""PDF text extraction using PyMuPDF4LLM."""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def extract_text(pdf_path: Path) -> str | None:
    """
    Extract Markdown-formatted text from a PDF file using PyMuPDF4LLM.

    Returns the extracted text, or None if extraction fails.
    """
    try:
        import pymupdf4llm  # type: ignore[import-untyped]

        text: str = pymupdf4llm.to_markdown(str(pdf_path))
        return text
    except Exception as exc:
        logger.warning("Failed to extract text from %s: %s", pdf_path, exc)
        return None
