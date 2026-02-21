"""PDF downloader and manual folder watcher."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_HEADERS = {"User-Agent": "citation-tracker/0.1 (academic research tool)"}


def _doi_to_path(doi: str) -> str:
    """Convert a DOI to a safe directory name."""
    return re.sub(r"[^a-zA-Z0-9_\-.]", "_", doi)


def download_pdf(
    pdf_url: str,
    pdfs_dir: Path,
    doi: str | None = None,
    filename_hint: str | None = None,
) -> Path | None:
    """
    Attempt to download a PDF from *pdf_url*.

    Saves to *pdfs_dir*/<doi-or-random>/<filename>.pdf.
    Returns the local path on success, None on failure.
    """
    subdir_name = _doi_to_path(doi) if doi else (filename_hint or "unknown")
    subdir = pdfs_dir / subdir_name
    subdir.mkdir(parents=True, exist_ok=True)
    dest = subdir / "paper.pdf"

    if dest.exists():
        logger.debug("PDF already downloaded: %s", dest)
        return dest

    try:
        with httpx.Client(timeout=60, follow_redirects=True, headers=_HEADERS) as client:
            resp = client.get(pdf_url)
            resp.raise_for_status()
            content_type = resp.headers.get("content-type", "")
            if "pdf" not in content_type and not pdf_url.lower().endswith(".pdf"):
                logger.warning(
                    "URL %s returned content-type %s, skipping", pdf_url, content_type
                )
                return None
            dest.write_bytes(resp.content)
            logger.info("Downloaded PDF to %s", dest)
            return dest
    except Exception as exc:
        logger.warning("Failed to download PDF from %s: %s", pdf_url, exc)
        return None


def try_download_citing_paper(
    paper: dict[str, Any], pdfs_dir: Path, email: str = "citation-tracker@example.com"
) -> Path | None:
    """
    Try all known PDF URLs for a citing paper dict.
    Returns local path if any succeeds, None otherwise.
    """
    doi = paper.get("doi")
    url = paper.get("pdf_url")
    if url:
        path = download_pdf(url, pdfs_dir, doi=doi)
        if path:
            return path

    # Fallback: try Unpaywall if DOI is known
    if doi:
        path = _try_unpaywall(doi, pdfs_dir, email=email)
        if path:
            return path

    return None


def _try_unpaywall(doi: str, pdfs_dir: Path, email: str = "citation-tracker@example.com") -> Path | None:
    """Try to get a PDF via the Unpaywall API."""
    try:
        url = f"https://api.unpaywall.org/v2/{doi}?email={email}"
        with httpx.Client(timeout=15) as client:
            resp = client.get(url)
            resp.raise_for_status()
            data = resp.json()
        best = data.get("best_oa_location") or {}
        pdf_url = best.get("url_for_pdf")
        if pdf_url:
            return download_pdf(pdf_url, pdfs_dir, doi=doi)
    except Exception as exc:
        logger.debug("Unpaywall lookup failed for %s: %s", doi, exc)
    return None


def scan_manual_dir(manual_dir: Path) -> list[Path]:
    """Return all PDF files in the manual directory."""
    if not manual_dir.exists():
        return []
    return list(manual_dir.glob("*.pdf"))
