"""Paper metadata resolution from URLs, DOIs, and Semantic Scholar IDs."""

from __future__ import annotations

import logging
import re
import tempfile
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _clean_query(text: str, max_len: int = 200) -> str:
    """Clean extracted text for use as a search query."""
    text = re.sub(r"[#*_\[\]()]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_len]


def _merge_source_results(
    doi: str,
    ss_result: dict[str, Any] | None,
    oa_result: dict[str, Any] | None,
    ads_result: dict[str, Any] | None,
) -> dict[str, Any]:
    """Merge paper metadata from multiple sources into a single dict.

    Prefers the first non-None value for each field.
    """
    _ss = ss_result or {}
    _oa = oa_result or {}
    _ads = ads_result or {}
    return {
        "title": _ss.get("title") or _oa.get("title") or _ads.get("title"),
        "authors": _ss.get("authors") or _oa.get("authors") or _ads.get("authors"),
        "year": _ss.get("year") or _oa.get("year") or _ads.get("year"),
        "abstract": _ss.get("abstract") or _oa.get("abstract") or _ads.get("abstract"),
        "doi": doi,
        "ss_id": _ss.get("ss_id"),
        "oa_id": _oa.get("oa_id"),
        "ads_bibcode": _ads.get("ads_bibcode"),
        "pdf_url": _ss.get("pdf_url") or _oa.get("pdf_url") or _ads.get("pdf_url"),
        "source_url": None,
    }


def _resolve_by_doi(doi: str, cfg: Any) -> dict[str, Any] | None:
    """Fetch and merge metadata for a DOI from all configured sources.

    Returns a merged dict or None if no source found the paper.
    """
    from citation_tracker.sources import semantic_scholar as ss
    from citation_tracker.sources import openalexapi as oa
    from citation_tracker.sources import adsapi as ads

    ss_result = ss.get_paper_by_doi(doi)
    oa_result = oa.get_paper_by_doi(doi)
    ads_result = ads.get_paper_by_doi(doi, cfg.ads.api_key)

    if not ss_result and not oa_result and not ads_result:
        return None

    return _merge_source_results(doi, ss_result, oa_result, ads_result)


def _resolve_by_ss_id(ss_id: str) -> dict[str, Any] | None:
    """Fetch paper metadata from Semantic Scholar by paper ID."""
    from citation_tracker.sources import semantic_scholar as ss

    paper = ss.get_paper_by_id(ss_id)
    if paper:
        paper["source_url"] = None
    return paper


def _extract_doi_from_url(url: str) -> str | None:
    """Extract a DOI from a URL string, returning it or None."""
    match = re.search(r"10\.\d{4,}/[^\s\"'<>]+", url)
    return match.group() if match else None


def _extract_arxiv_id_from_url(url: str) -> str | None:
    """Extract an arXiv paper ID from a URL string, returning it or None."""
    pattern = r"(\d{4}\.\d{4,5}(v\d+)?|arxiv:[a-z\-]+(\.[a-z\-]+)?/\d{7})"
    match = re.search(pattern, url.lower())
    return match.group(1) if match else None


def _resolve_by_arxiv_id(arxiv_id: str, source_url: str) -> dict[str, Any] | None:
    """Fetch paper metadata for an arXiv ID from SS or OA."""
    from citation_tracker.sources import semantic_scholar as ss
    from citation_tracker.sources import openalexapi as oa

    paper = ss.get_paper_by_arxiv(arxiv_id) or oa.get_paper_by_arxiv(arxiv_id)
    if paper:
        paper["source_url"] = source_url
    return paper


def _resolve_from_pdf(url: str, cfg: Any) -> dict[str, Any] | None:
    """Download a PDF and search for its metadata via APIs, then LLM fallback.

    Returns a paper dict or None if resolution fails entirely.
    """
    from citation_tracker.sources import semantic_scholar as ss
    from citation_tracker.sources import openalexapi as oa
    from citation_tracker.sources import adsapi as ads
    from citation_tracker.fetcher import download_pdf
    from citation_tracker.parser import extract_text

    filename = Path(url).name
    title_guess = filename
    for ext in [".pdf", ".html", ".htm"]:
        if title_guess.lower().endswith(ext):
            title_guess = title_guess[: -len(ext)]
    title_guess = title_guess.replace("_", " ").replace("-", " ")
    query = _clean_query(title_guess)

    # For short/ambiguous filenames, peek at PDF content for a better query
    pdf_text: str | None = None
    if len(title_guess) < 30:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = download_pdf(url, Path(tmpdir), filename_hint="resolve")
            if tmp_path:
                pdf_text = extract_text(tmp_path)
                if pdf_text:
                    query = _clean_query(pdf_text, max_len=200)

    paper = (
        ss.search_paper_by_query(query)
        or oa.search_paper_by_query(query)
        or ads.search_paper_by_query(query, cfg.ads.api_key)
    )

    if paper:
        if paper.get("doi"):
            official = _resolve_by_doi(paper["doi"], cfg)
            if official:
                official["source_url"] = url
                return official
        paper["source_url"] = url
        return paper

    if pdf_text:
        return _resolve_from_pdf_text(pdf_text, title_guess, url, cfg)

    return None


def _resolve_from_pdf_text(
    pdf_text: str, title_guess: str, url: str, cfg: Any
) -> dict[str, Any] | None:
    """Use the LLM to extract metadata from PDF text, with API re-resolution.

    Returns a paper dict or None.
    """
    from citation_tracker.analyser import parse_paper_metadata

    try:
        logger.info("API search failed; extracting metadata from PDF using LLM...")
        extracted = parse_paper_metadata(pdf_text, cfg)

        if extracted.get("doi"):
            logger.info("LLM extracted DOI: %s. Re-resolving...", extracted["doi"])
            official = _resolve_by_doi(extracted["doi"], cfg)
            if official:
                official["source_url"] = url
                return official

        return {
            "doi": extracted.get("doi"),
            "title": extracted.get("title") or title_guess,
            "authors": extracted.get("authors"),
            "year": extracted.get("year"),
            "abstract": extracted.get("abstract"),
            "source_url": url,
            "ss_id": None,
            "oa_id": None,
            "pdf_url": url,
        }
    except Exception as exc:
        logger.warning("LLM metadata extraction failed: %s", exc)
        return None


def _resolve_by_url(url: str, cfg: Any) -> dict[str, Any] | None:
    """Resolve paper metadata from a URL.

    Tries in order: DOI extraction, arXiv ID extraction, PDF download + search.
    Falls back to a stub entry if nothing else works.
    """
    doi = _extract_doi_from_url(url)
    if doi:
        paper = _resolve_by_doi(doi, cfg)
        if paper:
            paper["source_url"] = url
        return paper

    arxiv_id = _extract_arxiv_id_from_url(url)
    if arxiv_id:
        paper = _resolve_by_arxiv_id(arxiv_id, url)
        if paper:
            return paper

    paper = _resolve_from_pdf(url, cfg)
    if paper:
        return paper

    # Stub entry — we at least know the URL
    filename = Path(url).name
    title_guess = filename
    for ext in [".pdf", ".html", ".htm"]:
        if title_guess.lower().endswith(ext):
            title_guess = title_guess[: -len(ext)]
    title_guess = title_guess.replace("_", " ").replace("-", " ")

    return {
        "doi": None,
        "title": title_guess,
        "authors": None,
        "year": None,
        "abstract": None,
        "source_url": url,
        "ss_id": None,
        "oa_id": None,
        "pdf_url": url,
    }


def resolve_paper(
    url: str | None = None,
    doi: str | None = None,
    ss_id: str | None = None,
    cfg: Any = None,
) -> dict[str, Any] | None:
    """Resolve paper metadata from any combination of URL, DOI, or SS ID.

    Args:
        url: A URL (paper page, PDF, arXiv, journal) to resolve.
        doi: A DOI string.
        ss_id: A Semantic Scholar paper ID.
        cfg: Application config object.

    Returns:
        A paper metadata dict or None if resolution failed.
    """
    if doi:
        return _resolve_by_doi(doi, cfg)
    if ss_id:
        return _resolve_by_ss_id(ss_id)
    if url:
        return _resolve_by_url(url, cfg)
    return None
