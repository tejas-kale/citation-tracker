"""Paper metadata resolution from URLs, DOIs, and Semantic Scholar IDs."""

from __future__ import annotations

import logging
import re
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

logger = logging.getLogger(__name__)


_STOP_WORDS = {"a", "an", "the", "of", "in", "and", "or", "for", "to", "on", "is", "are"}


def _clean_query(text: str, max_len: int = 200) -> str:
    """Clean extracted text for use as a search query."""
    text = re.sub(r"[#*_\[\]()]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_len]


def _titles_match(query_title: str, result_title: str | None, threshold: float = 0.4) -> bool:
    """Return True if result_title shares enough words with query_title.

    Uses word-overlap Jaccard similarity after stripping punctuation and stop words.
    A threshold of 0.4 means at least 40% of the smaller title's words overlap.
    """
    if not result_title:
        return False

    def _words(text: str) -> set[str]:
        text = re.sub(r"[^\w\s]", " ", text.lower())
        return {w for w in text.split() if w not in _STOP_WORDS and len(w) > 1}

    q_words = _words(query_title)
    r_words = _words(result_title)
    if not q_words or not r_words:
        return False
    overlap = len(q_words & r_words) / min(len(q_words), len(r_words))
    return overlap >= threshold


def _title_from_pdf_markdown(text: str) -> str | None:
    """Extract paper title from markdown-formatted PDF text.

    Looks for the first markdown heading that looks like a title (not a section
    number like "1 Introduction"). Returns None if no suitable heading is found.
    """
    for line in text.splitlines()[:80]:
        line = line.strip()
        if not line:
            continue
        if line.startswith("#"):
            candidate = re.sub(r"^#+\s*", "", line).strip()
            # Skip section headings like "1 Introduction", "2.1 Methods"
            if re.match(r"^\d", candidate):
                continue
            if len(candidate) > 10:
                return candidate
    return None


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

    filename = unquote(Path(urlparse(url).path).name)
    title_guess = filename
    for ext in [".pdf", ".html", ".htm"]:
        if title_guess.lower().endswith(ext):
            title_guess = title_guess[: -len(ext)]
    title_guess = title_guess.replace("_", " ").replace("-", " ")
    query = _clean_query(title_guess)

    # For short/ambiguous filenames or hash-like names (no spaces), peek at PDF content for a better query
    pdf_text: str | None = None
    title: str | None = None
    if len(title_guess) < 30 or " " not in title_guess:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = download_pdf(url, Path(tmpdir), filename_hint="resolve")
            if tmp_path:
                pdf_text = extract_text(tmp_path)
                if pdf_text:
                    title = _title_from_pdf_markdown(pdf_text)
                    query = _clean_query(title) if title else _clean_query(pdf_text, max_len=200)

    paper = (
        ss.search_paper_by_query(query)
        or oa.search_paper_by_query(query)
        or ads.search_paper_by_query(query, cfg.ads.api_key)
    )

    # Reject the result if its title doesn't match the filename or PDF title
    # (guards against wrong-paper matches when the API returns unrelated results)
    compare_title = title or title_guess
    if paper and not _titles_match(compare_title, paper.get("title")):
        logger.warning(
            "Search result title mismatch: queried %r, got %r — falling back to LLM",
            title,
            paper.get("title"),
        )
        paper = None

    if paper:
        if paper.get("doi"):
            official = _resolve_by_doi(paper["doi"], cfg)
            if official:
                official["source_url"] = url
                return official
        paper["source_url"] = url
        return paper

    if pdf_text:
        return _resolve_from_pdf_text(pdf_text, title or title_guess, url, cfg)

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
    filename = unquote(Path(urlparse(url).path).name)
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


def resolve_from_stored_text(
    text: str, hint: str, url: str, cfg: Any
) -> dict[str, Any] | None:
    """Resolve paper metadata from already-extracted PDF text.

    Tries API search first (Semantic Scholar, OpenAlex, ADS), then falls back
    to LLM extraction. Used when a PDF was ingested without a DOI.

    Args:
        text: Full extracted text of the PDF.
        hint: Fallback title hint (e.g. filename stem).
        url: Source URL or local path to store as source_url/pdf_url.
        cfg: Application config.

    Returns:
        A paper metadata dict or None if resolution fails.
    """
    from citation_tracker.sources import semantic_scholar as ss
    from citation_tracker.sources import openalexapi as oa
    from citation_tracker.sources import adsapi as ads

    extracted_title = _title_from_pdf_markdown(text)
    query = _clean_query(extracted_title) if extracted_title else _clean_query(text, max_len=200)

    paper = (
        ss.search_paper_by_query(query)
        or oa.search_paper_by_query(query)
        or ads.search_paper_by_query(query, cfg.ads.api_key)
    )

    if paper and extracted_title and not _titles_match(extracted_title, paper.get("title")):
        logger.warning(
            "Search result title mismatch: queried %r, got %r — falling back to LLM",
            extracted_title,
            paper.get("title"),
        )
        paper = None

    if paper:
        if paper.get("doi"):
            official = _resolve_by_doi(paper["doi"], cfg)
            if official:
                official["source_url"] = url
                return official
        paper["source_url"] = url
        return paper

    return _resolve_from_pdf_text(text, hint, url, cfg)


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
