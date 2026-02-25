"""Semantic Scholar API client for fetching paper citations."""

from __future__ import annotations

import logging
from typing import Any

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

SS_BASE = "https://api.semanticscholar.org/graph/v1"
FIELDS = "title,authors,year,abstract,externalIds,openAccessPdf"


def _paper_to_dict(p: dict[str, Any]) -> dict[str, Any]:
    doi = (p.get("externalIds") or {}).get("DOI")
    oa_pdf = p.get("openAccessPdf") or {}
    return {
        "title": p.get("title"),
        "authors": ", ".join(a.get("name", "") for a in (p.get("authors") or [])),
        "year": p.get("year"),
        "abstract": p.get("abstract"),
        "doi": doi,
        "ss_id": p.get("paperId"),
        "oa_id": None,
        "pdf_url": oa_pdf.get("url"),
    }


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def _get(url: str, params: dict[str, Any]) -> Any:
    try:
        with httpx.Client(timeout=30) as client:
            resp = client.get(url, params=params)
            resp.raise_for_status()
            return resp.json()
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 429:
            logger.warning("Semantic Scholar rate limit (429) hit for %s", url)
            # If we've exhausted retries, tenacity will re-raise the exception.
            # But we can catch it here if we want to return a sentinel.
            raise
        raise


def search_paper_by_title(title: str) -> dict[str, Any] | None:
    """Search Semantic Scholar for a paper by title, return best match."""
    return search_paper_by_query(title)


def search_paper_by_query(query: str) -> dict[str, Any] | None:
    """Search Semantic Scholar with a general query string."""
    from tenacity import RetryError
    try:
        data = _get(
            f"{SS_BASE}/paper/search",
            {"query": query, "fields": FIELDS, "limit": 1},
        )
        items = data.get("data") or []
        if not items:
            return None
        return _paper_to_dict(items[0])
    except (RetryError, httpx.HTTPStatusError) as exc:
        logger.warning("Semantic Scholar query failed: %s", exc)
        return None


def get_paper_by_id(ss_id: str) -> dict[str, Any] | None:
    """Fetch paper metadata from Semantic Scholar by SS paper ID."""
    from tenacity import RetryError
    try:
        data = _get(f"{SS_BASE}/paper/{ss_id}", {"fields": FIELDS})
        return _paper_to_dict(data)
    except (RetryError, httpx.HTTPStatusError) as exc:
        logger.warning("Semantic Scholar get_id failed: %s", exc)
        return None


def get_paper_by_doi(doi: str) -> dict[str, Any] | None:
    """Fetch paper metadata from Semantic Scholar by DOI."""
    from tenacity import RetryError
    try:
        data = _get(f"{SS_BASE}/paper/DOI:{doi}", {"fields": FIELDS})
        return _paper_to_dict(data)
    except (RetryError, httpx.HTTPStatusError) as exc:
        logger.warning("Semantic Scholar get_doi failed: %s", exc)
        return None


def get_paper_by_arxiv(arxiv_id: str) -> dict[str, Any] | None:
    """Fetch paper metadata from Semantic Scholar by arXiv ID."""
    from tenacity import RetryError
    try:
        data = _get(f"{SS_BASE}/paper/ARXIV:{arxiv_id}", {"fields": FIELDS})
        return _paper_to_dict(data)
    except (RetryError, httpx.HTTPStatusError) as exc:
        logger.warning("Semantic Scholar get_arxiv failed: %s", exc)
        return None


def get_citations(ss_id: str, limit: int = 500) -> list[dict[str, Any]]:
    """Fetch all papers citing the given Semantic Scholar paper ID."""
    from tenacity import RetryError
    results: list[dict[str, Any]] = []
    offset = 0
    while True:
        try:
            data = _get(
                f"{SS_BASE}/paper/{ss_id}/citations",
                {
                    "fields": FIELDS,
                    "limit": min(limit, 1000),
                    "offset": offset,
                },
            )
            items = data.get("data") or []
            for item in items:
                citing = item.get("citingPaper") or {}
                if citing:
                    results.append(_paper_to_dict(citing))
            if len(items) < min(limit, 1000):
                break
            offset += len(items)
            if offset >= limit:
                break
        except (RetryError, httpx.HTTPStatusError) as exc:
            logger.warning("Semantic Scholar get_citations failed: %s", exc)
            break
    return results
