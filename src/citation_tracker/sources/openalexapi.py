"""OpenAlex API client for fetching paper citations."""

from __future__ import annotations

import logging
from typing import Any
from urllib.parse import quote

import httpx
from tenacity import RetryError, retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

OA_BASE = "https://api.openalex.org"


def _work_to_dict(w: dict[str, Any]) -> dict[str, Any]:
    doi_raw = w.get("doi") or ""
    doi = doi_raw.replace("https://doi.org/", "") if doi_raw else None
    oa_pdf_url: str | None = None
    best_oa = w.get("best_oa_location") or {}
    if best_oa.get("pdf_url"):
        oa_pdf_url = best_oa["pdf_url"]

    authors = ", ".join(
        (a.get("author") or {}).get("display_name", "")
        for a in (w.get("authorships") or [])
    )
    return {
        "title": w.get("display_name") or w.get("title"),
        "authors": authors,
        "year": w.get("publication_year"),
        "abstract": None,
        "doi": doi,
        "ss_id": None,
        "oa_id": w.get("id"),
        "pdf_url": oa_pdf_url,
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
            logger.warning("OpenAlex rate limit (429) hit for %s", url)
            raise
        raise


def search_paper_by_query(query: str) -> dict[str, Any] | None:
    """Search OpenAlex with a general query string."""
    try:
        data = _get(
            f"{OA_BASE}/works",
            {"search": query, "per_page": 1},
        )
        results = data.get("results") or []
        if not results:
            return None
        return _work_to_dict(results[0])
    except (RetryError, httpx.HTTPStatusError) as exc:
        logger.warning("OpenAlex query failed: %s", exc)
        return None


def get_paper_by_doi(doi: str) -> dict[str, Any] | None:
    """Fetch paper metadata from OpenAlex by DOI."""
    try:
        data = _get(f"{OA_BASE}/works/doi:{quote(doi, safe='')}", {})
        return _work_to_dict(data)
    except (RetryError, httpx.HTTPStatusError) as exc:
        logger.warning("OpenAlex get_doi failed: %s", exc)
        return None


def get_paper_by_arxiv(arxiv_id: str) -> dict[str, Any] | None:
    """Fetch paper metadata from OpenAlex by arXiv ID."""
    try:
        data = _get(f"{OA_BASE}/works", {"filter": f"ids.arxiv:{arxiv_id}"})
        results = data.get("results") or []
        if not results:
            return None
        return _work_to_dict(results[0])
    except (RetryError, httpx.HTTPStatusError) as exc:
        logger.warning("OpenAlex get_arxiv failed: %s", exc)
        return None


def get_citations(oa_id: str, max_results: int = 500) -> list[dict[str, Any]]:
    """Fetch papers citing the given OpenAlex work ID."""
    from tenacity import RetryError
    results: list[dict[str, Any]] = []
    page = 1
    per_page = 100
    while len(results) < max_results:
        try:
            data = _get(
                f"{OA_BASE}/works",
                {
                    "filter": f"cites:{oa_id}",
                    "per_page": per_page,
                    "page": page,
                    "select": "id,doi,display_name,publication_year,authorships,best_oa_location",
                },
            )
            items = data.get("results") or []
            results.extend(_work_to_dict(w) for w in items)
            if len(items) < per_page:
                break
            page += 1
        except (RetryError, httpx.HTTPStatusError) as exc:
            logger.warning("OpenAlex get_citations failed: %s", exc)
            break
    return results[:max_results]
