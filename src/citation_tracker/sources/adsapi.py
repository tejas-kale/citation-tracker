"""NASA ADS API client for fetching paper citations."""

from __future__ import annotations

import logging
from typing import Any
from urllib.parse import quote

import httpx
from tenacity import RetryError, retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

ADS_BASE = "https://api.adsabs.harvard.edu/v1"


def _doc_to_dict(doc: dict[str, Any]) -> dict[str, Any]:
    doi_list = doc.get("doi") or []
    doi = doi_list[0] if doi_list else None
    
    # ADS returns authors as a list of strings
    authors = ", ".join(doc.get("author") or [])
    
    # Try to find a PDF link in the 'property' or 'esource'
    bibcode = doc.get("bibcode")
    pdf_url = (
        f"https://ui.adsabs.harvard.edu/link_gateway/{bibcode}/EPRINT_PDF"
        if bibcode
        else None
    )

    return {
        "title": (doc.get("title") or ["Untitled"])[0],
        "authors": authors,
        "year": int(doc.get("year")) if doc.get("year") else None,
        "abstract": doc.get("abstract"),
        "doi": doi,
        "ss_id": None,
        "oa_id": None,
        "ads_bibcode": bibcode,
        "pdf_url": pdf_url,
    }


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def _get(url: str, params: dict[str, Any], token: str) -> Any:
    if not token:
        logger.warning("ADS API token missing, skipping request.")
        return None
        
    headers = {"Authorization": f"Bearer {token}"}
    try:
        with httpx.Client(timeout=30, headers=headers) as client:
            resp = client.get(url, params=params)
            resp.raise_for_status()
            return resp.json()
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 429:
            logger.warning("ADS rate limit (429) hit")
            raise
        if exc.response.status_code == 401:
            logger.error("ADS API token invalid or unauthorized")
            return None
        raise


def get_paper_by_doi(doi: str, token: str) -> dict[str, Any] | None:
    """Fetch paper metadata from ADS by DOI."""
    try:
        query = f'doi:"{doi}"'
        data = _get(
            f"{ADS_BASE}/search/query",
            {"q": query, "fl": "bibcode,title,author,year,abstract,doi", "rows": 1},
            token
        )
        if not data:
            return None
        docs = data.get("response", {}).get("docs") or []
        if not docs:
            return None
        return _doc_to_dict(docs[0])
    except (RetryError, httpx.HTTPStatusError) as exc:
        logger.warning("ADS get_paper_by_doi failed: %s", exc)
        return None


def get_citations(
    bibcode: str, token: str, max_results: int = 500
) -> list[dict[str, Any]]:
    """Fetch papers citing the given ADS bibcode."""
    from tenacity import RetryError
    results: list[dict[str, Any]] = []
    try:
        query = f"citations({bibcode})"
        data = _get(
            f"{ADS_BASE}/search/query",
            {
                "q": query,
                "fl": "bibcode,title,author,year,abstract,doi",
                "rows": max_results,
                "sort": "date desc"
            },
            token
        )
        if not data:
            return []
        docs = data.get("response", {}).get("docs") or []
        for doc in docs:
            results.append(_doc_to_dict(doc))
    except (RetryError, httpx.HTTPStatusError) as exc:
        logger.warning("ADS get_citations failed: %s", exc)
    return results


def search_paper_by_query(query: str, token: str) -> dict[str, Any] | None:
    """Search ADS with a general query string."""
    try:
        data = _get(
            f"{ADS_BASE}/search/query",
            {"q": query, "fl": "bibcode,title,author,year,abstract,doi", "rows": 1},
            token
        )
        if not data:
            return None
        docs = data.get("response", {}).get("docs") or []
        if not docs:
            return None
        return _doc_to_dict(docs[0])
    except (RetryError, httpx.HTTPStatusError) as exc:
        logger.warning("ADS query failed: %s", exc)
        return None
