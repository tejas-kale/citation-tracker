"""Deduplicator: merge citing paper lists from multiple sources."""

from __future__ import annotations

from typing import Any


def deduplicate(papers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Merge a list of paper dicts, deduplicating by DOI (case-insensitive).

    Papers without a DOI are kept as-is (deduplicated by ss_id if available,
    otherwise kept verbatim).

    When two records refer to the same paper, the merge keeps all non-None
    fields from both, preferring the first occurrence for conflicts.
    """
    doi_map: dict[str, dict[str, Any]] = {}
    ss_map: dict[str, dict[str, Any]] = {}
    ads_map: dict[str, dict[str, Any]] = {}
    unique: list[dict[str, Any]] = []

    for paper in papers:
        doi = (paper.get("doi") or "").strip().lower()
        ss_id = paper.get("ss_id") or ""
        bibcode = paper.get("ads_bibcode") or ""

        if doi:
            if doi in doi_map:
                _merge_into(doi_map[doi], paper)
                continue
            doi_map[doi] = paper
            unique.append(paper)
        elif ss_id:
            if ss_id in ss_map:
                _merge_into(ss_map[ss_id], paper)
                continue
            ss_map[ss_id] = paper
            unique.append(paper)
        elif bibcode:
            if bibcode in ads_map:
                _merge_into(ads_map[bibcode], paper)
                continue
            ads_map[bibcode] = paper
            unique.append(paper)
        else:
            unique.append(paper)

    return unique


def _merge_into(target: dict[str, Any], source: dict[str, Any]) -> None:
    """Fill None fields in *target* with values from *source*."""
    for key, value in source.items():
        if target.get(key) is None and value is not None:
            target[key] = value
