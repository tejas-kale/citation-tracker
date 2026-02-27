"""Tests and examples for the deduplicator.

The deduplicator merges paper lists from multiple API sources (Semantic Scholar,
OpenAlex, NASA ADS) that may overlap. It deduplicates by DOI first, then by
Semantic Scholar ID, then by ADS bibcode.

When two records refer to the same paper, fields from the *second* record fill
in any None fields in the *first* — the first record's non-None values win.
"""

from citation_tracker.sources.deduplicator import deduplicate, _merge_into


# ── _merge_into ────────────────────────────────────────────────────────────


def test_merge_into_fills_missing_fields():
    """Source fills None fields in target; existing values are not overwritten."""
    target = {"title": "Paper A", "authors": None, "year": None}
    source = {"title": "Different Title", "authors": "Smith et al.", "year": 2022}
    _merge_into(target, source)
    assert target["title"] == "Paper A"       # original kept
    assert target["authors"] == "Smith et al."  # filled in
    assert target["year"] == 2022               # filled in


def test_merge_into_ignores_none_source_values():
    """None values in source do not overwrite existing target values."""
    target = {"title": "Paper A", "year": 2020}
    source = {"title": None, "year": None}
    _merge_into(target, source)
    assert target["title"] == "Paper A"
    assert target["year"] == 2020


# ── deduplicate ────────────────────────────────────────────────────────────


def test_deduplicate_by_doi_keeps_one():
    """Two records with the same DOI (case-insensitive) become one."""
    papers = [
        {"doi": "10.1234/abc", "title": "Paper A", "ss_id": "ss1", "year": 2021},
        {"doi": "10.1234/ABC", "title": None, "ss_id": None, "year": None},
    ]
    result = deduplicate(papers)
    assert len(result) == 1
    assert result[0]["doi"] == "10.1234/abc"
    assert result[0]["title"] == "Paper A"  # first value kept
    assert result[0]["ss_id"] == "ss1"      # first value kept


def test_deduplicate_by_doi_merges_missing_fields():
    """When a duplicate DOI appears, its non-None fields fill gaps in the original."""
    papers = [
        {"doi": "10.5/x", "title": "Paper X", "ss_id": None, "year": None},
        {"doi": "10.5/x", "title": None, "ss_id": "ssX", "year": 2023},
    ]
    result = deduplicate(papers)
    assert len(result) == 1
    assert result[0]["title"] == "Paper X"   # original kept
    assert result[0]["ss_id"] == "ssX"        # filled from duplicate
    assert result[0]["year"] == 2023          # filled from duplicate


def test_deduplicate_by_ss_id_no_doi():
    """Papers without a DOI are deduplicated by Semantic Scholar ID."""
    papers = [
        {"doi": None, "ss_id": "ssABC", "title": "Paper A", "year": 2020},
        {"doi": None, "ss_id": "ssABC", "title": None, "year": None},
    ]
    result = deduplicate(papers)
    assert len(result) == 1
    assert result[0]["title"] == "Paper A"


def test_deduplicate_by_ads_bibcode():
    """Papers without DOI or SS ID are deduplicated by ADS bibcode."""
    papers = [
        {"doi": None, "ss_id": None, "ads_bibcode": "2023A&A...1A", "title": "ADS Paper"},
        {"doi": None, "ss_id": None, "ads_bibcode": "2023A&A...1A", "title": None},
    ]
    result = deduplicate(papers)
    assert len(result) == 1
    assert result[0]["title"] == "ADS Paper"


def test_deduplicate_different_dois_kept_separate():
    """Papers with different DOIs are all retained."""
    papers = [
        {"doi": "10.1/a", "title": "Paper A"},
        {"doi": "10.1/b", "title": "Paper B"},
        {"doi": "10.1/c", "title": "Paper C"},
    ]
    result = deduplicate(papers)
    assert len(result) == 3


def test_deduplicate_no_id_always_kept():
    """Papers with no DOI, ss_id, or bibcode are kept verbatim (no dedup key)."""
    papers = [
        {"doi": None, "ss_id": None, "ads_bibcode": None, "title": "Mystery Paper 1"},
        {"doi": None, "ss_id": None, "ads_bibcode": None, "title": "Mystery Paper 2"},
    ]
    result = deduplicate(papers)
    assert len(result) == 2


def test_deduplicate_cross_source_merge():
    """Realistic scenario: SS and OA both return the same paper, ADS adds a bibcode."""
    from_ss = {
        "doi": "10.1093/mnras/stac123",
        "title": "Stellar Formation in NGC 1234",
        "authors": "Smith, J.; Jones, K.",
        "year": 2022,
        "ss_id": "ss-xyz",
        "oa_id": None,
        "ads_bibcode": None,
        "pdf_url": "https://arxiv.org/pdf/2201.00001.pdf",
    }
    from_oa = {
        "doi": "10.1093/mnras/stac123",
        "title": None,
        "authors": None,
        "year": None,
        "ss_id": None,
        "oa_id": "W123456",
        "ads_bibcode": None,
        "pdf_url": None,
    }
    from_ads = {
        "doi": "10.1093/MNRAS/STAC123",  # uppercase — should still deduplicate
        "title": None,
        "authors": None,
        "year": None,
        "ss_id": None,
        "oa_id": None,
        "ads_bibcode": "2022MNRAS.stac123S",
        "pdf_url": None,
    }

    result = deduplicate([from_ss, from_oa, from_ads])

    assert len(result) == 1
    paper = result[0]
    assert paper["title"] == "Stellar Formation in NGC 1234"  # from SS
    assert paper["ss_id"] == "ss-xyz"                          # from SS
    assert paper["oa_id"] == "W123456"                         # merged from OA
    assert paper["ads_bibcode"] == "2022MNRAS.stac123S"        # merged from ADS
    assert paper["pdf_url"] == "https://arxiv.org/pdf/2201.00001.pdf"  # from SS
