"""Tests for paper metadata resolution."""

from citation_tracker import resolver


def test_clean_query_strips_markdown():
    result = resolver._clean_query("## **[Paper Title]** (2023)")
    assert "#" not in result
    assert "*" not in result
    assert "[" not in result


def test_clean_query_truncates():
    long_text = "word " * 100
    result = resolver._clean_query(long_text, max_len=20)
    assert len(result) <= 20


def test_extract_doi_from_url_found():
    url = "https://doi.org/10.1234/some.paper.2023"
    doi = resolver._extract_doi_from_url(url)
    assert doi == "10.1234/some.paper.2023"


def test_extract_doi_from_url_not_found():
    url = "https://arxiv.org/abs/2301.00001"
    assert resolver._extract_doi_from_url(url) is None


def test_extract_arxiv_id_from_url_new_format():
    url = "https://arxiv.org/abs/2301.00001"
    arxiv_id = resolver._extract_arxiv_id_from_url(url)
    assert arxiv_id == "2301.00001"


def test_extract_arxiv_id_from_url_with_version():
    url = "https://arxiv.org/abs/2301.00001v2"
    arxiv_id = resolver._extract_arxiv_id_from_url(url)
    assert arxiv_id == "2301.00001v2"


def test_extract_arxiv_id_from_url_not_found():
    url = "https://example.com/paper.pdf"
    assert resolver._extract_arxiv_id_from_url(url) is None


def test_merge_source_results_prefers_first():
    ss = {"title": "SS Title", "authors": "A, B", "year": 2022, "ss_id": "ss1",
          "oa_id": None, "ads_bibcode": None, "pdf_url": "https://arxiv.org/pdf/x"}
    oa = {"title": None, "authors": None, "year": None, "ss_id": None,
          "oa_id": "W123", "ads_bibcode": None, "pdf_url": None}
    ads = {"title": None, "authors": None, "year": None, "ss_id": None,
           "oa_id": None, "ads_bibcode": "2022Test", "pdf_url": None}

    merged = resolver._merge_source_results("10.1/x", ss, oa, ads)

    assert merged["doi"] == "10.1/x"
    assert merged["title"] == "SS Title"
    assert merged["ss_id"] == "ss1"
    assert merged["oa_id"] == "W123"
    assert merged["ads_bibcode"] == "2022Test"
    assert merged["pdf_url"] == "https://arxiv.org/pdf/x"
    assert merged["source_url"] is None


def test_merge_source_results_all_none():
    merged = resolver._merge_source_results("10.1/x", None, None, None)
    assert merged["doi"] == "10.1/x"
    assert merged["title"] is None


def test_resolve_paper_no_inputs_returns_none():
    result = resolver.resolve_paper()
    assert result is None


def test_resolve_by_doi_calls_all_sources(monkeypatch):
    """resolve_by_doi queries SS, OA, and ADS and merges the results."""
    import citation_tracker.sources.semantic_scholar as ss_mod
    import citation_tracker.sources.openalexapi as oa_mod
    import citation_tracker.sources.adsapi as ads_mod

    monkeypatch.setattr(ss_mod, "get_paper_by_doi", lambda doi: {
        "title": "T", "authors": "A", "year": 2020,
        "doi": doi, "ss_id": "ss1", "oa_id": None,
        "ads_bibcode": None, "pdf_url": None, "abstract": None,
    })
    monkeypatch.setattr(oa_mod, "get_paper_by_doi", lambda doi: None)
    monkeypatch.setattr(ads_mod, "get_paper_by_doi", lambda doi, key: None)

    class FakeCfg:
        class ads:
            api_key = ""

    result = resolver._resolve_by_doi("10.1/t", FakeCfg())
    assert result is not None
    assert result["title"] == "T"
    assert result["ss_id"] == "ss1"


def test_resolve_by_doi_returns_none_when_all_miss(monkeypatch):
    import citation_tracker.sources.semantic_scholar as ss_mod
    import citation_tracker.sources.openalexapi as oa_mod
    import citation_tracker.sources.adsapi as ads_mod

    monkeypatch.setattr(ss_mod, "get_paper_by_doi", lambda doi: None)
    monkeypatch.setattr(oa_mod, "get_paper_by_doi", lambda doi: None)
    monkeypatch.setattr(ads_mod, "get_paper_by_doi", lambda doi, key: None)

    class FakeCfg:
        class ads:
            api_key = ""

    result = resolver._resolve_by_doi("10.999/missing", FakeCfg())
    assert result is None


def test_resolve_by_url_doi_path(monkeypatch):
    """URL containing a DOI is resolved via _resolve_by_doi."""
    called_with = {}

    def fake_resolve_by_doi(doi, cfg):
        called_with["doi"] = doi
        return {"doi": doi, "title": "Found", "source_url": None}

    monkeypatch.setattr(resolver, "_resolve_by_doi", fake_resolve_by_doi)

    result = resolver._resolve_by_url(
        "https://doi.org/10.1234/mypaper", cfg=None
    )
    assert called_with["doi"] == "10.1234/mypaper"
    assert result["source_url"] == "https://doi.org/10.1234/mypaper"


def test_resolve_by_url_stub_fallback(monkeypatch):
    """A URL that cannot be resolved returns a stub with the URL as pdf_url."""
    monkeypatch.setattr(resolver, "_resolve_from_pdf", lambda url, cfg: None)

    result = resolver._resolve_by_url("https://example.com/my-paper.pdf", cfg=None)
    assert result is not None
    assert result["title"] == "my paper"
    assert result["pdf_url"] == "https://example.com/my-paper.pdf"
    assert result["doi"] is None
