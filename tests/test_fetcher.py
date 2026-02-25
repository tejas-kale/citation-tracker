from citation_tracker.fetcher import _extract_arxiv_id


def test_extract_arxiv_id_from_doi():
    assert _extract_arxiv_id("10.48550/arXiv.1706.03762", {}) == "1706.03762"
