from pathlib import Path

from citation_tracker.fetcher import _doi_to_arxiv_id, try_download_citing_paper


def test_doi_to_arxiv_id():
    assert _doi_to_arxiv_id("10.48550/arXiv.2501.01234") == "2501.01234"
    assert _doi_to_arxiv_id("10.1000/test") is None


def test_try_download_uses_crossref_after_unpaywall(mocker, tmp_path):
    unpaywall = mocker.patch("citation_tracker.fetcher._try_unpaywall", return_value=None)
    crossref_path = tmp_path / "x" / "paper.pdf"
    crossref = mocker.patch("citation_tracker.fetcher._try_crossref", return_value=crossref_path)
    arxiv = mocker.patch("citation_tracker.fetcher._try_arxiv", return_value=None)

    result = try_download_citing_paper({"doi": "10.1000/test"}, Path(tmp_path))

    assert result == crossref_path
    unpaywall.assert_called_once()
    crossref.assert_called_once_with("10.1000/test", Path(tmp_path))
    arxiv.assert_not_called()
