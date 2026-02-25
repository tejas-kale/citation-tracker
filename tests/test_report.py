import sqlite3
import pytest
from citation_tracker.report import build_report, render_full_report_html

@pytest.fixture
def mock_tracked_paper():
    return {
        "title": "Main Paper",
        "authors": "Author 1",
        "year": 2020,
        "doi": "10.1/main",
        "source_url": "https://example.com/main.pdf"
    }

@pytest.fixture
def mock_analyses():
    return [
        {
            "citing_title": "Citing A",
            "citing_authors": "Author A",
            "citing_year": 2021,
            "citing_doi": "10.1/a",
            "relationship_type": "supports",
            "summary": "Great paper.",
            "new_evidence": None,
            "flaws_identified": None,
            "assumptions_questioned": None,
            "other_notes": None
        }
    ]

def test_build_report(mock_tracked_paper, mock_analyses):
    report = build_report(mock_tracked_paper, mock_analyses, [])
    assert "# Citation Report: Main Paper" in report
    assert "*Author 1*" in report
    assert "Source URL: https://example.com/main.pdf" in report
    assert "```mermaid" in report
    assert "T -->|supports| C1" in report
    assert "### Citing A" in report
    assert "**Relationship:** `supports`" in report

def test_render_html():
    markdown = "# Title\n\nSome text."
    html = render_full_report_html(markdown)
    assert "<title>Citation Tracker Report</title>" in html
    assert "Title</h1>" in html
    assert "<p>Some text.</p>" in html
