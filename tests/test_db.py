import sqlite3
import pytest
from pathlib import Path
from citation_tracker.db import (
    init_db, get_conn, insert_tracked_paper, get_tracked_paper_by_id,
    upsert_citing_paper, list_citing_papers, _generate_id
)

@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "test.db"

def test_init_db(db_path):
    init_db(db_path)
    assert db_path.exists()
    with sqlite3.connect(db_path) as conn:
        res = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        tables = [r[0] for r in res]
        assert "tracked_papers" in tables
        assert "citing_papers" in tables
        assert "analyses" in tables
        assert "runs" in tables

def test_id_generation():
    id1 = _generate_id()
    id2 = _generate_id()
    assert len(id1) == 8
    assert id1 != id2
    # Check hex
    int(id1, 16)

def test_insert_and_get_tracked_paper(db_path):
    init_db(db_path)
    paper = {
        "doi": "10.1234/test",
        "title": "Test Paper",
        "authors": "Author A, Author B",
        "year": 2023,
        "ads_bibcode": "2023Test...1...1A"
    }
    with get_conn(db_path) as conn:
        paper_id = insert_tracked_paper(conn, paper)
        assert len(paper_id) == 8
        
        fetched = get_tracked_paper_by_id(conn, paper_id)
        assert fetched["title"] == "Test Paper"
        assert fetched["doi"] == "10.1234/test"
        assert fetched["ads_bibcode"] == "2023Test...1...1A"

def test_upsert_citing_paper(db_path):
    init_db(db_path)
    tracked_id = "tracked1"
    citing = {
        "doi": "10.5678/citing",
        "title": "Citing Paper",
        "ads_bibcode": "2023Citing..1...1C"
    }
    with get_conn(db_path) as conn:
        # Need to insert tracked paper first because of FK
        conn.execute("INSERT INTO tracked_papers (id, added_at) VALUES (?, 'now')", (tracked_id,))
        
        cid1, is_new1 = upsert_citing_paper(conn, tracked_id, citing)
        assert is_new1 is True
        
        # Verify bibcode was saved
        row = conn.execute("SELECT ads_bibcode FROM citing_papers WHERE id=?", (cid1,)).fetchone()
        assert row["ads_bibcode"] == "2023Citing..1...1C"
        
        cid2, is_new2 = upsert_citing_paper(conn, tracked_id, citing)
        assert is_new2 is False
        assert cid1 == cid2

def test_list_citing_papers(db_path):
    init_db(db_path)
    tracked_id = "tracked1"
    with get_conn(db_path) as conn:
        conn.execute("INSERT INTO tracked_papers (id, added_at) VALUES (?, 'now')", (tracked_id,))
        upsert_citing_paper(conn, tracked_id, {"title": "C1", "doi": "d1"})
        upsert_citing_paper(conn, tracked_id, {"title": "C2", "doi": "d2"})
        
        citations = list_citing_papers(conn, tracked_id)
        assert len(citations) == 2
        assert any(c["doi"] == "d1" for c in citations)
        assert any(c["doi"] == "d2" for c in citations)
