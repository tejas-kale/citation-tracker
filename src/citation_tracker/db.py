"""SQLite database schema and queries."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def init_db(db_path: Path) -> None:
    """Initialise the database, creating tables if they don't exist."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            PRAGMA journal_mode=WAL;
            PRAGMA foreign_keys=ON;

            CREATE TABLE IF NOT EXISTS tracked_papers (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                doi         TEXT,
                title       TEXT,
                authors     TEXT,
                year        INTEGER,
                abstract    TEXT,
                source_url  TEXT,
                ss_id       TEXT,
                oa_id       TEXT,
                added_at    TEXT NOT NULL,
                active      INTEGER NOT NULL DEFAULT 1
            );

            CREATE TABLE IF NOT EXISTS citing_papers (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                tracked_paper_id    INTEGER NOT NULL REFERENCES tracked_papers(id),
                doi                 TEXT,
                title               TEXT,
                authors             TEXT,
                year                INTEGER,
                abstract            TEXT,
                ss_id               TEXT,
                oa_id               TEXT,
                pdf_url             TEXT,
                pdf_status          TEXT NOT NULL DEFAULT 'pending',
                extracted_text      TEXT,
                text_extracted      INTEGER NOT NULL DEFAULT 0,
                created_at          TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS analyses (
                id                      INTEGER PRIMARY KEY AUTOINCREMENT,
                citing_paper_id         INTEGER NOT NULL REFERENCES citing_papers(id),
                tracked_paper_id        INTEGER NOT NULL REFERENCES tracked_papers(id),
                backend_used            TEXT,
                summary                 TEXT,
                relationship_type       TEXT,
                new_evidence            TEXT,
                flaws_identified        TEXT,
                assumptions_questioned  TEXT,
                other_notes             TEXT,
                raw_response            TEXT,
                analysed_at             TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS runs (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                triggered_by        TEXT NOT NULL,
                tracked_paper_id    INTEGER REFERENCES tracked_papers(id),
                started_at          TEXT NOT NULL,
                finished_at         TEXT,
                new_papers_found    INTEGER,
                papers_analysed     INTEGER,
                errors              TEXT
            );
            """
        )


@contextmanager
def get_conn(db_path: Path) -> Generator[sqlite3.Connection, None, None]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ── tracked_papers ─────────────────────────────────────────────────────────


def insert_tracked_paper(conn: sqlite3.Connection, paper: dict[str, Any]) -> int:
    cur = conn.execute(
        """
        INSERT INTO tracked_papers
            (doi, title, authors, year, abstract, source_url, ss_id, oa_id, added_at, active)
        VALUES
            (:doi, :title, :authors, :year, :abstract, :source_url, :ss_id, :oa_id, :added_at, 1)
        """,
        {
            "doi": paper.get("doi"),
            "title": paper.get("title"),
            "authors": paper.get("authors"),
            "year": paper.get("year"),
            "abstract": paper.get("abstract"),
            "source_url": paper.get("source_url"),
            "ss_id": paper.get("ss_id"),
            "oa_id": paper.get("oa_id"),
            "added_at": _now(),
        },
    )
    return cur.lastrowid  # type: ignore[return-value]


def get_tracked_paper_by_doi(conn: sqlite3.Connection, doi: str) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM tracked_papers WHERE doi = ?", (doi,)
    ).fetchone()


def get_tracked_paper_by_ss_id(conn: sqlite3.Connection, ss_id: str) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM tracked_papers WHERE ss_id = ?", (ss_id,)
    ).fetchone()


def get_tracked_paper_by_id(conn: sqlite3.Connection, paper_id: int) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM tracked_papers WHERE id = ?", (paper_id,)
    ).fetchone()


def list_tracked_papers(
    conn: sqlite3.Connection, active_only: bool = False
) -> list[sqlite3.Row]:
    if active_only:
        return conn.execute(
            "SELECT * FROM tracked_papers WHERE active = 1 ORDER BY added_at"
        ).fetchall()
    return conn.execute("SELECT * FROM tracked_papers ORDER BY added_at").fetchall()


def set_tracked_paper_active(
    conn: sqlite3.Connection, paper_id: int, active: bool
) -> None:
    conn.execute(
        "UPDATE tracked_papers SET active = ? WHERE id = ?",
        (1 if active else 0, paper_id),
    )


def delete_tracked_paper(conn: sqlite3.Connection, paper_id: int) -> None:
    conn.execute(
        "DELETE FROM analyses WHERE tracked_paper_id = ?", (paper_id,)
    )
    conn.execute(
        "DELETE FROM citing_papers WHERE tracked_paper_id = ?", (paper_id,)
    )
    conn.execute("DELETE FROM runs WHERE tracked_paper_id = ?", (paper_id,))
    conn.execute("DELETE FROM tracked_papers WHERE id = ?", (paper_id,))


# ── citing_papers ──────────────────────────────────────────────────────────


def upsert_citing_paper(
    conn: sqlite3.Connection, tracked_paper_id: int, paper: dict[str, Any]
) -> tuple[int, bool]:
    """
    Insert a citing paper if it doesn't exist yet (keyed on doi or ss_id).

    Returns ``(id, is_new)`` where *is_new* is True if the row was just inserted.
    """
    existing = None
    if paper.get("doi"):
        existing = conn.execute(
            "SELECT id FROM citing_papers WHERE tracked_paper_id = ? AND doi = ?",
            (tracked_paper_id, paper["doi"]),
        ).fetchone()
    if existing is None and paper.get("ss_id"):
        existing = conn.execute(
            "SELECT id FROM citing_papers WHERE tracked_paper_id = ? AND ss_id = ?",
            (tracked_paper_id, paper["ss_id"]),
        ).fetchone()

    if existing:
        return existing["id"], False

    cur = conn.execute(
        """
        INSERT INTO citing_papers
            (tracked_paper_id, doi, title, authors, year, abstract,
             ss_id, oa_id, pdf_url, pdf_status, created_at)
        VALUES
            (:tracked_paper_id, :doi, :title, :authors, :year, :abstract,
             :ss_id, :oa_id, :pdf_url, 'pending', :created_at)
        """,
        {
            "tracked_paper_id": tracked_paper_id,
            "doi": paper.get("doi"),
            "title": paper.get("title"),
            "authors": paper.get("authors"),
            "year": paper.get("year"),
            "abstract": paper.get("abstract"),
            "ss_id": paper.get("ss_id"),
            "oa_id": paper.get("oa_id"),
            "pdf_url": paper.get("pdf_url"),
            "created_at": _now(),
        },
    )
    return cur.lastrowid, True  # type: ignore[return-value]


def get_citing_papers_pending_pdf(
    conn: sqlite3.Connection, tracked_paper_id: int
) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM citing_papers WHERE tracked_paper_id = ? AND pdf_status = 'pending'",
        (tracked_paper_id,),
    ).fetchall()


def get_citing_papers_for_analysis(
    conn: sqlite3.Connection, tracked_paper_id: int
) -> list[sqlite3.Row]:
    return conn.execute(
        """
        SELECT cp.* FROM citing_papers cp
        LEFT JOIN analyses a ON a.citing_paper_id = cp.id
            AND a.tracked_paper_id = cp.tracked_paper_id
        WHERE cp.tracked_paper_id = ?
          AND cp.text_extracted = 1
          AND a.id IS NULL
        """,
        (tracked_paper_id,),
    ).fetchall()


def update_citing_paper_pdf(
    conn: sqlite3.Connection, citing_paper_id: int, status: str, pdf_url: str | None = None
) -> None:
    conn.execute(
        "UPDATE citing_papers SET pdf_status = ?, pdf_url = COALESCE(?, pdf_url) WHERE id = ?",
        (status, pdf_url, citing_paper_id),
    )


def update_citing_paper_text(
    conn: sqlite3.Connection, citing_paper_id: int, text: str
) -> None:
    conn.execute(
        "UPDATE citing_papers SET extracted_text = ?, text_extracted = 1 WHERE id = ?",
        (text, citing_paper_id),
    )


def get_citing_paper_by_doi(
    conn: sqlite3.Connection, tracked_paper_id: int, doi: str
) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM citing_papers WHERE tracked_paper_id = ? AND doi = ?",
        (tracked_paper_id, doi),
    ).fetchone()


def list_citing_papers(
    conn: sqlite3.Connection, tracked_paper_id: int
) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM citing_papers WHERE tracked_paper_id = ? ORDER BY created_at",
        (tracked_paper_id,),
    ).fetchall()


# ── analyses ───────────────────────────────────────────────────────────────


def insert_analysis(conn: sqlite3.Connection, analysis: dict[str, Any]) -> int:
    cur = conn.execute(
        """
        INSERT INTO analyses
            (citing_paper_id, tracked_paper_id, backend_used,
             summary, relationship_type, new_evidence,
             flaws_identified, assumptions_questioned, other_notes,
             raw_response, analysed_at)
        VALUES
            (:citing_paper_id, :tracked_paper_id, :backend_used,
             :summary, :relationship_type, :new_evidence,
             :flaws_identified, :assumptions_questioned, :other_notes,
             :raw_response, :analysed_at)
        """,
        {
            "citing_paper_id": analysis["citing_paper_id"],
            "tracked_paper_id": analysis["tracked_paper_id"],
            "backend_used": analysis.get("backend_used"),
            "summary": analysis.get("summary"),
            "relationship_type": analysis.get("relationship_type"),
            "new_evidence": analysis.get("new_evidence"),
            "flaws_identified": analysis.get("flaws_identified"),
            "assumptions_questioned": analysis.get("assumptions_questioned"),
            "other_notes": analysis.get("other_notes"),
            "raw_response": analysis.get("raw_response"),
            "analysed_at": _now(),
        },
    )
    return cur.lastrowid  # type: ignore[return-value]


def list_analyses(
    conn: sqlite3.Connection, tracked_paper_id: int
) -> list[sqlite3.Row]:
    return conn.execute(
        """
        SELECT a.*, cp.title AS citing_title, cp.authors AS citing_authors,
               cp.year AS citing_year, cp.doi AS citing_doi
        FROM analyses a
        JOIN citing_papers cp ON cp.id = a.citing_paper_id
        WHERE a.tracked_paper_id = ?
        ORDER BY a.analysed_at
        """,
        (tracked_paper_id,),
    ).fetchall()


# ── runs ───────────────────────────────────────────────────────────────────


def insert_run(
    conn: sqlite3.Connection,
    triggered_by: str,
    tracked_paper_id: int | None = None,
) -> int:
    cur = conn.execute(
        """
        INSERT INTO runs (triggered_by, tracked_paper_id, started_at)
        VALUES (?, ?, ?)
        """,
        (triggered_by, tracked_paper_id, _now()),
    )
    return cur.lastrowid  # type: ignore[return-value]


def finish_run(
    conn: sqlite3.Connection,
    run_id: int,
    new_papers_found: int,
    papers_analysed: int,
    errors: list[str] | None = None,
) -> None:
    conn.execute(
        """
        UPDATE runs
        SET finished_at = ?, new_papers_found = ?, papers_analysed = ?, errors = ?
        WHERE id = ?
        """,
        (
            _now(),
            new_papers_found,
            papers_analysed,
            json.dumps(errors or []),
            run_id,
        ),
    )


def db_summary(conn: sqlite3.Connection) -> dict[str, Any]:
    tracked = conn.execute("SELECT COUNT(*) FROM tracked_papers WHERE active=1").fetchone()[0]
    total_citing = conn.execute("SELECT COUNT(*) FROM citing_papers").fetchone()[0]
    pending_pdf = conn.execute(
        "SELECT COUNT(*) FROM citing_papers WHERE pdf_status='pending'"
    ).fetchone()[0]
    total_analyses = conn.execute("SELECT COUNT(*) FROM analyses").fetchone()[0]
    return {
        "active_tracked": tracked,
        "total_citing": total_citing,
        "pending_pdf": pending_pdf,
        "total_analyses": total_analyses,
    }
