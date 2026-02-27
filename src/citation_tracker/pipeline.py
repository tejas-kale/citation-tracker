"""Pipeline execution: discover → fetch → parse → analyse → report."""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Type alias for a report section tuple
ReportSection = tuple[dict[str, Any], list[Any], list[Any], str | None]


def _discover_citations(
    paper: dict[str, Any], cfg: Any, errors: list[str]
) -> list[dict[str, Any]]:
    """Query all configured citation sources and deduplicate results.

    Args:
        paper: Tracked paper dict (must include ss_id, oa_id, ads_bibcode).
        cfg: Application config.
        errors: List to append error strings to.

    Returns:
        Deduplicated list of citing paper dicts.
    """
    from citation_tracker.sources import semantic_scholar as ss
    from citation_tracker.sources import openalexapi as oa
    from citation_tracker.sources import adsapi as ads
    from citation_tracker.sources.deduplicator import deduplicate

    citations: list[dict[str, Any]] = []

    if paper.get("ss_id"):
        try:
            citations.extend(ss.get_citations(paper["ss_id"]))
        except Exception as exc:
            errors.append(f"SS citations failed: {exc}")

    if paper.get("oa_id"):
        try:
            citations.extend(oa.get_citations(paper["oa_id"]))
        except Exception as exc:
            errors.append(f"OA citations failed: {exc}")

    if paper.get("ads_bibcode") and cfg.ads.api_key:
        try:
            citations.extend(ads.get_citations(paper["ads_bibcode"], cfg.ads.api_key))
        except Exception as exc:
            errors.append(f"ADS citations failed: {exc}")

    return deduplicate(citations)


def _fetch_stage(
    tracked_id: str, cfg: Any, db_path: Path, workers: int
) -> None:
    """Download PDFs for all pending citing papers in parallel."""
    from citation_tracker.db import (
        get_citing_papers_pending_pdf,
        get_conn,
        update_citing_paper_pdf,
    )
    from citation_tracker.fetcher import try_download_citing_paper

    with get_conn(db_path) as conn:
        pending = get_citing_papers_pending_pdf(conn, tracked_id)

    def _fetch_one(cp_row: Any) -> tuple[str, bool]:
        cp_dict = dict(cp_row)
        email = cfg.unpaywall_email or "citation-tracker@example.com"
        path = try_download_citing_paper(cp_dict, cfg.pdfs_dir, email=email)
        return cp_dict["id"], bool(path)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_fetch_one, cp) for cp in pending]
        for future in as_completed(futures):
            cp_id, ok = future.result()
            with get_conn(db_path) as conn:
                update_citing_paper_pdf(conn, cp_id, "downloaded" if ok else "failed")


def _parse_stage(
    tracked_id: str, cfg: Any, db_path: Path, workers: int
) -> None:
    """Extract text from all downloaded PDFs in parallel."""
    from citation_tracker.db import get_conn, update_citing_paper_text
    from citation_tracker.fetcher import _doi_to_path
    from citation_tracker.parser import extract_text

    with get_conn(db_path) as conn:
        downloaded = conn.execute(
            """SELECT * FROM citing_papers
               WHERE tracked_paper_id=? AND pdf_status IN ('downloaded','manual')
               AND text_extracted=0""",
            (tracked_id,),
        ).fetchall()

    def _parse_one(cp_row: Any) -> tuple[str, str | None]:
        cp_dict = dict(cp_row)
        doi_safe = _doi_to_path(cp_dict["doi"]) if cp_dict["doi"] else "unknown"
        pdf_path = cfg.pdfs_dir / doi_safe / "paper.pdf"
        if not pdf_path.exists():
            return cp_dict["id"], None
        return cp_dict["id"], extract_text(pdf_path)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_parse_one, cp) for cp in downloaded]
        for future in as_completed(futures):
            cp_id, text = future.result()
            if text:
                with get_conn(db_path) as conn:
                    update_citing_paper_text(conn, cp_id, text)


def _analyse_stage(
    paper: dict[str, Any],
    tracked_id: str,
    cfg: Any,
    db_path: Path,
    workers: int,
    errors: list[str],
) -> int:
    """Run LLM analysis on all papers with extracted text but no analysis yet.

    Returns the number of papers successfully analysed.
    """
    from citation_tracker.db import (
        get_citing_papers_for_analysis,
        get_conn,
        insert_analysis,
    )
    from citation_tracker.analyser import analyse_citing_paper

    with get_conn(db_path) as conn:
        to_analyse = get_citing_papers_for_analysis(conn, tracked_id)

    def _analyse_one(cp_row: Any) -> tuple[str, dict[str, Any]]:
        cp_dict = dict(cp_row)
        result = analyse_citing_paper(
            tracked_paper=paper,
            citing_paper=cp_dict,
            extracted_text=cp_dict.get("extracted_text") or "",
            config=cfg,
        )
        return cp_dict["id"], result

    papers_analysed = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_analyse_one, cp) for cp in to_analyse]
        for future in as_completed(futures):
            try:
                cp_id, result = future.result()
                with get_conn(db_path) as conn:
                    insert_analysis(
                        conn,
                        {
                            "citing_paper_id": cp_id,
                            "tracked_paper_id": tracked_id,
                            "backend_used": cfg.backend,
                            "summary": result.get("summary"),
                            "relationship_type": result.get("relationship_type"),
                            "new_evidence": result.get("new_evidence"),
                            "flaws_identified": result.get("flaws_identified"),
                            "assumptions_questioned": result.get(
                                "assumptions_questioned"
                            ),
                            "other_notes": result.get("other_notes"),
                            "raw_response": json.dumps(result),
                        },
                    )
                papers_analysed += 1
            except Exception as exc:
                errors.append(f"Analysis failed: {exc}")
                logger.warning("Analysis failed: %s", exc)

    return papers_analysed


def _report_stage(
    paper: dict[str, Any],
    tracked_id: str,
    cfg: Any,
    db_path: Path,
    errors: list[str],
) -> ReportSection:
    """Collect analyses and generate a scholarly synthesis.

    Returns a tuple of (paper, analyses_rows, failed_pdfs, scholarly_synthesis).
    """
    from citation_tracker.db import get_conn, list_analyses
    from citation_tracker.analyser import generate_scholarly_synthesis

    with get_conn(db_path) as conn:
        analyses_rows = list_analyses(conn, tracked_id)
        failed_pdfs = conn.execute(
            "SELECT * FROM citing_papers"
            " WHERE tracked_paper_id=? AND pdf_status='failed'",
            (tracked_id,),
        ).fetchall()

    scholarly_synthesis = None
    if analyses_rows:
        try:
            logger.info("Generating scholarly synthesis...")
            scholarly_synthesis = generate_scholarly_synthesis(
                paper, [dict(a) for a in analyses_rows], cfg
            )
        except Exception as exc:
            errors.append(f"Synthesis failed: {exc}")
            logger.warning("Synthesis failed: %s", exc)

    return paper, analyses_rows, failed_pdfs, scholarly_synthesis


def process_paper(
    paper: dict[str, Any],
    cfg: Any,
    db_path: Path,
    workers: int = 8,
) -> tuple[int, int, list[str], ReportSection]:
    """Run the full pipeline for one tracked paper.

    Stages: discover → fetch → parse → analyse → report.

    Args:
        paper: Tracked paper dict.
        cfg: Application config.
        db_path: Path to the SQLite database.
        workers: Number of threads for parallel fetch/parse/analyse.

    Returns:
        Tuple of (new_papers, papers_analysed, errors, report_section).
    """
    from citation_tracker.db import get_conn, upsert_citing_paper
    from citation_tracker.fetcher import scan_manual_dir

    tracked_id = paper["id"]
    errors: list[str] = []

    logger.info("Processing: %s", paper.get("title", "N/A"))

    # 1. DISCOVER
    citations = _discover_citations(paper, cfg, errors)
    new_papers = 0
    with get_conn(db_path) as conn:
        for c in citations:
            _cid, is_new = upsert_citing_paper(conn, tracked_id, c)
            if is_new:
                new_papers += 1

    logger.info("Discovered %d citing papers", len(citations))

    # 2. FETCH
    _fetch_stage(tracked_id, cfg, db_path, workers)

    # Manual PDFs are handled via the `ingest` command; just log their presence
    for manual_pdf in scan_manual_dir(cfg.manual_dir):
        logger.debug("Manual PDF found: %s", manual_pdf)

    # 3. PARSE
    _parse_stage(tracked_id, cfg, db_path, workers)

    # 4. ANALYSE
    papers_analysed = _analyse_stage(paper, tracked_id, cfg, db_path, workers, errors)

    # 5. REPORT
    section = _report_stage(paper, tracked_id, cfg, db_path, errors)

    return new_papers, papers_analysed, errors, section
