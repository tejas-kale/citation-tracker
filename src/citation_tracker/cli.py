"""Click CLI entry point for citation-tracker."""

from __future__ import annotations

import logging
import sqlite3
import sys
from pathlib import Path
from typing import Any

import click
from rich.console import Console
from rich.table import Table

from citation_tracker.config import load_config

console = Console()
logger = logging.getLogger(__name__)


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")


def _get_db(cfg: Any) -> Path:
    from citation_tracker.db import init_db

    cfg.db_path.parent.mkdir(parents=True, exist_ok=True)
    init_db(cfg.db_path)
    return cfg.db_path


@click.group()
@click.option("--config", "config_path", default=None, help="Path to config.yaml")
@click.option("--env", "env_path", default=None, help="Path to .env file")
@click.option("--verbose", is_flag=True, default=False)
@click.pass_context
def main(ctx: click.Context, config_path: str | None, env_path: str | None, verbose: bool) -> None:
    """Citation Tracker — track and analyse citations of academic papers."""
    _setup_logging(verbose)
    ctx.ensure_object(dict)
    cfg = load_config(
        config_path=Path(config_path) if config_path else None,
        env_path=Path(env_path) if env_path else None,
    )
    ctx.obj["cfg"] = cfg


# ── add ────────────────────────────────────────────────────────────────────


@main.command("add")
@click.argument("url", required=False)
@click.option("--doi", default=None, help="DOI of the paper to track")
@click.option("--ss-id", default=None, help="Semantic Scholar paper ID")
@click.pass_context
def add_paper(ctx: click.Context, url: str | None, doi: str | None, ss_id: str | None) -> None:
    """Add a paper to track by URL, DOI, or Semantic Scholar ID."""
    cfg = ctx.obj["cfg"]
    db_path = _get_db(cfg)

    paper = _resolve_paper(url=url, doi=doi, ss_id=ss_id)
    if paper is None:
        console.print("[red]Could not resolve paper metadata.[/red]")
        sys.exit(1)

    from citation_tracker.db import get_conn, get_tracked_paper_by_doi, insert_tracked_paper

    with get_conn(db_path) as conn:
        if paper.get("doi"):
            existing = get_tracked_paper_by_doi(conn, paper["doi"])
            if existing:
                console.print(
                    f"[yellow]Paper already tracked (id={existing['id']}).[/yellow]"
                )
                return

        paper_id = insert_tracked_paper(conn, paper)
        console.print(f"[green]Added paper (id={paper_id}): {paper.get('title', 'N/A')}[/green]")


def _resolve_paper(
    url: str | None, doi: str | None, ss_id: str | None
) -> dict[str, Any] | None:
    from citation_tracker.sources import semantic_scholar as ss
    from citation_tracker.sources import openalexapi as oa

    if doi:
        paper = ss.get_paper_by_doi(doi)
        if paper is None:
            paper = oa.get_paper_by_doi(doi)
        if paper:
            paper["source_url"] = None
        return paper

    if ss_id:
        paper = ss.get_paper_by_id(ss_id)
        if paper:
            paper["source_url"] = None
        return paper

    if url:
        # Try to extract DOI from URL
        import re

        doi_match = re.search(r"10\.\d{4,}/[^\s\"'<>]+", url)
        if doi_match:
            return _resolve_paper(url=None, doi=doi_match.group(), ss_id=None)

        # Attempt to download PDF and search by filename/URL
        title_guess = Path(url).stem.replace("_", " ").replace("-", " ")
        paper = ss.search_paper_by_title(title_guess)
        if paper is None:
            paper = oa.search_paper_by_title(title_guess)
        if paper:
            paper["source_url"] = url
        else:
            # Create a stub entry with just the URL
            paper = {
                "doi": None,
                "title": title_guess,
                "authors": None,
                "year": None,
                "abstract": None,
                "source_url": url,
                "ss_id": None,
                "oa_id": None,
                "pdf_url": url,
            }
        return paper

    return None


# ── list ───────────────────────────────────────────────────────────────────


@main.command("list")
@click.pass_context
def list_papers(ctx: click.Context) -> None:
    """List all tracked papers."""
    cfg = ctx.obj["cfg"]
    db_path = _get_db(cfg)

    from citation_tracker.db import get_conn, list_tracked_papers

    with get_conn(db_path) as conn:
        papers = list_tracked_papers(conn)

    if not papers:
        console.print("No tracked papers.")
        return

    table = Table(title="Tracked Papers")
    table.add_column("ID", justify="right")
    table.add_column("Title")
    table.add_column("DOI")
    table.add_column("Year")
    table.add_column("Active")
    table.add_column("Added")

    for p in papers:
        table.add_row(
            str(p["id"]),
            (p["title"] or "")[:60],
            p["doi"] or "",
            str(p["year"] or ""),
            "✓" if p["active"] else "✗",
            (p["added_at"] or "")[:10],
        )
    console.print(table)


# ── pause / resume ─────────────────────────────────────────────────────────


@main.command("pause")
@click.option("--doi", required=True, help="DOI of the paper to pause")
@click.pass_context
def pause_paper(ctx: click.Context, doi: str) -> None:
    """Pause tracking a paper (keeps all history)."""
    _set_active(ctx, doi, active=False)


@main.command("resume")
@click.option("--doi", required=True, help="DOI of the paper to resume")
@click.pass_context
def resume_paper(ctx: click.Context, doi: str) -> None:
    """Resume tracking a paper."""
    _set_active(ctx, doi, active=True)


def _set_active(ctx: click.Context, doi: str, active: bool) -> None:
    cfg = ctx.obj["cfg"]
    db_path = _get_db(cfg)
    from citation_tracker.db import (
        get_conn,
        get_tracked_paper_by_doi,
        set_tracked_paper_active,
    )

    with get_conn(db_path) as conn:
        paper = get_tracked_paper_by_doi(conn, doi)
        if paper is None:
            console.print(f"[red]Paper with DOI {doi!r} not found.[/red]")
            sys.exit(1)
        set_tracked_paper_active(conn, paper["id"], active)
        action = "resumed" if active else "paused"
        console.print(f"[green]Paper {action}: {paper['title']}[/green]")


# ── remove ─────────────────────────────────────────────────────────────────


@main.command("remove")
@click.option("--doi", required=True, help="DOI of the paper to remove")
@click.confirmation_option(prompt="This will delete all data for this paper. Continue?")
@click.pass_context
def remove_paper(ctx: click.Context, doi: str) -> None:
    """Remove a paper and all its data."""
    cfg = ctx.obj["cfg"]
    db_path = _get_db(cfg)
    from citation_tracker.db import (
        delete_tracked_paper,
        get_conn,
        get_tracked_paper_by_doi,
    )

    with get_conn(db_path) as conn:
        paper = get_tracked_paper_by_doi(conn, doi)
        if paper is None:
            console.print(f"[red]Paper with DOI {doi!r} not found.[/red]")
            sys.exit(1)
        delete_tracked_paper(conn, paper["id"])
        console.print(f"[green]Removed paper: {paper['title']}[/green]")


# ── run ────────────────────────────────────────────────────────────────────


@main.command("run")
@click.option("--doi", default=None, help="Process a single tracked paper by DOI")
@click.option("--backend", default=None, help="Override backend (openrouter|claude_code)")
@click.option("--triggered-by", default="manual", hidden=True)
@click.pass_context
def run_pipeline(
    ctx: click.Context, doi: str | None, backend: str | None, triggered_by: str
) -> None:
    """Run the full discovery → fetch → parse → analyse → report pipeline."""
    cfg = ctx.obj["cfg"]
    if backend:
        cfg.backend = backend
    db_path = _get_db(cfg)

    from citation_tracker.db import (
        finish_run,
        get_conn,
        get_tracked_paper_by_doi,
        get_tracked_paper_by_id,
        insert_run,
        list_tracked_papers,
    )

    with get_conn(db_path) as conn:
        if doi:
            paper = get_tracked_paper_by_doi(conn, doi)
            if paper is None:
                console.print(f"[red]Paper with DOI {doi!r} not found.[/red]")
                sys.exit(1)
            papers_to_run = [paper]
        else:
            papers_to_run = list_tracked_papers(conn, active_only=True)

        if not papers_to_run:
            console.print("[yellow]No active tracked papers to process.[/yellow]")
            return

        run_id = insert_run(conn, triggered_by=triggered_by)

    total_new = 0
    total_analysed = 0
    all_errors: list[str] = []
    report_sections = []

    for paper_row in papers_to_run:
        paper_dict = dict(paper_row)
        new, analysed, errors, section = _process_paper(paper_dict, cfg, db_path)
        total_new += new
        total_analysed += analysed
        all_errors.extend(errors)
        if section:
            report_sections.append(section)

    # Build and optionally email report
    if report_sections:
        from citation_tracker.report import build_full_report

        report = build_full_report(report_sections)
        console.print("\n[bold]Report:[/bold]\n")
        console.print(report)

        if cfg.email_from and cfg.email_to and cfg.resend_api_key:
            from citation_tracker.mailer import send_report

            try:
                send_report(
                    report,
                    from_address=cfg.email_from,
                    to_address=cfg.email_to,
                    api_key=cfg.resend_api_key,
                )
                console.print("[green]Report emailed.[/green]")
            except Exception as exc:
                console.print(f"[yellow]Email failed: {exc}[/yellow]")

    with get_conn(db_path) as conn:
        finish_run(conn, run_id, total_new, total_analysed, all_errors)

    console.print(
        f"\n[bold green]Done.[/bold green] "
        f"New papers: {total_new}, Analysed: {total_analysed}, Errors: {len(all_errors)}"
    )


def _process_paper(
    paper: dict[str, Any],
    cfg: Any,
    db_path: Path,
) -> tuple[int, int, list[str], Any]:
    """Run the pipeline for one tracked paper. Returns (new, analysed, errors, report_section)."""
    from citation_tracker.db import (
        finish_run,
        get_citing_papers_for_analysis,
        get_citing_papers_pending_pdf,
        get_conn,
        insert_analysis,
        list_citing_papers,
        update_citing_paper_pdf,
        update_citing_paper_text,
        upsert_citing_paper,
    )
    from citation_tracker.sources import semantic_scholar as ss
    from citation_tracker.sources import openalexapi as oa
    from citation_tracker.sources.deduplicator import deduplicate
    from citation_tracker.fetcher import scan_manual_dir, try_download_citing_paper
    from citation_tracker.parser import extract_text
    from citation_tracker.analyser import analyse_citing_paper
    from citation_tracker.report import build_report

    tracked_id = paper["id"]
    errors: list[str] = []
    new_papers = 0

    console.print(f"\n[bold]Processing:[/bold] {paper.get('title', 'N/A')}")

    # ── 1. DISCOVER ────────────────────────────────────────────────────────
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

    citations = deduplicate(citations)

    with get_conn(db_path) as conn:
        for c in citations:
            _cid, is_new = upsert_citing_paper(conn, tracked_id, c)
            if is_new:
                new_papers += 1

    console.print(f"  Discovered {len(citations)} citing papers")

    # ── 2. FETCH ───────────────────────────────────────────────────────────
    with get_conn(db_path) as conn:
        pending = get_citing_papers_pending_pdf(conn, tracked_id)

    for cp in pending:
        cp_dict = dict(cp)
        path = try_download_citing_paper(cp_dict, cfg.pdfs_dir, email=cfg.unpaywall_email or "citation-tracker@example.com")
        with get_conn(db_path) as conn:
            if path:
                update_citing_paper_pdf(conn, cp["id"], "downloaded")
            else:
                update_citing_paper_pdf(conn, cp["id"], "failed")

    # Check manual dir
    for manual_pdf in scan_manual_dir(cfg.manual_dir):
        # Manual PDFs handled via `ingest` command; just log
        logger.debug("Manual PDF found: %s", manual_pdf)

    # ── 3. PARSE ───────────────────────────────────────────────────────────
    with get_conn(db_path) as conn:
        downloaded = conn.execute(
            """SELECT * FROM citing_papers
               WHERE tracked_paper_id=? AND pdf_status IN ('downloaded','manual')
               AND text_extracted=0""",
            (tracked_id,),
        ).fetchall()

    for cp in downloaded:
        from citation_tracker.fetcher import _doi_to_path
        doi_safe = _doi_to_path(cp["doi"]) if cp["doi"] else "unknown"
        pdf_path = cfg.pdfs_dir / doi_safe / "paper.pdf"
        if not pdf_path.exists():
            continue
        text = extract_text(pdf_path)
        if text:
            with get_conn(db_path) as conn:
                update_citing_paper_text(conn, cp["id"], text)

    # ── 4. ANALYSE ─────────────────────────────────────────────────────────
    with get_conn(db_path) as conn:
        to_analyse = get_citing_papers_for_analysis(conn, tracked_id)

    papers_analysed = 0
    for cp in to_analyse:
        try:
            result = analyse_citing_paper(
                tracked_paper=paper,
                citing_paper=dict(cp),
                extracted_text=cp["extracted_text"] or "",
                config=cfg,
            )
            import json

            with get_conn(db_path) as conn:
                insert_analysis(
                    conn,
                    {
                        "citing_paper_id": cp["id"],
                        "tracked_paper_id": tracked_id,
                        "backend_used": cfg.backend,
                        "summary": result.get("summary"),
                        "relationship_type": result.get("relationship_type"),
                        "new_evidence": result.get("new_evidence"),
                        "flaws_identified": result.get("flaws_identified"),
                        "assumptions_questioned": result.get("assumptions_questioned"),
                        "other_notes": result.get("other_notes"),
                        "raw_response": json.dumps(result),
                    },
                )
            papers_analysed += 1
        except Exception as exc:
            errors.append(f"Analysis failed for citing paper id={cp['id']}: {exc}")
            logger.warning("Analysis failed: %s", exc)

    # ── 5. REPORT ──────────────────────────────────────────────────────────
    from citation_tracker.db import list_analyses

    with get_conn(db_path) as conn:
        analyses = list_analyses(conn, tracked_id)
        failed_pdfs = conn.execute(
            "SELECT * FROM citing_papers WHERE tracked_paper_id=? AND pdf_status='failed'",
            (tracked_id,),
        ).fetchall()

    section = (paper, analyses, failed_pdfs)
    return new_papers, papers_analysed, errors, section


# ── ingest ─────────────────────────────────────────────────────────────────


@main.command("ingest")
@click.argument("pdf_path", type=click.Path(exists=True))
@click.option("--doi", required=True, help="DOI of the citing paper")
@click.pass_context
def ingest(ctx: click.Context, pdf_path: str, doi: str) -> None:
    """Ingest a manually downloaded PDF and link it to a citing paper."""
    cfg = ctx.obj["cfg"]
    db_path = _get_db(cfg)

    from citation_tracker.db import get_conn
    from citation_tracker.fetcher import _doi_to_path
    from citation_tracker.parser import extract_text
    import shutil

    with get_conn(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM citing_papers WHERE doi = ?", (doi,)
        ).fetchone()
        if row is None:
            console.print(f"[red]No citing paper with DOI {doi!r} found in database.[/red]")
            sys.exit(1)

    # Copy PDF to pdfs dir using the same DOI path scheme as fetcher.py
    safe_doi = _doi_to_path(doi)
    dest_dir = cfg.pdfs_dir / safe_doi
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / "paper.pdf"
    shutil.copy2(pdf_path, dest)

    text = extract_text(dest)
    with get_conn(db_path) as conn:
        from citation_tracker.db import update_citing_paper_pdf, update_citing_paper_text

        update_citing_paper_pdf(conn, row["id"], "manual")
        if text:
            update_citing_paper_text(conn, row["id"], text)

    console.print(f"[green]Ingested PDF for DOI {doi}[/green]")
    if not text:
        console.print("[yellow]Warning: could not extract text from PDF.[/yellow]")


# ── status ─────────────────────────────────────────────────────────────────


@main.command("status")
@click.pass_context
def status(ctx: click.Context) -> None:
    """Show DB summary: tracked papers, citation counts, pending items."""
    cfg = ctx.obj["cfg"]
    db_path = _get_db(cfg)

    from citation_tracker.db import db_summary, get_conn

    with get_conn(db_path) as conn:
        summary = db_summary(conn)

    table = Table(title="Citation Tracker Status")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("Active tracked papers", str(summary["active_tracked"]))
    table.add_row("Total citing papers", str(summary["total_citing"]))
    table.add_row("Pending PDF downloads", str(summary["pending_pdf"]))
    table.add_row("Total analyses", str(summary["total_analyses"]))
    console.print(table)


# ── show ───────────────────────────────────────────────────────────────────


@main.command("show")
@click.option("--doi", required=True, help="DOI of the tracked paper to show")
@click.pass_context
def show(ctx: click.Context, doi: str) -> None:
    """Show all analyses for a given tracked paper."""
    cfg = ctx.obj["cfg"]
    db_path = _get_db(cfg)

    from citation_tracker.db import (
        get_conn,
        get_tracked_paper_by_doi,
        list_analyses,
    )
    from citation_tracker.report import build_report

    with get_conn(db_path) as conn:
        paper = get_tracked_paper_by_doi(conn, doi)
        if paper is None:
            console.print(f"[red]Paper with DOI {doi!r} not found.[/red]")
            sys.exit(1)
        analyses = list_analyses(conn, paper["id"])
        failed_pdfs = conn.execute(
            "SELECT * FROM citing_papers WHERE tracked_paper_id=? AND pdf_status='failed'",
            (paper["id"],),
        ).fetchall()

    report = build_report(paper, analyses, failed_pdfs)
    console.print(report)
