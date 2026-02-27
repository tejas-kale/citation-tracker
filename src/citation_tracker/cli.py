"""Click CLI entry point for citation-tracker."""

from __future__ import annotations

import logging
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import click
from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table

from citation_tracker.config import load_config
from citation_tracker.db import (
    db_summary,
    delete_tracked_paper,
    finish_run,
    get_conn,
    get_tracked_paper_by_doi,
    get_tracked_paper_by_id,
    init_db,
    insert_run,
    insert_tracked_paper,
    list_analyses,
    list_citing_papers,
    list_tracked_papers,
    set_tracked_paper_active,
    upsert_citing_paper,
    update_citing_paper_pdf,
    update_citing_paper_text,
)
from citation_tracker.resolver import resolve_paper
from citation_tracker.pipeline import process_paper
from citation_tracker.report import build_report
from citation_tracker.analyser import generate_scholarly_synthesis
from citation_tracker.fetcher import _doi_to_path

console = Console()
logger = logging.getLogger(__name__)


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")


def _get_db(cfg: Any) -> Path:
    cfg.db_path.parent.mkdir(parents=True, exist_ok=True)
    init_db(cfg.db_path)
    return cfg.db_path


def _print_markdown(content: str) -> None:
    """Print Markdown content using glow (if available) or rich."""
    if shutil.which("glow"):
        try:
            subprocess.run(["glow", "-"], input=content.encode(), check=True)
            return
        except Exception:
            pass
    console.print(Markdown(content))


@click.group()
@click.option("--config", "config_path", default=None, help="Path to config.yaml")
@click.option("--env", "env_path", default=None, help="Path to .env file")
@click.option("--verbose", is_flag=True, default=False)
@click.pass_context
def main(
    ctx: click.Context,
    config_path: str | None,
    env_path: str | None,
    verbose: bool,
) -> None:
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
def add_paper(
    ctx: click.Context, url: str | None, doi: str | None, ss_id: str | None
) -> None:
    """Add a paper to track by URL, DOI, or Semantic Scholar ID."""
    cfg = ctx.obj["cfg"]
    db_path = _get_db(cfg)

    paper = resolve_paper(url=url, doi=doi, ss_id=ss_id, cfg=cfg)
    if paper is None:
        console.print("[red]Could not resolve paper metadata.[/red]")
        sys.exit(1)

    with get_conn(db_path) as conn:
        if paper.get("doi"):
            existing = get_tracked_paper_by_doi(conn, paper["doi"])
            if existing:
                console.print(
                    f"[yellow]Paper already tracked (id={existing['id']}).[/yellow]"
                )
                return

        paper_id = insert_tracked_paper(conn, paper)
        console.print(
            f"[green]Added paper (id={paper_id}): {paper.get('title', 'N/A')}[/green]"
        )


# ── list ───────────────────────────────────────────────────────────────────


@main.command("list")
@click.pass_context
def list_papers(ctx: click.Context) -> None:
    """List all tracked papers."""
    cfg = ctx.obj["cfg"]
    db_path = _get_db(cfg)

    with get_conn(db_path) as conn:
        papers = list_tracked_papers(conn)

    if not papers:
        console.print("No tracked papers.")
        return

    table = Table(title="Tracked Papers")
    table.add_column("ID", justify="right")
    table.add_column("Title")
    table.add_column("DOI/URL")
    table.add_column("Year")
    table.add_column("Active")
    table.add_column("Added")

    for p in papers:
        location = p["doi"] or p["source_url"] or ""
        table.add_row(
            str(p["id"]),
            (p["title"] or "")[:60],
            location[:40],
            str(p["year"] or ""),
            "✓" if p["active"] else "✗",
            (p["added_at"] or "")[:10],
        )
    console.print(table)


# ── pause / resume ─────────────────────────────────────────────────────────


@main.command("pause")
@click.option("--id", "paper_id", help="ID of the paper to pause")
@click.option("--doi", help="DOI of the paper to pause")
@click.pass_context
def pause_paper(ctx: click.Context, paper_id: str | None, doi: str | None) -> None:
    """Pause tracking a paper (keeps all history)."""
    _set_active(ctx, paper_id, doi, active=False)


@main.command("resume")
@click.option("--id", "paper_id", help="ID of the paper to resume")
@click.option("--doi", help="DOI of the paper to resume")
@click.pass_context
def resume_paper(ctx: click.Context, paper_id: str | None, doi: str | None) -> None:
    """Resume tracking a paper."""
    _set_active(ctx, paper_id, doi, active=True)


def _set_active(
    ctx: click.Context, paper_id: str | None, doi: str | None, active: bool
) -> None:
    cfg = ctx.obj["cfg"]
    db_path = _get_db(cfg)

    with get_conn(db_path) as conn:
        paper = None
        if paper_id:
            paper = get_tracked_paper_by_id(conn, paper_id)
        elif doi:
            paper = get_tracked_paper_by_doi(conn, doi)

        if paper is None:
            console.print("[red]Paper not found.[/red]")
            sys.exit(1)

        set_tracked_paper_active(conn, paper["id"], active)
        action = "resumed" if active else "paused"
        console.print(f"[green]Paper {action}: {paper['title']}[/green]")


# ── remove ─────────────────────────────────────────────────────────────────


@main.command("remove")
@click.option("--id", "paper_id", help="ID of the paper to remove")
@click.option("--doi", help="DOI of the paper to remove")
@click.confirmation_option(prompt="This will delete all data for this paper. Continue?")
@click.pass_context
def remove_paper(ctx: click.Context, paper_id: str | None, doi: str | None) -> None:
    """Remove a paper and all its data."""
    cfg = ctx.obj["cfg"]
    db_path = _get_db(cfg)

    with get_conn(db_path) as conn:
        paper = None
        if paper_id:
            paper = get_tracked_paper_by_id(conn, paper_id)
        elif doi:
            paper = get_tracked_paper_by_doi(conn, doi)

        if paper is None:
            console.print("[red]Paper not found.[/red]")
            sys.exit(1)

        delete_tracked_paper(conn, paper["id"])
        console.print(f"[green]Removed paper: {paper['title']}[/green]")


# ── run ────────────────────────────────────────────────────────────────────


def _select_papers(conn: Any, paper_id: str | None) -> list[Any]:
    """Return the list of papers to process for this run."""
    if paper_id:
        paper = get_tracked_paper_by_id(conn, paper_id)
        if paper is None:
            paper = get_tracked_paper_by_doi(conn, paper_id)
        if paper is None:
            console.print(f"[red]Paper with ID or DOI {paper_id!r} not found.[/red]")
            sys.exit(1)
        return [paper]
    return list_tracked_papers(conn, active_only=True)


def _save_reports(report_sections: list[Any], cfg: Any) -> None:
    """Write per-paper Markdown reports to disk and print them."""
    reports_dir = cfg.reports_dir
    reports_dir.mkdir(parents=True, exist_ok=True)

    for tracked_paper, analyses_rows, failed_pdfs, scholarly_synthesis in report_sections:
        report_md = build_report(
            tracked_paper, analyses_rows, failed_pdfs, scholarly_synthesis
        )
        console.print("\n[bold]Report:[/bold]\n")
        _print_markdown(report_md)

        report_path = reports_dir / f"{tracked_paper['id']}.md"
        report_path.write_text(report_md)
        console.print(f"[bold green]Report saved to: {report_path}[/bold green]")


@main.command("run")
@click.option(
    "--id", "paper_id", default=None,
    help="Process a single tracked paper by numeric ID or DOI",
)
@click.option(
    "--workers", default=8, show_default=True, type=int,
    help="Worker threads for fetch/parse/analyse",
)
@click.option("--triggered-by", default="manual", hidden=True)
@click.pass_context
def run_pipeline(
    ctx: click.Context,
    paper_id: str | None,
    workers: int,
    triggered_by: str,
) -> None:
    """Run the full discovery → fetch → parse → analyse → report pipeline."""
    cfg = ctx.obj["cfg"]
    db_path = _get_db(cfg)

    with get_conn(db_path) as conn:
        papers_to_run = _select_papers(conn, paper_id)

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
        console.print(f"\n[bold]Processing:[/bold] {paper_dict.get('title', 'N/A')}")
        new, analysed, errors, section = process_paper(
            paper_dict, cfg, db_path, workers=max(1, workers)
        )
        total_new += new
        total_analysed += analysed
        all_errors.extend(errors)
        if section:
            report_sections.append(section)

    if report_sections:
        _save_reports(report_sections, cfg)

    with get_conn(db_path) as conn:
        finish_run(conn, run_id, total_new, total_analysed, all_errors)

    console.print(
        f"\n[bold green]Done.[/bold green] "
        f"New papers: {total_new}, Analysed: {total_analysed}, "
        f"Errors: {len(all_errors)}"
    )


# ── ingest ─────────────────────────────────────────────────────────────────


@main.command("ingest")
@click.argument("pdf_path", type=click.Path(exists=True))
@click.option(
    "--id", "paper_id", required=True, help="ID of the tracked paper being cited"
)
@click.option(
    "--doi",
    help="DOI of the citing paper (optional, will try to extract if not provided)",
)
@click.pass_context
def ingest(ctx: click.Context, pdf_path: str, paper_id: str, doi: str | None) -> None:
    """Ingest a manually downloaded PDF as a citation for a tracked paper."""
    cfg = ctx.obj["cfg"]
    db_path = _get_db(cfg)

    from citation_tracker.parser import extract_text

    with get_conn(db_path) as conn:
        tracked = get_tracked_paper_by_id(conn, paper_id)
        if tracked is None:
            console.print(f"[red]Tracked paper with ID {paper_id} not found.[/red]")
            sys.exit(1)

    citing_paper = resolve_paper(url=None, doi=doi, ss_id=None, cfg=cfg)
    if citing_paper is None:
        title_guess = Path(pdf_path).stem.replace("_", " ").replace("-", " ")
        citing_paper = {
            "doi": doi,
            "title": title_guess,
            "authors": None,
            "year": None,
            "abstract": None,
            "ss_id": None,
            "oa_id": None,
            "pdf_url": None,
        }

    with get_conn(db_path) as conn:
        cid, _is_new = upsert_citing_paper(conn, paper_id, citing_paper)

        safe_doi = _doi_to_path(citing_paper.get("doi") or f"manual-{cid}")
        dest_dir = cfg.pdfs_dir / safe_doi
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / "paper.pdf"
        shutil.copy2(pdf_path, dest)

        text = extract_text(dest)
        update_citing_paper_pdf(conn, cid, "manual")
        if text:
            update_citing_paper_text(conn, cid, text)

    console.print(
        f"[green]Ingested citation for paper ID {paper_id}: "
        f"{citing_paper['title']}[/green]"
    )
    if not text:
        console.print("[yellow]Warning: could not extract text from PDF.[/yellow]")


# ── citations ──────────────────────────────────────────────────────────────


@main.command("citations")
@click.option("--id", "paper_id", required=True, help="ID of the tracked paper")
@click.pass_context
def list_citations(ctx: click.Context, paper_id: str) -> None:
    """List all citing papers for a given tracked paper."""
    cfg = ctx.obj["cfg"]
    db_path = _get_db(cfg)

    with get_conn(db_path) as conn:
        tracked = get_tracked_paper_by_id(conn, paper_id)
        if tracked is None:
            console.print(f"[red]Paper with ID {paper_id} not found.[/red]")
            sys.exit(1)
        citations = list_citing_papers(conn, paper_id)

    if not citations:
        console.print(f"No citations found for: {tracked['title']}")
        return

    table = Table(title=f"Citations for: {tracked['title']}")
    table.add_column("ID", justify="right")
    table.add_column("Title")
    table.add_column("DOI")
    table.add_column("Status")
    table.add_column("Summarised")

    for c in citations:
        table.add_row(
            str(c["id"]),
            (c["title"] or "")[:50],
            c["doi"] or "",
            c["pdf_status"],
            "✓" if c["has_analysis"] else "✗",
        )
    console.print(table)


# ── status ─────────────────────────────────────────────────────────────────


@main.command("status")
@click.pass_context
def status(ctx: click.Context) -> None:
    """Show DB summary: tracked papers, citation counts, pending items."""
    cfg = ctx.obj["cfg"]
    db_path = _get_db(cfg)

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
@click.option("--id", "paper_id", help="ID of the tracked paper")
@click.option("--doi", help="DOI of the tracked paper to show")
@click.pass_context
def show(ctx: click.Context, paper_id: str | None, doi: str | None) -> None:
    """Show all analyses for a given tracked paper."""
    cfg = ctx.obj["cfg"]
    db_path = _get_db(cfg)

    with get_conn(db_path) as conn:
        paper = None
        if paper_id:
            paper = get_tracked_paper_by_id(conn, paper_id)
        elif doi:
            paper = get_tracked_paper_by_doi(conn, doi)

        if paper is None:
            console.print("[red]Tracked paper not found.[/red]")
            sys.exit(1)

        analyses = list_analyses(conn, paper["id"])
        failed_pdfs = conn.execute(
            "SELECT * FROM citing_papers"
            " WHERE tracked_paper_id=? AND pdf_status='failed'",
            (paper["id"],),
        ).fetchall()

    scholarly_synthesis = None
    if analyses:
        try:
            console.print("  Generating scholarly synthesis...")
            scholarly_synthesis = generate_scholarly_synthesis(
                dict(paper), [dict(a) for a in analyses], cfg
            )
        except Exception as exc:
            console.print(f"[yellow]Warning: synthesis failed: {exc}[/yellow]")

    report = build_report(paper, analyses, failed_pdfs, scholarly_synthesis)
    _print_markdown(report)
