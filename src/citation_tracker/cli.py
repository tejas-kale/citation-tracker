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
    update_citing_paper_metadata,
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
@click.option("--config", "config_path", default=None, help="Path to config.yaml (default: ./config.yaml or ~/.citation-tracker/config.yaml)")
@click.option("--env", "env_path", default=None, help="Path to .env file (default: ./.env or ~/.citation-tracker/.env)")
@click.option("--verbose", is_flag=True, default=False, help="Enable DEBUG-level logging")
@click.pass_context
def main(
    ctx: click.Context,
    config_path: str | None,
    env_path: str | None,
    verbose: bool,
) -> None:
    """Citation Tracker — discover, fetch, parse, and analyse papers that cite your work.

    Runs a 5-stage pipeline (discover → fetch → parse → analyse → report) for each
    tracked paper. Results are stored in a local SQLite database and saved as Markdown
    reports in ~/.citation-tracker/reports/.

    Configuration is loaded from config.yaml (searched in ./ then ~/.citation-tracker/).
    API keys are loaded from .env (same search order). Use --config and --env to override.

    \b
    Required API keys (in .env):
      OPENROUTER_API_KEY   LLM analysis via OpenRouter (required)
      ADS_DEV_KEY          NASA ADS discovery for astronomy/physics papers (optional)
    """
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
@click.option("--doi", default=None, help="DOI of the paper (e.g. 10.1145/1234567.1234568)")
@click.option("--ss-id", default=None, help="Semantic Scholar paper ID (40-char hex or numeric)")
@click.pass_context
def add_paper(
    ctx: click.Context, url: str | None, doi: str | None, ss_id: str | None
) -> None:
    """Add a paper to track by URL, DOI, or Semantic Scholar ID.

    Resolves full metadata (title, authors, year, abstract, DOI) via Semantic Scholar
    and stores the paper in the database. Papers with a duplicate DOI are rejected with
    a message showing the existing record's ID.

    Provide exactly one of: URL argument, --doi, or --ss-id.

    \b
    Examples:
      citation-tracker add "https://arxiv.org/abs/2401.00001"
      citation-tracker add "https://example.com/paper.pdf"
      citation-tracker add --doi 10.1145/1234567.1234568
      citation-tracker add --ss-id 204e3073870fae3d05bcbc2f6a8e263d21195671
    """
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
    """List all tracked papers (ID, title, DOI/URL, year, active status, date added).

    Use the ID shown here with --id in other commands (run, show, citations, pause, etc.).
    Active=✓ means the paper is included in `run`; Active=✗ means it is paused.
    """
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
@click.option("--id", "paper_id", help="8-char hex ID of the paper (from `list`)")
@click.option("--doi", help="DOI of the paper")
@click.pass_context
def pause_paper(ctx: click.Context, paper_id: str | None, doi: str | None) -> None:
    """Pause tracking a paper (keeps all history and citations).

    Paused papers are skipped by `run` until resumed. All existing citations,
    analyses, and reports are preserved.
    """
    _set_active(ctx, paper_id, doi, active=False)


@main.command("resume")
@click.option("--id", "paper_id", help="8-char hex ID of the paper (from `list`)")
@click.option("--doi", help="DOI of the paper")
@click.pass_context
def resume_paper(ctx: click.Context, paper_id: str | None, doi: str | None) -> None:
    """Resume tracking a paused paper.

    The paper will be included in the next `run` after being resumed.
    """
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
@click.option("--id", "paper_id", help="8-char hex ID of the paper (from `list`)")
@click.option("--doi", help="DOI of the paper")
@click.confirmation_option(prompt="This will delete all data for this paper. Continue?")
@click.pass_context
def remove_paper(ctx: click.Context, paper_id: str | None, doi: str | None) -> None:
    """Remove a tracked paper and ALL its data (citations, analyses, PDFs).

    This is irreversible. A confirmation prompt is shown before proceeding.
    Use `pause` instead if you want to stop processing a paper without losing data.
    """
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

    for tracked_paper, analyses_rows, failed_pdfs, scholarly_synthesis, citing_stats, unconfirmed_analyses in report_sections:
        report_md = build_report(
            tracked_paper, analyses_rows, failed_pdfs, scholarly_synthesis, citing_stats,
            unconfirmed_analyses=unconfirmed_analyses,
        )
        console.print("\n[bold]Report:[/bold]\n")
        _print_markdown(report_md)

        report_path = reports_dir / f"{tracked_paper['id']}.md"
        report_path.write_text(report_md)
        console.print(f"[bold green]Report saved to: {report_path}[/bold green]")


@main.command("run")
@click.option(
    "--id", "paper_id", default=None,
    help="Process a single paper by 8-char hex ID or DOI (default: all active papers)",
)
@click.option(
    "--workers", default=8, show_default=True, type=int,
    help="Concurrent threads for the fetch/parse/analyse stages",
)
@click.option("--triggered-by", default="manual", hidden=True)
@click.pass_context
def run_pipeline(
    ctx: click.Context,
    paper_id: str | None,
    workers: int,
    triggered_by: str,
) -> None:
    """Run the full discovery → fetch → parse → analyse → report pipeline.

    Processes all active tracked papers (or a single paper with --id). For each paper:

    \b
    1. DISCOVER  Query Semantic Scholar, OpenAlex, and NASA ADS in parallel;
                 deduplicate citing papers across sources.
    2. FETCH     Download PDFs via: direct URL → ADS Link Gateway → Unpaywall
                 → CrossRef → arXiv. Status tracked as pending/downloaded/failed.
    3. PARSE     Extract Markdown-formatted text from PDFs using pymupdf4llm.
    4. ANALYSE   Send paper text + tracked paper metadata to the configured LLM.
                 Papers over 80 KB use map-reduce: chunk → summarise → reduce.
                 Produces: summary, relationship_type, new_evidence, flaws_identified,
                 assumptions_questioned, other_notes.
    5. REPORT    Assemble Markdown with a scholarly synthesis; save to
                 ~/.citation-tracker/reports/<id>.md.

    Fetch, parse, and analyse run concurrently across --workers threads.
    Previously analysed papers are not re-analysed unless the DB record is cleared.

    \b
    Examples:
      citation-tracker run
      citation-tracker run --id a1b2c3d4
      citation-tracker run --id 10.1145/1234567.1234568
      citation-tracker run --workers 16
    """
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
    "--id", "paper_id", required=True,
    help="8-char hex ID of the tracked paper being cited (from `list`)",
)
@click.option(
    "--doi",
    help="DOI of the citing paper (optional; title is inferred from filename if omitted)",
)
@click.pass_context
def ingest(ctx: click.Context, pdf_path: str, paper_id: str, doi: str | None) -> None:
    """Ingest a manually downloaded PDF as a citation for a tracked paper.

    Use this when automatic PDF fetching fails or for papers not in public databases.
    The PDF is copied to ~/.citation-tracker/pdfs/, text is extracted immediately,
    and the citing paper is registered in the database. It will be analysed on the
    next `run` call.

    PDF_PATH is the local path to the PDF file to ingest.

    If --doi is provided, metadata is resolved via Semantic Scholar. Otherwise,
    the title is guessed from the filename.

    \b
    Example:
      citation-tracker ingest ~/Downloads/smith2024.pdf --id a1b2c3d4
      citation-tracker ingest paper.pdf --id a1b2c3d4 --doi 10.1145/1234567.1234568
    """
    cfg = ctx.obj["cfg"]
    db_path = _get_db(cfg)

    from citation_tracker.parser import extract_text
    from citation_tracker.resolver import resolve_from_stored_text

    with get_conn(db_path) as conn:
        tracked = get_tracked_paper_by_id(conn, paper_id)
        if tracked is None:
            console.print(f"[red]Tracked paper with ID {paper_id} not found.[/red]")
            sys.exit(1)

    needs_metadata_resolution = False
    title_guess = Path(pdf_path).stem.replace("_", " ").replace("-", " ")

    citing_paper = resolve_paper(url=None, doi=doi, ss_id=None, cfg=cfg)
    if citing_paper is None:
        needs_metadata_resolution = True
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

    with get_conn(db_path) as conn:
        update_citing_paper_pdf(conn, cid, "manual")
        if text:
            update_citing_paper_text(conn, cid, text)

    if needs_metadata_resolution and text:
        console.print("  Resolving metadata from PDF content...")
        resolved = resolve_from_stored_text(text, title_guess, str(dest), cfg)
        if resolved and (resolved.get("title") or resolved.get("doi")):
            with get_conn(db_path) as conn:
                update_citing_paper_metadata(conn, cid, resolved)
            citing_paper.update({k: v for k, v in resolved.items() if v is not None})

    console.print(
        f"[green]Ingested citation for paper ID {paper_id}: "
        f"{citing_paper['title']}[/green]"
    )
    if not text:
        console.print("[yellow]Warning: could not extract text from PDF.[/yellow]")


# ── reprocess ──────────────────────────────────────────────────────────────


@main.command("reprocess")
@click.option(
    "--id", "paper_id", required=True,
    help="8-char hex ID of the tracked paper whose manually ingested citations should be reprocessed",
)
@click.pass_context
def reprocess_metadata(ctx: click.Context, paper_id: str) -> None:
    """Re-resolve metadata for manually ingested citing papers.

    Finds all manually ingested citing papers for the given tracked paper that
    have extracted text, and re-runs metadata resolution (API search + LLM
    extraction) to populate missing title, authors, year, and DOI fields.

    \b
    Example:
      citation-tracker reprocess --id a1b2c3d4
    """
    cfg = ctx.obj["cfg"]
    db_path = _get_db(cfg)

    from citation_tracker.resolver import resolve_from_stored_text

    with get_conn(db_path) as conn:
        tracked = get_tracked_paper_by_id(conn, paper_id)
        if tracked is None:
            console.print(f"[red]Paper with ID {paper_id} not found.[/red]")
            sys.exit(1)

        to_reprocess = conn.execute(
            """SELECT * FROM citing_papers
               WHERE tracked_paper_id=? AND pdf_status='manual' AND text_extracted=1""",
            (paper_id,),
        ).fetchall()

    if not to_reprocess:
        console.print("No manually ingested papers with extracted text found.")
        return

    updated = 0
    for cp_row in to_reprocess:
        cp = dict(cp_row)
        hint = cp.get("title") or "unknown"
        console.print(f"  Resolving: [dim]{hint}[/dim]")
        resolved = resolve_from_stored_text(
            cp["extracted_text"], hint, cp.get("pdf_url") or "", cfg
        )
        if resolved and (resolved.get("title") or resolved.get("doi")):
            with get_conn(db_path) as conn:
                update_citing_paper_metadata(conn, cp["id"], resolved)
            console.print(f"    → {resolved.get('title', 'N/A')} ({resolved.get('year', 'N/A')})")
            updated += 1
        else:
            console.print(f"    [yellow]Could not resolve metadata[/yellow]")

    console.print(f"\n[bold green]Updated {updated}/{len(to_reprocess)} papers.[/bold green]")


# ── citations ──────────────────────────────────────────────────────────────


@main.command("citations")
@click.option("--id", "paper_id", required=True, help="8-char hex ID of the tracked paper (from `list`)")
@click.pass_context
def list_citations(ctx: click.Context, paper_id: str) -> None:
    """List all papers that cite a given tracked paper.

    Shows each citing paper's ID, title, DOI, PDF status, and whether an LLM
    analysis has been completed.

    \b
    PDF status values:
      pending     Not yet attempted
      downloaded  PDF successfully fetched
      failed      All fetch strategies exhausted
      manual      Ingested via `ingest` command
    """
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
    """Show a database summary: tracked papers, citation counts, and pending work.

    Reports counts for: active tracked papers, total citing papers discovered,
    PDFs still pending download, and total LLM analyses completed.
    """
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
@click.option("--id", "paper_id", help="8-char hex ID of the tracked paper (from `list`)")
@click.option("--doi", help="DOI of the tracked paper")
@click.pass_context
def show(ctx: click.Context, paper_id: str | None, doi: str | None) -> None:
    """Render the full analysis report for a tracked paper in the terminal.

    Generates a scholarly synthesis on the fly using the configured LLM, then
    prints the full Markdown report. Uses `glow` for rendering if it is installed,
    otherwise falls back to `rich`.

    This does not save a file — use `run` to generate saved reports in
    ~/.citation-tracker/reports/.
    """
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
        row = conn.execute(
            """
            SELECT
                COUNT(*) AS total,
                SUM(CASE WHEN pdf_status='pending'    THEN 1 ELSE 0 END) AS pending,
                SUM(CASE WHEN pdf_status='downloaded' THEN 1 ELSE 0 END) AS downloaded,
                SUM(CASE WHEN pdf_status='failed'     THEN 1 ELSE 0 END) AS failed,
                SUM(CASE WHEN pdf_status='manual'     THEN 1 ELSE 0 END) AS manual
            FROM citing_papers WHERE tracked_paper_id=?
            """,
            (paper["id"],),
        ).fetchone()
        citing_stats = {
            "total":      row["total"]      or 0,
            "pending":    row["pending"]    or 0,
            "downloaded": row["downloaded"] or 0,
            "failed":     row["failed"]     or 0,
            "manual":     row["manual"]     or 0,
        }

    confirmed_analyses = [a for a in analyses if a["confirmed_citation"] != 0]
    unconfirmed_analyses = [a for a in analyses if a["confirmed_citation"] == 0]

    scholarly_synthesis = None
    if confirmed_analyses:
        try:
            console.print("  Generating scholarly synthesis...")
            scholarly_synthesis = generate_scholarly_synthesis(
                dict(paper), [dict(a) for a in confirmed_analyses], cfg
            )
        except Exception as exc:
            console.print(f"[yellow]Warning: synthesis failed: {exc}[/yellow]")

    report = build_report(
        paper, confirmed_analyses, failed_pdfs, scholarly_synthesis, citing_stats,
        unconfirmed_analyses=unconfirmed_analyses,
    )
    _print_markdown(report)
