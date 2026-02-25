# Citation Tracker: A Code Walkthrough

*2026-02-25T19:26:27Z by Showboat 0.6.1*
<!-- showboat-id: 3ae44c2c-d8ab-4d67-b371-50b7422a12bb -->

## Overview

Citation Tracker is a command-line tool that monitors academic papers and analyses how the scientific community is citing them. Given a paper you care about, it:

1. **Discovers** new papers that cite it via Semantic Scholar and OpenAlex APIs
2. **Fetches** PDFs of those citing papers from open-access sources
3. **Parses** the PDFs into Markdown text
4. **Analyses** each citing paper with an LLM to classify the relationship (supports / challenges / extends / uses / neutral)
5. **Reports** the results as formatted Markdown and a styled HTML page

The codebase lives entirely in `src/citation_tracker/` and is structured as a series of single-responsibility modules wired together by the CLI orchestrator.

## Project Layout

Let's start by understanding the file structure.

```bash
find src/citation_tracker -name '*.py' | sort
```

```output
src/citation_tracker/__init__.py
src/citation_tracker/analyser.py
src/citation_tracker/backends/__init__.py
src/citation_tracker/backends/claude_code.py
src/citation_tracker/backends/openrouter.py
src/citation_tracker/cli.py
src/citation_tracker/config.py
src/citation_tracker/db.py
src/citation_tracker/fetcher.py
src/citation_tracker/parser.py
src/citation_tracker/report.py
src/citation_tracker/sources/__init__.py
src/citation_tracker/sources/deduplicator.py
src/citation_tracker/sources/openalexapi.py
src/citation_tracker/sources/semantic_scholar.py
```

The modules group naturally into layers:

| Layer | Files | Role |
|---|---|---|
| **Entry point** | `cli.py` | Click commands; pipeline orchestration |
| **Configuration** | `config.py` | YAML + .env loading |
| **Persistence** | `db.py` | SQLite schema and all queries |
| **Discovery** | `sources/semantic_scholar.py`, `sources/openalexapi.py`, `sources/deduplicator.py` | Find citing papers |
| **Retrieval** | `fetcher.py` | Download PDFs |
| **Parsing** | `parser.py` | PDF → Markdown text |
| **Analysis** | `analyser.py` | LLM orchestration |
| **LLM backends** | `backends/openrouter.py`, `backends/claude_code.py` | API wrappers |
| **Reporting** | `report.py` | Markdown + HTML generation |

We will walk through each layer in the order data flows through the system.

## Step 1 — Configuration (`config.py`)

Before anything else, the tool needs to know which LLM backend to use, where to store data, and what API keys are available. All of that lives in `config.py`, which defines three dataclasses and a single loader function.

```bash
sed -n '1,91p' src/citation_tracker/config.py
```

```output
"""Configuration loading from .env and YAML files."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from dotenv import load_dotenv


@dataclass
class OpenRouterConfig:
    model: str = "minimax/minimax-m2.5"
    api_key_env: str = "OPENROUTER_API_KEY"

    @property
    def api_key(self) -> str:
        key = os.environ.get(self.api_key_env, "")
        return key


@dataclass
class ClaudeCodeConfig:
    flags: str = ""


@dataclass
class Config:
    backend: str = "openrouter"
    openrouter: OpenRouterConfig = field(default_factory=OpenRouterConfig)
    claude_code: ClaudeCodeConfig = field(default_factory=ClaudeCodeConfig)
    data_dir: Path = field(default_factory=lambda: Path.home() / ".citation-tracker")
    unpaywall_email: str = ""

    @property
    def db_path(self) -> Path:
        return self.data_dir / "tracker.db"

    @property
    def pdfs_dir(self) -> Path:
        return self.data_dir / "pdfs"

    @property
    def manual_dir(self) -> Path:
        return self.data_dir / "manual"

    @property
    def reports_dir(self) -> Path:
        return self.data_dir / "reports"


def load_config(config_path: Path | None = None, env_path: Path | None = None) -> Config:
    """Load configuration from .env and optional YAML config file."""
    # Load .env file
    env_file = env_path or Path(".env")
    if env_file.exists():
        load_dotenv(env_file)

    config = Config()

    # Load YAML config if provided or default config.yaml exists
    yaml_file = config_path or Path("config.yaml")
    if yaml_file.exists():
        with yaml_file.open() as f:
            raw = yaml.safe_load(f) or {}

        if "backend" in raw:
            config.backend = raw["backend"]

        if "openrouter" in raw:
            or_raw = raw["openrouter"]
            config.openrouter = OpenRouterConfig(
                model=or_raw.get("model", config.openrouter.model),
                api_key_env=or_raw.get("api_key_env", config.openrouter.api_key_env),
            )

        if "claude_code" in raw:
            cc_raw = raw["claude_code"]
            config.claude_code = ClaudeCodeConfig(
                flags=cc_raw.get("flags", config.claude_code.flags),
            )

        if "data_dir" in raw:
            config.data_dir = Path(raw["data_dir"]).expanduser()

        if "unpaywall_email" in raw:
            config.unpaywall_email = raw["unpaywall_email"]

    return config
```

Key points:

- **`OpenRouterConfig`** stores the model name and the *name* of the environment variable holding the API key (not the key itself). The `api_key` property reads from `os.environ` at call time, so the key is never baked into the config object.
- **`Config`** exposes four derived `Path` properties — `db_path`, `pdfs_dir`, `manual_dir`, `reports_dir` — all rooted under `data_dir` (default `~/.citation-tracker/`). This makes directory layout consistent throughout the codebase.
- **`load_config`** first calls `python-dotenv`'s `load_dotenv` to populate `os.environ`, then layering YAML values on top of the dataclass defaults. YAML wins over defaults but the code does *not* override what was already set in the environment, so a `.env` value can never accidentally be overwritten by config.yaml.

Let's also see the default `config.yaml` that ships with the repo:

```bash
cat config.yaml
```

```output
# Citation Tracker Configuration

# Backend to use for LLM analysis: openrouter or claude_code
backend: openrouter

openrouter:
  model: minimax/minimax-m2.5
  api_key_env: OPENROUTER_API_KEY

claude_code:
  flags: ""   # optional extra flags passed to claude -p

# Directory for database, PDFs, and reports
data_dir: "~/.citation-tracker"

# Your email address for the Unpaywall API (required by their usage policy)
unpaywall_email: "kaletejas2006@gmail.com"
```

## Step 2 — Database Schema (`db.py`)

All persistent state is kept in a single SQLite file. `db.py` owns every table definition and every query; nothing else reaches into SQLite directly.

### Schema

```bash
sed -n '23,89p' src/citation_tracker/db.py
```

```output
def init_db(db_path: Path) -> None:
    """Initialise the database, creating tables if they don't exist."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            PRAGMA journal_mode=WAL;
            PRAGMA foreign_keys=ON;

            CREATE TABLE IF NOT EXISTS tracked_papers (
                id          TEXT PRIMARY KEY,
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
                id                  TEXT PRIMARY KEY,
                tracked_paper_id    TEXT NOT NULL REFERENCES tracked_papers(id),
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
                id                      TEXT PRIMARY KEY,
                citing_paper_id         TEXT NOT NULL REFERENCES citing_papers(id),
                tracked_paper_id        TEXT NOT NULL REFERENCES tracked_papers(id),
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
                id                  TEXT PRIMARY KEY,
                triggered_by        TEXT NOT NULL,
                tracked_paper_id    TEXT REFERENCES tracked_papers(id),
                started_at          TEXT NOT NULL,
                finished_at         TEXT,
                new_papers_found    INTEGER,
                papers_analysed     INTEGER,
                errors              TEXT
            );
            """
        )
```

Four tables, each with an 8-character hex primary key (first 8 chars of a random UUID):

- **`tracked_papers`** — the papers the user wants to monitor. The `active` flag lets users pause tracking without losing data.
- **`citing_papers`** — papers discovered to be citing a tracked paper. Tracks the full PDF lifecycle: `pdf_status` moves through `pending → downloaded/failed/manual`, while `text_extracted` is a boolean optimisation flag so the SELECT for parsing never re-reads papers already done.
- **`analyses`** — the structured LLM output for each citing paper. Seven text fields (summary, relationship_type, new_evidence, flaws_identified, assumptions_questioned, other_notes, raw_response) plus a timestamp and the backend that produced them.
- **`runs`** — an audit log of every pipeline execution, including counts of new papers found and analysed, and a JSON array of error strings.

Two PRAGMAs are set on every connection: `WAL` journal mode (safe for concurrent readers while one writer is active, important because the pipeline uses `ThreadPoolExecutor`) and `foreign_keys=ON` (SQLite's FK enforcement is off by default).

### The Connection Context Manager

```bash
sed -n '92,104p' src/citation_tracker/db.py
```

```output
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
```

`get_conn` is a standard commit-on-success / rollback-on-exception context manager. Setting `row_factory = sqlite3.Row` means every query returns objects that support both positional and named column access (e.g. `row['title']`). Each call opens and closes its own connection, which is safe because WAL mode handles concurrent readers, and each thread only writes to its own paper's rows.

## Step 3 — The CLI Entry Point (`cli.py`)

`cli.py` is the largest file (694 lines) and serves two roles: exposing every user-facing command via Click, and orchestrating the five-stage pipeline.

### The Root Group and Global Options

```bash
sed -n '52,66p' src/citation_tracker/cli.py
```

```output
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

```

Every subcommand inherits these three global flags. `load_config` is called once here and the resulting `Config` object is stashed in `ctx.obj["cfg"]` — Click's standard mechanism for passing state to subcommands.

### The `add` Command and Paper Resolution

`add` is the starting point for any workflow. It accepts a URL, DOI, or Semantic Scholar ID and stores the paper in `tracked_papers`.

```bash
sed -n '71,99p' src/citation_tracker/cli.py
```

```output
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

```

Before inserting, it calls `_resolve_paper` to fetch canonical metadata. Let's look at that:

```bash
sed -n '101,151p' src/citation_tracker/cli.py
```

```output
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
```

The resolution strategy has three tiers:

1. **DOI provided** — try Semantic Scholar first, fall back to OpenAlex.
2. **Semantic Scholar ID provided** — fetch directly.
3. **URL provided** — extract a DOI from the URL with a regex (`10\\.\\ d{4,}/...`) and recurse. If no DOI is found, guess the title from the filename and search by title. If even that fails, a stub record is created so the paper can still be tracked manually.

This graceful degradation means the tool works even for papers not yet indexed.

### The Pipeline Command (`run`)

```bash
sed -n '277,360p' src/citation_tracker/cli.py
```

```output
@main.command("run")
@click.option("--doi", default=None, help="Process a single tracked paper by DOI")
@click.option("--backend", default=None, help="Override backend (openrouter|claude_code)")
@click.option("--workers", default=8, show_default=True, type=int, help="Worker threads for fetch/parse/analyse")
@click.option("--triggered-by", default="manual", hidden=True)
@click.pass_context
def run_pipeline(
    ctx: click.Context, doi: str | None, backend: str | None, workers: int, triggered_by: str
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
        new, analysed, errors, section = _process_paper(paper_dict, cfg, db_path, workers=max(1, workers))
        total_new += new
        total_analysed += analysed
        all_errors.extend(errors)
        if section:
            report_sections.append(section)

    # Build and save report
    if report_sections:
        from citation_tracker.report import build_full_report, render_full_report_html
        from datetime import datetime

        report_md = build_full_report(report_sections)
        console.print("\n[bold]Report (Markdown):[/bold]\n")
        _print_markdown(report_md)

        # Render HTML and save to reports_dir
        html_content = render_full_report_html(report_md)
        
        reports_dir = cfg.reports_dir
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = reports_dir / f"report_{timestamp}.html"
        
        with report_path.open("w") as f:
            f.write(html_content)
        
        console.print(f"\n[bold green]Report saved to: {report_path}[/bold green]")

    with get_conn(db_path) as conn:
        finish_run(conn, run_id, total_new, total_analysed, all_errors)

    console.print(
        f"\n[bold green]Done.[/bold green] "
        f"New papers: {total_new}, Analysed: {total_analysed}, Errors: {len(all_errors)}"
    )
```

`run_pipeline` is the orchestrator. It:

1. Resolves which papers to process (single DOI or all active papers).
2. Calls `insert_run` to start an audit record.
3. Calls `_process_paper` for each tracked paper — this is where the five-stage pipeline lives.
4. Assembles the individual report sections and renders them to HTML.
5. Calls `finish_run` to complete the audit record with counts and any error messages.

The `--triggered-by` flag is hidden from users but allows automation (cron jobs, GitHub Actions) to record who initiated the run.

## Step 4 — Discovery: Finding Citing Papers

The first stage of the pipeline queries two complementary academic APIs and merges the results. Let's walk through each in turn.

### Semantic Scholar (`sources/semantic_scholar.py`)

```bash
cat src/citation_tracker/sources/semantic_scholar.py
```

```output
"""Semantic Scholar API client for fetching paper citations."""

from __future__ import annotations

import logging
from typing import Any

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

SS_BASE = "https://api.semanticscholar.org/graph/v1"
FIELDS = "title,authors,year,abstract,externalIds,openAccessPdf"


def _paper_to_dict(p: dict[str, Any]) -> dict[str, Any]:
    doi = (p.get("externalIds") or {}).get("DOI")
    oa_pdf = p.get("openAccessPdf") or {}
    return {
        "title": p.get("title"),
        "authors": ", ".join(a.get("name", "") for a in (p.get("authors") or [])),
        "year": p.get("year"),
        "abstract": p.get("abstract"),
        "doi": doi,
        "ss_id": p.get("paperId"),
        "oa_id": None,
        "pdf_url": oa_pdf.get("url"),
    }


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def _get(url: str, params: dict[str, Any]) -> Any:
    with httpx.Client(timeout=30) as client:
        resp = client.get(url, params=params)
        resp.raise_for_status()
        return resp.json()


def search_paper_by_title(title: str) -> dict[str, Any] | None:
    """Search Semantic Scholar for a paper by title, return best match."""
    data = _get(
        f"{SS_BASE}/paper/search",
        {"query": title, "fields": FIELDS, "limit": 1},
    )
    items = data.get("data") or []
    if not items:
        return None
    return _paper_to_dict(items[0])


def get_paper_by_id(ss_id: str) -> dict[str, Any] | None:
    """Fetch paper metadata from Semantic Scholar by SS paper ID."""
    try:
        data = _get(f"{SS_BASE}/paper/{ss_id}", {"fields": FIELDS})
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 404:
            return None
        raise
    return _paper_to_dict(data)


def get_paper_by_doi(doi: str) -> dict[str, Any] | None:
    """Fetch paper metadata from Semantic Scholar by DOI."""
    try:
        data = _get(f"{SS_BASE}/paper/DOI:{doi}", {"fields": FIELDS})
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 404:
            return None
        raise
    return _paper_to_dict(data)


def get_citations(ss_id: str, limit: int = 500) -> list[dict[str, Any]]:
    """Fetch all papers citing the given Semantic Scholar paper ID."""
    results: list[dict[str, Any]] = []
    offset = 0
    while True:
        data = _get(
            f"{SS_BASE}/paper/{ss_id}/citations",
            {
                "fields": FIELDS,
                "limit": min(limit, 1000),
                "offset": offset,
            },
        )
        items = data.get("data") or []
        for item in items:
            citing = item.get("citingPaper") or {}
            if citing:
                results.append(_paper_to_dict(citing))
        if len(items) < min(limit, 1000):
            break
        offset += len(items)
        if offset >= limit:
            break
    return results
```

Two things to notice:

**Resilience via tenacity**: The private `_get` function is decorated with `@retry`, which will automatically retry up to 3 times with exponential back-off (2 s → 4 s → 10 s) on any exception. This handles transient API rate limits or network blips without the caller needing to know.

**Pagination in `get_citations`**: The Semantic Scholar citations endpoint is paginated. The loop increments `offset` by the number of items returned and stops when either the page is short (last page) or we hit the configured limit. This pattern avoids loading everything into memory at once for highly-cited papers.

### OpenAlex (`sources/openalexapi.py`)

```bash
cat src/citation_tracker/sources/openalexapi.py
```

```output
"""OpenAlex API client for fetching paper citations."""

from __future__ import annotations

import logging
from typing import Any
from urllib.parse import quote

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

OA_BASE = "https://api.openalex.org"


def _work_to_dict(w: dict[str, Any]) -> dict[str, Any]:
    doi_raw = w.get("doi") or ""
    doi = doi_raw.replace("https://doi.org/", "") if doi_raw else None
    oa_pdf_url: str | None = None
    best_oa = w.get("best_oa_location") or {}
    if best_oa.get("pdf_url"):
        oa_pdf_url = best_oa["pdf_url"]

    authors = ", ".join(
        (a.get("author") or {}).get("display_name", "")
        for a in (w.get("authorships") or [])
    )
    return {
        "title": w.get("display_name") or w.get("title"),
        "authors": authors,
        "year": w.get("publication_year"),
        "abstract": None,
        "doi": doi,
        "ss_id": None,
        "oa_id": w.get("id"),
        "pdf_url": oa_pdf_url,
    }


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def _get(url: str, params: dict[str, Any]) -> Any:
    with httpx.Client(timeout=30) as client:
        resp = client.get(url, params=params)
        resp.raise_for_status()
        return resp.json()


def search_paper_by_title(title: str) -> dict[str, Any] | None:
    """Search OpenAlex for a paper by title, return best match."""
    data = _get(
        f"{OA_BASE}/works",
        {"search": title, "per_page": 1},
    )
    results = data.get("results") or []
    if not results:
        return None
    return _work_to_dict(results[0])


def get_paper_by_doi(doi: str) -> dict[str, Any] | None:
    """Fetch paper metadata from OpenAlex by DOI."""
    try:
        data = _get(f"{OA_BASE}/works/doi:{quote(doi, safe='')}", {})
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 404:
            return None
        raise
    return _work_to_dict(data)


def get_citations(oa_id: str, max_results: int = 500) -> list[dict[str, Any]]:
    """Fetch papers citing the given OpenAlex work ID."""
    results: list[dict[str, Any]] = []
    page = 1
    per_page = 100
    while len(results) < max_results:
        data = _get(
            f"{OA_BASE}/works",
            {
                "filter": f"cites:{oa_id}",
                "per_page": per_page,
                "page": page,
                "select": "id,doi,display_name,publication_year,authorships,best_oa_location",
            },
        )
        items = data.get("results") or []
        results.extend(_work_to_dict(w) for w in items)
        if len(items) < per_page:
            break
        page += 1
    return results[:max_results]
```

OpenAlex uses a page-based (not offset-based) pagination style and a different API shape, but the same tenacity retry wrapper. One important normalisation: OpenAlex returns DOIs as full URLs (e.g. `https://doi.org/10.1234/...`), so `_work_to_dict` strips the prefix to produce a bare DOI, matching the format from Semantic Scholar.

Both clients produce dicts with the same keys (`title`, `authors`, `year`, `doi`, `ss_id`, `oa_id`, `pdf_url`) — a shared schema that lets downstream code stay source-agnostic.

### Deduplication (`sources/deduplicator.py`)

```bash
cat src/citation_tracker/sources/deduplicator.py
```

```output
"""Deduplicator: merge citing paper lists from multiple sources."""

from __future__ import annotations

from typing import Any


def deduplicate(papers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Merge a list of paper dicts, deduplicating by DOI (case-insensitive).

    Papers without a DOI are kept as-is (deduplicated by ss_id if available,
    otherwise kept verbatim).

    When two records refer to the same paper, the merge keeps all non-None
    fields from both, preferring the first occurrence for conflicts.
    """
    doi_map: dict[str, dict[str, Any]] = {}
    ss_map: dict[str, dict[str, Any]] = {}
    unique: list[dict[str, Any]] = []

    for paper in papers:
        doi = (paper.get("doi") or "").strip().lower()
        ss_id = paper.get("ss_id") or ""

        if doi:
            if doi in doi_map:
                _merge_into(doi_map[doi], paper)
                continue
            doi_map[doi] = paper
            unique.append(paper)
        elif ss_id:
            if ss_id in ss_map:
                _merge_into(ss_map[ss_id], paper)
                continue
            ss_map[ss_id] = paper
            unique.append(paper)
        else:
            unique.append(paper)

    return unique


def _merge_into(target: dict[str, Any], source: dict[str, Any]) -> None:
    """Fill None fields in *target* with values from *source*."""
    for key, value in source.items():
        if target.get(key) is None and value is not None:
            target[key] = value
```

Since both Semantic Scholar and OpenAlex may return the same paper, the combined list is deduplicated before writing to the database.

The algorithm is a single O(n) pass using two hash maps:
- **`doi_map`** keyed on lowercased DOI (case-insensitive)
- **`ss_map`** keyed on Semantic Scholar ID (for papers without a DOI)

When a duplicate is found, `_merge_into` fills any `None` fields in the first record with values from the duplicate. This means: if Semantic Scholar gave us the abstract but OpenAlex gave us the open-access PDF URL, the merged record will have both. The first occurrence wins on conflicts.

Papers without a DOI and without a Semantic Scholar ID are appended verbatim (no dedup is possible).

### Upserting into the Database

```bash
sed -n '186,233p' src/citation_tracker/db.py
```

```output
def upsert_citing_paper(
    conn: sqlite3.Connection, tracked_paper_id: str, paper: dict[str, Any]
) -> tuple[str, bool]:
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

    paper_id = _generate_id()
    conn.execute(
        """
        INSERT INTO citing_papers
            (id, tracked_paper_id, doi, title, authors, year, abstract,
             ss_id, oa_id, pdf_url, pdf_status, created_at)
        VALUES
            (:id, :tracked_paper_id, :doi, :title, :authors, :year, :abstract,
             :ss_id, :oa_id, :pdf_url, 'pending', :created_at)
        """,
        {
            "id": paper_id,
            "tracked_paper_id": tracked_paper_id,
            "doi": paper.get("doi"),
            "title": paper.get("title"),
            "authors": json.dumps(paper.get("authors")) if isinstance(paper.get("authors"), (list, dict)) else paper.get("authors"),
            "year": paper.get("year"),
            "abstract": paper.get("abstract"),
            "ss_id": paper.get("ss_id"),
            "oa_id": paper.get("oa_id"),
            "pdf_url": paper.get("pdf_url"),
            "created_at": _now(),
        },
    )
    return paper_id, True
```

The database layer performs a second deduplication check for safety: before inserting, it queries by DOI then by SS ID. This handles the case where the same paper was discovered in a *previous* run — the pipeline will skip re-inserting it and return `is_new=False`. The return tuple lets the caller count truly new papers.

## Step 5 — Fetching PDFs (`fetcher.py`)

After discovering which papers cite our tracked paper, the pipeline tries to download their PDFs. The main entry point is `try_download_citing_paper`, which works through a prioritised list of sources.

```bash
sed -n '62,88p' src/citation_tracker/fetcher.py
```

```output
def try_download_citing_paper(
    paper: dict[str, Any], pdfs_dir: Path, email: str = "citation-tracker@example.com"
) -> Path | None:
    """
    Try all known PDF URLs for a citing paper dict.
    Returns local path if any succeeds, None otherwise.
    """
    doi = paper.get("doi")
    url = paper.get("pdf_url")
    if url:
        path = download_pdf(url, pdfs_dir, doi=doi)
        if path:
            return path

    # Fallback: try Unpaywall if DOI is known
    if doi:
        path = _try_unpaywall(doi, pdfs_dir, email=email)
        if path:
            return path
        path = _try_crossref(doi, pdfs_dir)
        if path:
            return path
        path = _try_arxiv(doi=doi, paper=paper, pdfs_dir=pdfs_dir)
        if path:
            return path

    return None
```

The priority chain is:
1. **Direct PDF URL** — from the paper metadata (Semantic Scholar's `openAccessPdf` or OpenAlex's `best_oa_location`)
2. **Unpaywall** — an open-access lookup API that knows about legal, free-to-access copies of papers (requires your email per their usage policy)
3. **Crossref** — the publisher's own metadata, which sometimes includes direct PDF links
4. **arXiv** — for papers with arXiv DOIs (pattern `10.48550/arXiv.*`)

Each attempt returns `None` on failure; the function returns the first successful `Path`. Let's see how the base `download_pdf` function works:

```bash
sed -n '23,59p' src/citation_tracker/fetcher.py
```

```output
def download_pdf(
    pdf_url: str,
    pdfs_dir: Path,
    doi: str | None = None,
    filename_hint: str | None = None,
) -> Path | None:
    """
    Attempt to download a PDF from *pdf_url*.

    Saves to *pdfs_dir*/<doi-or-random>/<filename>.pdf.
    Returns the local path on success, None on failure.
    """
    subdir_name = _doi_to_path(doi) if doi else (filename_hint or "unknown")
    subdir = pdfs_dir / subdir_name
    subdir.mkdir(parents=True, exist_ok=True)
    dest = subdir / "paper.pdf"

    if dest.exists():
        logger.debug("PDF already downloaded: %s", dest)
        return dest

    try:
        with httpx.Client(timeout=60, follow_redirects=True, headers=_HEADERS) as client:
            resp = client.get(pdf_url)
            resp.raise_for_status()
            content_type = resp.headers.get("content-type", "")
            if "pdf" not in content_type and not pdf_url.lower().endswith(".pdf"):
                logger.warning(
                    "URL %s returned content-type %s, skipping", pdf_url, content_type
                )
                return None
            dest.write_bytes(resp.content)
            logger.info("Downloaded PDF to %s", dest)
            return dest
    except Exception as exc:
        logger.warning("Failed to download PDF from %s: %s", pdf_url, exc)
        return None
```

Key behaviours of `download_pdf`:

- **Directory naming**: DOIs are sanitised (non-alphanumeric chars → `_`) via `_doi_to_path` to create a safe filesystem path under `~/.citation-tracker/pdfs/`.
- **Idempotent**: If `paper.pdf` already exists, it is returned immediately — re-running the pipeline never re-downloads.
- **Content-type guard**: If the server responds with a non-PDF content type and the URL doesn't end in `.pdf`, the download is rejected. This prevents silently saving HTML error pages as PDFs.

The pipeline runs fetches in parallel using `ThreadPoolExecutor`. Each thread calls `try_download_citing_paper` independently; database updates (status → 'downloaded' or 'failed') are serialised by holding separate connections per write.

## Step 6 — Parsing PDFs (`parser.py`)

PDF text extraction is intentionally minimal — one function, one library.

```bash
cat src/citation_tracker/parser.py
```

```output
"""PDF text extraction using PyMuPDF4LLM."""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def extract_text(pdf_path: Path) -> str | None:
    """
    Extract Markdown-formatted text from a PDF file using PyMuPDF4LLM.

    Returns the extracted text, or None if extraction fails.
    """
    try:
        import pymupdf4llm  # type: ignore[import-untyped]

        text: str = pymupdf4llm.to_markdown(str(pdf_path))
        return text
    except Exception as exc:
        logger.warning("Failed to extract text from %s: %s", pdf_path, exc)
        return None
```

`pymupdf4llm` is a wrapper around PyMuPDF that outputs Markdown-formatted text with preserved layout information (headings, lists, tables) — intentionally designed for feeding into LLMs. The `import` is deferred inside the function so the rest of the codebase can load without it installed.

The parse stage mirrors the fetch stage: a `ThreadPoolExecutor` processes downloaded papers in parallel. After success, `update_citing_paper_text` stores the extracted text and sets `text_extracted = 1` in one atomic UPDATE, ensuring the paper is only ever parsed once.

## Step 7 — LLM Analysis (`analyser.py` and `backends/`)

This is where the intellectual work happens. The analyser converts raw PDF text into a structured assessment of how the citing paper engages with the tracked paper.

### The Analysis Prompt

```bash
sed -n '12,37p' src/citation_tracker/analyser.py
```

```output
ANALYSIS_PROMPT_TEMPLATE = """\
CONTEXT — ORIGINAL PAPER
Title: {tracked_title}
Authors: {tracked_authors}
Abstract: {tracked_abstract}

CITING PAPER
Title: {citing_title}
Full text (Markdown):
{extracted_text}

TASK
Analyse how this citing paper engages with the original paper above.
Return a JSON object with exactly these fields:

{{
  "summary": "2–3 sentence overview of the citing paper's argument",
  "relationship_type": "one of: supports | challenges | extends | uses | neutral",
  "new_evidence": "any new empirical or theoretical evidence introduced, or null",
  "flaws_identified": "any flaws or limitations of the original paper raised, or null",
  "assumptions_questioned": "any assumptions of the original paper challenged, or null",
  "other_notes": "anything else notable about how this paper engages, or null"
}}

Return only valid JSON. No preamble.\
"""
```

The prompt gives the LLM both papers' context and demands a strict JSON envelope with six fields. The `relationship_type` vocabulary is tightly constrained to five values so the report can categorise and colour-code analyses consistently.

### Handling Long Papers: Map-Reduce

```bash
sed -n '39,118p' src/citation_tracker/analyser.py
```

```output
# Rough token estimate: ~4 chars per token. Keep well below 100k tokens.
_MAX_TEXT_CHARS = 300_000
_CHUNK_SIZE = 80_000
_CHUNK_SUMMARY_PROMPT = """\
Summarise the following section of an academic paper in 3–5 sentences:

{chunk}

Return only the summary text, no preamble.\
"""
_REDUCE_SUMMARY_PROMPT = """\
You are combining section summaries from a long academic paper.

Section summaries:
{summaries}

Produce a coherent synthesis (8-12 sentences) that preserves key nuance:
- main argument and method
- strongest evidence/results
- caveats or limitations
- points most relevant to how this paper may engage with another work

Return only the synthesis text, no preamble.\
"""


def _get_backend(config: Config) -> Any:
    if config.backend == "openrouter":
        from citation_tracker.backends import openrouter

        return lambda prompt: openrouter.analyse(prompt, config.openrouter)
    elif config.backend == "claude_code":
        from citation_tracker.backends import claude_code

        return lambda prompt: claude_code.analyse(prompt, config.claude_code)
    else:
        raise ValueError(f"Unknown backend: {config.backend!r}")


def _call_llm_text(prompt: str, config: Config) -> str:
    """Call the LLM and return raw text (not JSON-parsed)."""
    if config.backend == "openrouter":
        from openai import OpenAI
        from citation_tracker.backends.openrouter import OPENROUTER_BASE_URL

        client = OpenAI(api_key=config.openrouter.api_key, base_url=OPENROUTER_BASE_URL)
        resp = client.chat.completions.create(
            model=config.openrouter.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )
        return resp.choices[0].message.content or ""
    elif config.backend == "claude_code":
        import subprocess

        cmd = ["claude", "-p"]
        if config.claude_code.flags:
            cmd.extend(config.claude_code.flags.split())
        result = subprocess.run(cmd, input=prompt, capture_output=True, text=True, check=True)
        return result.stdout.strip()
    else:
        raise ValueError(f"Unknown backend: {config.backend!r}")


def _map_reduce(text: str, config: Config) -> str:
    """Chunk text and summarise each chunk, then synthesise."""
    chunks = [text[i : i + _CHUNK_SIZE] for i in range(0, len(text), _CHUNK_SIZE)]
    summaries: list[str] = []
    for i, chunk in enumerate(chunks):
        logger.debug("Summarising chunk %d/%d", i + 1, len(chunks))
        prompt = _CHUNK_SUMMARY_PROMPT.format(chunk=chunk)
        summaries.append(_call_llm_text(prompt, config))
    if len(summaries) == 1:
        return summaries[0]
    return _call_llm_text(
        _REDUCE_SUMMARY_PROMPT.format(
            summaries="\n\n".join(f"Section {i + 1}: {s}" for i, s in enumerate(summaries))
        ),
        config,
    )
```

Academic papers can be very long. If the extracted text exceeds 300,000 characters (~75k tokens), a two-pass map-reduce strategy is used:

1. **Map** — split the text into 80,000-character chunks and ask the LLM to summarise each chunk in 3–5 sentences.
2. **Reduce** — if there was more than one chunk, send all section summaries back to the LLM and ask for a coherent 8–12 sentence synthesis.

This keeps every individual LLM call within context limits while preserving nuance that naive truncation would lose.

Notice that `_call_llm_text` duplicates the backend dispatch logic from `_get_backend`. That is because map-reduce needs raw text responses (not JSON-parsed), while the final analysis call needs a parsed dict. The two functions serve different return types.

### The Backend Factory (`_get_backend`)

```bash
sed -n '65,75p' src/citation_tracker/analyser.py
```

```output
def _get_backend(config: Config) -> Any:
    if config.backend == "openrouter":
        from citation_tracker.backends import openrouter

        return lambda prompt: openrouter.analyse(prompt, config.openrouter)
    elif config.backend == "claude_code":
        from citation_tracker.backends import claude_code

        return lambda prompt: claude_code.analyse(prompt, config.claude_code)
    else:
        raise ValueError(f"Unknown backend: {config.backend!r}")
```

`_get_backend` returns a callable `prompt -> dict`. The caller (`analyse_citing_paper`) never needs to know which backend was chosen — it just calls the returned lambda.

### The Two LLM Backends

```bash
cat src/citation_tracker/backends/openrouter.py
```

```output
"""OpenRouter backend for LLM analysis via the OpenAI-compatible API."""

from __future__ import annotations

import json
import logging
from typing import Any

from openai import OpenAI

from citation_tracker.config import OpenRouterConfig

logger = logging.getLogger(__name__)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def analyse(
    prompt: str,
    config: OpenRouterConfig,
) -> dict[str, Any]:
    """
    Send *prompt* to OpenRouter and return the parsed JSON response dict.

    Raises ValueError if the response cannot be parsed as JSON.
    """
    client = OpenAI(
        api_key=config.api_key,
        base_url=OPENROUTER_BASE_URL,
    )

    response = client.chat.completions.create(
        model=config.model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
    )

    raw = response.choices[0].message.content or ""
    logger.debug("OpenRouter raw response: %s", raw[:200])

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Try to extract JSON block from markdown code fences
        import re

        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
        if match:
            return json.loads(match.group(1))
        raise ValueError(f"Could not parse JSON from OpenRouter response: {raw[:200]}")
```

```bash
cat src/citation_tracker/backends/claude_code.py
```

```output
"""Claude Code backend: subprocess wrapper for `claude -p`."""

from __future__ import annotations

import json
import logging
import re
import subprocess
from typing import Any

from citation_tracker.config import ClaudeCodeConfig

logger = logging.getLogger(__name__)


def analyse(prompt: str, config: ClaudeCodeConfig) -> dict[str, Any]:
    """
    Run `claude -p` with *prompt* as stdin and return the parsed JSON response.

    Raises subprocess.CalledProcessError if claude exits non-zero.
    Raises ValueError if the response cannot be parsed as JSON.
    """
    cmd = ["claude", "-p"]
    if config.flags:
        cmd.extend(config.flags.split())

    result = subprocess.run(
        cmd,
        input=prompt,
        capture_output=True,
        text=True,
        check=True,
    )

    raw = result.stdout.strip()
    logger.debug("claude -p raw response: %s", raw[:200])

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
        if match:
            return json.loads(match.group(1))
        raise ValueError(f"Could not parse JSON from claude response: {raw[:200]}")
```

Both backends share the same JSON-parsing strategy:

1. Try `json.loads` on the raw response.
2. If that fails, hunt for a ```json ... ``` code fence using a regex — some LLMs wrap their JSON in markdown even when told not to.
3. Raise `ValueError` if neither works.

The difference is the transport layer:
- **OpenRouter** — uses the `openai` Python SDK with `base_url` overridden to `https://openrouter.ai/api/v1`. This makes OpenRouter a drop-in replacement for the OpenAI API, giving access to 200+ models.
- **Claude Code** — runs `claude -p` as a subprocess with the prompt on stdin. This avoids API keys entirely and runs using whatever Claude Code session is active locally.

`temperature=0.1` is used throughout for consistency — low randomness ensures repeatable analysis results when re-running the same papers.

## Step 8 — Report Generation (`report.py`)

After analysis, the pipeline collects all results and renders them as Markdown (displayed in the terminal) and HTML (saved to disk).

### Building the Influence Graph

```bash
sed -n '14,26p' src/citation_tracker/report.py
```

```output
def _build_influence_graph(
    tracked_title: str,
    analyses: list[sqlite3.Row],
) -> str:
    lines = ["```mermaid", "graph TD", '  T["Tracked: ' + tracked_title.replace('"', '\\"') + '"]']
    for idx, analysis in enumerate(analyses, 1):
        node = f"C{idx}"
        citing_title = (analysis["citing_title"] or "Untitled").replace('"', '\\"')
        relation = analysis["relationship_type"] or "neutral"
        lines.append(f'  {node}["{citing_title}"]')
        lines.append(f"  T -->|{relation}| {node}")
    lines.append("```")
    return "\n".join(lines)
```

The influence graph is a Mermaid `graph TD` (top-down directed graph) embedded as a code block in the Markdown. Each citing paper becomes a node labelled with its title, connected to the tracked paper with an edge labelled with the relationship type (e.g. `supports`, `challenges`). GitHub renders Mermaid diagrams natively.

### Building the Full Report

```bash
sed -n '29,89p' src/citation_tracker/report.py
```

```output
def build_report(
    tracked_paper: sqlite3.Row,
    analyses: list[sqlite3.Row],
    failed_pdfs: list[sqlite3.Row],
) -> str:
    """
    Build a Markdown report for a single tracked paper.

    *analyses* rows must include citing_title, citing_authors, citing_year,
    citing_doi plus the analysis fields.
    *failed_pdfs* are citing_papers rows with pdf_status='failed'.
    """
    lines: list[str] = []
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    title = tracked_paper["title"] or "Untitled"
    authors = _format_authors(tracked_paper["authors"])
    lines.append(f"# Citation Report: {title}")
    lines.append(f"*{authors}*  ")
    if tracked_paper["year"]:
        lines.append(f"*Year: {tracked_paper['year']}*  ")
    if tracked_paper["doi"]:
        lines.append(f"*DOI: {tracked_paper['doi']}*  ")
    lines.append(f"\n*Generated: {now}*\n")

    if analyses:
        lines.append(f"## New Citing Papers ({len(analyses)} analysed)\n")
        lines.append("## Influence Tree\n")
        lines.append(_build_influence_graph(title, analyses))
        lines.append("")
        for a in analyses:
            lines.append(f"### {a['citing_title'] or 'Untitled'}")
            lines.append(
                f"**Authors:** {_format_authors(a['citing_authors'])}  "
                f"**Year:** {a['citing_year'] or 'N/A'}  "
                f"**DOI:** {a['citing_doi'] or 'N/A'}"
            )
            lines.append(f"\n**Relationship:** `{a['relationship_type'] or 'N/A'}`\n")
            if a["summary"]:
                lines.append(f"**Summary:** {a['summary']}\n")
            if a["new_evidence"]:
                lines.append(f"**New Evidence:** {a['new_evidence']}\n")
            if a["flaws_identified"]:
                lines.append(f"**Flaws Identified:** {a['flaws_identified']}\n")
            if a["assumptions_questioned"]:
                lines.append(f"**Assumptions Questioned:** {a['assumptions_questioned']}\n")
            if a["other_notes"]:
                lines.append(f"**Other Notes:** {a['other_notes']}\n")
            lines.append("---\n")
    else:
        lines.append("## No new analysed citations in this run.\n")

    if failed_pdfs:
        lines.append("## PDFs Requiring Manual Download\n")
        lines.append("The following papers could not be automatically downloaded:\n")
        for fp in failed_pdfs:
            doi_str = f" (DOI: {fp['doi']})" if fp["doi"] else ""
            lines.append(f"- {fp['title'] or 'Untitled'}{doi_str}")
        lines.append("")

    return "\n".join(lines)
```

`build_report` produces a Markdown string for a single tracked paper. The structure is:
- Header: title, authors, year, DOI, generation timestamp
- Influence Tree Mermaid diagram
- One `###` section per analysed citing paper with all six LLM fields
- A trailing list of papers whose PDFs could not be downloaded (so users can manually retrieve them)

`build_full_report` simply concatenates multiple `build_report` outputs separated by `---` dividers.

### HTML Rendering

```bash
sed -n '102,147p' src/citation_tracker/report.py
```

```output
def render_full_report_html(markdown_content: str) -> str:
    """Render the Markdown report as a full HTML page with basic styling."""
    try:
        import markdown
    except ImportError:
        return f"<html><body><pre>{markdown_content}</pre></body></html>"

    html_body = markdown.markdown(markdown_content, extensions=["extra", "toc"])

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Citation Tracker Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 800px;
            margin: 40px auto;
            padding: 0 20px;
            background-color: #f9f9f9;
        }}
        .container {{
            background: white;
            padding: 40px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1, h2, h3 {{ color: #1a1a1a; }}
        h1 {{ border-bottom: 2px solid #eee; padding-bottom: 10px; }}
        h2 {{ margin-top: 40px; border-bottom: 1px solid #eee; }}
        code {{ background: #f4f4f4; padding: 2px 4px; border-radius: 4px; }}
        hr {{ border: 0; border-top: 2px solid #eee; margin: 40px 0; }}
        .metadata {{ color: #666; font-style: italic; }}
    </style>
</head>
<body>
    <div class="container">
        {html_body}
    </div>
</body>
</html>
"""
```

The `markdown` library converts the Markdown string to HTML with the `extra` extension (tables, fenced code blocks) and `toc` (auto-generated table of contents anchors). If the library isn't installed, the function falls back to wrapping raw Markdown in a `<pre>` tag.

The embedded CSS keeps the report readable without any external dependencies: a centred white card on a grey background, system fonts, and section separators. Reports are saved as `report_YYYYMMDD_HHMMSS.html` so each run produces a distinct, timestamped snapshot.

## Step 9 — Tying It All Together: `_process_paper`

Now we can read the full pipeline function with all the pieces in context:

```bash
sed -n '363,523p' src/citation_tracker/cli.py
```

```output
def _process_paper(
    paper: dict[str, Any],
    cfg: Any,
    db_path: Path,
    workers: int = 8,
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

    def _fetch_one(cp_row: Any) -> tuple[str, bool]:
        cp_dict = dict(cp_row)
        path = try_download_citing_paper(
            cp_dict, cfg.pdfs_dir, email=cfg.unpaywall_email or "citation-tracker@example.com"
        )
        return cp_dict["id"], bool(path)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_fetch_one, cp) for cp in pending]
        for future in as_completed(futures):
            cp_id, ok = future.result()
            with get_conn(db_path) as conn:
                update_citing_paper_pdf(conn, cp_id, "downloaded" if ok else "failed")

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

    def _parse_one(cp_row: Any) -> tuple[str, str | None]:
        from citation_tracker.fetcher import _doi_to_path

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

    # ── 4. ANALYSE ─────────────────────────────────────────────────────────
    with get_conn(db_path) as conn:
        to_analyse = get_citing_papers_for_analysis(conn, tracked_id)

    papers_analysed = 0
    def _analyse_one(cp_row: Any) -> tuple[str, dict[str, Any]]:
        cp_dict = dict(cp_row)
        result = analyse_citing_paper(
            tracked_paper=paper,
            citing_paper=cp_dict,
            extracted_text=cp_dict.get("extracted_text") or "",
            config=cfg,
        )
        return cp_dict["id"], result

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_analyse_one, cp) for cp in to_analyse]
        for future in as_completed(futures):
            try:
                cp_id, result = future.result()
                import json

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
                            "assumptions_questioned": result.get("assumptions_questioned"),
                            "other_notes": result.get("other_notes"),
                            "raw_response": json.dumps(result),
                        },
                    )
                papers_analysed += 1
            except Exception as exc:
                errors.append(f"Analysis failed: {exc}")
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
```

Reading `_process_paper` linearly, the five-stage flow is clearly visible in the comments:

**Stage 1 — Discover**: Try Semantic Scholar then OpenAlex (each in a `try/except` so one failure doesn't kill the other), combine results, deduplicate, upsert into the DB. Errors are collected but don't stop the run.

**Stage 2 — Fetch**: Read only `pdf_status='pending'` rows (idempotent — already-downloaded papers are never re-fetched). Run up to `workers` downloads in parallel. Each thread returns `(id, success)`; the main thread serialises the DB update.

**Stage 3 — Parse**: Read only rows with `pdf_status IN ('downloaded','manual') AND text_extracted=0`. Parallelise extraction. The `text_extracted` flag ensures re-runs skip already-parsed papers.

**Stage 4 — Analyse**: `get_citing_papers_for_analysis` uses a `LEFT JOIN` to find papers with text but no analysis yet — another idempotency guard. Analysis errors are caught per-paper so one bad LLM response doesn't block others.

**Stage 5 — Report**: Collect all analyses (including from previous runs) and failed PDFs, then return as a tuple for the outer loop to assemble into the full report.

This staged, idempotent design means the pipeline can be interrupted at any point and safely re-run without duplicating work.

## Step 10 — Supporting Commands

Beyond `run`, the CLI provides several management commands. Here is a quick survey:

### `ingest` — Manual PDF Upload

```bash
sed -n '529,586p' src/citation_tracker/cli.py
```

```output
@main.command("ingest")
@click.argument("pdf_path", type=click.Path(exists=True))
@click.option("--id", "paper_id", required=True, help="ID of the tracked paper being cited")
@click.option("--doi", help="DOI of the citing paper (optional, will try to extract if not provided)")
@click.pass_context
def ingest(ctx: click.Context, pdf_path: str, paper_id: str, doi: str | None) -> None:
    """Ingest a manually downloaded PDF as a citation for a tracked paper."""
    cfg = ctx.obj["cfg"]
    db_path = _get_db(cfg)

    from citation_tracker.db import get_conn, get_tracked_paper_by_id, upsert_citing_paper
    from citation_tracker.fetcher import _doi_to_path
    from citation_tracker.parser import extract_text
    import shutil

    with get_conn(db_path) as conn:
        tracked = get_tracked_paper_by_id(conn, paper_id)
        if tracked is None:
            console.print(f"[red]Tracked paper with ID {paper_id} not found.[/red]")
            sys.exit(1)

    # Resolve citing paper metadata
    citing_paper = _resolve_paper(url=None, doi=doi, ss_id=None)
    if citing_paper is None:
        # If we can't resolve metadata, create a stub from filename
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
        cid, is_new = upsert_citing_paper(conn, paper_id, citing_paper)
        
        # Copy PDF to pdfs dir
        safe_doi = _doi_to_path(citing_paper.get("doi") or f"manual-{cid}")
        dest_dir = cfg.pdfs_dir / safe_doi
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / "paper.pdf"
        shutil.copy2(pdf_path, dest)

        text = extract_text(dest)
        from citation_tracker.db import update_citing_paper_pdf, update_citing_paper_text

        update_citing_paper_pdf(conn, cid, "manual")
        if text:
            update_citing_paper_text(conn, cid, text)

    console.print(f"[green]Ingested citation for paper ID {paper_id}: {citing_paper['title']}[/green]")
    if not text:
        console.print("[yellow]Warning: could not extract text from PDF.[/yellow]")

```

`ingest` handles PDFs that the automated downloader couldn't retrieve (paywalled journals, etc.). It:
1. Resolves metadata via DOI if given, or creates a stub from the filename.
2. Upserts the citing paper record.
3. Copies the PDF into the managed `pdfs/` directory with status `manual`.
4. Immediately extracts text — so the paper is ready to analyse on the next `run`.

### The `get_citing_papers_for_analysis` Query

This query is the linchpin of idempotency for the analysis stage:

```bash
sed -n '245,258p' src/citation_tracker/db.py
```

```output
def get_citing_papers_for_analysis(
    conn: sqlite3.Connection, tracked_paper_id: str
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
```

The `LEFT JOIN ... WHERE a.id IS NULL` pattern is a standard SQL "find rows without a matching row in another table" idiom. It selects only citing papers that:
- have extracted text (`text_extracted = 1`)
- do **not** yet have an entry in `analyses`

This means re-running the pipeline never re-analyses a paper, even if the `run` command was killed mid-way through.

## Summary: End-to-End Data Flow

Here is the complete journey from user command to final report:

```
citation-tracker add --doi 10.xxxx/yyyy
    └─ _resolve_paper()
        ├─ semantic_scholar.get_paper_by_doi()
        └─ (fallback) openalexapi.get_paper_by_doi()
    └─ db.insert_tracked_paper()

citation-tracker run
    └─ _process_paper()
        │
        ├─ 1. DISCOVER
        │   ├─ semantic_scholar.get_citations(ss_id)  [paginated, retried]
        │   ├─ openalexapi.get_citations(oa_id)       [paginated, retried]
        │   ├─ deduplicator.deduplicate()             [O(n), merge by DOI/ss_id]
        │   └─ db.upsert_citing_paper()               [idempotent insert]
        │
        ├─ 2. FETCH  [ThreadPoolExecutor, up to 8 workers]
        │   └─ fetcher.try_download_citing_paper()
        │       ├─ direct pdf_url
        │       ├─ Unpaywall API
        │       ├─ Crossref API
        │       └─ arXiv direct URL
        │
        ├─ 3. PARSE  [ThreadPoolExecutor, up to 8 workers]
        │   └─ parser.extract_text()  [pymupdf4llm → Markdown]
        │
        ├─ 4. ANALYSE  [ThreadPoolExecutor, up to 8 workers]
        │   └─ analyser.analyse_citing_paper()
        │       ├─ (if text > 300k chars) _map_reduce() → condensed text
        │       ├─ fill ANALYSIS_PROMPT_TEMPLATE
        │       └─ backend_fn(prompt)
        │           ├─ openrouter.analyse()  [OpenAI SDK → openrouter.ai]
        │           └─ claude_code.analyse() [subprocess: claude -p]
        │
        └─ 5. REPORT
            ├─ report.build_report()           [Markdown per paper]
            ├─ report.build_full_report()      [combined Markdown]
            ├─ report.render_full_report_html() [styled HTML page]
            └─ save to ~/.citation-tracker/reports/report_TIMESTAMP.html
```

Each stage reads only the rows it needs and writes back just enough state to let the next stage proceed — making the entire pipeline safely re-runnable after any failure.

## Key Design Principles

A few cross-cutting principles show up throughout the codebase:

**Idempotency everywhere** — Every DB write is guarded by a prior check. Re-running `citation-tracker run` after an interruption will pick up exactly where it left off, never duplicating work or data.

**Error isolation** — Each stage wraps per-item failures in `try/except` and appends to an `errors` list. One bad API response or failed LLM call does not abort the whole run.

**Shared paper schema** — Both `semantic_scholar` and `openalexapi` normalise their responses to the same dict keys before returning. The rest of the code never needs to know which source a paper came from.

**Pluggable LLM backends** — Adding a new backend means implementing a single `analyse(prompt, config) -> dict` function and registering it in `_get_backend`. The rest of the pipeline is unchanged.

**Minimal dependencies at import time** — Heavy libraries (`pymupdf4llm`, `openai`) are imported inside the functions that use them. This keeps startup fast and makes the codebase testable with lightweight mocks.
