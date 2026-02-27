# Citation Tracker: A Code Walkthrough

*2026-02-27T13:35:30Z by Showboat 0.6.1*
<!-- showboat-id: c76aa6ee-1b41-4a95-b264-2232a7cf36d3 -->

citation-tracker is a Python CLI that automatically discovers papers citing your research, downloads their PDFs, extracts text, and uses an LLM to analyse how each paper engages with your work.

This walkthrough follows the code in execution order: startup and configuration → adding a paper → running the pipeline (discover → fetch → parse → analyse → report). Every stage lives in its own module with a single responsibility.

## 1. Configuration

The entry point is `cli.py`. The `@click.group()` on `main()` registers all subcommands and calls `load_config()` first. The resulting `Config` object is stashed on Click's context dict so every subcommand can read it.

```bash
sed -n '61,78p' src/citation_tracker/cli.py
```

```output
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
```

Config is loaded from two sources: a `config.yaml` for model/backend settings and a `.env` file for secrets. `load_config()` merges both into a typed `Config` dataclass.

```bash
sed -n '58,97p' src/citation_tracker/config.py
```

```output
def load_config(
    config_path: Path | None = None, env_path: Path | None = None
) -> Config:
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

        if "ads" in raw:
            ads_raw = raw["ads"]
            config.ads = ADSConfig(
                api_key_env=ads_raw.get("api_key_env", config.ads.api_key_env),
            )

        if "data_dir" in raw:
            config.data_dir = Path(raw["data_dir"]).expanduser()

        if "unpaywall_email" in raw:
            config.unpaywall_email = raw["unpaywall_email"]

    return config
```

## 2. Database

The SQLite database lives at `~/.citation-tracker/tracker.db`. `init_db()` creates four tables if they don't exist and runs a lightweight migration to add any columns added since an existing database was created.

```bash
sed -n '33,115p' src/citation_tracker/db.py
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
                ads_bibcode TEXT,
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
                ads_bibcode         TEXT,
                pdf_url             TEXT,
                pdf_status          TEXT NOT NULL DEFAULT 'pending',
                extracted_text      TEXT,
                text_extracted      INTEGER NOT NULL DEFAULT 0,
                created_at          TEXT NOT NULL
            );
            """
        )
        
        # Migration: Add ads_bibcode column if it doesn't exist
        # (This is needed for existing databases created before the ADS integration)
        for table in ["tracked_papers", "citing_papers"]:
            cursor = conn.execute(f"PRAGMA table_info({table})")
            columns = [r[1] for r in cursor.fetchall()]
            if "ads_bibcode" not in columns:
                logger.info("Migrating database: adding ads_bibcode to %s", table)
                conn.execute(f"ALTER TABLE {table} ADD COLUMN ads_bibcode TEXT")

        conn.executescript(
            """
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

The four tables are:
- **tracked_papers** — papers you are monitoring
- **citing_papers** — papers discovered to cite a tracked paper (includes pdf_status and extracted_text)
- **analyses** — one LLM analysis per citing paper per tracked paper
- **runs** — history of pipeline executions

All IDs are 8-character hex UUIDs. WAL mode is on for safe concurrent reads during a run.

## 3. Adding a Paper

`citation-tracker add` is how you register a paper to track. It accepts a URL, a DOI, or a Semantic Scholar ID, and delegates metadata resolution to `resolver.py`.

```bash
grep -n 'def resolve_paper\|def _resolve_by\|def _extract_\|def _resolve_from' src/citation_tracker/resolver.py
```

```output
48:def _resolve_by_doi(doi: str, cfg: Any) -> dict[str, Any] | None:
67:def _resolve_by_ss_id(ss_id: str) -> dict[str, Any] | None:
77:def _extract_doi_from_url(url: str) -> str | None:
83:def _extract_arxiv_id_from_url(url: str) -> str | None:
90:def _resolve_by_arxiv_id(arxiv_id: str, source_url: str) -> dict[str, Any] | None:
101:def _resolve_from_pdf(url: str, cfg: Any) -> dict[str, Any] | None:
151:def _resolve_from_pdf_text(
187:def _resolve_by_url(url: str, cfg: Any) -> dict[str, Any] | None:
231:def resolve_paper(
```

`resolve_paper()` is the public entry point. Internally it tries three strategies in order:

1. **DOI** — query Semantic Scholar, OpenAlex, and NASA ADS in parallel, then merge the results (SS wins on abstract, ADS contributes its bibcode).
2. **Semantic Scholar ID** — direct fetch.
3. **URL** — extract a DOI or arXiv ID from the URL; if neither is present, download the PDF, build a search query from the text, and fall back to LLM-based metadata extraction if the API search fails. A stub entry is created as a last resort.

```bash
sed -n '48,100p' src/citation_tracker/resolver.py
```

```output
def _resolve_by_doi(doi: str, cfg: Any) -> dict[str, Any] | None:
    """Fetch and merge metadata for a DOI from all configured sources.

    Returns a merged dict or None if no source found the paper.
    """
    from citation_tracker.sources import semantic_scholar as ss
    from citation_tracker.sources import openalexapi as oa
    from citation_tracker.sources import adsapi as ads

    ss_result = ss.get_paper_by_doi(doi)
    oa_result = oa.get_paper_by_doi(doi)
    ads_result = ads.get_paper_by_doi(doi, cfg.ads.api_key)

    if not ss_result and not oa_result and not ads_result:
        return None

    return _merge_source_results(doi, ss_result, oa_result, ads_result)


def _resolve_by_ss_id(ss_id: str) -> dict[str, Any] | None:
    """Fetch paper metadata from Semantic Scholar by paper ID."""
    from citation_tracker.sources import semantic_scholar as ss

    paper = ss.get_paper_by_id(ss_id)
    if paper:
        paper["source_url"] = None
    return paper


def _extract_doi_from_url(url: str) -> str | None:
    """Extract a DOI from a URL string, returning it or None."""
    match = re.search(r"10\.\d{4,}/[^\s\"'<>]+", url)
    return match.group() if match else None


def _extract_arxiv_id_from_url(url: str) -> str | None:
    """Extract an arXiv paper ID from a URL string, returning it or None."""
    pattern = r"(\d{4}\.\d{4,5}(v\d+)?|arxiv:[a-z\-]+(\.[a-z\-]+)?/\d{7})"
    match = re.search(pattern, url.lower())
    return match.group(1) if match else None


def _resolve_by_arxiv_id(arxiv_id: str, source_url: str) -> dict[str, Any] | None:
    """Fetch paper metadata for an arXiv ID from SS or OA."""
    from citation_tracker.sources import semantic_scholar as ss
    from citation_tracker.sources import openalexapi as oa

    paper = ss.get_paper_by_arxiv(arxiv_id) or oa.get_paper_by_arxiv(arxiv_id)
    if paper:
        paper["source_url"] = source_url
    return paper


```

## 4. The Pipeline

`citation-tracker run` executes the 5-stage pipeline for every active tracked paper. The orchestration lives in `pipeline.py`. `cli.py` calls `process_paper()` and then writes the resulting Markdown report to `~/.citation-tracker/reports/<id>.md`.

```bash
sed -n '157,205p' src/citation_tracker/pipeline.py
```

```output
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
```

```bash
sed -n '215,260p' src/citation_tracker/pipeline.py
```

```output

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
```

### Stage 1: Discover

`_discover_citations()` fans out to three citation sources in sequence (Semantic Scholar, OpenAlex, NASA ADS) and deduplicates the combined list before writing new rows to `citing_papers`.

```bash
sed -n '1,57p' src/citation_tracker/sources/deduplicator.py
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
    ads_map: dict[str, dict[str, Any]] = {}
    unique: list[dict[str, Any]] = []

    for paper in papers:
        doi = (paper.get("doi") or "").strip().lower()
        ss_id = paper.get("ss_id") or ""
        bibcode = paper.get("ads_bibcode") or ""

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
        elif bibcode:
            if bibcode in ads_map:
                _merge_into(ads_map[bibcode], paper)
                continue
            ads_map[bibcode] = paper
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

The deduplicator works in a single pass. It maintains three identity maps (by DOI, SS ID, ADS bibcode) and a `unique` output list. When it sees a paper it has already encountered, it calls `_merge_into` to copy any fields that are None in the original from the duplicate — so the *first* non-None value for any field wins. Here is a concrete example:

```bash
uv run python3 -c "
from citation_tracker.sources.deduplicator import deduplicate
from_ss  = {\"doi\": \"10.1/x\", \"title\": \"Great Paper\", \"ss_id\": \"ss1\",  \"oa_id\": None,  \"ads_bibcode\": None,       \"year\": 2023}
from_oa  = {\"doi\": \"10.1/x\", \"title\": None,           \"ss_id\": None,   \"oa_id\": \"W99\", \"ads_bibcode\": None,       \"year\": None}
from_ads = {\"doi\": \"10.1/X\", \"title\": None,           \"ss_id\": None,   \"oa_id\": None,  \"ads_bibcode\": \"2023Test\", \"year\": None}
result = deduplicate([from_ss, from_oa, from_ads])
p = result[0]
print(\"Records in: 3  ->  Records out:\", len(result))
print(\"title       :\", p[\"title\"])
print(\"ss_id       :\", p[\"ss_id\"])
print(\"oa_id       :\", p[\"oa_id\"])
print(\"ads_bibcode :\", p[\"ads_bibcode\"])
"
```

```output
Records in: 3  ->  Records out: 1
title       : Great Paper
ss_id       : ss1
oa_id       : W99
ads_bibcode : 2023Test
```

### Stage 2: Fetch

`_fetch_stage()` runs `try_download_citing_paper()` concurrently for all pending citing papers. It tries PDF sources in this order:
1. The direct `pdf_url` from the discovery metadata
2. ADS Link Gateway (when an ADS bibcode is known)
3. Unpaywall (open-access PDF lookup by DOI)
4. CrossRef links
5. arXiv (when the DOI is an arXiv DOI)

```bash
sed -n '64,97p' src/citation_tracker/fetcher.py
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

    # Fallback: try ADS if bibcode is known
    bibcode = paper.get("ads_bibcode")
    if bibcode:
        path = _try_ads(bibcode, pdfs_dir, doi=doi)
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

### Stage 3: Parse

`_parse_stage()` calls `extract_text()` from `parser.py` on every newly downloaded PDF. Under the hood, `pymupdf4llm` converts the PDF to Markdown-formatted text, preserving tables and structure. The result is stored in `citing_papers.extracted_text`.

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

### Stage 4: Analyse

`_analyse_stage()` sends each extracted paper text to the LLM through `analyse_citing_paper()` in `analyser.py`. The LLM receives a prompt containing the tracked paper's metadata and the full citing paper text, and returns a structured JSON object with six fields.

```bash
sed -n '12,37p' src/citation_tracker/analyser.py
```

```output

logger = logging.getLogger(__name__)

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
```

For papers whose extracted text exceeds ~300 KB, a map-reduce strategy is used: the text is split into 80 KB chunks, each chunk is summarised independently by the LLM, and then those summaries are reduced into a single coherent synthesis before the main analysis prompt is run.

```bash
sed -n '93,110p' src/citation_tracker/analyser.py
```

```output
    summaries: list[str] = []
    for i, chunk in enumerate(chunks):
        logger.debug("Summarising chunk %d/%d", i + 1, len(chunks))
        prompt = _CHUNK_SUMMARY_PROMPT.format(chunk=chunk)
        summaries.append(_call_llm_text(prompt, config))
    if len(summaries) == 1:
        return summaries[0]
    combined = "\n\n".join(
        f"Section {i + 1}: {s}" for i, s in enumerate(summaries)
    )
    return _call_llm_text(
        _REDUCE_SUMMARY_PROMPT.format(summaries=combined),
        config,
    )


def analyse_citing_paper(
    tracked_paper: dict[str, Any],
```

The LLM backend is OpenRouter (OpenAI-compatible API). `openrouter.analyse()` sends the prompt and parses the response JSON — handling both raw JSON and JSON wrapped in Markdown code fences.

```bash
cat src/citation_tracker/backends/openrouter.py
```

```output
"""OpenRouter backend for LLM analysis via the OpenAI-compatible API."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from openai import OpenAI

from citation_tracker.config import OpenRouterConfig

logger = logging.getLogger(__name__)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def _parse_json_response(raw: str) -> dict[str, Any]:
    """Parse a JSON dict from a raw LLM response string.

    Handles plain JSON and JSON wrapped in markdown code fences.
    Raises ValueError if no valid JSON dict can be extracted.
    """
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
        if match:
            return json.loads(match.group(1))
        raise ValueError(f"Could not parse JSON from OpenRouter response: {raw[:200]}")


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

    return _parse_json_response(raw)
```

### Stage 5: Report

`_report_stage()` assembles the final Markdown report. First it generates a **Scholarly Synthesis** — a 3-paragraph academic narrative synthesising all citation analyses. Then `build_report()` assembles the full Markdown, with each citing paper in a collapsible `<details>` block.

```bash
sed -n '14,88p' src/citation_tracker/report.py
```

```output
def build_report(
    tracked_paper: sqlite3.Row,
    analyses: list[sqlite3.Row],
    failed_pdfs: list[sqlite3.Row],
    scholarly_synthesis: str | None = None,
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
    if tracked_paper["source_url"]:
        lines.append(f"*Source URL: {tracked_paper['source_url']}*  ")
    lines.append(f"\n*Generated: {now}*\n")

    if scholarly_synthesis:
        lines.append("## Scholarly Synthesis & Impact Assessment\n")
        lines.append(f"{scholarly_synthesis}\n")
        lines.append("---\n")

    if analyses:
        lines.append(f"## Individual Citing Papers ({len(analyses)} analysed)\n")
        for a in analyses:
            lines.append("<details markdown=\"1\">")
            title_str = a["citing_title"] or "Untitled"
            year_str = a["citing_year"] or "N/A"
            lines.append(
                f"<summary><b>{title_str}</b> ({year_str})</summary>\n"
            )
            lines.append(
                f"**Authors:** {_format_authors(a['citing_authors'])}  \n"
                f"**Year:** {a['citing_year'] or 'N/A'}  \n"
                f"**DOI:** {a['citing_doi'] or 'N/A'}\n"
            )
            lines.append(f"\n**Relationship:** `{a['relationship_type'] or 'N/A'}`\n")
            if a["summary"]:
                lines.append(f"**Summary:** {a['summary']}\n")
            if a["new_evidence"]:
                lines.append(f"**New Evidence:** {a['new_evidence']}\n")
            if a["flaws_identified"]:
                lines.append(f"**Flaws Identified:** {a['flaws_identified']}\n")
            if a["assumptions_questioned"]:
                lines.append(
                    f"**Assumptions Questioned:** {a['assumptions_questioned']}\n"
                )
            if a["other_notes"]:
                lines.append(f"**Other Notes:** {a['other_notes']}\n")
            lines.append("\n</details>")
            lines.append("")
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

Each report is saved as `~/.citation-tracker/reports/<paper-id>.md`. The report is also printed to the terminal using `glow` (if available) or `rich`'s Markdown renderer.

## 5. Module Map

A quick reference for where each piece of logic lives:

```bash
grep -rn 'def ' src/citation_tracker/*.py src/citation_tracker/backends/*.py src/citation_tracker/sources/*.py | grep -v '__pycache__' | grep -v 'test_' | sed 's|src/citation_tracker/||' | sort
```

```output
analyser.py:109:def analyse_citing_paper(
analyser.py:143:def generate_scholarly_synthesis(
analyser.py:198:def parse_paper_metadata(text: str, config: Config) -> dict[str, Any]:
analyser.py:68:def _get_backend(config: Config) -> Any:
analyser.py:76:def _call_llm_text(prompt: str, config: Config) -> str:
analyser.py:90:def _map_reduce(text: str, config: Config) -> str:
backends/openrouter.py:19:def _parse_json_response(raw: str) -> dict[str, Any]:
backends/openrouter.py:34:def analyse(
cli.py:129:def list_papers(ctx: click.Context) -> None:
cli.py:169:def pause_paper(ctx: click.Context, paper_id: str | None, doi: str | None) -> None:
cli.py:178:def resume_paper(ctx: click.Context, paper_id: str | None, doi: str | None) -> None:
cli.py:183:def _set_active(
cli.py:213:def remove_paper(ctx: click.Context, paper_id: str | None, doi: str | None) -> None:
cli.py:236:def _select_papers(conn: Any, paper_id: str | None) -> list[Any]:
cli.py:249:def _save_reports(report_sections: list[Any], cfg: Any) -> None:
cli.py:277:def run_pipeline(
cli.py:339:def ingest(ctx: click.Context, pdf_path: str, paper_id: str, doi: str | None) -> None:
cli.py:394:def list_citations(ctx: click.Context, paper_id: str) -> None:
cli.py:433:def status(ctx: click.Context) -> None:
cli.py:458:def show(ctx: click.Context, paper_id: str | None, doi: str | None) -> None:
cli.py:46:def _setup_logging(verbose: bool) -> None:
cli.py:51:def _get_db(cfg: Any) -> Path:
cli.py:57:def _print_markdown(content: str) -> None:
cli.py:73:def main(
cli.py:97:def add_paper(
config.py:19:    def api_key(self) -> str:
config.py:29:    def api_key(self) -> str:
config.py:42:    def db_path(self) -> Path:
config.py:46:    def pdfs_dir(self) -> Path:
config.py:50:    def manual_dir(self) -> Path:
config.py:54:    def reports_dir(self) -> Path:
config.py:58:def load_config(
db.py:118:def get_conn(db_path: Path) -> Generator[sqlite3.Connection, None, None]:
db.py:135:def insert_tracked_paper(conn: sqlite3.Connection, paper: dict[str, Any]) -> str:
db.py:161:def get_tracked_paper_by_doi(conn: sqlite3.Connection, doi: str) -> sqlite3.Row | None:
db.py:167:def get_tracked_paper_by_id(
db.py:17:def _now() -> str:
db.py:175:def list_tracked_papers(
db.py:185:def set_tracked_paper_active(
db.py:194:def delete_tracked_paper(conn: sqlite3.Connection, paper_id: str) -> None:
db.py:208:def upsert_citing_paper(
db.py:21:def _generate_id() -> str:
db.py:259:def get_citing_papers_pending_pdf(
db.py:26:def _serialise_authors(authors: Any) -> str | None:
db.py:269:def get_citing_papers_for_analysis(
db.py:285:def update_citing_paper_pdf(
db.py:298:def update_citing_paper_text(
db.py:307:def list_citing_papers(
db.py:325:def insert_analysis(conn: sqlite3.Connection, analysis: dict[str, Any]) -> str:
db.py:33:def init_db(db_path: Path) -> None:
db.py:358:def list_analyses(
db.py:377:def insert_run(
db.py:393:def finish_run(
db.py:416:def db_summary(conn: sqlite3.Connection) -> dict[str, Any]:
fetcher.py:100:def _try_unpaywall(
fetcher.py:119:def _try_crossref(doi: str, pdfs_dir: Path) -> Path | None:
fetcher.py:140:def _extract_arxiv_id(doi: str | None, paper: dict[str, Any]) -> str | None:
fetcher.py:149:def _try_arxiv(doi: str | None, paper: dict[str, Any], pdfs_dir: Path) -> Path | None:
fetcher.py:159:def _try_ads(bibcode: str, pdfs_dir: Path, doi: str | None = None) -> Path | None:
fetcher.py:170:def scan_manual_dir(manual_dir: Path) -> list[Path]:
fetcher.py:18:def _doi_to_path(doi: str) -> str:
fetcher.py:23:def download_pdf(
fetcher.py:64:def try_download_citing_paper(
parser.py:11:def extract_text(pdf_path: Path) -> str | None:
pipeline.py:102:    def _parse_one(cp_row: Any) -> tuple[str, str | None]:
pipeline.py:119:def _analyse_stage(
pipeline.py:141:    def _analyse_one(cp_row: Any) -> tuple[str, dict[str, Any]]:
pipeline.py:17:def _discover_citations(
pipeline.py:183:def _report_stage(
pipeline.py:219:def process_paper(
pipeline.py:58:def _fetch_stage(
pipeline.py:72:    def _fetch_one(cp_row: Any) -> tuple[str, bool]:
pipeline.py:86:def _parse_stage(
report.py:10:def _format_authors(authors: str | None) -> str:
report.py:14:def build_report(
report.py:91:def render_full_report_html(markdown_content: str) -> str:
resolver.py:101:def _resolve_from_pdf(url: str, cfg: Any) -> dict[str, Any] | None:
resolver.py:14:def _clean_query(text: str, max_len: int = 200) -> str:
resolver.py:151:def _resolve_from_pdf_text(
resolver.py:187:def _resolve_by_url(url: str, cfg: Any) -> dict[str, Any] | None:
resolver.py:21:def _merge_source_results(
resolver.py:231:def resolve_paper(
resolver.py:48:def _resolve_by_doi(doi: str, cfg: Any) -> dict[str, Any] | None:
resolver.py:67:def _resolve_by_ss_id(ss_id: str) -> dict[str, Any] | None:
resolver.py:77:def _extract_doi_from_url(url: str) -> str | None:
resolver.py:83:def _extract_arxiv_id_from_url(url: str) -> str | None:
resolver.py:90:def _resolve_by_arxiv_id(arxiv_id: str, source_url: str) -> dict[str, Any] | None:
sources/adsapi.py:115:def search_paper_by_query(query: str, token: str) -> dict[str, Any] | None:
sources/adsapi.py:17:def _doc_to_dict(doc: dict[str, Any]) -> dict[str, Any]:
sources/adsapi.py:46:def _get(url: str, params: dict[str, Any], token: str) -> Any:
sources/adsapi.py:67:def get_paper_by_doi(doi: str, token: str) -> dict[str, Any] | None:
sources/adsapi.py:87:def get_citations(
sources/deduplicator.py:52:def _merge_into(target: dict[str, Any], source: dict[str, Any]) -> None:
sources/deduplicator.py:8:def deduplicate(papers: list[dict[str, Any]]) -> list[dict[str, Any]]:
sources/openalexapi.py:17:def _work_to_dict(w: dict[str, Any]) -> dict[str, Any]:
sources/openalexapi.py:42:def _get(url: str, params: dict[str, Any]) -> Any:
sources/openalexapi.py:55:def search_paper_by_query(query: str) -> dict[str, Any] | None:
sources/openalexapi.py:71:def get_paper_by_doi(doi: str) -> dict[str, Any] | None:
sources/openalexapi.py:81:def get_paper_by_arxiv(arxiv_id: str) -> dict[str, Any] | None:
sources/openalexapi.py:94:def get_citations(oa_id: str, max_results: int = 500) -> list[dict[str, Any]]:
sources/semantic_scholar.py:17:def _paper_to_dict(p: dict[str, Any]) -> dict[str, Any]:
sources/semantic_scholar.py:33:def _get(url: str, params: dict[str, Any]) -> Any:
sources/semantic_scholar.py:48:def search_paper_by_query(query: str) -> dict[str, Any] | None:
sources/semantic_scholar.py:64:def get_paper_by_id(ss_id: str) -> dict[str, Any] | None:
sources/semantic_scholar.py:74:def get_paper_by_doi(doi: str) -> dict[str, Any] | None:
sources/semantic_scholar.py:84:def get_paper_by_arxiv(arxiv_id: str) -> dict[str, Any] | None:
sources/semantic_scholar.py:94:def get_citations(ss_id: str, limit: int = 500) -> list[dict[str, Any]]:
```

| Module | Responsibility |
|--------|---------------|
| `cli.py` | Click commands, user I/O, ties everything together |
| `config.py` | Config dataclasses, YAML + .env loading |
| `db.py` | SQLite schema, all queries and migrations |
| `resolver.py` | Paper metadata resolution from URL/DOI/SS ID |
| `pipeline.py` | 5-stage pipeline orchestration |
| `fetcher.py` | PDF download with multi-source fallback |
| `parser.py` | PDF → Markdown text extraction |
| `analyser.py` | LLM calls, map-reduce, scholarly synthesis |
| `report.py` | Markdown report assembly |
| `backends/openrouter.py` | OpenRouter API wrapper |
| `sources/semantic_scholar.py` | Semantic Scholar API |
| `sources/openalexapi.py` | OpenAlex API |
| `sources/adsapi.py` | NASA ADS API |
| `sources/deduplicator.py` | Cross-source deduplication |

## 6. Other Commands

Beyond `add` and `run`, the CLI provides:
- `citation-tracker list` — table of all tracked papers
- `citation-tracker citations --id <ID>` — list citing papers and their PDF/analysis status
- `citation-tracker status` — DB summary counts
- `citation-tracker show --id <ID>` — regenerate and display the latest report for a paper
- `citation-tracker ingest <pdf> --id <ID>` — manually add a PDF as a citation
- `citation-tracker pause/resume --id <ID>` — toggle active tracking
- `citation-tracker remove --id <ID>` — delete a paper and all its data

```bash
uv run citation-tracker --help
```

```output
Usage: citation-tracker [OPTIONS] COMMAND [ARGS]...

  Citation Tracker — track and analyse citations of academic papers.

Options:
  --config TEXT  Path to config.yaml
  --env TEXT     Path to .env file
  --verbose
  --help         Show this message and exit.

Commands:
  add        Add a paper to track by URL, DOI, or Semantic Scholar ID.
  citations  List all citing papers for a given tracked paper.
  ingest     Ingest a manually downloaded PDF as a citation for a tracked...
  list       List all tracked papers.
  pause      Pause tracking a paper (keeps all history).
  remove     Remove a paper and all its data.
  resume     Resume tracking a paper.
  run        Run the full discovery → fetch → parse → analyse → report...
  show       Show all analyses for a given tracked paper.
  status     Show DB summary: tracked papers, citation counts, pending...
```
