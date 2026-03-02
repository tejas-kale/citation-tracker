# Citation Tracker

Discover and analyse academic papers that cite your work. Runs a 5-stage pipeline
(discover → fetch → parse → analyse → report) for each tracked paper, producing
LLM-generated analyses and a scholarly synthesis saved as a Markdown report.

## Quick Start

```bash
# 1. Install
uv tool install .

# 2. Set your API key
echo "OPENROUTER_API_KEY=sk-or-..." >> ~/.citation-tracker/.env

# 3. Add a paper to track
citation-tracker add --doi 10.1145/1234567.1234568

# 4. Run the pipeline
citation-tracker run
```

## Installation

```bash
# Install globally with uv
uv tool install .

# During development (editable install)
uv tool install --editable .
```

**Requirements:** Python 3.11+, [uv](https://docs.astral.sh/uv/).

## Configuration

All data is stored in `~/.citation-tracker/` by default.

### API Keys (`.env`)

Place a `.env` file in your project directory or `~/.citation-tracker/`. See `.env.example`:

```env
OPENROUTER_API_KEY=sk-or-...   # Required: LLM analysis via OpenRouter
ADS_DEV_KEY=...                # Optional: NASA ADS discovery (astronomy/physics papers)
```

### Config File (`config.yaml`)

Place `config.yaml` in your project directory or `~/.citation-tracker/`. Full schema:

```yaml
# LLM backend (only openrouter is supported)
backend: openrouter

openrouter:
  model: minimax/minimax-m2.5     # Any model on openrouter.ai
  api_key_env: OPENROUTER_API_KEY # Env var name to read the key from

ads:
  api_key_env: ADS_DEV_KEY        # Env var name to read the NASA ADS key from

# Where to store the database, PDFs, and reports
data_dir: "~/.citation-tracker"

# Your email for the Unpaywall API (required by their usage policy)
unpaywall_email: "you@example.com"
```

Config and `.env` are searched in this order: explicit `--config`/`--env` flag →
current directory → `~/.citation-tracker/`.

## Commands

All commands support `--help`. Global options (`--config`, `--env`, `--verbose`) must
come before the subcommand name:

```bash
citation-tracker --verbose run
citation-tracker --config /path/to/config.yaml run --id a1b2c3d4
```

### `add` — Track a new paper

```bash
# By arXiv or any URL (metadata resolved via Semantic Scholar)
citation-tracker add "https://arxiv.org/abs/2401.00001"

# By DOI
citation-tracker add --doi 10.1145/1234567.1234568

# By Semantic Scholar ID
citation-tracker add --ss-id 204e3073870fae3d05bcbc2f6a8e263d21195671
```

Duplicate papers (same DOI) are rejected with a message showing the existing ID.

### `list` — Show all tracked papers

```bash
citation-tracker list
```

Prints ID, title, DOI/URL, year, active status, and date added. Use the ID from this
output in `--id` arguments to other commands.

### `run` — Execute the pipeline

```bash
# Process all active tracked papers
citation-tracker run

# Process a single paper
citation-tracker run --id a1b2c3d4
citation-tracker run --id 10.1145/1234567.1234568

# Use more workers for faster parallel fetch/parse/analyse
citation-tracker run --workers 16
```

### `show` — Print report in terminal

```bash
citation-tracker show --id a1b2c3d4
citation-tracker show --doi 10.1145/1234567.1234568
```

Generates a scholarly synthesis on the fly and renders the report using `glow`
(if installed) or `rich`. Does not save to disk — use `run` for saved reports.

### `citations` — List citing papers

```bash
citation-tracker citations --id a1b2c3d4
```

Shows each citing paper's title, DOI, PDF status (`pending` / `downloaded` / `failed` /
`manual`), and whether an analysis has been completed.

### `status` — Database summary

```bash
citation-tracker status
```

Prints counts for: active tracked papers, total citing papers, PDFs pending download,
and total analyses completed.

### `ingest` — Manually add a PDF

```bash
# Register a PDF as a citation for a tracked paper
citation-tracker ingest ~/Downloads/smith2024.pdf --id a1b2c3d4

# With DOI for richer metadata
citation-tracker ingest paper.pdf --id a1b2c3d4 --doi 10.1145/1234567.1234568
```

The PDF is copied to `~/.citation-tracker/pdfs/`, text is extracted immediately,
and the paper is queued for analysis on the next `run`.

### `pause` / `resume` — Suspend tracking

```bash
citation-tracker pause --id a1b2c3d4    # Skip in future runs, keep all data
citation-tracker resume --id a1b2c3d4   # Include again in runs
```

### `remove` — Delete a paper and all its data

```bash
citation-tracker remove --id a1b2c3d4
```

Irreversible. Prompts for confirmation. Deletes all citations, analyses, and PDFs for
the paper. Use `pause` instead to stop processing without losing data.

## Pipeline

The pipeline has 5 sequential stages executed per tracked paper:

| Stage | Description |
|---|---|
| **DISCOVER** | Queries Semantic Scholar, OpenAlex, and NASA ADS in parallel; deduplicates results across sources. |
| **FETCH** | Downloads PDFs via: direct URL → ADS Link Gateway → Unpaywall → CrossRef → arXiv. |
| **PARSE** | Extracts Markdown-formatted text from PDFs using `pymupdf4llm`. |
| **ANALYSE** | Sends paper text + tracked paper metadata to the LLM. Papers over 80 KB use map-reduce (chunk → summarise → reduce). |
| **REPORT** | Assembles Markdown with a scholarly synthesis and saves to `~/.citation-tracker/reports/<id>.md`. |

Fetch, parse, and analyse run concurrently across `--workers` threads (default: 8).
Previously analysed papers are not re-analysed unless the DB record is cleared.

## Reports

HTML reports are saved to `~/.citation-tracker/reports/<id>.md` after each `run`.
Each report contains:
- A **Scholarly Synthesis** — an LLM-generated academic summary of how the citing
  literature engages with the tracked paper.
- Per-citation analyses with relationship type, new evidence, identified flaws, and notes.
- A list of papers where PDF download failed.

## Analysis JSON Structure

Each citing paper produces a structured analysis stored in the `analyses` table:

```python
{
    "summary": str,                  # 1-2 sentence summary of the citing paper's use
    "relationship_type": str,        # "supports" | "challenges" | "extends" | "uses" | "neutral"
    "new_evidence": str | None,      # New evidence or data the citing paper contributes
    "flaws_identified": str | None,  # Methodological or theoretical flaws noted
    "assumptions_questioned": str | None,  # Assumptions of the tracked paper challenged
    "other_notes": str | None,       # Anything else noteworthy
}
```

## Data Directory Layout

```
~/.citation-tracker/
├── tracker.db          # SQLite database (WAL mode)
├── .env                # API keys (optional location)
├── config.yaml         # Config overrides (optional location)
├── pdfs/               # Downloaded PDFs, organised by DOI slug
│   └── 10.1145-.../
│       └── paper.pdf
└── reports/            # Saved Markdown reports
    └── a1b2c3d4.md
```

## Database Schema

SQLite at `~/.citation-tracker/tracker.db`. Four tables:

| Table | Description |
|---|---|
| `tracked_papers` | Papers being monitored. Key columns: `id` (8-char hex), `doi`, `title`, `active`. |
| `citing_papers` | Papers that cite a tracked paper. Key columns: `tracked_paper_id`, `pdf_status`, `extracted_text`. |
| `analyses` | Structured LLM analysis per citing paper. Linked to `citing_papers.id`. |
| `runs` | Execution history. Columns: `triggered_by`, `new_papers`, `analysed`, `errors`. |

IDs are 8-character hex strings (e.g., `a1b2c3d4`). Migrations run automatically at
startup via `init_db()`. WAL mode is enabled.

## Project Structure

```
citation-tracker/
├── src/citation_tracker/
│   ├── cli.py              # Click CLI entry point and pipeline orchestration
│   ├── db.py               # SQLite schema, migrations, and all query functions
│   ├── config.py           # Config dataclasses and YAML/.env loading
│   ├── pipeline.py         # Per-paper pipeline runner (ThreadPoolExecutor)
│   ├── resolver.py         # Paper metadata resolution (URL / DOI / SS-ID)
│   ├── analyser.py         # LLM calls, map-reduce strategy, scholarly synthesis
│   ├── fetcher.py          # PDF download with multi-source fallback chain
│   ├── parser.py           # PDF text extraction via pymupdf4llm
│   ├── report.py           # Markdown/HTML report assembly
│   ├── backends/
│   │   └── openrouter.py   # OpenRouter API wrapper (OpenAI-compatible)
│   └── sources/
│       ├── semantic_scholar.py
│       ├── openalex.py
│       ├── ads.py
│       └── deduplicator.py
├── tests/
├── config.yaml
├── .env.example
└── pyproject.toml
```

## Cron Setup

To run the tracker automatically every Saturday at 7:00 AM:

```cron
0 7 * * 6 /Users/yourusername/.local/bin/citation-tracker run >> /Users/yourusername/.citation-tracker/cron.log 2>&1
```

Find your binary path with `which citation-tracker`.

## Development

```bash
# Install in editable mode
uv tool install --editable .

# Run tests
pytest tests/
pytest tests/test_db.py                          # single file
pytest tests/test_db.py::test_init_db -v         # single test
```

Tests use `tmp_path` fixtures for isolated SQLite databases and `monkeypatch` to mock
API calls and LLM responses.
