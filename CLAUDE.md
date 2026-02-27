# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

**Install for development:**
```bash
uv tool install --editable .
```

**Run tests:**
```bash
pytest tests/
pytest tests/test_db.py                          # single file
pytest tests/test_db.py::test_init_db -v         # single test
```

**Run the CLI:**
```bash
citation-tracker run                             # process all active tracked papers
citation-tracker run --id <ID|DOI> --workers 8
citation-tracker add <URL|DOI|SS-ID>
citation-tracker show --id <ID>
citation-tracker status
```

## Architecture

The pipeline has 5 sequential stages executed in `cli.py:_process_paper()`:

1. **DISCOVER** — Queries Semantic Scholar, OpenAlex, and NASA ADS in parallel, then deduplicates across sources (`sources/deduplicator.py`).
2. **FETCH** — Downloads PDFs via multiple fallbacks: direct URL → ADS Link Gateway → Unpaywall → CrossRef → arXiv (`fetcher.py`).
3. **PARSE** — Extracts Markdown-formatted text from PDFs using `pymupdf4llm` (`parser.py`).
4. **ANALYSE** — Sends paper text + tracked paper metadata to an LLM for structured analysis (`analyser.py`). Long papers use a map-reduce strategy (80KB chunks → summarize → reduce).
5. **REPORT** — Assembles Markdown and renders HTML saved to `~/.citation-tracker/reports/` (`report.py`).

Fetch, parse, and analyse stages run with `ThreadPoolExecutor` (default 8 workers, `--workers` flag).

## Key Files

| File | Role |
|---|---|
| `src/citation_tracker/cli.py` | All CLI commands and pipeline orchestration |
| `src/citation_tracker/db.py` | SQLite schema, migrations, and all queries |
| `src/citation_tracker/analyser.py` | LLM calls, map-reduce, executive synthesis, metadata extraction |
| `src/citation_tracker/config.py` | Config loading from `config.yaml` + `.env` |
| `src/citation_tracker/fetcher.py` | PDF download with multi-source fallback |
| `src/citation_tracker/sources/` | API clients (Semantic Scholar, OpenAlex, NASA ADS) + deduplicator |
| `src/citation_tracker/backends/openrouter.py` | OpenRouter API wrapper (OpenAI-compatible) |
| `src/citation_tracker/report.py` | Markdown/HTML report assembly |

## Database

SQLite at `~/.citation-tracker/tracker.db`. Four tables:
- `tracked_papers` — papers being monitored
- `citing_papers` — papers that cite tracked papers (includes `pdf_status`, `extracted_text`)
- `analyses` — structured LLM analysis per citing paper
- `runs` — execution history

IDs are 8-character hex UUIDs. Migrations run automatically via `init_db()` at startup. WAL mode is enabled.

## Configuration

`config.yaml` (in project root) controls the LLM model and API key env var names. `.env` holds secrets (see `.env.example`). Required: `OPENROUTER_API_KEY`. Optional: `ADS_DEV_KEY` (enables NASA ADS source).

## Analysis JSON Structure

LLM analyses produce this structure (stored in `analyses` table and returned by `analyser.analyse_citing_paper()`):
```python
{
    "summary": str,
    "relationship_type": "supports|challenges|extends|uses|neutral",
    "new_evidence": str | None,
    "flaws_identified": str | None,
    "assumptions_questioned": str | None,
    "other_notes": str | None,
}
```

## Testing Conventions

Tests use `tmp_path` fixtures for isolated SQLite databases and `monkeypatch` to mock API calls and LLM responses. Each test creates a fresh DB via `init_db()`.
