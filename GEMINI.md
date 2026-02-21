# Citation Tracker - Project Context

`citation-tracker` is a Python-based CLI tool designed to track citations of academic papers and analyze how citing works engage with the tracked research using Large Language Models (LLMs).

## Project Overview

*   **Purpose**: Automates the pipeline of discovering new citations, fetching PDFs, extracting text, and generating LLM-powered analyses of the relationship between citing and cited papers.
*   **Main Technologies**:
    *   **Python 3.12+**: Core language.
    *   **Click**: CLI framework.
    *   **SQLite**: Local data persistence (using WAL mode for concurrency).
    *   **LLM Backends**: OpenRouter (OpenAI-compatible) and Claude Code.
    *   **PDF Extraction**: `pymupdf4llm` for converting PDFs to Markdown-rich text.
    *   **API Sources**: Semantic Scholar and OpenAlex for citation discovery.
    *   **Reporting**: Generates Markdown and HTML reports saved locally.
    *   **Package Management**: `uv` is the preferred tool for installation and development.

## Architecture

*   **`src/citation_tracker/cli.py`**: Entry point for all commands (`add`, `run`, `list`, etc.).
*   **`src/citation_tracker/db.py`**: Manages the SQLite schema and all database operations. Tables include `tracked_papers`, `citing_papers`, `analyses`, and `runs`.
*   **`src/citation_tracker/sources/`**: Contains API clients for external academic databases.
*   **`src/citation_tracker/fetcher.py`**: Handles PDF downloads, utilizing Unpaywall for open-access discovery.
*   **`src/citation_tracker/analyser.py`**: Orchestrates LLM interactions to analyze the extracted text of citing papers.
*   **`src/citation_tracker/report.py`**: Assembly and HTML rendering of analysis reports, which are saved to `~/.citation-tracker/reports`.

## Building and Running

### Installation
```bash
# Install with uv (recommended)
uv tool install .

# Development mode
uv tool install --editable .
```

### Core Commands
*   **Add a paper**: `citation-tracker add --doi 10.xxxx/xxxxx`
*   **List papers**: `citation-tracker list` (Displays numeric IDs for each tracked paper)
*   **List citations**: `citation-tracker citations --id <ID>` (Shows citing papers and status)
*   **Process citations**: `citation-tracker run` (Discovers, fetches, and analyses new citations)
*   **Show analyses**: `citation-tracker show --id <ID>` (Shows the analysis report for a paper)
*   **Ingest manual PDF**: `citation-tracker ingest data/manual/paper.pdf --id <ID>` (Links a PDF to a tracked paper)

## Development Conventions

*   **Configuration**: Managed via `config.yaml` and `.env`. Use `src/citation_tracker/config.py` to add new configuration fields.
*   **Database**: SQLite with `PRAGMA foreign_keys=ON`. Connections are handled via the `get_conn` context manager in `db.py`.
*   **Error Handling**: Uses `tenacity` for retrying API calls and robust error logging via `rich`.
*   **Type Hinting**: Strong adherence to Python type hints (`from __future__ import annotations`).
*   **Data Storage**: By default, all data (DB, PDFs, and HTML reports) is stored in `~/.citation-tracker/`, configurable via `config.yaml`.

## Key Files
*   `pyproject.toml`: Dependency definitions and script entry points.
*   `config.yaml`: Main application configuration (backend choice, API settings).
*   `.env.example`: Template for required environment variables (API keys).
*   `src/citation_tracker/db.py`: The "source of truth" for data structures.
