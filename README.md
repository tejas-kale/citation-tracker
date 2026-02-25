# Citation Tracker

Track citations of academic papers and analyse how they engage with your work using LLMs.

## Installation

```bash
# Install globally with uv
uv tool install .

# Or during development
uv tool install --editable .
```

## Usage

### Add a paper to track

```bash
# Add by URL (handles arXiv, DOIs, and peeks into PDF for ambiguous filenames)
citation-tracker add "https://ericnrobertson.github.io/files/jmp.pdf"

# Add by DOI
citation-tracker add --doi 10.xxxx/xxxxx

# Add by Semantic Scholar ID
citation-tracker add --ss-id 12345678
```

### List tracked papers and citations

```bash
# List all papers you are tracking (shows IDs and DOI/URL)
citation-tracker list

# List citations for a specific paper
citation-tracker citations --id <ID>
```

### Run the pipeline (discover → fetch → parse → analyse → report)

```bash
# Process all active tracked papers
citation-tracker run

# Process a single paper by numeric ID or DOI
citation-tracker run --id <ID|DOI>

# Specify number of concurrent workers
citation-tracker run --workers 12
```

### Manage tracked papers

```bash
# Pause tracking (keeps history)
citation-tracker pause --id <ID>

# Resume tracking
citation-tracker resume --id <ID>

# Remove paper and all its data
citation-tracker remove --id <ID>
```

### Ingest a manually downloaded PDF

```bash
# Link a PDF to a tracked paper as a citation
citation-tracker ingest path/to/paper.pdf --id <ID>
```

### Inspect and Reports

```bash
# Show current status summary
citation-tracker status

# Show analysis report for a specific paper
citation-tracker show --id <ID>
```

Reports are automatically generated as HTML files in `~/.citation-tracker/reports/` after each `run`.
Reports feature an **interactive Mermaid.js influence tree** that categorizes citations (EXTENDS, CHALLENGES, USES, etc.) in a vertical, directory-style layout.

### Simple PWA (fresh run, no persistence)

A lightweight PWA is available at `src/pwa/index.html`. It features a progress bar and structured result view for quick paper lookups via OpenRouter.
This app is deployed to GitHub Pages via `.github/workflows/pages.yml`.

## Configuration

All data is stored in `~/.citation-tracker/` by default.

1. Copy `.env.example` to `.env` and fill in:
   - `OPENROUTER_API_KEY`: Required for LLM analysis.
   - `ADS_DEV_KEY`: Optional, for NASA ADS discovery (Astronomy/Physics).
2. Edit `config.yaml` to change the default model, worker count, or data directory.

## Cron setup

To run the tracker automatically every Saturday at 7:00 AM:

```cron
0 7 * * 6 /Users/yourusername/.local/bin/citation-tracker run >> /Users/yourusername/.citation-tracker/cron.log 2>&1
```

## Project structure

```
citation-tracker/
├── src/
│   ├── citation_tracker/
│   │   ├── cli.py          # Click entry point & ThreadPoolExecutor
│   │   ├── config.py       # Config loading & ADS/OpenRouter settings
│   │   ├── db.py           # SQLite schema, short UUIDs & migrations
│   │   ├── sources/        # API clients: SS, OpenAlex, NASA ADS, Deduplicator
│   │   ├── fetcher.py      # PDF download (Unpaywall, ADS Gateway, arXiv)
│   │   ├── parser.py       # PDF text extraction (PyMuPDF4LLM)
│   │   ├── analyser.py     # LLM analysis & Map-Reduce orchestration
│   │   └── report.py       # Markdown & HTML report assembly (Mermaid.js)
│   └── pwa/                # Standalone lookup tool
├── config.yaml
├── .env.example
└── pyproject.toml
```
