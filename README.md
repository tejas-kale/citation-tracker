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
# Add by PDF URL (metadata fetched automatically)
citation-tracker add "https://example.com/paper.pdf"

# Add by DOI
citation-tracker add --doi 10.xxxx/xxxxx

# Add by Semantic Scholar ID
citation-tracker add --ss-id 12345678
```

### List tracked papers and citations

```bash
# List all papers you are tracking
citation-tracker list

# List citations for a specific paper
citation-tracker citations --id <ID>
```

### Run the pipeline (discover → fetch → parse → analyse → report)

```bash
# Process all active tracked papers
citation-tracker run

# Process a single paper
citation-tracker run --id <ID>

# Use a specific LLM backend
citation-tracker run --backend claude_code
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

# Show analysis report for a specific paper (renders in terminal with glow/rich)
citation-tracker show --id <ID>
```

Reports are automatically generated as HTML files in `~/.citation-tracker/reports/` after each `run`.

## Configuration

All data is stored in `~/.citation-tracker/` by default.

1. Copy `.env.example` to `.env` and fill in your `OPENROUTER_API_KEY`.
2. Edit `config.yaml` to change the default model or data directory if needed.

## Cron setup

To run the tracker automatically every Saturday at 7:00 AM:

```cron
0 7 * * 6 /Users/yourusername/.local/bin/citation-tracker run >> /Users/yourusername/.citation-tracker/cron.log 2>&1
```

## Project structure

```
citation-tracker/
├── src/
│   └── citation_tracker/
│       ├── cli.py          # Click entry point
│       ├── config.py       # Config loading
│       ├── db.py           # SQLite schema and short UUIDs
│       ├── sources/        # Semantic Scholar + OpenAlex clients
│       ├── fetcher.py      # PDF download (Unpaywall)
│       ├── parser.py       # PDF text extraction (PyMuPDF4LLM)
│       ├── analyser.py     # LLM analysis orchestration
│       ├── backends/       # OpenRouter + Claude Code backends
│       └── report.py       # Markdown & HTML report assembly
├── config.yaml
├── .env.example
└── pyproject.toml
```
