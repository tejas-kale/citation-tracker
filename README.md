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

### List tracked papers

```bash
citation-tracker list
```

### Run the pipeline (discover → fetch → parse → analyse → report)

```bash
# Process all active tracked papers
citation-tracker run

# Process a single paper
citation-tracker run --doi 10.xxxx/xxxxx

# Use a specific LLM backend
citation-tracker run --backend claude_code
```

### Manage tracked papers

```bash
# Pause tracking (keeps history)
citation-tracker pause --doi 10.xxxx/xxxxx

# Resume tracking
citation-tracker resume --doi 10.xxxx/xxxxx

# Remove paper and all its data
citation-tracker remove --doi 10.xxxx/xxxxx
```

### Ingest a manually downloaded PDF

```bash
citation-tracker ingest data/manual/paper.pdf --doi 10.xxxx/xxxxx
```

### Inspect the database

```bash
citation-tracker status
citation-tracker show --doi 10.xxxx/xxxxx
```

## Configuration

Copy `.env.example` to `.env` and fill in your API keys.
Edit `config.yaml` to set your backend and email settings.

## Cron setup

```cron
# Every Monday at 08:00
0 8 * * 1 citation-tracker run >> ~/.local/share/citation-tracker/cron.log 2>&1
```

## Project structure

```
citation-tracker/
├── src/
│   └── citation_tracker/
│       ├── cli.py          # Click entry point
│       ├── config.py       # Config loading
│       ├── db.py           # SQLite schema and queries
│       ├── sources/        # Semantic Scholar + OpenAlex clients
│       ├── fetcher.py      # PDF download
│       ├── parser.py       # PDF text extraction
│       ├── analyser.py     # LLM analysis orchestration
│       ├── backends/       # OpenRouter + Claude Code backends
│       ├── report.py       # Report assembly
│       └── mailer.py       # Email via Resend
├── data/
│   ├── pdfs/               # Downloaded PDFs
│   └── manual/             # Drop manually downloaded PDFs here
├── config.yaml
├── .env.example
└── pyproject.toml
```