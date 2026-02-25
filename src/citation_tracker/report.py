"""Markdown report assembly."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from typing import Any


def _format_authors(authors: str | None) -> str:
    return authors or "Unknown authors"


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
    if tracked_paper["source_url"]:
        lines.append(f"*Source URL: {tracked_paper['source_url']}*  ")
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


def build_full_report(
    sections: list[tuple[sqlite3.Row, list[sqlite3.Row], list[sqlite3.Row]]],
) -> str:
    """Build a combined report for multiple tracked papers."""
    if not sections:
        return "# Citation Tracker Report\n\nNo data to report.\n"
    parts = [build_report(tp, analyses, failed) for tp, analyses, failed in sections]
    return "\n\n---\n\n".join(parts)


def render_full_report_html(markdown_content: str) -> str:
    """Render the Markdown report as a full HTML page with basic styling."""
    try:
        import markdown
    except ImportError:
        return f"<html><body><pre>{markdown_content}</pre></body></html>"

    html_body = markdown.markdown(markdown_content, extensions=["extra", "toc"])

    # Mermaid JS requires the graph to be inside a div with class "mermaid"
    # We replace the markdown-generated <pre><code class="language-mermaid">...</code></pre> 
    # with the simple <div class="mermaid">...</div>
    import re
    html_body = re.sub(
        r'<pre><code class="language-mermaid">(.*?)</code></pre>',
        r'<div class="mermaid">\1</div>',
        html_body,
        flags=re.DOTALL
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Citation Tracker Report</title>
    <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
    <script>
        mermaid.initialize({{ startOnLoad: true }});
    </script>
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
