"""Simple local PWA server for fresh-run OpenRouter analyses."""

from __future__ import annotations

import json
import os
import tempfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from citation_tracker.config import load_config
from citation_tracker.db import get_conn, get_tracked_paper_by_id, init_db, insert_tracked_paper
from citation_tracker.report import build_full_report, render_full_report_html

_INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <meta name="theme-color" content="#1f2937">
  <link rel="manifest" href="/manifest.json">
  <title>Citation Tracker PWA</title>
  <style>
    body{font-family:system-ui,-apple-system,sans-serif;max-width:900px;margin:2rem auto;padding:0 1rem}
    input,button{padding:.6rem;margin:.25rem 0;width:100%} button{cursor:pointer}
    #status{white-space:pre-wrap;color:#374151} #report{margin-top:1rem;border:1px solid #e5e7eb;border-radius:8px;padding:1rem}
  </style>
</head>
<body>
  <h1>Citation Tracker</h1>
  <p>Fresh run each time (no persistence).</p>
  <form id="run-form">
    <input id="doi" placeholder="Tracked paper DOI (e.g., 10.1038/nature12373)" required>
    <input id="openrouterKey" placeholder="OpenRouter API key" required>
    <button type="submit">Run analysis</button>
  </form>
  <div id="status"></div>
  <div id="report"></div>
  <script>
    if ("serviceWorker" in navigator) navigator.serviceWorker.register("/sw.js");
    const form = document.getElementById("run-form");
    const status = document.getElementById("status");
    const report = document.getElementById("report");
    form.addEventListener("submit", async (e) => {
      e.preventDefault();
      status.textContent = "Running pipeline...";
      report.innerHTML = "";
      const resp = await fetch("/run", {
        method: "POST",
        headers: {"content-type":"application/json"},
        body: JSON.stringify({
          doi: document.getElementById("doi").value,
          openrouterKey: document.getElementById("openrouterKey").value
        })
      });
      const data = await resp.json();
      if (!resp.ok) {
        status.textContent = data.error || "Failed";
        return;
      }
      status.textContent = `New papers: ${data.new_papers}, Analysed: ${data.analysed}, Errors: ${data.errors.length}`;
      report.innerHTML = data.report_html || "<em>No report generated.</em>";
    });
  </script>
</body>
</html>
"""

_MANIFEST = {
    "name": "Citation Tracker",
    "short_name": "CitationTracker",
    "start_url": "/",
    "display": "standalone",
    "background_color": "#ffffff",
    "theme_color": "#1f2937",
    "icons": [],
}

_SW_JS = """self.addEventListener('install', () => self.skipWaiting());"""


def _run_fresh_pipeline(doi: str, openrouter_key: str) -> dict[str, object]:
    from citation_tracker.cli import _process_paper, _resolve_paper

    with tempfile.TemporaryDirectory(prefix="citation-tracker-pwa-") as tmpdir:
        cfg = load_config()
        cfg.backend = "openrouter"
        cfg.data_dir = Path(tmpdir)
        os.environ[cfg.openrouter.api_key_env] = openrouter_key
        init_db(cfg.db_path)

        paper = _resolve_paper(url=None, doi=doi, ss_id=None)
        if paper is None:
            raise ValueError(f"Could not resolve paper metadata for DOI {doi!r}")

        with get_conn(cfg.db_path) as conn:
            tracked_id = insert_tracked_paper(conn, paper)
            tracked = get_tracked_paper_by_id(conn, tracked_id)
            if tracked is None:
                raise ValueError("Tracked paper insert failed")

        new_papers, analysed, errors, section = _process_paper(dict(tracked), cfg, cfg.db_path)
        report_html = ""
        if section:
            report_md = build_full_report([section])
            report_html = render_full_report_html(report_md)

        return {
            "new_papers": new_papers,
            "analysed": analysed,
            "errors": errors,
            "report_html": report_html,
        }


def serve(host: str = "127.0.0.1", port: int = 8765) -> None:
    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, body: str, content_type: str) -> None:
            body_bytes = body.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body_bytes)))
            self.end_headers()
            self.wfile.write(body_bytes)

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/":
                self._send(200, _INDEX_HTML, "text/html; charset=utf-8")
            elif self.path == "/manifest.json":
                self._send(200, json.dumps(_MANIFEST), "application/manifest+json")
            elif self.path == "/sw.js":
                self._send(200, _SW_JS, "application/javascript")
            else:
                self._send(404, "Not found", "text/plain; charset=utf-8")

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/run":
                self._send(404, "Not found", "text/plain; charset=utf-8")
                return
            try:
                content_length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(content_length).decode("utf-8") or "{}")
                doi = str(payload.get("doi") or "").strip()
                key = str(payload.get("openrouterKey") or "").strip()
                if not doi or not key:
                    raise ValueError("Both DOI and OpenRouter key are required")
                result = _run_fresh_pipeline(doi, key)
                self._send(200, json.dumps(result), "application/json")
            except Exception as exc:  # noqa: BLE001
                self._send(400, json.dumps({"error": str(exc)}), "application/json")

        def log_message(self, _format: str, *args: object) -> None:
            return

    server = ThreadingHTTPServer((host, port), Handler)
    print(f"Citation Tracker PWA running at http://{host}:{port}")
    server.serve_forever()

