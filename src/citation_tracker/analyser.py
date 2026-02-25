"""LLM analysis orchestration."""

from __future__ import annotations

import logging
from typing import Any

from citation_tracker.config import Config

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

Return only valid JSON. No preamble.\
"""

# Rough token estimate: ~4 chars per token. Keep well below 100k tokens.
_MAX_TEXT_CHARS = 300_000
_CHUNK_SIZE = 80_000
_CHUNK_SUMMARY_PROMPT = """\
Summarise the following section of an academic paper in 3–5 sentences:

{chunk}

Return only the summary text, no preamble.\
"""
_REDUCE_SUMMARY_PROMPT = """\
You are combining partial summaries of one academic paper.
Synthesize them into a cohesive summary that preserves key nuance, methods, findings, and caveats in 6-10 sentences.

Partial summaries:
{summaries}

Return only the final summary text, no preamble.\
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
    reduce_prompt = _REDUCE_SUMMARY_PROMPT.format(
        summaries="\n\n".join(f"Section {i + 1}:\n{s}" for i, s in enumerate(summaries))
    )
    return _call_llm_text(reduce_prompt, config)


def analyse_citing_paper(
    tracked_paper: dict[str, Any],
    citing_paper: dict[str, Any],
    extracted_text: str,
    config: Config,
) -> dict[str, Any]:
    """
    Analyse how *citing_paper* engages with *tracked_paper*.

    Uses map-reduce if the extracted text exceeds the context limit.
    Returns parsed JSON dict from the LLM.
    """
    backend_fn = _get_backend(config)

    text = extracted_text
    if len(text) > _MAX_TEXT_CHARS:
        logger.info(
            "Text too long (%d chars), using map-reduce for '%s'",
            len(text),
            citing_paper.get("title", ""),
        )
        text = _map_reduce(extracted_text, config)

    prompt = ANALYSIS_PROMPT_TEMPLATE.format(
        tracked_title=tracked_paper.get("title", ""),
        tracked_authors=tracked_paper.get("authors", ""),
        tracked_abstract=tracked_paper.get("abstract", ""),
        citing_title=citing_paper.get("title", ""),
        extracted_text=text,
    )

    return backend_fn(prompt)
