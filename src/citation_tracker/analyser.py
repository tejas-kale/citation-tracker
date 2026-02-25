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
You are combining section summaries from a long academic paper.

Section summaries:
{summaries}

Produce a coherent synthesis (8-12 sentences) that preserves key nuance:
- main argument and method
- strongest evidence/results
- caveats or limitations
- points most relevant to how this paper may engage with another work

Return only the synthesis text, no preamble.\
"""


def _get_backend(config: Config) -> Any:
    if config.backend == "openrouter":
        from citation_tracker.backends import openrouter

        return lambda prompt: openrouter.analyse(prompt, config.openrouter)
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
    return _call_llm_text(
        _REDUCE_SUMMARY_PROMPT.format(
            summaries="\n\n".join(f"Section {i + 1}: {s}" for i, s in enumerate(summaries))
        ),
        config,
    )


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


def generate_executive_synthesis(
    tracked_paper: dict[str, Any],
    analyses: list[dict[str, Any]],
    config: Config
) -> str:
    """Generate a high-level academic synthesis of all citation engagement."""
    if not analyses:
        return "No citations available for synthesis."

    backend_fn = _get_backend(config)
    
    # Prepare a condensed version of all analyses for the prompt
    summaries = []
    for a in analyses:
        summaries.append(
            f"Title: {a.get('citing_title')}\n"
            f"Relationship: {a.get('relationship_type')}\n"
            f"Summary: {a.get('summary')}\n"
            f"New Evidence: {a.get('new_evidence')}\n"
            f"Challenges/Flaws: {a.get('flaws_identified') or a.get('assumptions_questioned')}\n"
        )
    
    all_summaries = "\n---\n".join(summaries)

    prompt = f"""\
You are a senior academic advisor and research lead. Your task is to provide a sophisticated scholarly synthesis of how the research community is engaging with the following paper.

TRACKED PAPER:
Title: {tracked_paper.get('title')}
Abstract: {tracked_paper.get('abstract')}

CITING ANALYSES:
{all_summaries}

INSTRUCTIONS:
Write a "Scholarly Synthesis & Impact Assessment" for a tenured professor. 
STRICT CONSTRAINTS:
1. Maximum 3 paragraphs.
2. Each paragraph must be exactly 4-5 lines long.
3. Use a rigorous, dense academic tone.
4. Do NOT include a title or heading in your response.
5. Do NOT list individual papers; synthesize the collective findings into a narrative.

FOCUS ON:
- Field trajectory and thematic clusters.
- Critical discourse (confirming vs. challenging).
- High-level 'state of the art' impact.

Return raw Markdown.
"""
    # Use the text-only caller to get a raw narrative
    return _call_llm_text(prompt, config)


def parse_paper_metadata(text: str, config: Config) -> dict[str, Any]:
    """Extract paper metadata (title, authors, year, abstract, DOI) from PDF text using LLM."""
    backend_fn = _get_backend(config)

    prompt = f"""\
Extract the metadata for this academic paper from its text.
Return a JSON object with these fields: title, authors, year, abstract, doi.
If a field is unknown, use null.
The 'authors' field should be a comma-separated string.
For 'doi', look for a string starting with '10.' (e.g., 10.1051/0004-6361/202450011).

TEXT:
{text[:4000]}

Return only valid JSON. No preamble.
"""
    return backend_fn(prompt)
