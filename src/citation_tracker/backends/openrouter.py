"""OpenRouter backend for LLM analysis via the OpenAI-compatible API."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from openai import OpenAI

from citation_tracker.config import OpenRouterConfig

logger = logging.getLogger(__name__)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def _parse_json_response(raw: str) -> dict[str, Any]:
    """Parse a JSON dict from a raw LLM response string.

    Handles plain JSON and JSON wrapped in markdown code fences.
    Raises ValueError if no valid JSON dict can be extracted.
    """
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
        if match:
            return json.loads(match.group(1))
        raise ValueError(f"Could not parse JSON from OpenRouter response: {raw[:200]}")


def analyse(
    prompt: str,
    config: OpenRouterConfig,
) -> dict[str, Any]:
    """
    Send *prompt* to OpenRouter and return the parsed JSON response dict.

    Raises ValueError if the response cannot be parsed as JSON.
    """
    client = OpenAI(
        api_key=config.api_key,
        base_url=OPENROUTER_BASE_URL,
    )

    response = client.chat.completions.create(
        model=config.model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
    )

    raw = response.choices[0].message.content or ""
    logger.debug("OpenRouter raw response: %s", raw[:200])

    return _parse_json_response(raw)
