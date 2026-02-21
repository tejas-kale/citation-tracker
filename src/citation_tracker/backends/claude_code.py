"""Claude Code backend: subprocess wrapper for `claude -p`."""

from __future__ import annotations

import json
import logging
import re
import subprocess
from typing import Any

from citation_tracker.config import ClaudeCodeConfig

logger = logging.getLogger(__name__)


def analyse(prompt: str, config: ClaudeCodeConfig) -> dict[str, Any]:
    """
    Run `claude -p` with *prompt* as stdin and return the parsed JSON response.

    Raises subprocess.CalledProcessError if claude exits non-zero.
    Raises ValueError if the response cannot be parsed as JSON.
    """
    cmd = ["claude", "-p"]
    if config.flags:
        cmd.extend(config.flags.split())

    result = subprocess.run(
        cmd,
        input=prompt,
        capture_output=True,
        text=True,
        check=True,
    )

    raw = result.stdout.strip()
    logger.debug("claude -p raw response: %s", raw[:200])

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
        if match:
            return json.loads(match.group(1))
        raise ValueError(f"Could not parse JSON from claude response: {raw[:200]}")
