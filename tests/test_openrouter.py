"""Tests for the OpenRouter backend."""

import json
import pytest
from citation_tracker.backends.openrouter import _parse_json_response


def test_parse_plain_json():
    raw = '{"summary": "Great paper.", "relationship_type": "supports"}'
    result = _parse_json_response(raw)
    assert result["summary"] == "Great paper."
    assert result["relationship_type"] == "supports"


def test_parse_json_in_code_fence():
    raw = '```json\n{"summary": "Fenced.", "relationship_type": "extends"}\n```'
    result = _parse_json_response(raw)
    assert result["summary"] == "Fenced."


def test_parse_json_in_plain_fence():
    raw = '```\n{"summary": "No lang tag.", "relationship_type": "uses"}\n```'
    result = _parse_json_response(raw)
    assert result["summary"] == "No lang tag."


def test_parse_invalid_json_raises():
    with pytest.raises(ValueError, match="Could not parse JSON"):
        _parse_json_response("This is not JSON at all.")
