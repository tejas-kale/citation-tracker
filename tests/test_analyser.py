from citation_tracker import analyser


def test_map_reduce_performs_reduce(monkeypatch):
    calls: list[str] = []

    def fake_call(prompt: str, _config):
        calls.append(prompt)
        if "Section summaries" in prompt:
            return "reduced"
        return "chunk-summary"

    monkeypatch.setattr(analyser, "_call_llm_text", fake_call)
    monkeypatch.setattr(analyser, "_CHUNK_SIZE", 5)
    result = analyser._map_reduce("abcdefghij", config=None)  # type: ignore[arg-type]
    assert result == "reduced"
    assert len(calls) == 3

def test_parse_paper_metadata(monkeypatch):
    def fake_get_backend(_config):
        return lambda _prompt: {"title": "Parsed Title", "authors": "A, B", "year": 2024, "abstract": "Abs"}

    monkeypatch.setattr(analyser, "_get_backend", fake_get_backend)
    
    result = analyser.parse_paper_metadata("Some paper text", config=None) # type: ignore[arg-type]
    assert result["title"] == "Parsed Title"
    assert result["authors"] == "A, B"
    assert result["year"] == 2024
