from citation_tracker.analyser import _map_reduce
from citation_tracker.config import Config


def test_map_reduce_uses_reduce_step(mocker, monkeypatch):
    monkeypatch.setattr("citation_tracker.analyser._CHUNK_SIZE", 3)
    llm_call = mocker.patch(
        "citation_tracker.analyser._call_llm_text",
        side_effect=["chunk summary 1", "chunk summary 2", "final combined summary"],
    )

    text = "abcdef"
    result = _map_reduce(text, Config())

    assert result == "final combined summary"
    assert llm_call.call_count == 3
    assert "Section 1" in llm_call.call_args_list[-1].args[0]
