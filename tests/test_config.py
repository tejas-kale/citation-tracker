from pathlib import Path
from unittest.mock import MagicMock, patch
from citation_tracker.config import load_config, Config, OpenRouterConfig


def test_config_defaults():
    config = Config()
    assert config.backend == "openrouter"
    assert config.data_dir == Path.home() / ".citation-tracker"
    assert config.db_path == Path.home() / ".citation-tracker" / "tracker.db"


def test_load_config():
    mock_cfg = MagicMock()
    mock_cfg.citation_tracker.model = "gpt-4"
    mock_cfg.citation_tracker.data_dir = Path.home() / "custom-dir"
    mock_cfg.citation_tracker.unpaywall_email = ""
    mock_cfg.openrouter.model = "default-model"
    mock_cfg.openrouter.api_key = "test-key"

    with patch("citation_tracker.config._load_tejas_config", return_value=mock_cfg), \
         patch("citation_tracker.config.get_secret", return_value=None):
        config = load_config()

    assert config.openrouter.model == "gpt-4"
    assert config.openrouter.api_key == "test-key"
    assert config.data_dir == Path.home() / "custom-dir"


def test_load_config_falls_back_to_openrouter_model():
    mock_cfg = MagicMock()
    mock_cfg.citation_tracker.model = None
    mock_cfg.citation_tracker.data_dir = Path.home() / ".citation-tracker"
    mock_cfg.citation_tracker.unpaywall_email = ""
    mock_cfg.openrouter.model = "fallback-model"
    mock_cfg.openrouter.api_key = ""

    with patch("citation_tracker.config._load_tejas_config", return_value=mock_cfg), \
         patch("citation_tracker.config.get_secret", return_value=None):
        config = load_config()

    assert config.openrouter.model == "fallback-model"


def test_openrouter_config_api_key_env_fallback(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "env-test-key")
    orc = OpenRouterConfig()
    assert orc.api_key == "env-test-key"
