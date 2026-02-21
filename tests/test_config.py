import os
import yaml
from pathlib import Path
from citation_tracker.config import load_config, Config

def test_config_defaults():
    config = Config()
    assert config.backend == "openrouter"
    assert config.data_dir == Path.home() / ".citation-tracker"
    assert config.db_path == Path.home() / ".citation-tracker" / "tracker.db"

def test_load_config_yaml(tmp_path):
    yaml_file = tmp_path / "config.yaml"
    raw = {
        "backend": "claude_code",
        "data_dir": "~/custom-dir",
        "openrouter": {
            "model": "gpt-4",
            "api_key_env": "CUSTOM_KEY"
        }
    }
    with yaml_file.open("w") as f:
        yaml.dump(raw, f)
    
    config = load_config(config_path=yaml_file)
    assert config.backend == "claude_code"
    assert config.data_dir == Path.home() / "custom-dir"
    assert config.openrouter.model == "gpt-4"
    assert config.openrouter.api_key_env == "CUSTOM_KEY"

def test_load_config_env(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    with env_file.open("w") as f:
        f.write("OPENROUTER_API_KEY=test-key\n")
    
    config = load_config(env_path=env_file)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    assert config.openrouter.api_key == "test-key"
