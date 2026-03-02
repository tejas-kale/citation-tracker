"""Configuration loading from .env and YAML files."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from dotenv import load_dotenv


@dataclass
class OpenRouterConfig:
    model: str = "minimax/minimax-m2.5"
    api_key_env: str = "OPENROUTER_API_KEY"

    @property
    def api_key(self) -> str:
        key = os.environ.get(self.api_key_env, "")
        return key


@dataclass
class ADSConfig:
    api_key_env: str = "ADS_DEV_KEY"

    @property
    def api_key(self) -> str:
        return os.environ.get(self.api_key_env, "")


@dataclass
class Config:
    backend: str = "openrouter"
    openrouter: OpenRouterConfig = field(default_factory=OpenRouterConfig)
    ads: ADSConfig = field(default_factory=ADSConfig)
    data_dir: Path = field(default_factory=lambda: Path.home() / ".citation-tracker")
    unpaywall_email: str = ""

    @property
    def db_path(self) -> Path:
        return self.data_dir / "tracker.db"

    @property
    def pdfs_dir(self) -> Path:
        return self.data_dir / "pdfs"

    @property
    def manual_dir(self) -> Path:
        return self.data_dir / "manual"

    @property
    def reports_dir(self) -> Path:
        return self.data_dir / "reports"


_DATA_DIR = Path.home() / ".citation-tracker"


def load_config(
    config_path: Path | None = None, env_path: Path | None = None
) -> Config:
    """Load configuration from .env and optional YAML config file."""
    # Load .env — check explicit path, CWD, then ~/.citation-tracker/
    env_candidates = [env_path, Path(".env"), _DATA_DIR / ".env"]
    for candidate in env_candidates:
        if candidate and candidate.exists():
            load_dotenv(candidate)
            break

    config = Config()

    # Load YAML config — check explicit path, CWD, then ~/.citation-tracker/
    yaml_candidates = [config_path, Path("config.yaml"), _DATA_DIR / "config.yaml"]
    yaml_file = next((p for p in yaml_candidates if p and p.exists()), None)
    if yaml_file is None:
        return config
    with yaml_file.open() as f:
        raw = yaml.safe_load(f) or {}

    if "backend" in raw:
        config.backend = raw["backend"]

    if "openrouter" in raw:
        or_raw = raw["openrouter"]
        config.openrouter = OpenRouterConfig(
            model=or_raw.get("model", config.openrouter.model),
            api_key_env=or_raw.get("api_key_env", config.openrouter.api_key_env),
        )

    if "ads" in raw:
        ads_raw = raw["ads"]
        config.ads = ADSConfig(
            api_key_env=ads_raw.get("api_key_env", config.ads.api_key_env),
        )

    if "data_dir" in raw:
        config.data_dir = Path(raw["data_dir"]).expanduser()

    if "unpaywall_email" in raw:
        config.unpaywall_email = raw["unpaywall_email"]

    return config
