"""Configuration loading via tejas-config."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from tejas_config import load_config as _load_tejas_config
from tejas_config.secrets import get_secret


@dataclass
class OpenRouterConfig:
    model: str = "minimax/minimax-m2.5"
    api_key_env: str = "OPENROUTER_API_KEY"
    _api_key: str = ""

    @property
    def api_key(self) -> str:
        if self._api_key:
            return self._api_key
        return os.environ.get(self.api_key_env, "")


@dataclass
class ADSConfig:
    api_key_env: str = "ADS_DEV_KEY"
    _api_key: str = ""

    @property
    def api_key(self) -> str:
        if self._api_key:
            return self._api_key
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


def load_config(**_kwargs) -> Config:
    """Load configuration via tejas-config."""
    cfg = _load_tejas_config("citation_tracker")
    ct = cfg.citation_tracker
    return Config(
        openrouter=OpenRouterConfig(
            model=ct.model or cfg.openrouter.model,
            api_key_env="",
            _api_key=cfg.openrouter.api_key,
        ),
        ads=ADSConfig(
            api_key_env="",
            _api_key=get_secret("ads_dev_key") or "",
        ),
        data_dir=ct.data_dir,
        unpaywall_email=ct.unpaywall_email,
    )
