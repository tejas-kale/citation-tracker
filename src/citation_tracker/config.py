"""Configuration loading from .env and YAML files."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from dotenv import load_dotenv


@dataclass
class OpenRouterConfig:
    model: str = "qwen/qwen3-235b-a22b"
    api_key_env: str = "OPENROUTER_API_KEY"

    @property
    def api_key(self) -> str:
        key = os.environ.get(self.api_key_env, "")
        return key


@dataclass
class ClaudeCodeConfig:
    flags: str = ""


@dataclass
class Config:
    backend: str = "openrouter"
    openrouter: OpenRouterConfig = field(default_factory=OpenRouterConfig)
    claude_code: ClaudeCodeConfig = field(default_factory=ClaudeCodeConfig)
    data_dir: Path = field(default_factory=lambda: Path("data"))
    resend_api_key_env: str = "RESEND_API_KEY"
    email_from: str = ""
    email_to: str = ""
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
    def resend_api_key(self) -> str:
        return os.environ.get(self.resend_api_key_env, "")


def load_config(config_path: Path | None = None, env_path: Path | None = None) -> Config:
    """Load configuration from .env and optional YAML config file."""
    # Load .env file
    env_file = env_path or Path(".env")
    if env_file.exists():
        load_dotenv(env_file)

    config = Config()

    # Load YAML config if provided or default config.yaml exists
    yaml_file = config_path or Path("config.yaml")
    if yaml_file.exists():
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

        if "claude_code" in raw:
            cc_raw = raw["claude_code"]
            config.claude_code = ClaudeCodeConfig(
                flags=cc_raw.get("flags", config.claude_code.flags),
            )

        if "data_dir" in raw:
            config.data_dir = Path(raw["data_dir"])

        if "resend_api_key_env" in raw:
            config.resend_api_key_env = raw["resend_api_key_env"]

        if "email_from" in raw:
            config.email_from = raw["email_from"]

        if "email_to" in raw:
            config.email_to = raw["email_to"]

        if "unpaywall_email" in raw:
            config.unpaywall_email = raw["unpaywall_email"]

    return config
