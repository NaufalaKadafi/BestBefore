"""Configuration read from the environment (.env supported)."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

try:  # optional: load .env when python-dotenv is installed
    from dotenv import load_dotenv

    # Explicit path: the default search depends on the caller's cwd/frame and can miss it.
    load_dotenv(ROOT / ".env")
except ImportError:  # pragma: no cover
    pass


class ConfigError(RuntimeError):
    """A required setting is missing."""


@dataclass(frozen=True)
class Settings:
    bifrost_url: str
    virtual_key: str
    model_extract: str
    model_judge: str
    model_judge_2: str
    model_chat: str
    fallbacks: tuple[str, ...]
    tavily_key: str
    tavily_url: str
    max_claims: int
    workers: int
    timeout: int
    port: int

    def require_models(self) -> None:
        missing = [
            name
            for name, value in (("MODEL_EXTRACT", self.model_extract), ("MODEL_JUDGE", self.model_judge))
            if not value or value.startswith("provider/")
        ]
        if missing:
            raise ConfigError(
                "Set " + " and ".join(missing) + " in .env to model IDs your Bifrost can reach. "
                "No defaults are shipped because hard-coded model IDs go stale."
            )

    def require_tavily(self) -> None:
        if not self.tavily_key or self.tavily_key.startswith("tvly-your"):
            raise ConfigError("Set TAVILY_API_KEY in .env.")


def get_settings() -> Settings:
    env = os.environ.get
    return Settings(
        bifrost_url=env("BIFROST_URL", "http://localhost:8080").rstrip("/"),
        virtual_key=env("BIFROST_VIRTUAL_KEY", "").strip(),
        model_extract=env("MODEL_EXTRACT", "").strip(),
        model_judge=env("MODEL_JUDGE", "").strip(),
        model_judge_2=env("MODEL_JUDGE_2", "").strip(),
        model_chat=env("MODEL_CHAT", "").strip(),
        fallbacks=tuple(m.strip() for m in env("MODEL_FALLBACKS", "").split(",") if m.strip()),
        tavily_key=env("TAVILY_API_KEY", "").strip(),
        tavily_url=env("TAVILY_URL", "https://api.tavily.com").rstrip("/"),
        max_claims=int(env("MAX_CLAIMS", "8") or 8),
        workers=int(env("WORKERS", "4") or 4),
        timeout=int(env("HTTP_TIMEOUT", "60") or 60),
        port=int(env("PORT", "8787") or 8787),
    )


def today() -> date:
    """Today's date; BESTBEFORE_TODAY (YYYY-MM-DD) pins it for tests and demos."""
    pinned = os.environ.get("BESTBEFORE_TODAY", "").strip()
    return date.fromisoformat(pinned) if pinned else date.today()


def load_claim_types() -> dict:
    data = json.loads((ROOT / "presets" / "claim_types.json").read_text(encoding="utf-8"))
    return {k: v for k, v in data.items() if not k.startswith("_")}


def load_demo_answers() -> list:
    return json.loads((ROOT / "demo" / "answers.json").read_text(encoding="utf-8"))
