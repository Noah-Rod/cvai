from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration, read from `JASS_*` environment variables (or `.env`)."""

    model_config = SettingsConfigDict(env_prefix="JASS_", env_file=".env", extra="ignore")

    model_backend: Literal["auto", "yolo", "dummy"] = "auto"
    model_path: Path = Path("models/best.pt")
    device: str = "cpu"
    conf_threshold: float = 0.25
    iou_threshold: float = 0.45
    img_size: int = 640
    max_upload_mb: int = 10
    api_url: str = "http://localhost:8000"


@lru_cache
def get_settings() -> Settings:
    return Settings()
