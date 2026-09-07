from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


# settings.py
#   -> config/
#   -> dltime/
#   -> src/
#   -> backend/
#   -> repository root
PROJECT_ROOT = Path(__file__).resolve().parents[4]


class Settings(BaseSettings):
    app_name: str = "Data Loader Execution Time Prediction API"
    app_version: str = "0.1.0"
    environment: str = "development"
    debug: bool = False

    model_artifact_path: Path = PROJECT_ROOT / "backend" / "artifacts" / (
        "total_execution_model.joblib"
    )

    model_config = SettingsConfigDict(
        env_prefix="DLTIME_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
