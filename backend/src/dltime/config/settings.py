from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Data Loader Execution Time Prediction API"
    app_version: str = "0.1.0"
    environment: str = "development"
    debug: bool = False

    model_config = SettingsConfigDict(
        env_prefix="DLTIME_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
