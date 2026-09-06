from fastapi import FastAPI

from dltime.config.settings import get_settings


settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Backend API for predicting DataZap Data Loader execution times.",
)


@app.get("/health")
def health_check() -> dict[str, str]:
    return {
        "status": "ok",
        "service": "data-loader-execution-time-prediction",
        "environment": settings.environment,
    }