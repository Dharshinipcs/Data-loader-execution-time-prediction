from fastapi import FastAPI

app = FastAPI(
    title="Data Loader Execution Time Prediction API",
    version="0.1.0",
    description="Backend API for predicting DataZap Data Loader execution times.",
)


@app.get("/health")
def health_check() -> dict[str, str]:
    return {
        "status": "ok",
        "service": "data-loader-execution-time-prediction",
    }