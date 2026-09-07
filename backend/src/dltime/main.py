from __future__ import annotations

from fastapi import FastAPI, HTTPException

from dltime.api.schemas import (
    TotalExecutionPredictionRequest,
    TotalExecutionPredictionResponse,
)
from dltime.config.settings import get_settings
from dltime.data.feedback_store import SQLiteFeedbackStore
from dltime.features.online_historical_features import (
    OnlineHistoricalFeatureProvider,
)
from dltime.services.manual_configuration_service import (
    ManualConfigurationError,
    ManualLoaderConfigurationService,
)
from dltime.services.prediction_service import (
    PredictionServiceError,
    TotalExecutionPredictionService,
)


settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Backend API for predicting DataZap Data Loader execution times.",
)


def _build_prediction_service() -> TotalExecutionPredictionService:
    """Construct the prediction service from application configuration."""
    feedback_db_path = (
        settings.model_artifact_path.parent / "feedback.db"
    )

    feedback_store = SQLiteFeedbackStore(feedback_db_path)
    historical_provider = OnlineHistoricalFeatureProvider(feedback_store)

    return TotalExecutionPredictionService(
        historical_provider=historical_provider,
        artifact_path=settings.model_artifact_path,
    )


def _build_configuration_service() -> ManualLoaderConfigurationService:
    """Construct the manual loader configuration resolver."""
    return ManualLoaderConfigurationService()


@app.get("/health")
def health_check() -> dict[str, str]:
    return {
        "status": "ok",
        "service": "data-loader-execution-time-prediction",
        "environment": settings.environment,
    }


@app.post(
    "/predict/total",
    response_model=TotalExecutionPredictionResponse,
)
def predict_total(
    request: TotalExecutionPredictionRequest,
) -> TotalExecutionPredictionResponse:
    """
    Validate manual loader configuration and predict total execution time.

    Configuration resolution happens before prediction so invalid or
    unknown loader configurations cannot reach the prediction engine.
    """
    try:
        configuration_service = _build_configuration_service()

        configuration_service.resolve(
            loader_name=request.loader_name,
            sprint=request.sprint,
            ldr_connection_name=request.ldr_connection_name,
            datasetname=request.datasetname,
            prevalidation_enabled=request.prevalidation_enabled,
            transformation_enabled=request.transformation_enabled,
        )

        prediction_service = _build_prediction_service()

        prediction = prediction_service.predict_total(
            loader_name=request.loader_name,
            sprint=request.sprint,
            prediction_timestamp=request.prediction_timestamp,
        )

        return TotalExecutionPredictionResponse(
            predicted_total_seconds=prediction.predicted_total_seconds,
            prediction_timestamp=prediction.prediction_timestamp,
            loader_name=prediction.loader_name,
            sprint=prediction.sprint,
            model_version=prediction.model_version,
            historical_features=prediction.historical_features,
        )

    except ManualConfigurationError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except PredictionServiceError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc