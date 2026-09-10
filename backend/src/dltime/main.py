from __future__ import annotations

from datetime import datetime, timezone
from math import isfinite
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from dltime.api.schemas import (
    AnalyticsOverviewResponse,
    DurationTrendResponse,
    ExecutionHistoryItem,
    ExecutionHistoryResponse,
    HistoricalFeaturesResponse,
    LoaderIntelligenceItem,
    LoaderIntelligenceResponse,
    ModelInformationResponse,
    PredictionHistoryItem,
    PredictionHistoryResponse,
    StageAnalyticsItem,
    StageAnalyticsResponse,
    TotalExecutionPredictionRequest,
    TotalExecutionPredictionResponse,
)
from dltime.config.settings import get_settings
from dltime.data.feedback_store import SQLiteFeedbackStore
from dltime.features.online_historical_features import (
    OnlineHistoricalFeatureProvider,
)
from dltime.model.total_execution_model import (
    TotalExecutionModelError,
    load_total_execution_model,
)
from dltime.services.analytics_service import (
    AnalyticsService,
    AnalyticsServiceError,
)
from dltime.services.manual_configuration_service import (
    ManualConfigurationError,
    ManualLoaderConfigurationService,
)
from dltime.services.prediction_service import (
    TotalExecutionPredictionService,
)


settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Backend API for predicting DataZap Data Loader execution times.",
)


# ---------------------------------------------------------------------------
# Frontend development CORS
# ---------------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Shared service builders
# ---------------------------------------------------------------------------


def _get_feedback_database_path() -> Path:
    """Return the configured SQLite feedback database path."""

    return settings.model_artifact_path.parent / "feedback.db"


def _build_feedback_store() -> SQLiteFeedbackStore:
    """Construct the feedback persistence service."""

    return SQLiteFeedbackStore(
        _get_feedback_database_path()
    )


def _build_prediction_service() -> TotalExecutionPredictionService:
    """Construct the prediction service from application configuration."""

    feedback_store = _build_feedback_store()

    historical_provider = OnlineHistoricalFeatureProvider(
        feedback_store
    )

    return TotalExecutionPredictionService(
        historical_provider=historical_provider,
        artifact_path=settings.model_artifact_path,
    )


def _build_configuration_service() -> ManualLoaderConfigurationService:
    """Construct the manual loader configuration resolver."""

    return ManualLoaderConfigurationService()


def _build_analytics_service() -> AnalyticsService:
    """Construct the read-only analytics service."""

    return AnalyticsService(
        project_root=settings.model_artifact_path.parents[2],
        feedback_store=_build_feedback_store(),
    )


def _safe_api_float(value: object) -> float | None:
    """
    Convert a numeric value into a JSON-safe float.

    Pandas and NumPy calculations can produce NaN or infinity for missing
    historical statistics. Those values must be represented as null at the
    public API boundary.
    """

    if value is None:
        return None

    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return None

    if not isfinite(numeric_value):
        return None

    return numeric_value


def _build_historical_features_response(
    historical_features: dict[str, object],
) -> HistoricalFeaturesResponse:
    """
    Map internal historical feature names to the public API contract.

    The trained model intentionally uses the canonical internal feature
    name `seconds_since_loader_previous`. The API exposes the clearer
    response field `seconds_since_loader_previous_execution`.
    """

    return HistoricalFeaturesResponse(
        global_prior_execution_count=int(
            historical_features.get(
                "global_prior_execution_count",
                0,
            )
            or 0
        ),
        global_prior_mean_duration_sec=_safe_api_float(
            historical_features.get(
                "global_prior_mean_duration_sec"
            )
        ),
        global_prior_median_duration_sec=_safe_api_float(
            historical_features.get(
                "global_prior_median_duration_sec"
            )
        ),
        global_prior_std_duration_sec=_safe_api_float(
            historical_features.get(
                "global_prior_std_duration_sec"
            )
        ),
        loader_prior_execution_count=int(
            historical_features.get(
                "loader_prior_execution_count",
                0,
            )
            or 0
        ),
        loader_prior_mean_duration_sec=_safe_api_float(
            historical_features.get(
                "loader_prior_mean_duration_sec"
            )
        ),
        loader_prior_median_duration_sec=_safe_api_float(
            historical_features.get(
                "loader_prior_median_duration_sec"
            )
        ),
        loader_prior_std_duration_sec=_safe_api_float(
            historical_features.get(
                "loader_prior_std_duration_sec"
            )
        ),
        loader_previous_duration_sec=_safe_api_float(
            historical_features.get(
                "loader_previous_duration_sec"
            )
        ),
        seconds_since_loader_previous_execution=_safe_api_float(
            historical_features.get(
                "seconds_since_loader_previous"
            )
        ),
    )


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


@app.get("/health")
def health_check() -> dict[str, str]:
    return {
        "status": "ok",
        "service": "data-loader-execution-time-prediction",
        "environment": settings.environment,
    }


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------


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

        prediction_timestamp = request.prediction_timestamp

        if prediction_timestamp is None:
            prediction_timestamp = datetime.now(timezone.utc)

        prediction = prediction_service.predict_total(
            loader_name=request.loader_name,
            sprint=request.sprint,
            loader_connection_name=request.ldr_connection_name,
            dataset_name=request.datasetname,
            prediction_timestamp=prediction_timestamp,
        )

        return TotalExecutionPredictionResponse(
            predicted_total_seconds=prediction.predicted_total_seconds,
            prediction_timestamp=prediction.prediction_timestamp,
            loader_name=prediction.loader_name,
            sprint=prediction.sprint,
            model_version=prediction.model_version,
            prediction_source=prediction.prediction_source,
            historical_features=_build_historical_features_response(
                prediction.historical_features
            ),
            configuration_features=prediction.configuration_features,
            cold_start_confidence=(
                None
                if prediction.cold_start_confidence is None
                else {
                    "level": prediction.cold_start_confidence.level,
                    "score": prediction.cold_start_confidence.score,
                    "reason": prediction.cold_start_confidence.reason,
                }
            ),
            similar_loaders=list(prediction.similar_loaders),
        )

    except ManualConfigurationError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc


# ---------------------------------------------------------------------------
# Analytics - overview
# ---------------------------------------------------------------------------


@app.get(
    "/analytics/overview",
    response_model=AnalyticsOverviewResponse,
)
def analytics_overview() -> AnalyticsOverviewResponse:
    """Return high-level execution analytics."""

    try:
        overview = _build_analytics_service().get_overview()

        return AnalyticsOverviewResponse(
            total_executions=overview.total_executions,
            successful_executions=overview.successful_executions,
            stopped_executions=overview.stopped_executions,
            other_executions=overview.other_executions,
            valid_training_targets=overview.valid_training_targets,
            unique_loaders=overview.unique_loaders,
            total_records=overview.total_records,
            average_duration_seconds=overview.average_duration_seconds,
            median_duration_seconds=overview.median_duration_seconds,
            p90_duration_seconds=overview.p90_duration_seconds,
            maximum_duration_seconds=overview.maximum_duration_seconds,
            earliest_execution=overview.earliest_execution,
            latest_execution=overview.latest_execution,
        )

    except AnalyticsServiceError as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc


# ---------------------------------------------------------------------------
# Analytics - execution history
# ---------------------------------------------------------------------------


@app.get(
    "/analytics/executions",
    response_model=ExecutionHistoryResponse,
)
def analytics_executions(
    offset: int = Query(
        default=0,
        ge=0,
        description="Number of records to skip.",
    ),
    limit: int = Query(
        default=50,
        ge=1,
        le=200,
        description="Maximum number of records to return.",
    ),
    loader_name: str | None = Query(
        default=None,
        description="Optional exact loader-name filter.",
    ),
    status: str | None = Query(
        default=None,
        description="Optional execution-status filter.",
    ),
) -> ExecutionHistoryResponse:
    """Return paginated execution history."""

    try:
        records, total = _build_analytics_service().get_execution_history(
            offset=offset,
            limit=limit,
            loader_name=loader_name,
            status=status,
        )

        items = [
            ExecutionHistoryItem(
                execution_id=record.execution_id,
                loader_name=record.loader_name,
                status=record.status,
                sprint=record.sprint,
                start_time=record.start_time,
                end_time=record.end_time,
                duration_seconds=record.duration_seconds,
                dataset_count=record.dataset_count,
                total_records=record.total_records,
                success_records=record.success_records,
                error_records=record.error_records,
                prevalidation_duration_seconds=(
                    record.prevalidation_duration_seconds
                ),
                transformation_duration_seconds=(
                    record.transformation_duration_seconds
                ),
                staging_duration_minutes=(
                    record.staging_duration_minutes
                ),
                data_loading_duration_seconds=(
                    record.data_loading_duration_seconds
                ),
                datamart_approval_duration_seconds=(
                    record.datamart_approval_duration_seconds
                ),
                dataloading_approval_duration_seconds=(
                    record.dataloading_approval_duration_seconds
                ),
            )
            for record in records
        ]

        return ExecutionHistoryResponse(
            items=items,
            total=total,
            offset=offset,
            limit=limit,
        )

    except AnalyticsServiceError as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc


# ---------------------------------------------------------------------------
# Analytics - duration trend
# ---------------------------------------------------------------------------


@app.get(
    "/analytics/duration-trend",
    response_model=DurationTrendResponse,
)
def analytics_duration_trend() -> DurationTrendResponse:
    """Return chronological execution-duration statistics."""

    try:
        records = _build_analytics_service().get_duration_trend()

        return DurationTrendResponse(
            items=[
                {
                    "execution_date": record.execution_date,
                    "execution_count": record.execution_count,
                    "average_duration_seconds": (
                        record.average_duration_seconds
                    ),
                    "median_duration_seconds": (
                        record.median_duration_seconds
                    ),
                }
                for record in records
            ]
        )

    except AnalyticsServiceError as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc


# ---------------------------------------------------------------------------
# Analytics - loader intelligence
# ---------------------------------------------------------------------------


@app.get(
    "/analytics/loaders",
    response_model=LoaderIntelligenceResponse,
)
def analytics_loaders() -> LoaderIntelligenceResponse:
    """Return loader-level execution statistics."""

    try:
        records = _build_analytics_service().get_loader_intelligence()

        return LoaderIntelligenceResponse(
            items=[
                LoaderIntelligenceItem(
                    loader_name=record.loader_name,
                    execution_count=record.execution_count,
                    successful_execution_count=(
                        record.successful_execution_count
                    ),
                    average_duration_seconds=(
                        record.average_duration_seconds
                    ),
                    median_duration_seconds=(
                        record.median_duration_seconds
                    ),
                    p90_duration_seconds=(
                        record.p90_duration_seconds
                    ),
                    maximum_duration_seconds=(
                        record.maximum_duration_seconds
                    ),
                    total_records=record.total_records,
                    latest_execution=record.latest_execution,
                )
                for record in records
            ]
        )

    except AnalyticsServiceError as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc


# ---------------------------------------------------------------------------
# Analytics - stage statistics
# ---------------------------------------------------------------------------


@app.get(
    "/analytics/stages",
    response_model=StageAnalyticsResponse,
)
def analytics_stages() -> StageAnalyticsResponse:
    """Return duration statistics for available execution stages."""

    try:
        records = _build_analytics_service().get_stage_analytics()

        return StageAnalyticsResponse(
            items=[
                StageAnalyticsItem(
                    stage_name=record.stage_name,
                    unit=record.unit,
                    observations=record.observations,
                    average_duration=record.average_duration,
                    median_duration=record.median_duration,
                    p90_duration=record.p90_duration,
                    maximum_duration=record.maximum_duration,
                )
                for record in records
            ]
        )

    except AnalyticsServiceError as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc


# ---------------------------------------------------------------------------
# Analytics - model information
# ---------------------------------------------------------------------------


@app.get(
    "/analytics/model",
    response_model=ModelInformationResponse,
)
def analytics_model() -> ModelInformationResponse:
    """Return metadata from the currently loaded production model."""

    try:
        artifact = load_total_execution_model(
            settings.model_artifact_path
        )

        pipeline = artifact["model"]
        model_version = str(artifact["model_version"])
        target_column = str(artifact["target_column"])
        target_transform = str(artifact["target_transform"])

        estimator = pipeline.named_steps.get("model")

        if estimator is None:
            raise TotalExecutionModelError(
                "Persisted model pipeline does not contain the model step."
            )

        return ModelInformationResponse(
            model_version=model_version,
            estimator_type=type(estimator).__name__,
            target_column=target_column,
            target_transform=target_transform,
            training_rows=0,
            training_loaders=0,
            training_sprints=0,
            n_estimators=int(
                getattr(
                    estimator,
                    "n_estimators",
                    0,
                )
            ),
            min_samples_leaf=int(
                getattr(
                    estimator,
                    "min_samples_leaf",
                    0,
                )
            ),
            max_features=getattr(
                estimator,
                "max_features",
                None,
            ),
            random_state=int(
                getattr(
                    estimator,
                    "random_state",
                    0,
                )
            ),
            artifact_format="joblib",
        )

    except TotalExecutionModelError as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc


# ---------------------------------------------------------------------------
# Analytics - prediction history
# ---------------------------------------------------------------------------


@app.get(
    "/analytics/predictions",
    response_model=PredictionHistoryResponse,
)
def analytics_predictions(
    offset: int = Query(
        default=0,
        ge=0,
        description="Number of prediction records to skip.",
    ),
    limit: int = Query(
        default=50,
        ge=1,
        le=200,
        description="Maximum number of prediction records to return.",
    ),
) -> PredictionHistoryResponse:
    """Return persisted prediction/feedback records."""

    try:
        records = _build_analytics_service().get_feedback_records()

        total = len(records)
        page = records[offset : offset + limit]

        items = [
            PredictionHistoryItem(
                execution_id=record.execution_id,
                loader_name=record.loader_name,
                sprint=record.sprint,
                prediction_timestamp=_parse_feedback_datetime(
                    record.prediction_timestamp
                ),
                predicted_total_seconds=(
                    record.predicted_total_seconds
                ),
                actual_total_seconds=record.actual_total_seconds,
                prediction_error_seconds=(
                    record.prediction_error_seconds
                ),
                absolute_error_seconds=(
                    record.absolute_error_seconds
                ),
                relative_error=record.relative_error,
                execution_status=record.execution_status,
                used_prediction_source=(
                    record.used_prediction_source
                ),
                reliability_score=record.reliability_score,
                model_version=record.model_version,
                training_eligible=record.training_eligible,
                recorded_at=_parse_feedback_datetime(
                    record.recorded_at
                ),
            )
            for record in page
        ]

        return PredictionHistoryResponse(
            items=items,
            total=total,
            offset=offset,
            limit=limit,
        )

    except AnalyticsServiceError as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc


def _parse_feedback_datetime(
    value: str | None,
) -> datetime | None:
    """Parse an optional persisted ISO timestamp for API responses."""

    if value is None:
        return None

    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)

    return parsed.astimezone(timezone.utc)