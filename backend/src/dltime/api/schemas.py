from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class HistoricalFeaturesResponse(BaseModel):
    """Historical features used by the prediction engine."""

    model_config = ConfigDict(extra="forbid")

    global_prior_execution_count: int
    global_prior_mean_duration_sec: float | None
    global_prior_median_duration_sec: float | None
    global_prior_std_duration_sec: float | None
    loader_prior_execution_count: int
    loader_prior_mean_duration_sec: float | None
    loader_prior_median_duration_sec: float | None
    loader_prior_std_duration_sec: float | None
    loader_previous_duration_sec: float | None
    seconds_since_loader_previous_execution: float | None


class ManualLoaderConfigurationRequest(BaseModel):
    """
    Validated manual Data Loader configuration input.

    The four identity fields form the configuration business key.
    Stage flags are optional because the backend can resolve actual
    stage availability from the authoritative configuration source.

    Configuration-derived numeric features and data-quality fields are
    intentionally excluded from user input.
    """

    model_config = ConfigDict(extra="forbid")

    loader_name: str = Field(
        min_length=1,
        description="DataZap Data Loader name.",
    )
    sprint: str = Field(
        min_length=1,
        description="Sprint or execution grouping identifier.",
    )
    ldr_connection_name: str = Field(
        min_length=1,
        description="DataZap loader connection name.",
    )
    datasetname: str = Field(
        min_length=1,
        description="Configured dataset name.",
    )
    prevalidation_enabled: bool | None = Field(
        default=None,
        description=(
            "Optional prevalidation stage flag. If supplied, it must "
            "match the resolved Data Loader configuration."
        ),
    )
    transformation_enabled: bool | None = Field(
        default=None,
        description=(
            "Optional transformation stage flag. If supplied, it must "
            "match the resolved Data Loader configuration."
        ),
    )


class TotalExecutionPredictionRequest(ManualLoaderConfigurationRequest):
    """
    Input required to request a pre-execution total-time prediction.

    The manual loader configuration fields are required for resolving
    the authoritative pre-execution configuration. The prediction
    timestamp is optional and defaults to the backend's current UTC time.
    """

    prediction_timestamp: datetime | None = Field(
        default=None,
        description=(
            "Timestamp at which the prediction is requested. "
            "If omitted, the backend uses the current UTC time."
        ),
    )


class ColdStartConfidenceResponse(BaseModel):
    """Evidence-based confidence indicator for an unseen loader."""

    model_config = ConfigDict(extra="forbid")

    level: Literal["LOW", "MEDIUM", "HIGH"]
    score: float = Field(
        ge=0,
        le=1,
        description=(
            "Evidence score derived from similar-loader retrieval. "
            "This is not a calibrated probability."
        ),
    )
    reason: str


class SimilarLoaderResponse(BaseModel):
    """Historical loader profile used as cold-start supporting evidence."""

    model_config = ConfigDict(extra="forbid")

    loader_name: str
    sprint: str
    connection: str
    similarity_score: float = Field(
        ge=0,
        le=1,
    )


class TotalExecutionPredictionResponse(BaseModel):
    """Validated prediction returned by the backend."""

    model_config = ConfigDict(extra="forbid")

    predicted_total_seconds: float = Field(
        gt=0,
        description="Predicted total execution duration in seconds.",
    )
    prediction_timestamp: datetime
    loader_name: str
    sprint: str
    model_version: str
    prediction_source: str
    historical_features: HistoricalFeaturesResponse
    configuration_features: dict[str, float | str]
    cold_start_confidence: ColdStartConfidenceResponse | None = None
    similar_loaders: list[SimilarLoaderResponse] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Frontend analytics response schemas
# ---------------------------------------------------------------------------


class AnalyticsOverviewResponse(BaseModel):
    """High-level execution analytics for the frontend dashboard."""

    model_config = ConfigDict(extra="forbid")

    total_executions: int
    successful_executions: int
    stopped_executions: int
    other_executions: int
    valid_training_targets: int
    unique_loaders: int
    total_records: int
    average_duration_seconds: float | None
    median_duration_seconds: float | None
    p90_duration_seconds: float | None
    maximum_duration_seconds: float | None
    earliest_execution: datetime | None
    latest_execution: datetime | None


class ExecutionHistoryItem(BaseModel):
    """Single reconstructed execution for the history analytics view."""

    model_config = ConfigDict(extra="forbid")

    execution_id: str
    loader_name: str | None
    status: str | None
    sprint: str | None
    start_time: datetime | None
    end_time: datetime | None
    duration_seconds: float | None
    dataset_count: int
    total_records: int
    success_records: int
    error_records: int
    prevalidation_duration_seconds: float | None
    transformation_duration_seconds: float | None
    staging_duration_minutes: float | None
    data_loading_duration_seconds: float | None
    datamart_approval_duration_seconds: float | None
    dataloading_approval_duration_seconds: float | None


class ExecutionHistoryResponse(BaseModel):
    """Paginated execution history response."""

    model_config = ConfigDict(extra="forbid")

    items: list[ExecutionHistoryItem]
    total: int
    offset: int
    limit: int


class DurationTrendPoint(BaseModel):
    """Aggregated duration observation used by analytics charts."""

    model_config = ConfigDict(extra="forbid")

    execution_date: str
    execution_count: int
    average_duration_seconds: float | None
    median_duration_seconds: float | None


class DurationTrendResponse(BaseModel):
    """Chronological duration trend."""

    model_config = ConfigDict(extra="forbid")

    items: list[DurationTrendPoint]


class LoaderIntelligenceItem(BaseModel):
    """Loader-level historical execution statistics."""

    model_config = ConfigDict(extra="forbid")

    loader_name: str
    execution_count: int
    successful_execution_count: int
    average_duration_seconds: float | None
    median_duration_seconds: float | None
    p90_duration_seconds: float | None
    maximum_duration_seconds: float | None
    total_records: int
    latest_execution: datetime | None


class LoaderIntelligenceResponse(BaseModel):
    """Loader intelligence dataset for the frontend."""

    model_config = ConfigDict(extra="forbid")

    items: list[LoaderIntelligenceItem]


StageUnit = Literal["seconds", "minutes"]


class StageAnalyticsItem(BaseModel):
    """Duration statistics for one execution stage."""

    model_config = ConfigDict(extra="forbid")

    stage_name: str
    unit: StageUnit
    observations: int
    average_duration: float | None
    median_duration: float | None
    p90_duration: float | None
    maximum_duration: float | None


class StageAnalyticsResponse(BaseModel):
    """Stage-duration analytics."""

    model_config = ConfigDict(extra="forbid")

    items: list[StageAnalyticsItem]


class ModelInformationResponse(BaseModel):
    """Metadata describing the currently loaded prediction model."""

    model_config = ConfigDict(extra="forbid")

    model_version: str
    estimator_type: str
    target_column: str
    target_transform: str
    training_rows: int
    training_loaders: int
    training_sprints: int
    n_estimators: int
    min_samples_leaf: int
    max_features: float | str | None
    random_state: int
    artifact_format: str


class PredictionHistoryItem(BaseModel):
    """Persisted prediction/feedback record."""

    model_config = ConfigDict(extra="forbid")

    execution_id: str
    loader_name: str | None
    sprint: str | None
    prediction_timestamp: datetime | None
    predicted_total_seconds: float | None
    actual_total_seconds: float | None
    prediction_error_seconds: float | None
    absolute_error_seconds: float | None
    relative_error: float | None
    execution_status: str | None
    used_prediction_source: str | None
    reliability_score: float | None
    model_version: str | None
    training_eligible: bool
    recorded_at: datetime | None


class PredictionHistoryResponse(BaseModel):
    """Persisted prediction history for monitoring."""

    model_config = ConfigDict(extra="forbid")

    items: list[PredictionHistoryItem]
    total: int
    offset: int
    limit: int


__all__ = [
    "HistoricalFeaturesResponse",
    "ManualLoaderConfigurationRequest",
    "TotalExecutionPredictionRequest",
    "TotalExecutionPredictionResponse",
    "AnalyticsOverviewResponse",
    "ExecutionHistoryItem",
    "ExecutionHistoryResponse",
    "DurationTrendPoint",
    "DurationTrendResponse",
    "LoaderIntelligenceItem",
    "LoaderIntelligenceResponse",
    "StageAnalyticsItem",
    "StageAnalyticsResponse",
    "ModelInformationResponse",
    "PredictionHistoryItem",
    "PredictionHistoryResponse",
]