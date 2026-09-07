from __future__ import annotations

from datetime import datetime

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
    historical_features: HistoricalFeaturesResponse


__all__ = [
    "HistoricalFeaturesResponse",
    "ManualLoaderConfigurationRequest",
    "TotalExecutionPredictionRequest",
    "TotalExecutionPredictionResponse",
]