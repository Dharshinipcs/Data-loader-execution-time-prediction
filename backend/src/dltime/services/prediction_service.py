from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from dltime.features.online_historical_features import (
    HISTORICAL_FEATURE_COLUMNS,
    OnlineHistoricalFeatureProvider,
)
from dltime.model.total_execution_model import (
    FEATURE_COLUMNS,
    MODEL_VERSION,
    load_total_execution_model,
)


class PredictionServiceError(ValueError):
    """Raised when a prediction request cannot be processed safely."""


@dataclass(frozen=True)
class TotalExecutionPrediction:
    """Validated total execution-time prediction."""

    predicted_total_seconds: float
    prediction_timestamp: str
    loader_name: str | None
    sprint: str | None
    model_version: str
    historical_features: dict[str, float | int | None]

    def as_dict(self) -> dict[str, object]:
        """Return the prediction as a JSON-compatible dictionary."""
        return {
            "predicted_total_seconds": self.predicted_total_seconds,
            "prediction_timestamp": self.prediction_timestamp,
            "loader_name": self.loader_name,
            "sprint": self.sprint,
            "model_version": self.model_version,
            "historical_features": self.historical_features,
        }


class TotalExecutionPredictionService:
    """Build pre-execution features and predict total execution time."""

    def __init__(
        self,
        historical_provider: OnlineHistoricalFeatureProvider,
        artifact_path: Path,
    ) -> None:
        self._historical_provider = historical_provider
        self._artifact = load_total_execution_model(artifact_path)

        if self._artifact["model_version"] != MODEL_VERSION:
            raise PredictionServiceError(
                "Loaded model version does not match the prediction service."
            )

    @staticmethod
    def _validate_prediction_timestamp(
        prediction_timestamp: str | datetime | None,
    ) -> str:
        if prediction_timestamp is None:
            timestamp = datetime.now(timezone.utc)
        elif isinstance(prediction_timestamp, datetime):
            timestamp = prediction_timestamp
        elif isinstance(prediction_timestamp, str):
            try:
                timestamp = datetime.fromisoformat(
                    prediction_timestamp.replace("Z", "+00:00")
                )
            except ValueError as exc:
                raise PredictionServiceError(
                    "prediction_timestamp must be a valid ISO-8601 timestamp."
                ) from exc
        else:
            raise PredictionServiceError(
                "prediction_timestamp must be a string, datetime, or None."
            )

        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        else:
            timestamp = timestamp.astimezone(timezone.utc)

        return timestamp.isoformat()

    @staticmethod
    def _validate_prediction_value(
        prediction_log_seconds: float,
    ) -> float:
        if not np.isfinite(prediction_log_seconds):
            raise PredictionServiceError(
                "Model returned a non-finite prediction."
            )

        prediction_seconds = float(np.expm1(prediction_log_seconds))

        if not np.isfinite(prediction_seconds):
            raise PredictionServiceError(
                "Inverse-transformed prediction is non-finite."
            )

        if prediction_seconds <= 0:
            raise PredictionServiceError(
                "Model returned a non-positive execution-time prediction."
            )

        return prediction_seconds

    def predict_total(
        self,
        loader_name: str | None,
        sprint: str | None,
        prediction_timestamp: str | datetime | None = None,
    ) -> TotalExecutionPrediction:
        timestamp = self._validate_prediction_timestamp(
            prediction_timestamp
        )

        historical_features = self._historical_provider.build(
            loader_name=loader_name,
            prediction_timestamp=timestamp,
        )

        feature_values = historical_features.as_dict()

        model_input = {
            "Loader_Name": loader_name,
            "Sprint": sprint,
        }

        for column in HISTORICAL_FEATURE_COLUMNS:
            model_input[column] = feature_values[column]

        X = pd.DataFrame(
            [model_input],
            columns=FEATURE_COLUMNS,
        )

        try:
            prediction_log_seconds = float(
                self._artifact["pipeline"].predict(X)[0]
            )
        except Exception as exc:
            raise PredictionServiceError(
                "Model prediction failed."
            ) from exc

        predicted_total_seconds = self._validate_prediction_value(
            prediction_log_seconds
        )

        return TotalExecutionPrediction(
            predicted_total_seconds=predicted_total_seconds,
            prediction_timestamp=timestamp,
            loader_name=loader_name,
            sprint=sprint,
            model_version=MODEL_VERSION,
            historical_features=feature_values,
        )


__all__ = [
    "PredictionServiceError",
    "TotalExecutionPrediction",
    "TotalExecutionPredictionService",
]