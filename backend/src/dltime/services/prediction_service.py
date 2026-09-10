from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from dltime.features.cold_start_intelligence import (
    ColdStartConfidence,
    ColdStartIntelligence,
)
from dltime.features.online_configuration_features import (
    CONFIGURATION_FEATURE_COLUMNS,
    OnlineConfigurationRequest,
    build_online_configuration_features,
)
from dltime.features.online_historical_features import (
    HISTORICAL_FEATURE_COLUMNS,
    OnlineHistoricalFeatureProvider,
)
from dltime.model.total_execution_model import (
    FEATURE_COLUMNS,
    MODEL_VERSION,
    load_total_execution_model,
)


LOADER_MODEL_SOURCE = "LOADER_MODEL"
GLOBAL_MEDIAN_COLD_START_SOURCE = "GLOBAL_MEDIAN_COLD_START"

PROJECT_ROOT = Path(__file__).resolve().parents[4]

HISTORICAL_DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "execution_historical_features.csv"
)

TARGET_COLUMN = "timetaken_in_sec"


@dataclass(frozen=True)
class TotalExecutionPrediction:
    predicted_total_seconds: float
    prediction_timestamp: datetime
    loader_name: str
    sprint: str
    loader_connection_name: str
    dataset_name: str
    model_version: str
    prediction_source: str
    historical_features: dict[str, float]
    configuration_features: dict[str, float | str]
    cold_start_confidence: Optional[ColdStartConfidence] = None
    similar_loaders: tuple[dict[str, object], ...] = ()


class TotalExecutionPredictionService:
    """
    Runtime total-execution prediction service.

    Known loader:
        eligible historical loader history
        -> ExtraTrees total model

    Unseen loader:
        global historical median
        + similar-loader intelligence
        -> cold-start response

    The global cold-start median comes from the historical
    training dataset. It does not depend on the online feedback
    database being populated.

    Similar-loader intelligence is supporting evidence.
    It does not override the experimentally selected
    global-median cold-start prediction.
    """

    def __init__(
        self,
        historical_provider: OnlineHistoricalFeatureProvider,
        artifact_path: Optional[str | Path] = None,
        cold_start_intelligence: Optional[
            ColdStartIntelligence
        ] = None,
    ) -> None:
        self.historical_provider = historical_provider

        self.cold_start_intelligence = (
            cold_start_intelligence
            or ColdStartIntelligence()
        )

        artifact = load_total_execution_model(
            Path(artifact_path)
            if artifact_path is not None
            else None
        )

        self.model = artifact["model"]
        self.model_version = artifact["model_version"]
        self.target_transform = artifact["target_transform"]

        if self.model_version != MODEL_VERSION:
            raise ValueError(
                "Loaded model version does not match expected model version."
            )

        if not hasattr(self.model, "predict"):
            raise ValueError(
                "Loaded model does not expose predict()."
            )

        self.global_historical_median_seconds = (
            self._load_global_historical_median()
        )

    @staticmethod
    def _validate_text(
        value: str,
        field_name: str,
    ) -> str:
        if value is None:
            raise ValueError(
                f"{field_name} is required."
            )

        value = str(value).strip()

        if not value:
            raise ValueError(
                f"{field_name} cannot be empty."
            )

        return value

    @staticmethod
    def _validate_prediction_timestamp(
        value: datetime,
    ) -> datetime:
        if not isinstance(value, datetime):
            raise ValueError(
                "prediction_timestamp must be a datetime."
            )

        if value.tzinfo is None:
            raise ValueError(
                "prediction_timestamp must be timezone-aware."
            )

        return value

    @staticmethod
    def _load_global_historical_median() -> float:
        """
        Load the global historical median duration from the
        validated historical training dataset.

        This is intentionally independent of the online feedback
        database because a new deployment may have zero feedback
        records.
        """
        if not HISTORICAL_DATASET_PATH.exists():
            raise ValueError(
                "Historical training dataset not found: "
                f"{HISTORICAL_DATASET_PATH}"
            )

        try:
            dataframe = pd.read_csv(
                HISTORICAL_DATASET_PATH
            )
        except Exception as exc:
            raise ValueError(
                "Failed to load historical training dataset: "
                f"{HISTORICAL_DATASET_PATH}"
            ) from exc

        if TARGET_COLUMN not in dataframe.columns:
            raise ValueError(
                "Historical training dataset is missing target column: "
                f"{TARGET_COLUMN}"
            )

        target = pd.to_numeric(
            dataframe[TARGET_COLUMN],
            errors="coerce",
        )

        target = target[
            np.isfinite(target.to_numpy())
            & (target > 0)
        ]

        if target.empty:
            raise ValueError(
                "No valid positive historical execution durations "
                "are available for cold-start fallback."
            )

        median = float(target.median())

        if not np.isfinite(median) or median <= 0:
            raise ValueError(
                "Global historical median must be finite and positive."
            )

        return median

    @staticmethod
    def _build_model_input(
        *,
        loader_name: str,
        sprint: str,
        configuration_coverage_status: str,
        historical_features: dict[str, float | int | None],
        configuration_features: dict[str, float | str],
    ) -> dict[str, object]:
        model_input: dict[str, object] = {
            "Loader_Name": loader_name,
            "Sprint": sprint,
            "configuration_coverage_status": (
                configuration_coverage_status
            ),
        }

        for column in HISTORICAL_FEATURE_COLUMNS:
            value = historical_features.get(
                column,
                np.nan,
            )

            model_input[column] = (
                np.nan
                if value is None
                else float(value)
            )

        for column in CONFIGURATION_FEATURE_COLUMNS:
            value = configuration_features.get(
                column,
                np.nan,
            )

            model_input[column] = (
                np.nan
                if value is None
                else value
            )

        missing = [
            column
            for column in FEATURE_COLUMNS
            if column not in model_input
        ]

        if missing:
            raise ValueError(
                f"Model input missing features: {missing}"
            )

        if len(model_input) != len(FEATURE_COLUMNS):
            raise ValueError(
                "Model input feature count does not match model contract."
            )

        return model_input

    def _predict_with_loader_model(
        self,
        *,
        loader_name: str,
        sprint: str,
        configuration_coverage_status: str,
        historical_features: dict[str, float | int | None],
        configuration_features: dict[str, float | str],
    ) -> float:
        model_input = self._build_model_input(
            loader_name=loader_name,
            sprint=sprint,
            configuration_coverage_status=(
                configuration_coverage_status
            ),
            historical_features=historical_features,
            configuration_features=configuration_features,
        )

        frame = pd.DataFrame(
            [model_input],
            columns=list(FEATURE_COLUMNS),
        )

        transformed_prediction = float(
            self.model.predict(frame)[0]
        )

        if not np.isfinite(transformed_prediction):
            raise ValueError(
                "Model produced a non-finite prediction."
            )

        if self.target_transform == "log1p":
            prediction = float(
                np.expm1(transformed_prediction)
            )
        else:
            prediction = transformed_prediction

        if not np.isfinite(prediction):
            raise ValueError(
                "Inverse-transformed prediction is non-finite."
            )

        return max(0.0, prediction)

    @staticmethod
    def _serialize_similar_loaders(
        similar_loaders: tuple,
    ) -> tuple[dict[str, object], ...]:
        return tuple(
            {
                "loader_name": item.loader_name,
                "sprint": item.sprint,
                "connection": item.connection,
                "similarity_score": float(
                    item.similarity_score
                ),
            }
            for item in similar_loaders
        )

    def predict_total(
        self,
        *,
        loader_name: str,
        sprint: str,
        loader_connection_name: str,
        dataset_name: str,
        prediction_timestamp: datetime,
    ) -> TotalExecutionPrediction:
        loader_name = self._validate_text(
            loader_name,
            "loader_name",
        )

        sprint = self._validate_text(
            sprint,
            "sprint",
        )

        loader_connection_name = self._validate_text(
            loader_connection_name,
            "loader_connection_name",
        )

        dataset_name = self._validate_text(
            dataset_name,
            "dataset_name",
        )

        prediction_timestamp = (
            self._validate_prediction_timestamp(
                prediction_timestamp
            )
        )

        historical = self.historical_provider.get_features(
            loader_name=loader_name,
            prediction_timestamp=prediction_timestamp,
        )

        historical_features = historical.as_dict()

        configuration = build_online_configuration_features(
            OnlineConfigurationRequest(
                sprint=sprint,
                loader_name=loader_name,
                loader_connection_name=(
                    loader_connection_name
                ),
                dataset_name=dataset_name,
            )
        )

        configuration_features = {
            column: configuration[column]
            for column in CONFIGURATION_FEATURE_COLUMNS
        }

        configuration_coverage_status = str(
            configuration["configuration_coverage_status"]
        )

        loader_prior_count = int(
            historical_features[
                "loader_prior_execution_count"
            ]
            or 0
        )

        if loader_prior_count > 0:
            predicted_seconds = (
                self._predict_with_loader_model(
                    loader_name=loader_name,
                    sprint=sprint,
                    configuration_coverage_status=(
                        configuration_coverage_status
                    ),
                    historical_features=historical_features,
                    configuration_features=configuration_features,
                )
            )

            return TotalExecutionPrediction(
                predicted_total_seconds=predicted_seconds,
                prediction_timestamp=prediction_timestamp,
                loader_name=loader_name,
                sprint=sprint,
                loader_connection_name=loader_connection_name,
                dataset_name=dataset_name,
                model_version=self.model_version,
                prediction_source=LOADER_MODEL_SOURCE,
                historical_features={
                    column: float(
                        historical_features[column]
                    )
                    if historical_features[column] is not None
                    else float("nan")
                    for column in HISTORICAL_FEATURE_COLUMNS
                },
                configuration_features=configuration,
            )

        # Cold-start prediction uses the global median from the
        # historical training dataset, not the online feedback store.
        global_median = (
            self.global_historical_median_seconds
        )

        cold_start = self.cold_start_intelligence.evaluate(
            predicted_seconds=global_median,
            loader_name=loader_name,
            sprint=sprint,
            loader_connection_name=loader_connection_name,
            dataset_name=dataset_name,
            configuration_features=configuration_features,
            top_k=3,
        )

        return TotalExecutionPrediction(
            predicted_total_seconds=(
                cold_start.predicted_seconds
            ),
            prediction_timestamp=prediction_timestamp,
            loader_name=loader_name,
            sprint=sprint,
            loader_connection_name=loader_connection_name,
            dataset_name=dataset_name,
            model_version=self.model_version,
            prediction_source=(
                cold_start.prediction_source
            ),
            historical_features={
                column: float(
                    historical_features[column]
                )
                if historical_features[column] is not None
                else float("nan")
                for column in HISTORICAL_FEATURE_COLUMNS
            },
            configuration_features=configuration,
            cold_start_confidence=(
                cold_start.confidence
            ),
            similar_loaders=(
                self._serialize_similar_loaders(
                    cold_start.similar_loaders
                )
            ),
        )