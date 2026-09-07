from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


DATASET_PATH = Path("data/processed/execution_historical_features.csv")

CATEGORICAL_FEATURES = [
    "Loader_Name",
    "Sprint",
]

NUMERIC_FEATURES = [
    "global_prior_execution_count",
    "global_prior_mean_duration_sec",
    "global_prior_median_duration_sec",
    "global_prior_std_duration_sec",
    "loader_prior_execution_count",
    "loader_prior_mean_duration_sec",
    "loader_prior_median_duration_sec",
    "loader_prior_std_duration_sec",
    "loader_previous_duration_sec",
    "seconds_since_loader_previous_execution",
]

FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES

TARGET_COLUMN = "timetaken_in_sec"

MODEL_VERSION = "total-et-v1"

FORBIDDEN_FEATURES = {
    "LDR_Execution_Id",
    "Dataset_Row_Count",
    "Dataset_Count",
    "LDR_Status",
    "ldr_workflow_execution_id",
    "Start_Time",
    "End_Time",
    "staging_timetaken_in_min",
    "data_loading_timetaken_in_sec",
    "data_loading_timetaken_in_sec_1",
    "datamart_approval_timetaken_in_sec",
    "dataloading_approval_timetaken_in_sec",
    "Total_Records",
    "Success_Records",
    "Error_Records",
    "Prevalidation_Errors",
    "Format_Errors",
    "Constraint_Errors",
    "Application_Errors",
    "Cascade_Errors",
    "prevaliadtion_timetaken_in_sec",
    "transformation_timetaken_in_sec",
    "timestamp_duration_sec",
    "known_stage_seconds",
    "target_timestamp_difference_sec",
    "target_stage_ratio",
    "target_quality_status",
}


class ModelTrainingError(ValueError):
    """Raised when the production model training contract is violated."""


@dataclass(frozen=True)
class ModelMetadata:
    model_version: str
    algorithm: str
    target_column: str
    target_transformation: str
    feature_columns: list[str]
    categorical_features: list[str]
    numeric_features: list[str]
    training_rows: int
    training_loader_count: int
    training_sprint_count: int
    training_timestamp_utc: str
    random_state: int
    n_estimators: int
    min_samples_leaf: int
    max_features: float
    artifact_format: str = "joblib"


def _validate_training_frame(df: pd.DataFrame) -> None:
    if df.empty:
        raise ModelTrainingError("Training dataset is empty.")

    if TARGET_COLUMN not in df.columns:
        raise ModelTrainingError(
            f"Required target column '{TARGET_COLUMN}' is missing."
        )

    missing_features = [
        column for column in FEATURE_COLUMNS if column not in df.columns
    ]

    if missing_features:
        raise ModelTrainingError(
            f"Required feature columns are missing: {missing_features}"
        )

    accidental_forbidden = sorted(
        FORBIDDEN_FEATURES.intersection(FEATURE_COLUMNS)
    )

    if accidental_forbidden:
        raise ModelTrainingError(
            "Forbidden features entered the production feature contract: "
            f"{accidental_forbidden}"
        )

    target = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")

    if target.isna().any():
        raise ModelTrainingError(
            "Target contains missing or non-numeric values."
        )

    if not np.isfinite(target.to_numpy()).all():
        raise ModelTrainingError(
            "Target contains non-finite values."
        )

    if (target <= 0).any():
        raise ModelTrainingError(
            "Target must contain only positive execution durations."
        )

    if df["Loader_Name"].notna().sum() == 0:
        raise ModelTrainingError(
            "Training dataset contains no usable Loader_Name values."
        )

    if df["Sprint"].notna().sum() == 0:
        raise ModelTrainingError(
            "Training dataset contains no usable Sprint values."
        )


def _build_pipeline() -> Pipeline:
    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="constant",
                    fill_value="__MISSING__",
                ),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES,
            ),
            (
                "numeric",
                numeric_pipeline,
                NUMERIC_FEATURES,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    model = ExtraTreesRegressor(
        n_estimators=200,
        min_samples_leaf=2,
        max_features=1.0,
        random_state=42,
        n_jobs=-1,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def train_total_execution_model(
    dataset_path: Path = DATASET_PATH,
    artifact_path: Path = Path(
        "backend/artifacts/total_execution_model.joblib"
    ),
) -> ModelMetadata:
    dataset_path = Path(dataset_path)
    artifact_path = Path(artifact_path)

    if not dataset_path.exists():
        raise ModelTrainingError(
            f"Training dataset does not exist: {dataset_path}"
        )

    df = pd.read_csv(dataset_path)

    _validate_training_frame(df)

    X = df[FEATURE_COLUMNS].copy()

    y = (
        pd.to_numeric(
            df[TARGET_COLUMN],
            errors="raise",
        )
        .to_numpy(dtype=float)
    )

    pipeline = _build_pipeline()

    log_target = np.log1p(y)

    pipeline.fit(X, log_target)

    metadata = ModelMetadata(
        model_version=MODEL_VERSION,
        algorithm="ExtraTreesRegressor",
        target_column=TARGET_COLUMN,
        target_transformation="log1p",
        feature_columns=list(FEATURE_COLUMNS),
        categorical_features=list(CATEGORICAL_FEATURES),
        numeric_features=list(NUMERIC_FEATURES),
        training_rows=len(df),
        training_loader_count=int(
            df["Loader_Name"].nunique(dropna=True)
        ),
        training_sprint_count=int(
            df["Sprint"].nunique(dropna=True)
        ),
        training_timestamp_utc=datetime.now(timezone.utc).isoformat(),
        random_state=42,
        n_estimators=200,
        min_samples_leaf=2,
        max_features=1.0,
    )

    artifact = {
        "artifact_type": "total_execution_prediction_model",
        "artifact_version": "1",
        "model_version": MODEL_VERSION,
        "pipeline": pipeline,
        "metadata": asdict(metadata),
    }

    artifact_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        artifact,
        artifact_path,
    )

    return metadata


def load_total_execution_model(
    artifact_path: Path,
) -> dict[str, Any]:
    artifact_path = Path(artifact_path)

    if not artifact_path.exists():
        raise ModelTrainingError(
            f"Model artifact does not exist: {artifact_path}"
        )

    artifact = joblib.load(artifact_path)

    if not isinstance(artifact, dict):
        raise ModelTrainingError(
            "Model artifact has an invalid structure."
        )

    required_keys = {
        "artifact_type",
        "artifact_version",
        "model_version",
        "pipeline",
        "metadata",
    }

    missing_keys = required_keys.difference(artifact)

    if missing_keys:
        raise ModelTrainingError(
            "Model artifact is missing required keys: "
            f"{sorted(missing_keys)}"
        )

    if artifact["artifact_type"] != "total_execution_prediction_model":
        raise ModelTrainingError(
            "Model artifact type does not match the total execution model."
        )

    if artifact["model_version"] != MODEL_VERSION:
        raise ModelTrainingError(
            f"Unsupported model version: {artifact['model_version']}"
        )

    if not isinstance(artifact["pipeline"], Pipeline):
        raise ModelTrainingError(
            "Model artifact does not contain a valid sklearn Pipeline."
        )

    return artifact


__all__ = [
    "CATEGORICAL_FEATURES",
    "DATASET_PATH",
    "FEATURE_COLUMNS",
    "MODEL_VERSION",
    "ModelMetadata",
    "ModelTrainingError",
    "NUMERIC_FEATURES",
    "TARGET_COLUMN",
    "load_total_execution_model",
    "train_total_execution_model",
]