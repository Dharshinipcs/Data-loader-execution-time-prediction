from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


PROJECT_ROOT = Path(__file__).resolve().parents[4]

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "execution_historical_features.csv"
)

CONFIGURATION_FEATURES_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "execution_configuration_features.csv"
)

MODEL_OUTPUT_PATH = (
    PROJECT_ROOT
    / "backend"
    / "artifacts"
    / "total_execution_model.joblib"
)


MODEL_VERSION = "total-et-v2"


CATEGORICAL_FEATURES = (
    "Loader_Name",
    "Sprint",
    "configuration_coverage_status",
)


HISTORICAL_FEATURES = (
    "global_prior_execution_count",
    "global_prior_mean_duration_sec",
    "global_prior_median_duration_sec",
    "global_prior_std_duration_sec",
    "loader_prior_execution_count",
    "loader_prior_mean_duration_sec",
    "loader_prior_median_duration_sec",
    "loader_prior_std_duration_sec",
    "loader_previous_duration_sec",
    "seconds_since_loader_previous",
)


CONFIGURATION_FEATURES = (
    "configured_dataset_count",
    "matched_configured_dataset_count",
    "configuration_coverage_pct",
    "configured_total_complexity_count",
    "configured_prevalidation_count",
    "configured_transformation_count",
    "configured_detail_row_count",
    "configured_unique_preval_trans_name_count",
    "configured_unique_preval_trans_type_count",
    "configured_unique_calling_type_count",
    "configured_execution_order_count",
    "configured_min_execution_order",
    "configured_max_execution_order",
    "configured_execution_order_span",
    "configured_max_dataset_complexity",
)


FEATURE_COLUMNS = (
    *CATEGORICAL_FEATURES,
    *HISTORICAL_FEATURES,
    *CONFIGURATION_FEATURES,
)


TARGET_COLUMN = "timetaken_in_sec"


FORBIDDEN_FEATURES = {
    "LDR_Execution_Id",
    "Dataset_display_name",
    "Dataset_Count",
    "Dataset_Row_Count",
    "Total_Records",
    "Success_Records",
    "Error_Records",
    "Prevalidation_Errors",
    "Format_Errors",
    "Constraint_Errors",
    "Application_Errors",
    "Cascade_Errors",
    "LDR_Status",
    "Start_Time",
    "End_Time",
    "Time_Taken",
    "staging_start_time",
    "staging_end_time",
    "staging_timetaken_in_min",
    "prevalidation_start_time",
    "prevalidation_end_time",
    "prevaliadtion_timetaken",
    "prevaliadtion_timetaken_in_sec",
    "transformation_start_time",
    "transformation_end_time",
    "transformation_timetaken",
    "transformation_timetaken_in_sec",
    "pre_load_start_time",
    "pre_load_end_time",
    "data_loading_timetaken",
    "data_loading_timetaken_in_sec",
    "data_loading_start_time",
    "data_loading_end_time",
    "data_loading_timetaken_1",
    "data_loading_timetaken_in_sec_1",
    "datamart_approval_timetaken",
    "datamart_approval_timetaken_in_sec",
    "dataloading_approval_timetaken",
    "dataloading_approval_timetaken_in_sec",
    "actual_stage_durations",
    TARGET_COLUMN,
}


class TotalExecutionModelError(Exception):
    """Raised when total execution model operations fail safely."""


def _validate_columns(
    dataframe: pd.DataFrame,
    required_columns: tuple[str, ...],
    dataset_name: str,
) -> None:
    missing = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing:
        raise TotalExecutionModelError(
            f"{dataset_name} is missing required columns: "
            + ", ".join(missing)
        )


def _validate_forbidden_features() -> None:
    """
    Validate that no forbidden post-execution or target-derived field
    has entered the model feature contract.
    """
    forbidden_present = sorted(
        FORBIDDEN_FEATURES.intersection(
            FEATURE_COLUMNS
        )
    )

    if forbidden_present:
        raise TotalExecutionModelError(
            "Forbidden post-execution or target-derived features "
            "are included in the model feature contract: "
            + ", ".join(forbidden_present)
        )


def _validate_training_data(
    dataframe: pd.DataFrame,
) -> None:
    if dataframe.empty:
        raise TotalExecutionModelError(
            "Training dataset is empty."
        )

    _validate_columns(
        dataframe,
        FEATURE_COLUMNS + (TARGET_COLUMN,),
        "Training dataset",
    )

    _validate_forbidden_features()

    target = pd.to_numeric(
        dataframe[TARGET_COLUMN],
        errors="coerce",
    )

    if target.isna().any():
        raise TotalExecutionModelError(
            "Training target contains missing or non-numeric values."
        )

    if (~np.isfinite(target.to_numpy())).any():
        raise TotalExecutionModelError(
            "Training target contains non-finite values."
        )

    if (target <= 0).any():
        raise TotalExecutionModelError(
            "Training target contains non-positive values."
        )

    if dataframe["Loader_Name"].isna().all():
        raise TotalExecutionModelError(
            "Loader_Name is completely missing."
        )

    if dataframe["Sprint"].isna().all():
        raise TotalExecutionModelError(
            "Sprint is completely missing."
        )


def _load_training_data() -> pd.DataFrame:
    if not DATASET_PATH.exists():
        raise TotalExecutionModelError(
            f"Historical feature dataset not found: {DATASET_PATH}"
        )

    if not CONFIGURATION_FEATURES_PATH.exists():
        raise TotalExecutionModelError(
            "Execution configuration feature dataset not found: "
            f"{CONFIGURATION_FEATURES_PATH}"
        )

    historical = pd.read_csv(
        DATASET_PATH
    )

    configuration = pd.read_csv(
        CONFIGURATION_FEATURES_PATH
    )

    _validate_columns(
        historical,
        (
            "LDR_Execution_Id",
            "Loader_Name",
            "Sprint",
            "Start_Time",
            TARGET_COLUMN,
            *HISTORICAL_FEATURES,
        ),
        "Historical feature dataset",
    )

    _validate_columns(
        configuration,
        (
            "LDR_Execution_Id",
            *CATEGORICAL_FEATURES[2:],
            *CONFIGURATION_FEATURES,
        ),
        "Configuration feature dataset",
    )

    if historical[
        "LDR_Execution_Id"
    ].duplicated().any():
        raise TotalExecutionModelError(
            "Historical feature dataset must contain one row per execution."
        )

    if configuration[
        "LDR_Execution_Id"
    ].duplicated().any():
        raise TotalExecutionModelError(
            "Configuration feature dataset must contain one row per execution."
        )

    configuration_subset = configuration[
        [
            "LDR_Execution_Id",
            *CATEGORICAL_FEATURES[2:],
            *CONFIGURATION_FEATURES,
        ]
    ].copy()

    merged = historical.merge(
        configuration_subset,
        on="LDR_Execution_Id",
        how="left",
        validate="one_to_one",
    )

    _validate_training_data(
        merged
    )

    return merged


def _build_pipeline() -> Pipeline:
    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent",
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
                SimpleImputer(
                    strategy="median"
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                categorical_pipeline,
                list(CATEGORICAL_FEATURES),
            ),
            (
                "numeric",
                numeric_pipeline,
                list(
                    HISTORICAL_FEATURES
                    + CONFIGURATION_FEATURES
                ),
            ),
        ],
        remainder="drop",
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
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "model",
                model,
            ),
        ]
    )


def train_total_execution_model() -> Path:
    dataframe = _load_training_data()

    X = dataframe[
        list(FEATURE_COLUMNS)
    ].copy()

    y = pd.to_numeric(
        dataframe[TARGET_COLUMN],
        errors="coerce",
    ).to_numpy(
        dtype=float
    )

    if not np.isfinite(y).all() or (
        y <= 0
    ).any():
        raise TotalExecutionModelError(
            "Target contains invalid values after preparation."
        )

    pipeline = _build_pipeline()

    # Execution time is highly right-skewed,
    # so train in log space.
    y_log = np.log1p(y)

    pipeline.fit(
        X,
        y_log,
    )

    artifact = {
        "model": pipeline,
        "model_version": MODEL_VERSION,
        "feature_columns": list(
            FEATURE_COLUMNS
        ),
        "categorical_features": list(
            CATEGORICAL_FEATURES
        ),
        "historical_features": list(
            HISTORICAL_FEATURES
        ),
        "configuration_features": list(
            CONFIGURATION_FEATURES
        ),
        "target_column": TARGET_COLUMN,
        "target_transform": "log1p",
    }

    MODEL_OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        artifact,
        MODEL_OUTPUT_PATH,
    )

    return MODEL_OUTPUT_PATH


def load_total_execution_model(
    artifact_path: Path | None = None,
) -> dict[str, object]:
    """
    Load and validate the trained total execution model artifact.

    The persisted trained pipeline is stored under the ``model`` key.
    """
    path = (
        artifact_path
        if artifact_path is not None
        else MODEL_OUTPUT_PATH
    )

    if not path.exists():
        raise TotalExecutionModelError(
            f"Model artifact not found: {path}"
        )

    try:
        artifact = joblib.load(
            path
        )
    except Exception as exc:
        raise TotalExecutionModelError(
            f"Failed to load model artifact: {path}"
        ) from exc

    if not isinstance(
        artifact,
        dict,
    ):
        raise TotalExecutionModelError(
            "Model artifact must be a dictionary."
        )

    required_keys = {
        "model",
        "model_version",
        "feature_columns",
        "categorical_features",
        "historical_features",
        "configuration_features",
        "target_column",
        "target_transform",
    }

    missing_keys = sorted(
        required_keys.difference(
            artifact.keys()
        )
    )

    if missing_keys:
        raise TotalExecutionModelError(
            "Model artifact is missing required keys: "
            + ", ".join(missing_keys)
        )

    if artifact["model_version"] != MODEL_VERSION:
        raise TotalExecutionModelError(
            "Model artifact version mismatch: "
            f"expected {MODEL_VERSION}, "
            f"found {artifact['model_version']}"
        )

    if artifact["target_transform"] != "log1p":
        raise TotalExecutionModelError(
            "Unsupported target transform in model artifact: "
            f"{artifact['target_transform']}"
        )

    stored_features = tuple(
        artifact["feature_columns"]
    )

    if stored_features != FEATURE_COLUMNS:
        raise TotalExecutionModelError(
            "Model artifact feature contract does not match "
            "the current model feature contract."
        )

    if not hasattr(
        artifact["model"],
        "predict",
    ):
        raise TotalExecutionModelError(
            "Persisted model does not expose predict()."
        )

    return artifact


if __name__ == "__main__":
    output = train_total_execution_model()

    print(
        "total execution model training complete"
    )
    print(
        f"output={output}"
    )
    print(
        f"model_version={MODEL_VERSION}"
    )
    print(
        f"feature_count={len(FEATURE_COLUMNS)}"
    )