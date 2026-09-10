from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


class ExecutionConfigurationFeatureError(Exception):
    """Raised when execution-level configuration features cannot be built."""


PROJECT_ROOT = Path(__file__).resolve().parents[4]

EXECUTION_TARGETS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "execution_training_targets.csv"
)

RAW_EXECUTIONS_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "LOADER_EXECTUIONS.xlsx"
)

PRE_EXECUTION_FEATURES_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "pre_execution_features.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "execution_configuration_features.csv"
)


REQUIRED_EXECUTION_COLUMNS: tuple[str, ...] = (
    "LDR_Execution_Id",
    "Loader_Name",
    "Sprint",
    "Dataset_display_name",
)

REQUIRED_TARGET_COLUMNS: tuple[str, ...] = (
    "LDR_Execution_Id",
    "Loader_Name",
    "Sprint",
    "ldr_connection_name",
)

REQUIRED_CONFIG_COLUMNS: tuple[str, ...] = (
    "sprint_name",
    "ldr_connection_name",
    "loader_display_name",
    "datasetname",
    "detailed_configuration_row_count",
    "unique_preval_trans_name_count",
    "unique_preval_trans_type_count",
    "unique_calling_type_count",
    "unique_execution_order_count",
    "min_execution_order",
    "max_execution_order",
    "prevalidation_count",
    "transformation_count",
    "configuration_complexity_proxy",
)


OUTPUT_FEATURE_COLUMNS: tuple[str, ...] = (
    "LDR_Execution_Id",
    "configured_dataset_count",
    "matched_configured_dataset_count",
    "configuration_coverage_pct",
    "configuration_coverage_status",
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


def _require_columns(
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
        raise ExecutionConfigurationFeatureError(
            f"{dataset_name} is missing required columns: {missing}"
        )


def _normalise_text(value: object) -> str | None:
    if pd.isna(value):
        return None

    normalized = str(value).strip().upper()

    return normalized or None


def _build_keys(
    dataframe: pd.DataFrame,
    *,
    sprint_column: str,
    loader_column: str,
    dataset_column: str,
    connection_column: str | None = None,
) -> pd.Series:
    parts = [
        dataframe[sprint_column].map(_normalise_text),
        dataframe[loader_column].map(_normalise_text),
    ]

    if connection_column is not None:
        parts.append(
            dataframe[connection_column].map(_normalise_text)
        )

    parts.append(
        dataframe[dataset_column].map(_normalise_text)
    )

    result = parts[0].astype("string")

    for part in parts[1:]:
        result = result + "|" + part.astype("string")

    return result


def _validate_unique_execution_ids(
    dataframe: pd.DataFrame,
) -> None:
    duplicate_mask = dataframe["LDR_Execution_Id"].duplicated()

    if duplicate_mask.any():
        raise ExecutionConfigurationFeatureError(
            "Training-target dataset must contain one row per execution. "
            f"Found {int(duplicate_mask.sum())} duplicate execution rows."
        )


def _numeric(
    dataframe: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    result = dataframe.copy()

    for column in columns:
        result[column] = pd.to_numeric(
            result[column],
            errors="coerce",
        )

    return result


def build_execution_configuration_features(
    execution_targets_path: Path = EXECUTION_TARGETS_PATH,
    raw_executions_path: Path = RAW_EXECUTIONS_PATH,
    pre_execution_features_path: Path = PRE_EXECUTION_FEATURES_PATH,
    output_path: Path = OUTPUT_PATH,
) -> Path:
    """
    Build one pre-execution configuration feature row per training execution.

    The raw execution workbook is used only to identify which datasets
    belonged to each historical execution. No execution outcome, duration,
    status, timestamp, or post-execution metric is used as a feature.

    Configuration is matched using the canonical four-part business key:

        Sprint + Loader + Connection + Dataset

    Only configuration information that can be matched to a historical
    execution dataset is aggregated.

    Coverage is explicitly retained:

        FULL    -> every execution dataset has configuration
        PARTIAL -> some execution datasets have configuration
        NONE    -> no execution dataset has configuration

    Missing configuration is never converted into zero complexity.
    """

    execution_targets_path = Path(execution_targets_path)
    raw_executions_path = Path(raw_executions_path)
    pre_execution_features_path = Path(pre_execution_features_path)
    output_path = Path(output_path)

    for path, label in (
        (execution_targets_path, "training-target dataset"),
        (raw_executions_path, "raw execution workbook"),
        (pre_execution_features_path, "pre-execution feature dataset"),
    ):
        if not path.exists():
            raise ExecutionConfigurationFeatureError(
                f"{label} not found: {path}"
            )

    targets = pd.read_csv(execution_targets_path)
    raw_executions = pd.read_excel(raw_executions_path)
    config = pd.read_csv(pre_execution_features_path)

    _require_columns(
        targets,
        REQUIRED_TARGET_COLUMNS,
        "training-target dataset",
    )

    _require_columns(
        raw_executions,
        REQUIRED_EXECUTION_COLUMNS,
        "raw execution workbook",
    )

    _require_columns(
        config,
        REQUIRED_CONFIG_COLUMNS,
        "pre-execution feature dataset",
    )

    _validate_unique_execution_ids(targets)

    target_execution_ids = set(
        targets["LDR_Execution_Id"].astype(str)
    )

    executions = raw_executions[
        raw_executions["LDR_Execution_Id"].astype(str).isin(
            target_execution_ids
        )
    ].copy()

    target_context = targets[
        [
            "LDR_Execution_Id",
            "Loader_Name",
            "Sprint",
            "ldr_connection_name",
        ]
    ].copy()

    executions = executions.merge(
        target_context,
        on="LDR_Execution_Id",
        how="left",
        suffixes=("", "_target"),
        validate="many_to_one",
    )

    executions["execution_config_key"] = _build_keys(
        executions,
        sprint_column="Sprint",
        loader_column="Loader_Name",
        connection_column="ldr_connection_name",
        dataset_column="Dataset_display_name",
    )

    config["execution_config_key"] = _build_keys(
        config,
        sprint_column="sprint_name",
        loader_column="loader_display_name",
        connection_column="ldr_connection_name",
        dataset_column="datasetname",
    )

    config = _numeric(
        config,
        [
            "detailed_configuration_row_count",
            "unique_preval_trans_name_count",
            "unique_preval_trans_type_count",
            "unique_calling_type_count",
            "unique_execution_order_count",
            "min_execution_order",
            "max_execution_order",
            "prevalidation_count",
            "transformation_count",
            "configuration_complexity_proxy",
        ],
    )

    duplicate_config_keys = config[
        config["execution_config_key"].duplicated(
            keep=False
        )
    ]

    if not duplicate_config_keys.empty:
        raise ExecutionConfigurationFeatureError(
            "Pre-execution configuration contains duplicate canonical "
            "configuration keys. Cannot safely aggregate ambiguous "
            f"configuration rows: {len(duplicate_config_keys)} rows."
        )

    dataset_rows = executions[
        [
            "LDR_Execution_Id",
            "Dataset_display_name",
            "execution_config_key",
        ]
    ].drop_duplicates()

    matched = dataset_rows.merge(
        config[
            [
                "execution_config_key",
                "detailed_configuration_row_count",
                "unique_preval_trans_name_count",
                "unique_preval_trans_type_count",
                "unique_calling_type_count",
                "unique_execution_order_count",
                "min_execution_order",
                "max_execution_order",
                "prevalidation_count",
                "transformation_count",
                "configuration_complexity_proxy",
            ]
        ],
        on="execution_config_key",
        how="left",
        indicator=True,
        validate="many_to_one",
    )

    matched["configuration_matched"] = (
        matched["_merge"] == "both"
    )

    coverage = (
        matched.groupby(
            "LDR_Execution_Id",
            sort=False,
        )
        .agg(
            configured_dataset_count=(
                "Dataset_display_name",
                "size",
            ),
            matched_configured_dataset_count=(
                "configuration_matched",
                "sum",
            ),
        )
        .reset_index()
    )

    coverage["configuration_coverage_pct"] = (
        100.0
        * coverage["matched_configured_dataset_count"]
        / coverage["configured_dataset_count"]
    )

    coverage["configuration_coverage_status"] = np.select(
        [
            coverage["matched_configured_dataset_count"]
            == coverage["configured_dataset_count"],
            coverage["matched_configured_dataset_count"] > 0,
        ],
        [
            "FULL",
            "PARTIAL",
        ],
        default="NONE",
    )

    numeric_config_columns = [
        "detailed_configuration_row_count",
        "unique_preval_trans_name_count",
        "unique_preval_trans_type_count",
        "unique_calling_type_count",
        "unique_execution_order_count",
        "min_execution_order",
        "max_execution_order",
        "prevalidation_count",
        "transformation_count",
        "configuration_complexity_proxy",
    ]

    matched_config = matched[
        matched["configuration_matched"]
    ].copy()

    if matched_config.empty:
        aggregates = pd.DataFrame(
            columns=[
                "LDR_Execution_Id",
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
            ]
        )

    else:
        grouped = matched_config.groupby(
            "LDR_Execution_Id",
            sort=False,
        )

        aggregates = grouped.agg(
            configured_prevalidation_count=(
                "prevalidation_count",
                "sum",
            ),
            configured_transformation_count=(
                "transformation_count",
                "sum",
            ),
            configured_detail_row_count=(
                "detailed_configuration_row_count",
                "sum",
            ),
            configured_unique_preval_trans_name_count=(
                "unique_preval_trans_name_count",
                "sum",
            ),
            configured_unique_preval_trans_type_count=(
                "unique_preval_trans_type_count",
                "sum",
            ),
            configured_unique_calling_type_count=(
                "unique_calling_type_count",
                "sum",
            ),
            configured_execution_order_count=(
                "unique_execution_order_count",
                "sum",
            ),
            configured_min_execution_order=(
                "min_execution_order",
                "min",
            ),
            configured_max_execution_order=(
                "max_execution_order",
                "max",
            ),
            configured_max_dataset_complexity=(
                "configuration_complexity_proxy",
                "max",
            ),
        ).reset_index()

        aggregates["configured_total_complexity_count"] = (
            aggregates["configured_prevalidation_count"].fillna(0)
            + aggregates["configured_transformation_count"].fillna(0)
        )

        aggregates["configured_execution_order_span"] = (
            aggregates["configured_max_execution_order"]
            - aggregates["configured_min_execution_order"]
        )

    result = coverage.merge(
        aggregates,
        on="LDR_Execution_Id",
        how="left",
        validate="one_to_one",
    )

    numeric_output_columns = [
        column
        for column in OUTPUT_FEATURE_COLUMNS
        if column not in {
            "LDR_Execution_Id",
            "configuration_coverage_status",
        }
    ]

    for column in numeric_output_columns:
        result[column] = pd.to_numeric(
            result[column],
            errors="coerce",
        )

    result = result[
        list(OUTPUT_FEATURE_COLUMNS)
    ].sort_values(
        "LDR_Execution_Id",
        kind="mergesort",
    ).reset_index(drop=True)

    if len(result) != len(targets):
        raise ExecutionConfigurationFeatureError(
            "Configuration feature output does not contain exactly one row "
            "per training execution: "
            f"expected {len(targets)}, got {len(result)}."
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        output_path,
        index=False,
    )

    return output_path


if __name__ == "__main__":
    output = build_execution_configuration_features()

    result = pd.read_csv(output)

    print("execution configuration feature build complete")
    print(f"output={output}")
    print(f"rows={len(result)}")
    print(
        "coverage_status="
        + result["configuration_coverage_status"]
        .value_counts()
        .to_string()
    )
