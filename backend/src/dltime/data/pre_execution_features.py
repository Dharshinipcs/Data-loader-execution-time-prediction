from __future__ import annotations

from pathlib import Path

import pandas as pd

from dltime.data.excel_loader import load_excel
from dltime.data.raw_schemas import (
    LOADER_DEFINITIONS_1_COLUMNS,
    LOADER_DEFINITIONS_COLUMNS,
)
from dltime.data.schema import validate_required_columns


class PreExecutionFeatureError(Exception):
    """Raised when pre-execution feature construction fails."""


# These fields identify a concrete loader configuration context.
CONFIGURATION_KEY_COLUMNS: tuple[str, ...] = (
    "sprint_name",
    "ldr_connection_name",
    "loader_display_name",
    "datasetname",
)


# Configuration-detail columns that are converted into aggregate features.
CONFIGURATION_DETAIL_COLUMNS: tuple[str, ...] = (
    "preval_trans_display_name",
    "type",
    "calling_type",
    "execution_order",
)


PROJECT_ROOT = Path(__file__).resolve().parents[4]

RAW_DEFINITIONS_PATH = (
    PROJECT_ROOT / "data" / "raw" / "LOADER_DEFINATIONS.xlsx"
)

RAW_DEFINITIONS_1_PATH = (
    PROJECT_ROOT / "data" / "raw" / "LOADER_DEFINATIONS_1.xlsx"
)

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

OUTPUT_PATH = (
    PROCESSED_DIR / "pre_execution_features.csv"
)


def _validate_configuration_keys(
    dataframe: pd.DataFrame,
) -> None:
    """Validate that configuration identity fields are not missing."""
    missing_key_values = dataframe[list(CONFIGURATION_KEY_COLUMNS)].isna().any(
        axis=1
    )

    if missing_key_values.any():
        count = int(missing_key_values.sum())

        raise PreExecutionFeatureError(
            "Configuration data contains "
            f"{count} rows with missing business-key values."
        )


def _build_detail_features(
    definitions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate configuration-detail rows into one row per configuration.

    No execution outcomes or execution-time information is used here.
    """

    grouped = definitions.groupby(
        list(CONFIGURATION_KEY_COLUMNS),
        dropna=False,
        sort=False,
    )

    features = grouped.agg(
        detailed_configuration_row_count=(
            "preval_trans_display_name",
            "size",
        ),
        unique_preval_trans_name_count=(
            "preval_trans_display_name",
            "nunique",
        ),
        unique_preval_trans_type_count=(
            "type",
            "nunique",
        ),
        unique_calling_type_count=(
            "calling_type",
            "nunique",
        ),
        unique_execution_order_count=(
            "execution_order",
            "nunique",
        ),
        min_execution_order=(
            "execution_order",
            "min",
        ),
        max_execution_order=(
            "execution_order",
            "max",
        ),
    ).reset_index()

    return features


def _build_summary_features(
    definitions_1: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build one summary row per configuration.

    LOADER_DEFINATIONS_1 already contains aggregate prevalidation and
    transformation counts, so duplicate business keys are rejected rather
    than silently aggregated.
    """

    duplicate_mask = definitions_1.duplicated(
        subset=list(CONFIGURATION_KEY_COLUMNS),
        keep=False,
    )

    if duplicate_mask.any():
        duplicate_count = int(duplicate_mask.sum())

        raise PreExecutionFeatureError(
            "LOADER_DEFINATIONS_1 contains duplicate configuration keys: "
            f"{duplicate_count} rows."
        )

    return definitions_1[
        [
            *CONFIGURATION_KEY_COLUMNS,
            "prevalidation_count",
            "transformation_count",
        ]
    ].copy()


def build_pre_execution_features(
    definitions_path: Path = RAW_DEFINITIONS_PATH,
    definitions_1_path: Path = RAW_DEFINITIONS_1_PATH,
    output_path: Path = OUTPUT_PATH,
) -> Path:
    """
    Build the canonical pre-execution feature dataset.

    Pipeline
    --------
    LOADER_DEFINATIONS.xlsx
        -> schema validation
        -> configuration-detail aggregation

    LOADER_DEFINATIONS_1.xlsx
        -> schema validation
        -> summary validation

    Both
        -> business-key join
        -> derived complexity features
        -> pre_execution_features.csv

    Important
    ---------
    This dataset contains only information available from the
    pre-execution configuration sources. It intentionally does not use
    historical execution outcomes or execution-time targets.
    """

    definitions = load_excel(definitions_path)
    definitions_1 = load_excel(definitions_1_path)

    validate_required_columns(
        definitions,
        LOADER_DEFINITIONS_COLUMNS,
    )

    validate_required_columns(
        definitions_1,
        LOADER_DEFINITIONS_1_COLUMNS,
    )

    _validate_configuration_keys(definitions)
    _validate_configuration_keys(definitions_1)

    detail_features = _build_detail_features(definitions)

    summary_features = _build_summary_features(definitions_1)

    features = detail_features.merge(
        summary_features,
        on=list(CONFIGURATION_KEY_COLUMNS),
        how="outer",
        validate="one_to_one",
        indicator=True,
    )

    features["pre_execution_match_status"] = features["_merge"].map(
        {
            "both": "MATCHED",
            "left_only": "DETAIL_ONLY",
            "right_only": "SUMMARY_ONLY",
        }
    )

    features = features.drop(columns=["_merge"])

    count_columns = [
        "prevalidation_count",
        "transformation_count",
        "detailed_configuration_row_count",
        "unique_preval_trans_name_count",
        "unique_preval_trans_type_count",
        "unique_calling_type_count",
        "unique_execution_order_count",
    ]

    for column in count_columns:
        features[column] = pd.to_numeric(
            features[column],
            errors="coerce",
        )

    features["total_complexity_count"] = (
        features["prevalidation_count"].fillna(0)
        + features["transformation_count"].fillna(0)
    )

    features["configuration_complexity_proxy"] = (
        features["detailed_configuration_row_count"].fillna(0)
        + features["unique_preval_trans_name_count"].fillna(0)
        + features["unique_preval_trans_type_count"].fillna(0)
        + features["unique_calling_type_count"].fillna(0)
    )

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    features.to_csv(
        output_path,
        index=False,
    )

    return output_path


if __name__ == "__main__":
    output = build_pre_execution_features()

    print("pre-execution feature build complete")
    print(f"output={output}")