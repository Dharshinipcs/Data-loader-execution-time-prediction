from __future__ import annotations

from pathlib import Path

import pandas as pd

from dltime.data.excel_loader import load_excel
from dltime.data.raw_schemas import (
    LOADER_DEFINITIONS_1_COLUMNS,
    LOADER_DEFINITIONS_COLUMNS,
)
from dltime.data.schema import validate_required_columns


PROJECT_ROOT = Path(__file__).resolve().parents[4]

LOADER_DEFINITIONS_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "LOADER_DEFINATIONS.xlsx"
)

LOADER_DEFINITIONS_1_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "LOADER_DEFINATIONS_1.xlsx"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "pre_execution_features.csv"
)


CONFIGURATION_KEY_COLUMNS: tuple[str, ...] = (
    "sprint_name",
    "ldr_connection_name",
    "loader_display_name",
    "datasetname",
)

CONFIGURATION_DETAIL_COLUMNS: tuple[str, ...] = (
    "preval_trans_display_name",
    "type",
    "calling_type",
    "execution_order",
)


class PreExecutionFeatureError(Exception):
    """Raised when pre-execution feature construction fails."""


def _normalise_text(value: object) -> str | None:
    """Normalize configuration identity text consistently."""
    if pd.isna(value):
        return None

    normalized = (
        str(value)
        .strip()
        .upper()
    )

    return normalized or None


def _validate_configuration_keys(
    dataframe: pd.DataFrame,
    dataset_name: str,
) -> None:
    """Ensure all configuration business-key fields are available."""
    missing_key_values = dataframe[
        list(CONFIGURATION_KEY_COLUMNS)
    ].isna().any(axis=1)

    if missing_key_values.any():
        raise PreExecutionFeatureError(
            f"{dataset_name} contains "
            f"{int(missing_key_values.sum())} rows with missing "
            "configuration business-key values."
        )


def _build_configuration_key(
    dataframe: pd.DataFrame,
) -> pd.Series:
    """Build the canonical four-part configuration key."""
    normalized = pd.DataFrame(
        {
            column: dataframe[column].map(_normalise_text)
            for column in CONFIGURATION_KEY_COLUMNS
        },
        index=dataframe.index,
    )

    return (
        normalized["sprint_name"].astype("string")
        + "|"
        + normalized["ldr_connection_name"].astype("string")
        + "|"
        + normalized["loader_display_name"].astype("string")
        + "|"
        + normalized["datasetname"].astype("string")
    )


def _build_detail_features(
    definitions: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate detailed loader configuration metadata."""
    required_columns = [
        *CONFIGURATION_KEY_COLUMNS,
        *CONFIGURATION_DETAIL_COLUMNS,
    ]

    validate_required_columns(
        definitions,
        required_columns,
    )

    detail = definitions.copy()

    _validate_configuration_keys(
        detail,
        "LOADER_DEFINATIONS.xlsx",
    )

    for column in CONFIGURATION_KEY_COLUMNS:
        detail[column] = detail[column].map(
            _normalise_text,
        )

    detail["execution_order"] = pd.to_numeric(
        detail["execution_order"],
        errors="coerce",
    )

    grouped = (
        detail.groupby(
            list(CONFIGURATION_KEY_COLUMNS),
            dropna=False,
            sort=False,
        )
        .agg(
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
        )
        .reset_index()
    )

    grouped["configuration_complexity_proxy"] = (
        grouped["detailed_configuration_row_count"]
        * (
            1
            + grouped["unique_preval_trans_type_count"]
            .fillna(0)
            + grouped["unique_calling_type_count"]
            .fillna(0)
        )
    )

    return grouped


def _build_summary_features(
    definitions_1: pd.DataFrame,
) -> pd.DataFrame:
    """Validate and prepare the primary configuration summary."""
    validate_required_columns(
        definitions_1,
        list(LOADER_DEFINITIONS_1_COLUMNS),
    )

    summary = definitions_1.copy()

    _validate_configuration_keys(
        summary,
        "LOADER_DEFINATIONS_1.xlsx",
    )

    for column in CONFIGURATION_KEY_COLUMNS:
        summary[column] = summary[column].map(
            _normalise_text,
        )

    duplicate_mask = summary.duplicated(
        subset=list(CONFIGURATION_KEY_COLUMNS),
        keep=False,
    )

    if duplicate_mask.any():
        duplicate_count = int(
            duplicate_mask.sum()
        )

        raise PreExecutionFeatureError(
            "LOADER_DEFINATIONS_1.xlsx contains duplicate canonical "
            "configuration keys: "
            f"{duplicate_count} rows."
        )

    numeric_columns = [
        "prevalidation_count",
        "transformation_count",
    ]

    for column in numeric_columns:
        summary[column] = pd.to_numeric(
            summary[column],
            errors="coerce",
        )

    return summary[
        [
            *CONFIGURATION_KEY_COLUMNS,
            "prevalidation_count",
            "transformation_count",
        ]
    ]


def build_pre_execution_features(
    definitions_path: Path = LOADER_DEFINITIONS_PATH,
    definitions_1_path: Path = LOADER_DEFINITIONS_1_PATH,
    output_path: Path = OUTPUT_PATH,
) -> Path:
    """
    Build one pre-execution configuration feature row per configured dataset.

    Only information available from loader configuration is used.

    Unknown configuration remains missing rather than being represented as
    zero complexity.
    """
    definitions_path = Path(
        definitions_path,
    )

    definitions_1_path = Path(
        definitions_1_path,
    )

    output_path = Path(
        output_path,
    )

    for path, label in (
        (
            definitions_path,
            "LOADER_DEFINATIONS.xlsx",
        ),
        (
            definitions_1_path,
            "LOADER_DEFINATIONS_1.xlsx",
        ),
    ):
        if not path.exists():
            raise PreExecutionFeatureError(
                f"{label} not found: {path}"
            )

    definitions = load_excel(
        definitions_path,
    )

    definitions_1 = load_excel(
        definitions_1_path,
    )

    validate_required_columns(
        definitions,
        list(LOADER_DEFINITIONS_COLUMNS),
    )

    validate_required_columns(
        definitions_1,
        list(LOADER_DEFINITIONS_1_COLUMNS),
    )

    detail_features = _build_detail_features(
        definitions,
    )

    summary_features = _build_summary_features(
        definitions_1,
    )

    detail_features["configuration_key"] = (
        _build_configuration_key(
            detail_features,
        )
    )

    summary_features["configuration_key"] = (
        _build_configuration_key(
            summary_features,
        )
    )

    detail_columns = [
        "configuration_key",
        "detailed_configuration_row_count",
        "unique_preval_trans_name_count",
        "unique_preval_trans_type_count",
        "unique_calling_type_count",
        "unique_execution_order_count",
        "min_execution_order",
        "max_execution_order",
        "configuration_complexity_proxy",
    ]

    summary_columns = [
        "configuration_key",
        "prevalidation_count",
        "transformation_count",
    ]

    result = summary_features[
        summary_columns
    ].merge(
        detail_features[
            detail_columns
        ],
        on="configuration_key",
        how="outer",
        validate="one_to_one",
    )

    result["pre_execution_match_status"] = (
        result["configuration_key"]
        .map(
            lambda value: (
                "MATCHED"
                if pd.notna(value)
                else "UNMATCHED"
            )
        )
    )

    result["total_complexity_count"] = (
        result["prevalidation_count"]
        + result["transformation_count"]
    )

    result["configuration_complexity_proxy"] = pd.to_numeric(
        result["configuration_complexity_proxy"],
        errors="coerce",
    )

    result["total_complexity_count"] = pd.to_numeric(
        result["total_complexity_count"],
        errors="coerce",
    )

    # IMPORTANT:
    # Do not fill missing configuration values with zero.
    #
    # Missing means that the configuration feature is unknown. Zero would
    # incorrectly imply that the configured complexity was actually zero.
    #
    # This distinction is required for downstream FULL/PARTIAL/NONE coverage
    # handling and cold-start prediction.
    result["total_complexity_count"] = (
        result["total_complexity_count"]
    )

    result["configuration_complexity_proxy"] = (
        result["configuration_complexity_proxy"]
    )

    output_columns = [
        "sprint_name",
        "ldr_connection_name",
        "loader_display_name",
        "datasetname",
        "prevalidation_count",
        "transformation_count",
        "detailed_configuration_row_count",
        "unique_preval_trans_name_count",
        "unique_preval_trans_type_count",
        "unique_calling_type_count",
        "unique_execution_order_count",
        "min_execution_order",
        "max_execution_order",
        "configuration_complexity_proxy",
        "total_complexity_count",
        "pre_execution_match_status",
    ]

    # Reconstruct the identity columns from the canonical key source.
    identity = summary_features[
        list(CONFIGURATION_KEY_COLUMNS)
    ].copy()

    identity["configuration_key"] = (
        _build_configuration_key(
            identity,
        )
    )

    result = identity.merge(
        result.drop(
            columns=list(CONFIGURATION_KEY_COLUMNS),
            errors="ignore",
        ),
        on="configuration_key",
        how="outer",
        validate="one_to_one",
    )

    result = result[
        [
            *CONFIGURATION_KEY_COLUMNS,
            "prevalidation_count",
            "transformation_count",
            "detailed_configuration_row_count",
            "unique_preval_trans_name_count",
            "unique_preval_trans_type_count",
            "unique_calling_type_count",
            "unique_execution_order_count",
            "min_execution_order",
            "max_execution_order",
            "configuration_complexity_proxy",
            "total_complexity_count",
            "pre_execution_match_status",
        ]
    ].sort_values(
        [
            "sprint_name",
            "loader_display_name",
            "datasetname",
        ],
        kind="mergesort",
    ).reset_index(
        drop=True,
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
    output = build_pre_execution_features()

    result = pd.read_csv(
        output,
    )

    print(
        "pre-execution feature build complete"
    )
    print(
        f"output={output}"
    )
    print(
        f"rows={len(result)}"
    )
    print(
        "match_status="
        + result[
            "pre_execution_match_status"
        ]
        .value_counts()
        .to_string()
    )