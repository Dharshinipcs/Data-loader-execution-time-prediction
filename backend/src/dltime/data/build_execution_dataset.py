from __future__ import annotations

from pathlib import Path

import pandas as pd

from dltime.data.data_quality import assess_execution_data_quality
from dltime.data.execution_reconstruction import reconstruct_executions
from dltime.data.excel_loader import load_excel
from dltime.data.raw_schemas import (
    LOADER_DEFINITIONS_1_COLUMNS,
    LOADER_EXECUTIONS_COLUMNS,
)
from dltime.data.schema import validate_required_columns
from dltime.data.target_quality import (
    classify_target_quality,
    get_valid_targets,
)


PROJECT_ROOT = Path(__file__).resolve().parents[4]

RAW_EXECUTIONS_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "LOADER_EXECTUIONS.xlsx"
)

RAW_DEFINITIONS_1_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "LOADER_DEFINATIONS_1.xlsx"
)

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

EXECUTION_LEVEL_OUTPUT_PATH = (
    PROCESSED_DIR
    / "execution_level.csv"
)

EXECUTION_DATA_QUALITY_OUTPUT_PATH = (
    PROCESSED_DIR
    / "execution_data_quality.csv"
)

TARGET_QUALITY_OUTPUT_PATH = (
    PROCESSED_DIR
    / "execution_target_quality.csv"
)

TRAINING_TARGET_OUTPUT_PATH = (
    PROCESSED_DIR
    / "execution_training_targets.csv"
)


def _normalize_identity(series: pd.Series) -> pd.Series:
    """
    Normalize business-key text for deterministic configuration matching.

    Matching is case-insensitive and whitespace-normalized.
    """
    return (
        series.astype("string")
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
        .str.upper()
    )


def _build_connection_mapping(
    definitions_1: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build the authoritative loader-connection mapping.

    Exact business key
        Sprint + Loader + Dataset

    maps to exactly one loader connection.

    A key with multiple distinct connections is marked ambiguous rather than
    silently choosing one.
    """
    required_columns = list(
        LOADER_DEFINITIONS_1_COLUMNS
    )

    validate_required_columns(
        definitions_1,
        required_columns,
    )

    mapping_source = definitions_1.copy()

    mapping_source["_sprint"] = _normalize_identity(
        mapping_source["sprint_name"]
    )

    mapping_source["_loader"] = _normalize_identity(
        mapping_source["loader_display_name"]
    )

    mapping_source["_dataset"] = _normalize_identity(
        mapping_source["datasetname"]
    )

    mapping_source["_connection"] = (
        mapping_source["ldr_connection_name"]
        .astype("string")
        .str.strip()
    )

    grouped = (
        mapping_source.groupby(
            [
                "_sprint",
                "_loader",
                "_dataset",
            ],
            dropna=False,
            sort=False,
        )["_connection"]
        .agg(
            lambda values: sorted(
                {
                    value
                    for value in values.dropna()
                    if str(value).strip()
                }
            )
        )
        .reset_index(name="connections")
    )

    grouped["connection_count"] = (
        grouped["connections"].apply(len)
    )

    grouped["exact_connection"] = (
        grouped["connections"].apply(
            lambda values: (
                values[0]
                if len(values) == 1
                else pd.NA
            )
        )
    )

    grouped["exact_mapping_status"] = (
        grouped["connection_count"].map(
            {
                0: "NO_CONFIGURATION",
                1: "RESOLVED",
            }
        )
    )

    grouped.loc[
        grouped["connection_count"] > 1,
        "exact_mapping_status",
    ] = "AMBIGUOUS"

    return grouped


def _build_loader_sprint_mapping(
    definitions_1: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build the secondary Sprint + Loader connection mapping.

    The fallback is safe only when exactly one distinct connection exists
    for the loader+sprint pair.
    """
    mapping_source = definitions_1.copy()

    mapping_source["_sprint"] = _normalize_identity(
        mapping_source["sprint_name"]
    )

    mapping_source["_loader"] = _normalize_identity(
        mapping_source["loader_display_name"]
    )

    mapping_source["_connection"] = (
        mapping_source["ldr_connection_name"]
        .astype("string")
        .str.strip()
    )

    grouped = (
        mapping_source.groupby(
            [
                "_sprint",
                "_loader",
            ],
            dropna=False,
            sort=False,
        )["_connection"]
        .agg(
            lambda values: sorted(
                {
                    value
                    for value in values.dropna()
                    if str(value).strip()
                }
            )
        )
        .reset_index(name="loader_connections")
    )

    grouped["loader_connection_count"] = (
        grouped["loader_connections"].apply(len)
    )

    grouped["loader_fallback_connection"] = (
        grouped["loader_connections"].apply(
            lambda values: (
                values[0]
                if len(values) == 1
                else pd.NA
            )
        )
    )

    return grouped


def _resolve_execution_connections(
    executions: pd.DataFrame,
    definitions_1: pd.DataFrame,
) -> pd.DataFrame:
    """
    Resolve loader connections before execution reconstruction.

    Resolution hierarchy
    --------------------
    1. Exact Sprint + Loader + Dataset match.
    2. If exact match is unavailable, use Sprint + Loader only when that
       pair has exactly one configured connection.
    3. Otherwise leave the connection unresolved.

    Ambiguous configuration is never silently resolved.
    """
    result = executions.copy()

    result["_sprint"] = _normalize_identity(
        result["Sprint"]
    )

    result["_loader"] = _normalize_identity(
        result["Loader_Name"]
    )

    result["_dataset"] = _normalize_identity(
        result["Dataset_display_name"]
    )

    exact_mapping = _build_connection_mapping(
        definitions_1,
    )

    loader_mapping = _build_loader_sprint_mapping(
        definitions_1,
    )

    result = result.merge(
        exact_mapping[
            [
                "_sprint",
                "_loader",
                "_dataset",
                "exact_connection",
                "exact_mapping_status",
            ]
        ],
        on=[
            "_sprint",
            "_loader",
            "_dataset",
        ],
        how="left",
        validate="many_to_one",
    )

    result = result.merge(
        loader_mapping[
            [
                "_sprint",
                "_loader",
                "loader_connection_count",
                "loader_fallback_connection",
            ]
        ],
        on=[
            "_sprint",
            "_loader",
        ],
        how="left",
        validate="many_to_one",
    )

    result["ldr_connection_name"] = (
        result["exact_connection"]
    )

    exact_resolved = (
        result["exact_connection"].notna()
    )

    fallback_resolved = (
        result["exact_connection"].isna()
        & result["loader_fallback_connection"].notna()
    )

    result.loc[
        fallback_resolved,
        "ldr_connection_name",
    ] = result.loc[
        fallback_resolved,
        "loader_fallback_connection",
    ]

    result["connection_resolution_method"] = (
        "UNRESOLVED"
    )

    result.loc[
        exact_resolved,
        "connection_resolution_method",
    ] = "EXACT_DATASET_MATCH"

    result.loc[
        fallback_resolved,
        "connection_resolution_method",
    ] = "LOADER_SPRINT_FALLBACK"

    result.loc[
        result["loader_connection_count"] > 1,
        "connection_resolution_method",
    ] = "AMBIGUOUS"

    result["ldr_connection_name"] = (
        result["ldr_connection_name"].astype("string")
    )

    ambiguous_fallback = (
        result["exact_connection"].isna()
        & (
            result["loader_connection_count"]
            > 1
        )
    )

    result.loc[
        ambiguous_fallback,
        "ldr_connection_name",
    ] = pd.NA

    result["connection_resolution_status"] = (
        result["connection_resolution_method"]
    )

    return result.drop(
        columns=[
            "_sprint",
            "_loader",
            "_dataset",
            "exact_connection",
            "exact_mapping_status",
            "loader_connection_count",
            "loader_fallback_connection",
            "connection_resolution_method",
        ]
    )


def _validate_execution_connection_consistency(
    execution_level: pd.DataFrame,
) -> None:
    """
    Validate that each reconstructed execution has at most one connection.

    Multiple non-null connections for one execution indicate a configuration
    conflict and must stop the dataset build.
    """
    if execution_level.empty:
        return

    connection_counts = (
        execution_level.groupby(
            "LDR_Execution_Id",
            dropna=False,
        )["ldr_connection_name"]
        .nunique(dropna=True)
    )

    conflicting = connection_counts[
        connection_counts > 1
    ]

    if not conflicting.empty:
        examples = conflicting.index.tolist()[
            :10
        ]

        raise ValueError(
            "Execution-level connection resolution produced multiple "
            "connections for execution IDs: "
            + ", ".join(
                str(value)
                for value in examples
            )
        )


def build_execution_dataset(
    raw_path: Path = RAW_EXECUTIONS_PATH,
    definitions_1_path: Path = RAW_DEFINITIONS_1_PATH,
    execution_output_path: Path = EXECUTION_LEVEL_OUTPUT_PATH,
    execution_data_quality_output_path: Path = (
        EXECUTION_DATA_QUALITY_OUTPUT_PATH
    ),
    target_quality_output_path: Path = (
        TARGET_QUALITY_OUTPUT_PATH
    ),
    training_target_output_path: Path = (
        TRAINING_TARGET_OUTPUT_PATH
    ),
) -> tuple[Path, Path, Path, Path]:
    """
    Build reproducible execution-level datasets from raw execution data.

    Pipeline
    --------
    Raw execution Excel
        -> schema validation
        -> authoritative connection resolution
        -> execution reconstruction
        -> general data-quality assessment
        -> target-quality classification
        -> valid training-target dataset

    Important separation
    --------------------
    General data quality and target quality have different responsibilities.

    General data quality:
        Assesses structural, temporal, workload, and stage-level health.
        Historical executions are retained and anomalies are flagged.

    Target quality:
        Determines whether a successful execution is suitable as a
        supervised-learning target.

    The raw source files are never modified.
    """
    dataframe = load_excel(
        raw_path,
    )

    validate_required_columns(
        dataframe,
        LOADER_EXECUTIONS_COLUMNS,
    )

    definitions_1 = load_excel(
        definitions_1_path,
    )

    validate_required_columns(
        definitions_1,
        LOADER_DEFINITIONS_1_COLUMNS,
    )

    enriched_execution_rows = (
        _resolve_execution_connections(
            dataframe,
            definitions_1,
        )
    )

    execution_level = reconstruct_executions(
        enriched_execution_rows,
    )

    _validate_execution_connection_consistency(
        execution_level,
    )

    # --------------------------------------------------------------
    # General historical data-quality assessment.
    #
    # This does NOT remove rows. Legitimate extreme executions remain
    # available for historical analysis while anomalies are flagged.
    # --------------------------------------------------------------
    execution_data_quality = (
        assess_execution_data_quality(
            execution_level,
        )
    )

    # --------------------------------------------------------------
    # Supervised-learning target quality.
    #
    # This is intentionally separate from general DQ.
    # --------------------------------------------------------------
    target_quality = classify_target_quality(
        execution_level,
    )

    training_targets = get_valid_targets(
        execution_level,
    )

    execution_output_path = Path(
        execution_output_path
    )

    execution_data_quality_output_path = Path(
        execution_data_quality_output_path
    )

    target_quality_output_path = Path(
        target_quality_output_path
    )

    training_target_output_path = Path(
        training_target_output_path
    )

    for output_path in (
        execution_output_path,
        execution_data_quality_output_path,
        target_quality_output_path,
        training_target_output_path,
    ):
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    execution_level.to_csv(
        execution_output_path,
        index=False,
    )

    execution_data_quality.to_csv(
        execution_data_quality_output_path,
        index=False,
    )

    target_quality.to_csv(
        target_quality_output_path,
        index=False,
    )

    training_targets.to_csv(
        training_target_output_path,
        index=False,
    )

    return (
        execution_output_path,
        execution_data_quality_output_path,
        target_quality_output_path,
        training_target_output_path,
    )


if __name__ == "__main__":
    (
        execution_output,
        execution_data_quality_output,
        target_quality_output,
        training_target_output,
    ) = build_execution_dataset()

    print("execution dataset build complete")
    print(
        f"execution_level_output={execution_output}"
    )
    print(
        "execution_data_quality_output="
        f"{execution_data_quality_output}"
    )
    print(
        f"target_quality_output={target_quality_output}"
    )
    print(
        f"training_target_output={training_target_output}"
    )