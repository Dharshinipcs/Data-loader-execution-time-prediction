from __future__ import annotations

from dataclasses import dataclass
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


CONFIGURATION_FEATURE_COLUMNS: tuple[str, ...] = (
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


@dataclass(frozen=True)
class OnlineConfigurationRequest:
    """Pre-execution configuration identity supplied by the caller."""

    sprint: str
    loader_name: str
    loader_connection_name: str
    dataset_name: str


class OnlineConfigurationFeatureError(Exception):
    """Raised when online configuration features cannot be built."""


def _normalise_text(value: object) -> str | None:
    """Normalize configuration identity text consistently."""
    if pd.isna(value):
        return None

    normalized = str(value).strip().upper()

    return normalized or None


def _require_identity(
    request: OnlineConfigurationRequest,
) -> None:
    """Validate that all required pre-execution identity fields exist."""
    values = {
        "sprint": request.sprint,
        "loader_name": request.loader_name,
        "loader_connection_name": request.loader_connection_name,
        "dataset_name": request.dataset_name,
    }

    missing = [
        name
        for name, value in values.items()
        if _normalise_text(value) is None
    ]

    if missing:
        raise OnlineConfigurationFeatureError(
            "Missing required pre-execution configuration identity: "
            + ", ".join(missing)
        )


def _prepare_detail_definitions(
    definitions: pd.DataFrame,
) -> pd.DataFrame:
    """Prepare detailed loader configuration records."""
    validate_required_columns(
        definitions,
        list(LOADER_DEFINITIONS_COLUMNS),
    )

    detail = definitions.copy()

    identity_columns = [
        "sprint_name",
        "ldr_connection_name",
        "loader_display_name",
        "datasetname",
    ]

    for column in identity_columns:
        detail[column] = detail[column].map(_normalise_text)

    detail["execution_order"] = pd.to_numeric(
        detail["execution_order"],
        errors="coerce",
    )

    detail["type"] = detail["type"].map(_normalise_text)

    detail["calling_type"] = detail["calling_type"].map(
        _normalise_text
    )

    detail["preval_trans_display_name"] = detail[
        "preval_trans_display_name"
    ].map(_normalise_text)

    return detail


def _prepare_summary_definitions(
    definitions_1: pd.DataFrame,
) -> pd.DataFrame:
    """Prepare primary loader configuration summary records."""
    validate_required_columns(
        definitions_1,
        list(LOADER_DEFINITIONS_1_COLUMNS),
    )

    summary = definitions_1.copy()

    identity_columns = [
        "sprint_name",
        "ldr_connection_name",
        "loader_display_name",
        "datasetname",
    ]

    for column in identity_columns:
        summary[column] = summary[column].map(_normalise_text)

    summary["prevalidation_count"] = pd.to_numeric(
        summary["prevalidation_count"],
        errors="coerce",
    )

    summary["transformation_count"] = pd.to_numeric(
        summary["transformation_count"],
        errors="coerce",
    )

    duplicate_mask = summary.duplicated(
        subset=identity_columns,
        keep=False,
    )

    if duplicate_mask.any():
        duplicate_count = int(duplicate_mask.sum())

        raise OnlineConfigurationFeatureError(
            "LOADER_DEFINATIONS_1.xlsx contains duplicate "
            "configuration keys: "
            f"{duplicate_count} rows."
        )

    return summary


def _build_detail_lookup(
    definitions: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate detailed configuration by canonical dataset identity."""
    grouped = (
        definitions.groupby(
            [
                "sprint_name",
                "ldr_connection_name",
                "loader_display_name",
                "datasetname",
            ],
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
            + grouped["unique_preval_trans_type_count"].fillna(0)
            + grouped["unique_calling_type_count"].fillna(0)
        )
    )

    return grouped


def _build_configuration_dataset(
    definitions: pd.DataFrame,
    definitions_1: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build one row per configured dataset.

    This is the same dataset-level configuration foundation used by the
    offline pre-execution feature engineering pipeline.
    """
    detail = _prepare_detail_definitions(definitions)
    summary = _prepare_summary_definitions(definitions_1)

    detail_lookup = _build_detail_lookup(detail)

    identity_columns = [
        "sprint_name",
        "ldr_connection_name",
        "loader_display_name",
        "datasetname",
    ]

    result = summary[
        [
            *identity_columns,
            "prevalidation_count",
            "transformation_count",
        ]
    ].merge(
        detail_lookup[
            [
                *identity_columns,
                "detailed_configuration_row_count",
                "unique_preval_trans_name_count",
                "unique_preval_trans_type_count",
                "unique_calling_type_count",
                "unique_execution_order_count",
                "min_execution_order",
                "max_execution_order",
                "configuration_complexity_proxy",
            ]
        ],
        on=identity_columns,
        how="outer",
        validate="one_to_one",
    )

    result["total_complexity_count"] = (
        result["prevalidation_count"]
        + result["transformation_count"]
    )

    result["execution_order_span"] = (
        result["max_execution_order"]
        - result["min_execution_order"]
    )

    result["max_dataset_complexity"] = (
        result["configuration_complexity_proxy"]
    )

    return result


def _build_loader_scope(
    configuration: pd.DataFrame,
    request: OnlineConfigurationRequest,
) -> pd.DataFrame:
    """Return all configured datasets belonging to the requested loader."""
    sprint = _normalise_text(request.sprint)
    loader_name = _normalise_text(request.loader_name)
    connection = _normalise_text(
        request.loader_connection_name
    )

    return configuration[
        (configuration["sprint_name"] == sprint)
        & (configuration["loader_display_name"] == loader_name)
        & (configuration["ldr_connection_name"] == connection)
    ].copy()


def _build_features(
    configuration: pd.DataFrame,
    request: OnlineConfigurationRequest,
) -> dict[str, float | int | str]:
    """
    Build online configuration features for the requested dataset.

    The caller supplies one dataset identity. We first identify the complete
    configured dataset scope for the loader, then match the requested dataset
    inside that scope.

    Numeric configuration values come only from the matched dataset because
    those are the configuration attributes actually identified by the
    pre-execution request.
    """
    scope = _build_loader_scope(
        configuration,
        request,
    )

    configured_dataset_count = int(
        scope["datasetname"].nunique()
    )

    requested_dataset = _normalise_text(
        request.dataset_name
    )

    matched = scope[
        scope["datasetname"] == requested_dataset
    ].copy()

    matched_configured_dataset_count = int(
        matched["datasetname"].nunique()
    )

    if configured_dataset_count > 0:
        coverage_pct = (
            matched_configured_dataset_count
            / configured_dataset_count
            * 100.0
        )
    else:
        coverage_pct = 0.0

    if configured_dataset_count == 0:
        coverage_status = "NONE"
    elif matched_configured_dataset_count == 0:
        coverage_status = "NONE"
    elif matched_configured_dataset_count < configured_dataset_count:
        coverage_status = "PARTIAL"
    else:
        coverage_status = "FULL"

    def _sum_numeric(
        frame: pd.DataFrame,
        column: str,
    ) -> float:
        values = pd.to_numeric(
            frame[column],
            errors="coerce",
        )

        if values.notna().any():
            return float(values.sum(min_count=1))

        return float("nan")

    def _min_numeric(
        frame: pd.DataFrame,
        column: str,
    ) -> float:
        values = pd.to_numeric(
            frame[column],
            errors="coerce",
        )

        if values.notna().any():
            return float(values.min())

        return float("nan")

    def _max_numeric(
        frame: pd.DataFrame,
        column: str,
    ) -> float:
        values = pd.to_numeric(
            frame[column],
            errors="coerce",
        )

        if values.notna().any():
            return float(values.max())

        return float("nan")

    feature_source = matched

    if feature_source.empty:
        nan_features = {
            column: float("nan")
            for column in CONFIGURATION_FEATURE_COLUMNS
            if column
            not in {
                "configured_dataset_count",
                "matched_configured_dataset_count",
                "configuration_coverage_pct",
            }
        }

        return {
            "configured_dataset_count": configured_dataset_count,
            "matched_configured_dataset_count": (
                matched_configured_dataset_count
            ),
            "configuration_coverage_pct": coverage_pct,
            **nan_features,
            "configuration_coverage_status": coverage_status,
        }

    configured_min_execution_order = _min_numeric(
        feature_source,
        "min_execution_order",
    )

    configured_max_execution_order = _max_numeric(
        feature_source,
        "max_execution_order",
    )

    return {
        "configured_dataset_count": configured_dataset_count,
        "matched_configured_dataset_count": (
            matched_configured_dataset_count
        ),
        "configuration_coverage_pct": coverage_pct,
        "configured_total_complexity_count": _sum_numeric(
            feature_source,
            "total_complexity_count",
        ),
        "configured_prevalidation_count": _sum_numeric(
            feature_source,
            "prevalidation_count",
        ),
        "configured_transformation_count": _sum_numeric(
            feature_source,
            "transformation_count",
        ),
        "configured_detail_row_count": _sum_numeric(
            feature_source,
            "detailed_configuration_row_count",
        ),
        "configured_unique_preval_trans_name_count": _sum_numeric(
            feature_source,
            "unique_preval_trans_name_count",
        ),
        "configured_unique_preval_trans_type_count": _sum_numeric(
            feature_source,
            "unique_preval_trans_type_count",
        ),
        "configured_unique_calling_type_count": _sum_numeric(
            feature_source,
            "unique_calling_type_count",
        ),
        "configured_execution_order_count": _sum_numeric(
            feature_source,
            "unique_execution_order_count",
        ),
        "configured_min_execution_order": (
            configured_min_execution_order
        ),
        "configured_max_execution_order": (
            configured_max_execution_order
        ),
        "configured_execution_order_span": (
            configured_max_execution_order
            - configured_min_execution_order
        ),
        "configured_max_dataset_complexity": _max_numeric(
            feature_source,
            "max_dataset_complexity",
        ),
        "configuration_coverage_status": coverage_status,
    }


def build_online_configuration_features(
    request: OnlineConfigurationRequest,
    definitions_path: Path = LOADER_DEFINITIONS_PATH,
    definitions_1_path: Path = LOADER_DEFINITIONS_1_PATH,
) -> dict[str, float | int | str]:
    """
    Build configuration features available before execution starts.

    No execution outcome, timestamp, actual duration, row-count result,
    status, or post-execution information is used.
    """
    _require_identity(request)

    definitions_path = Path(definitions_path)
    definitions_1_path = Path(definitions_1_path)

    if not definitions_path.exists():
        raise OnlineConfigurationFeatureError(
            "LOADER_DEFINATIONS.xlsx not found: "
            f"{definitions_path}"
        )

    if not definitions_1_path.exists():
        raise OnlineConfigurationFeatureError(
            "LOADER_DEFINATIONS_1.xlsx not found: "
            f"{definitions_1_path}"
        )

    definitions = load_excel(definitions_path)
    definitions_1 = load_excel(definitions_1_path)

    validate_required_columns(
        definitions,
        list(LOADER_DEFINITIONS_COLUMNS),
    )

    validate_required_columns(
        definitions_1,
        list(LOADER_DEFINITIONS_1_COLUMNS),
    )

    configuration = _build_configuration_dataset(
        definitions,
        definitions_1,
    )

    return _build_features(
        configuration,
        request,
    )


if __name__ == "__main__":
    sample_request = OnlineConfigurationRequest(
        sprint="DEV",
        loader_name="2902_LATEST_LOADER",
        loader_connection_name="DEV_DM",
        dataset_name="2902_LATEST_LOADER",
    )

    features = build_online_configuration_features(
        sample_request,
    )

    print("online configuration feature build complete")

    for column in [
        *CONFIGURATION_FEATURE_COLUMNS,
        "configuration_coverage_status",
    ]:
        print(f"{column}={features.get(column)}")