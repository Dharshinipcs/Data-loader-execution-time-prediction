from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd


class DataQualityError(Exception):
    """Raised when execution data-quality analysis cannot be completed."""


DQ_STATUS_GOOD: Final[str] = "GOOD"
DQ_STATUS_WARNING: Final[str] = "WARNING"
DQ_STATUS_INVALID: Final[str] = "INVALID"

DQ_SEVERITY_NONE: Final[str] = "NONE"
DQ_SEVERITY_LOW: Final[str] = "LOW"
DQ_SEVERITY_MEDIUM: Final[str] = "MEDIUM"
DQ_SEVERITY_HIGH: Final[str] = "HIGH"


STAGE_DURATION_COLUMNS: tuple[str, ...] = (
    "staging_timetaken_in_min",
    "prevaliadtion_timetaken_in_sec",
    "transformation_timetaken_in_sec",
    "data_loading_timetaken_in_sec",
    "data_loading_timetaken_in_sec_1",
    "datamart_approval_timetaken_in_sec",
    "dataloading_approval_timetaken_in_sec",
)

REQUIRED_COLUMNS: tuple[str, ...] = (
    "LDR_Execution_Id",
    "Loader_Name",
    "LDR_Status",
    "Sprint",
    "Start_Time",
    "End_Time",
    "timetaken_in_sec",
    "Total_Records",
    "Success_Records",
    "Error_Records",
    *STAGE_DURATION_COLUMNS,
)


def _validate_required_columns(
    dataframe: pd.DataFrame,
) -> None:
    """Validate columns required for data-quality analysis."""
    missing = [
        column
        for column in REQUIRED_COLUMNS
        if column not in dataframe.columns
    ]

    if missing:
        raise DataQualityError(
            "Missing columns required for data-quality analysis: "
            + ", ".join(missing)
        )


def _numeric(
    dataframe: pd.DataFrame,
    column: str,
) -> pd.Series:
    """Convert a column to numeric values, coercing invalid values to NaN."""
    return pd.to_numeric(
        dataframe[column],
        errors="coerce",
    )


def _append_flag(
    flags: pd.Series,
    mask: pd.Series,
    flag: str,
) -> pd.Series:
    """
    Append a semicolon-separated quality flag without duplicating it.

    Existing flags are preserved. The same flag is added at most once to
    each row.
    """
    result = flags.copy()

    normalized_mask = (
        mask.fillna(False)
        .astype(bool)
    )

    empty_mask = (
        normalized_mask
        & result.eq("")
    )

    append_mask = (
        normalized_mask
        & result.ne("")
        & ~result.str.split(";").apply(
            lambda values: flag in values
        )
    )

    result.loc[
        empty_mask,
    ] = flag

    result.loc[
        append_mask,
    ] = (
        result.loc[append_mask]
        + ";"
        + flag
    )

    return result


def _maximum_stage_duration_seconds(
    dataframe: pd.DataFrame,
) -> pd.Series:
    """
    Return the maximum observed stage duration in seconds.

    Staging is stored in minutes and is converted to seconds.

    Missing stage values remain missing and are ignored when taking the
    row-wise maximum.
    """
    stage_values = pd.DataFrame(
        {
            "staging": _numeric(
                dataframe,
                "staging_timetaken_in_min",
            )
            * 60,
            "prevalidation": _numeric(
                dataframe,
                "prevaliadtion_timetaken_in_sec",
            ),
            "transformation": _numeric(
                dataframe,
                "transformation_timetaken_in_sec",
            ),
            "data_loading": _numeric(
                dataframe,
                "data_loading_timetaken_in_sec",
            ),
            "data_loading_1": _numeric(
                dataframe,
                "data_loading_timetaken_in_sec_1",
            ),
            "datamart_approval": _numeric(
                dataframe,
                "datamart_approval_timetaken_in_sec",
            ),
            "dataloading_approval": _numeric(
                dataframe,
                "dataloading_approval_timetaken_in_sec",
            ),
        },
        index=dataframe.index,
    )

    return stage_values.max(
        axis=1,
        skipna=True,
    )


def assess_execution_data_quality(
    dataframe: pd.DataFrame,
    timestamp_tolerance_seconds: float = 60.0,
    extreme_quantile: float = 0.99,
) -> pd.DataFrame:
    """
    Assess structural, temporal, workload, and stage-level data quality.

    The original DataFrame is never modified.

    Important design rules
    ----------------------
    1. Historical executions are never deleted by this function.
    2. Extreme values are flagged, not automatically rejected.
    3. Missing stage durations are not automatically treated as errors because
       stage execution is path-dependent.
    4. Timestamp/duration disagreement is a quality anomaly.
    5. Total records must equal successful records plus error records.
    6. Stage durations must be non-negative when present.
    7. Successful target eligibility remains the responsibility of
       target_quality.py.
    """
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError(
            "dataframe must be a pandas DataFrame."
        )

    if timestamp_tolerance_seconds < 0:
        raise ValueError(
            "timestamp_tolerance_seconds must be non-negative."
        )

    if not 0.0 < extreme_quantile < 1.0:
        raise ValueError(
            "extreme_quantile must be between 0 and 1."
        )

    _validate_required_columns(
        dataframe,
    )

    result = dataframe.copy()

    result["dq_flags"] = ""
    result["dq_status"] = DQ_STATUS_GOOD
    result["dq_severity"] = DQ_SEVERITY_NONE

    result["dq_timestamp_duration_difference_sec"] = np.nan
    result["dq_record_balance_difference"] = np.nan
    result["dq_stage_count"] = 0
    result["dq_max_stage_duration_sec"] = np.nan

    # ------------------------------------------------------------------
    # Structural checks
    # ------------------------------------------------------------------

    duplicate_execution_ids = (
        result["LDR_Execution_Id"]
        .duplicated(keep=False)
    )

    result["dq_duplicate_execution_id"] = (
        duplicate_execution_ids
    )

    result["dq_flags"] = _append_flag(
        result["dq_flags"],
        duplicate_execution_ids,
        "DUPLICATE_EXECUTION_ID",
    )

    missing_execution_id = (
        result["LDR_Execution_Id"].isna()
    )

    result["dq_missing_execution_id"] = (
        missing_execution_id
    )

    result["dq_flags"] = _append_flag(
        result["dq_flags"],
        missing_execution_id,
        "MISSING_EXECUTION_ID",
    )

    missing_loader_identity = (
        result["Loader_Name"].isna()
        | result["Sprint"].isna()
    )

    result["dq_missing_loader_identity"] = (
        missing_loader_identity
    )

    result["dq_flags"] = _append_flag(
        result["dq_flags"],
        missing_loader_identity,
        "MISSING_LOADER_IDENTITY",
    )

    # ------------------------------------------------------------------
    # Timestamp checks
    # ------------------------------------------------------------------

    start = pd.to_datetime(
        result["Start_Time"],
        errors="coerce",
    )

    end = pd.to_datetime(
        result["End_Time"],
        errors="coerce",
    )

    duration = _numeric(
        result,
        "timetaken_in_sec",
    )

    timestamp_duration = (
        end - start
    ).dt.total_seconds()

    timestamp_difference = (
        duration - timestamp_duration
    ).abs()

    result["dq_timestamp_duration_difference_sec"] = (
        timestamp_difference
    )

    missing_start = start.isna()
    missing_end = end.isna()

    result["dq_missing_start_time"] = missing_start
    result["dq_missing_end_time"] = missing_end

    result["dq_flags"] = _append_flag(
        result["dq_flags"],
        missing_start,
        "MISSING_START_TIME",
    )

    result["dq_flags"] = _append_flag(
        result["dq_flags"],
        missing_end,
        "MISSING_END_TIME",
    )

    end_before_start = (
        start.notna()
        & end.notna()
        & (end < start)
    )

    result["dq_end_before_start"] = (
        end_before_start
    )

    result["dq_flags"] = _append_flag(
        result["dq_flags"],
        end_before_start,
        "END_BEFORE_START",
    )

    timestamp_mismatch = (
        start.notna()
        & end.notna()
        & duration.notna()
        & (
            timestamp_difference
            > timestamp_tolerance_seconds
        )
    )

    result["dq_timestamp_mismatch"] = (
        timestamp_mismatch
    )

    result["dq_flags"] = _append_flag(
        result["dq_flags"],
        timestamp_mismatch,
        "TIMESTAMP_DURATION_MISMATCH",
    )

    # ------------------------------------------------------------------
    # Total-duration checks
    # ------------------------------------------------------------------

    missing_duration = duration.isna()

    non_positive_duration = (
        duration.notna()
        & (duration <= 0)
    )

    result["dq_missing_duration"] = (
        missing_duration
    )

    result["dq_non_positive_duration"] = (
        non_positive_duration
    )

    result["dq_flags"] = _append_flag(
        result["dq_flags"],
        missing_duration,
        "MISSING_TOTAL_DURATION",
    )

    result["dq_flags"] = _append_flag(
        result["dq_flags"],
        non_positive_duration,
        "NON_POSITIVE_TOTAL_DURATION",
    )

    # ------------------------------------------------------------------
    # Workload checks
    # ------------------------------------------------------------------

    total_records = _numeric(
        result,
        "Total_Records",
    )

    success_records = _numeric(
        result,
        "Success_Records",
    )

    error_records = _numeric(
        result,
        "Error_Records",
    )

    result["dq_record_balance_difference"] = (
        total_records
        - success_records
        - error_records
    ).abs()

    missing_workload = (
        total_records.isna()
        | success_records.isna()
        | error_records.isna()
    )

    negative_workload = (
        (total_records < 0)
        | (success_records < 0)
        | (error_records < 0)
    )

    workload_balance_mismatch = (
        total_records.notna()
        & success_records.notna()
        & error_records.notna()
        & (
            result["dq_record_balance_difference"]
            > 0
        )
    )

    result["dq_missing_workload"] = (
        missing_workload
    )

    result["dq_negative_workload"] = (
        negative_workload
    )

    result["dq_workload_balance_mismatch"] = (
        workload_balance_mismatch
    )

    result["dq_flags"] = _append_flag(
        result["dq_flags"],
        missing_workload,
        "MISSING_WORKLOAD_COUNT",
    )

    result["dq_flags"] = _append_flag(
        result["dq_flags"],
        negative_workload,
        "NEGATIVE_WORKLOAD_COUNT",
    )

    result["dq_flags"] = _append_flag(
        result["dq_flags"],
        workload_balance_mismatch,
        "WORKLOAD_BALANCE_MISMATCH",
    )

    # ------------------------------------------------------------------
    # Stage checks
    # ------------------------------------------------------------------

    stage_numeric = pd.DataFrame(
        {
            column: _numeric(
                result,
                column,
            )
            for column in STAGE_DURATION_COLUMNS
        },
        index=result.index,
    )

    result["dq_stage_count"] = (
        stage_numeric.notna().sum(axis=1)
    )

    result["dq_max_stage_duration_sec"] = (
        _maximum_stage_duration_seconds(
            result,
        )
    )

    negative_stage_mask = (
        stage_numeric.lt(0)
        .any(axis=1)
    )

    # NaN means the stage duration was not observed.
    # It is path-dependent and is NOT a non-finite anomaly.
    present_stage_values = stage_numeric.notna()

    non_finite_stage_values = (
        ~np.isfinite(stage_numeric)
        & present_stage_values
    )

    non_finite_stage_mask = (
        non_finite_stage_values.any(axis=1)
    )

    result["dq_negative_stage_duration"] = (
        negative_stage_mask
    )

    result["dq_non_finite_stage_duration"] = (
        non_finite_stage_mask
    )

    result["dq_flags"] = _append_flag(
        result["dq_flags"],
        negative_stage_mask,
        "NEGATIVE_STAGE_DURATION",
    )

    result["dq_flags"] = _append_flag(
        result["dq_flags"],
        non_finite_stage_mask,
        "NON_FINITE_STAGE_DURATION",
    )

    # ------------------------------------------------------------------
    # Statistical extreme-value flags
    # ------------------------------------------------------------------

    duration_positive = duration[
        duration > 0
    ]

    duration_extreme_threshold = (
        duration_positive.quantile(
            extreme_quantile,
        )
        if not duration_positive.empty
        else np.nan
    )

    extreme_total_duration = (
        duration.notna()
        & pd.notna(
            duration_extreme_threshold,
        )
        & (
            duration
            > duration_extreme_threshold
        )
    )

    result["dq_extreme_total_duration"] = (
        extreme_total_duration
    )

    result["dq_total_duration_extreme_threshold_sec"] = (
        duration_extreme_threshold
    )

    result["dq_flags"] = _append_flag(
        result["dq_flags"],
        extreme_total_duration,
        "EXTREME_TOTAL_DURATION",
    )

    stage_extreme_mask = pd.Series(
        False,
        index=result.index,
    )

    for column in STAGE_DURATION_COLUMNS:
        values = stage_numeric[column]

        if column == "staging_timetaken_in_min":
            values = values * 60

        positive_values = values[
            values > 0
        ]

        if positive_values.empty:
            continue

        threshold = positive_values.quantile(
            extreme_quantile,
        )

        stage_extreme_mask = (
            stage_extreme_mask
            | (
                values.notna()
                & (values > threshold)
            )
        )

    result["dq_extreme_stage_duration"] = (
        stage_extreme_mask
    )

    result["dq_flags"] = _append_flag(
        result["dq_flags"],
        stage_extreme_mask,
        "EXTREME_STAGE_DURATION",
    )

    # ------------------------------------------------------------------
    # Known timestamp-corruption pattern
    # ------------------------------------------------------------------

    known_test_map_pattern = (
        result["Loader_Name"]
        .astype("string")
        .str.strip()
        .str.upper()
        .eq("TEST_MAP_CHECK_2")
        & timestamp_mismatch
    )

    result["dq_known_timestamp_corruption"] = (
        known_test_map_pattern
    )

    result["dq_flags"] = _append_flag(
        result["dq_flags"],
        known_test_map_pattern,
        "KNOWN_TIMESTAMP_CORRUPTION",
    )

    # ------------------------------------------------------------------
    # Severity and overall status
    # ------------------------------------------------------------------

    high_severity = (
        missing_execution_id
        | duplicate_execution_ids
        | end_before_start
        | timestamp_mismatch
        | non_positive_duration
        | negative_workload
        | workload_balance_mismatch
        | negative_stage_mask
        | non_finite_stage_mask
    )

    medium_severity = (
        missing_loader_identity
        | missing_start
        | missing_end
        | missing_duration
        | missing_workload
        | extreme_total_duration
        | stage_extreme_mask
    )

    result.loc[
        medium_severity,
        "dq_severity",
    ] = DQ_SEVERITY_MEDIUM

    result.loc[
        high_severity,
        "dq_severity",
    ] = DQ_SEVERITY_HIGH

    result.loc[
        medium_severity,
        "dq_status",
    ] = DQ_STATUS_WARNING

    result.loc[
        high_severity,
        "dq_status",
    ] = DQ_STATUS_INVALID

    # Known timestamp corruption is explicitly invalid for training
    # purposes but remains available for historical analytics.
    result.loc[
        known_test_map_pattern,
        "dq_status",
    ] = DQ_STATUS_INVALID

    result.loc[
        known_test_map_pattern,
        "dq_severity",
    ] = DQ_SEVERITY_HIGH

    result["dq_flag_count"] = result[
        "dq_flags"
    ].apply(
        lambda value: (
            0
            if not value
            else len(
                str(value).split(";")
            )
        )
    )

    result["dq_has_anomaly"] = (
        result["dq_flag_count"] > 0
    )

    return result.reset_index(
        drop=True,
    )


def get_data_quality_summary(
    quality_dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Return independent summaries for status and severity.

    Each row represents one category. The category_type column distinguishes
    status categories from severity categories.
    """
    required = {
        "dq_status",
        "dq_severity",
    }

    missing = required.difference(
        quality_dataframe.columns,
    )

    if missing:
        raise DataQualityError(
            "Quality DataFrame is missing required summary columns: "
            + ", ".join(sorted(missing))
        )

    status_summary = (
        quality_dataframe["dq_status"]
        .value_counts(dropna=False)
        .rename_axis("category")
        .reset_index(name="execution_count")
    )

    status_summary["category_type"] = "STATUS"

    severity_summary = (
        quality_dataframe["dq_severity"]
        .value_counts(dropna=False)
        .rename_axis("category")
        .reset_index(name="execution_count")
    )

    severity_summary["category_type"] = "SEVERITY"

    return pd.concat(
        [
            status_summary,
            severity_summary,
        ],
        ignore_index=True,
    )[
        [
            "category_type",
            "category",
            "execution_count",
        ]
    ]


def get_anomalous_executions(
    quality_dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Return executions with at least one quality flag.
    """
    if "dq_has_anomaly" not in quality_dataframe.columns:
        raise DataQualityError(
            "DataFrame must first be processed by "
            "assess_execution_data_quality()."
        )

    return quality_dataframe.loc[
        quality_dataframe["dq_has_anomaly"]
    ].reset_index(
        drop=True,
    )