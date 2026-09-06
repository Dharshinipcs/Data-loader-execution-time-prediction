from __future__ import annotations

import pandas as pd


class TargetQualityError(Exception):
    """Raised when target-quality analysis cannot be completed."""


TARGET_QUALITY_VALID = "VALID"
TARGET_QUALITY_INVALID_TARGET = "INVALID_TARGET"
TARGET_QUALITY_TIMESTAMP_MISMATCH = "TIMESTAMP_MISMATCH"
TARGET_QUALITY_STAGE_INCONSISTENCY = "STAGE_TARGET_INCONSISTENCY"


STAGE_DURATION_COLUMNS: tuple[str, ...] = (
    "staging_timetaken_in_min",
    "prevaliadtion_timetaken_in_sec",
    "transformation_timetaken_in_sec",
    "data_loading_timetaken_in_sec",
    "data_loading_timetaken_in_sec_1",
    "datamart_approval_timetaken_in_sec",
    "dataloading_approval_timetaken_in_sec",
)


def _validate_required_columns(dataframe: pd.DataFrame) -> None:
    """Validate columns required for target-quality analysis."""
    required_columns = {
        "LDR_Status",
        "Start_Time",
        "End_Time",
        "timetaken_in_sec",
        *STAGE_DURATION_COLUMNS,
    }

    missing_columns = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise TargetQualityError(
            "Missing columns required for target-quality analysis: "
            + ", ".join(sorted(missing_columns))
        )


def _calculate_known_stage_seconds(dataframe: pd.DataFrame) -> pd.Series:
    """
    Calculate the sum of available stage durations in seconds.

    Staging is stored in minutes, while all other stage-duration columns used
    here are stored in seconds.
    """
    known_stage_seconds = (
        dataframe["staging_timetaken_in_min"].fillna(0) * 60
        + dataframe["prevaliadtion_timetaken_in_sec"].fillna(0)
        + dataframe["transformation_timetaken_in_sec"].fillna(0)
        + dataframe["data_loading_timetaken_in_sec"].fillna(0)
        + dataframe["data_loading_timetaken_in_sec_1"].fillna(0)
        + dataframe["datamart_approval_timetaken_in_sec"].fillna(0)
        + dataframe["dataloading_approval_timetaken_in_sec"].fillna(0)
    )

    return known_stage_seconds


def classify_target_quality(
    dataframe: pd.DataFrame,
    timestamp_tolerance_seconds: float = 1.0,
    stage_ratio_threshold: float = 1000.0,
) -> pd.DataFrame:
    """
    Classify successful executions according to target quality.

    The returned DataFrame is a copy and the input DataFrame is never modified.

    Quality rules
    -------------
    1. Only successful executions are evaluated.
    2. Target duration must be positive and non-null.
    3. Stored total duration must agree with End_Time - Start_Time within
       the configured timestamp tolerance.
    4. If known stage durations exist, an extreme target-to-stage ratio is
       flagged as a stage/target inconsistency.

    Important
    ---------
    A large execution duration is not automatically considered invalid.
    Legitimate long-running executions are retained when their timing data
    is internally consistent.
    """
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("dataframe must be a pandas DataFrame.")

    if timestamp_tolerance_seconds < 0:
        raise ValueError("timestamp_tolerance_seconds must be non-negative.")

    if stage_ratio_threshold <= 0:
        raise ValueError("stage_ratio_threshold must be greater than zero.")

    _validate_required_columns(dataframe)

    result = dataframe.copy()

    result["Start_Time"] = pd.to_datetime(
        result["Start_Time"],
        errors="coerce",
    )
    result["End_Time"] = pd.to_datetime(
        result["End_Time"],
        errors="coerce",
    )

    successful_mask = (
        result["LDR_Status"]
        .astype(str)
        .str.upper()
        .eq("SUCCEED")
    )

    result = result.loc[successful_mask].copy()

    result["timestamp_duration_sec"] = (
        result["End_Time"] - result["Start_Time"]
    ).dt.total_seconds()

    result["known_stage_seconds"] = _calculate_known_stage_seconds(result)

    result["target_timestamp_difference_sec"] = (
        result["timetaken_in_sec"]
        - result["timestamp_duration_sec"]
    )

    result["target_stage_ratio"] = (
        result["timetaken_in_sec"]
        / result["known_stage_seconds"].replace(0, pd.NA)
    )

    result["target_quality_status"] = TARGET_QUALITY_VALID

    invalid_target_mask = (
        result["timetaken_in_sec"].isna()
        | (result["timetaken_in_sec"] <= 0)
    )

    result.loc[
        invalid_target_mask,
        "target_quality_status",
    ] = TARGET_QUALITY_INVALID_TARGET

    timestamp_mismatch_mask = (
        result["target_quality_status"].eq(TARGET_QUALITY_VALID)
        & (
            result["timestamp_duration_sec"].isna()
            | (
                result["target_timestamp_difference_sec"].abs()
                > timestamp_tolerance_seconds
            )
        )
    )

    result.loc[
        timestamp_mismatch_mask,
        "target_quality_status",
    ] = TARGET_QUALITY_TIMESTAMP_MISMATCH

    stage_inconsistency_mask = (
        result["target_quality_status"].eq(TARGET_QUALITY_VALID)
        & result["target_stage_ratio"].notna()
        & (result["target_stage_ratio"] > stage_ratio_threshold)
    )

    result.loc[
        stage_inconsistency_mask,
        "target_quality_status",
    ] = TARGET_QUALITY_STAGE_INCONSISTENCY

    return result.reset_index(drop=True)


def get_valid_targets(
    dataframe: pd.DataFrame,
    timestamp_tolerance_seconds: float = 1.0,
    stage_ratio_threshold: float = 1000.0,
) -> pd.DataFrame:
    """
    Return only successful executions with valid execution-time targets.
    """
    classified = classify_target_quality(
        dataframe,
        timestamp_tolerance_seconds=timestamp_tolerance_seconds,
        stage_ratio_threshold=stage_ratio_threshold,
    )

    return classified.loc[
        classified["target_quality_status"].eq(TARGET_QUALITY_VALID)
    ].reset_index(drop=True)


def get_target_quality_summary(
    classified_dataframe: pd.DataFrame,
) -> pd.Series:
    """
    Return the count of executions in each target-quality category.
    """
    if "target_quality_status" not in classified_dataframe.columns:
        raise TargetQualityError(
            "DataFrame must first be processed by classify_target_quality()."
        )

    return classified_dataframe["target_quality_status"].value_counts()
