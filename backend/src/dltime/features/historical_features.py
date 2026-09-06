from __future__ import annotations

from pathlib import Path

import pandas as pd


class HistoricalFeatureError(Exception):
    """Raised when historical feature construction fails."""


PROJECT_ROOT = Path(__file__).resolve().parents[4]

TRAINING_TARGETS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "execution_training_targets.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "execution_historical_features.csv"
)

REQUIRED_COLUMNS: tuple[str, ...] = (
    "LDR_Execution_Id",
    "Loader_Name",
    "Start_Time",
    "timetaken_in_sec",
)


HISTORICAL_FEATURE_COLUMNS: tuple[str, ...] = (
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
)


def _validate_input(dataframe: pd.DataFrame) -> None:
    """Validate the minimum columns required for historical features."""

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise HistoricalFeatureError(
            "Missing columns required for historical feature construction: "
            + ", ".join(missing_columns)
        )

    if dataframe.empty:
        raise HistoricalFeatureError(
            "Cannot construct historical features from an empty dataset."
        )

    if dataframe["LDR_Execution_Id"].duplicated().any():
        duplicate_count = int(
            dataframe["LDR_Execution_Id"].duplicated().sum()
        )

        raise HistoricalFeatureError(
            "Historical feature construction requires one row per execution. "
            f"Found {duplicate_count} duplicate execution rows."
        )


def _safe_mean(values: list[float]) -> float | None:
    """Return the arithmetic mean or None when no history exists."""

    if not values:
        return None

    return float(sum(values) / len(values))


def _safe_median(values: list[float]) -> float | None:
    """Return the median or None when no history exists."""

    if not values:
        return None

    return float(pd.Series(values).median())


def _safe_std(values: list[float]) -> float | None:
    """
    Return sample standard deviation.

    Standard deviation is undefined for fewer than two observations, so
    None is returned until sufficient history exists.
    """

    if len(values) < 2:
        return None

    return float(pd.Series(values).std(ddof=1))


def _normalise_loader_name(loader_name: object) -> str | None:
    """
    Normalize loader identity for historical grouping.

    Missing or blank loader names are treated as unavailable rather than
    being converted into an artificial loader identity.
    """

    if pd.isna(loader_name):
        return None

    normalized = str(loader_name).strip()

    if not normalized:
        return None

    return normalized


def build_historical_features(
    dataframe: pd.DataFrame,
    output_path: Path | None = OUTPUT_PATH,
) -> pd.DataFrame:
    """
    Build leakage-safe historical execution features.

    Historical information is strictly causal:

    * Only executions with Start_Time strictly earlier than the current
      execution can contribute historical information.
    * Executions sharing the exact same Start_Time are treated as concurrent.
    * Concurrent executions cannot contribute their targets to one another.
    * Loader-specific history is calculated only when Loader_Name exists.
    * Missing loader identities receive global history only.
    * The current execution's target is added to history only after its
      features have been calculated.
    """

    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("dataframe must be a pandas DataFrame.")

    _validate_input(dataframe)

    result = dataframe.copy()

    result["Start_Time"] = pd.to_datetime(
        result["Start_Time"],
        errors="coerce",
    )

    result["timetaken_in_sec"] = pd.to_numeric(
        result["timetaken_in_sec"],
        errors="coerce",
    )

    invalid_start_time = result["Start_Time"].isna()

    if invalid_start_time.any():
        raise HistoricalFeatureError(
            "Historical feature construction requires a valid Start_Time "
            f"for every execution. Found {int(invalid_start_time.sum())} "
            "invalid timestamps."
        )

    invalid_target = (
        result["timetaken_in_sec"].isna()
        | (result["timetaken_in_sec"] <= 0)
    )

    if invalid_target.any():
        raise HistoricalFeatureError(
            "Historical feature construction requires a positive target "
            f"duration for every execution. Found {int(invalid_target.sum())} "
            "invalid targets."
        )

    result = result.sort_values(
        ["Start_Time", "LDR_Execution_Id"],
        kind="mergesort",
    ).reset_index(drop=True)

    global_history: list[float] = []
    loader_history: dict[str, list[float]] = {}
    loader_last_start: dict[str, pd.Timestamp] = {}

    feature_records: list[dict[str, object]] = []

    # Process one timestamp group at a time.
    #
    # This is important for strict temporal leakage prevention. Every
    # execution in the group must see the same history state: only
    # executions whose Start_Time is strictly earlier than this timestamp.
    for start_time, timestamp_group in result.groupby(
        "Start_Time",
        sort=True,
    ):
        pending_updates: list[tuple[str | None, float, pd.Timestamp]] = []

        # ---------------------------------------------------------------
        # Phase 1: calculate features for every execution at this timestamp
        # without updating historical state.
        # ---------------------------------------------------------------
        for row in timestamp_group.itertuples(index=False):
            execution_id = getattr(row, "LDR_Execution_Id")
            loader_name = getattr(row, "Loader_Name")
            target_duration = float(getattr(row, "timetaken_in_sec"))

            loader_key = _normalise_loader_name(loader_name)

            if global_history:
                global_mean = _safe_mean(global_history)
                global_median = _safe_median(global_history)
                global_std = _safe_std(global_history)
            else:
                global_mean = None
                global_median = None
                global_std = None

            if loader_key is None:
                current_loader_history: list[float] = []
                previous_loader_start = None
            else:
                current_loader_history = loader_history.get(
                    loader_key,
                    [],
                )

                previous_loader_start = loader_last_start.get(
                    loader_key,
                )

            if current_loader_history:
                loader_mean = _safe_mean(current_loader_history)
                loader_median = _safe_median(current_loader_history)
                loader_std = _safe_std(current_loader_history)
                previous_duration = float(current_loader_history[-1])
            else:
                loader_mean = None
                loader_median = None
                loader_std = None
                previous_duration = None

            if previous_loader_start is None:
                seconds_since_previous = None
            else:
                seconds_since_previous = float(
                    (
                        start_time
                        - previous_loader_start
                    ).total_seconds()
                )

            feature_records.append(
                {
                    "LDR_Execution_Id": execution_id,
                    "global_prior_execution_count": len(global_history),
                    "global_prior_mean_duration_sec": global_mean,
                    "global_prior_median_duration_sec": global_median,
                    "global_prior_std_duration_sec": global_std,
                    "loader_prior_execution_count": len(
                        current_loader_history
                    ),
                    "loader_prior_mean_duration_sec": loader_mean,
                    "loader_prior_median_duration_sec": loader_median,
                    "loader_prior_std_duration_sec": loader_std,
                    "loader_previous_duration_sec": previous_duration,
                    "seconds_since_loader_previous_execution": (
                        seconds_since_previous
                    ),
                }
            )

            pending_updates.append(
                (
                    loader_key,
                    target_duration,
                    start_time,
                )
            )

        # ---------------------------------------------------------------
        # Phase 2: only after ALL executions at this timestamp have had
        # their features calculated, update historical state.
        # ---------------------------------------------------------------
        for loader_key, target_duration, execution_start in pending_updates:
            global_history.append(target_duration)

            if loader_key is not None:
                loader_history.setdefault(
                    loader_key,
                    [],
                ).append(target_duration)

                loader_last_start[loader_key] = execution_start

    historical_features = pd.DataFrame(feature_records)

    result = result.merge(
        historical_features,
        on="LDR_Execution_Id",
        how="left",
        validate="one_to_one",
    )

    if output_path is not None:
        output_path = Path(output_path)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        result.to_csv(
            output_path,
            index=False,
        )

    return result


def build_historical_features_from_file(
    input_path: Path = TRAINING_TARGETS_PATH,
    output_path: Path = OUTPUT_PATH,
) -> Path:
    """
    Build historical features from the canonical training-target CSV.
    """

    input_path = Path(input_path)

    if not input_path.exists():
        raise HistoricalFeatureError(
            f"Training-target file not found: {input_path}"
        )

    dataframe = pd.read_csv(input_path)

    build_historical_features(
        dataframe,
        output_path=output_path,
    )

    return Path(output_path)


if __name__ == "__main__":
    output = build_historical_features_from_file()

    print("historical feature build complete")
    print(f"output={output}")