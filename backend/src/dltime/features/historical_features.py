from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[4]
TRAINING_TARGETS_PATH = (
    PROJECT_ROOT / "data" / "processed" / "execution_training_targets.csv"
)
OUTPUT_PATH = (
    PROJECT_ROOT / "data" / "processed" / "execution_historical_features.csv"
)


REQUIRED_COLUMNS = (
    "LDR_Execution_Id",
    "Loader_Name",
    "Start_Time",
    "timetaken_in_sec",
)


OUTPUT_COLUMNS = (
    "LDR_Execution_Id",
    "Loader_Name",
    "Sprint",
    "Start_Time",
    "timetaken_in_sec",
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


class HistoricalFeatureError(Exception):
    """Raised when historical feature generation cannot be completed safely."""


def _normalise_loader_name(value: object) -> str | None:
    """
    Normalize loader identity consistently with the canonical project identity.

    Missing or blank values remain None.
    Non-missing values are stripped, internal whitespace is collapsed,
    and the result is upper-cased.
    """
    if value is None:
        return None

    if pd.isna(value):
        return None

    normalized = " ".join(str(value).strip().split())

    if not normalized:
        return None

    return normalized.upper()


def _safe_mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _safe_median(values: list[float]) -> float | None:
    if not values:
        return None

    series = pd.Series(values, dtype="float64")
    return float(series.median())


def _safe_std(values: list[float]) -> float | None:
    if len(values) < 2:
        return None

    series = pd.Series(values, dtype="float64")
    return float(series.std(ddof=1))


def _validate_input(dataframe: pd.DataFrame) -> None:
    missing = [
        column for column in REQUIRED_COLUMNS
        if column not in dataframe.columns
    ]

    if missing:
        raise HistoricalFeatureError(
            "Missing required columns: " + ", ".join(missing)
        )

    if dataframe.empty:
        raise HistoricalFeatureError(
            "Training target dataset is empty."
        )

    if dataframe["LDR_Execution_Id"].duplicated().any():
        duplicate_count = int(
            dataframe["LDR_Execution_Id"].duplicated().sum()
        )
        raise HistoricalFeatureError(
            f"Expected one row per execution, found {duplicate_count} "
            "duplicate execution rows."
        )


def _prepare_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    prepared = dataframe.copy()

    prepared["Loader_Name"] = prepared["Loader_Name"].map(
        _normalise_loader_name
    )

    prepared["Start_Time"] = pd.to_datetime(
        prepared["Start_Time"],
        errors="coerce",
    )

    if prepared["Start_Time"].isna().any():
        invalid_count = int(prepared["Start_Time"].isna().sum())
        raise HistoricalFeatureError(
            f"Found {invalid_count} rows with invalid Start_Time."
        )

    prepared["timetaken_in_sec"] = pd.to_numeric(
        prepared["timetaken_in_sec"],
        errors="coerce",
    )

    if prepared["timetaken_in_sec"].isna().any():
        invalid_count = int(prepared["timetaken_in_sec"].isna().sum())
        raise HistoricalFeatureError(
            f"Found {invalid_count} rows with invalid target duration."
        )

    if (prepared["timetaken_in_sec"] <= 0).any():
        invalid_count = int(
            (prepared["timetaken_in_sec"] <= 0).sum()
        )
        raise HistoricalFeatureError(
            f"Found {invalid_count} rows with non-positive target duration."
        )

    prepared = prepared.sort_values(
        ["Start_Time", "LDR_Execution_Id"],
        kind="mergesort",
    ).reset_index(drop=True)

    return prepared


def _build_historical_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    global_history: list[float] = []
    loader_history: dict[str, list[float]] = {}
    loader_previous: dict[str, tuple[pd.Timestamp, float]] = {}

    output_rows: list[dict[str, object]] = []

    index = 0

    while index < len(dataframe):
        timestamp = dataframe.iloc[index]["Start_Time"]

        same_time_indices: list[int] = []

        while (
            index + len(same_time_indices) < len(dataframe)
            and dataframe.iloc[index + len(same_time_indices)]["Start_Time"]
            == timestamp
        ):
            same_time_indices.append(index + len(same_time_indices))

        # Phase 1:
        # Calculate historical features for every execution at this timestamp
        # BEFORE adding any same-timestamp execution to history.
        pending_history_updates: list[tuple[int, str | None, float]] = []

        for row_index in same_time_indices:
            row = dataframe.iloc[row_index]

            loader_name = row["Loader_Name"]
            duration = float(row["timetaken_in_sec"])

            global_values = list(global_history)

            if loader_name is None:
                loader_values: list[float] = []
                previous_record = None
            else:
                loader_values = list(
                    loader_history.get(loader_name, [])
                )
                previous_record = loader_previous.get(loader_name)

            output_rows.append(
                {
                    "LDR_Execution_Id": row["LDR_Execution_Id"],
                    "Loader_Name": loader_name,
                    "Sprint": row["Sprint"]
                    if "Sprint" in dataframe.columns
                    else None,
                    "Start_Time": timestamp,
                    "timetaken_in_sec": duration,
                    "global_prior_execution_count": len(global_values),
                    "global_prior_mean_duration_sec": _safe_mean(
                        global_values
                    ),
                    "global_prior_median_duration_sec": _safe_median(
                        global_values
                    ),
                    "global_prior_std_duration_sec": _safe_std(
                        global_values
                    ),
                    "loader_prior_execution_count": len(loader_values),
                    "loader_prior_mean_duration_sec": _safe_mean(
                        loader_values
                    ),
                    "loader_prior_median_duration_sec": _safe_median(
                        loader_values
                    ),
                    "loader_prior_std_duration_sec": _safe_std(
                        loader_values
                    ),
                    "loader_previous_duration_sec": (
                        previous_record[1]
                        if previous_record is not None
                        else None
                    ),
                    "seconds_since_loader_previous": (
                        float(
                            (
                                timestamp - previous_record[0]
                            ).total_seconds()
                        )
                        if previous_record is not None
                        else None
                    ),
                }
            )

            pending_history_updates.append(
                (row_index, loader_name, duration)
            )

        # Phase 2:
        # Only after all same-timestamp features are calculated do these
        # executions become available to later executions.
        for _, loader_name, duration in pending_history_updates:
            global_history.append(duration)

            if loader_name is not None:
                loader_history.setdefault(loader_name, []).append(
                    duration
                )
                loader_previous[loader_name] = (
                    timestamp,
                    duration,
                )

        index += len(same_time_indices)

    result = pd.DataFrame(output_rows)

    return result[list(OUTPUT_COLUMNS)]


def build_historical_features(
    input_path: str | Path = TRAINING_TARGETS_PATH,
    output_path: str | Path = OUTPUT_PATH,
) -> Path:
    input_file = Path(input_path)
    output_file = Path(output_path)

    if not input_file.exists():
        raise HistoricalFeatureError(
            f"Training target file not found: {input_file}"
        )

    dataframe = pd.read_csv(input_file)

    _validate_input(dataframe)

    prepared = _prepare_dataframe(dataframe)

    features = _build_historical_features(prepared)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(output_file, index=False)

    return output_file


if __name__ == "__main__":
    output = build_historical_features()

    print("historical feature build complete")
    print(f"output={output}")
    print(f"rows={len(pd.read_csv(output))}")
