from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from dltime.data.feedback_store import (
    ExecutionFeedback,
    SQLiteFeedbackStore,
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
    "seconds_since_loader_previous",
)


@dataclass(frozen=True)
class OnlineHistoricalFeatures:
    global_prior_execution_count: int
    global_prior_mean_duration_sec: float | None
    global_prior_median_duration_sec: float | None
    global_prior_std_duration_sec: float | None
    loader_prior_execution_count: int
    loader_prior_mean_duration_sec: float | None
    loader_prior_median_duration_sec: float | None
    loader_prior_std_duration_sec: float | None
    loader_previous_duration_sec: float | None
    seconds_since_loader_previous: float | None

    def as_dict(
        self,
    ) -> dict[str, float | int | None]:
        """Return historical features using canonical feature names."""
        return {
            column: getattr(self, column)
            for column in HISTORICAL_FEATURE_COLUMNS
        }


class OnlineHistoricalFeatureError(Exception):
    """Raised when online historical features cannot be built safely."""


def _normalise_loader_name(
    value: object,
) -> str | None:
    """
    Normalize loader identity consistently with offline historical features.

    Missing or blank values remain None.
    Non-missing values are stripped, internal whitespace is collapsed,
    and the result is upper-cased.
    """
    if value is None:
        return None

    if pd.isna(value):
        return None

    normalized = " ".join(
        str(value).strip().split()
    )

    if not normalized:
        return None

    return normalized.upper()


def _safe_mean(
    values: list[float],
) -> float | None:
    if not values:
        return None

    return float(
        sum(values) / len(values)
    )


def _safe_median(
    values: list[float],
) -> float | None:
    if not values:
        return None

    return float(
        pd.Series(
            values,
            dtype="float64",
        ).median()
    )


def _safe_std(
    values: list[float],
) -> float | None:
    if len(values) < 2:
        return None

    return float(
        pd.Series(
            values,
            dtype="float64",
        ).std(
            ddof=1,
        )
    )


def _parse_prediction_timestamp(
    value: object,
) -> pd.Timestamp:
    timestamp = pd.to_datetime(
        value,
        errors="coerce",
    )

    if pd.isna(timestamp):
        raise OnlineHistoricalFeatureError(
            f"Invalid prediction timestamp: {value!r}"
        )

    return timestamp


class OnlineHistoricalFeatureProvider:
    """
    Build leakage-safe historical features from persisted feedback.

    Only training-eligible successful executions with positive actual
    durations are considered.

    A feedback record is usable only when its prediction timestamp is
    strictly earlier than the current prediction timestamp.
    """

    def __init__(
        self,
        store: SQLiteFeedbackStore,
    ) -> None:
        self.store = store

    def _eligible_records(
        self,
        prediction_timestamp: pd.Timestamp,
    ) -> list[ExecutionFeedback]:
        records = self.store.all_records()

        eligible: list[ExecutionFeedback] = []

        for record in records:
            if not record.training_eligible:
                continue

            if (
                record.execution_status is None
                or record.execution_status.upper() != "SUCCEED"
            ):
                continue

            if record.actual_total_seconds is None:
                continue

            if record.actual_total_seconds <= 0:
                continue

            if record.prediction_timestamp is None:
                continue

            try:
                record_timestamp = (
                    _parse_prediction_timestamp(
                        record.prediction_timestamp
                    )
                )
            except OnlineHistoricalFeatureError:
                continue

            # Strictly earlier only: prevents same-time leakage.
            if record_timestamp >= prediction_timestamp:
                continue

            eligible.append(record)

        eligible.sort(
            key=lambda record: (
                _parse_prediction_timestamp(
                    record.prediction_timestamp
                ),
                str(record.execution_id),
            )
        )

        return eligible

    def get_features(
        self,
        *,
        loader_name: object,
        prediction_timestamp: object,
    ) -> OnlineHistoricalFeatures:
        """
        Build leakage-safe historical features for one prediction request.
        """
        return self.build(
            loader_name=loader_name,
            prediction_timestamp=prediction_timestamp,
        )

    def build(
        self,
        loader_name: object,
        prediction_timestamp: object,
    ) -> OnlineHistoricalFeatures:
        current_timestamp = (
            _parse_prediction_timestamp(
                prediction_timestamp
            )
        )

        normalized_loader = (
            _normalise_loader_name(
                loader_name
            )
        )

        records = self._eligible_records(
            current_timestamp
        )

        global_durations: list[float] = []
        loader_durations: list[float] = []

        loader_previous_timestamp: (
            pd.Timestamp | None
        ) = None

        loader_previous_duration: (
            float | None
        ) = None

        for record in records:
            duration = float(
                record.actual_total_seconds
            )

            record_loader = (
                _normalise_loader_name(
                    record.loader_name
                )
            )

            global_durations.append(
                duration
            )

            if (
                normalized_loader is not None
                and record_loader == normalized_loader
            ):
                loader_durations.append(
                    duration
                )

                record_timestamp = (
                    _parse_prediction_timestamp(
                        record.prediction_timestamp
                    )
                )

                loader_previous_timestamp = (
                    record_timestamp
                )

                loader_previous_duration = (
                    duration
                )

        seconds_since_previous: (
            float | None
        ) = None

        if loader_previous_timestamp is not None:
            seconds_since_previous = float(
                (
                    current_timestamp
                    - loader_previous_timestamp
                ).total_seconds()
            )

        return OnlineHistoricalFeatures(
            global_prior_execution_count=(
                len(global_durations)
            ),
            global_prior_mean_duration_sec=(
                _safe_mean(
                    global_durations
                )
            ),
            global_prior_median_duration_sec=(
                _safe_median(
                    global_durations
                )
            ),
            global_prior_std_duration_sec=(
                _safe_std(
                    global_durations
                )
            ),
            loader_prior_execution_count=(
                len(loader_durations)
            ),
            loader_prior_mean_duration_sec=(
                _safe_mean(
                    loader_durations
                )
            ),
            loader_prior_median_duration_sec=(
                _safe_median(
                    loader_durations
                )
            ),
            loader_prior_std_duration_sec=(
                _safe_std(
                    loader_durations
                )
            ),
            loader_previous_duration_sec=(
                loader_previous_duration
            ),
            seconds_since_loader_previous=(
                seconds_since_previous
            ),
        )


__all__ = [
    "HISTORICAL_FEATURE_COLUMNS",
    "OnlineHistoricalFeatureError",
    "OnlineHistoricalFeatures",
    "OnlineHistoricalFeatureProvider",
]