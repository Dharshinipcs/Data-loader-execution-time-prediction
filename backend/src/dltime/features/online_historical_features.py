from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from math import sqrt
from pathlib import Path

import pandas as pd

from dltime.data.feedback_store import ExecutionFeedback, SQLiteFeedbackStore


class OnlineHistoricalFeatureError(Exception):
    """Raised when online historical feature construction fails."""


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


@dataclass(frozen=True)
class OnlineHistoricalFeatures:
    """
    Causal historical features available immediately before execution.

    All values are derived exclusively from previously persisted,
    training-eligible successful executions.
    """

    global_prior_execution_count: int
    global_prior_mean_duration_sec: float | None
    global_prior_median_duration_sec: float | None
    global_prior_std_duration_sec: float | None

    loader_prior_execution_count: int
    loader_prior_mean_duration_sec: float | None
    loader_prior_median_duration_sec: float | None
    loader_prior_std_duration_sec: float | None

    loader_previous_duration_sec: float | None
    seconds_since_loader_previous_execution: float | None

    def as_dict(self) -> dict[str, int | float | None]:
        """Return features in the canonical historical-feature schema."""

        return {
            "global_prior_execution_count": (
                self.global_prior_execution_count
            ),
            "global_prior_mean_duration_sec": (
                self.global_prior_mean_duration_sec
            ),
            "global_prior_median_duration_sec": (
                self.global_prior_median_duration_sec
            ),
            "global_prior_std_duration_sec": (
                self.global_prior_std_duration_sec
            ),
            "loader_prior_execution_count": (
                self.loader_prior_execution_count
            ),
            "loader_prior_mean_duration_sec": (
                self.loader_prior_mean_duration_sec
            ),
            "loader_prior_median_duration_sec": (
                self.loader_prior_median_duration_sec
            ),
            "loader_prior_std_duration_sec": (
                self.loader_prior_std_duration_sec
            ),
            "loader_previous_duration_sec": (
                self.loader_previous_duration_sec
            ),
            "seconds_since_loader_previous_execution": (
                self.seconds_since_loader_previous_execution
            ),
        }


def _normalise_loader_name(loader_name: object) -> str | None:
    """Normalize loader identity without creating synthetic identities."""

    if loader_name is None:
        return None

    if pd.isna(loader_name):
        return None

    normalized = str(loader_name).strip()

    return normalized or None


def _parse_prediction_timestamp(
    value: str | None,
) -> datetime | None:
    """Parse a persisted prediction timestamp."""

    if value is None:
        return None

    try:
        timestamp = pd.Timestamp(value)

    except (TypeError, ValueError):
        return None

    if pd.isna(timestamp):
        return None

    return timestamp.to_pydatetime()


def _safe_mean(values: list[float]) -> float | None:
    """Return the arithmetic mean or None when history is empty."""

    if not values:
        return None

    return float(sum(values) / len(values))


def _safe_median(values: list[float]) -> float | None:
    """Return the median or None when history is empty."""

    if not values:
        return None

    return float(pd.Series(values).median())


def _safe_std(values: list[float]) -> float | None:
    """
    Return population-independent sample standard deviation.

    The offline historical feature implementation uses ddof=1, so the
    online implementation must use the same convention for consistency.
    """

    if len(values) < 2:
        return None

    mean = sum(values) / len(values)

    squared_deviations = [
        (value - mean) ** 2
        for value in values
    ]

    return float(
        sqrt(
            sum(squared_deviations)
            / (len(values) - 1)
        )
    )


class OnlineHistoricalFeatureProvider:
    """
    Build causal historical features from persisted feedback.

    The provider is deliberately read-only with respect to feedback.
    It never writes the current execution into history.

    A caller should:
        1. create prediction_timestamp at execution-trigger time;
        2. request features using that timestamp;
        3. generate the prediction;
        4. execute the loader;
        5. persist post-execution feedback separately.

    Only records satisfying the feedback store's training-eligibility
    policy are allowed to contribute to historical duration features.
    """

    def __init__(
        self,
        store: SQLiteFeedbackStore,
    ) -> None:
        if not isinstance(store, SQLiteFeedbackStore):
            raise TypeError(
                "store must be a SQLiteFeedbackStore instance."
            )

        self.store = store

    @staticmethod
    def _eligible_records(
        records: list[ExecutionFeedback],
    ) -> list[
        tuple[
            ExecutionFeedback,
            datetime,
        ]
    ]:
        """
        Return valid training records with parseable prediction timestamps.

        Records are sorted by execution/prediction timestamp, not by
        recorded_at, because causal ordering depends on when prediction
        was generated.
        """

        eligible: list[
            tuple[
                ExecutionFeedback,
                datetime,
            ]
        ] = []

        for record in records:
            if not record.training_eligible:
                continue

            if record.execution_status != "SUCCEED":
                continue

            if (
                record.actual_total_seconds is None
                or record.actual_total_seconds <= 0
            ):
                continue

            timestamp = _parse_prediction_timestamp(
                record.prediction_timestamp
            )

            if timestamp is None:
                continue

            eligible.append(
                (
                    record,
                    timestamp,
                )
            )

        eligible.sort(
            key=lambda item: (
                item[1],
                item[0].execution_id,
            )
        )

        return eligible

    def build(
        self,
        *,
        loader_name: str | None,
        prediction_timestamp: str | datetime,
    ) -> OnlineHistoricalFeatures:
        """
        Build causal features for one prediction request.

        Only persisted eligible executions with timestamps strictly
        earlier than prediction_timestamp contribute to history.

        Executions sharing the exact prediction timestamp are therefore
        treated as concurrent and cannot influence one another.
        """

        if isinstance(prediction_timestamp, datetime):
            current_timestamp = prediction_timestamp

        else:
            try:
                current_timestamp = pd.Timestamp(
                    prediction_timestamp
                ).to_pydatetime()

            except (TypeError, ValueError) as exc:
                raise OnlineHistoricalFeatureError(
                    "prediction_timestamp must be a valid datetime "
                    "or ISO timestamp string."
                ) from exc

        if pd.isna(current_timestamp):
            raise OnlineHistoricalFeatureError(
                "prediction_timestamp must not be missing."
            )

        loader_key = _normalise_loader_name(loader_name)

        records = self._eligible_records(
            self.store.all_records()
        )

        global_history: list[float] = []
        loader_history: list[float] = []
        previous_loader_timestamp: datetime | None = None

        for record, record_timestamp in records:
            if record_timestamp >= current_timestamp:
                break

            duration = float(record.actual_total_seconds)

            global_history.append(duration)

            record_loader = _normalise_loader_name(
                record.loader_name
            )

            if (
                loader_key is not None
                and record_loader == loader_key
            ):
                loader_history.append(duration)
                previous_loader_timestamp = record_timestamp

        global_mean = _safe_mean(global_history)
        global_median = _safe_median(global_history)
        global_std = _safe_std(global_history)

        loader_mean = _safe_mean(loader_history)
        loader_median = _safe_median(loader_history)
        loader_std = _safe_std(loader_history)

        if loader_history:
            previous_duration = float(
                loader_history[-1]
            )
        else:
            previous_duration = None

        if previous_loader_timestamp is None:
            seconds_since_previous = None

        else:
            seconds_since_previous = float(
                (
                    current_timestamp
                    - previous_loader_timestamp
                ).total_seconds()
            )

        return OnlineHistoricalFeatures(
            global_prior_execution_count=len(
                global_history
            ),
            global_prior_mean_duration_sec=global_mean,
            global_prior_median_duration_sec=global_median,
            global_prior_std_duration_sec=global_std,
            loader_prior_execution_count=len(
                loader_history
            ),
            loader_prior_mean_duration_sec=loader_mean,
            loader_prior_median_duration_sec=loader_median,
            loader_prior_std_duration_sec=loader_std,
            loader_previous_duration_sec=previous_duration,
            seconds_since_loader_previous_execution=(
                seconds_since_previous
            ),
        )


__all__ = [
    "HISTORICAL_FEATURE_COLUMNS",
    "OnlineHistoricalFeatureError",
    "OnlineHistoricalFeatures",
    "OnlineHistoricalFeatureProvider",
]