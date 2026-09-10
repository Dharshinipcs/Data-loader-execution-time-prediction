from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from dltime.data.feedback_store import ExecutionFeedback, SQLiteFeedbackStore


StageUnit = Literal["seconds", "minutes"]


class AnalyticsServiceError(Exception):
    """Raised when analytics data cannot be loaded or processed."""


@dataclass(frozen=True)
class AnalyticsOverview:
    """High-level execution analytics."""

    total_executions: int
    successful_executions: int
    stopped_executions: int
    other_executions: int
    valid_training_targets: int
    unique_loaders: int
    total_records: int
    average_duration_seconds: float | None
    median_duration_seconds: float | None
    p90_duration_seconds: float | None
    maximum_duration_seconds: float | None
    earliest_execution: datetime | None
    latest_execution: datetime | None


@dataclass(frozen=True)
class ExecutionHistoryRecord:
    """Single execution-history observation."""

    execution_id: str
    loader_name: str | None
    status: str | None
    sprint: str | None
    start_time: datetime | None
    end_time: datetime | None
    duration_seconds: float | None
    dataset_count: int
    total_records: int
    success_records: int
    error_records: int
    prevalidation_duration_seconds: float | None
    transformation_duration_seconds: float | None
    staging_duration_minutes: float | None
    data_loading_duration_seconds: float | None
    datamart_approval_duration_seconds: float | None
    dataloading_approval_duration_seconds: float | None


@dataclass(frozen=True)
class DurationTrendRecord:
    """One date-level duration trend observation."""

    execution_date: str
    execution_count: int
    average_duration_seconds: float | None
    median_duration_seconds: float | None


@dataclass(frozen=True)
class LoaderIntelligenceRecord:
    """Loader-level historical statistics."""

    loader_name: str
    execution_count: int
    successful_execution_count: int
    average_duration_seconds: float | None
    median_duration_seconds: float | None
    p90_duration_seconds: float | None
    maximum_duration_seconds: float | None
    total_records: int
    latest_execution: datetime | None


@dataclass(frozen=True)
class StageAnalyticsRecord:
    """One stage's duration statistics with explicit source unit."""

    stage_name: str
    unit: StageUnit
    observations: int
    average_duration: float | None
    median_duration: float | None
    p90_duration: float | None
    maximum_duration: float | None


class AnalyticsService:
    """
    Read-only analytics service for the frontend.

    The service deliberately reads already validated/reconstructed
    processed datasets instead of raw Excel workbooks. This preserves
    the existing data-engineering boundary:

        raw source
            -> reconstruction/validation
            -> processed execution datasets
            -> analytics service
            -> API
            -> frontend
    """

    EXECUTION_COLUMNS = {
        "LDR_Execution_Id",
        "Dataset_Count",
        "Loader_Name",
        "LDR_Status",
        "Sprint",
        "Start_Time",
        "End_Time",
        "timetaken_in_sec",
        "Total_Records",
        "Success_Records",
        "Error_Records",
        "prevaliadtion_timetaken_in_sec",
        "transformation_timetaken_in_sec",
        "staging_timetaken_in_min",
        "data_loading_timetaken_in_sec",
        "data_loading_timetaken_in_sec_1",
        "datamart_approval_timetaken_in_sec",
        "dataloading_approval_timetaken_in_sec",
    }

    STAGE_COLUMNS: dict[str, tuple[str, StageUnit]] = {
        "prevalidation": (
            "prevaliadtion_timetaken_in_sec",
            "seconds",
        ),
        "transformation": (
            "transformation_timetaken_in_sec",
            "seconds",
        ),
        "staging": (
            "staging_timetaken_in_min",
            "minutes",
        ),
        "data_loading": (
            "data_loading_timetaken_in_sec",
            "seconds",
        ),
        "data_loading_1": (
            "data_loading_timetaken_in_sec_1",
            "seconds",
        ),
        "datamart_approval": (
            "datamart_approval_timetaken_in_sec",
            "seconds",
        ),
        "dataloading_approval": (
            "dataloading_approval_timetaken_in_sec",
            "seconds",
        ),
    }

    SUCCESS_STATUS = "SUCCEED"
    STOPPED_STATUSES = frozenset({"STOPPED", "KILLED"})

    def __init__(
        self,
        project_root: str | Path,
        feedback_store: SQLiteFeedbackStore,
    ) -> None:
        self.project_root = Path(project_root)
        self.feedback_store = feedback_store

        self.execution_dataset_path = (
            self.project_root
            / "data"
            / "processed"
            / "execution_level.csv"
        )

        self.training_target_path = (
            self.project_root
            / "data"
            / "processed"
            / "execution_training_targets.csv"
        )

        self._execution_dataframe: pd.DataFrame | None = None
        self._training_target_count: int | None = None

    # ------------------------------------------------------------------
    # Data loading and validation
    # ------------------------------------------------------------------

    def _load_execution_dataframe(self) -> pd.DataFrame:
        """Load and validate the canonical execution-level dataset."""

        if self._execution_dataframe is not None:
            return self._execution_dataframe.copy()

        if not self.execution_dataset_path.exists():
            raise AnalyticsServiceError(
                f"Execution dataset not found: {self.execution_dataset_path}"
            )

        try:
            dataframe = pd.read_csv(self.execution_dataset_path)
        except (OSError, pd.errors.ParserError) as exc:
            raise AnalyticsServiceError(
                f"Unable to read execution dataset: "
                f"{self.execution_dataset_path}"
            ) from exc

        missing = sorted(self.EXECUTION_COLUMNS - set(dataframe.columns))
        if missing:
            raise AnalyticsServiceError(
                "Execution dataset is missing required columns: "
                + ", ".join(missing)
            )

        if dataframe.empty:
            raise AnalyticsServiceError(
                "Execution dataset contains no execution records."
            )

        self._execution_dataframe = dataframe.copy()
        return dataframe

    def _load_training_target_count(self) -> int:
        """Return the number of validated training-target rows."""

        if self._training_target_count is not None:
            return self._training_target_count

        if not self.training_target_path.exists():
            raise AnalyticsServiceError(
                f"Training-target dataset not found: "
                f"{self.training_target_path}"
            )

        try:
            dataframe = pd.read_csv(self.training_target_path)
        except (OSError, pd.errors.ParserError) as exc:
            raise AnalyticsServiceError(
                f"Unable to read training-target dataset: "
                f"{self.training_target_path}"
            ) from exc

        self._training_target_count = len(dataframe)
        return self._training_target_count

    # ------------------------------------------------------------------
    # Normalization helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _clean_text(value: object) -> str | None:
        """Convert a nullable dataframe value into clean API text."""

        if value is None or pd.isna(value):
            return None

        text = str(value).strip()
        return text if text else None

    @staticmethod
    def _clean_number(value: object) -> float | None:
        """Convert a nullable numeric value into a finite float."""

        if value is None or pd.isna(value):
            return None

        try:
            number = float(value)
        except (TypeError, ValueError):
            return None

        if not np.isfinite(number):
            return None

        return number

    @classmethod
    def _clean_integer(cls, value: object) -> int:
        """Convert a nullable count value into a non-negative integer."""

        number = cls._clean_number(value)

        if number is None:
            return 0

        return max(0, int(round(number)))

    @classmethod
    def _parse_datetime(cls, value: object) -> datetime | None:
        """Parse a timestamp and normalize it to UTC."""

        if value is None or pd.isna(value):
            return None

        timestamp = pd.to_datetime(value, errors="coerce", utc=True)

        if pd.isna(timestamp):
            return None

        return timestamp.to_pydatetime().astimezone(timezone.utc)

    @staticmethod
    def _percentile(
        series: pd.Series,
        percentile: float,
    ) -> float | None:
        """Return a finite percentile from a numeric series."""

        numeric = pd.to_numeric(series, errors="coerce").dropna()

        if numeric.empty:
            return None

        value = float(
            np.percentile(
                numeric.to_numpy(dtype=float),
                percentile,
            )
        )

        return value if np.isfinite(value) else None

    # ------------------------------------------------------------------
    # Overview
    # ------------------------------------------------------------------

    def get_overview(self) -> AnalyticsOverview:
        """Build high-level analytics from real execution records."""

        dataframe = self._load_execution_dataframe()

        statuses = dataframe["LDR_Status"].map(self._clean_text)

        total_executions = len(dataframe)

        successful_executions = int(
            statuses.eq(self.SUCCESS_STATUS).sum()
        )

        stopped_executions = int(
            statuses.isin(self.STOPPED_STATUSES).sum()
        )

        other_executions = (
            total_executions
            - successful_executions
            - stopped_executions
        )

        durations = pd.to_numeric(
            dataframe["timetaken_in_sec"],
            errors="coerce",
        )

        positive_durations = durations[
            np.isfinite(durations) & (durations > 0)
        ]

        loaders = dataframe["Loader_Name"].map(self._clean_text)
        unique_loaders = int(loaders.dropna().nunique())

        total_records = int(
            pd.to_numeric(
                dataframe["Total_Records"],
                errors="coerce",
            )
            .fillna(0)
            .clip(lower=0)
            .sum()
        )

        start_times = pd.to_datetime(
            dataframe["Start_Time"],
            errors="coerce",
            utc=True,
        ).dropna()

        earliest_execution = (
            start_times.min()
            .to_pydatetime()
            .astimezone(timezone.utc)
            if not start_times.empty
            else None
        )

        latest_execution = (
            start_times.max()
            .to_pydatetime()
            .astimezone(timezone.utc)
            if not start_times.empty
            else None
        )

        return AnalyticsOverview(
            total_executions=total_executions,
            successful_executions=successful_executions,
            stopped_executions=stopped_executions,
            other_executions=other_executions,
            valid_training_targets=self._load_training_target_count(),
            unique_loaders=unique_loaders,
            total_records=total_records,
            average_duration_seconds=(
                float(positive_durations.mean())
                if not positive_durations.empty
                else None
            ),
            median_duration_seconds=(
                float(positive_durations.median())
                if not positive_durations.empty
                else None
            ),
            p90_duration_seconds=self._percentile(
                positive_durations,
                90,
            ),
            maximum_duration_seconds=(
                float(positive_durations.max())
                if not positive_durations.empty
                else None
            ),
            earliest_execution=earliest_execution,
            latest_execution=latest_execution,
        )

    # ------------------------------------------------------------------
    # Execution history
    # ------------------------------------------------------------------

    def get_execution_history(
        self,
        *,
        offset: int = 0,
        limit: int = 50,
        loader_name: str | None = None,
        status: str | None = None,
    ) -> tuple[list[ExecutionHistoryRecord], int]:
        """
        Return paginated execution history.

        Results are sorted newest-first by Start_Time and then execution
        ID for deterministic pagination.
        """

        if offset < 0:
            raise AnalyticsServiceError("offset must be non-negative.")

        if limit < 1 or limit > 200:
            raise AnalyticsServiceError(
                "limit must be between 1 and 200."
            )

        dataframe = self._load_execution_dataframe().copy()

        if loader_name is not None:
            loader_filter = loader_name.strip()

            if loader_filter:
                dataframe = dataframe[
                    dataframe["Loader_Name"]
                    .fillna("")
                    .astype(str)
                    .str.strip()
                    .eq(loader_filter)
                ]

        if status is not None:
            status_filter = status.strip()

            if status_filter:
                dataframe = dataframe[
                    dataframe["LDR_Status"]
                    .fillna("")
                    .astype(str)
                    .str.strip()
                    .str.upper()
                    .eq(status_filter.upper())
                ]

        dataframe["_sort_start_time"] = pd.to_datetime(
            dataframe["Start_Time"],
            errors="coerce",
            utc=True,
        )

        dataframe = dataframe.sort_values(
            by=["_sort_start_time", "LDR_Execution_Id"],
            ascending=[False, False],
            na_position="last",
        )

        total = len(dataframe)

        page = dataframe.iloc[offset : offset + limit]

        records: list[ExecutionHistoryRecord] = []

        for _, row in page.iterrows():
            records.append(
                ExecutionHistoryRecord(
                    execution_id=str(row["LDR_Execution_Id"]),
                    loader_name=self._clean_text(row["Loader_Name"]),
                    status=self._clean_text(row["LDR_Status"]),
                    sprint=self._clean_text(row["Sprint"]),
                    start_time=self._parse_datetime(row["Start_Time"]),
                    end_time=self._parse_datetime(row["End_Time"]),
                    duration_seconds=self._clean_number(
                        row["timetaken_in_sec"]
                    ),
                    dataset_count=self._clean_integer(
                        row["Dataset_Count"]
                    ),
                    total_records=self._clean_integer(
                        row["Total_Records"]
                    ),
                    success_records=self._clean_integer(
                        row["Success_Records"]
                    ),
                    error_records=self._clean_integer(
                        row["Error_Records"]
                    ),
                    prevalidation_duration_seconds=self._clean_number(
                        row["prevaliadtion_timetaken_in_sec"]
                    ),
                    transformation_duration_seconds=self._clean_number(
                        row["transformation_timetaken_in_sec"]
                    ),
                    staging_duration_minutes=self._clean_number(
                        row["staging_timetaken_in_min"]
                    ),
                    data_loading_duration_seconds=self._clean_number(
                        row["data_loading_timetaken_in_sec"]
                    ),
                    datamart_approval_duration_seconds=self._clean_number(
                        row["datamart_approval_timetaken_in_sec"]
                    ),
                    dataloading_approval_duration_seconds=self._clean_number(
                        row["dataloading_approval_timetaken_in_sec"]
                    ),
                )
            )

        return records, total

    # ------------------------------------------------------------------
    # Duration trend
    # ------------------------------------------------------------------

    def get_duration_trend(self) -> list[DurationTrendRecord]:
        """Aggregate execution duration by calendar date."""

        dataframe = self._load_execution_dataframe().copy()

        dataframe["_start_time"] = pd.to_datetime(
            dataframe["Start_Time"],
            errors="coerce",
            utc=True,
        )

        dataframe["_duration"] = pd.to_numeric(
            dataframe["timetaken_in_sec"],
            errors="coerce",
        )

        dataframe = dataframe[
            dataframe["_start_time"].notna()
            & dataframe["_duration"].notna()
            & np.isfinite(dataframe["_duration"])
            & (dataframe["_duration"] > 0)
        ].copy()

        if dataframe.empty:
            return []

        dataframe["_execution_date"] = (
            dataframe["_start_time"].dt.strftime("%Y-%m-%d")
        )

        grouped = (
            dataframe.groupby("_execution_date")["_duration"]
            .agg(["count", "mean", "median"])
            .reset_index()
            .sort_values("_execution_date")
        )

        return [
            DurationTrendRecord(
                execution_date=str(row["_execution_date"]),
                execution_count=int(row["count"]),
                average_duration_seconds=float(row["mean"]),
                median_duration_seconds=float(row["median"]),
            )
            for _, row in grouped.iterrows()
        ]

    # ------------------------------------------------------------------
    # Loader intelligence
    # ------------------------------------------------------------------

    def get_loader_intelligence(self) -> list[LoaderIntelligenceRecord]:
        """Build loader-level historical statistics."""

        dataframe = self._load_execution_dataframe().copy()

        dataframe["_loader"] = dataframe["Loader_Name"].map(
            self._clean_text
        )

        dataframe["_duration"] = pd.to_numeric(
            dataframe["timetaken_in_sec"],
            errors="coerce",
        )

        dataframe["_records"] = pd.to_numeric(
            dataframe["Total_Records"],
            errors="coerce",
        ).fillna(0).clip(lower=0)

        dataframe["_start_time"] = pd.to_datetime(
            dataframe["Start_Time"],
            errors="coerce",
            utc=True,
        )

        dataframe = dataframe[dataframe["_loader"].notna()].copy()

        records: list[LoaderIntelligenceRecord] = []

        for loader_name, group in dataframe.groupby(
            "_loader",
            sort=True,
        ):
            durations = group["_duration"]

            positive_durations = durations[
                durations.notna()
                & np.isfinite(durations)
                & (durations > 0)
            ]

            statuses = group["LDR_Status"].map(self._clean_text)

            latest_start = group["_start_time"].dropna()

            latest_execution = (
                latest_start.max()
                .to_pydatetime()
                .astimezone(timezone.utc)
                if not latest_start.empty
                else None
            )

            records.append(
                LoaderIntelligenceRecord(
                    loader_name=str(loader_name),
                    execution_count=len(group),
                    successful_execution_count=int(
                        statuses.eq(self.SUCCESS_STATUS).sum()
                    ),
                    average_duration_seconds=(
                        float(positive_durations.mean())
                        if not positive_durations.empty
                        else None
                    ),
                    median_duration_seconds=(
                        float(positive_durations.median())
                        if not positive_durations.empty
                        else None
                    ),
                    p90_duration_seconds=self._percentile(
                        positive_durations,
                        90,
                    ),
                    maximum_duration_seconds=(
                        float(positive_durations.max())
                        if not positive_durations.empty
                        else None
                    ),
                    total_records=int(group["_records"].sum()),
                    latest_execution=latest_execution,
                )
            )

        records.sort(
            key=lambda item: (
                item.execution_count,
                item.loader_name,
            ),
            reverse=True,
        )

        return records

    # ------------------------------------------------------------------
    # Stage analytics
    # ------------------------------------------------------------------

    def get_stage_analytics(self) -> list[StageAnalyticsRecord]:
        """
        Build statistics for every available stage-duration field.

        Stage units follow the authoritative processed dataset:
        staging is stored in minutes; all other exposed stages are
        stored in seconds.
        """

        dataframe = self._load_execution_dataframe()

        records: list[StageAnalyticsRecord] = []

        for stage_name, (column_name, unit) in self.STAGE_COLUMNS.items():
            values = pd.to_numeric(
                dataframe[column_name],
                errors="coerce",
            )

            values = values[
                values.notna()
                & np.isfinite(values)
                & (values >= 0)
            ]

            records.append(
                StageAnalyticsRecord(
                    stage_name=stage_name,
                    unit=unit,
                    observations=len(values),
                    average_duration=(
                        float(values.mean())
                        if not values.empty
                        else None
                    ),
                    median_duration=(
                        float(values.median())
                        if not values.empty
                        else None
                    ),
                    p90_duration=self._percentile(
                        values,
                        90,
                    ),
                    maximum_duration=(
                        float(values.max())
                        if not values.empty
                        else None
                    ),
                )
            )

        return records

    # ------------------------------------------------------------------
    # Model / prediction information
    # ------------------------------------------------------------------

    def get_feedback_records(self) -> list[ExecutionFeedback]:
        """Return persisted prediction feedback records."""

        return self.feedback_store.all_records()


__all__ = [
    "AnalyticsOverview",
    "ExecutionHistoryRecord",
    "DurationTrendRecord",
    "LoaderIntelligenceRecord",
    "StageAnalyticsRecord",
    "AnalyticsService",
    "AnalyticsServiceError",
    "StageUnit",
]