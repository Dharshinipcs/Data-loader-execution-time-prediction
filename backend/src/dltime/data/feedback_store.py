from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class ExecutionFeedback:
    """
    Persisted post-execution feedback record.

    The execution_id is the idempotency key. A single DataZap execution
    must produce at most one feedback record.

    Feedback records are persisted for auditability even when an execution
    is not eligible for duration-model training.
    """

    execution_id: str
    loader_name: str | None = None
    sprint: str | None = None

    prediction_timestamp: str | None = None
    predicted_total_seconds: float | None = None
    actual_total_seconds: float | None = None

    prediction_error_seconds: float | None = None
    absolute_error_seconds: float | None = None
    relative_error: float | None = None

    execution_status: str | None = None

    actual_stage_durations: dict[str, float | None] | None = None

    used_prediction_source: str | None = None
    reliability_score: float | None = None
    model_version: str | None = None

    training_eligible: bool = False

    recorded_at: str | None = None


class SQLiteFeedbackStore:
    """
    SQLite-backed persistence for post-execution feedback.

    The storage implementation is intentionally isolated from the
    prediction engine so the persistence layer can later be replaced
    by a DataZap/enterprise database without changing prediction logic.
    """

    TABLE_NAME = "execution_feedback"
    TRAINING_STATUS = "SUCCEED"

    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_database()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _initialize_database(self) -> None:
        with self._connect() as connection:
            connection.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.TABLE_NAME} (
                    execution_id TEXT PRIMARY KEY,

                    loader_name TEXT,
                    sprint TEXT,

                    prediction_timestamp TEXT,
                    predicted_total_seconds REAL,
                    actual_total_seconds REAL,

                    prediction_error_seconds REAL,
                    absolute_error_seconds REAL,
                    relative_error REAL,

                    execution_status TEXT,

                    actual_stage_durations_json TEXT,

                    used_prediction_source TEXT,
                    reliability_score REAL,
                    model_version TEXT,

                    training_eligible INTEGER NOT NULL DEFAULT 0,

                    recorded_at TEXT NOT NULL
                )
                """
            )

            connection.execute(
                f"""
                CREATE INDEX IF NOT EXISTS idx_{self.TABLE_NAME}_loader
                ON {self.TABLE_NAME}(loader_name)
                """
            )

            connection.execute(
                f"""
                CREATE INDEX IF NOT EXISTS idx_{self.TABLE_NAME}_status
                ON {self.TABLE_NAME}(execution_status)
                """
            )

            connection.execute(
                f"""
                CREATE INDEX IF NOT EXISTS idx_{self.TABLE_NAME}_training_eligible
                ON {self.TABLE_NAME}(training_eligible)
                """
            )

            connection.execute(
                f"""
                CREATE INDEX IF NOT EXISTS idx_{self.TABLE_NAME}_recorded_at
                ON {self.TABLE_NAME}(recorded_at)
                """
            )

    @staticmethod
    def _utc_now() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _serialize_stage_durations(
        stage_durations: dict[str, float | None] | None,
    ) -> str | None:
        if stage_durations is None:
            return None

        return json.dumps(
            stage_durations,
            sort_keys=True,
            separators=(",", ":"),
        )

    @staticmethod
    def _deserialize_stage_durations(
        value: str | None,
    ) -> dict[str, float | None] | None:
        if value is None:
            return None

        parsed = json.loads(value)

        if not isinstance(parsed, dict):
            raise ValueError("Stored stage durations must be a JSON object.")

        return parsed

    @classmethod
    def _is_training_eligible(cls, feedback: ExecutionFeedback) -> bool:
        """
        Apply the persistence-layer training eligibility invariant.

        A duration-model training example must represent a naturally
        completed successful execution with a positive actual duration.

        Target-quality validation, such as timestamp consistency, is
        intentionally handled upstream because the store does not have
        enough context to validate those semantics.
        """

        return (
            feedback.execution_status == cls.TRAINING_STATUS
            and feedback.actual_total_seconds is not None
            and feedback.actual_total_seconds > 0
        )

    def save(self, feedback: ExecutionFeedback) -> bool:
        """
        Insert one feedback record.

        Returns:
            True  -> a new record was inserted.
            False -> the execution_id already existed, so no duplicate
                     feedback record was created.

        Training eligibility is computed from the persisted execution
        outcome and actual duration rather than trusted from the caller.

        This method intentionally does not overwrite an existing record.
        That makes feedback ingestion idempotent and preserves the first
        persisted prediction/actual pair for auditability.
        """

        if not feedback.execution_id.strip():
            raise ValueError("execution_id must not be empty.")

        recorded_at = feedback.recorded_at or self._utc_now()
        training_eligible = self._is_training_eligible(feedback)

        with self._connect() as connection:
            cursor = connection.execute(
                f"""
                INSERT OR IGNORE INTO {self.TABLE_NAME} (
                    execution_id,
                    loader_name,
                    sprint,
                    prediction_timestamp,
                    predicted_total_seconds,
                    actual_total_seconds,
                    prediction_error_seconds,
                    absolute_error_seconds,
                    relative_error,
                    execution_status,
                    actual_stage_durations_json,
                    used_prediction_source,
                    reliability_score,
                    model_version,
                    training_eligible,
                    recorded_at
                )
                VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?, ?
                )
                """,
                (
                    feedback.execution_id,
                    feedback.loader_name,
                    feedback.sprint,
                    feedback.prediction_timestamp,
                    feedback.predicted_total_seconds,
                    feedback.actual_total_seconds,
                    feedback.prediction_error_seconds,
                    feedback.absolute_error_seconds,
                    feedback.relative_error,
                    feedback.execution_status,
                    self._serialize_stage_durations(
                        feedback.actual_stage_durations
                    ),
                    feedback.used_prediction_source,
                    feedback.reliability_score,
                    feedback.model_version,
                    int(training_eligible),
                    recorded_at,
                ),
            )

            return cursor.rowcount == 1

    def get(self, execution_id: str) -> ExecutionFeedback | None:
        """Return one feedback record by execution ID."""

        with self._connect() as connection:
            row = connection.execute(
                f"""
                SELECT
                    execution_id,
                    loader_name,
                    sprint,
                    prediction_timestamp,
                    predicted_total_seconds,
                    actual_total_seconds,
                    prediction_error_seconds,
                    absolute_error_seconds,
                    relative_error,
                    execution_status,
                    actual_stage_durations_json,
                    used_prediction_source,
                    reliability_score,
                    model_version,
                    training_eligible,
                    recorded_at
                FROM {self.TABLE_NAME}
                WHERE execution_id = ?
                """,
                (execution_id,),
            ).fetchone()

        if row is None:
            return None

        return ExecutionFeedback(
            execution_id=row["execution_id"],
            loader_name=row["loader_name"],
            sprint=row["sprint"],
            prediction_timestamp=row["prediction_timestamp"],
            predicted_total_seconds=row["predicted_total_seconds"],
            actual_total_seconds=row["actual_total_seconds"],
            prediction_error_seconds=row["prediction_error_seconds"],
            absolute_error_seconds=row["absolute_error_seconds"],
            relative_error=row["relative_error"],
            execution_status=row["execution_status"],
            actual_stage_durations=self._deserialize_stage_durations(
                row["actual_stage_durations_json"]
            ),
            used_prediction_source=row["used_prediction_source"],
            reliability_score=row["reliability_score"],
            model_version=row["model_version"],
            training_eligible=bool(row["training_eligible"]),
            recorded_at=row["recorded_at"],
        )

    def count(self) -> int:
        """Return the number of persisted feedback records."""

        with self._connect() as connection:
            row = connection.execute(
                f"SELECT COUNT(*) AS count FROM {self.TABLE_NAME}"
            ).fetchone()

        return int(row["count"])

    def count_training_eligible(self) -> int:
        """Return the number of persisted training-eligible records."""

        with self._connect() as connection:
            row = connection.execute(
                f"""
                SELECT COUNT(*) AS count
                FROM {self.TABLE_NAME}
                WHERE training_eligible = 1
                """
            ).fetchone()

        return int(row["count"])

    def exists(self, execution_id: str) -> bool:
        """Return whether feedback already exists for an execution."""

        with self._connect() as connection:
            row = connection.execute(
                f"""
                SELECT 1
                FROM {self.TABLE_NAME}
                WHERE execution_id = ?
                LIMIT 1
                """,
                (execution_id,),
            ).fetchone()

        return row is not None

    def all_records(self) -> list[ExecutionFeedback]:
        """Return all feedback records in recorded order."""

        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT
                    execution_id,
                    loader_name,
                    sprint,
                    prediction_timestamp,
                    predicted_total_seconds,
                    actual_total_seconds,
                    prediction_error_seconds,
                    absolute_error_seconds,
                    relative_error,
                    execution_status,
                    actual_stage_durations_json,
                    used_prediction_source,
                    reliability_score,
                    model_version,
                    training_eligible,
                    recorded_at
                FROM {self.TABLE_NAME}
                ORDER BY recorded_at ASC
                """
            ).fetchall()

        return [
            ExecutionFeedback(
                execution_id=row["execution_id"],
                loader_name=row["loader_name"],
                sprint=row["sprint"],
                prediction_timestamp=row["prediction_timestamp"],
                predicted_total_seconds=row["predicted_total_seconds"],
                actual_total_seconds=row["actual_total_seconds"],
                prediction_error_seconds=row["prediction_error_seconds"],
                absolute_error_seconds=row["absolute_error_seconds"],
                relative_error=row["relative_error"],
                execution_status=row["execution_status"],
                actual_stage_durations=self._deserialize_stage_durations(
                    row["actual_stage_durations_json"]
                ),
                used_prediction_source=row["used_prediction_source"],
                reliability_score=row["reliability_score"],
                model_version=row["model_version"],
                training_eligible=bool(row["training_eligible"]),
                recorded_at=row["recorded_at"],
            )
            for row in rows
        ]


__all__ = [
    "ExecutionFeedback",
    "SQLiteFeedbackStore",
]
