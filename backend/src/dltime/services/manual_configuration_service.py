from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


class ManualConfigurationError(ValueError):
    """Raised when manual loader configuration cannot be resolved safely."""


CONFIGURATION_KEY_COLUMNS: tuple[str, ...] = (
    "sprint_name",
    "ldr_connection_name",
    "loader_display_name",
    "datasetname",
)

CONFIGURATION_FEATURE_COLUMNS: tuple[str, ...] = (
    "detailed_configuration_row_count",
    "unique_preval_trans_name_count",
    "unique_preval_trans_type_count",
    "unique_calling_type_count",
    "unique_execution_order_count",
    "min_execution_order",
    "max_execution_order",
    "prevalidation_count",
    "transformation_count",
    "total_complexity_count",
    "configuration_complexity_proxy",
)

PROJECT_ROOT = Path(__file__).resolve().parents[4]

DEFAULT_CONFIGURATION_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "pre_execution_features.csv"
)


@dataclass(frozen=True)
class ManualLoaderConfiguration:
    """Resolved Data Loader configuration for a prediction request."""

    loader_name: str
    sprint: str
    ldr_connection_name: str
    datasetname: str

    prevalidation_enabled: bool
    transformation_enabled: bool

    detailed_configuration_row_count: float
    unique_preval_trans_name_count: float
    unique_preval_trans_type_count: float
    unique_calling_type_count: float
    unique_execution_order_count: float
    min_execution_order: float
    max_execution_order: float
    prevalidation_count: float
    transformation_count: float
    total_complexity_count: float
    configuration_complexity_proxy: float

    pre_execution_match_status: str

    def as_dict(self) -> dict[str, object]:
        """Return the resolved configuration as a JSON-compatible dictionary."""
        return {
            "loader_name": self.loader_name,
            "sprint": self.sprint,
            "ldr_connection_name": self.ldr_connection_name,
            "datasetname": self.datasetname,
            "prevalidation_enabled": self.prevalidation_enabled,
            "transformation_enabled": self.transformation_enabled,
            "detailed_configuration_row_count": (
                self.detailed_configuration_row_count
            ),
            "unique_preval_trans_name_count": (
                self.unique_preval_trans_name_count
            ),
            "unique_preval_trans_type_count": (
                self.unique_preval_trans_type_count
            ),
            "unique_calling_type_count": self.unique_calling_type_count,
            "unique_execution_order_count": (
                self.unique_execution_order_count
            ),
            "min_execution_order": self.min_execution_order,
            "max_execution_order": self.max_execution_order,
            "prevalidation_count": self.prevalidation_count,
            "transformation_count": self.transformation_count,
            "total_complexity_count": self.total_complexity_count,
            "configuration_complexity_proxy": (
                self.configuration_complexity_proxy
            ),
            "pre_execution_match_status": self.pre_execution_match_status,
        }


class ManualLoaderConfigurationService:
    """
    Resolve manually supplied loader identity fields against processed
    pre-execution configuration features.

    The configuration source is authoritative for derived features and
    stage availability. Manual stage flags are validated against the
    resolved configuration rather than used to override it.
    """

    def __init__(
        self,
        configuration_path: Path = DEFAULT_CONFIGURATION_PATH,
    ) -> None:
        self._configuration_path = Path(configuration_path)
        self._configuration = self._load_configuration()

    def _load_configuration(self) -> pd.DataFrame:
        if not self._configuration_path.exists():
            raise ManualConfigurationError(
                "Pre-execution configuration file was not found: "
                f"{self._configuration_path}"
            )

        try:
            dataframe = pd.read_csv(self._configuration_path)
        except Exception as exc:
            raise ManualConfigurationError(
                "Failed to load pre-execution configuration data."
            ) from exc

        required_columns = {
            *CONFIGURATION_KEY_COLUMNS,
            *CONFIGURATION_FEATURE_COLUMNS,
            "pre_execution_match_status",
        }

        missing_columns = sorted(
            required_columns.difference(dataframe.columns)
        )

        if missing_columns:
            raise ManualConfigurationError(
                "Pre-execution configuration data is missing required "
                f"columns: {', '.join(missing_columns)}"
            )

        if dataframe.empty:
            raise ManualConfigurationError(
                "Pre-execution configuration data is empty."
            )

        if dataframe[list(CONFIGURATION_KEY_COLUMNS)].isna().any().any():
            raise ManualConfigurationError(
                "Pre-execution configuration contains null configuration "
                "key values."
            )

        duplicate_mask = dataframe.duplicated(
            subset=list(CONFIGURATION_KEY_COLUMNS),
            keep=False,
        )

        if duplicate_mask.any():
            duplicate_count = int(duplicate_mask.sum())
            raise ManualConfigurationError(
                "Pre-execution configuration contains duplicate "
                f"configuration keys: {duplicate_count} rows."
            )

        return dataframe

    @staticmethod
    def _validate_text(value: str, field_name: str) -> str:
        if not isinstance(value, str):
            raise ManualConfigurationError(
                f"{field_name} must be a string."
            )

        normalized = value.strip()

        if not normalized:
            raise ManualConfigurationError(
                f"{field_name} must not be empty."
            )

        return normalized

    def resolve(
        self,
        *,
        loader_name: str,
        sprint: str,
        ldr_connection_name: str,
        datasetname: str,
        prevalidation_enabled: bool | None = None,
        transformation_enabled: bool | None = None,
    ) -> ManualLoaderConfiguration:
        """
        Resolve a manually supplied configuration.

        The four identity fields form the authoritative configuration key.
        Optional stage flags are checked against the resolved configuration
        when supplied.
        """
        loader_name = self._validate_text(
            loader_name,
            "loader_name",
        )
        sprint = self._validate_text(
            sprint,
            "sprint",
        )
        ldr_connection_name = self._validate_text(
            ldr_connection_name,
            "ldr_connection_name",
        )
        datasetname = self._validate_text(
            datasetname,
            "datasetname",
        )

        matches = self._configuration[
            (self._configuration["sprint_name"] == sprint)
            & (
                self._configuration["ldr_connection_name"]
                == ldr_connection_name
            )
            & (
                self._configuration["loader_display_name"]
                == loader_name
            )
            & (self._configuration["datasetname"] == datasetname)
        ]

        if matches.empty:
            raise ManualConfigurationError(
                "No matching Data Loader configuration was found for the "
                "supplied sprint, connection, loader, and dataset."
            )

        if len(matches) != 1:
            raise ManualConfigurationError(
                "Multiple matching Data Loader configurations were found. "
                "Prediction cannot continue safely."
            )

        row = matches.iloc[0]

        resolved_prevalidation = (
            float(row["prevalidation_count"]) > 0
        )
        resolved_transformation = (
            float(row["transformation_count"]) > 0
        )

        if (
            prevalidation_enabled is not None
            and bool(prevalidation_enabled) != resolved_prevalidation
        ):
            raise ManualConfigurationError(
                "prevalidation_enabled does not match the resolved "
                "configuration."
            )

        if (
            transformation_enabled is not None
            and bool(transformation_enabled) != resolved_transformation
        ):
            raise ManualConfigurationError(
                "transformation_enabled does not match the resolved "
                "configuration."
            )

        match_status = str(row["pre_execution_match_status"])

        if match_status != "MATCHED":
            raise ManualConfigurationError(
                "Resolved configuration is not marked as MATCHED."
            )

        numeric_values: dict[str, float] = {}

        for column in CONFIGURATION_FEATURE_COLUMNS:
            value = pd.to_numeric(
                row[column],
                errors="coerce",
            )

            if pd.isna(value):
                raise ManualConfigurationError(
                    "Resolved configuration contains an invalid value "
                    f"for '{column}'."
                )

            numeric_values[column] = float(value)

        return ManualLoaderConfiguration(
            loader_name=loader_name,
            sprint=sprint,
            ldr_connection_name=ldr_connection_name,
            datasetname=datasetname,
            prevalidation_enabled=resolved_prevalidation,
            transformation_enabled=resolved_transformation,
            detailed_configuration_row_count=(
                numeric_values["detailed_configuration_row_count"]
            ),
            unique_preval_trans_name_count=(
                numeric_values["unique_preval_trans_name_count"]
            ),
            unique_preval_trans_type_count=(
                numeric_values["unique_preval_trans_type_count"]
            ),
            unique_calling_type_count=(
                numeric_values["unique_calling_type_count"]
            ),
            unique_execution_order_count=(
                numeric_values["unique_execution_order_count"]
            ),
            min_execution_order=(
                numeric_values["min_execution_order"]
            ),
            max_execution_order=(
                numeric_values["max_execution_order"]
            ),
            prevalidation_count=(
                numeric_values["prevalidation_count"]
            ),
            transformation_count=(
                numeric_values["transformation_count"]
            ),
            total_complexity_count=(
                numeric_values["total_complexity_count"]
            ),
            configuration_complexity_proxy=(
                numeric_values["configuration_complexity_proxy"]
            ),
            pre_execution_match_status=match_status,
        )


__all__ = [
    "CONFIGURATION_FEATURE_COLUMNS",
    "CONFIGURATION_KEY_COLUMNS",
    "DEFAULT_CONFIGURATION_PATH",
    "ManualConfigurationError",
    "ManualLoaderConfiguration",
    "ManualLoaderConfigurationService",
]
