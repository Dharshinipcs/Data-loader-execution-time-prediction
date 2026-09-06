from __future__ import annotations

from collections.abc import Callable

import pandas as pd


class ExecutionReconstructionError(Exception):
    """Raised when execution-level reconstruction cannot be completed."""


# Fields that should have one consistent value for an execution.
EXECUTION_REPRESENTATIVE_COLUMNS: tuple[str, ...] = (
    "Loader_Name",
    "LDR_Status",
    "Sprint",
    "ldr_workflow_execution_id",
    "Start_Time",
    "End_Time",
    "timetaken_in_sec",
    "staging_timetaken_in_min",
    "data_loading_timetaken_in_sec",
    "data_loading_timetaken_in_sec_1",
    "datamart_approval_timetaken_in_sec",
    "dataloading_approval_timetaken_in_sec",
)


# Dataset-level fields are additive when reconstructing one execution.
DATASET_SUM_COLUMNS: tuple[str, ...] = (
    "Total_Records",
    "Success_Records",
    "Error_Records",
    "Prevalidation_Errors",
    "Format_Errors",
    "Constraint_Errors",
    "Application_Errors",
    "Cascade_Errors",
)


# These stages can appear on multiple dataset rows but represent the
# execution-level elapsed stage time. Historical investigation established
# that MAX is the correct aggregation rule.
STAGE_MAX_COLUMNS: tuple[str, ...] = (
    "prevaliadtion_timetaken_in_sec",
    "transformation_timetaken_in_sec",
)


def _first_non_null(series: pd.Series) -> object:
    """Return the first non-null value from a Series, or None."""
    non_null = series.dropna()

    if non_null.empty:
        return None

    return non_null.iloc[0]


def _validate_consistency(
    execution_id: object,
    execution: pd.DataFrame,
    column: str,
) -> None:
    """Validate that a representative execution-level field is consistent."""
    values = execution[column].dropna().unique()

    if len(values) <= 1:
        return

    raise ExecutionReconstructionError(
        f"Execution '{execution_id}' has conflicting values in "
        f"execution-level column '{column}'."
    )


def reconstruct_executions(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Reconstruct dataset-level execution rows into one row per execution.

    The raw DataFrame is never modified.

    Aggregation rules
    -----------------
    * Execution-level fields must be internally consistent.
    * Dataset-level record/error counts are summed.
    * Prevalidation and transformation durations use MAX because multiple
      dataset rows can represent concurrent stage activity.
    * Other stored execution-level timing values use their representative
      non-null value.
    """
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("dataframe must be a pandas DataFrame.")

    if "LDR_Execution_Id" not in dataframe.columns:
        raise ExecutionReconstructionError(
            "Required column 'LDR_Execution_Id' is missing."
        )

    if dataframe.empty:
        return dataframe.copy()

    if dataframe["LDR_Execution_Id"].isna().any():
        raise ExecutionReconstructionError(
            "Cannot reconstruct executions because some rows have a missing "
            "'LDR_Execution_Id'."
        )

    missing_columns = [
        column
        for column in (
            *EXECUTION_REPRESENTATIVE_COLUMNS,
            *DATASET_SUM_COLUMNS,
            *STAGE_MAX_COLUMNS,
        )
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ExecutionReconstructionError(
            "Missing columns required for execution reconstruction: "
            + ", ".join(missing_columns)
        )

    records: list[dict[str, object]] = []

    for execution_id, execution in dataframe.groupby(
        "LDR_Execution_Id",
        sort=False,
    ):
        record: dict[str, object] = {
            "LDR_Execution_Id": execution_id,
            "Dataset_Row_Count": len(execution),
            "Dataset_Count": execution["Dataset_display_name"].nunique(
                dropna=True
            )
            if "Dataset_display_name" in execution.columns
            else 0,
        }

        for column in EXECUTION_REPRESENTATIVE_COLUMNS:
            _validate_consistency(execution_id, execution, column)
            record[column] = _first_non_null(execution[column])

        for column in DATASET_SUM_COLUMNS:
            record[column] = execution[column].sum(min_count=1)

        for column in STAGE_MAX_COLUMNS:
            record[column] = execution[column].max(skipna=True)

        records.append(record)

    result = pd.DataFrame(records)

    if result.empty:
        return result

    result["LDR_Execution_Id"] = result["LDR_Execution_Id"].astype(
        dataframe["LDR_Execution_Id"].dtype,
        copy=False,
    )

    return result.reset_index(drop=True)