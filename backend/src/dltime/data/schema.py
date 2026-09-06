from collections.abc import Iterable

import pandas as pd


class SchemaValidationError(Exception):
    """Raised when a DataFrame does not contain the expected schema."""


def validate_required_columns(
    dataframe: pd.DataFrame,
    required_columns: Iterable[str],
) -> None:
    """
    Validate that all required columns exist in a DataFrame.

    Parameters
    ----------
    dataframe:
        DataFrame whose schema should be validated.
    required_columns:
        Column names that must be present.

    Raises
    ------
    SchemaValidationError
        If one or more required columns are missing.
    """
    required = list(required_columns)
    actual = set(dataframe.columns)

    missing = [column for column in required if column not in actual]

    if missing:
        missing_display = ", ".join(missing)
        raise SchemaValidationError(
            f"Missing required columns: {missing_display}"
        )


def get_column_names(dataframe: pd.DataFrame) -> list[str]:
    """
    Return DataFrame column names in their existing order.
    """
    return [str(column) for column in dataframe.columns]