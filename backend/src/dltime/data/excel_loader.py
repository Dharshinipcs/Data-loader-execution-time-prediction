from pathlib import Path

import pandas as pd


class ExcelLoaderError(Exception):
    """Raised when an Excel workbook cannot be loaded safely."""


def load_excel(
    file_path: str | Path,
    sheet_name: str | None = None,
) -> pd.DataFrame:
    """
    Load an Excel worksheet into a pandas DataFrame.

    Parameters
    ----------
    file_path:
        Path to the Excel workbook.
    sheet_name:
        Worksheet name to load. If omitted, pandas loads the first sheet.

    Returns
    -------
    pandas.DataFrame
        The contents of the selected worksheet.

    Raises
    ------
    ExcelLoaderError
        If the workbook does not exist, cannot be read, or the requested
        worksheet does not exist.
    """
    path = Path(file_path)

    if not path.exists():
        raise ExcelLoaderError(f"Excel file not found: {path}")

    if not path.is_file():
        raise ExcelLoaderError(f"Excel path is not a file: {path}")

    try:
        if sheet_name is None:
            dataframe = pd.read_excel(path)
        else:
            dataframe = pd.read_excel(path, sheet_name=sheet_name)
    except ValueError as exc:
        raise ExcelLoaderError(
            f"Worksheet '{sheet_name}' was not found in {path.name}."
        ) from exc
    except (OSError, ImportError) as exc:
        raise ExcelLoaderError(
            f"Unable to read Excel workbook '{path}'."
        ) from exc

    return dataframe