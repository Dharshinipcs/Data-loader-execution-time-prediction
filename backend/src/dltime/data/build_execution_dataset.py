from __future__ import annotations

from pathlib import Path

from dltime.data.execution_reconstruction import reconstruct_executions
from dltime.data.excel_loader import load_excel
from dltime.data.raw_schemas import LOADER_EXECUTIONS_COLUMNS
from dltime.data.schema import validate_required_columns


PROJECT_ROOT = Path(__file__).resolve().parents[4]

RAW_EXECUTIONS_PATH = PROJECT_ROOT / "data" / "raw" / "LOADER_EXECTUIONS.xlsx"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_PATH = PROCESSED_DIR / "execution_level.csv"


def build_execution_dataset(
    raw_path: Path = RAW_EXECUTIONS_PATH,
    output_path: Path = OUTPUT_PATH,
) -> Path:
    """
    Build the canonical execution-level dataset from the raw execution file.

    Pipeline:
        Raw Excel
            -> schema validation
            -> execution reconstruction
            -> processed CSV

    The raw source file is never modified.
    """

    dataframe = load_excel(raw_path)

    validate_required_columns(
        dataframe,
        LOADER_EXECUTIONS_COLUMNS,
    )

    execution_level = reconstruct_executions(dataframe)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    execution_level.to_csv(
        output_path,
        index=False,
    )

    return output_path


if __name__ == "__main__":
    output = build_execution_dataset()

    print("execution dataset build complete")
    print(f"output={output}")