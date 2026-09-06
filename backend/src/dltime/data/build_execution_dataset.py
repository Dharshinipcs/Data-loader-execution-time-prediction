from __future__ import annotations

from pathlib import Path

from dltime.data.execution_reconstruction import reconstruct_executions
from dltime.data.excel_loader import load_excel
from dltime.data.raw_schemas import LOADER_EXECUTIONS_COLUMNS
from dltime.data.schema import validate_required_columns
from dltime.data.target_quality import (
    classify_target_quality,
    get_valid_targets,
)


PROJECT_ROOT = Path(__file__).resolve().parents[4]

RAW_EXECUTIONS_PATH = PROJECT_ROOT / "data" / "raw" / "LOADER_EXECTUIONS.xlsx"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

EXECUTION_LEVEL_OUTPUT_PATH = (
    PROCESSED_DIR / "execution_level.csv"
)

TARGET_QUALITY_OUTPUT_PATH = (
    PROCESSED_DIR / "execution_target_quality.csv"
)

TRAINING_TARGET_OUTPUT_PATH = (
    PROCESSED_DIR / "execution_training_targets.csv"
)


def build_execution_dataset(
    raw_path: Path = RAW_EXECUTIONS_PATH,
    execution_output_path: Path = EXECUTION_LEVEL_OUTPUT_PATH,
    target_quality_output_path: Path = TARGET_QUALITY_OUTPUT_PATH,
    training_target_output_path: Path = TRAINING_TARGET_OUTPUT_PATH,
) -> tuple[Path, Path, Path]:
    """
    Build reproducible execution-level datasets from the raw execution file.

    Pipeline
    --------
    Raw Excel
        -> schema validation
        -> execution reconstruction
        -> canonical execution dataset
        -> target-quality classification
        -> valid training-target dataset

    Outputs
    -------
    execution_level.csv
        One row per reconstructed execution, including successful,
        stopped, killed, warning, and error executions.

    execution_target_quality.csv
        Successful executions with target-quality diagnostics and
        classification.

    execution_training_targets.csv
        Successful executions whose total execution-time target passed
        the validated quality rules.

    The raw source file is never modified.
    """

    dataframe = load_excel(raw_path)

    validate_required_columns(
        dataframe,
        LOADER_EXECUTIONS_COLUMNS,
    )

    execution_level = reconstruct_executions(dataframe)

    target_quality = classify_target_quality(
        execution_level,
    )

    training_targets = get_valid_targets(
        execution_level,
    )

    execution_output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    execution_output_path = Path(execution_output_path)
    target_quality_output_path = Path(target_quality_output_path)
    training_target_output_path = Path(training_target_output_path)

    execution_level.to_csv(
        execution_output_path,
        index=False,
    )

    target_quality.to_csv(
        target_quality_output_path,
        index=False,
    )

    training_targets.to_csv(
        training_target_output_path,
        index=False,
    )

    return (
        execution_output_path,
        target_quality_output_path,
        training_target_output_path,
    )


if __name__ == "__main__":
    (
        execution_output,
        target_quality_output,
        training_target_output,
    ) = build_execution_dataset()

    print("execution dataset build complete")
    print(f"execution_level_output={execution_output}")
    print(f"target_quality_output={target_quality_output}")
    print(f"training_target_output={training_target_output}")