from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[4]

DEFAULT_FEATURE_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "pre_execution_features.csv"
)


NUMERIC_FEATURE_COLUMNS = (
    "prevalidation_count",
    "transformation_count",
    "detailed_configuration_row_count",
    "unique_preval_trans_name_count",
    "unique_preval_trans_type_count",
    "unique_calling_type_count",
    "unique_execution_order_count",
    "min_execution_order",
    "max_execution_order",
    "configuration_complexity_proxy",
    "total_complexity_count",
)

CATEGORICAL_FEATURE_COLUMNS = (
    "sprint_name",
    "ldr_connection_name",
)

IDENTITY_COLUMNS = (
    "sprint_name",
    "ldr_connection_name",
    "loader_display_name",
    "datasetname",
)


@dataclass(frozen=True)
class SimilarLoader:
    loader_name: str
    sprint: str
    connection: str
    similarity_score: float


class SimilarLoaderRetrievalError(ValueError):
    """Raised when similar-loader retrieval cannot be performed safely."""


def _normalize_text(value: object) -> str:
    if value is None:
        return ""

    text = str(value).strip().upper()

    if text in {"", "NAN", "NONE", "NULL"}:
        return ""

    return text


def _safe_numeric(value: object) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0

    if not np.isfinite(result):
        return 0.0

    return result


class SimilarLoaderRetriever:
    """
    Retrieves historically similar loader configurations.

    This component provides:
      - top similar historical loaders
      - similarity scores
      - cold-start evidence

    It does NOT replace the selected Global Median cold-start
    prediction unless a future experiment proves that it improves
    predictive accuracy.
    """

    def __init__(
        self,
        feature_path: str | Path = DEFAULT_FEATURE_PATH,
    ) -> None:
        self.feature_path = Path(feature_path)

        if not self.feature_path.exists():
            raise SimilarLoaderRetrievalError(
                "Similarity feature dataset not found: "
                f"{self.feature_path}"
            )

        try:
            frame = pd.read_csv(self.feature_path)
        except Exception as exc:
            raise SimilarLoaderRetrievalError(
                "Failed to load similarity feature dataset: "
                f"{self.feature_path}"
            ) from exc

        required_columns = set(
            NUMERIC_FEATURE_COLUMNS
            + CATEGORICAL_FEATURE_COLUMNS
            + (
                "loader_display_name",
                "datasetname",
            )
        )

        missing = sorted(
            required_columns - set(frame.columns)
        )

        if missing:
            raise SimilarLoaderRetrievalError(
                "Similarity dataset missing columns: "
                f"{missing}"
            )

        frame = frame.copy()

        for column in (
            CATEGORICAL_FEATURE_COLUMNS
            + (
                "loader_display_name",
                "datasetname",
            )
        ):
            frame[column] = frame[column].map(
                _normalize_text
            )

        for column in NUMERIC_FEATURE_COLUMNS:
            frame[column] = pd.to_numeric(
                frame[column],
                errors="coerce",
            ).replace(
                [np.inf, -np.inf],
                np.nan,
            )

        frame[list(NUMERIC_FEATURE_COLUMNS)] = frame[
            list(NUMERIC_FEATURE_COLUMNS)
        ].fillna(0.0)

        frame = frame[
            frame["loader_display_name"].ne("")
        ].reset_index(drop=True)

        if frame.empty:
            raise SimilarLoaderRetrievalError(
                "No valid loader configuration rows are available."
            )

        self._frame = frame
        self._profile_frame = self._build_loader_profiles(
            frame
        )

    @staticmethod
    def _build_loader_profiles(
        frame: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Build one representative configuration profile per
        loader + sprint + connection.

        Numeric values use medians so unusually large configuration
        rows do not dominate similarity.
        """

        group_columns = [
            "loader_display_name",
            "sprint_name",
            "ldr_connection_name",
        ]

        numeric_profile = (
            frame.groupby(
                group_columns,
                dropna=False,
            )[list(NUMERIC_FEATURE_COLUMNS)]
            .median()
            .reset_index()
        )

        return numeric_profile

    def _numeric_distance(
        self,
        query: dict[str, float],
        candidate: pd.Series,
    ) -> float:
        """
        Compute normalized numeric distance.

        Log1p reduces the influence of highly skewed configuration
        counts.
        """

        distances: list[float] = []

        for column in NUMERIC_FEATURE_COLUMNS:
            query_value = max(
                0.0,
                _safe_numeric(
                    query.get(column, 0.0)
                ),
            )

            candidate_value = max(
                0.0,
                _safe_numeric(
                    candidate[column]
                ),
            )

            query_log = np.log1p(
                query_value
            )

            candidate_log = np.log1p(
                candidate_value
            )

            distances.append(
                abs(
                    query_log
                    - candidate_log
                )
            )

        if not distances:
            return 0.0

        return float(
            np.mean(distances)
        )

    @staticmethod
    def _categorical_similarity(
        query: dict[str, str],
        candidate: pd.Series,
    ) -> float:
        """
        Compare contextual categorical fields.

        Sprint and connection are supporting signals rather than
        hard filters.
        """

        comparisons: list[float] = []

        for column in CATEGORICAL_FEATURE_COLUMNS:
            query_value = _normalize_text(
                query.get(column, "")
            )

            candidate_value = _normalize_text(
                candidate[column]
            )

            if not query_value or not candidate_value:
                continue

            comparisons.append(
                1.0
                if query_value == candidate_value
                else 0.0
            )

        if not comparisons:
            return 0.5

        return float(
            np.mean(comparisons)
        )

    def retrieve(
        self,
        *,
        loader_name: str,
        sprint: str,
        loader_connection_name: str,
        dataset_name: str,
        configuration_features: Optional[
            dict[str, float]
        ] = None,
        top_k: int = 3,
    ) -> list[SimilarLoader]:
        """
        Retrieve the top-k historically similar loaders.

        The current loader itself is excluded.

        dataset_name is part of the request contract but is not used
        as a hard filter, allowing useful analogues for a new dataset.
        """

        if top_k < 1:
            raise SimilarLoaderRetrievalError(
                "top_k must be at least 1."
            )

        normalized_loader = _normalize_text(
            loader_name
        )

        normalized_sprint = _normalize_text(
            sprint
        )

        normalized_connection = _normalize_text(
            loader_connection_name
        )

        if not normalized_loader:
            raise SimilarLoaderRetrievalError(
                "loader_name is required."
            )

        if not normalized_sprint:
            raise SimilarLoaderRetrievalError(
                "sprint is required."
            )

        if configuration_features is None:
            configuration_features = {}

        query_numeric = {
            column: _safe_numeric(
                configuration_features.get(
                    column,
                    0.0,
                )
            )
            for column in NUMERIC_FEATURE_COLUMNS
        }

        query_categorical = {
            "sprint_name": normalized_sprint,
            "ldr_connection_name": normalized_connection,
        }

        results: list[SimilarLoader] = []

        for _, candidate in self._profile_frame.iterrows():
            candidate_loader = _normalize_text(
                candidate["loader_display_name"]
            )

            if candidate_loader == normalized_loader:
                continue

            numeric_distance = self._numeric_distance(
                query_numeric,
                candidate,
            )

            categorical_score = (
                self._categorical_similarity(
                    query_categorical,
                    candidate,
                )
            )

            numeric_similarity = 1.0 / (
                1.0 + numeric_distance
            )

            similarity = (
                0.80 * numeric_similarity
                + 0.20 * categorical_score
            )

            similarity = float(
                np.clip(
                    similarity,
                    0.0,
                    1.0,
                )
            )

            results.append(
                SimilarLoader(
                    loader_name=candidate_loader,
                    sprint=_normalize_text(
                        candidate["sprint_name"]
                    ),
                    connection=_normalize_text(
                        candidate[
                            "ldr_connection_name"
                        ]
                    ),
                    similarity_score=similarity,
                )
            )

        results.sort(
            key=lambda item: (
                -item.similarity_score,
                item.loader_name,
            )
        )

        return results[:top_k]

    def profile_count(self) -> int:
        return int(
            len(self._profile_frame)
        )

    def source_row_count(self) -> int:
        return int(
            len(self._frame)
        )


__all__ = [
    "DEFAULT_FEATURE_PATH",
    "NUMERIC_FEATURE_COLUMNS",
    "CATEGORICAL_FEATURE_COLUMNS",
    "IDENTITY_COLUMNS",
    "SimilarLoader",
    "SimilarLoaderRetrievalError",
    "SimilarLoaderRetriever",
]