from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from dltime.features.similar_loader_retrieval import (
    SimilarLoader,
    SimilarLoaderRetriever,
)


GLOBAL_MEDIAN_COLD_START_SOURCE = "GLOBAL_MEDIAN_COLD_START"


@dataclass(frozen=True)
class ColdStartConfidence:
    level: str
    score: float
    reason: str


@dataclass(frozen=True)
class ColdStartResult:
    prediction_source: str
    predicted_seconds: float
    confidence: ColdStartConfidence
    similar_loaders: tuple[SimilarLoader, ...]


class ColdStartIntelligence:
    """
    Cold-start intelligence for loaders with no eligible execution history.

    Similar-loader retrieval is used as supporting evidence and
    explainability. The selected V1 prediction remains the global
    historical median because chronological experiments did not show
    a reliable accuracy improvement from similarity blending.

    Confidence is an evidence score, not a calibrated probability.
    """

    def __init__(
        self,
        retriever: Optional[SimilarLoaderRetriever] = None,
    ) -> None:
        self.retriever = retriever or SimilarLoaderRetriever()

    @staticmethod
    def _confidence_from_similar_loaders(
        similar_loaders: list[SimilarLoader],
    ) -> ColdStartConfidence:
        """
        Convert retrieval evidence into a simple confidence indicator.

        This is deliberately conservative:
          - top similarity contributes evidence
          - agreement between top results contributes stability
          - very weak similarity remains low confidence

        The result is NOT interpreted as a probability.
        """

        if not similar_loaders:
            return ColdStartConfidence(
                level="LOW",
                score=0.0,
                reason="No similar historical loader profiles were found.",
            )

        scores = np.asarray(
            [
                float(item.similarity_score)
                for item in similar_loaders
            ],
            dtype=float,
        )

        scores = np.clip(
            np.nan_to_num(scores, nan=0.0),
            0.0,
            1.0,
        )

        top_score = float(scores[0])

        if len(scores) >= 2:
            agreement = float(
                np.mean(scores[: min(3, len(scores))])
            )
        else:
            agreement = top_score

        evidence_score = float(
            0.70 * top_score
            + 0.30 * agreement
        )

        evidence_score = float(
            np.clip(evidence_score, 0.0, 1.0)
        )

        if evidence_score >= 0.85:
            level = "HIGH"
            reason = (
                "Strong configuration similarity to historical loaders."
            )
        elif evidence_score >= 0.65:
            level = "MEDIUM"
            reason = (
                "Moderate configuration similarity to historical loaders."
            )
        else:
            level = "LOW"
            reason = (
                "Weak similarity evidence; cold-start uncertainty remains high."
            )

        return ColdStartConfidence(
            level=level,
            score=round(evidence_score, 6),
            reason=reason,
        )

    def evaluate(
        self,
        *,
        predicted_seconds: float,
        loader_name: str,
        sprint: str,
        loader_connection_name: str,
        dataset_name: str,
        configuration_features: Optional[dict[str, float]] = None,
        top_k: int = 3,
    ) -> ColdStartResult:
        """
        Produce the cold-start prediction contract.

        The prediction itself remains the supplied global-median
        estimate. Similar loaders and confidence are attached as
        supporting intelligence.
        """

        try:
            prediction = float(predicted_seconds)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "predicted_seconds must be numeric."
            ) from exc

        if not np.isfinite(prediction) or prediction < 0:
            raise ValueError(
                "predicted_seconds must be finite and non-negative."
            )

        similar_loaders = self.retriever.retrieve(
            loader_name=loader_name,
            sprint=sprint,
            loader_connection_name=loader_connection_name,
            dataset_name=dataset_name,
            configuration_features=configuration_features,
            top_k=top_k,
        )

        confidence = self._confidence_from_similar_loaders(
            similar_loaders
        )

        return ColdStartResult(
            prediction_source=GLOBAL_MEDIAN_COLD_START_SOURCE,
            predicted_seconds=prediction,
            confidence=confidence,
            similar_loaders=tuple(similar_loaders),
        )