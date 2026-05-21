from __future__ import annotations

import logging

from semble.interfaces import Chunk, Reranker

logger = logging.getLogger(__name__)


class CrossEncoderReranker(Reranker):
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3") -> None:
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        combined_scores: dict[Chunk, float],
        all_chunks: list[Chunk],
        top_k: int,
        *,
        penalise_paths: bool = True,
    ) -> list[tuple[Chunk, float]]:
        if not combined_scores:
            return []

        candidates = sorted(combined_scores.items(), key=lambda x: -x[1])
        candidates = candidates[: top_k * 5]

        pairs = [(query, chunk.content) for chunk, _ in candidates]
        scores = self._model.predict(pairs)

        ranked = sorted(zip(candidates, scores), key=lambda x: -x[1])
        return [(chunk, float(score)) for (chunk, _), score in ranked[:top_k]]
