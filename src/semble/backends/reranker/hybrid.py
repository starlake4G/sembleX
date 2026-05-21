from __future__ import annotations

import logging

from semble.backends.reranker.cross_encoder import CrossEncoderReranker
from semble.backends.reranker.rules import RulesReranker
from semble.interfaces import Chunk, Reranker

logger = logging.getLogger(__name__)


class HybridReranker(Reranker):
    def __init__(
        self,
        rules: RulesReranker,
        model: CrossEncoderReranker,
        coarse_multiplier: int = 5,
    ) -> None:
        self._rules = rules
        self._model = model
        self._coarse_multiplier = coarse_multiplier

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

        coarse_k = top_k * self._coarse_multiplier
        coarse = self._rules.rerank(
            query, combined_scores, all_chunks, coarse_k, penalise_paths=penalise_paths
        )

        fine_input = {chunk: score for chunk, score in coarse}
        return self._model.rerank(query, fine_input, all_chunks, top_k)
