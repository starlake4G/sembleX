from __future__ import annotations

from semble.interfaces import Chunk, Reranker
from semble.ranking import apply_query_boost, boost_multi_chunk_files, rerank_topk


class RulesReranker(Reranker):
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

        boosted = dict(combined_scores)
        boost_multi_chunk_files(boosted)
        boosted = apply_query_boost(boosted, query, all_chunks)
        return rerank_topk(boosted, top_k, penalise_paths=penalise_paths)
