from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from semble.ranking import resolve_alpha
from semble.types import Chunk, SearchResult

if TYPE_CHECKING:
    from semble.interfaces import EmbeddingProvider, Reranker, SparseIndex, VectorStore

_RRF_K = 60


def _rrf_scores(scores: dict[Chunk, float]) -> dict[Chunk, float]:
    if not scores:
        return scores
    ranked = sorted(scores, key=lambda c: -scores[c])
    return {chunk: 1.0 / (_RRF_K + rank) for rank, chunk in enumerate(ranked, 1)}


def _search_semantic(
    query: str,
    model: "EmbeddingProvider",
    semantic_index: "VectorStore",
    namespace: str,
    by_id: dict[str, Chunk],
    top_k: int,
    selector_ids: Sequence[str] | None,
) -> list[SearchResult]:
    query_embedding = model.encode([query])
    hits = semantic_index.query(namespace, query_embedding, k=top_k, selector_ids=selector_ids)
    return [
        SearchResult(chunk=by_id[cid], score=float(score))
        for cid, score in hits
        if cid in by_id
    ]


def _search_bm25(
    query: str,
    sparse_index: "SparseIndex",
    by_id: dict[str, Chunk],
    top_k: int,
    selector_ids: Sequence[str] | None,
) -> list[SearchResult]:
    hits = sparse_index.query(query, k=top_k, selector_ids=selector_ids)
    return [
        SearchResult(chunk=by_id[cid], score=float(score))
        for cid, score in hits
        if cid in by_id
    ]


def search(
    query: str,
    model: "EmbeddingProvider",
    semantic_index: "VectorStore",
    sparse_index: "SparseIndex",
    chunks: list[Chunk],
    by_id: dict[str, Chunk],
    top_k: int,
    namespace: str,
    alpha: float | None = None,
    selector_ids: Sequence[str] | None = None,
    rerank: bool = True,
    reranker: "Reranker | None" = None,
    coarse_k: int | None = None,
) -> list[SearchResult]:
    alpha_weight = resolve_alpha(query, alpha)
    candidate_count = top_k * 5

    semantic = _search_semantic(
        query, model, semantic_index, namespace, by_id, candidate_count, selector_ids,
    )
    semantic_scores: dict[Chunk, float] = {result.chunk: result.score for result in semantic}

    bm25_scores: dict[Chunk, float] = {}
    for result in _search_bm25(query, sparse_index, by_id, candidate_count, selector_ids):
        if result.score:
            bm25_scores[result.chunk] = result.score

    normalized_semantic = _rrf_scores(semantic_scores)
    normalized_bm25 = _rrf_scores(bm25_scores)

    all_candidates = sorted(
        {*normalized_semantic, *normalized_bm25},
        key=lambda c: c.start_line,
    )
    combined_scores: dict[Chunk, float] = {
        chunk: alpha_weight * normalized_semantic.get(chunk, 0.0)
        + (1.0 - alpha_weight) * normalized_bm25.get(chunk, 0.0)
        for chunk in all_candidates
    }

    if rerank:
        if reranker is not None:
            ranked = reranker.rerank(
                query,
                combined_scores,
                chunks,
                top_k,
                penalise_paths=alpha_weight < 1.0,
                coarse_k=coarse_k,
            )
        else:
            from semble.ranking import apply_query_boost, boost_multi_chunk_files, rerank_topk

            boost_multi_chunk_files(combined_scores)
            combined_scores = apply_query_boost(combined_scores, query, chunks)
            ranked = rerank_topk(combined_scores, top_k, penalise_paths=alpha_weight < 1.0)
    else:
        sorted_by_score = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        ranked = sorted_by_score[:top_k]

    return [SearchResult(chunk=chunk, score=score) for chunk, score in ranked]
