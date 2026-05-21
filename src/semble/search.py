from __future__ import annotations

from typing import TYPE_CHECKING

import bm25s
import numpy as np
import numpy.typing as npt

from semble.index.sparse import selector_to_mask
from semble.ranking import resolve_alpha
from semble.tokens import tokenize
from semble.types import Chunk, SearchResult

if TYPE_CHECKING:
    from semble.interfaces import EmbeddingProvider, Reranker, VectorStore

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
    chunks: list[Chunk],
    top_k: int,
    selector: npt.NDArray[np.int_] | None,
) -> list[SearchResult]:
    query_embedding = model.encode([query])
    results = semantic_index.query(query_embedding, k=top_k, selector=selector)
    if not results:
        return []
    indices, distances = results[0]
    return [
        SearchResult(chunk=chunks[idx], score=1.0 - float(dist))
        for idx, dist in zip(indices, distances)
        if 0 <= idx < len(chunks)
    ]


def _sort_top_k(arr: npt.NDArray, top_k: int) -> npt.NDArray[np.int_]:
    neg_arr = -arr
    if top_k >= len(arr):
        return np.argsort(neg_arr)
    partitioned = np.argpartition(neg_arr, kth=top_k)[:top_k]
    return partitioned[np.argsort(neg_arr[partitioned])]


def _search_bm25(
    query: str,
    bm25_index: bm25s.BM25,
    chunks: list[Chunk],
    top_k: int,
    selector: npt.NDArray[np.int_] | None,
) -> list[SearchResult]:
    tokens = tokenize(query)
    if not tokens:
        return []
    mask = selector_to_mask(selector, len(chunks))
    scores: npt.NDArray[np.float32] = bm25_index.get_scores(tokens, weight_mask=mask)
    indices = _sort_top_k(scores, top_k)
    return [SearchResult(chunk=chunks[i], score=float(scores[i])) for i in indices if scores[i] > 0]


def search(
    query: str,
    model: "EmbeddingProvider",
    semantic_index: "VectorStore",
    bm25_index: bm25s.BM25,
    chunks: list[Chunk],
    top_k: int,
    alpha: float | None = None,
    selector: npt.NDArray[np.int_] | None = None,
    rerank: bool = True,
    reranker: "Reranker | None" = None,
) -> list[SearchResult]:
    alpha_weight = resolve_alpha(query, alpha)
    candidate_count = top_k * 5

    semantic = _search_semantic(query, model, semantic_index, chunks, candidate_count, selector)
    semantic_scores: dict[Chunk, float] = {result.chunk: result.score for result in semantic}
    bm25_scores: dict[Chunk, float] = {}
    for result in _search_bm25(query, bm25_index, chunks, candidate_count, selector):
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
                query, combined_scores, chunks, top_k, penalise_paths=alpha_weight < 1.0,
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
