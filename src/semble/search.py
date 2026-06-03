from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from semble.ranking import apply_query_boost, boost_multi_chunk_files, rerank_topk, resolve_alpha
from semble.types import Chunk, SearchResult

_RRF_K = 60


@dataclass(frozen=True, slots=True)
class Candidate:
    """A retrieval candidate resolved to its chunk, source repo, and fused score."""

    chunk_id: str
    repo_id: str
    repo_name: str
    repo_source: str
    chunk: Chunk
    score: float


def _rrf(ranked_ids: list[str]) -> dict[str, float]:
    """Reciprocal-rank-fusion weights for a ranked list of chunk ids."""
    return {cid: 1.0 / (_RRF_K + rank) for rank, cid in enumerate(ranked_ids, 1)}


def fuse_hybrid(
    dense: list[tuple[str, float]],
    sparse: list[tuple[str, float]],
    alpha: float,
) -> dict[str, float]:
    """Fuse dense and sparse ranked lists by chunk_id using RRF + alpha blending."""
    dense_rrf = _rrf([cid for cid, _ in dense])
    sparse_rrf = _rrf([cid for cid, _ in sparse])
    fused: dict[str, float] = {}
    for cid in dense_rrf.keys() | sparse_rrf.keys():
        fused[cid] = alpha * dense_rrf.get(cid, 0.0) + (1.0 - alpha) * sparse_rrf.get(cid, 0.0)
    return fused


def rank_candidates(
    query: str | None,
    candidates: list[Candidate],
    top_k: int,
    *,
    apply_query_boosts: bool,
) -> list[SearchResult]:
    """Rank candidates by running the code-aware ranking per source repo, then merging.

    Grouping by repo keeps file-coherence boosts and file-saturation penalties
    repo-scoped (a ``src/main.py`` in repo A must not be conflated with one in
    repo B), and avoids cross-repo collisions of equal ``Chunk`` objects.

    :param query: The query string, or None for find_related (no query-text boosts).
    :param candidates: Fused candidates with resolved chunks and repo attribution.
    :param top_k: Number of merged results to return.
    :param apply_query_boosts: Whether to apply symbol/stem query boosts (search only).
    :return: Globally merged, ranked results with repo attribution.
    """
    if not candidates:
        return []

    by_repo: dict[str, list[Candidate]] = defaultdict(list)
    for cand in candidates:
        by_repo[cand.repo_id].append(cand)

    merged: list[SearchResult] = []
    for repo_id, repo_cands in by_repo.items():
        repo_name = repo_cands[0].repo_name
        repo_source = repo_cands[0].repo_source
        scores: dict[Chunk, float] = {c.chunk: c.score for c in repo_cands}
        if apply_query_boosts and query:
            boost_multi_chunk_files(scores)
            scores = apply_query_boost(scores, query, list(scores))
        ranked = rerank_topk(scores, len(scores), penalise_paths=True)
        for chunk, score in ranked:
            merged.append(
                SearchResult(
                    chunk=chunk,
                    score=score,
                    repo_id=repo_id,
                    repo_name=repo_name,
                    repo_source=repo_source,
                )
            )

    merged.sort(key=lambda r: -r.score)
    return merged[:top_k]


def alpha_for(query: str) -> float:
    """Resolve the dense/sparse blend weight for *query* (symbol vs natural language)."""
    return resolve_alpha(query, None)
