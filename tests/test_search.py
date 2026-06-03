from __future__ import annotations

from semble.search import Candidate, fuse_hybrid, rank_candidates
from semble.types import chunk_id
from tests.conftest import make_chunk


def _cand(content: str, path: str, repo_id: str, repo_name: str, score: float) -> Candidate:
    chunk = make_chunk(content, path)
    return Candidate(
        chunk_id=chunk_id(chunk, repo_id),
        repo_id=repo_id,
        repo_name=repo_name,
        repo_source=repo_name,
        chunk=chunk,
        score=score,
    )


def test_fuse_hybrid_blends_both_retrievers() -> None:
    dense = [("a", 0.9), ("b", 0.5)]
    sparse = [("b", 3.0), ("c", 1.0)]
    fused = fuse_hybrid(dense, sparse, alpha=0.5)
    assert set(fused) == {"a", "b", "c"}
    # "b" is ranked by both retrievers, so it should outscore single-retriever hits.
    assert fused["b"] > fused["a"]
    assert fused["b"] > fused["c"]


def test_fuse_hybrid_alpha_extremes() -> None:
    dense = [("a", 1.0)]
    sparse = [("b", 1.0)]
    only_dense = fuse_hybrid(dense, sparse, alpha=1.0)
    assert only_dense["a"] > 0 and only_dense["b"] == 0.0
    only_sparse = fuse_hybrid(dense, sparse, alpha=0.0)
    assert only_sparse["b"] > 0 and only_sparse["a"] == 0.0


def test_rank_candidates_attributes_and_truncates() -> None:
    cands = [
        _cand("def authenticate(token): pass", "auth.py", "r1", "repoA", 0.9),
        _cand("class UserService: pass", "users.py", "r2", "repoB", 0.5),
    ]
    results = rank_candidates("authenticate", cands, top_k=1, apply_query_boosts=True)
    assert len(results) == 1
    assert results[0].repo_id == "r1"
    assert results[0].repo_name == "repoA"


def test_rank_candidates_does_not_collapse_across_repos() -> None:
    shared = "def helper():\n    pass"
    cands = [
        _cand(shared, "module.py", "r1", "repoA", 0.8),
        _cand(shared, "module.py", "r2", "repoB", 0.8),
    ]
    results = rank_candidates("helper", cands, top_k=5, apply_query_boosts=True)
    assert len(results) == 2
    assert {r.repo_id for r in results} == {"r1", "r2"}


def test_rank_candidates_empty() -> None:
    assert rank_candidates("x", [], top_k=3, apply_query_boosts=True) == []


def test_rank_candidates_find_related_without_query() -> None:
    cands = [_cand("def f(): pass", "f.py", "r1", "repoA", 0.7)]
    results = rank_candidates(None, cands, top_k=3, apply_query_boosts=False)
    assert len(results) == 1
    assert results[0].repo_id == "r1"
