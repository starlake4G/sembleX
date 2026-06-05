"""End-to-end test of GlobalIndex with an in-memory fake Milvus + deterministic embedder.

Validates the cross-repo engine logic (index -> hybrid search -> cross-repo
find_related with source-repo exclusion) without any external services.
"""
from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest

from semble.config import SembleConfig
from semble.interfaces import EmbeddingMatrix

_DIM = 32


class FakeEmbedder:
    """Deterministic embedder: identical text -> identical unit vector (so similar code matches)."""

    name = "fake"

    @property
    def dim(self) -> int:
        return _DIM

    def encode(self, texts: Sequence[str]) -> EmbeddingMatrix:
        rows = []
        for text in texts:
            seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")
            vec = np.random.default_rng(seed).standard_normal(_DIM).astype(np.float32)
            vec /= np.linalg.norm(vec) + 1e-9
            rows.append(vec)
        return np.asarray(rows, dtype=np.float32)


class FakeVectorStore:
    """In-memory multi-namespace cosine store implementing the VectorStore surface used by GlobalIndex."""

    def __init__(self, config: object, dim: int) -> None:
        del config
        self._dim = dim
        self._data: dict[str, dict[str, np.ndarray]] = {}

    @property
    def dim(self) -> int:
        return self._dim

    def add(self, namespace: str, chunk_ids: Sequence[str], vectors: EmbeddingMatrix) -> None:
        ns = self._data.setdefault(namespace, {})
        for cid, vec in zip(chunk_ids, np.asarray(vectors, dtype=np.float32)):
            ns[cid] = vec

    def delete(self, namespace: str, chunk_ids: Sequence[str]) -> None:
        ns = self._data.get(namespace, {})
        for cid in chunk_ids:
            ns.pop(cid, None)

    def clear(self, namespace: str) -> None:
        self._data.pop(namespace, None)

    def _rank(self, pairs: list[tuple[str, str, float]], k: int) -> list[tuple[str, str, float]]:
        pairs.sort(key=lambda t: -t[2])
        return pairs[:k]

    def query(self, namespace, vector, k, selector_ids=None):  # noqa: ANN001, ANN201
        q = np.asarray(vector, dtype=np.float32).reshape(-1)
        sel = set(selector_ids) if selector_ids is not None else None
        pairs = [
            (cid, namespace, float(np.dot(q, vec)))
            for cid, vec in self._data.get(namespace, {}).items()
            if sel is None or cid in sel
        ]
        return [(cid, score) for cid, _ns, score in self._rank(pairs, k)]

    def query_global(self, vector, k, exclude_namespace=None):  # noqa: ANN001, ANN201
        q = np.asarray(vector, dtype=np.float32).reshape(-1)
        pairs: list[tuple[str, str, float]] = []
        for ns, items in self._data.items():
            if ns == exclude_namespace:
                continue
            pairs.extend((cid, ns, float(np.dot(q, vec))) for cid, vec in items.items())
        return self._rank(pairs, k)


_AUTH_CODE = "def authenticate(token):\n    return token == 'secret'\n"


@pytest.fixture
def index(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):  # noqa: ANN201
    monkeypatch.setattr("semble.server.indexer.create_embedding_provider", lambda cfg: FakeEmbedder())
    monkeypatch.setattr("semble.server.indexer.MilvusVectorStore", FakeVectorStore)
    from semble.server.indexer import GlobalIndex

    cfg = SembleConfig(core={"dir": str(tmp_path / "core")})
    return GlobalIndex(cfg)


def _make_repo(root: Path, name: str, extra: str = "") -> Path:
    repo = root / name
    repo.mkdir(parents=True)
    (repo / "auth.py").write_text(_AUTH_CODE + extra)
    return repo


def test_index_search_and_cross_repo_find_related(index, tmp_path: Path) -> None:  # noqa: ANN001
    repo_a = _make_repo(tmp_path / "src", "repo_a")
    repo_b = _make_repo(tmp_path / "src", "repo_b", extra="\ndef other():\n    return 1\n")

    out_a = index.index_repo(str(repo_a))
    out_b = index.index_repo(str(repo_b))
    assert out_a.indexed and out_a.chunk_count >= 1
    assert out_b.indexed and out_b.chunk_count >= 1

    # Re-indexing without --force is a no-op.
    assert index.index_repo(str(repo_a)).indexed is False

    # Cross-repo search attributes results to their source repos.
    results = index.search("authenticate token", top_k=10)
    assert results
    repos = {r.repo_name for r in results}
    assert {"repo_a", "repo_b"} <= repos
    assert any("authenticate" in r.chunk.content for r in results)

    # Single-repo restriction (debug path) only returns the named repo.
    only_a = index.search("authenticate", top_k=10, repo=str(repo_a))
    assert only_a
    assert {r.repo_name for r in only_a} == {"repo_a"}

    # Cross-repo find_related from repo_a finds the identical implementation in repo_b and excludes repo_a.
    related = index.find_related("auth.py", 1, repo=str(repo_a), top_k=5)
    assert related
    assert all(r.repo_name != "repo_a" for r in related)
    assert any(r.repo_name == "repo_b" and "authenticate" in r.chunk.content for r in related)


def test_workspace_scope_resolves_subdirectory(index, tmp_path: Path) -> None:  # noqa: ANN001
    from semble.server.metadata import RepoNotIndexedError

    repo_a = _make_repo(tmp_path / "src", "repo_a")
    _make_repo(tmp_path / "src", "repo_b")
    index.index_repo(str(repo_a))
    index.index_repo(str(tmp_path / "src" / "repo_b"))

    # A subdirectory under repo_a resolves to repo_a.
    subdir = repo_a / "nested"
    subdir.mkdir()
    assert index.resolve_repo_id(str(subdir)) == index.resolve_repo_id(str(repo_a))

    # Restricting search to repo_a (via a path under it) only returns repo_a.
    results = index.search("authenticate", top_k=10, repo=str(subdir))
    assert results and {r.repo_name for r in results} == {"repo_a"}

    # An unindexed path raises so callers can fall back to global.
    with pytest.raises(RepoNotIndexedError):
        index.search("authenticate", repo=str(tmp_path / "not_indexed"))


def test_deferred_persist_defers_bm25_save_until_flush(index, tmp_path: Path) -> None:  # noqa: ANN001
    repo_a = _make_repo(tmp_path / "src", "repo_a")
    marker = index._sparse_marker()
    before = marker.stat().st_mtime

    index.index_repo(str(repo_a), persist=False)
    # The repo's chunks are in memory and searchable, but the BM25 file is untouched.
    assert index._sparse.doc_count >= 1
    assert marker.stat().st_mtime == before

    index.flush()
    assert marker.stat().st_mtime > before


def test_force_reindex_refreshes_missing_vectors(index, tmp_path: Path) -> None:  # noqa: ANN001
    repo = _make_repo(tmp_path / "src", "repo")
    index.index_repo(str(repo))
    repo_id = index.resolve_repo_id(str(repo))
    assert repo_id is not None
    assert index._vector_store._data[repo_id]

    index._vector_store._data[repo_id].clear()
    assert not index._vector_store._data[repo_id]

    outcome = index.index_repo(str(repo), force=True)
    assert outcome.indexed is True
    assert index._vector_store._data[repo_id]


def test_empty_repo_reindex_removes_stale_chunks(index, tmp_path: Path) -> None:  # noqa: ANN001
    repo = _make_repo(tmp_path / "src", "repo")
    index.index_repo(str(repo))
    assert index.search("authenticate", top_k=5, repo=str(repo))

    (repo / "auth.py").unlink()
    with pytest.raises(ValueError, match="No supported files"):
        index.index_repo(str(repo))

    assert index._metadata.total_chunk_count() == 0
    assert index._sparse.doc_count == 0
    assert index.search("authenticate", top_k=5) == []


def test_load_rebuilds_bm25_when_inconsistent_with_metadata(index, tmp_path: Path) -> None:  # noqa: ANN001
    from semble.server.indexer import GlobalIndex

    repo_a = _make_repo(tmp_path / "src", "repo_a")
    # Index without persisting and without flushing: metadata (SQLite) has the chunks
    # but the on-disk BM25 index does not — i.e. an interrupted deferred-save batch.
    index.index_repo(str(repo_a), persist=False)
    expected = index._metadata.total_chunk_count()
    assert expected >= 1

    # A fresh index over the same core dir must self-heal by rebuilding from metadata.
    reopened = GlobalIndex(SembleConfig(core={"dir": str(tmp_path / "core")}))
    assert reopened._sparse.doc_count == expected


def test_status_reports_repos_and_chunks(index, tmp_path: Path) -> None:  # noqa: ANN001
    repo = _make_repo(tmp_path / "src", "solo")
    index.index_repo(str(repo))
    status = index.status()
    assert status["repos"] == 1
    assert status["chunks"] >= 1
    assert str(repo) in status["sources"]
