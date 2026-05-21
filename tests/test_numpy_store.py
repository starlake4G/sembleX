from __future__ import annotations

from pathlib import Path

import numpy as np

from semble.backends.vector_store.numpy_store import NumpyVectorStore
from semble.interfaces import LOCAL_NAMESPACE


def _unit(rows: int, dim: int = 4, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    arr = rng.standard_normal((rows, dim)).astype(np.float32)
    arr /= np.linalg.norm(arr, axis=1, keepdims=True) + 1e-9
    return arr.astype(np.float32)


def test_add_and_query_returns_chunk_ids() -> None:
    vs = NumpyVectorStore(dim=4)
    vecs = _unit(3)
    vs.add(LOCAL_NAMESPACE, ["a", "b", "c"], vecs)
    hits = vs.query(LOCAL_NAMESPACE, vecs[0:1], k=2)
    assert [cid for cid, _ in hits] == ["a", "b"] or [cid for cid, _ in hits[:1]] == ["a"]


def test_selector_restricts_results() -> None:
    vs = NumpyVectorStore(dim=4)
    vs.add(LOCAL_NAMESPACE, ["a", "b", "c"], _unit(3))
    hits = vs.query(LOCAL_NAMESPACE, _unit(1, seed=99), k=5, selector_ids=["b"])
    assert {cid for cid, _ in hits} == {"b"}


def test_delete_removes_chunks() -> None:
    vs = NumpyVectorStore(dim=4)
    vs.add(LOCAL_NAMESPACE, ["a", "b", "c"], _unit(3))
    vs.delete(LOCAL_NAMESPACE, ["b"])
    hits = vs.query(LOCAL_NAMESPACE, _unit(1, seed=99), k=5)
    assert {cid for cid, _ in hits} == {"a", "c"}


def test_clear_resets() -> None:
    vs = NumpyVectorStore(dim=4)
    vs.add(LOCAL_NAMESPACE, ["a"], _unit(1))
    vs.clear(LOCAL_NAMESPACE)
    assert vs.query(LOCAL_NAMESPACE, _unit(1, seed=99), k=5) == []


def test_duplicate_chunk_id_rejected() -> None:
    vs = NumpyVectorStore(dim=4)
    vs.add(LOCAL_NAMESPACE, ["a"], _unit(1))
    try:
        vs.add(LOCAL_NAMESPACE, ["a"], _unit(1, seed=99))
    except ValueError as exc:
        assert "already present" in str(exc)
    else:
        raise AssertionError("expected ValueError for duplicate chunk_id")


def test_save_load_roundtrip(tmp_path: Path) -> None:
    vs = NumpyVectorStore(dim=4)
    vs.add(LOCAL_NAMESPACE, ["a", "b"], _unit(2))
    vs.save(tmp_path / "vs")

    loaded = NumpyVectorStore(dim=4)
    loaded.load(tmp_path / "vs")
    assert {cid for cid, _ in loaded.query(LOCAL_NAMESPACE, _unit(1), k=5)} == {"a", "b"}
