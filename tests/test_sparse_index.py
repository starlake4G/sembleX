from __future__ import annotations

from pathlib import Path

from semble.backends.sparse import Bm25sSparseIndex
from semble.interfaces import LOCAL_NAMESPACE
from semble.types import chunk_id
from tests.conftest import make_chunk


def _build() -> tuple[Bm25sSparseIndex, list[str]]:
    chunks = [
        make_chunk("def authenticate(token): pass", "auth.py"),
        make_chunk("class UserService: pass", "users.py"),
        make_chunk("def format_date(dt): return dt", "utils.py"),
    ]
    ids = [chunk_id(c, LOCAL_NAMESPACE) for c in chunks]
    idx = Bm25sSparseIndex()
    idx.build(chunks, ids)
    return idx, ids


def test_query_returns_relevant_chunk() -> None:
    idx, ids = _build()
    hits = idx.query("authenticate token", k=3)
    assert hits
    assert hits[0][0] == ids[0]


def test_query_with_selector_restricts() -> None:
    idx, ids = _build()
    hits = idx.query("format", k=3, selector_ids=[ids[-1]])
    assert all(cid == ids[-1] for cid, _ in hits)


def test_empty_query_returns_empty() -> None:
    idx, _ = _build()
    assert idx.query("", k=3) == []


def test_save_and_load_roundtrip(tmp_path: Path) -> None:
    idx, ids = _build()
    out = tmp_path / "sparse"
    idx.save(out)

    loaded = Bm25sSparseIndex()
    loaded.load(out)
    hits = loaded.query("authenticate", k=3)
    assert hits
    assert hits[0][0] == ids[0]
