from __future__ import annotations

from pathlib import Path

from semble.backends.sparse import IncrementalSparseIndex
from semble.types import chunk_id
from tests.conftest import make_chunk

LOCAL_NAMESPACE = "local"


def _build() -> tuple[IncrementalSparseIndex, list[str]]:
    chunks = [
        make_chunk("def authenticate(token): pass", "auth.py"),
        make_chunk("class UserService: pass", "users.py"),
        make_chunk("def format_date(dt): return dt", "utils.py"),
    ]
    ids = [chunk_id(c, LOCAL_NAMESPACE) for c in chunks]
    idx = IncrementalSparseIndex()
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


def test_incremental_add_and_remove() -> None:
    idx, ids = _build()
    assert idx.doc_count == 3
    extra = make_chunk("def parse_config(path): pass", "config.py")
    extra_id = chunk_id(extra, LOCAL_NAMESPACE)
    idx.add_documents([extra], [extra_id])
    assert idx.doc_count == 4
    hits = idx.query("parse config", k=3)
    assert hits and hits[0][0] == extra_id
    idx.remove_documents([extra_id])
    assert idx.doc_count == 3
    assert all(cid != extra_id for cid, _ in idx.query("parse config", k=5))


def test_save_and_load_roundtrip(tmp_path: Path) -> None:
    idx, ids = _build()
    out = tmp_path / "sparse"
    idx.save(out)

    loaded = IncrementalSparseIndex()
    loaded.load(out)
    hits = loaded.query("authenticate", k=3)
    assert hits
    assert hits[0][0] == ids[0]
