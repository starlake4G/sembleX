from __future__ import annotations

from pathlib import Path

import pytest

from semble.server.metadata import MetadataStore, RepoNotIndexedError, StoredChunk
from semble.types import Chunk


@pytest.fixture
def store(tmp_path: Path) -> MetadataStore:
    return MetadataStore(tmp_path / "metadata.sqlite3")


def _stored(chunk_id: str, content: str, file_path: str, start: int, end: int) -> StoredChunk:
    return StoredChunk(
        chunk_id=chunk_id,
        chunk=Chunk(content=content, file_path=file_path, start_line=start, end_line=end, language="python"),
        content_hash="h",
    )


def test_initially_empty(store: MetadataStore) -> None:
    assert store.has_repo("r1") is False
    assert store.repo_chunk_count("r1") == 0
    assert store.repo_ids_for_source("https://x") == []


def test_insert_and_retrieve(store: MetadataStore) -> None:
    chunks = [_stored("c1", "x", "a.py", 1, 5), _stored("c2", "y", "a.py", 6, 10)]
    store.insert_chunks("r1", chunks)
    store.finish_repo_replace(
        repo_id="r1",
        source="https://x",
        source_path="/tmp/r1",
        include_text_files=False,
        embedding_model="m",
        embedding_dim=4,
        chunker_version="v1",
        commit_sha="abc",
        indexed_at="2026-05-21T00:00:00Z",
        chunk_count=2,
    )
    assert store.has_repo("r1") is True
    assert store.repo_chunk_count("r1") == 2
    assert store.repo_ids_for_source("https://x") == ["r1"]
    out = store.all_chunks("r1")
    assert [cid for cid, _ in out] == ["c1", "c2"]


def test_find_chunk_picks_smallest_enclosing(store: MetadataStore) -> None:
    store.insert_chunks("r1", [
        _stored("big", "x", "a.py", 1, 50),
        _stored("small", "y", "a.py", 10, 20),
    ])
    cid, _ = store.find_chunk("r1", "a.py", 15)
    assert cid == "small"


def test_find_chunk_missing(store: MetadataStore) -> None:
    assert store.find_chunk("r1", "missing.py", 1) is None


def test_start_repo_replace_clears(store: MetadataStore) -> None:
    store.insert_chunks("r1", [_stored("c1", "x", "a.py", 1, 5)])
    store.finish_repo_replace(
        repo_id="r1",
        source="src",
        source_path="/tmp/r1",
        include_text_files=False,
        embedding_model="m",
        embedding_dim=4,
        chunker_version="v1",
        commit_sha=None,
        indexed_at="t",
        chunk_count=1,
    )
    store.start_repo_replace("r1")
    assert store.has_repo("r1") is False
    assert store.all_chunks("r1") == []


def test_repo_not_indexed_error_carries_source() -> None:
    exc = RepoNotIndexedError("rid", source="https://x")
    assert exc.repo_id == "rid"
    assert exc.source == "https://x"
