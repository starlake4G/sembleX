from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from semble.backends.sparse import Bm25sSparseIndex
from semble.backends.vector_store.numpy_store import NumpyVectorStore
from semble.interfaces import LOCAL_NAMESPACE
from semble.search import _search_bm25, _search_semantic, search
from semble.types import Chunk, chunk_id
from tests.conftest import make_chunk


@pytest.fixture
def chunks() -> list[Chunk]:
    """Four small code chunks covering authentication, login, user service, and utils."""
    return [
        make_chunk("def authenticate(token):\n    return token == 'secret'", "auth.py"),
        make_chunk("def login(username, password):\n    pass", "auth.py"),
        make_chunk("class UserService:\n    pass", "users.py"),
        make_chunk("def format_date(dt):\n    return str(dt)", "utils.py"),
    ]


@pytest.fixture
def chunk_ids(chunks: list[Chunk]) -> list[str]:
    return [chunk_id(c, LOCAL_NAMESPACE) for c in chunks]


@pytest.fixture
def by_id(chunks: list[Chunk], chunk_ids: list[str]) -> dict[str, Chunk]:
    return dict(zip(chunk_ids, chunks))


@pytest.fixture
def embeddings(chunks: list[Chunk]) -> npt.NDArray[np.float32]:
    rng = np.random.default_rng(0)
    embs = rng.standard_normal((len(chunks), 256)).astype(np.float32)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    return (embs / (norms + 1e-8)).astype(np.float32)


@pytest.fixture
def sparse(chunks: list[Chunk], chunk_ids: list[str]) -> Bm25sSparseIndex:
    idx = Bm25sSparseIndex()
    idx.build(chunks, chunk_ids)
    return idx


@pytest.fixture
def semantic(
    embeddings: npt.NDArray[np.float32], chunk_ids: list[str]
) -> NumpyVectorStore:
    vs = NumpyVectorStore(dim=embeddings.shape[1])
    vs.add(LOCAL_NAMESPACE, chunk_ids, embeddings)
    return vs


def test_sparse_returns_relevant(
    sparse: Bm25sSparseIndex, by_id: dict[str, Chunk], chunks: list[Chunk]
) -> None:
    """BM25 returns most-relevant chunk first."""
    results = _search_bm25("authenticate token", sparse, by_id, top_k=4, selector_ids=None)
    assert results
    assert "authenticate" in results[0].chunk.content


def test_sparse_selector_restricts(
    sparse: Bm25sSparseIndex, by_id: dict[str, Chunk], chunks: list[Chunk], chunk_ids: list[str]
) -> None:
    """selector_ids restricts BM25 results to the supplied chunk_ids."""
    only = [chunk_ids[-1]]
    results = _search_bm25("format", sparse, by_id, top_k=4, selector_ids=only)
    assert all(r.chunk is chunks[-1] for r in results)


@pytest.mark.parametrize("query", ["", "   ", "\n\n", "zzzznonexistentterm"])
def test_sparse_empty_for_no_match(sparse: Bm25sSparseIndex, by_id: dict[str, Chunk], query: str) -> None:
    assert _search_bm25(query, sparse, by_id, top_k=3, selector_ids=None) == []


def test_semantic_search(
    semantic: NumpyVectorStore, by_id: dict[str, Chunk], mock_model: Any
) -> None:
    results = _search_semantic("login", mock_model, semantic, LOCAL_NAMESPACE, by_id, top_k=3, selector_ids=None)
    assert results
    assert all(-1.0 <= r.score <= 1.0 for r in results)


def test_search_hybrid(
    chunks: list[Chunk],
    chunk_ids: list[str],
    by_id: dict[str, Chunk],
    semantic: NumpyVectorStore,
    sparse: Bm25sSparseIndex,
    mock_model: Any,
) -> None:
    results = search(
        "authenticate token", mock_model, semantic, sparse, chunks, by_id,
        top_k=3, namespace=LOCAL_NAMESPACE,
    )
    assert results

    # Identical content in different files should not be deduplicated.
    shared_content = "def helper():\n    pass"
    a = make_chunk(shared_content, "module_a.py")
    b = make_chunk(shared_content, "module_b.py")
    pair = [a, b]
    pair_ids = [chunk_id(c, LOCAL_NAMESPACE) for c in pair]
    pair_by_id = dict(zip(pair_ids, pair))

    rng = np.random.default_rng(1)
    embs = rng.standard_normal((2, 256)).astype(np.float32)
    embs /= np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8

    vs = NumpyVectorStore(dim=256)
    vs.add(LOCAL_NAMESPACE, pair_ids, embs)
    sp = Bm25sSparseIndex()
    sp.build(pair, pair_ids)

    deduped = search(
        "helper", mock_model, vs, sp, pair, pair_by_id, top_k=5, namespace=LOCAL_NAMESPACE,
    )
    locations = {r.chunk.file_path for r in deduped}
    assert "module_a.py" in locations
    assert "module_b.py" in locations
