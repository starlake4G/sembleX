from semble.backends.sparse.bm25s_index import Bm25sSparseIndex
from semble.interfaces import SparseIndex


def create_sparse_index() -> SparseIndex:
    """Default sparse index factory (currently only bm25s)."""
    return Bm25sSparseIndex()


__all__ = ["Bm25sSparseIndex", "SparseIndex", "create_sparse_index"]
