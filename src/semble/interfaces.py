from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import numpy.typing as npt

from semble.types import Chunk

EmbeddingMatrix = npt.NDArray[np.float32]

LOCAL_NAMESPACE = "local"


class EmbeddingFailure(RuntimeError):
    """Raised when an embedding provider cannot encode some or all inputs.

    Always carries the indices (into the original input sequence) that failed.
    For partial failures, ``partial`` holds the successfully encoded rows
    (in original input order, with failed rows omitted) so callers can salvage
    the work that succeeded.
    """

    def __init__(
        self,
        message: str,
        failed_indices: Sequence[int],
        partial: "EmbeddingMatrix | None" = None,
    ) -> None:
        super().__init__(message)
        self.failed_indices: list[int] = list(failed_indices)
        self.partial: "EmbeddingMatrix | None" = partial


class EmbeddingProvider(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def dim(self) -> int: ...

    @abstractmethod
    def encode(self, texts: Sequence[str]) -> EmbeddingMatrix: ...


class VectorStore(ABC):
    """Namespace + chunk_id keyed dense vector store.

    Local backends (numpy/faiss) operate with a single namespace ("local").
    Remote backends like Milvus use one namespace per indexed repository.

    All identifiers are caller-supplied stable strings; backends keep the
    chunk_id→vector mapping themselves so callers never reason about row
    positions.
    """

    @property
    @abstractmethod
    def dim(self) -> int: ...

    @abstractmethod
    def add(
        self, namespace: str, chunk_ids: Sequence[str], vectors: EmbeddingMatrix
    ) -> None: ...

    @abstractmethod
    def delete(self, namespace: str, chunk_ids: Sequence[str]) -> None: ...

    @abstractmethod
    def clear(self, namespace: str) -> None: ...

    @abstractmethod
    def query(
        self,
        namespace: str,
        vector: EmbeddingMatrix,
        k: int,
        selector_ids: Sequence[str] | None = None,
    ) -> list[tuple[str, float]]:
        """Return ``[(chunk_id, similarity)]`` sorted by similarity descending."""
        ...

    @abstractmethod
    def save(self, path: Path) -> None: ...

    @abstractmethod
    def load(self, path: Path) -> None: ...


class SparseIndex(ABC):
    """Sparse (BM25-style) index keyed by chunk_id."""

    @abstractmethod
    def build(self, chunks: Sequence[Chunk], chunk_ids: Sequence[str]) -> None: ...

    @abstractmethod
    def query(
        self,
        query: str,
        k: int,
        selector_ids: Sequence[str] | None = None,
    ) -> list[tuple[str, float]]: ...

    @abstractmethod
    def save(self, path: Path) -> None: ...

    @abstractmethod
    def load(self, path: Path) -> None: ...


class Reranker(ABC):
    @abstractmethod
    def rerank(
        self,
        query: str,
        combined_scores: dict[Chunk, float],
        all_chunks: list[Chunk],
        top_k: int,
        *,
        penalise_paths: bool = True,
        coarse_k: int | None = None,
    ) -> list[tuple[Chunk, float]]: ...
