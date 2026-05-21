from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from semble.types import Chunk

EmbeddingMatrix = npt.NDArray[np.float32]


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
    @property
    @abstractmethod
    def count(self) -> int: ...

    @abstractmethod
    def build(self, vectors: EmbeddingMatrix) -> None: ...

    @abstractmethod
    def add(self, ids: list[int], vectors: EmbeddingMatrix) -> None: ...

    @abstractmethod
    def remove(self, ids: list[int]) -> None: ...

    @abstractmethod
    def query(
        self,
        vector: EmbeddingMatrix,
        k: int,
        selector: npt.NDArray[np.int_] | None = None,
    ) -> list[tuple[list[int], npt.NDArray[np.float32]]]: ...

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
    ) -> list[tuple[Chunk, float]]: ...
