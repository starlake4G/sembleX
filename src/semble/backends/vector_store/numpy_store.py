from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import numpy.typing as npt
from vicinity.utils import normalize

from semble.interfaces import VectorStore, EmbeddingMatrix


class NumpyVectorStore(VectorStore):
    def __init__(self, dim: int = 256) -> None:
        self._vectors: npt.NDArray[np.float32] = np.empty((0, dim), dtype=np.float32)
        self._dim = dim

    @property
    def count(self) -> int:
        return len(self._vectors)

    def build(self, vectors: EmbeddingMatrix) -> None:
        self._vectors = np.asarray(vectors, dtype=np.float32)

    def add(self, ids: list[int], vectors: EmbeddingMatrix) -> None:
        vectors = np.asarray(vectors, dtype=np.float32)
        if len(self._vectors) == 0:
            self._vectors = vectors
        else:
            self._vectors = np.vstack([self._vectors, vectors])

    def remove(self, ids: list[int]) -> None:
        if not ids or len(self._vectors) == 0:
            return
        mask = np.ones(len(self._vectors), dtype=bool)
        for i in ids:
            if 0 <= i < len(mask):
                mask[i] = False
        self._vectors = self._vectors[mask]

    def _dist(self, x: npt.NDArray) -> npt.NDArray:
        x_norm = normalize(x)
        sim = x_norm.dot(self._vectors.T)
        return 1 - sim

    def _selector_dist(self, x: npt.NDArray, selector: npt.NDArray[np.int_]) -> npt.NDArray:
        x_norm = normalize(x)
        sim = x_norm.dot(self._vectors[selector].T)
        return 1 - sim

    def query(
        self,
        vector: EmbeddingMatrix,
        k: int,
        selector: npt.NDArray[np.int_] | None = None,
    ) -> list[tuple[list[int], npt.NDArray[np.float32]]]:
        effective_k = min(k, len(self._vectors))
        if selector is not None:
            effective_k = min(effective_k, len(selector))

        if effective_k < 1:
            return []

        distances = self._selector_dist(vector, selector) if selector is not None else self._dist(vector)

        indices = np.argpartition(distances, kth=effective_k - 1, axis=1)[:, :effective_k]
        sorted_indices = np.take_along_axis(
            indices, np.argsort(np.take_along_axis(distances, indices, axis=1)), axis=1
        )
        sorted_distances = np.take_along_axis(distances, sorted_indices, axis=1)

        if selector is not None:
            sorted_indices = selector[sorted_indices]

        out: list[tuple[list[int], npt.NDArray[np.float32]]] = []
        for row_idx, row_dist in zip(sorted_indices, sorted_distances):
            out.append((row_idx.tolist(), row_dist.astype(np.float32)))
        return out

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / "vectors.npy", self._vectors)
        meta = {"dim": self._dim, "count": len(self._vectors)}
        (path / "meta.json").write_text(json.dumps(meta))

    def load(self, path: Path) -> None:
        self._vectors = np.load(path / "vectors.npy")
        if len(self._vectors) > 0:
            self._dim = self._vectors.shape[1]
