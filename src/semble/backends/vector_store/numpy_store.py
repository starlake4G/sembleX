from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import numpy.typing as npt
from vicinity.utils import normalize

from semble.interfaces import LOCAL_NAMESPACE, EmbeddingMatrix, VectorStore


class NumpyVectorStore(VectorStore):
    """Dense vector store backed by a single ``(N, dim)`` numpy array.

    Single-namespace: any namespace value is accepted but treated as the local
    one. The store maintains a ``chunk_id -> row`` mapping so callers always
    reason about stable ids, never row positions.
    """

    def __init__(self, dim: int = 256) -> None:
        self._dim = dim
        self._vectors: npt.NDArray[np.float32] = np.empty((0, dim), dtype=np.float32)
        self._chunk_ids: list[str] = []
        self._index_by_id: dict[str, int] = {}

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def count(self) -> int:
        return len(self._chunk_ids)

    def _resize(self, new_dim: int) -> None:
        if self._vectors.size == 0:
            self._dim = new_dim
            self._vectors = np.empty((0, new_dim), dtype=np.float32)

    def add(self, namespace: str, chunk_ids: Sequence[str], vectors: EmbeddingMatrix) -> None:
        del namespace
        if len(chunk_ids) == 0:
            return
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.shape[0] != len(chunk_ids):
            raise ValueError(f"chunk_ids ({len(chunk_ids)}) and vectors ({vectors.shape[0]}) length mismatch")
        if self._vectors.shape[0] == 0 and vectors.shape[1] != self._dim:
            self._dim = vectors.shape[1]
        if self._vectors.size == 0:
            self._vectors = vectors.copy()
        else:
            self._vectors = np.vstack([self._vectors, vectors])
        for cid in chunk_ids:
            if cid in self._index_by_id:
                raise ValueError(f"chunk_id {cid!r} already present")
            self._index_by_id[cid] = len(self._chunk_ids)
            self._chunk_ids.append(cid)

    def delete(self, namespace: str, chunk_ids: Sequence[str]) -> None:
        del namespace
        positions = sorted({self._index_by_id[cid] for cid in chunk_ids if cid in self._index_by_id})
        if not positions:
            return
        mask = np.ones(len(self._chunk_ids), dtype=bool)
        mask[positions] = False
        self._vectors = self._vectors[mask]
        kept_ids = [cid for i, cid in enumerate(self._chunk_ids) if mask[i]]
        self._chunk_ids = kept_ids
        self._index_by_id = {cid: i for i, cid in enumerate(self._chunk_ids)}

    def clear(self, namespace: str) -> None:
        del namespace
        self._vectors = np.empty((0, self._dim), dtype=np.float32)
        self._chunk_ids = []
        self._index_by_id = {}

    def query(
        self,
        namespace: str,
        vector: EmbeddingMatrix,
        k: int,
        selector_ids: Sequence[str] | None = None,
    ) -> list[tuple[str, float]]:
        del namespace
        if len(self._chunk_ids) == 0:
            return []
        v = np.asarray(vector, dtype=np.float32)
        if v.ndim == 1:
            v = v.reshape(1, -1)
        v_norm = normalize(v)

        if selector_ids is not None:
            sel_positions = np.array(
                [self._index_by_id[cid] for cid in selector_ids if cid in self._index_by_id],
                dtype=np.int_,
            )
            if sel_positions.size == 0:
                return []
            sub_vectors = self._vectors[sel_positions]
            sim = (v_norm @ sub_vectors.T)[0]
            effective_k = min(k, sel_positions.size)
            top = np.argpartition(-sim, kth=effective_k - 1)[:effective_k]
            top = top[np.argsort(-sim[top])]
            return [(self._chunk_ids[sel_positions[i]], float(sim[i])) for i in top]

        sim = (v_norm @ self._vectors.T)[0]
        effective_k = min(k, len(self._chunk_ids))
        top = np.argpartition(-sim, kth=effective_k - 1)[:effective_k]
        top = top[np.argsort(-sim[top])]
        return [(self._chunk_ids[i], float(sim[i])) for i in top]

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / "vectors.npy", self._vectors)
        (path / "chunk_ids.json").write_text(json.dumps(self._chunk_ids))
        (path / "meta.json").write_text(json.dumps({"dim": self._dim, "count": len(self._chunk_ids)}))

    def load(self, path: Path) -> None:
        self._vectors = np.load(path / "vectors.npy").astype(np.float32, copy=False)
        if self._vectors.size > 0:
            self._dim = self._vectors.shape[1]
        self._chunk_ids = json.loads((path / "chunk_ids.json").read_text())
        self._index_by_id = {cid: i for i, cid in enumerate(self._chunk_ids)}


# Convenience re-export so internal code can pin the namespace constant.
__all__ = ["LOCAL_NAMESPACE", "NumpyVectorStore"]
