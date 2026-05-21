from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import numpy.typing as npt

try:
    import faiss

    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

from semble.interfaces import LOCAL_NAMESPACE, EmbeddingMatrix, VectorStore

logger = logging.getLogger(__name__)


class FaissVectorStore(VectorStore):
    """Single-namespace FAISS-backed dense vector store keyed by chunk_id."""

    def __init__(
        self,
        dim: int = 256,
        use_gpu: bool = False,
        index_type: str = "flat",
        metric: str = "ip",
        nlist: int = 100,
    ) -> None:
        if not FAISS_AVAILABLE:
            raise RuntimeError("FAISS is not installed. Install with: pip install faiss-cpu")

        self._dim = dim
        self._index_type = index_type
        self._metric = metric
        self._nlist = nlist
        self._use_gpu = use_gpu and faiss.get_num_gpus() > 0
        self._res = None
        self._chunk_ids: list[str] = []
        self._index_by_id: dict[str, int] = {}

        self._index = self._fresh_index()
        if self._use_gpu:
            self._res = faiss.StandardGpuResources()
            self._index = faiss.index_cpu_to_gpu(self._res, 0, self._index)

    def _fresh_index(self) -> "faiss.Index":
        if self._index_type == "flat":
            return faiss.IndexFlatIP(self._dim) if self._metric == "ip" else faiss.IndexFlatL2(self._dim)
        if self._index_type == "ivf":
            quantizer = faiss.IndexFlatIP(self._dim) if self._metric == "ip" else faiss.IndexFlatL2(self._dim)
            metric = faiss.METRIC_INNER_PRODUCT if self._metric == "ip" else faiss.METRIC_L2
            return faiss.IndexIVFFlat(quantizer, self._dim, self._nlist, metric)
        raise ValueError(f"Unknown index type: {self._index_type}")

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def count(self) -> int:
        return len(self._chunk_ids)

    def _normalize_rows(self, vectors: npt.NDArray) -> npt.NDArray:
        vectors = np.ascontiguousarray(np.asarray(vectors, dtype=np.float32))
        if self._metric == "ip" and vectors.size:
            faiss.normalize_L2(vectors)
        return vectors

    def add(self, namespace: str, chunk_ids: Sequence[str], vectors: EmbeddingMatrix) -> None:
        del namespace
        if len(chunk_ids) == 0:
            return
        vectors = self._normalize_rows(vectors)
        if vectors.shape[0] != len(chunk_ids):
            raise ValueError(f"chunk_ids ({len(chunk_ids)}) and vectors ({vectors.shape[0]}) length mismatch")
        if self._index_type == "ivf" and not self._index.is_trained:
            if vectors.shape[0] < 100:
                logger.warning("Not enough vectors (%d) for IVF, falling back to flat", vectors.shape[0])
                self._index_type = "flat"
                self._index = self._fresh_index()
                if self._use_gpu:
                    self._res = faiss.StandardGpuResources()
                    self._index = faiss.index_cpu_to_gpu(self._res, 0, self._index)
            else:
                self._index.train(vectors)
        self._index.add(vectors)
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
        try:
            selector = faiss.IDSelectorArray(np.array(positions, dtype=np.int64))
            self._index.remove_ids(selector)
            mask = np.ones(len(self._chunk_ids), dtype=bool)
            mask[positions] = False
            kept = [cid for i, cid in enumerate(self._chunk_ids) if mask[i]]
            self._chunk_ids = kept
            self._index_by_id = {cid: i for i, cid in enumerate(self._chunk_ids)}
        except (RuntimeError, AttributeError):
            logger.warning("FAISS index does not support remove_ids, rebuilding")
            all_vectors = self._index.reconstruct_n(0, self._index.ntotal)
            mask = np.ones(len(self._chunk_ids), dtype=bool)
            mask[positions] = False
            kept_vectors = all_vectors[mask]
            kept_ids = [cid for i, cid in enumerate(self._chunk_ids) if mask[i]]
            self._index = self._fresh_index()
            if self._use_gpu:
                self._res = faiss.StandardGpuResources()
                self._index = faiss.index_cpu_to_gpu(self._res, 0, self._index)
            if len(kept_vectors) > 0:
                self._index.add(kept_vectors)
            self._chunk_ids = kept_ids
            self._index_by_id = {cid: i for i, cid in enumerate(self._chunk_ids)}

    def clear(self, namespace: str) -> None:
        del namespace
        self._index = self._fresh_index()
        if self._use_gpu:
            self._res = faiss.StandardGpuResources()
            self._index = faiss.index_cpu_to_gpu(self._res, 0, self._index)
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
        v = np.ascontiguousarray(np.asarray(vector, dtype=np.float32))
        if v.ndim == 1:
            v = v.reshape(1, -1)
        if self._metric == "ip":
            faiss.normalize_L2(v)

        if selector_ids is not None:
            sel_positions = np.array(
                [self._index_by_id[cid] for cid in selector_ids if cid in self._index_by_id],
                dtype=np.int64,
            )
            if sel_positions.size == 0:
                return []
            sub_vectors = self._index.reconstruct_n(0, self._index.ntotal)[sel_positions]
            sub_index = faiss.IndexFlatIP(self._dim) if self._metric == "ip" else faiss.IndexFlatL2(self._dim)
            sub_index.add(sub_vectors)
            effective_k = min(k, sel_positions.size)
            scores, local_idx = sub_index.search(v, effective_k)
            global_positions = sel_positions[local_idx[0]]
            return [
                (self._chunk_ids[int(pos)], self._similarity(float(score)))
                for pos, score in zip(global_positions, scores[0])
                if int(pos) >= 0
            ]

        effective_k = min(k, len(self._chunk_ids))
        scores, indices = self._index.search(v, effective_k)
        return [
            (self._chunk_ids[int(pos)], self._similarity(float(score)))
            for pos, score in zip(indices[0], scores[0])
            if int(pos) >= 0
        ]

    def _similarity(self, score: float) -> float:
        if self._metric == "l2":
            return 1.0 / (1.0 + score)
        return score

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        cpu_index = faiss.index_gpu_to_cpu(self._index) if self._use_gpu else self._index
        faiss.write_index(cpu_index, str(path / "faiss.index"))
        (path / "chunk_ids.json").write_text(json.dumps(self._chunk_ids))
        (path / "config.json").write_text(json.dumps({
            "dim": self._dim,
            "use_gpu": self._use_gpu,
            "index_type": self._index_type,
            "metric": self._metric,
            "nlist": self._nlist,
        }))

    def load(self, path: Path) -> None:
        self._index = faiss.read_index(str(path / "faiss.index"))
        if self._use_gpu and self._res is not None:
            self._index = faiss.index_cpu_to_gpu(self._res, 0, self._index)
        self._chunk_ids = json.loads((path / "chunk_ids.json").read_text())
        self._index_by_id = {cid: i for i, cid in enumerate(self._chunk_ids)}


__all__ = ["LOCAL_NAMESPACE", "FaissVectorStore"]
