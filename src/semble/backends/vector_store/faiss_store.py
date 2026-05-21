from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import numpy.typing as npt

try:
    import faiss

    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

from semble.interfaces import VectorStore, EmbeddingMatrix

logger = logging.getLogger(__name__)


class FaissVectorStore(VectorStore):
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
        self._use_gpu = use_gpu and FAISS_AVAILABLE and faiss.get_num_gpus() > 0
        self._res = None

        if index_type == "flat":
            if metric == "ip":
                self._index = faiss.IndexFlatIP(dim)
            else:
                self._index = faiss.IndexFlatL2(dim)
        elif index_type == "ivf":
            quantizer = faiss.IndexFlatIP(dim) if metric == "ip" else faiss.IndexFlatL2(dim)
            self._index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT if metric == "ip" else faiss.METRIC_L2)
        else:
            raise ValueError(f"Unknown index type: {index_type}")

        if self._use_gpu:
            self._res = faiss.StandardGpuResources()
            self._index = faiss.index_cpu_to_gpu(self._res, 0, self._index)

    @property
    def count(self) -> int:
        return self._index.ntotal

    def _normalize(self, vectors: npt.NDArray) -> None:
        if self._metric == "ip":
            faiss.normalize_L2(vectors)

    def build(self, vectors: EmbeddingMatrix) -> None:
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.size > 0:
            self._normalize(vectors)
        if self._index_type == "ivf" and not self._index.is_trained:
            if vectors.shape[0] < 100:
                logger.warning("Not enough vectors (%d) for IVF, falling back to flat", vectors.shape[0])
                if self._metric == "ip":
                    self._index = faiss.IndexFlatIP(self._dim)
                else:
                    self._index = faiss.IndexFlatL2(self._dim)
                if self._use_gpu:
                    self._res = faiss.StandardGpuResources()
                    self._index = faiss.index_cpu_to_gpu(self._res, 0, self._index)
            else:
                self._index.train(vectors)
        if vectors.size > 0:
            self._index.add(vectors)

    def add(self, ids: list[int], vectors: EmbeddingMatrix) -> None:
        vectors = np.asarray(vectors, dtype=np.float32)
        self._normalize(vectors)
        self._index.add(vectors)

    def remove(self, ids: list[int]) -> None:
        if not ids or self._index.ntotal == 0:
            return
        try:
            id_selector = faiss.IDSelectorArray(np.array(ids, dtype=np.int64))
            self._index.remove_ids(id_selector)
        except (RuntimeError, AttributeError):
            logger.warning("FAISS index does not support remove_ids, rebuilding")
            all_vectors = self._index.reconstruct_n(0, self._index.ntotal)
            keep_mask = np.ones(len(all_vectors), dtype=bool)
            for i in ids:
                if 0 <= i < len(keep_mask):
                    keep_mask[i] = False
            kept = all_vectors[keep_mask]
            self._index.reset()
            if len(kept) > 0:
                self._index.add(kept)

    def query(
        self,
        vector: EmbeddingMatrix,
        k: int,
        selector: npt.NDArray[np.int_] | None = None,
    ) -> list[tuple[list[int], npt.NDArray[np.float32]]]:
        vector = np.asarray(vector, dtype=np.float32).copy()
        self._normalize(vector)

        if selector is not None:
            ntotal = self._index.ntotal
            all_vectors = self._index.reconstruct_n(0, ntotal)
            sub_vectors = all_vectors[selector]
            sub_index = faiss.IndexFlatIP(self._dim) if self._metric == "ip" else faiss.IndexFlatL2(self._dim)
            sub_index.add(sub_vectors)
            effective_k = min(k, len(selector))
            scores, local_indices = sub_index.search(vector, effective_k)
            global_indices = selector[local_indices]
        else:
            effective_k = min(k, self._index.ntotal)
            scores, global_indices = self._index.search(vector, effective_k)

        if self._metric == "ip":
            distances = 1.0 - scores.astype(np.float32)
        else:
            distances = scores.astype(np.float32)

        out: list[tuple[list[int], npt.NDArray[np.float32]]] = []
        for row_idx, row_dist in zip(global_indices, distances):
            out.append((row_idx.tolist(), row_dist))
        return out

    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        cpu_index = faiss.index_gpu_to_cpu(self._index) if self._use_gpu else self._index
        faiss.write_index(cpu_index, str(path / "faiss.index"))
        config = {
            "dim": self._dim,
            "use_gpu": self._use_gpu,
            "index_type": self._index_type,
            "metric": self._metric,
            "nlist": self._nlist,
        }
        (path / "config.json").write_text(json.dumps(config))

    def load(self, path: Path) -> None:
        path = Path(path)
        self._index = faiss.read_index(str(path / "faiss.index"))
        if self._use_gpu and self._res is not None:
            self._index = faiss.index_cpu_to_gpu(self._res, 0, self._index)
