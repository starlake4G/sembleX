from __future__ import annotations

import contextlib
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import numpy.typing as npt

from semble.config import MilvusConfig
from semble.interfaces import EmbeddingMatrix, VectorStore


class MilvusVectorStore(VectorStore):
    """Milvus-backed multi-namespace vector store.

    Each ``namespace`` (typically a repo_id) maps to rows in a shared
    collection. Only ``chunk_id``, ``namespace`` and the vector live in
    Milvus — content/metadata live in the metadata store maintained by the
    server, which keeps the collection schema lean.
    """

    def __init__(self, config: MilvusConfig, dim: int) -> None:
        try:
            from pymilvus import DataType, MilvusClient
        except ImportError:
            raise RuntimeError("Milvus dependencies are missing. Install with: pip install 'semble[server]'") from None

        self._config = config
        self._dim = dim
        client_kwargs: dict[str, object] = {"uri": config.uri}
        if config.token:
            client_kwargs["token"] = config.token
        if config.db_name:
            client_kwargs["db_name"] = config.db_name
        self._client = MilvusClient(**client_kwargs)
        self._data_type = DataType
        self._ensure_collection()

    @property
    def dim(self) -> int:
        return self._dim

    def _ensure_collection(self) -> None:
        if self._client.has_collection(self._config.collection):
            with contextlib.suppress(Exception):
                self._client.load_collection(collection_name=self._config.collection)
            return

        schema = self._client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field(field_name="chunk_id", datatype=self._data_type.VARCHAR, is_primary=True, max_length=64)
        schema.add_field(field_name="namespace", datatype=self._data_type.VARCHAR, max_length=64)
        schema.add_field(field_name=self._config.vector_field, datatype=self._data_type.FLOAT_VECTOR, dim=self._dim)

        index_params = self._client.prepare_index_params()
        index_params.add_index(
            field_name=self._config.vector_field,
            index_type=self._config.index_type,
            metric_type=self._config.metric_type,
            params=self._config.index_params,
        )
        self._client.create_collection(
            collection_name=self._config.collection,
            schema=schema,
            index_params=index_params,
            consistency_level=self._config.consistency_level,
        )
        with contextlib.suppress(Exception):
            self._client.load_collection(collection_name=self._config.collection)

    def add(self, namespace: str, chunk_ids: Sequence[str], vectors: EmbeddingMatrix) -> None:
        if len(chunk_ids) == 0:
            return
        prepared = self._prepare_vectors(np.asarray(vectors, dtype=np.float32))
        if prepared.shape[0] != len(chunk_ids):
            raise ValueError(f"chunk_ids ({len(chunk_ids)}) and vectors ({prepared.shape[0]}) length mismatch")
        rows = [
            {
                "chunk_id": cid,
                "namespace": namespace,
                self._config.vector_field: vec.tolist(),
            }
            for cid, vec in zip(chunk_ids, prepared)
        ]
        self._client.insert(collection_name=self._config.collection, data=rows)

    def delete(self, namespace: str, chunk_ids: Sequence[str]) -> None:
        if not chunk_ids:
            return
        # Milvus filter expression: chunk_id is the primary key.
        quoted = ",".join(f'"{cid}"' for cid in chunk_ids)
        self._client.delete(
            collection_name=self._config.collection,
            filter=f'namespace == "{namespace}" && chunk_id in [{quoted}]',
        )

    def clear(self, namespace: str) -> None:
        self._client.delete(
            collection_name=self._config.collection,
            filter=f'namespace == "{namespace}"',
        )

    def query(
        self,
        namespace: str,
        vector: EmbeddingMatrix,
        k: int,
        selector_ids: Sequence[str] | None = None,
    ) -> list[tuple[str, float]]:
        prepared = self._prepare_vectors(np.asarray(vector, dtype=np.float32))
        filter_expr = f'namespace == "{namespace}"'
        if selector_ids is not None:
            ids = list(selector_ids)
            if not ids:
                return []
            quoted = ",".join(f'"{cid}"' for cid in ids)
            filter_expr = f"{filter_expr} && chunk_id in [{quoted}]"
        search_kwargs: dict[str, object] = {
            "collection_name": self._config.collection,
            "data": prepared.tolist(),
            "anns_field": self._config.vector_field,
            "filter": filter_expr,
            "limit": k,
            "output_fields": ["chunk_id"],
        }
        if self._config.search_params:
            search_kwargs["search_params"] = self._config.search_params
        results = self._client.search(**search_kwargs)
        hits: list[tuple[str, float]] = []
        first = results[0] if results else []
        for hit in first:
            cid, _, distance = self._parse_hit(hit)
            if cid is None:
                continue
            hits.append((cid, self._similarity(distance)))
        return hits

    def query_global(
        self,
        vector: EmbeddingMatrix,
        k: int,
        exclude_namespace: str | None = None,
    ) -> list[tuple[str, str, float]]:
        """Search across all namespaces, returning ``[(chunk_id, namespace, similarity)]``."""
        prepared = self._prepare_vectors(np.asarray(vector, dtype=np.float32))
        search_kwargs: dict[str, object] = {
            "collection_name": self._config.collection,
            "data": prepared.tolist(),
            "anns_field": self._config.vector_field,
            "limit": k,
            "output_fields": ["chunk_id", "namespace"],
        }
        if exclude_namespace is not None:
            search_kwargs["filter"] = f'namespace != "{exclude_namespace}"'
        if self._config.search_params:
            search_kwargs["search_params"] = self._config.search_params
        results = self._client.search(**search_kwargs)
        hits: list[tuple[str, str, float]] = []
        first = results[0] if results else []
        for hit in first:
            cid, namespace, distance = self._parse_hit(hit)
            if cid is None:
                continue
            hits.append((cid, namespace or "", self._similarity(distance)))
        return hits

    @staticmethod
    def _parse_hit(hit: object) -> tuple[str | None, str | None, float]:
        """Extract ``(chunk_id, namespace, distance)`` from a Milvus hit (dict or object)."""
        if isinstance(hit, dict):
            entity = hit.get("entity", {})
            cid = entity.get("chunk_id") or hit.get("id")
            namespace = entity.get("namespace")
            distance = float(hit.get("distance", 0.0))
        else:
            cid = getattr(hit, "id", None)
            namespace = None
            distance = float(getattr(hit, "distance", 0.0))
        return (str(cid) if cid is not None else None, str(namespace) if namespace is not None else None, distance)

    def save(self, path: Path) -> None:
        # Milvus is the source of truth; nothing to dump.
        del path

    def load(self, path: Path) -> None:
        del path

    def _prepare_vectors(self, vectors: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        arr = np.ascontiguousarray(np.asarray(vectors, dtype=np.float32))
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if self._config.metric_type == "IP":
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            arr = (arr / np.where(norms > 1e-9, norms, 1.0)).astype(np.float32, copy=False)
        return arr

    def _similarity(self, distance: float) -> float:
        """Convert Milvus' raw `distance` into a uniform similarity (larger=better)."""
        if self._config.metric_type == "L2":
            return 1.0 / (1.0 + distance)
        return distance
