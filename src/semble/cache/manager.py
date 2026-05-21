from __future__ import annotations

import json
import logging
from collections import OrderedDict
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING

from semble.backends.sparse import Bm25sSparseIndex
from semble.backends.vector_store import create_vector_store
from semble.cache.hash_tracker import HashTracker
from semble.config import SembleConfig
from semble.index.files import get_extensions
from semble.types import Chunk

if TYPE_CHECKING:
    from semble.index.index import SembleIndex
    from semble.interfaces import SparseIndex, VectorStore

logger = logging.getLogger(__name__)

_CHUNKS_JSONL = "chunks.jsonl"


def _chunk_to_dict(chunk: Chunk, chunk_id: str) -> dict[str, object]:
    return {
        "chunk_id": chunk_id,
        "content": chunk.content,
        "file_path": chunk.file_path,
        "start_line": chunk.start_line,
        "end_line": chunk.end_line,
        "language": chunk.language,
    }


def _chunk_from_dict(data: dict[str, object]) -> tuple[Chunk, str]:
    chunk = Chunk(
        content=str(data["content"]),
        file_path=str(data["file_path"]),
        start_line=int(data["start_line"]),
        end_line=int(data["end_line"]),
        language=data.get("language") if isinstance(data.get("language"), str) else None,
    )
    return chunk, str(data["chunk_id"])


class CacheManager:
    def __init__(self, cache_dir: Path, config: SembleConfig) -> None:
        self._cache_dir = cache_dir
        self._config = config
        self._in_memory: OrderedDict[str, "SembleIndex"] = OrderedDict()
        self._max_cache = 10

    def _repo_key(self, source: str) -> str:
        return sha256(source.encode()).hexdigest()[:16]

    def _cache_path(self, source: str) -> Path:
        return self._cache_dir / self._repo_key(source)

    def get_in_memory(self, source: str) -> "SembleIndex | None":
        key = self._repo_key(source)
        if key in self._in_memory:
            self._in_memory.move_to_end(key)
            return self._in_memory[key]
        return None

    def put_in_memory(self, source: str, index: "SembleIndex") -> None:
        key = self._repo_key(source)
        self._in_memory[key] = index
        if len(self._in_memory) > self._max_cache:
            self._in_memory.popitem(last=False)

    def is_disk_valid(
        self,
        source: str,
        current_hashes: dict[str, str] | None = None,
        extensions: frozenset[str] | set[str] | None = None,
    ) -> bool:
        cache_path = self._cache_path(source)
        hash_file = cache_path / "file_hashes.json"
        if not hash_file.exists():
            return False

        root = Path(source).resolve() if Path(source).exists() else None
        if root is None:
            # git URL caches expire only when explicitly invalidated by deletion.
            return True

        try:
            saved = json.loads(hash_file.read_text())
            current = current_hashes
            if current is None:
                current_extensions = extensions or frozenset(get_extensions(False, None))
                current = HashTracker.compute_hashes(root, frozenset(current_extensions))
            return saved == current
        except (OSError, json.JSONDecodeError):
            return False

    def _load_chunks(self, cache_path: Path) -> tuple[list[Chunk], list[str]]:
        chunks: list[Chunk] = []
        chunk_ids: list[str] = []
        with (cache_path / _CHUNKS_JSONL).open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                chunk, cid = _chunk_from_dict(json.loads(line))
                chunks.append(chunk)
                chunk_ids.append(cid)
        return chunks, chunk_ids

    def load_from_disk(
        self, source: str
    ) -> tuple[list[Chunk], list[str], "VectorStore", "SparseIndex"] | None:
        cache_path = self._cache_path(source)
        if not (cache_path / "meta.json").exists():
            return None

        try:
            meta = json.loads((cache_path / "meta.json").read_text())
            chunks, chunk_ids = self._load_chunks(cache_path)

            dim = meta.get("embedding_dim", 256)
            vs = create_vector_store(self._config.vector_store, dim)
            vs.load(cache_path / "vectors")

            sparse = Bm25sSparseIndex()
            sparse.load(cache_path / "sparse")

            return chunks, chunk_ids, vs, sparse
        except Exception as e:  # noqa: BLE001 — cache is best-effort
            logger.warning("Failed to load cache for %s: %s", source, e)
            return None

    def save_to_disk(
        self,
        source: str,
        index: "SembleIndex",
        file_hashes: dict[str, str],
    ) -> None:
        cache_path = self._cache_path(source)
        try:
            cache_path.mkdir(parents=True, exist_ok=True)

            meta = {
                "source": source,
                "indexed_at": datetime.now(timezone.utc).isoformat(),
                "chunk_count": len(index.chunks),
                "embedding_backend": self._config.embedding.backend,
                "embedding_dim": index.model.dim,
            }
            (cache_path / "meta.json").write_text(json.dumps(meta, indent=2))

            index._semantic_index.save(cache_path / "vectors")
            index._sparse_index.save(cache_path / "sparse")

            with (cache_path / _CHUNKS_JSONL).open("w", encoding="utf-8") as f:
                for chunk, cid in zip(index.chunks, index.chunk_ids):
                    f.write(json.dumps(_chunk_to_dict(chunk, cid), separators=(",", ":")))
                    f.write("\n")

            (cache_path / "file_hashes.json").write_text(json.dumps(file_hashes))
            logger.info("Saved cache for %s (%d chunks)", source, len(index.chunks))
        except Exception as e:  # noqa: BLE001 — cache is best-effort
            logger.error("Failed to save cache for %s: %s", source, e)
