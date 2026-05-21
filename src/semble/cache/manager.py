from __future__ import annotations

import json
import logging
from collections import OrderedDict
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from semble.backends.vector_store import create_vector_store
from semble.cache.hash_tracker import HashTracker
from semble.config import SembleConfig
from semble.index.files import get_extensions
from semble.types import Chunk

if TYPE_CHECKING:
    from semble.interfaces import VectorStore
    from semble.index.index import SembleIndex

logger = logging.getLogger(__name__)


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

    def is_disk_valid(self, source: str) -> bool:
        cache_path = self._cache_path(source)
        hash_file = cache_path / "file_hashes.json"
        if not hash_file.exists():
            return False

        root = Path(source).resolve() if Path(source).exists() else None
        if root is None:
            return True

        try:
            saved = json.loads(hash_file.read_text())
            extensions = get_extensions(False, None)
            current = HashTracker.compute_hashes(root, frozenset(extensions))
            return saved == current
        except (OSError, json.JSONDecodeError):
            return False

    def load_from_disk(self, source: str) -> tuple[list[Chunk], "VectorStore", object] | None:
        cache_path = self._cache_path(source)
        if not (cache_path / "meta.json").exists():
            return None

        try:
            meta = json.loads((cache_path / "meta.json").read_text())
            chunks_data = json.loads((cache_path / "chunks.json").read_text())
            chunks = [
                Chunk(
                    content=c["content"],
                    file_path=c["file_path"],
                    start_line=c["start_line"],
                    end_line=c["end_line"],
                    language=c.get("language"),
                )
                for c in chunks_data
            ]

            dim = meta.get("embedding_dim", 256)
            vs = create_vector_store(self._config.vector_store, dim)
            vs.load(cache_path / "vectors")

            import bm25s

            bm25_index = bm25s.BM25.load(str(cache_path / "bm25"), load_corpus=True)

            return chunks, vs, bm25_index
        except Exception as e:
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

            chunks_data = []
            for c in index.chunks:
                chunks_data.append(
                    {
                        "content": c.content,
                        "file_path": c.file_path,
                        "start_line": c.start_line,
                        "end_line": c.end_line,
                        "language": c.language,
                    }
                )
            (cache_path / "chunks.json").write_text(json.dumps(chunks_data))

            index._bm25_index.save(str(cache_path / "bm25"))

            (cache_path / "file_hashes.json").write_text(json.dumps(file_hashes))
            logger.info("Saved cache for %s (%d chunks)", source, len(index.chunks))
        except Exception as e:
            logger.error("Failed to save cache for %s: %s", source, e)
