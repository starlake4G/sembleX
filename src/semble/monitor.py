from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from semble.backends.vector_store import create_vector_store
from semble.cache.hash_tracker import HashTracker
from semble.chunking import chunk_source
from semble.config import SembleConfig
from semble.index.files import get_extensions
from semble.index.sparse import enrich_for_bm25
from semble.tokens import tokenize

if TYPE_CHECKING:
    from semble.cache.manager import CacheManager
    from semble.index.index import SembleIndex

logger = logging.getLogger(__name__)


class FileMonitor:
    def __init__(
        self,
        root: Path,
        index: "SembleIndex",
        cache_manager: "CacheManager",
        config: SembleConfig,
    ) -> None:
        self._root = root
        self._index = index
        self._cache_manager = cache_manager
        self._config = config
        self._hash_tracker = HashTracker()
        self._current_hashes: dict[str, str] = {}
        self._file_to_chunks: dict[str, list[int]] = {}

    def initialize_hashes(self) -> None:
        extensions = get_extensions(False, None)
        self._current_hashes = self._hash_tracker.compute_hashes(self._root, frozenset(extensions))
        self._rebuild_file_mapping()

    def _rebuild_file_mapping(self) -> None:
        self._file_to_chunks.clear()
        for i, chunk in enumerate(self._index.chunks):
            self._file_to_chunks.setdefault(chunk.file_path, []).append(i)

    async def watch(self) -> None:
        try:
            import watchfiles
        except ImportError:
            logger.warning("watchfiles not installed, file monitoring disabled")
            return
        async for _ in watchfiles.awatch(self._root):
            await asyncio.sleep(self._config.monitor.debounce_ms / 1000)
            await self._on_change()

    async def _on_change(self) -> None:
        extensions = get_extensions(False, None)
        new_hashes = self._hash_tracker.compute_hashes(self._root, frozenset(extensions))
        added, removed, modified = self._hash_tracker.diff(self._current_hashes, new_hashes)
        changed = added | modified
        affected_paths = changed | removed

        if not affected_paths:
            return

        affected_indices = sorted(
            [i for p in affected_paths for i in self._file_to_chunks.get(p, [])],
            reverse=True,
        )
        if not affected_indices:
            self._current_hashes = new_hashes
            return

        logger.info(
            "File changes detected: %d added, %d removed, %d modified (%d chunks affected)",
            len(added), len(removed), len(modified), len(affected_indices),
        )

        old_count = len(self._index.chunks)

        keep_mask = np.ones(old_count, dtype=bool)
        for i in affected_indices:
            if 0 <= i < old_count:
                keep_mask[i] = False

        self._index._semantic_index.remove(affected_indices)

        new_chunks = []
        from semble.index.files import detect_language

        for fpath in sorted(changed):
            full_path = self._root / fpath
            try:
                if full_path.stat().st_size > 1_000_000:
                    continue
                source = full_path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            language = detect_language(full_path)
            new_chunks.extend(chunk_source(source, fpath, language))

        if new_chunks:
            new_embeddings = np.array(
                self._index.model.encode([c.content for c in new_chunks]),
                dtype=np.float32,
            )
            start_id = int(keep_mask.sum())
            new_ids = list(range(start_id, start_id + len(new_chunks)))
            self._index._semantic_index.add(new_ids, new_embeddings)

        kept_chunks = [c for c, keep in zip(self._index.chunks, keep_mask) if keep]
        self._index.chunks = kept_chunks + new_chunks

        import bm25s

        bm25_index = bm25s.BM25()
        bm25_index.index(
            [tokenize(enrich_for_bm25(c)) for c in self._index.chunks],
            show_progress=False,
        )
        self._index._bm25_index = bm25_index

        self._rebuild_file_mapping()
        self._current_hashes = new_hashes
        self._cache_manager.save_to_disk(str(self._root), self._index, new_hashes)
        logger.info("Index updated: %d chunks (was %d)", len(self._index.chunks), old_count)
