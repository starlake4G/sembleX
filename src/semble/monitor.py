from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from semble.cache.hash_tracker import HashTracker
from semble.chunking import chunk_source
from semble.config import SembleConfig
from semble.index.files import detect_language, get_extensions
from semble.interfaces import LOCAL_NAMESPACE, EmbeddingFailure
from semble.types import chunk_id

if TYPE_CHECKING:
    from semble.cache.manager import CacheManager
    from semble.index.index import SembleIndex

logger = logging.getLogger(__name__)


class FileMonitor:
    """Watches *root* and incrementally updates *index* on file changes."""

    def __init__(
        self,
        root: Path,
        index: "SembleIndex",
        cache_manager: "CacheManager | None",
        config: SembleConfig,
    ) -> None:
        self._root = root
        self._index = index
        self._cache_manager = cache_manager
        self._config = config
        self._hash_tracker = HashTracker()
        self._current_hashes: dict[str, str] = {}

    def initialize_hashes(self) -> None:
        extensions = get_extensions(False, None)
        self._current_hashes = self._hash_tracker.compute_hashes(self._root, frozenset(extensions))

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

        affected_ids: list[str] = []
        for fpath in affected_paths:
            affected_ids.extend(self._index._file_mapping.get(fpath, []))

        if not affected_ids and not changed:
            self._current_hashes = new_hashes
            return

        old_count = len(self._index.chunks)
        logger.info(
            "File changes detected: %d added, %d removed, %d modified (%d chunks affected)",
            len(added), len(removed), len(modified), len(affected_ids),
        )

        if affected_ids:
            self._index._semantic_index.delete(LOCAL_NAMESPACE, affected_ids)
            affected_set = set(affected_ids)
            kept_pairs = [
                (c, cid)
                for c, cid in zip(self._index.chunks, self._index.chunk_ids)
                if cid not in affected_set
            ]
            self._index.chunks = [c for c, _ in kept_pairs]
            self._index.chunk_ids = [cid for _, cid in kept_pairs]
            self._index._by_id = dict(zip(self._index.chunk_ids, self._index.chunks))

        new_chunks = []
        for fpath in sorted(changed):
            full_path = self._root / fpath
            try:
                if full_path.stat().st_size > self._config.indexing.max_file_bytes:
                    continue
                source = full_path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            language = detect_language(full_path)
            new_chunks.extend(chunk_source(source, fpath, language))

        new_ids = [chunk_id(c, LOCAL_NAMESPACE) for c in new_chunks]
        if new_chunks:
            try:
                new_embeddings = np.asarray(
                    self._index.model.encode([c.content for c in new_chunks]),
                    dtype=np.float32,
                )
                survivors = list(zip(new_chunks, new_ids))
            except EmbeddingFailure as exc:
                failed = set(exc.failed_indices)
                survivors = [
                    (c, cid) for i, (c, cid) in enumerate(zip(new_chunks, new_ids)) if i not in failed
                ]
                if exc.partial is None or not survivors:
                    logger.warning("All %d new chunks failed to embed; skipping", len(new_chunks))
                    new_chunks, new_ids = [], []
                else:
                    new_embeddings = np.asarray(exc.partial, dtype=np.float32)
                    new_chunks = [c for c, _ in survivors]
                    new_ids = [cid for _, cid in survivors]
                    logger.warning(
                        "Dropped %d/%d new chunks (embedding failed)",
                        len(failed), len(survivors) + len(failed),
                    )

            if new_chunks:
                self._index._semantic_index.add(LOCAL_NAMESPACE, new_ids, new_embeddings)
                self._index.chunks = self._index.chunks + new_chunks
                self._index.chunk_ids = self._index.chunk_ids + new_ids
                self._index._by_id = dict(zip(self._index.chunk_ids, self._index.chunks))

        # Rebuild BM25 — incremental updates are not supported by bm25s.
        self._index._sparse_index.build(self._index.chunks, self._index.chunk_ids)

        # Recompute file/language mappings now that chunks have shifted.
        self._index._file_mapping, self._index._language_mapping = self._index._populate_mapping()

        self._current_hashes = new_hashes
        if self._cache_manager is not None:
            self._cache_manager.save_to_disk(str(self._root), self._index, new_hashes)
        logger.info("Index updated: %d chunks (was %d)", len(self._index.chunks), old_count)
