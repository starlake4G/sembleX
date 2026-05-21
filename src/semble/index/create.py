from __future__ import annotations

import contextlib
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from semble.backends.sparse import Bm25sSparseIndex
from semble.chunking import chunk_source
from semble.config import IndexingConfig
from semble.index.file_walker import walk_files
from semble.index.files import detect_language, get_extensions
from semble.interfaces import LOCAL_NAMESPACE, EmbeddingFailure
from semble.types import Chunk, chunk_id

if TYPE_CHECKING:
    from semble.interfaces import EmbeddingProvider, SparseIndex, VectorStore

logger = logging.getLogger(__name__)


def _embed_chunks_batched(
    model: "EmbeddingProvider",
    chunks: list[Chunk],
    chunk_ids: list[str],
    vector_store: "VectorStore",
    namespace: str,
    batch_size: int,
) -> tuple[list[Chunk], list[str]]:
    """Embed *chunks* in batches; return the chunks (and their ids) that succeeded.

    Failures from the embedding provider drop the corresponding chunks rather than
    silently inserting zero vectors.
    """
    kept_chunks: list[Chunk] = []
    kept_ids: list[str] = []
    total_failed = 0
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        batch_ids = chunk_ids[start : start + batch_size]
        try:
            batch_emb = np.asarray(model.encode([c.content for c in batch]), dtype=np.float32)
            survivor_chunks, survivor_ids = batch, batch_ids
        except EmbeddingFailure as exc:
            failed = set(exc.failed_indices)
            total_failed += len(failed)
            survivor_chunks = [c for i, c in enumerate(batch) if i not in failed]
            survivor_ids = [cid for i, cid in enumerate(batch_ids) if i not in failed]
            if exc.partial is None or not survivor_chunks:
                logger.warning("Batch starting at %d fully failed; dropping %d chunks", start, len(batch))
                continue
            batch_emb = np.asarray(exc.partial, dtype=np.float32)
            logger.warning(
                "Batch starting at %d partially failed; dropped %d/%d chunks",
                start, len(failed), len(batch),
            )

        vector_store.add(namespace, survivor_ids, batch_emb)
        kept_chunks.extend(survivor_chunks)
        kept_ids.extend(survivor_ids)
        logger.info("Embedded %d/%d chunks", len(kept_chunks), len(chunks))

    if total_failed:
        logger.warning("Embedding completed with %d failed chunks dropped", total_failed)
    return kept_chunks, kept_ids


def create_index_from_path(
    path: Path,
    model: "EmbeddingProvider",
    extensions: Sequence[str] | None = None,
    include_text_files: bool = False,
    display_root: Path | None = None,
    vector_store: "VectorStore | None" = None,
    indexing: IndexingConfig | None = None,
    namespace: str = LOCAL_NAMESPACE,
) -> tuple["SparseIndex", "VectorStore", list[Chunk], list[str]]:
    """Chunk *path*, embed, and build sparse + dense indices.

    :returns: ``(sparse_index, vector_store, chunks, chunk_ids)`` where each
        ``chunks[i]`` corresponds to ``chunk_ids[i]``.
    """
    cfg = indexing or IndexingConfig()
    chunks: list[Chunk] = []
    extensions = get_extensions(include_text_files, extensions)
    file_count = 0
    for file_path in walk_files(path, extensions):
        language = detect_language(file_path)
        with contextlib.suppress(OSError):
            if file_path.stat().st_size > cfg.max_file_bytes:
                continue
            source = file_path.read_text(encoding="utf-8", errors="replace")
            chunk_path = file_path.relative_to(display_root) if display_root else file_path
            chunks.extend(chunk_source(source, str(chunk_path), language))
            file_count += 1
    logger.info("Scanned %d files, %d chunks", file_count, len(chunks))

    if not chunks:
        raise ValueError(f"No supported files found under {path}.")

    chunk_ids = [chunk_id(c, namespace) for c in chunks]

    logger.info("Embedding %d chunks (dim=%d)...", len(chunks), model.dim)
    if vector_store is None:
        from semble.backends.vector_store.numpy_store import NumpyVectorStore

        vector_store = NumpyVectorStore(dim=model.dim)
    chunks, chunk_ids = _embed_chunks_batched(
        model, chunks, chunk_ids, vector_store, namespace, batch_size=cfg.embed_batch_size,
    )
    if not chunks:
        raise ValueError(f"All chunks failed to embed for {path}.")

    logger.info("Building sparse (BM25) index...")
    sparse_index = Bm25sSparseIndex()
    sparse_index.build(chunks, chunk_ids)

    logger.info("Index ready: %d chunks", len(chunks))
    return sparse_index, vector_store, chunks, chunk_ids
