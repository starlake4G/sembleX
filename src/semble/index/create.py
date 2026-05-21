from __future__ import annotations

import contextlib
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import bm25s
import numpy as np

from semble.chunking import chunk_source
from semble.index.file_walker import walk_files
from semble.index.files import detect_language, get_extensions
from semble.index.sparse import enrich_for_bm25
from semble.tokens import tokenize
from semble.types import Chunk

if TYPE_CHECKING:
    from semble.interfaces import EmbeddingProvider, VectorStore

logger = logging.getLogger(__name__)

_MAX_FILE_BYTES = 1_000_000
_EMBED_BATCH_SIZE = 4096


def _embed_chunks_batched(
    model: "EmbeddingProvider",
    chunks: list[Chunk],
    vector_store: "VectorStore",
) -> None:
    total = len(chunks)
    for start in range(0, total, _EMBED_BATCH_SIZE):
        end = min(start + _EMBED_BATCH_SIZE, total)
        batch_texts = [c.content for c in chunks[start:end]]
        batch_emb = np.array(model.encode(batch_texts), dtype=np.float32)
        vector_store.add(list(range(start, end)), batch_emb)
        logger.info("Embedded %d/%d chunks", end, total)
        del batch_texts, batch_emb


def create_index_from_path(
    path: Path,
    model: "EmbeddingProvider",
    extensions: Sequence[str] | None = None,
    include_text_files: bool = False,
    display_root: Path | None = None,
    vector_store: "VectorStore | None" = None,
) -> tuple[bm25s.BM25, "VectorStore", list[Chunk]]:
    chunks: list[Chunk] = []
    extensions = get_extensions(include_text_files, extensions)
    file_count = 0
    for file_path in walk_files(path, extensions):
        language = detect_language(file_path)
        with contextlib.suppress(OSError):
            if file_path.stat().st_size > _MAX_FILE_BYTES:
                continue
            source = file_path.read_text(encoding="utf-8", errors="replace")
            chunk_path = file_path.relative_to(display_root) if display_root else file_path
            chunks.extend(chunk_source(source, str(chunk_path), language))
            file_count += 1
    logger.info("Scanned %d files, %d chunks", file_count, len(chunks))

    if chunks:
        logger.info("Embedding %d chunks (dim=%d)...", len(chunks), model.dim)
        if vector_store is None:
            from semble.backends.vector_store.numpy_store import NumpyVectorStore

            vector_store = NumpyVectorStore(dim=model.dim)
        vector_store.build(np.empty((0, model.dim), dtype=np.float32))
        _embed_chunks_batched(model, chunks, vector_store)
        logger.info("Building BM25 index...")
        bm25_index = bm25s.BM25()
        bm25_index.index(
            [tokenize(enrich_for_bm25(chunk)) for chunk in chunks],
            show_progress=False,
        )
        logger.info("Index ready: %d chunks", len(chunks))
    else:
        raise ValueError(f"No supported files found under {path}.")

    return bm25_index, vector_store, chunks
