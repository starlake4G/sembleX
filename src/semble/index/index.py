from __future__ import annotations

import logging
import os
import subprocess
import tempfile
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from bm25s import BM25

from semble.backends.embedding import create_embedding_provider
from semble.backends.reranker import create_reranker
from semble.backends.vector_store import create_vector_store
from semble.cache.hash_tracker import HashTracker
from semble.cache.manager import CacheManager
from semble.config import SembleConfig
from semble.index.create import create_index_from_path
from semble.index.files import get_extensions
from semble.search import _search_semantic, search
from semble.stats import save_search_stats
from semble.types import CallType, Chunk, IndexStats, SearchResult

if TYPE_CHECKING:
    from semble.interfaces import EmbeddingProvider, Reranker, VectorStore

logger = logging.getLogger(__name__)

_GIT_CLONE_TIMEOUT = int(os.environ.get("SEMBLE_CLONE_TIMEOUT", 60))


class SembleIndex:
    """Fast local code index with hybrid search."""

    def __init__(
        self,
        model: "EmbeddingProvider",
        bm25_index: BM25,
        semantic_index: "VectorStore",
        chunks: list[Chunk],
        root: Path | None = None,
        reranker: "Reranker | None" = None,
        config: SembleConfig | None = None,
    ) -> None:
        self.model: "EmbeddingProvider" = model
        self.chunks: list[Chunk] = chunks
        self._bm25_index: BM25 = bm25_index
        self._semantic_index: "VectorStore" = semantic_index
        self._root: Path | None = root
        self._reranker: "Reranker | None" = reranker
        self._config: SembleConfig = config or SembleConfig()
        self._file_sizes: dict[str, int] = self._compute_file_sizes(root) if root else {}
        self._file_mapping, self._language_mapping = self._populate_mapping()

    def _populate_mapping(self) -> tuple[dict[str, list[int]], dict[str, list[int]]]:
        language_to_id: dict[str, list[int]] = defaultdict(list)
        file_to_id: dict[str, list[int]] = defaultdict(list)
        for i, chunk in enumerate(self.chunks):
            if chunk.language:
                language_to_id[chunk.language].append(i)
            file_to_id[chunk.file_path].append(i)
        return dict(file_to_id), dict(language_to_id)

    def _compute_file_sizes(self, root: Path) -> dict[str, int]:
        sizes: dict[str, int] = {}
        for chunk in self.chunks:
            if chunk.file_path in sizes:
                continue
            try:
                sizes[chunk.file_path] = len((root / chunk.file_path).read_text(encoding="utf-8", errors="replace"))
            except OSError:
                pass
        return sizes

    @property
    def stats(self) -> IndexStats:
        language_counts: dict[str, int] = defaultdict(int)
        for chunk in self.chunks:
            if chunk.language:
                language_counts[chunk.language] += 1
        return IndexStats(
            indexed_files=len(self._file_mapping),
            total_chunks=len(self.chunks),
            languages=dict(language_counts),
        )

    @classmethod
    def _resolve_model(cls, model: object | None, config: SembleConfig) -> "EmbeddingProvider":
        if model is not None:
            return model  # type: ignore[return-value]
        return create_embedding_provider(config.embedding)

    @classmethod
    def from_path(
        cls,
        path: str | Path,
        model: object | None = None,
        extensions: Sequence[str] | None = None,
        include_text_files: bool = False,
        config: SembleConfig | None = None,
    ) -> SembleIndex:
        cfg = config or SembleConfig()
        resolved_model = cls._resolve_model(model, cfg)
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Path does not exist: {path}")
        if not path.is_dir():
            raise NotADirectoryError(f"Path is not a directory: {path}")
        path = path.resolve()

        reranker = create_reranker(cfg.reranker) if cfg.reranker.backend != "rules" else None

        logger.info("Indexing %s...", path)
        if cfg.cache.enabled:
            cache_mgr = CacheManager(cfg.cache.dir, cfg)
            ext_set = frozenset(get_extensions(include_text_files, extensions))
            logger.info("Computing file hashes...")
            file_hashes = HashTracker.compute_hashes(path, ext_set)
            if cache_mgr.is_disk_valid(str(path)):
                logger.info("Loading cached index for %s...", path)
                cached = cache_mgr.load_from_disk(str(path))
                if cached is not None:
                    chunks, vs, bm25_idx = cached
                    logger.info("Loaded cached index: %d chunks", len(chunks))
                    return cls(resolved_model, bm25_idx, vs, chunks, root=path, reranker=reranker, config=cfg)

        logger.info("Building index for %s...", path)
        vs = create_vector_store(cfg.vector_store, resolved_model.dim)
        bm25_idx, vs, chunks = create_index_from_path(
            path,
            model=resolved_model,
            extensions=extensions,
            include_text_files=include_text_files,
            display_root=path,
            vector_store=vs,
        )

        if cfg.cache.enabled:
            logger.info("Saving index to cache...")
            cache_mgr = CacheManager(cfg.cache.dir, cfg)
            ext_set = frozenset(get_extensions(include_text_files, extensions))
            file_hashes = HashTracker.compute_hashes(path, ext_set)
            cache_mgr.save_to_disk(str(path), cls(resolved_model, bm25_idx, vs, chunks, root=path, config=cfg), file_hashes)
            logger.info("Cache saved.")

        return cls(resolved_model, bm25_idx, vs, chunks, root=path, reranker=reranker, config=cfg)

    @classmethod
    def from_git(
        cls,
        url: str,
        ref: str | None = None,
        model: object | None = None,
        extensions: Sequence[str] | None = None,
        include_text_files: bool = False,
        config: SembleConfig | None = None,
    ) -> SembleIndex:
        cfg = config or SembleConfig()
        resolved_model = cls._resolve_model(model, cfg)
        reranker = create_reranker(cfg.reranker) if cfg.reranker.backend != "rules" else None

        with tempfile.TemporaryDirectory() as tmp_dir:
            cmd = ["git", "clone", "--depth", "1", *(["--branch", ref] if ref else []), "--", url, tmp_dir]
            try:
                result = subprocess.run(
                    cmd, capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=_GIT_CLONE_TIMEOUT
                )
            except FileNotFoundError:
                raise RuntimeError("git is not installed or not on PATH") from None
            except subprocess.TimeoutExpired:
                raise RuntimeError(f"git clone timed out for {url!r} (limit: {_GIT_CLONE_TIMEOUT} s)") from None
            if result.returncode != 0:
                raise RuntimeError(f"git clone failed for {url!r}:\n{result.stderr.strip()}")

            resolved_path = Path(tmp_dir).resolve()
            vs = create_vector_store(cfg.vector_store, resolved_model.dim)
            bm25_idx, vs, chunks = create_index_from_path(
                resolved_path,
                model=resolved_model,
                extensions=extensions,
                include_text_files=include_text_files,
                display_root=resolved_path,
                vector_store=vs,
            )
            return cls(resolved_model, bm25_idx, vs, chunks, root=resolved_path, reranker=reranker, config=cfg)

    def find_related(self, source: Chunk | SearchResult, *, top_k: int = 5) -> list[SearchResult]:
        target = source.chunk if isinstance(source, SearchResult) else source
        selector = self._get_selector_vector(filter_languages=[target.language]) if target.language else None
        results = _search_semantic(target.content, self.model, self._semantic_index, self.chunks, top_k + 1, selector)
        results = [r for r in results if r.chunk != target][:top_k]
        save_search_stats(results, CallType.FIND_RELATED, self._file_sizes)
        return results

    def _get_selector_vector(
        self, filter_languages: list[str] | None = None, filter_paths: list[str] | None = None
    ) -> npt.NDArray[np.int_] | None:
        selector: list[int] = []
        for language in filter_languages or []:
            selector.extend(self._language_mapping.get(language, []))
        for filename in filter_paths or []:
            selector.extend(self._file_mapping.get(filename, []))
        return np.unique(selector) if selector else None

    def search(
        self,
        query: str,
        top_k: int = 10,
        alpha: float | None = None,
        filter_languages: list[str] | None = None,
        filter_paths: list[str] | None = None,
        rerank: bool = True,
    ) -> list[SearchResult]:
        if not self.chunks or not query.strip():
            return []
        selector = self._get_selector_vector(filter_languages, filter_paths)
        results = search(
            query,
            self.model,
            self._semantic_index,
            self._bm25_index,
            self.chunks,
            top_k,
            alpha=alpha,
            selector=selector,
            rerank=rerank,
            reranker=self._reranker,
        )
        save_search_stats(results, CallType.SEARCH, self._file_sizes)
        return results
