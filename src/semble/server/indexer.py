from __future__ import annotations

import contextlib
import hashlib
import logging
import shutil
import subprocess
import threading
from collections import OrderedDict
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from semble.backends.embedding import create_embedding_provider
from semble.backends.reranker import create_reranker
from semble.backends.sparse import Bm25sSparseIndex
from semble.backends.vector_store.milvus_store import MilvusVectorStore
from semble.chunking import chunk_source
from semble.config import SembleConfig
from semble.index.file_walker import walk_files
from semble.index.files import detect_language, get_extensions
from semble.interfaces import EmbeddingFailure
from semble.search import search as run_search
from semble.server.metadata import MetadataStore, RepoNotIndexedError, StoredChunk
from semble.types import Chunk, SearchResult
from semble.utils import _is_git_url

logger = logging.getLogger(__name__)

_CHUNKER_VERSION = "v1"
_REMOTE_INDEX_CACHE = 4  # LRU size for loaded per-repo sparse indices


@dataclass(frozen=True, slots=True)
class IndexOutcome:
    repo_id: str
    chunk_count: int
    indexed: bool


@dataclass(slots=True)
class _LoadedIndex:
    chunks: list[Chunk]
    chunk_ids: list[str]
    by_id: dict[str, Chunk]
    sparse: Bm25sSparseIndex


class RemoteIndexer:
    """Indexes repositories on the server and serves hybrid (BM25 + dense) search."""

    def __init__(self, config: SembleConfig) -> None:
        self._config = config
        self._work_dir = self._config.server.work_dir.expanduser()
        self._work_dir.mkdir(parents=True, exist_ok=True)
        self._metadata = MetadataStore(self._work_dir / "metadata.sqlite3")
        self._model = create_embedding_provider(self._config.embedding)
        self._vector_store = MilvusVectorStore(self._config.milvus, dim=self._model.dim)
        self._reranker = create_reranker(self._config.reranker) if self._config.reranker.backend != "rules" else None
        self._lock = threading.RLock()
        self._loaded: OrderedDict[str, _LoadedIndex] = OrderedDict()

    # ------------------------------------------------------------------ public API

    def index_repo(
        self,
        repo: str,
        *,
        include_text_files: bool = False,
        force: bool = False,
        ref: str | None = None,
    ) -> IndexOutcome:
        repo_id = self._repo_id(repo, include_text_files)
        with self._lock:
            if not force and self._metadata.has_repo(repo_id):
                return IndexOutcome(
                    repo_id=repo_id,
                    chunk_count=self._metadata.repo_chunk_count(repo_id),
                    indexed=False,
                )

            source_path, commit_sha = self._prepare_source(repo, ref=ref, force=force)
            self._clear_orphan_repo_ids(repo, keep=repo_id)
            self._metadata.start_repo_replace(repo_id)
            self._vector_store.clear(repo_id)
            self._drop_sparse(repo_id)
            self._loaded.pop(repo_id, None)

            indexed_chunks, indexed_ids = self._index_chunks_streaming(
                repo_id, source_path, include_text_files,
            )
            if not indexed_chunks:
                raise ValueError(f"No supported files found under {source_path}.")

            sparse = Bm25sSparseIndex()
            sparse.build(indexed_chunks, indexed_ids)
            sparse.save(self._sparse_path(repo_id))

            self._metadata.finish_repo_replace(
                repo_id=repo_id,
                source=repo,
                source_path=str(source_path),
                include_text_files=include_text_files,
                embedding_model=self._embedding_model_name(),
                embedding_dim=self._model.dim,
                chunker_version=_CHUNKER_VERSION,
                commit_sha=commit_sha,
                indexed_at=datetime.now(timezone.utc).isoformat(),
                chunk_count=len(indexed_chunks),
            )
            return IndexOutcome(repo_id=repo_id, chunk_count=len(indexed_chunks), indexed=True)

    def search(
        self,
        query: str,
        *,
        repo: str,
        top_k: int = 5,
        include_text_files: bool = False,
    ) -> list[SearchResult]:
        if not query.strip():
            return []
        repo_id = self._repo_id(repo, include_text_files)
        if not self._metadata.has_repo(repo_id):
            raise RepoNotIndexedError(repo_id, source=repo)
        loaded = self._open(repo_id)
        coarse_k = top_k * self._config.reranker.coarse_multiplier if self._reranker else None
        return run_search(
            query,
            self._model,
            self._vector_store,
            loaded.sparse,
            loaded.chunks,
            loaded.by_id,
            top_k,
            namespace=repo_id,
            rerank=True,
            reranker=self._reranker,
            coarse_k=coarse_k,
        )

    def find_related(
        self,
        file_path: str,
        line: int,
        *,
        repo: str,
        top_k: int = 5,
        include_text_files: bool = False,
    ) -> list[SearchResult]:
        repo_id = self._repo_id(repo, include_text_files)
        if not self._metadata.has_repo(repo_id):
            raise RepoNotIndexedError(repo_id, source=repo)
        loaded = self._open(repo_id)
        found = self._metadata.find_chunk(repo_id, file_path, line)
        if found is None:
            return []
        target_id, target = found
        query_vector = np.asarray(self._model.encode([target.content]), dtype=np.float32)
        hits = self._vector_store.query(repo_id, query_vector, k=top_k + 1)
        out: list[SearchResult] = []
        for cid, score in hits:
            if cid == target_id:
                continue
            chunk = loaded.by_id.get(cid)
            if chunk is not None:
                out.append(SearchResult(chunk=chunk, score=float(score)))
            if len(out) >= top_k:
                break
        return out

    # ----------------------------------------------------------------- internals

    def _index_chunks_streaming(
        self,
        repo_id: str,
        source_path: Path,
        include_text_files: bool,
    ) -> tuple[list[Chunk], list[str]]:
        kept_chunks: list[Chunk] = []
        kept_ids: list[str] = []
        batch: list[StoredChunk] = []
        for chunk in self._iter_chunks(source_path, include_text_files):
            batch.append(self._stored_chunk(repo_id, chunk))
            if len(batch) >= self._config.indexing.embed_batch_size:
                kc, kids = self._flush_batch(repo_id, batch)
                kept_chunks.extend(kc)
                kept_ids.extend(kids)
                batch.clear()
        if batch:
            kc, kids = self._flush_batch(repo_id, batch)
            kept_chunks.extend(kc)
            kept_ids.extend(kids)
        return kept_chunks, kept_ids

    def _flush_batch(
        self, repo_id: str, batch: Sequence[StoredChunk]
    ) -> tuple[list[Chunk], list[str]]:
        texts = [stored.chunk.content for stored in batch]
        try:
            vectors = np.asarray(self._model.encode(texts), dtype=np.float32)
            survivors = list(batch)
        except EmbeddingFailure as exc:
            failed = set(exc.failed_indices)
            survivors = [s for i, s in enumerate(batch) if i not in failed]
            if exc.partial is None or not survivors:
                logger.warning("Batch fully failed; dropping %d chunks", len(batch))
                return [], []
            vectors = np.asarray(exc.partial, dtype=np.float32)
            logger.warning("Batch partially failed; dropped %d/%d chunks", len(failed), len(batch))

        ids = [s.chunk_id for s in survivors]
        self._vector_store.add(repo_id, ids, vectors)
        self._metadata.insert_chunks(repo_id, survivors)
        return [s.chunk for s in survivors], ids

    def _iter_chunks(self, source_path: Path, include_text_files: bool) -> Iterator[Chunk]:
        extensions = get_extensions(include_text_files, None)
        for file_path in walk_files(source_path, extensions):
            with contextlib.suppress(OSError, UnicodeError):
                if file_path.stat().st_size > self._config.indexing.max_file_bytes:
                    continue
                source = file_path.read_text(encoding="utf-8", errors="replace")
                relative_path = file_path.relative_to(source_path).as_posix()
                language = detect_language(file_path)
                yield from chunk_source(source, relative_path, language)

    def _open(self, repo_id: str) -> _LoadedIndex:
        with self._lock:
            cached = self._loaded.get(repo_id)
            if cached is not None:
                self._loaded.move_to_end(repo_id)
                return cached
            pairs = self._metadata.all_chunks(repo_id)
            chunks = [chunk for _, chunk in pairs]
            chunk_ids = [cid for cid, _ in pairs]
            sparse = Bm25sSparseIndex()
            sparse.load(self._sparse_path(repo_id))
            loaded = _LoadedIndex(chunks=chunks, chunk_ids=chunk_ids, by_id=dict(pairs), sparse=sparse)
            self._loaded[repo_id] = loaded
            while len(self._loaded) > _REMOTE_INDEX_CACHE:
                self._loaded.popitem(last=False)
            return loaded

    def _prepare_source(
        self, repo: str, *, ref: str | None, force: bool
    ) -> tuple[Path, str | None]:
        if _is_git_url(repo):
            if not repo.startswith(("https://", "http://")):
                raise ValueError(
                    f"Only https:// or http:// git URLs can be cloned by the remote server: {repo!r}"
                )
            path = self._clone_or_update(repo, ref=ref, force=force)
            commit = self._git_head_sha(path)
            return path, commit

        path = Path(repo).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Repository path does not exist on the remote server: {repo}")
        if not path.is_dir():
            raise NotADirectoryError(f"Repository path is not a directory on the remote server: {repo}")
        return path, None

    def _clone_or_update(self, url: str, *, ref: str | None, force: bool) -> Path:
        key = self._hash(f"{url}|{ref or 'HEAD'}", size=16)
        repo_dir = self._work_dir / "repos" / key
        if force and repo_dir.exists():
            shutil.rmtree(repo_dir)

        if repo_dir.exists():
            self._git_fetch_reset(repo_dir, ref=ref)
            return repo_dir

        repo_dir.parent.mkdir(parents=True, exist_ok=True)
        cmd = ["git", "clone", "--depth", "1", *(["--branch", ref] if ref else []), "--", url, str(repo_dir)]
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                stdin=subprocess.DEVNULL,
                timeout=self._config.server.clone_timeout,
            )
        except FileNotFoundError:
            raise RuntimeError("git is not installed or not on PATH on the remote server") from None
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"git clone timed out for {url!r}") from None
        if result.returncode != 0:
            raise RuntimeError(f"git clone failed for {url!r}:\n{result.stderr.strip()}")
        return repo_dir

    def _git_fetch_reset(self, repo_dir: Path, *, ref: str | None) -> None:
        target = ref or "HEAD"
        for cmd in (
            ["git", "-C", str(repo_dir), "fetch", "--depth", "1", "origin", target],
            ["git", "-C", str(repo_dir), "reset", "--hard", "FETCH_HEAD"],
        ):
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    stdin=subprocess.DEVNULL,
                    timeout=self._config.server.clone_timeout,
                )
            except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
                logger.warning("git update failed (%s); using stale clone", exc)
                return
            if result.returncode != 0:
                logger.warning("git update failed: %s; using stale clone", result.stderr.strip())
                return

    def _git_head_sha(self, repo_dir: Path) -> str | None:
        try:
            result = subprocess.run(
                ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                stdin=subprocess.DEVNULL,
                timeout=10,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return None
        if result.returncode != 0:
            return None
        return result.stdout.strip() or None

    def _clear_orphan_repo_ids(self, source: str, *, keep: str) -> None:
        for stale in self._metadata.repo_ids_for_source(source):
            if stale == keep:
                continue
            logger.info("Removing stale repo_id %s for source %s", stale, source)
            self._vector_store.clear(stale)
            self._metadata.start_repo_replace(stale)
            self._drop_sparse(stale)
            self._loaded.pop(stale, None)

    def _sparse_path(self, repo_id: str) -> Path:
        return self._work_dir / "repos_index" / repo_id / "bm25s"

    def _drop_sparse(self, repo_id: str) -> None:
        target = self._sparse_path(repo_id).parent
        if target.exists():
            shutil.rmtree(target, ignore_errors=True)

    def _repo_id(self, repo: str, include_text_files: bool) -> str:
        model_key = self._embedding_model_name()
        raw = f"{repo}|text={include_text_files}|model={model_key}|dim={self._model.dim}|chunker={_CHUNKER_VERSION}"
        return self._hash(raw, size=32)

    def _stored_chunk(self, repo_id: str, chunk: Chunk) -> StoredChunk:
        content_hash = self._hash(chunk.content, size=32)
        raw = f"{repo_id}:{chunk.file_path}:{chunk.start_line}:{chunk.end_line}:{content_hash}"
        return StoredChunk(chunk_id=self._hash(raw, size=32), chunk=chunk, content_hash=content_hash)

    def _embedding_model_name(self) -> str:
        provider_name = getattr(self._model, "name", type(self._model).__name__)
        if self._config.embedding.backend == "openai_compat":
            return f"{provider_name}:{self._config.embedding.openai_model}"
        return f"{provider_name}:{self._config.embedding.model}"

    @staticmethod
    def _hash(value: str, *, size: int) -> str:
        return hashlib.sha256(value.encode("utf-8")).hexdigest()[:size]
