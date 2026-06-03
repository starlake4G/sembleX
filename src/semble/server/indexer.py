from __future__ import annotations

import contextlib
import hashlib
import logging
import shutil
import subprocess
import threading
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from semble.backends.embedding import create_embedding_provider
from semble.backends.sparse import IncrementalSparseIndex
from semble.backends.vector_store.milvus_store import MilvusVectorStore
from semble.chunking import chunk_source
from semble.config import SembleConfig
from semble.index.file_walker import walk_files
from semble.index.files import detect_language, get_extensions
from semble.interfaces import EmbeddingFailure
from semble.search import Candidate, alpha_for, fuse_hybrid, rank_candidates
from semble.server.metadata import MetadataStore, RepoNotIndexedError, StoredChunk
from semble.types import Chunk, SearchResult
from semble.utils import _is_git_url, display_name_for

logger = logging.getLogger(__name__)

_CHUNKER_VERSION = "v1"


@dataclass(frozen=True, slots=True)
class IndexOutcome:
    repo_id: str
    chunk_count: int
    indexed: bool


class GlobalIndex:
    """Cross-repo code index: global Milvus dense store + one global BM25 + local metadata.

    Dense vectors live in a remote Milvus collection (one ``namespace`` per repo);
    chunk metadata lives in a local SQLite store; a single in-memory BM25 index
    (persisted to disk) covers all repos. Search and find_related operate globally
    across every indexed repo, attributing each result to its source repo.
    """

    def __init__(self, config: SembleConfig | None = None) -> None:
        """Connect to Milvus and prewarm the local metadata store and global BM25 index."""
        self._config = config or SembleConfig()
        self._core_dir = self._config.core.dir.expanduser()
        self._core_dir.mkdir(parents=True, exist_ok=True)
        self._metadata = MetadataStore(self._core_dir / "metadata.sqlite3")
        self._model = create_embedding_provider(self._config.embedding)
        self._vector_store = MilvusVectorStore(self._config.milvus, dim=self._model.dim)
        self._lock = threading.RLock()
        self._sparse = IncrementalSparseIndex()
        self._repo_sources: dict[str, str] = {}
        self._sparse_loaded_mtime: float = 0.0
        self._load_or_build_sparse()
        self._refresh_repos()

    # ------------------------------------------------------------------ prewarm / reload

    def _sparse_dir(self) -> Path:
        return self._core_dir / "sparse"

    def _sparse_marker(self) -> Path:
        return self._sparse_dir() / "incremental_meta.json"

    def _load_or_build_sparse(self) -> None:
        marker = self._sparse_marker()
        if marker.exists():
            try:
                self._sparse.load(self._sparse_dir())
                self._sparse_loaded_mtime = marker.stat().st_mtime
                expected = self._metadata.total_chunk_count()
                if self._sparse.doc_count != expected:
                    logger.warning(
                        "BM25 index has %d docs but metadata has %d chunks; rebuilding "
                        "(likely an interrupted deferred-save batch)",
                        self._sparse.doc_count,
                        expected,
                    )
                    self._rebuild_sparse_from_metadata()
                    return
                logger.info("Loaded global BM25 index from %s", self._sparse_dir())
                return
            except Exception:
                logger.warning("Failed to load BM25 index; rebuilding from metadata", exc_info=True)
        self._rebuild_sparse_from_metadata()

    def _rebuild_sparse_from_metadata(self) -> None:
        records = self._metadata.iter_all_chunks()
        chunks = [chunk for _, _, chunk in records]
        ids = [cid for cid, _, _ in records]
        self._sparse = IncrementalSparseIndex()
        if chunks:
            self._sparse.build(chunks, ids)
        self._save_sparse()
        logger.info("Rebuilt global BM25 index from %d chunks", len(chunks))

    def _save_sparse(self) -> None:
        self._sparse.save(self._sparse_dir())
        self._sparse_loaded_mtime = self._sparse_marker().stat().st_mtime

    def _refresh_repos(self) -> None:
        self._repo_sources = self._metadata.repo_sources()

    def _maybe_reload(self) -> None:
        """Reload the BM25 index if another process re-indexed since our last load."""
        marker = self._sparse_marker()
        if not marker.exists():
            return
        mtime = marker.stat().st_mtime
        if mtime > self._sparse_loaded_mtime:
            with self._lock:
                if mtime > self._sparse_loaded_mtime:
                    try:
                        self._sparse.load(self._sparse_dir())
                        self._sparse_loaded_mtime = mtime
                        self._refresh_repos()
                        logger.info("Reloaded global BM25 index after external update")
                    except Exception:
                        logger.warning("Failed to reload BM25 index", exc_info=True)

    # ------------------------------------------------------------------ indexing

    def index_repo(
        self,
        repo: str,
        *,
        include_text_files: bool = False,
        force: bool = False,
        ref: str | None = None,
        persist: bool = True,
    ) -> IndexOutcome:
        """Index (or re-index with *force*) a repo into the global Milvus + BM25 + metadata stores.

        :param repo: Local path or git URL of the repository.
        :param include_text_files: Also index non-code text files.
        :param force: Rebuild even if the repo is already indexed.
        :param ref: Branch or tag to check out (git URLs only).
        :param persist: Write the global BM25 index to disk before returning. Set
            ``False`` when indexing many repos in a loop and call :meth:`flush` once at
            the end (and periodically) — saving the whole BM25 index per repo is O(total
            chunks) and dominates the time between repos. Vector and SQLite stores are
            always persisted regardless of this flag.
        :returns: The outcome (repo id, chunk count, whether it was (re)indexed).
        """
        repo_id = self._repo_id(repo)
        with self._lock:
            if not force and self._metadata.has_repo(repo_id):
                return IndexOutcome(repo_id, self._metadata.repo_chunk_count(repo_id), indexed=False)

            source_path, commit_sha = self._prepare_source(repo, ref=ref, force=force)

            old_ids = self._metadata.chunk_ids(repo_id)
            self._metadata.start_repo_replace(repo_id)
            self._vector_store.clear(repo_id)
            if old_ids:
                self._sparse.remove_documents(old_ids)

            chunks, ids = self._index_chunks_streaming(repo_id, source_path, include_text_files)
            if not chunks:
                if persist:
                    self._save_sparse()
                raise ValueError(f"No supported files found under {source_path}.")

            self._sparse.add_documents(chunks, ids)
            if persist:
                self._save_sparse()

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
                chunk_count=len(chunks),
            )
            self._refresh_repos()
            return IndexOutcome(repo_id, len(chunks), indexed=True)

    def flush(self) -> None:
        """Persist the in-memory global BM25 index to disk.

        Call after a batch of ``index_repo(..., persist=False)`` calls to write the
        accumulated index once instead of once per repo.
        """
        with self._lock:
            self._save_sparse()

    # ------------------------------------------------------------------ search

    def resolve_repo_id(self, repo: str) -> str | None:
        """Resolve a repo *source* string or a filesystem path to an indexed ``repo_id``.

        Tries an exact source match first (the string used at index time), then falls
        back to locating the indexed repo whose ``source_path`` contains *repo* as a
        path. This lets a working directory — or any file/subdirectory under a repo —
        resolve to the repo that owns it. Returns ``None`` if nothing matches.

        :param repo: A repo source string, git URL, or local path (e.g. a workspace dir).
        :returns: The matching ``repo_id``, or ``None`` if no indexed repo contains it.
        """
        exact = self._repo_id(repo)
        if self._metadata.has_repo(exact):
            return exact
        if _is_git_url(repo):
            return None
        try:
            target = Path(repo).expanduser().resolve()
        except (OSError, ValueError):
            return None
        best_id: str | None = None
        best_depth = -1
        for repo_id, source_path in self._metadata.repo_source_paths().items():
            try:
                base = Path(source_path).resolve()
            except (OSError, ValueError):
                continue
            if target == base or base in target.parents:
                depth = len(base.parts)
                if depth > best_depth:
                    best_id, best_depth = repo_id, depth
        return best_id

    def search(self, query: str, *, top_k: int = 5, repo: str | None = None) -> list[SearchResult]:
        """Hybrid (dense + BM25) search across all repos, or one repo when *repo* is given.

        :param query: Natural-language or code/symbol query.
        :param top_k: Number of results to return.
        :param repo: Optional repo source/path to restrict to. A working directory or
            any path under an indexed repo resolves to that repo.
        :returns: Ranked search results, each attributed to its source repo.
        :raises RepoNotIndexedError: If *repo* is given but no indexed repo contains it.
        """
        if not query.strip():
            return []
        self._maybe_reload()
        coarse = max(top_k * self._config.reranker.coarse_multiplier, top_k)
        qvec = self._embed(query)

        restrict_id = None
        if repo is not None:
            restrict_id = self.resolve_repo_id(repo)
            if restrict_id is None:
                raise RepoNotIndexedError(self._repo_id(repo), source=repo)
        if restrict_id is not None:
            dense = [(cid, sim) for cid, sim in self._vector_store.query(restrict_id, qvec, k=coarse)]
        else:
            dense = [(cid, sim) for cid, _ns, sim in self._vector_store.query_global(qvec, k=coarse)]

        sparse = self._sparse.query(query, k=coarse)
        if restrict_id is not None:
            allowed = set(self._metadata.chunk_ids(restrict_id))
            sparse = [(cid, s) for cid, s in sparse if cid in allowed]

        fused = fuse_hybrid(dense, sparse, alpha_for(query))
        candidates = self._resolve_candidates(fused)
        return rank_candidates(query, candidates, top_k, apply_query_boosts=True)

    def find_related(
        self,
        file_path: str,
        line: int,
        *,
        repo: str,
        top_k: int = 5,
        exclude_source_repo: bool = True,
    ) -> list[SearchResult]:
        """Find code across all repos similar to the chunk at *file_path*:*line* in *repo*."""
        self._maybe_reload()
        repo_id = self._repo_id(repo)
        if not self._metadata.has_repo(repo_id):
            raise RepoNotIndexedError(repo_id, source=repo)
        found = self._metadata.find_chunk(repo_id, file_path, line)
        if found is None:
            return []
        target_id, target = found
        qvec = self._embed(target.content)
        exclude = repo_id if exclude_source_repo else None
        dense = self._vector_store.query_global(qvec, k=top_k * 5 + 1, exclude_namespace=exclude)
        fused = {cid: sim for cid, _ns, sim in dense if cid != target_id}
        candidates = self._resolve_candidates(fused)
        return rank_candidates(None, candidates, top_k, apply_query_boosts=False)

    def status(self) -> dict[str, object]:
        """Return a summary of the global index: repo count, chunk count, embedding, and sources."""
        self._maybe_reload()
        sources = self._repo_sources
        return {
            "repos": len(sources),
            "chunks": self._sparse.doc_count,
            "embedding_model": self._embedding_model_name(),
            "embedding_dim": self._model.dim,
            "milvus": self._config.milvus.uri,
            "core_dir": str(self._core_dir),
            "sources": sorted(sources.values()),
        }

    # ------------------------------------------------------------------ retrieval internals

    def _embed(self, text: str) -> np.ndarray:
        return np.asarray(self._model.encode([text]), dtype=np.float32)

    def _resolve_candidates(self, fused: dict[str, float]) -> list[Candidate]:
        if not fused:
            return []
        recs = self._metadata.get_chunks_with_repo(list(fused))
        candidates: list[Candidate] = []
        for cid, score in fused.items():
            rec = recs.get(cid)
            if rec is None:
                continue
            repo_id, chunk = rec
            candidates.append(
                Candidate(
                    chunk_id=cid,
                    repo_id=repo_id,
                    repo_name=self._repo_name(repo_id),
                    repo_source=self._repo_sources.get(repo_id, ""),
                    chunk=chunk,
                    score=score,
                )
            )
        return candidates

    def _repo_name(self, repo_id: str) -> str:
        source = self._repo_sources.get(repo_id)
        return display_name_for(source) if source else repo_id

    # ------------------------------------------------------------------ indexing internals

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

    def _flush_batch(self, repo_id: str, batch: Sequence[StoredChunk]) -> tuple[list[Chunk], list[str]]:
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

    def _prepare_source(self, repo: str, *, ref: str | None, force: bool) -> tuple[Path, str | None]:
        if _is_git_url(repo):
            if not repo.startswith(("https://", "http://")):
                raise ValueError(f"Only https:// or http:// git URLs can be cloned: {repo!r}")
            path = self._clone_or_update(repo, ref=ref, force=force)
            return path, self._git_head_sha(path)

        path = Path(repo).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Repository path does not exist: {repo}")
        if not path.is_dir():
            raise NotADirectoryError(f"Repository path is not a directory: {repo}")
        return path, None

    def _clone_or_update(self, url: str, *, ref: str | None, force: bool) -> Path:
        key = self._hash(f"{url}|{ref or 'HEAD'}", size=16)
        repo_dir = self._core_dir / "repos" / key
        if force and repo_dir.exists():
            shutil.rmtree(repo_dir, ignore_errors=True)

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
                timeout=self._config.core.clone_timeout,
            )
        except FileNotFoundError:
            raise RuntimeError("git is not installed or not on PATH") from None
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
                    timeout=self._config.core.clone_timeout,
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
        return result.stdout.strip() or None if result.returncode == 0 else None

    def _repo_id(self, repo: str) -> str:
        raw = f"{repo}|model={self._embedding_model_name()}|dim={self._model.dim}|chunker={_CHUNKER_VERSION}"
        return self._hash(raw, size=32)

    def _stored_chunk(self, repo_id: str, chunk: Chunk) -> StoredChunk:
        content_hash = self._hash(chunk.content, size=32)
        raw = f"{repo_id}:{chunk.file_path}:{chunk.start_line}:{chunk.end_line}:{content_hash}"
        return StoredChunk(chunk_id=self._hash(raw, size=32), chunk=chunk, content_hash=content_hash)

    def _embedding_model_name(self) -> str:
        provider_name = getattr(self._model, "name", type(self._model).__name__)
        return f"{provider_name}:{self._config.embedding.openai_model}"

    @staticmethod
    def _hash(value: str, *, size: int) -> str:
        return hashlib.sha256(value.encode("utf-8")).hexdigest()[:size]
