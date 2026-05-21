from __future__ import annotations

import asyncio
import logging
from collections import OrderedDict
from pathlib import Path
from typing import Annotated

from mcp.server.fastmcp import FastMCP
from pydantic import Field

from semble.backends.embedding import create_embedding_provider
from semble.cache.manager import CacheManager
from semble.config import SembleConfig
from semble.index.index import SembleIndex
from semble.interfaces import EmbeddingProvider
from semble.monitor import FileMonitor
from semble.remote import RemoteSembleClient
from semble.utils import _format_results, _is_git_url, _resolve_chunk

logger = logging.getLogger(__name__)

_REPO_DESCRIPTION = (
    "https:// or http:// git URL (e.g. https://github.com/org/repo) or local directory path to index and search. "
    "Required when no default index was configured at startup. "
    "The index is cached after the first call, so repeat queries are fast."
)

_CACHE_MAX_SIZE = 10


def _resolve_source(repo: str | None, default_source: str | None) -> str:
    if repo is not None and _is_git_url(repo) and not repo.startswith(("https://", "http://")):
        raise ValueError(f"Only https://, http://, or local directory paths are accepted as `repo`. Got: {repo!r}")
    source = repo or default_source
    if not source:
        raise ValueError(
            "No repo specified and no default index. "
            "Pass an https:// or http:// git URL or local directory path as `repo`."
        )
    return source


async def _get_index(
    repo: str | None,
    default_source: str | None,
    cache: _IndexCache,
) -> "SembleIndex":
    source = _resolve_source(repo, default_source)
    try:
        return await cache.get(source)
    except Exception as exc:
        raise ValueError(f"Failed to index {source!r}: {exc}") from exc


def create_server(cache: _IndexCache, default_source: str | None = None) -> FastMCP:
    """Create the local-index MCP server."""
    server = FastMCP(
        "semble",
        instructions=(
            "Instant code search for any local or remote git repository. "
            "Call `search` to find relevant code; call `find_related` on a result to discover similar code elsewhere. "
            "When working in a local project, pass the project root as `repo`. "
            "For remote repos, pass an explicit https:// URL. Never guess or infer URLs. "
            "Prefer these tools over Grep, Glob, or Read for any question about how code works."
        ),
    )

    @server.tool()
    async def search(
        query: Annotated[str, Field(description="Natural language or code query.")],
        repo: Annotated[str | None, Field(description=_REPO_DESCRIPTION)] = None,
        top_k: Annotated[int, Field(description="Number of results to return.", ge=1)] = 5,
    ) -> str:
        try:
            index = await _get_index(repo, default_source, cache)
        except ValueError as exc:
            return str(exc)
        results = index.search(query, top_k=top_k)
        if not results:
            return "No results found."
        return _format_results(f"Search results for: {query!r}", results)

    @server.tool()
    async def find_related(
        file_path: Annotated[
            str,
            Field(description="Path to the file as stored in the index (use file_path from a search result)."),
        ],
        line: Annotated[int, Field(description="Line number (1-indexed).")],
        repo: Annotated[str | None, Field(description=_REPO_DESCRIPTION)] = None,
        top_k: Annotated[int, Field(description="Number of similar chunks to return.", ge=1)] = 5,
    ) -> str:
        try:
            index = await _get_index(repo, default_source, cache)
        except ValueError as exc:
            return str(exc)
        chunk = _resolve_chunk(index.chunks, file_path, line)
        if chunk is None:
            return (
                f"No chunk found at {file_path}:{line}. "
                "Make sure the file is indexed and the line number is within a known chunk."
            )
        results = index.find_related(chunk, top_k=top_k)
        if not results:
            return f"No related chunks found for {file_path}:{line}."
        return _format_results(f"Chunks related to {file_path}:{line}", results)

    return server


def create_remote_server(
    client: RemoteSembleClient,
    default_source: str | None = None,
    include_text_files: bool = False,
) -> FastMCP:
    """Create the remote-index MCP server."""
    server = FastMCP(
        "semble",
        instructions=(
            "Remote Semble code search. Call `search` to find relevant code; call `find_related` on a result "
            "to discover similar code elsewhere. Local indexing, embeddings, and vector search run in the "
            "remote service."
        ),
    )

    @server.tool()
    async def search(
        query: Annotated[str, Field(description="Natural language or code query.")],
        repo: Annotated[str | None, Field(description=_REPO_DESCRIPTION)] = None,
        top_k: Annotated[int, Field(description="Number of results to return.", ge=1)] = 5,
    ) -> str:
        try:
            source = _resolve_source(repo, default_source)
            return await asyncio.to_thread(
                client.search,
                query,
                repo=source,
                top_k=top_k,
                include_text_files=include_text_files,
            )
        except (RuntimeError, ValueError) as exc:
            return str(exc)

    @server.tool()
    async def find_related(
        file_path: Annotated[
            str,
            Field(description="Path to the file as stored in the index (use file_path from a search result)."),
        ],
        line: Annotated[int, Field(description="Line number (1-indexed).")],
        repo: Annotated[str | None, Field(description=_REPO_DESCRIPTION)] = None,
        top_k: Annotated[int, Field(description="Number of similar chunks to return.", ge=1)] = 5,
    ) -> str:
        try:
            source = _resolve_source(repo, default_source)
            return await asyncio.to_thread(
                client.find_related,
                file_path,
                line,
                repo=source,
                top_k=top_k,
                include_text_files=include_text_files,
            )
        except (RuntimeError, ValueError) as exc:
            return str(exc)

    return server


async def serve(
    path: str | None = None,
    ref: str | None = None,
    include_text_files: bool = False,
    config: SembleConfig | None = None,
) -> None:
    """Run the Semble MCP server."""
    cfg = config or SembleConfig()
    if cfg.index.backend == "remote":
        server = create_remote_server(
            RemoteSembleClient.from_config(cfg.remote),
            default_source=path,
            include_text_files=include_text_files,
        )
        await server.run_stdio_async()
        return

    model = await asyncio.to_thread(create_embedding_provider, cfg.embedding)
    cache = _IndexCache(model=model, config=cfg, include_text_files=include_text_files)
    if path:
        await cache.get(path, ref=ref)
        if not _is_git_url(path) and cfg.monitor.enabled:
            await cache.start_watcher(path)

    server = create_server(cache, default_source=path)
    await server.run_stdio_async()


class _IndexCache:
    def __init__(
        self,
        model: EmbeddingProvider,
        config: SembleConfig | None = None,
        include_text_files: bool = False,
    ) -> None:
        self._model = model
        self._config = config or SembleConfig()
        self._include_text_files = include_text_files
        self._tasks: OrderedDict[str, asyncio.Task] = OrderedDict()
        self._watcher_task: asyncio.Task | None = None
        self._disk_cache = CacheManager(self._config.cache.dir, self._config) if self._config.cache.enabled else None

    def _compute_cache_key(self, source: str, ref: str | None = None) -> str:
        is_git = _is_git_url(source)
        return (f"{source}@{ref}" if ref else source) if is_git else str(Path(source).resolve())

    def evict(self, source: str) -> None:
        self._tasks.pop(self._compute_cache_key(source), None)

    async def start_watcher(self, path: str) -> None:
        self._watcher_task = asyncio.create_task(self._watch_loop(path))

    async def _watch_loop(self, path: str) -> None:
        try:
            idx = await self.get(path)
            monitor = FileMonitor(Path(path), idx, self._disk_cache, self._config)
            monitor.initialize_hashes()
            await monitor.watch()
        except Exception:
            logger.warning("Watcher failed for %r", path, exc_info=True)

    async def get(self, source: str, ref: str | None = None) -> "SembleIndex":
        cache_key = self._compute_cache_key(source, ref)

        if cache_key in self._tasks:
            self._tasks.move_to_end(cache_key)
        else:
            if len(self._tasks) >= _CACHE_MAX_SIZE:
                self._tasks.popitem(last=False)
            if _is_git_url(source):
                self._tasks[cache_key] = asyncio.create_task(
                    asyncio.to_thread(
                        SembleIndex.from_git,
                        source,
                        ref=ref,
                        model=self._model,
                        include_text_files=self._include_text_files,
                        config=self._config,
                    )
                )
            else:
                self._tasks[cache_key] = asyncio.create_task(
                    asyncio.to_thread(
                        SembleIndex.from_path,
                        source,
                        model=self._model,
                        include_text_files=self._include_text_files,
                        config=self._config,
                    )
                )
        task = self._tasks[cache_key]
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:  # pragma: no cover
            if task.done():
                self.evict(source)
            raise
        except Exception:
            if self._tasks.get(cache_key) is task:
                self.evict(source)
            raise
