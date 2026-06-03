from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Annotated, Literal

from mcp.server.fastmcp import FastMCP
from pydantic import Field

from semble.config import SembleConfig
from semble.server.indexer import GlobalIndex
from semble.server.metadata import RepoNotIndexedError
from semble.utils import _format_results

logger = logging.getLogger(__name__)

_REPO_DESCRIPTION = (
    "Source repo of the location: a local directory path or https:// git URL, "
    "exactly as shown on the `repo:` line of a prior search result."
)

Scope = Literal["global", "workspace"]


def create_server(index: GlobalIndex, *, workspace: str | None = None) -> FastMCP:
    """Create the cross-repo MCP server backed by a warm GlobalIndex.

    :param index: The warm cross-repo index to serve.
    :param workspace: Local path treated as the "current working directory" for
        ``scope="workspace"`` searches (defaults to the server's launch directory).
        Resolved to whichever indexed repo contains it.
    """
    server = FastMCP(
        "semble",
        instructions=(
            "Cross-repo code search over all indexed repositories. "
            "Call `search` to find relevant code by description or symbol; results are attributed to "
            "their source repo. By default it searches every indexed repo (`scope=global`), which is "
            "best for finding reference implementations elsewhere; pass `scope=workspace` to restrict "
            "to the current project. Call `find_related` with a result's `repo`, `file_path`, and `line` "
            "to discover similar implementations in other repos. "
            "Prefer these tools over Grep, Glob, or Read for any question about how code works."
        ),
    )

    @server.tool()
    async def search(
        query: Annotated[str, Field(description="Natural language or code/symbol query.")],
        top_k: Annotated[int, Field(description="Number of results to return.", ge=1)] = 5,
        scope: Annotated[
            Scope,
            Field(
                description=(
                    "'global' (default) searches every indexed repo — use this to find how something is "
                    "implemented across other projects. 'workspace' restricts to the current working "
                    "directory's repo. Ignored when `repo` is given."
                )
            ),
        ] = "global",
        repo: Annotated[
            str | None,
            Field(
                description=(
                    "Optional: restrict to a specific indexed repo by its local path or git URL "
                    "(a subdirectory or file path under it also works). Overrides `scope`."
                )
            ),
        ] = None,
    ) -> str:
        target = repo
        if target is None and scope == "workspace":
            if workspace is None:
                return "No workspace directory is configured; pass `repo` explicitly or use scope='global'."
            target = workspace
        try:
            results = await asyncio.to_thread(index.search, query, top_k=top_k, repo=target)
        except RepoNotIndexedError as exc:
            hint = "this workspace" if (repo is None and scope == "workspace") else target
            return f"{exc}. {hint} is not in the index — index it with `semble index {target}`, or use scope='global'."
        if not results:
            return "No results found."
        label = "workspace" if (repo is None and scope == "workspace") else (target or "all repos")
        return _format_results(f"Search results ({label}) for: {query!r}", results)

    @server.tool()
    async def find_related(
        repo: Annotated[str, Field(description=_REPO_DESCRIPTION)],
        file_path: Annotated[str, Field(description="File path as shown in a search result.")],
        line: Annotated[int, Field(description="Line number (1-indexed).")],
        top_k: Annotated[int, Field(description="Number of similar chunks to return.", ge=1)] = 5,
    ) -> str:
        try:
            results = await asyncio.to_thread(
                index.find_related, file_path, line, repo=repo, top_k=top_k, exclude_source_repo=True
            )
        except RepoNotIndexedError as exc:
            return f"{exc}. Index it first with `semble index {repo}`."
        if not results:
            return f"No related code found for {file_path}:{line} in {repo}."
        return _format_results(f"Code related to {file_path}:{line} (excluding {repo})", results)

    return server


async def serve(config: SembleConfig | None = None, *, workspace: str | None = None) -> None:
    """Run the Semble cross-repo MCP server over stdio.

    :param config: Semble configuration (defaults applied when omitted).
    :param workspace: Directory used for ``scope="workspace"`` searches; defaults to
        the process launch directory, which MCP clients set to the active project.
    """
    cfg = config or SembleConfig()
    workspace = workspace or str(Path.cwd())
    index = await asyncio.to_thread(GlobalIndex, cfg)
    server = create_server(index, workspace=workspace)
    await server.run_stdio_async()
