from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from semble.mcp import create_server, serve
from semble.server.metadata import RepoNotIndexedError
from semble.types import SearchResult
from semble.utils import _format_results, _is_git_url
from tests.conftest import make_chunk


def _tool_text(result: Any) -> str:
    """Extract the text string from a FastMCP call_tool result."""
    return result[0][0].text


def _result(content: str, path: str, repo_name: str, source: str, score: float) -> SearchResult:
    return SearchResult(
        chunk=make_chunk(content, path),
        score=score,
        repo_id=repo_name,
        repo_name=repo_name,
        repo_source=source,
    )


@pytest.mark.anyio
async def test_search_tool_formats_repo_attributed_results() -> None:
    index = MagicMock()
    index.search.return_value = [_result("def bar(): pass", "src/bar.py", "repoA", "/x/repoA", 0.9)]
    server = create_server(index)
    text = _tool_text(await server.call_tool("search", {"query": "bar"}))
    assert "repoA :: src/bar.py" in text
    assert "repo: /x/repoA" in text
    assert "0.900" in text
    index.search.assert_called_once()


@pytest.mark.anyio
async def test_search_tool_no_results() -> None:
    index = MagicMock()
    index.search.return_value = []
    server = create_server(index)
    assert "No results found" in _tool_text(await server.call_tool("search", {"query": "x"}))


@pytest.mark.anyio
async def test_search_global_scope_searches_all_repos() -> None:
    index = MagicMock()
    index.search.return_value = []
    server = create_server(index, workspace="/x/repoA")
    await server.call_tool("search", {"query": "x"})  # default scope=global
    _, kwargs = index.search.call_args
    assert kwargs["repo"] is None


@pytest.mark.anyio
async def test_search_workspace_scope_restricts_to_workspace() -> None:
    index = MagicMock()
    index.search.return_value = [_result("def bar(): pass", "bar.py", "repoA", "/x/repoA", 0.9)]
    server = create_server(index, workspace="/x/repoA")
    text = _tool_text(await server.call_tool("search", {"query": "bar", "scope": "workspace"}))
    _, kwargs = index.search.call_args
    assert kwargs["repo"] == "/x/repoA"
    assert "workspace" in text


@pytest.mark.anyio
async def test_search_explicit_repo_overrides_scope() -> None:
    index = MagicMock()
    index.search.return_value = []
    server = create_server(index, workspace="/x/repoA")
    await server.call_tool("search", {"query": "x", "scope": "workspace", "repo": "/x/repoB"})
    _, kwargs = index.search.call_args
    assert kwargs["repo"] == "/x/repoB"


@pytest.mark.anyio
async def test_search_workspace_scope_without_workspace() -> None:
    index = MagicMock()
    server = create_server(index, workspace=None)
    text = _tool_text(await server.call_tool("search", {"query": "x", "scope": "workspace"}))
    assert "workspace" in text.lower()
    index.search.assert_not_called()


@pytest.mark.anyio
async def test_search_workspace_not_indexed_message() -> None:
    index = MagicMock()
    index.search.side_effect = RepoNotIndexedError("rid", source="/x/repoA")
    server = create_server(index, workspace="/x/repoA")
    text = _tool_text(await server.call_tool("search", {"query": "x", "scope": "workspace"}))
    assert "not in the index" in text.lower()
    assert "scope='global'" in text


@pytest.mark.anyio
async def test_find_related_tool_returns_cross_repo_matches() -> None:
    index = MagicMock()
    index.find_related.return_value = [_result("class Foo: pass", "foo.py", "repoB", "/x/repoB", 0.8)]
    server = create_server(index)
    text = _tool_text(
        await server.call_tool("find_related", {"repo": "/x/repoA", "file_path": "a.py", "line": 3})
    )
    assert "repoB :: foo.py" in text
    _, kwargs = index.find_related.call_args
    assert kwargs["exclude_source_repo"] is True


@pytest.mark.anyio
async def test_find_related_tool_repo_not_indexed() -> None:
    index = MagicMock()
    index.find_related.side_effect = RepoNotIndexedError("rid", source="/x/repoA")
    server = create_server(index)
    text = _tool_text(
        await server.call_tool("find_related", {"repo": "/x/repoA", "file_path": "a.py", "line": 3})
    )
    assert "not indexed" in text.lower()


@pytest.mark.anyio
async def test_serve_runs_stdio() -> None:
    with (
        patch("semble.server.kernel.connect_or_spawn", return_value=MagicMock()),
        patch("mcp.server.fastmcp.FastMCP.run_stdio_async", new_callable=AsyncMock) as mock_run,
    ):
        await serve()
    mock_run.assert_called_once()


def test_format_results_shows_repo_and_source() -> None:
    results = [_result("def f(): pass", "f.py", "repoA", "/x/repoA", 0.5)]
    out = _format_results("Header", results)
    assert "repoA :: f.py" in out
    assert "repo: /x/repoA" in out
    assert "0.500" in out


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("https://github.com/org/repo", True),
        ("git@github.com:org/repo", True),
        ("/local/path/to/repo", False),
        ("./relative/path", False),
    ],
)
def test_is_git_url(path: str, expected: bool) -> None:
    assert _is_git_url(path) is expected
