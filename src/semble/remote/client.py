from __future__ import annotations

import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from semble.config import RemoteConfig
from semble.types import Chunk, SearchResult
from semble.utils import _format_results


class RemoteSembleClient:
    """HTTP client for a remote Semble index service."""

    def __init__(self, base_url: str, api_key: str | None = None, timeout: float = 120.0) -> None:
        """Create a client for a remote Semble service."""
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout

    @classmethod
    def from_config(cls, config: RemoteConfig) -> "RemoteSembleClient":
        """Create a client from remote configuration."""
        return cls(base_url=config.base_url, api_key=config.api_key, timeout=config.timeout)

    def search(self, query: str, *, repo: str, top_k: int = 5, include_text_files: bool = False) -> str:
        """Search a repository through the remote service."""
        data = self._post(
            "/v1/search",
            {"query": query, "repo": repo, "top_k": top_k, "include_text_files": include_text_files},
        )
        return self._format_response(f"Search results for: {query!r}", data)

    def index(
        self,
        *,
        repo: str,
        include_text_files: bool = False,
        force: bool = False,
        ref: str | None = None,
    ) -> str:
        """Request explicit remote indexing for a repository visible to the server."""
        payload: dict[str, object] = {
            "repo": repo,
            "include_text_files": include_text_files,
            "force": force,
        }
        if ref is not None:
            payload["ref"] = ref
        data = self._post("/v1/index", payload)
        if isinstance(data.get("message"), str):
            return str(data["message"])
        repo_id = data.get("repo_id", "unknown")
        chunk_count = data.get("chunk_count", 0)
        indexed = data.get("indexed", False)
        status = "indexed" if indexed else "already indexed"
        return f"Remote repository {status}: {repo_id} ({chunk_count} chunks)"

    def find_related(
        self,
        file_path: str,
        line: int,
        *,
        repo: str,
        top_k: int = 5,
        include_text_files: bool = False,
    ) -> str:
        """Find chunks related to a file location through the remote service."""
        data = self._post(
            "/v1/find-related",
            {
                "file_path": file_path,
                "line": line,
                "repo": repo,
                "top_k": top_k,
                "include_text_files": include_text_files,
            },
        )
        return self._format_response(f"Chunks related to {file_path}:{line}", data)

    def _post(self, path: str, payload: dict[str, object]) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        request = Request(f"{self._base_url}{path}", data=body, headers=headers, method="POST")
        try:
            with urlopen(request, timeout=self._timeout) as response:
                raw = response.read().decode("utf-8")
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            if exc.code == 409:
                raise RuntimeError(
                    "Repository not indexed on the remote server. Run `semble index <repo>` first."
                ) from exc
            raise RuntimeError(f"Remote Semble request failed ({exc.code}): {detail or exc.reason}") from exc
        except URLError as exc:
            raise RuntimeError(f"Remote Semble request failed: {exc.reason}") from exc

        try:
            data = json.loads(raw) if raw else {}
        except json.JSONDecodeError as exc:
            raise RuntimeError("Remote Semble returned invalid JSON") from exc
        if not isinstance(data, dict):
            raise RuntimeError("Remote Semble returned an unexpected response")
        return data

    def _format_response(self, title: str, data: dict[str, Any]) -> str:
        text = data.get("text")
        if isinstance(text, str):
            return text

        message = data.get("message")
        if isinstance(message, str):
            return message

        results_data = data.get("results", [])
        if not isinstance(results_data, list) or not results_data:
            return "No results found."

        results: list[SearchResult] = []
        for item in results_data:
            if isinstance(item, dict):
                results.append(self._result_from_dict(item))
        return _format_results(title, results) if results else "No results found."

    def _result_from_dict(self, data: dict[str, Any]) -> SearchResult:
        chunk_data = data.get("chunk") if isinstance(data.get("chunk"), dict) else data
        assert isinstance(chunk_data, dict)
        chunk = Chunk(
            content=str(chunk_data.get("content", "")),
            file_path=str(chunk_data.get("file_path", "")),
            start_line=int(chunk_data.get("start_line", 1)),
            end_line=int(chunk_data.get("end_line", chunk_data.get("start_line", 1))),
            language=chunk_data.get("language") if isinstance(chunk_data.get("language"), str) else None,
        )
        return SearchResult(chunk=chunk, score=float(data.get("score", 0.0)))
