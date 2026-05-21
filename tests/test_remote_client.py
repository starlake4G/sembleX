from __future__ import annotations

import io
import json
from typing import Any
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError

import pytest

from semble.remote.client import RemoteSembleClient


@pytest.fixture
def client() -> RemoteSembleClient:
    return RemoteSembleClient(base_url="http://srv:8080/", api_key="tok", timeout=5)


def _ok_response(body: dict[str, Any]) -> MagicMock:
    fake = MagicMock()
    fake.__enter__ = MagicMock(return_value=fake)
    fake.__exit__ = MagicMock(return_value=False)
    fake.read.return_value = json.dumps(body).encode("utf-8")
    return fake


def test_search_returns_formatted_results(client: RemoteSembleClient) -> None:
    response = _ok_response({
        "results": [
            {"score": 0.9, "chunk": {"content": "foo", "file_path": "a.py", "start_line": 1, "end_line": 2}},
        ]
    })
    with patch("semble.remote.client.urlopen", return_value=response):
        text = client.search("foo", repo="https://x")
    assert "foo" in text
    assert "a.py" in text


def test_index_response_text(client: RemoteSembleClient) -> None:
    response = _ok_response({"repo_id": "rid", "chunk_count": 3, "indexed": True})
    with patch("semble.remote.client.urlopen", return_value=response):
        text = client.index(repo="https://x", ref="main")
    assert "indexed" in text
    assert "rid" in text


def test_409_translates_to_friendly_message(client: RemoteSembleClient) -> None:
    err = HTTPError(
        "http://srv:8080/v1/search", 409, "Conflict", {}, io.BytesIO(b'{"error":"not_indexed"}'),
    )
    with patch("semble.remote.client.urlopen", side_effect=err):
        with pytest.raises(RuntimeError, match="not indexed"):
            client.search("foo", repo="https://x")


def test_other_http_error_passthrough(client: RemoteSembleClient) -> None:
    err = HTTPError(
        "http://srv:8080/v1/search", 500, "boom", {}, io.BytesIO(b"server error"),
    )
    with patch("semble.remote.client.urlopen", side_effect=err):
        with pytest.raises(RuntimeError, match="500"):
            client.search("foo", repo="https://x")


def test_bearer_header_is_sent(client: RemoteSembleClient) -> None:
    response = _ok_response({"results": []})

    captured: dict[str, Any] = {}

    def fake_urlopen(request: Any, timeout: float) -> Any:
        captured["headers"] = dict(request.headers)
        captured["url"] = request.full_url
        return response

    with patch("semble.remote.client.urlopen", side_effect=fake_urlopen):
        client.search("foo", repo="https://x")
    # urllib lower-cases custom header names internally
    auth = captured["headers"].get("Authorization") or captured["headers"].get("authorization")
    assert auth == "Bearer tok"
    assert captured["url"].endswith("/v1/search")
