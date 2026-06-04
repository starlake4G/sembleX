import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from semble.cli import main
from semble.projects import load_projects
from semble.server.indexer import IndexOutcome
from semble.types import SearchResult
from tests.conftest import make_chunk


def _result(content: str, path: str, repo_name: str, source: str, score: float) -> SearchResult:
    return SearchResult(
        chunk=make_chunk(content, path),
        score=score,
        repo_id=repo_name,
        repo_name=repo_name,
        repo_source=source,
    )


def test_search_command(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    fake = MagicMock()
    fake.search.return_value = [_result("def foo(): pass", "src/foo.py", "repoA", "/x/repoA", 0.9)]
    monkeypatch.setattr(sys, "argv", ["semble", "search", "foo"])
    with patch("semble.cli._query_backend", return_value=fake):
        main()
    out = capsys.readouterr().out
    assert "repoA :: src/foo.py" in out
    assert "0.900" in out


def test_search_no_results(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    fake = MagicMock()
    fake.search.return_value = []
    monkeypatch.setattr(sys, "argv", ["semble", "search", "nothing"])
    with patch("semble.cli._query_backend", return_value=fake):
        main()
    assert "No results found" in capsys.readouterr().out


def test_find_related_command(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    fake = MagicMock()
    fake.find_related.return_value = [_result("class Bar: pass", "src/bar.py", "repoB", "/y", 0.8)]
    monkeypatch.setattr(sys, "argv", ["semble", "find-related", "/y", "src/bar.py", "1"])
    with patch("semble.cli._query_backend", return_value=fake):
        main()
    out = capsys.readouterr().out
    assert "src/bar.py" in out
    _, kwargs = fake.find_related.call_args
    assert kwargs["exclude_source_repo"] is True


def test_index_command(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    fake = MagicMock()
    fake.index_repo.return_value = IndexOutcome(repo_id="r1", chunk_count=5, indexed=True)
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.setattr(sys, "argv", ["semble", "index", str(repo)])
    with patch("semble.cli._build_index", return_value=fake):
        main()
    assert "5 chunks" in capsys.readouterr().out
    fake.index_repo.assert_called_once()


def test_reindex_forces(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    fake = MagicMock()
    fake.index_repo.return_value = IndexOutcome(repo_id="r1", chunk_count=3, indexed=True)
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.setattr(sys, "argv", ["semble", "reindex", str(repo)])
    with patch("semble.cli._build_index", return_value=fake):
        main()
    _, kwargs = fake.index_repo.call_args
    assert kwargs["force"] is True


def test_status_command(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    fake = MagicMock()
    fake.status.return_value = {
        "repos": 2, "chunks": 100, "embedding_model": "m", "embedding_dim": 768,
        "milvus": "http://x", "core_dir": "/c", "sources": ["/a", "/b"],
    }
    monkeypatch.setattr(sys, "argv", ["semble", "status"])
    with patch("semble.cli._query_backend", return_value=fake):
        main()
    out = capsys.readouterr().out
    assert "Repos indexed : 2" in out
    assert "/a" in out and "/b" in out


def test_projects_add_list_remove(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    registry = tmp_path / "projects.json"
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.setenv("SEMBLE_PROJECTS_FILE", str(registry))

    monkeypatch.setattr(sys, "argv", ["semble", "projects", "add", str(repo), "--name", "api"])
    main()
    assert "Added project: api" in capsys.readouterr().out

    monkeypatch.setattr(sys, "argv", ["semble", "projects", "list"])
    main()
    assert "api" in capsys.readouterr().out

    monkeypatch.setattr(sys, "argv", ["semble", "projects", "remove", str(repo)])
    main()
    assert "Removed project" in capsys.readouterr().out
    assert load_projects().projects == []


def test_projects_scan_registers_git_repos(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    registry = tmp_path / "projects.json"
    (tmp_path / "repo-a" / ".git").mkdir(parents=True)
    (tmp_path / "nested" / "repo-b" / ".git").mkdir(parents=True)
    monkeypatch.setenv("SEMBLE_PROJECTS_FILE", str(registry))
    monkeypatch.setattr(sys, "argv", ["semble", "projects", "scan", str(tmp_path), "--depth", "2"])
    main()
    assert "Registered 2 projects" in capsys.readouterr().out
    sources = {entry.source for entry in load_projects().projects}
    assert sources == {str((tmp_path / "repo-a").resolve()), str((tmp_path / "nested" / "repo-b").resolve())}
