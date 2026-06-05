from pathlib import Path

from semble.config import SembleConfig
from semble.server.kernel import (
    KernelInfo,
    _result_from_dict,
    _result_to_dict,
    _write_kernel_info,
    connect_or_spawn,
    read_kernel_info,
)
from semble.types import Chunk, SearchResult


def test_result_roundtrip() -> None:
    """A SearchResult survives serialize -> deserialize unchanged."""
    result = SearchResult(
        chunk=Chunk(content="x = 1\n", file_path="a/b.py", start_line=3, end_line=5, language="python"),
        score=0.42,
        repo_id="rid",
        repo_name="proj",
        repo_source="C:\\repos\\proj",
    )
    back = _result_from_dict(_result_to_dict(result))
    assert back == result


def test_result_roundtrip_minimal() -> None:
    """Optional attribution fields round-trip as None."""
    result = SearchResult(
        chunk=Chunk(content="y", file_path="f", start_line=1, end_line=1, language=None),
        score=0.0,
    )
    assert _result_from_dict(_result_to_dict(result)) == result


def test_kernel_info_read_write(tmp_path: Path) -> None:
    """KernelInfo persists to and loads from the core dir; missing/corrupt -> None."""
    assert read_kernel_info(tmp_path) is None
    info = KernelInfo(host="127.0.0.1", port=5555, token="abc", pid=123, version="9.9.9", started_at=1.5)
    _write_kernel_info(tmp_path, info)
    loaded = read_kernel_info(tmp_path)
    assert loaded == info

    (tmp_path / "kernel.json").write_text("not json{", encoding="utf-8")
    assert read_kernel_info(tmp_path) is None


def test_connect_or_spawn_disabled_returns_cold(monkeypatch, tmp_path: Path) -> None:
    """With the kernel disabled, connect_or_spawn never spawns; it cold-loads.

    We stub the cold loader so the test doesn't need Milvus.
    """
    sentinel = object()
    monkeypatch.setattr("semble.server.kernel._cold_index", lambda cfg: sentinel)
    spawned = False

    def _fail_spawn(_core: Path) -> None:
        nonlocal spawned
        spawned = True

    monkeypatch.setattr("semble.server.kernel._spawn_kernel", _fail_spawn)

    cfg = SembleConfig()
    cfg.kernel.enabled = False
    cfg.core.dir = tmp_path
    assert connect_or_spawn(cfg) is sentinel
    assert spawned is False


def test_connect_or_spawn_no_autostart_uses_cold(monkeypatch, tmp_path: Path) -> None:
    """When no kernel is running and autostart is off, it falls back to cold without spawning."""
    sentinel = object()
    monkeypatch.setattr("semble.server.kernel._cold_index", lambda cfg: sentinel)
    monkeypatch.setattr("semble.server.kernel._spawn_kernel", lambda core: (_ for _ in ()).throw(AssertionError()))

    cfg = SembleConfig()
    cfg.kernel.autostart = False
    cfg.core.dir = tmp_path
    assert connect_or_spawn(cfg) is sentinel


def test_connect_or_spawn_holds_lock_until_kernel_ready(monkeypatch, tmp_path: Path) -> None:
    """The spawning client keeps the lock while waiting for kernel.json to become healthy."""
    sentinel = object()
    spawned = False
    saw_lock_after_spawn: list[bool] = []

    def _healthy(core: Path, _fingerprint: str | None = None):  # noqa: ANN202
        if spawned:
            saw_lock_after_spawn.append((core / "kernel.lock").exists())
            return sentinel
        return None

    def _spawn(_core: Path) -> None:
        nonlocal spawned
        spawned = True

    monkeypatch.setattr("semble.server.kernel._healthy_client", _healthy)
    monkeypatch.setattr("semble.server.kernel._spawn_kernel", _spawn)
    monkeypatch.setattr("semble.server.kernel.time.sleep", lambda _seconds: None)
    monkeypatch.setattr("semble.server.kernel._cold_index", lambda cfg: (_ for _ in ()).throw(AssertionError()))
    monkeypatch.delenv("SEMBLE_NO_KERNEL", raising=False)

    cfg = SembleConfig()
    cfg.core.dir = tmp_path
    cfg.kernel.startup_timeout = 1

    assert connect_or_spawn(cfg) is sentinel
    assert saw_lock_after_spawn == [True]
