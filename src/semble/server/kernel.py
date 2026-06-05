"""Shared warm-index kernel: a single localhost daemon both CLI and MCP talk to.

Loading the global BM25 index costs ~tens of seconds, so a cold CLI call paid it
every time. The kernel holds one warm :class:`GlobalIndex` and serves search /
find_related / status over a localhost TCP socket using line-delimited JSON.
Clients (:class:`KernelClient`) connect to it, so only the first call after a
(re)start pays the warmup; subsequent searches are milliseconds. The kernel
auto-stops after an idle period and is auto-spawned on demand.

Indexing stays a separate cold process (it's rare and long); it bumps the BM25
file mtime, which the kernel's ``GlobalIndex._maybe_reload`` picks up on the next
request, so a running kernel never serves stale results after a reindex.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import os
import secrets
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator

from semble.config import SembleConfig
from semble.server.metadata import RepoNotIndexedError
from semble.types import Chunk, SearchResult
from semble.version import __version__

if TYPE_CHECKING:
    from semble.server.indexer import GlobalIndex

logger = logging.getLogger(__name__)

_KERNEL_FILE = "kernel.json"
_KERNEL_LOG = "kernel.log"
_SPAWN_LOCK = "kernel.lock"
# Newline-delimited JSON; one request and one response per connection.
_ENCODING = "utf-8"


# --------------------------------------------------------------------------- state file


def _config_fingerprint(config: SembleConfig) -> str:
    raw = json.dumps(config.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode(_ENCODING)).hexdigest()


@dataclass(frozen=True)
class KernelInfo:
    """Connection details for a running kernel, persisted to the core dir."""

    host: str
    port: int
    token: str
    pid: int
    version: str
    started_at: float
    config_fingerprint: str = ""


def _kernel_file(core_dir: Path) -> Path:
    return core_dir / _KERNEL_FILE


def read_kernel_info(core_dir: Path) -> KernelInfo | None:
    """Return the recorded kernel connection info, or None if absent/corrupt."""
    path = _kernel_file(core_dir)
    try:
        raw = json.loads(path.read_text(encoding=_ENCODING))
        return KernelInfo(
            host=raw["host"],
            port=int(raw["port"]),
            token=raw["token"],
            pid=int(raw["pid"]),
            version=raw["version"],
            started_at=float(raw["started_at"]),
            config_fingerprint=str(raw.get("config_fingerprint", "")),
        )
    except (OSError, ValueError, KeyError):
        return None


def _write_kernel_info(core_dir: Path, info: KernelInfo) -> None:
    tmp = _kernel_file(core_dir).with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps(
            {
                "host": info.host,
                "port": info.port,
                "token": info.token,
                "pid": info.pid,
                "version": info.version,
                "started_at": info.started_at,
                "config_fingerprint": info.config_fingerprint,
            }
        ),
        encoding=_ENCODING,
    )
    tmp.replace(_kernel_file(core_dir))


def _remove_kernel_info(core_dir: Path) -> None:
    try:
        _kernel_file(core_dir).unlink()
    except OSError:
        pass


@contextlib.contextmanager
def _spawn_lock(core_dir: Path, *, timeout: float) -> Iterator[bool]:
    path = core_dir / _SPAWN_LOCK
    deadline = time.monotonic() + timeout
    stale_after = max(timeout, 30.0)
    while True:
        try:
            fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            with contextlib.suppress(OSError):
                if time.time() - path.stat().st_mtime > stale_after:
                    path.unlink()
                    continue
            if time.monotonic() >= deadline:
                yield False
                return
            time.sleep(0.2)
            continue
        try:
            with os.fdopen(fd, "w", encoding=_ENCODING) as f:
                f.write(str(os.getpid()))
            yield True
        finally:
            with contextlib.suppress(OSError):
                path.unlink()
        return


# --------------------------------------------------------------------------- (de)serialize


def _chunk_to_dict(chunk: Chunk) -> dict[str, Any]:
    return {
        "content": chunk.content,
        "file_path": chunk.file_path,
        "start_line": chunk.start_line,
        "end_line": chunk.end_line,
        "language": chunk.language,
    }


def _result_to_dict(result: SearchResult) -> dict[str, Any]:
    return {
        "chunk": _chunk_to_dict(result.chunk),
        "score": result.score,
        "repo_id": result.repo_id,
        "repo_name": result.repo_name,
        "repo_source": result.repo_source,
    }


def _result_from_dict(raw: dict[str, Any]) -> SearchResult:
    c = raw["chunk"]
    return SearchResult(
        chunk=Chunk(
            content=c["content"],
            file_path=c["file_path"],
            start_line=c["start_line"],
            end_line=c["end_line"],
            language=c.get("language"),
        ),
        score=raw["score"],
        repo_id=raw.get("repo_id"),
        repo_name=raw.get("repo_name"),
        repo_source=raw.get("repo_source"),
    )


# --------------------------------------------------------------------------- server


class _KernelServer:
    """Owns one warm GlobalIndex and serves it over a localhost socket."""

    def __init__(self, config: SembleConfig, index: GlobalIndex) -> None:
        self._config = config
        self._index = index
        self._core_dir = Path(config.core.dir).expanduser()
        self._token = secrets.token_hex(16)
        self._idle_timeout = config.kernel.idle_timeout
        self._last_active = time.monotonic()
        self._inflight = 0
        self._stop = asyncio.Event()

    async def serve(self) -> None:
        server = await asyncio.start_server(
            self._handle, self._config.kernel.host, self._config.kernel.port
        )
        sock = server.sockets[0]
        port = sock.getsockname()[1]
        info = KernelInfo(
            host=self._config.kernel.host,
            port=port,
            token=self._token,
            pid=os.getpid(),
            version=__version__,
            started_at=time.time(),
            config_fingerprint=_config_fingerprint(self._config),
        )
        _write_kernel_info(self._core_dir, info)
        logger.info("Kernel ready on %s:%d (pid %d, v%s)", info.host, port, info.pid, __version__)
        watcher = asyncio.create_task(self._idle_watcher())
        try:
            async with server:
                await self._stop.wait()
        finally:
            watcher.cancel()
            server.close()
            _remove_kernel_info(self._core_dir)
            logger.info("Kernel stopped")

    async def _idle_watcher(self) -> None:
        while not self._stop.is_set():
            await asyncio.sleep(min(30.0, max(1.0, self._idle_timeout / 4)))
            if self._inflight == 0 and time.monotonic() - self._last_active > self._idle_timeout:
                logger.info("Kernel idle for >%.0fs; shutting down", self._idle_timeout)
                self._stop.set()
                return

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            line = await reader.readline()
            if not line:
                return
            request = json.loads(line.decode(_ENCODING))
            response = await self._dispatch(request)
        except Exception as exc:  # noqa: BLE001 — never let one client crash the kernel.
            response = {"ok": False, "error": str(exc), "error_type": "InternalError"}
        try:
            writer.write((json.dumps(response) + "\n").encode(_ENCODING))
            await writer.drain()
        except (OSError, ConnectionError):
            pass
        finally:
            writer.close()

    async def _dispatch(self, request: dict[str, Any]) -> dict[str, Any]:
        if request.get("token") != self._token:
            return {"ok": False, "error": "bad token", "error_type": "Unauthorized"}
        method = request.get("method")
        params = request.get("params") or {}

        if method == "ping":
            return {"ok": True, "result": {"version": __version__, "pid": os.getpid()}}
        if method == "shutdown":
            self._stop.set()
            return {"ok": True, "result": None}

        self._last_active = time.monotonic()
        self._inflight += 1
        try:
            return await self._run_method(method, params)
        except RepoNotIndexedError as exc:
            return {"ok": False, "error": str(exc), "error_type": "RepoNotIndexedError"}
        except Exception as exc:  # noqa: BLE001 — surface as an error response.
            logger.warning("Kernel method %s failed", method, exc_info=True)
            return {"ok": False, "error": str(exc), "error_type": "InternalError"}
        finally:
            self._inflight -= 1
            self._last_active = time.monotonic()

    async def _run_method(self, method: str | None, params: dict[str, Any]) -> dict[str, Any]:
        if method == "search":
            results = await asyncio.to_thread(
                self._index.search,
                params["query"],
                top_k=params.get("top_k", 5),
                repo=params.get("repo"),
            )
            return {"ok": True, "result": [_result_to_dict(r) for r in results]}
        if method == "find_related":
            results = await asyncio.to_thread(
                self._index.find_related,
                params["file_path"],
                params["line"],
                repo=params["repo"],
                top_k=params.get("top_k", 5),
                exclude_source_repo=params.get("exclude_source_repo", True),
            )
            return {"ok": True, "result": [_result_to_dict(r) for r in results]}
        if method == "status":
            status = await asyncio.to_thread(self._index.status)
            return {"ok": True, "result": status}
        return {"ok": False, "error": f"unknown method: {method}", "error_type": "BadRequest"}


# --------------------------------------------------------------------------- client


class KernelClient:
    """Thin synchronous client; proxies GlobalIndex's interface over the socket."""

    def __init__(self, info: KernelInfo, *, timeout: float = 120.0) -> None:
        """Bind the client to a kernel's connection *info*."""
        self._info = info
        self._timeout = timeout

    def _request(self, method: str, params: dict[str, Any] | None = None) -> Any:
        payload = json.dumps(
            {"method": method, "params": params or {}, "token": self._info.token}
        )
        with socket.create_connection((self._info.host, self._info.port), timeout=self._timeout) as sock:
            sock.sendall((payload + "\n").encode(_ENCODING))
            sock.settimeout(self._timeout)
            buf = bytearray()
            while not buf.endswith(b"\n"):
                part = sock.recv(65536)
                if not part:
                    break
                buf.extend(part)
        response = json.loads(bytes(buf).decode(_ENCODING))
        if not response.get("ok"):
            message = response.get("error", "kernel error")
            if response.get("error_type") == "RepoNotIndexedError":
                exc = RepoNotIndexedError("")
                exc.args = (message,)  # preserve the server's already-formatted message
                raise exc
            raise RuntimeError(message)
        return response.get("result")

    def ping(self) -> dict[str, Any] | None:
        """Return the kernel's ``{version, pid}`` if reachable, else None."""
        try:
            return self._request("ping")
        except (OSError, ConnectionError, RuntimeError, ValueError):
            return None

    def search(self, query: str, *, top_k: int = 5, repo: str | None = None) -> list[SearchResult]:
        """Proxy :meth:`GlobalIndex.search` to the kernel."""
        raw = self._request("search", {"query": query, "top_k": top_k, "repo": repo})
        return [_result_from_dict(r) for r in raw]

    def find_related(
        self,
        file_path: str,
        line: int,
        *,
        repo: str,
        top_k: int = 5,
        exclude_source_repo: bool = True,
    ) -> list[SearchResult]:
        """Proxy :meth:`GlobalIndex.find_related` to the kernel."""
        raw = self._request(
            "find_related",
            {
                "file_path": file_path,
                "line": line,
                "repo": repo,
                "top_k": top_k,
                "exclude_source_repo": exclude_source_repo,
            },
        )
        return [_result_from_dict(r) for r in raw]

    def status(self) -> dict[str, Any]:
        """Proxy :meth:`GlobalIndex.status` to the kernel."""
        return self._request("status")

    def shutdown(self) -> bool:
        """Ask the kernel to exit; return True if the request was delivered."""
        try:
            self._request("shutdown")
            return True
        except (OSError, ConnectionError, RuntimeError, ValueError):
            return False


# --------------------------------------------------------------------------- discovery / spawn


def _healthy_client(core_dir: Path, config_fingerprint: str | None = None) -> KernelClient | None:
    """Return a client for a running, version-matching kernel, or None."""
    info = read_kernel_info(core_dir)
    if info is None:
        return None
    client = KernelClient(info)
    pong = client.ping()
    if pong is None:
        return None
    if pong.get("version") != __version__:
        # Stale kernel running older code; ask it to exit so we can respawn.
        logger.info("Kernel v%s != client v%s; restarting", pong.get("version"), __version__)
        client.shutdown()
        return None
    if config_fingerprint is not None and info.config_fingerprint != config_fingerprint:
        logger.info("Kernel config changed; restarting")
        client.shutdown()
        return None
    return client


def _spawn_kernel(core_dir: Path) -> None:
    r"""Launch a detached kernel via ``python -m semble`` (not the .exe shim).

    Using the interpreter directly keeps the Windows ``Scripts\\semble.exe``
    launcher unlocked, so a running kernel never blocks ``pip`` reinstalls.
    """
    core_dir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "semble", "kernel", "run"]
    log = open(core_dir / _KERNEL_LOG, "ab")  # noqa: SIM115 — child inherits the handle during Popen.
    kwargs: dict[str, Any] = {"stdout": log, "stderr": log, "stdin": subprocess.DEVNULL}
    if os.name == "nt":
        # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP: no console, survives parent exit.
        kwargs["creationflags"] = 0x00000008 | 0x00000200
    else:
        kwargs["start_new_session"] = True
    try:
        subprocess.Popen(cmd, **kwargs)  # noqa: S603 — fixed argv, no shell.
    finally:
        log.close()


def connect_or_spawn(config: SembleConfig, *, allow_spawn: bool = True) -> Any:
    """Return a backend exposing ``search``/``find_related``/``status``.

    Prefers a shared kernel: reuses a running one, else (when enabled) spawns a
    detached kernel and waits for it to warm up. Concurrent first-time callers
    are serialized by a spawn lock so only one kernel ever starts; the others
    wait for it to come up. Falls back to an in-process cold :class:`GlobalIndex`
    if the kernel is disabled or unreachable.
    """
    core_dir = Path(config.core.dir).expanduser()
    fingerprint = _config_fingerprint(config)
    use_kernel = config.kernel.enabled and not os.environ.get("SEMBLE_NO_KERNEL")
    if not use_kernel:
        return _cold_index(config)

    client = _healthy_client(core_dir, fingerprint)
    if client is not None:
        return client
    if not (allow_spawn and config.kernel.autostart):
        return _cold_index(config)

    try:
        core_dir.mkdir(parents=True, exist_ok=True)
        lock_timeout = min(max(config.kernel.startup_timeout, 1.0), 30.0)
        deadline = time.monotonic() + config.kernel.startup_timeout
        with _spawn_lock(core_dir, timeout=lock_timeout) as have_lock:
            # Another caller may have spawned while we contended for the lock.
            client = _healthy_client(core_dir, fingerprint)
            if client is not None:
                return client
            if have_lock:
                _spawn_kernel(core_dir)
            while time.monotonic() < deadline:
                time.sleep(0.5)
                client = _healthy_client(core_dir, fingerprint)
                if client is not None:
                    return client
    except OSError:
        logger.warning("Kernel spawn setup failed; using cold load", exc_info=True)
        return _cold_index(config)
    logger.warning("Kernel did not become ready in %.0fs; using cold load", config.kernel.startup_timeout)
    return _cold_index(config)


def _cold_index(config: SembleConfig) -> GlobalIndex:
    from semble.server.indexer import GlobalIndex

    return GlobalIndex(config)


# --------------------------------------------------------------------------- entrypoints


def run_kernel(config: SembleConfig) -> None:
    """Run the kernel daemon (foreground). Exits if a healthy kernel already runs."""
    core_dir = Path(config.core.dir).expanduser()
    core_dir.mkdir(parents=True, exist_ok=True)
    if _healthy_client(core_dir, _config_fingerprint(config)) is not None:
        logger.info("A healthy kernel is already running; exiting")
        return

    from semble.server.indexer import GlobalIndex

    logger.info("Kernel warming up (loading global index)...")
    index = GlobalIndex(config)
    server = _KernelServer(config, index)
    try:
        asyncio.run(server.serve())
    except KeyboardInterrupt:
        _remove_kernel_info(core_dir)


def kernel_status(config: SembleConfig) -> KernelInfo | None:
    """Return info for a running kernel, or None."""
    core_dir = Path(config.core.dir).expanduser()
    client = _healthy_client(core_dir, _config_fingerprint(config))
    if client is None:
        return None
    return read_kernel_info(core_dir)


def stop_kernel(config: SembleConfig) -> bool:
    """Ask a running kernel to shut down. Returns True if one was stopped."""
    core_dir = Path(config.core.dir).expanduser()
    info = read_kernel_info(core_dir)
    if info is None:
        return False
    stopped = KernelClient(info).shutdown()
    if stopped:
        _remove_kernel_info(core_dir)
    return stopped
