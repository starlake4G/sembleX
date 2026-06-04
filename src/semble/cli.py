import argparse
import asyncio
import logging
import sys
import time
from importlib.util import find_spec
from pathlib import Path

from semble.config import SembleConfig, load_config
from semble.projects import ProjectEntry, add_project, load_projects, remove_project
from semble.utils import _format_results, _is_git_url

logger = logging.getLogger(__name__)

# Timestamped, thread-tagged log lines. The thread name matters now that indexing runs
# a producer/embed-workers/writer pipeline — it lets you see embed requests overlapping
# parsing and writes, and read off real per-stage throughput from the timestamps.
_LOG_FORMAT = "%(asctime)s %(levelname)s [%(threadName)s] %(message)s"
_LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format=_LOG_FORMAT,
        datefmt=_LOG_DATEFMT,
        stream=sys.stderr,
    )


def main() -> None:
    """Run the Semble command-line entrypoint. With no subcommand, runs the MCP server."""
    parser = _build_parser()
    args = parser.parse_args()

    if getattr(args, "no_kernel", False):
        import os

        os.environ["SEMBLE_NO_KERNEL"] = "1"

    if args.command in (None, "serve"):
        _run_serve(args)
        return

    _configure_logging()

    if args.command == "projects":
        _run_projects(args)
        return

    cfg = load_config(args.config)
    if args.command == "kernel":
        _run_kernel(args, cfg)
    elif args.command in ("index", "reindex"):
        _run_index(args, cfg)
    elif args.command == "search":
        _run_search(args, cfg)
    elif args.command == "find-related":
        _run_find_related(args, cfg)
    elif args.command == "status":
        _run_status(cfg)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="semble",
        description="Cross-repo code search for agents (MCP-first). Run with no subcommand to start the MCP server.",
    )
    parser.add_argument("--config", default=None, help="Path to semble config file (YAML or JSON).")
    parser.add_argument(
        "--no-kernel",
        action="store_true",
        help="Bypass the shared warm-index kernel; cold-load in-process (slower, for debugging/CI).",
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("serve", help="Run the cross-repo MCP server (default when no subcommand is given).")

    kernel_p = sub.add_parser("kernel", help="Manage the shared warm-index daemon.")
    kernel_p.add_argument(
        "kernel_command",
        nargs="?",
        default="status",
        choices=("start", "stop", "status", "restart", "run"),
        help="start | stop | status | restart (default: status). 'run' stays in the foreground.",
    )

    for name, help_text in (
        ("index", "Index a repo into the global index."),
        ("reindex", "Force re-index a repo (alias for index --force)."),
    ):
        idx = sub.add_parser(name, help=help_text)
        idx.add_argument("source", nargs="?", help="Local path or https/http git URL.")
        idx.add_argument("--all", action="store_true", help="Index all registered projects.")
        idx.add_argument("--ref", default=None, help="Branch or tag (git URLs only).")
        idx.add_argument("--include-text-files", action="store_true", help="Also index non-code text files.")
        if name == "index":
            idx.add_argument("--force", action="store_true", help="Rebuild even if already indexed.")

    s = sub.add_parser("search", help="Search across all indexed repos.")
    s.add_argument("query", help="Natural language or code/symbol query.")
    s.add_argument(
        "--scope",
        choices=("global", "workspace"),
        default="global",
        help="'global' (default) searches every repo; 'workspace' restricts to the repo containing the "
        "current directory. Ignored when --repo is given.",
    )
    s.add_argument("--repo", default=None, help="Restrict to a specific indexed repo by path/URL.")
    s.add_argument("-k", "--top-k", type=int, default=5, help="Number of results (default: 5).")

    fr = sub.add_parser("find-related", help="Find similar code across repos for a location.")
    fr.add_argument("repo", help="Source repo path/URL of the location (from a search result's repo: line).")
    fr.add_argument("file_path", help="File path as shown in search results.")
    fr.add_argument("line", type=int, help="Line number (1-indexed).")
    fr.add_argument("-k", "--top-k", type=int, default=5, help="Number of results (default: 5).")
    fr.add_argument("--include-source-repo", action="store_true", help="Do not exclude the source repo.")

    sub.add_parser("status", help="Show global index status.")

    projects_p = sub.add_parser("projects", help="Manage the repo registry used by `index --all`.")
    projects_sub = projects_p.add_subparsers(dest="projects_command")
    add_p = projects_sub.add_parser("add", help="Register a local path or https/http git URL.")
    add_p.add_argument("source")
    add_p.add_argument("--name", default=None)
    add_p.add_argument("--ref", default=None)
    add_p.add_argument("--include-text-files", action="store_true")
    rm_p = projects_sub.add_parser("remove", help="Unregister a project.")
    rm_p.add_argument("source")
    projects_sub.add_parser("list", help="List registered projects.")
    scan_p = projects_sub.add_parser("scan", help="Register all git repositories under a directory.")
    scan_p.add_argument("root")
    scan_p.add_argument("--depth", type=int, default=3, help="Maximum recursion depth (default: 3).")

    return parser


def _run_serve(args: argparse.Namespace) -> None:
    if find_spec("mcp") is None:
        print("MCP dependencies are not installed. Run: pip install 'semblex[mcp]'", file=sys.stderr)
        raise SystemExit(1)
    from semble.mcp import serve

    _configure_logging()
    asyncio.run(serve(load_config(args.config)))


def _build_index(cfg: SembleConfig):  # noqa: ANN202 — cold in-process GlobalIndex (for indexing).
    from semble.server.indexer import GlobalIndex

    return GlobalIndex(cfg)


def _query_backend(cfg: SembleConfig):  # noqa: ANN202 — kernel client or cold GlobalIndex.
    """Backend for read-only queries: a shared warm kernel, or a cold fallback."""
    from semble.server.kernel import connect_or_spawn

    return connect_or_spawn(cfg)


def _run_kernel(args: argparse.Namespace, cfg: SembleConfig) -> None:
    from semble.server import kernel

    command = args.kernel_command
    if command == "run":
        kernel.run_kernel(cfg)
        return
    if command == "stop":
        print("Kernel stopped." if kernel.stop_kernel(cfg) else "No running kernel.")
        return
    if command == "restart":
        kernel.stop_kernel(cfg)
    if command in ("start", "restart"):
        backend = kernel.connect_or_spawn(cfg)
        if isinstance(backend, kernel.KernelClient):
            pong = backend.ping() or {}
            info = kernel.read_kernel_info(Path(cfg.core.dir).expanduser())
            where = f" on {info.host}:{info.port}" if info else ""
            print(f"Kernel running{where} (pid {pong.get('pid')}, v{pong.get('version')}).")
        else:
            print("Kernel unavailable; would cold-load in-process.", file=sys.stderr)
            sys.exit(1)
        return
    # status
    info = kernel.kernel_status(cfg)
    if info is None:
        print("No running kernel.")
    else:
        age = max(0, int(time.time() - info.started_at))
        print(f"Kernel running on {info.host}:{info.port} (pid {info.pid}, v{info.version}, up {age}s).")


def _run_index(args: argparse.Namespace, cfg: SembleConfig) -> None:
    force = args.command == "reindex" or getattr(args, "force", False)
    index = _build_index(cfg)

    if args.all:
        entries = load_projects().projects
        if not entries:
            print("No projects registered. Use `semble projects add <path>` first.", file=sys.stderr)
            sys.exit(1)
    elif args.source:
        entries = [ProjectEntry(source=args.source, ref=args.ref, include_text_files=args.include_text_files)]
    else:
        print("Provide a <source> or use --all.", file=sys.stderr)
        sys.exit(1)

    # Saving the global BM25 index is O(total chunks); doing it per repo dominates the
    # time between repos. For multi-repo runs, defer the save and flush periodically.
    flush_every = 25
    defer = len(entries) > 1
    failures = 0
    indexed_since_flush = 0
    for entry in entries:
        try:
            outcome = index.index_repo(
                entry.source,
                include_text_files=entry.include_text_files or args.include_text_files,
                force=force,
                ref=entry.ref or args.ref,
                persist=not defer,
            )
            state = "indexed" if outcome.indexed else "already indexed (use --force)"
            print(f"{entry.display_name}: {state}, {outcome.chunk_count} chunks")
            if defer and outcome.indexed:
                indexed_since_flush += 1
                if indexed_since_flush >= flush_every:
                    index.flush()
                    indexed_since_flush = 0
        except Exception as exc:  # noqa: BLE001 — report per-repo and continue.
            failures += 1
            print(f"{entry.display_name}: FAILED — {exc}", file=sys.stderr)
    if defer:
        index.flush()
    if failures:
        sys.exit(1)


def _run_search(args: argparse.Namespace, cfg: SembleConfig) -> None:
    from semble.server.metadata import RepoNotIndexedError

    index = _query_backend(cfg)
    target = args.repo
    if target is None and args.scope == "workspace":
        target = str(Path.cwd())
    try:
        results = index.search(args.query, top_k=args.top_k, repo=target)
    except RepoNotIndexedError as exc:
        print(f"{exc}. Index it first with `semble index {target}`, or use --scope global.", file=sys.stderr)
        sys.exit(1)
    if not results:
        print("No results found.")
        return
    scope = f" in {target}" if target else ""
    print(_format_results(f"Cross-repo search results{scope} for: {args.query!r}", results))


def _run_find_related(args: argparse.Namespace, cfg: SembleConfig) -> None:
    from semble.server.metadata import RepoNotIndexedError

    index = _query_backend(cfg)
    try:
        results = index.find_related(
            args.file_path,
            args.line,
            repo=args.repo,
            top_k=args.top_k,
            exclude_source_repo=not args.include_source_repo,
        )
    except RepoNotIndexedError as exc:
        print(f"{exc}. Index it first with `semble index {args.repo}`.", file=sys.stderr)
        sys.exit(1)
    if not results:
        print(f"No related code found for {args.file_path}:{args.line}.")
        return
    print(_format_results(f"Code related to {args.file_path}:{args.line}", results))


def _run_status(cfg: SembleConfig) -> None:
    index = _query_backend(cfg)
    status = index.status()
    print(f"Repos indexed : {status['repos']}")
    print(f"Chunks        : {status['chunks']}")
    print(f"Embedding     : {status['embedding_model']} (dim={status['embedding_dim']})")
    print(f"Milvus        : {status['milvus']}")
    print(f"Core dir      : {status['core_dir']}")
    sources = status["sources"]
    if isinstance(sources, list) and sources:
        print("Sources:")
        for source in sources:
            print(f"  {source}")


def _run_projects(args: argparse.Namespace) -> None:
    if args.projects_command == "add":
        _projects_add(args.source, name=args.name, ref=args.ref, include_text_files=args.include_text_files)
    elif args.projects_command == "remove":
        if remove_project(args.source):
            print(f"Removed project: {args.source}")
        else:
            print(f"Project not found: {args.source}", file=sys.stderr)
            sys.exit(1)
    elif args.projects_command == "list":
        _projects_list()
    elif args.projects_command == "scan":
        _projects_scan(args.root, depth=args.depth)
    else:
        print("Missing projects subcommand: add, remove, list, or scan", file=sys.stderr)
        sys.exit(1)


def _projects_add(source: str, *, name: str | None, ref: str | None, include_text_files: bool) -> None:
    if _is_git_url(source):
        if not source.startswith(("https://", "http://")):
            print("Only https://, http://, or local directory paths are accepted.", file=sys.stderr)
            sys.exit(1)
    elif not Path(source).expanduser().is_dir():
        print(f"Project path does not exist or is not a directory: {source}", file=sys.stderr)
        sys.exit(1)
    entry = add_project(source, name=name, ref=ref, include_text_files=include_text_files)
    print(f"Added project: {entry.display_name} ({entry.source})")


def _projects_list() -> None:
    config = load_projects()
    if not config.projects:
        print("No projects registered. Use `semble projects add <path>` first.")
        return
    print(f"Projects ({len(config.projects)}):")
    for entry in config.projects:
        ref = f" @ {entry.ref}" if entry.ref else ""
        text = " + text" if entry.include_text_files else ""
        print(f"  {entry.display_name}: {entry.source}{ref}{text}")


def _projects_scan(root_arg: str, *, depth: int) -> None:
    root = Path(root_arg).expanduser().resolve()
    if not root.is_dir():
        print(f"Scan root does not exist or is not a directory: {root_arg}", file=sys.stderr)
        sys.exit(1)
    repos = _scan_git_repos(root, max_depth=depth)
    if not repos:
        print("No git repositories found.")
        return
    for repo in repos:
        entry = add_project(str(repo))
        print(f"Added project: {entry.display_name} ({entry.source})")
    print(f"Registered {len(repos)} projects.")


def _scan_git_repos(root: Path, *, max_depth: int) -> list[Path]:
    """Return git repositories under *root* up to *max_depth* directories deep."""
    repos: list[Path] = []
    skip_dirs = {".git", "node_modules", "__pycache__", "venv", ".venv", "dist", "build"}

    def _walk(directory: Path, depth: int) -> None:
        if depth > max_depth:
            return
        if (directory / ".git").is_dir():
            repos.append(directory)
            return
        try:
            children = sorted(directory.iterdir())
        except OSError:
            return
        for child in children:
            if not child.is_dir() or child.name in skip_dirs or child.name.startswith("."):
                continue
            _walk(child, depth + 1)

    _walk(root, 0)
    return repos
