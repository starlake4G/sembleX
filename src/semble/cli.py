import argparse
import asyncio
import logging
import sys
from enum import Enum
from importlib.resources import files
from importlib.util import find_spec
from pathlib import Path

from model2vec.utils import get_package_extras

from semble.config import SembleConfig, load_config
from semble.index import SembleIndex
from semble.remote import RemoteSembleClient
from semble.stats import format_savings_report
from semble.utils import _format_results, _is_git_url, _resolve_chunk

logger = logging.getLogger(__name__)


class Agent(str, Enum):
    CLAUDE = "claude"
    COPILOT = "copilot"
    CURSOR = "cursor"
    GEMINI = "gemini"
    KIRO = "kiro"
    OPENCODE = "opencode"


_DEFAULT_AGENT = Agent.CLAUDE
_CLI_DISPATCH_ARGS = frozenset({"search", "find-related", "index", "init", "savings", "server", "-h", "--help"})


def _agent_path(agent: Agent) -> Path:
    base_dir = ".github" if agent is Agent.COPILOT else f".{agent.value}"
    return Path(base_dir) / "agents" / "semble-search.md"


def main() -> None:
    """Run the Semble command-line entrypoint."""
    if len(sys.argv) > 1 and sys.argv[1] in _CLI_DISPATCH_ARGS:
        _cli_main()
    else:
        _mcp_main()


def _mcp_main() -> None:
    parser = argparse.ArgumentParser(
        prog="semble",
        description="Instant local code search for agents.",
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=None,
        help="Local directory or git URL to pre-index at startup (optional).",
    )
    parser.add_argument("--ref", default=None, help="Branch or tag to check out (git URLs only).")
    parser.add_argument(
        "--include-text-files",
        action="store_true",
        help="Also index non-code text files (.md, .yaml, .json, etc.).",
    )
    parser.add_argument("--config", default=None, help="Path to semble config file (YAML or JSON).")
    args = parser.parse_args()
    if any(find_spec(dep) is None for dep in get_package_extras("semble", "mcp")):
        print("MCP dependencies are not installed. Run: pip install 'semble[mcp]'", file=sys.stderr)
        raise SystemExit(1)
    from semble.mcp import serve

    cfg = load_config(args.config)
    asyncio.run(serve(args.path, ref=args.ref, include_text_files=args.include_text_files, config=cfg))


def _run_init(*, agent: Agent = _DEFAULT_AGENT, force: bool = False) -> None:
    dest = _agent_path(agent)
    if dest.exists() and not force:
        print(f"{dest} already exists. Run with --force to overwrite.", file=sys.stderr)
        sys.exit(1)
    dest.parent.mkdir(parents=True, exist_ok=True)
    content = files("semble").joinpath(f"agents/{agent.value}.md").read_text(encoding="utf-8")
    dest.write_text(content, encoding="utf-8")
    print(f"Created {dest}")


def _run_remote_cli(args: argparse.Namespace, cfg: SembleConfig, include_text: bool) -> None:
    client = RemoteSembleClient.from_config(cfg.remote)
    try:
        if args.command == "index":
            print(
                client.index(
                    repo=args.path,
                    include_text_files=include_text,
                    force=args.force,
                    ref=getattr(args, "ref", None),
                )
            )
        elif args.command == "search":
            print(client.search(args.query, repo=args.path, top_k=args.top_k, include_text_files=include_text))
        elif args.command == "find-related":
            print(
                client.find_related(
                    args.file_path,
                    args.line,
                    repo=args.path,
                    top_k=args.top_k,
                    include_text_files=include_text,
                )
            )
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)


def _run_server(args: argparse.Namespace) -> None:
    if find_spec("fastapi") is None or find_spec("uvicorn") is None or find_spec("pymilvus") is None:
        print("Server dependencies are not installed. Run: pip install 'semble[server]'", file=sys.stderr)
        sys.exit(1)

    import uvicorn

    from semble.server import create_app

    cfg = load_config(args.config)
    host = args.host or cfg.server.host
    port = args.port or cfg.server.port
    uvicorn.run(create_app(cfg), host=host, port=port)


def _cli_main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        stream=sys.stderr,
    )
    parser = argparse.ArgumentParser(prog="semble")
    sub = parser.add_subparsers(dest="command")

    search_p = sub.add_parser("search", help="Search a codebase.")
    search_p.add_argument("query", help="Natural language or code query.")
    search_p.add_argument("path", nargs="?", default=".", help="Local path or git URL (default: current directory).")
    search_p.add_argument("-k", "--top-k", type=int, default=5, help="Number of results (default: 5).")
    search_p.add_argument(
        "--include-text-files",
        action="store_true",
        help="Also index non-code text files (.md, .yaml, .json, etc.).",
    )
    search_p.add_argument("--config", default=None, help="Path to semble config file (YAML or JSON).")

    related_p = sub.add_parser("find-related", help="Find code similar to a specific location.")
    related_p.add_argument("file_path", help="File path as shown in search results.")
    related_p.add_argument("line", type=int, help="Line number (1-indexed).")
    related_p.add_argument("path", nargs="?", default=".", help="Local path or git URL (default: current directory).")
    related_p.add_argument("-k", "--top-k", type=int, default=5, help="Number of results (default: 5).")
    related_p.add_argument(
        "--include-text-files",
        action="store_true",
        help="Also index non-code text files (.md, .yaml, .json, etc.).",
    )
    related_p.add_argument("--config", default=None, help="Path to semble config file (YAML or JSON).")

    index_p = sub.add_parser("index", help="Build or refresh a remote index.")
    index_p.add_argument("path", help="Local server path or https/http git URL to index remotely.")
    index_p.add_argument(
        "--include-text-files",
        action="store_true",
        help="Also index non-code text files (.md, .yaml, .json, etc.).",
    )
    index_p.add_argument("--force", action="store_true", help="Rebuild the remote index even if it already exists.")
    index_p.add_argument("--ref", default=None, help="Branch or tag to check out (git URLs only).")
    index_p.add_argument("--config", default=None, help="Path to semble config file (YAML or JSON).")

    init_p = sub.add_parser("init", help="Write a semble sub-agent file for your coding agent.")
    init_p.add_argument(
        "--agent",
        "-a",
        default=_DEFAULT_AGENT.value,
        choices=[a.value for a in Agent],
        help=f"Coding agent to set up (default: {_DEFAULT_AGENT.value}).",
    )
    init_p.add_argument("--force", action="store_true", help="Overwrite if the file already exists.")

    savings_p = sub.add_parser("savings", help="Show token savings and usage stats.")
    savings_p.add_argument("--verbose", action="store_true", help="Also show usage breakdown by call type.")

    server_p = sub.add_parser("server", help="Run the remote Milvus-backed index service.")
    server_p.add_argument("--host", default=None, help="Host to bind (default: config server.host).")
    server_p.add_argument("--port", type=int, default=None, help="Port to bind (default: config server.port).")
    server_p.add_argument("--config", default=None, help="Path to semble config file (YAML or JSON).")

    args = parser.parse_args()

    if args.command == "init":
        _run_init(agent=Agent(args.agent), force=args.force)
        return

    if args.command == "savings":
        print(format_savings_report(verbose=args.verbose), end="")
        return

    if args.command == "server":
        _run_server(args)
        return

    cfg = load_config(getattr(args, "config", None))
    include_text = args.include_text_files

    if args.command == "index" and cfg.index.backend != "remote":
        print("The index command requires index.backend: remote.", file=sys.stderr)
        sys.exit(1)

    if cfg.index.backend == "remote":
        _run_remote_cli(args, cfg, include_text)
        return

    index = (
        SembleIndex.from_git(args.path, include_text_files=include_text, config=cfg)
        if _is_git_url(args.path)
        else SembleIndex.from_path(args.path, include_text_files=include_text, config=cfg)
    )

    if args.command == "search":
        results = index.search(args.query, top_k=args.top_k)
        if not results:
            print("No results found.")
        else:
            print(_format_results(f"Search results for: {args.query!r}", results))

    elif args.command == "find-related":
        chunk = _resolve_chunk(index.chunks, args.file_path, args.line)
        if chunk is None:
            print(f"No chunk found at {args.file_path}:{args.line}.", file=sys.stderr)
            sys.exit(1)
        results = index.find_related(chunk, top_k=args.top_k)
        if not results:
            print(f"No related chunks found for {args.file_path}:{args.line}.")
        else:
            print(_format_results(f"Chunks related to {args.file_path}:{args.line}", results))
