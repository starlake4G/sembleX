<h2 align="center">
  SembleX — Cross-Repo Code Search for AI Agents<br/>
  <sub>Index many repositories once, then search and find similar code across all of them</sub>
</h2>

<div align="center">

[What it does](#what-it-does) •
[Architecture](#architecture) •
[Prerequisites](#prerequisites) •
[Install](#install) •
[Configuration](#configuration) •
[Quickstart](#quickstart) •
[MCP](#mcp-integration) •
[CLI](#cli-reference) •
[How it works](#how-it-works) •
[中文](README_CN.md)

</div>

SembleX is an aggressive fork of [Semble](https://github.com/MinishLab/semble) rebuilt for one job: **give an AI agent a single, global index across a large number of code repositories.** The agent can then (1) search every repo at once with a natural-language or symbol query, and (2) take any code location and find similar implementations in *other* repos to use as reference.

It is **MCP-first**: the MCP server is the primary interface and holds a warm in-process index; the CLI is a thin client for indexing, debugging, and scripting. Dense vectors live in a remote **Milvus** collection; chunk content and the lexical index live in a compact **local core database**. This split keeps storage small (only `id + namespace + vector` go to Milvus) and indexing fast (embeddings are produced by a remote, code-specialized endpoint).

## What it does

Two capabilities, one engine:

- **Cross-repo search** — `search("how is authentication handled")` returns the most relevant code chunks from *any* indexed repo, each attributed to its source repo (`repo :: file:line`).
- **Cross-repo "find similar"** — `find_related(repo, file, line)` embeds the code at that location and returns semantically similar implementations from *other* repos (the source repo is excluded by default), so the agent can learn from how the same problem was solved elsewhere.

Both reduce to the same pipeline: *produce a query vector → retrieve globally → fuse → rank*. A natural-language query and a code location are just two ways to produce that vector.

## Architecture

```
        ┌──────────── warm in-process index (MCP daemon) ────────────┐
  MCP ─▶│  GlobalIndex                                                │
(core)  │   • MetadataStore  — SQLite: chunk content + repo_id        │─▶ Remote Milvus
        │   • global BM25    — one IncrementalSparseIndex (persisted) │     (one collection,
  CLI ─▶│   • rules ranking  — code-aware boosts/penalties            │      namespace = repo_id,
(debug) │                                                             │      stores id + ns + vector only)
        └─────────────────────────────────────────────────────────────┘─▶ Remote OpenAI-compatible
                                                                            embedding endpoint
```

- **One global Milvus collection** holds every repo's dense vectors, partitioned by `namespace = repo_id`. Global search drops the namespace filter; single-repo search and `find_related`'s source-repo exclusion use it.
- **One global BM25 index** (lexical) covers all chunks, keyed by a globally-unique `chunk_id` (the hash includes `repo_id`). Re-indexing a repo removes its old chunk ids and adds the new ones incrementally.
- **Local core database** (default `~/.semble/core`) holds `metadata.sqlite3` (chunk content + repo metadata) and `sparse/` (the persisted BM25 index). The MCP process keeps these warm; CLI invocations reload the BM25 index when its file mtime changes.

## Prerequisites

Unlike upstream Semble, SembleX is **not** zero-setup — it relies on two services you point it at:

1. **A Milvus instance** (local or remote). For a quick local one:
   ```bash
   docker run -d --name milvus -p 19530:19530 milvusdb/milvus:standalone
   ```
2. **An OpenAI-compatible embedding endpoint**, ideally a self-hosted / code-specialized model. Any server exposing `POST /v1/embeddings` works (vLLM, TEI, Ollama, OpenAI itself, …). The embedding **dimension is configurable** — set it to match your model.

## Install

From a built wheel (recommended for deployment — installs standalone, no dependency on the source tree):

```bash
# build once
uv build --wheel                # produces dist/semblex-<version>-py3-none-any.whl

# install into your environment (conda base, venv, etc.) with all extras
pip install "dist/semblex-<version>-py3-none-any.whl[all]"
```

Or directly from source:

```bash
pip install ".[all]"            # or: uv tool install ".[all]"
```

Extras: `mcp` (MCP server), `yaml` (YAML config), `tokenizer` (tiktoken token counting), `all` (= all three). The console command is `semble`.

## Configuration

Create `semble.yaml` (YAML or JSON; env vars like `${VAR:-default}` are substituted). The quickest path is:

```bash
semble init
```

All keys have defaults; the example below shows the ones you'll usually set:

```yaml
embedding:
  backend: openai_compat
  openai_base_url: http://your-embed-host:8000/v1   # OpenAI-compatible endpoint
  openai_api_key: ${SEMBLE_EMBED_KEY:-dummy}        # many self-hosted servers ignore this
  openai_model: your-code-embedding-model
  openai_dim: 2048                                  # MUST match your model's output dim
  max_context_tokens: 32768

milvus:
  uri: http://your-milvus-host:19530
  collection: semble_global
  metric_type: IP                                   # vectors are L2-normalized → IP == cosine

core:
  dir: ~/.semble/core                               # local SQLite + BM25 + git clones

reranker:
  backend: rules                                    # lightweight code-aware ranking
```

Config is loaded via the `SEMBLE_CONFIG` env var or auto-discovery (`./semble.yaml`, `~/.config/semble/config.yaml`, `~/.semble/config.yaml`, ...).

## Quickstart

```bash
# 1. Register repos. Either add them one by one…
semble projects add /path/to/repo-a
semble projects add https://github.com/some-org/repo-b
# …or scan a directory tree and register every git repo under it:
semble projects scan /path/to/all/my/repos --depth 3

# 2. Build the global index (embeds + writes to Milvus + builds BM25)
semble index --all

# 3. Search across every indexed repo
semble search "incremental sparse BM25 index add and remove" -k 8

# 4. From a result's repo/file/line, find similar code in OTHER repos
semble find-related /path/to/repo-a src/foo/chunking.py 42 -k 6

# 5. Inspect the index
semble status
```

A single repo can be (re)indexed by passing its path/URL instead of `--all`: `semble index /path/to/repo-a` (no-op if unchanged; use `reindex` or `--force` to rebuild).

## MCP integration

The MCP server is the primary interface. It connects to the shared warm kernel and serves two tools over stdio.

### Tools

| Tool | Description |
|------|-------------|
| `search(query, top_k=5, scope="global", repo=None)` | Search indexed repos. `scope="global"` (default) spans every repo — best for finding reference implementations elsewhere; `scope="workspace"` restricts to the repo containing the server's working directory (the active project). `repo` optionally pins a specific repo by path/URL and overrides `scope`. Each result is attributed to its source repo. |
| `find_related(repo, file_path, line, top_k=5)` | Given a location (use the `repo`/`file_path`/`line` from a prior `search` result), return semantically similar code from *other* repos. The source repo is excluded. |

> **Scope.** `scope="workspace"` resolves the MCP server's launch directory to whichever indexed repo contains it (a subdirectory works too). MCP clients spawn the server in the active project directory, so this maps to "search only the current project". If that directory isn't indexed, the tool says so and suggests `scope="global"`.

### Register with Claude Code

```bash
claude mcp add semblex -s user -- semble serve
```

<details>
<summary>Other agents (Cursor, Codex, VS Code, …)</summary>

Use the same command in each harness's MCP config. For example, Cursor (`~/.cursor/mcp.json`):

```json
{
  "mcpServers": {
    "semblex": {
      "command": "semble",
      "args": ["serve"]
    }
  }
}
```

The pattern is identical for Codex (`~/.codex/config.toml`), VS Code (`.vscode/mcp.json`), Windsurf, Gemini CLI, etc. — set `command` to `semble` (or an absolute path to it) and `args` to `["serve"]`. If the config is not auto-discoverable, set `SEMBLE_CONFIG` in the MCP client's environment.

</details>

> Indexing is done out-of-band via the CLI (`semble index ...`). The MCP server reads the index that the CLI builds; it reloads the BM25 portion automatically when the on-disk index changes.

## CLI reference

`semble <command>`

| Command | Purpose |
|---------|---------|
| `init [--path PATH] [--force]` | Create a local `semble.yaml` configuration template. |
| `serve` | Run the cross-repo MCP server over stdio (default when no subcommand is given). |
| `index [source] [--all] [--ref REF] [--include-text-files] [--force]` | Index a repo (local path or git URL) into the global index, or `--all` registered projects. `--ref` selects a branch/tag for git URLs; `--include-text-files` also indexes non-code text; `--force` rebuilds. |
| `reindex [source] [--all] ...` | Alias for `index --force`. |
| `search <query> [--scope global\|workspace] [--repo REPO] [-k N]` | Search indexed repos. `--scope workspace` restricts to the repo containing the current directory; `--repo` pins a specific repo by path/URL (overrides `--scope`). |
| `find-related <repo> <file_path> <line> [-k N] [--include-source-repo]` | Find code similar to a location. `repo` is the source repo path/URL from a search result's `repo:` line. By default the source repo is excluded; `--include-source-repo` keeps it. |
| `status` | Show repo count, chunk count, embedding model/dim, Milvus URI, and registered sources. |
| `projects add <source> [--name N] [--ref REF] [--include-text-files]` | Register a local path or git URL. |
| `projects scan <root> [--depth N]` | Register every git repo found under `root` (default depth 3). |
| `projects list` / `projects remove <source>` | List / unregister projects. |

## How it works

SembleX splits each file into code-aware chunks with [tree-sitter](https://github.com/tree-sitter/py-tree-sitter), then retrieves with two complementary signals fused by Reciprocal Rank Fusion (RRF):

- **Dense** — embeddings from your OpenAI-compatible endpoint, L2-normalized and stored in Milvus (cosine via inner product).
- **Lexical** — a global BM25 index over identifiers and API names.

After fusion, results are reranked with code-aware signals (adaptive symbol/NL weighting, definition boosts, identifier-stem matching, file coherence, and noise penalties for test/legacy/example/stub code). In cross-repo mode these ranking signals are computed **per repo and then merged**, so file-coherence stays repo-scoped and identically-named files in different repos never collide.

### Storage footprint

The split-storage design is what keeps SembleX small. Milvus stores **only** `chunk_id + namespace + vector`; chunk *content* lives in the local SQLite database. As a reference point, indexing two repos (~600 chunks at dim 2048) produced ≈ **3.9 MB** local core (1.1 MB SQLite content + 2.9 MB BM25) and ≈ **4.9 MB** of raw vectors in Milvus. You can shrink it further by lowering `embedding.openai_dim`.

## License

MIT. SembleX is a fork of [Semble](https://github.com/MinishLab/semble) by the MinishLab team; the original retrieval and ranking work is theirs.
