# AGENTS.md — SembleX

SembleX is a cross-repo code-search **MCP** for AI agents: it indexes many repositories into one
global index, then serves `search` (cross-repo) and `find_related` (cross-repo "find similar").
MCP-first; the CLI is a thin client for indexing and debugging. Dense vectors live in a remote
Milvus collection; chunk content + a global BM25 index live in a local core database.

## Commands

All commands use `uv run` (no global install needed):

```
make install          # uv sync --all-extras + pre-commit install
make test             # uv run pytest
make lint             # ruff check + pydoclint on src/
make typecheck        # mypy on src/
make fix              # ruff check --fix + ruff format
make pre-commit       # all hooks
```

Run a single test file: `uv run pytest tests/test_global_index.py`
Run a single test: `uv run pytest tests/test_global_index.py::test_name -k "keyword"`

## Verify before committing

```
make lint && make typecheck && make test
```

Pre-commit enforces the same (ruff, pydoclint, mypy). The `main` branch is protected.

## Project structure

- **Package**: `src/semble/` — installed as `semblex` (dist name) with CLI entry point `semble`
- **CLI**: `src/semble/cli.py` — `semble serve | index | reindex | search | find-related | status | projects`
- **Core engine**: `src/semble/server/indexer.py` — `GlobalIndex` (`from semble.server import GlobalIndex`).
  The single warm object holding Milvus + SQLite `MetadataStore` + one global BM25 index.
- **Metadata store**: `src/semble/server/metadata.py` — SQLite chunk content + repo metadata
- **Search engine**: `src/semble/search.py` — `fuse_hybrid` (RRF) + `rank_candidates` (per-repo grouped ranking)
- **Pluggable backends**: `src/semble/backends/` — `embedding/` (openai_compat), `vector_store/` (milvus), `sparse/` (incremental BM25)
- **Chunking**: `src/semble/chunking/` — tree-sitter-based code chunking
- **Ranking**: `src/semble/ranking/` — code-aware signals (boosting, penalties, weighting); the `rules` reranker
- **MCP server**: `src/semble/mcp.py` — `create_server(GlobalIndex)` exposing `search` + `find_related`
- **Config / registry**: `src/semble/config.py`, `src/semble/projects.py`
- **Tests**: `tests/` — fixtures in `conftest.py`; `tests/test_global_index.py` is the end-to-end test with an in-memory fake Milvus + deterministic embedder

## Code style

- **Formatter/linter**: ruff, 120-char line length, target Python 3.10
- **Type checking**: mypy — all new code must be fully typed (`ignore_missing_imports = true`)
- **Docstrings**: Sphinx style, enforced by pydoclint. Skip `__init__` docstrings.
- **Print statements forbidden** (`T20` rule) except in `cli.py`
- **Tests**: no type-annotation enforcement (`ANN` rule skipped for `tests/`)
- **Imports**: isort-enforced order via ruff

## Testing notes

- pytest config in `pyproject.toml`: `--tb=short --strict-markers --cov=semble --cov-report=term-missing`
- **No external services needed**: the engine tests stub Milvus and the embedder with in-memory fakes
  (`FakeVectorStore` / `FakeEmbedder` in `tests/test_global_index.py`), so the full index→search→find_related
  path is exercised offline. Live runs need a real Milvus + an OpenAI-compatible embedding endpoint.
- Async tests use `anyio` with the `asyncio` backend (see `conftest.py`)
- Tests create temp files/projects via the `tmp_path` pytest fixture

## Build & versioning

- Version comes from `semble.version.__version__` (`src/semble/version.py`)
- Source layout: `src/semble/` (not flat)
- Optional dependency groups: `mcp`, `yaml`, `tokenizer`, `all` (= the first three), `dev`
- Build a wheel with `uv build --wheel`; deploy standalone via `pip install "dist/semblex-<version>-py3-none-any.whl[all]"`

## Runtime dependencies

SembleX is **not** zero-setup. It requires:
- A **Milvus** instance (global dense vector store; `milvus.uri` in config)
- An **OpenAI-compatible embedding endpoint** (`embedding.openai_base_url`); the dimension is configurable (`embedding.openai_dim`)

Indexing is done out-of-band via the CLI; the MCP server reads the index and reloads the BM25 portion when its on-disk mtime changes.
