from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


def _expand_path(value: Path | str) -> Path:
    return Path(value).expanduser()


class EmbeddingConfig(BaseModel):
    """Self-hosted / code-specialized embedding via an OpenAI-compatible endpoint."""

    backend: Literal["openai_compat"] = "openai_compat"
    openai_base_url: str = "https://api.openai.com/v1"
    openai_api_key: str | None = None
    openai_model: str = "text-embedding-3-small"
    openai_dim: int = 1536
    batch_size: int = 512
    max_retries: int = 3
    max_concurrent: int = 10
    max_context_tokens: int = 8192
    # Optional HF/modelscope id or local path of the serving model's tokenizer. When set
    # (and `transformers` is installed), token counts match the server exactly so inputs
    # are truncated precisely. Falls back to tiktoken / a heuristic when unset or unloadable.
    tokenizer: str | None = None


class RerankerConfig(BaseModel):
    backend: Literal["rules"] = "rules"
    coarse_multiplier: int = 5


class IndexingConfig(BaseModel):
    """Settings that govern how files become chunks/embeddings."""

    max_file_bytes: int = 1_000_000
    embed_batch_size: int = 512


class CoreDbConfig(BaseModel):
    """Local core database: chunk metadata (SQLite) + global BM25 index + git clones."""

    dir: Path = Field(default_factory=lambda: Path.home() / ".semble" / "core")
    clone_timeout: int = 300

    @field_validator("dir", mode="before")
    @classmethod
    def _expand_dir(cls, value: object) -> object:
        return _expand_path(value) if isinstance(value, (str, Path)) else value


class MilvusConfig(BaseModel):
    uri: str = "http://127.0.0.1:19530"
    token: str | None = None
    db_name: str | None = None
    collection: str = "semble_global"
    vector_field: str = "vector"
    metric_type: Literal["COSINE", "IP", "L2"] = "IP"
    index_type: str = "AUTOINDEX"
    index_params: dict[str, Any] = Field(default_factory=dict)
    search_params: dict[str, Any] = Field(default_factory=dict)
    consistency_level: str = "Bounded"


class SembleConfig(BaseModel):
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    reranker: RerankerConfig = Field(default_factory=RerankerConfig)
    indexing: IndexingConfig = Field(default_factory=IndexingConfig)
    core: CoreDbConfig = Field(default_factory=CoreDbConfig)
    milvus: MilvusConfig = Field(default_factory=MilvusConfig)


_ENV_VAR_RE = re.compile(r"\$\{([^}:]+)(?::-([^}]*))?\}")


def _substitute_env_vars(value: str) -> str:
    def _replacer(match: re.Match[str]) -> str:
        var_name = match.group(1)
        default_value = match.group(2)
        env_value = os.environ.get(var_name)
        if env_value is not None:
            return env_value
        if default_value is not None:
            return default_value
        return match.group(0)

    return _ENV_VAR_RE.sub(_replacer, value)


def _process_config_values(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _process_config_values(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_process_config_values(item) for item in obj]
    if isinstance(obj, str):
        return _substitute_env_vars(obj)
    return obj


def _find_default_config() -> Path | None:
    candidates = [
        Path("semble.yaml"),
        Path("semble.yml"),
        Path("semble.json"),
        Path.home() / ".config" / "semble" / "config.yaml",
        Path.home() / ".config" / "semble" / "config.yml",
        Path.home() / ".config" / "semble" / "config.json",
        Path.home() / ".semble" / "config.yaml",
        Path.home() / ".semble" / "config.yml",
        Path.home() / ".semble" / "config.json",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def load_config(path: str | Path | None = None) -> SembleConfig:
    """Load a Semble configuration from disk or return defaults."""
    if path is None:
        env_path = os.environ.get("SEMBLE_CONFIG")
        if env_path:
            path = env_path
        else:
            found = _find_default_config()
            if found is None:
                return SembleConfig()
            path = found

    path = Path(path)
    if not path.exists():
        return SembleConfig()

    raw: dict[str, Any]
    if path.suffix in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError:
            raise ImportError("PyYAML is required for YAML config. Install with: pip install pyyaml") from None
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
    elif path.suffix == ".json":
        import json

        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
    else:
        raise ValueError(f"Unsupported config file format: {path.suffix}")

    raw = _process_config_values(raw)
    return SembleConfig(**raw)
