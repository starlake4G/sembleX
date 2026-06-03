from __future__ import annotations

import json
from pathlib import Path

import pytest

from semble.config import IndexingConfig, SembleConfig, load_config


def test_defaults() -> None:
    cfg = SembleConfig()
    assert cfg.indexing.max_file_bytes == 1_000_000
    assert cfg.indexing.embed_batch_size == 512
    assert cfg.embedding.backend == "openai_compat"
    assert cfg.reranker.backend == "rules"
    assert cfg.milvus.collection == "semble_global"


def test_core_dir_expands_user() -> None:
    cfg = SembleConfig(core={"dir": "~/.semble-test-core"})
    assert "~" not in str(cfg.core.dir)
    assert cfg.core.dir.is_absolute()


def test_load_config_yaml(tmp_path: Path) -> None:
    pytest.importorskip("yaml")
    cfg_path = tmp_path / "semble.yaml"
    cfg_path.write_text(
        "indexing:\n  max_file_bytes: 42\n  embed_batch_size: 7\n"
        "milvus:\n  uri: http://example.org:19530\n"
    )
    cfg = load_config(cfg_path)
    assert cfg.indexing.max_file_bytes == 42
    assert cfg.indexing.embed_batch_size == 7
    assert cfg.milvus.uri == "http://example.org:19530"


def test_load_config_json(tmp_path: Path) -> None:
    cfg_path = tmp_path / "semble.json"
    cfg_path.write_text(json.dumps({"indexing": {"max_file_bytes": 99}}))
    cfg = load_config(cfg_path)
    assert cfg.indexing.max_file_bytes == 99


def test_env_var_substitution_with_default(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    pytest.importorskip("yaml")
    monkeypatch.delenv("SEMBLE_TEST_VAR", raising=False)
    cfg_path = tmp_path / "semble.yaml"
    cfg_path.write_text("milvus:\n  uri: ${SEMBLE_TEST_VAR:-http://fallback:19530}\n")
    cfg = load_config(cfg_path)
    assert cfg.milvus.uri == "http://fallback:19530"


def test_env_var_substitution_overrides(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    pytest.importorskip("yaml")
    monkeypatch.setenv("SEMBLE_TEST_VAR", "http://from-env:19530")
    cfg_path = tmp_path / "semble.yaml"
    cfg_path.write_text("milvus:\n  uri: ${SEMBLE_TEST_VAR}\n")
    cfg = load_config(cfg_path)
    assert cfg.milvus.uri == "http://from-env:19530"


def test_load_config_missing_returns_defaults(tmp_path: Path) -> None:
    missing = tmp_path / "does-not-exist.yaml"
    cfg = load_config(missing)
    assert isinstance(cfg, SembleConfig)
    assert cfg.indexing == IndexingConfig()


def test_load_config_unknown_suffix_raises(tmp_path: Path) -> None:
    bogus = tmp_path / "config.toml"
    bogus.write_text("[indexing]\nmax_file_bytes = 1\n")
    with pytest.raises(ValueError, match="Unsupported config file format"):
        load_config(bogus)
