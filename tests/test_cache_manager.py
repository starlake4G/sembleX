from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from semble.cache.manager import CacheManager
from semble.config import SembleConfig
from semble.index.index import SembleIndex


@pytest.fixture
def config(tmp_path: Path) -> SembleConfig:
    return SembleConfig(cache={"enabled": True, "dir": str(tmp_path / "cache")})


def test_save_then_load_returns_chunks(config: SembleConfig, tmp_project: Path, mock_model) -> None:
    idx = SembleIndex.from_path(tmp_project, model=mock_model, config=config)
    saved_ids = list(idx.chunk_ids)
    saved_count = len(idx.chunks)

    mgr = CacheManager(config.cache.dir, config)
    cached = mgr.load_from_disk(str(tmp_project.resolve()))
    assert cached is not None
    chunks, chunk_ids, vs, sparse = cached
    assert len(chunks) == saved_count
    assert chunk_ids == saved_ids
    # Vector store should round-trip a non-empty query.
    hits = vs.query("local", np.asarray(mock_model.encode(["x"]), dtype=np.float32), k=3)
    assert hits


def test_is_disk_valid_detects_hash_change(config: SembleConfig, tmp_project: Path, mock_model) -> None:
    SembleIndex.from_path(tmp_project, model=mock_model, config=config)
    mgr = CacheManager(config.cache.dir, config)

    assert mgr.is_disk_valid(str(tmp_project.resolve())) is True

    # Modify a tracked file and the cache must invalidate.
    (tmp_project / "auth.py").write_text("def changed(): pass\n")
    assert mgr.is_disk_valid(str(tmp_project.resolve())) is False
