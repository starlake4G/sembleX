from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import bm25s
import numpy as np

from semble.index.sparse import BM25TokenCorpus
from semble.interfaces import SparseIndex
from semble.tokens import tokenize
from semble.types import Chunk


class Bm25sSparseIndex(SparseIndex):
    """BM25 index over chunks, exposing the unified ``SparseIndex`` interface."""

    def __init__(self) -> None:
        self._index: bm25s.BM25 | None = None
        self._chunk_ids: list[str] = []
        self._index_by_id: dict[str, int] = {}

    def build(self, chunks: Sequence[Chunk], chunk_ids: Sequence[str]) -> None:
        if len(chunks) != len(chunk_ids):
            raise ValueError(f"chunks ({len(chunks)}) and chunk_ids ({len(chunk_ids)}) length mismatch")
        self._chunk_ids = list(chunk_ids)
        self._index_by_id = {cid: i for i, cid in enumerate(self._chunk_ids)}
        self._index = bm25s.BM25()
        self._index.index(BM25TokenCorpus(list(chunks)), show_progress=False)

    def query(
        self,
        query: str,
        k: int,
        selector_ids: Sequence[str] | None = None,
    ) -> list[tuple[str, float]]:
        if self._index is None or not self._chunk_ids:
            return []
        tokens = tokenize(query)
        if not tokens:
            return []
        mask: np.ndarray | None = None
        if selector_ids is not None:
            mask = np.zeros(len(self._chunk_ids), dtype=bool)
            for cid in selector_ids:
                pos = self._index_by_id.get(cid)
                if pos is not None:
                    mask[pos] = True
            if not mask.any():
                return []
        scores: np.ndarray = self._index.get_scores(tokens, weight_mask=mask)
        effective_k = min(k, scores.size)
        if effective_k <= 0:
            return []
        order = np.argpartition(-scores, kth=effective_k - 1)[:effective_k]
        order = order[np.argsort(-scores[order])]
        return [
            (self._chunk_ids[int(pos)], float(scores[int(pos)]))
            for pos in order
            if scores[int(pos)] > 0
        ]

    def save(self, path: Path) -> None:
        if self._index is None:
            raise RuntimeError("Cannot save an empty Bm25sSparseIndex")
        path.mkdir(parents=True, exist_ok=True)
        self._index.save(str(path / "bm25s"))
        (path / "chunk_ids.json").write_text(json.dumps(self._chunk_ids))

    def load(self, path: Path) -> None:
        self._index = bm25s.BM25.load(str(path / "bm25s"), load_corpus=False)
        self._chunk_ids = json.loads((path / "chunk_ids.json").read_text())
        self._index_by_id = {cid: i for i, cid in enumerate(self._chunk_ids)}

    @property
    def chunk_ids(self) -> list[str]:
        return list(self._chunk_ids)


__all__ = ["Bm25sSparseIndex"]
