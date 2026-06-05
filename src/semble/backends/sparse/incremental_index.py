from __future__ import annotations

import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from semble.index.sparse import tokenize_for_bm25
from semble.interfaces import SparseIndex
from semble.tokens import tokenize
from semble.types import Chunk


class IncrementalSparseIndex(SparseIndex):
    """BM25 index with incremental add/remove support.

    Uses a dictionary-backed inverted index instead of a batch-only library,
    allowing individual documents to be added or removed without a full rebuild.
    Query scoring uses standard BM25 (Okapi BM25 with k1=1.5, b=0.75).
    """

    _K1 = 1.5
    _B = 0.75

    def __init__(self) -> None:
        self._inverted: dict[str, dict[str, int]] = {}
        self._doc_lengths: dict[str, int] = {}
        self._chunk_ids: list[str] = []
        self._tokens_by_chunk: dict[str, list[str]] = {}
        self._avgdl: float = 0.0
        self._N: int = 0

    @property
    def supports_incremental(self) -> bool:
        return True

    @property
    def doc_count(self) -> int:
        """Number of documents currently in the index."""
        return self._N

    def build(self, chunks: Sequence[Chunk], chunk_ids: Sequence[str]) -> None:
        if len(chunks) != len(chunk_ids):
            raise ValueError(f"chunks ({len(chunks)}) and chunk_ids ({len(chunk_ids)}) length mismatch")
        self._inverted.clear()
        self._doc_lengths.clear()
        self._chunk_ids = []
        self._tokens_by_chunk.clear()
        self._N = 0
        self._avgdl = 0.0
        self.add_documents(chunks, chunk_ids)

    def add_documents(self, chunks: Sequence[Chunk], chunk_ids: Sequence[str]) -> None:
        if len(chunks) != len(chunk_ids):
            raise ValueError(f"chunks ({len(chunks)}) and chunk_ids ({len(chunk_ids)}) length mismatch")
        existing = [cid for cid in chunk_ids if cid in self._tokens_by_chunk]
        if existing:
            self.remove_documents(existing)
        total_length = self._avgdl * self._N
        for chunk, cid in zip(chunks, chunk_ids):
            tokens = tokenize_for_bm25(chunk)
            self._tokens_by_chunk[cid] = tokens
            self._doc_lengths[cid] = len(tokens)
            total_length += len(tokens)
            tf: dict[str, int] = {}
            for tok in tokens:
                tf[tok] = tf.get(tok, 0) + 1
            for tok, freq in tf.items():
                posting = self._inverted.setdefault(tok, {})
                posting[cid] = freq
            self._chunk_ids.append(cid)
            self._N += 1
        self._avgdl = total_length / self._N if self._N > 0 else 0.0

    def remove_documents(self, chunk_ids: Sequence[str]) -> None:
        remove_set = set(chunk_ids)
        total_length = self._avgdl * self._N
        for cid in chunk_ids:
            if cid not in self._tokens_by_chunk:
                continue
            tokens = self._tokens_by_chunk.pop(cid)
            doc_len = self._doc_lengths.pop(cid, 0)
            total_length -= doc_len
            tf: dict[str, int] = {}
            for tok in tokens:
                tf[tok] = tf.get(tok, 0) + 1
            for tok in tf:
                posting = self._inverted.get(tok)
                if posting is not None:
                    posting.pop(cid, None)
                    if not posting:
                        del self._inverted[tok]
            self._N -= 1
        self._chunk_ids = [cid for cid in self._chunk_ids if cid not in remove_set]
        self._avgdl = total_length / self._N if self._N > 0 else 0.0

    def query(
        self,
        query: str,
        k: int,
        selector_ids: Sequence[str] | None = None,
    ) -> list[tuple[str, float]]:
        if not self._chunk_ids or not self._inverted:
            return []
        tokens = tokenize(query)
        if not tokens:
            return []
        selector_set: set[str] | None = set(selector_ids) if selector_ids is not None else None
        scores: dict[str, float] = {}
        n_docs = self._N if self._N > 0 else 1
        for tok in tokens:
            posting = self._inverted.get(tok)
            if posting is None:
                continue
            df = len(posting)
            idf = math.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)
            for cid, freq in posting.items():
                if selector_set is not None and cid not in selector_set:
                    continue
                dl = self._doc_lengths.get(cid, 1)
                avgdl = self._avgdl if self._avgdl > 0 else 1.0
                tf_norm = (freq * (self._K1 + 1)) / (freq + self._K1 * (1 - self._B + self._B * dl / avgdl))
                scores[cid] = scores.get(cid, 0.0) + idf * tf_norm
        if not scores:
            return []
        ranked = sorted(scores.items(), key=lambda x: -x[1])
        return [(cid, score) for cid, score in ranked[:k] if score > 0]

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        data: dict[str, Any] = {
            "chunk_ids": self._chunk_ids,
            "doc_lengths": self._doc_lengths,
            "tokens_by_chunk": self._tokens_by_chunk,
            "N": self._N,
            "avgdl": self._avgdl,
        }
        (path / "incremental_meta.json").write_text(json.dumps(data))
        inv_serialized: dict[str, list[list[Any]]] = {}
        for tok, posting in self._inverted.items():
            inv_serialized[tok] = [[cid, freq] for cid, freq in posting.items()]
        (path / "inverted.json").write_text(json.dumps(inv_serialized))

    def load(self, path: Path) -> None:
        data = json.loads((path / "incremental_meta.json").read_text())
        self._chunk_ids = data["chunk_ids"]
        self._doc_lengths = data["doc_lengths"]
        self._tokens_by_chunk = data["tokens_by_chunk"]
        self._N = data["N"]
        self._avgdl = data["avgdl"]
        inv_serialized = json.loads((path / "inverted.json").read_text())
        self._inverted = {}
        for tok, entries in inv_serialized.items():
            self._inverted[tok] = {str(cid): int(freq) for cid, freq in entries}


__all__ = ["IncrementalSparseIndex"]
