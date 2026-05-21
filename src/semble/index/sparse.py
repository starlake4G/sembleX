from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np
import numpy.typing as npt

from semble.tokens import tokenize
from semble.types import Chunk


def selector_to_mask(selector: npt.NDArray[np.int_] | None, size: int) -> npt.NDArray[np.bool_] | None:
    """Convert a selector array of indices into a boolean mask of length ``size``."""
    if selector is None:
        return None
    mask = np.zeros(size, dtype=bool)
    mask[selector] = True
    return mask


def enrich_for_bm25(chunk: Chunk) -> str:
    """Append file path components to BM25 content to boost path-based queries.

    Assumes ``chunk.file_path`` is already repo-relative (set by ``create_index_from_path``)
    so machine-specific directory components are never indexed.
    """
    path = Path(chunk.file_path)
    stem = path.stem
    dir_parts = [part for part in path.parent.parts if part not in (".", "/")]
    dir_text = " ".join(dir_parts[-3:])  # Last 3 directory components
    # Repeat the stem twice to up-weight file-path matches in BM25.
    return f"{chunk.content} {stem} {stem} {dir_text}"


def tokenize_for_bm25(chunk: Chunk) -> list[str]:
    """Tokenize a chunk for BM25 without copying the full chunk content."""
    tokens = tokenize(chunk.content)
    path = Path(chunk.file_path)

    stem_tokens = tokenize(path.stem)
    tokens.extend(stem_tokens)
    tokens.extend(stem_tokens)

    dir_parts = [part for part in path.parent.parts if part not in (".", "/")]
    if dir_parts:
        tokens.extend(tokenize(" ".join(dir_parts[-3:])))
    return tokens


class BM25TokenCorpus:
    """Re-iterable BM25 corpus that tokenizes chunks lazily."""

    def __init__(self, chunks: Sequence[Chunk]) -> None:
        self._chunks = chunks

    def __iter__(self) -> Iterator[list[str]]:
        for chunk in self._chunks:
            yield tokenize_for_bm25(chunk)
