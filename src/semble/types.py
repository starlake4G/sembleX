import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol, TypeAlias

import numpy as np
import numpy.typing as npt

EmbeddingMatrix: TypeAlias = npt.NDArray[np.float32]


class Encoder(Protocol):
    """Protocol for embedding models."""

    @property
    def dim(self) -> int:
        """The dimensionality of the embedding."""
        ...

    def encode(self, texts: Sequence[str], /, **kwargs: Any) -> EmbeddingMatrix:
        """Encode texts into embeddings as a 2D float32 array."""
        ...  # pragma: no cover


@dataclass(frozen=True, slots=True)
class Chunk:
    """A single indexable unit of code."""

    content: str
    file_path: str
    start_line: int
    end_line: int
    language: str | None = None

    @property
    def location(self) -> str:
        """File path and line range as a string."""
        return f"{self.file_path}:{self.start_line}-{self.end_line}"


def chunk_id(chunk: "Chunk", namespace: str = "") -> str:
    """Stable identifier for a chunk inside *namespace*.

    Derived from path + line range + content hash so the same source chunk
    always maps to the same id regardless of insertion order, and trivial
    content edits invalidate the id.
    """
    content_digest = hashlib.sha256(chunk.content.encode("utf-8")).hexdigest()[:16]
    raw = f"{namespace}|{chunk.file_path}|{chunk.start_line}|{chunk.end_line}|{content_digest}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


@dataclass(frozen=True, slots=True)
class SearchResult:
    """A single search result with score and optional source-repo attribution."""

    chunk: Chunk
    score: float
    repo_id: str | None = None
    repo_name: str | None = None
    repo_source: str | None = None
