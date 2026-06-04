import logging

from semble.chunking.core import ChunkBoundary, chunk, chunk_lines, is_supported_language
from semble.types import Chunk

logger = logging.getLogger(__name__)

# The desired length of chunks in chars.
# TODO: makes this configurable
_DESIRED_CHUNK_LENGTH_CHARS = 1500
# Default hard cap on a single chunk's char length; see IndexingConfig.max_chunk_chars.
# Used when chunk_source is called without an explicit cap (e.g. from tests).
_DEFAULT_MAX_CHUNK_CHARS = 12_000


def _hard_split(boundaries: list[ChunkBoundary], max_chars: int) -> list[ChunkBoundary]:
    """Split any boundary longer than *max_chars* into fixed-size pieces by char offset.

    The tree-sitter chunker only enforces ``desired_length`` by recursing into a node's
    children; a leaf with no children (a huge string/comment/data literal) and the
    line-chunker's single oversized line both escape it. This pass is the safety net that
    guarantees no chunk exceeds the embedding context, so nothing gets silently truncated.
    """
    if max_chars <= 0:
        return boundaries
    out: list[ChunkBoundary] = []
    for boundary in boundaries:
        if boundary.end - boundary.start <= max_chars:
            out.append(boundary)
            continue
        start = boundary.start
        while start < boundary.end:
            end = min(start + max_chars, boundary.end)
            out.append(ChunkBoundary(start=start, end=end))
            start = end
    return out


def chunk_source(
    source: str,
    file_path: str,
    language: str | None,
    *,
    max_chunk_chars: int = _DEFAULT_MAX_CHUNK_CHARS,
) -> list[Chunk]:
    """Chunk pre-read source text, hard-capping each chunk at *max_chunk_chars*."""
    if not source.strip():
        return []
    chunk_boundaries = None
    if language is not None and is_supported_language(language):
        chunk_boundaries = chunk(source, language, _DESIRED_CHUNK_LENGTH_CHARS)
    # This is an if because the error state of the parser above
    # is a None.
    if chunk_boundaries is None:
        chunk_boundaries = chunk_lines(source, _DESIRED_CHUNK_LENGTH_CHARS)

    chunk_boundaries = _hard_split(chunk_boundaries, max_chunk_chars)

    chunks: list[Chunk] = []
    for boundary in chunk_boundaries:
        # Clamp to start_index so zero-length chunks don't produce an off-by-one.
        end_index = max(boundary.end - 1, boundary.start)
        text = source[boundary.start : end_index + 1]
        chunks.append(
            Chunk(
                content=text,
                file_path=file_path,
                start_line=source[: boundary.start].count("\n") + 1,
                end_line=source[:end_index].count("\n") + 1,
                language=language,
            )
        )
    return chunks
