from __future__ import annotations

from pydantic import BaseModel, Field


class IndexRequest(BaseModel):
    repo: str
    include_text_files: bool = False
    force: bool = False
    ref: str | None = None


class SearchRequest(BaseModel):
    query: str
    repo: str
    top_k: int = Field(default=5, ge=1, le=100)
    include_text_files: bool = False


class FindRelatedRequest(BaseModel):
    file_path: str
    line: int = Field(ge=1)
    repo: str
    top_k: int = Field(default=5, ge=1, le=100)
    include_text_files: bool = False


class ChunkResponse(BaseModel):
    content: str
    file_path: str
    start_line: int
    end_line: int
    language: str | None = None


class SearchResultResponse(BaseModel):
    chunk: ChunkResponse
    score: float


class SearchResponse(BaseModel):
    results: list[SearchResultResponse]


class MessageResponse(BaseModel):
    message: str


class IndexResponse(BaseModel):
    repo_id: str
    chunk_count: int
    indexed: bool
