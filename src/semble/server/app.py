from __future__ import annotations

import asyncio
import secrets
from typing import Annotated, NoReturn

from semble.config import SembleConfig
from semble.server.indexer import RemoteIndexer
from semble.server.metadata import RepoNotIndexedError
from semble.server.models import FindRelatedRequest, IndexRequest, IndexResponse, SearchRequest
from semble.types import SearchResult


def create_app(config: SembleConfig | None = None):
    """Create the FastAPI app for the remote Semble index service."""
    try:
        from fastapi import Depends, FastAPI, Header, HTTPException
    except ImportError:
        raise RuntimeError("Server dependencies are missing. Install with: pip install 'semble[server]'") from None

    cfg = config or SembleConfig()
    indexer = RemoteIndexer(cfg)
    app = FastAPI(title="Semble Remote Index Service", version="1")

    def authorize(authorization: Annotated[str | None, Header()] = None) -> None:
        if not cfg.server.api_key:
            return
        expected = f"Bearer {cfg.server.api_key}"
        if authorization is None or not secrets.compare_digest(authorization, expected):
            raise HTTPException(status_code=401, detail="Invalid or missing bearer token")

    def handle_error(exc: Exception) -> NoReturn:
        if isinstance(exc, RepoNotIndexedError):
            raise HTTPException(
                status_code=409,
                detail={
                    "error": "not_indexed",
                    "repo_id": exc.repo_id,
                    "source": exc.source,
                    "message": "Repository not indexed. Call /v1/index first.",
                },
            ) from exc
        if isinstance(exc, (FileNotFoundError, NotADirectoryError, ValueError)):
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/index", dependencies=[Depends(authorize)])
    async def index_repo(request: IndexRequest) -> IndexResponse:
        try:
            outcome = await asyncio.to_thread(
                indexer.index_repo,
                request.repo,
                include_text_files=request.include_text_files,
                force=request.force,
                ref=request.ref,
            )
            return IndexResponse(repo_id=outcome.repo_id, chunk_count=outcome.chunk_count, indexed=outcome.indexed)
        except Exception as exc:
            handle_error(exc)

    @app.post("/v1/search", dependencies=[Depends(authorize)])
    async def search(request: SearchRequest) -> dict[str, object]:
        try:
            results = await asyncio.to_thread(
                indexer.search,
                request.query,
                repo=request.repo,
                top_k=request.top_k,
                include_text_files=request.include_text_files,
            )
            return {"results": [_result_to_dict(result) for result in results]}
        except Exception as exc:
            handle_error(exc)

    @app.post("/v1/find-related", dependencies=[Depends(authorize)])
    async def find_related(request: FindRelatedRequest) -> dict[str, object]:
        try:
            results = await asyncio.to_thread(
                indexer.find_related,
                request.file_path,
                request.line,
                repo=request.repo,
                top_k=request.top_k,
                include_text_files=request.include_text_files,
            )
            return {"results": [_result_to_dict(result) for result in results]}
        except Exception as exc:
            handle_error(exc)

    return app


def _result_to_dict(result: SearchResult) -> dict[str, object]:
    return {
        "score": result.score,
        "chunk": {
            "content": result.chunk.content,
            "file_path": result.chunk.file_path,
            "start_line": result.chunk.start_line,
            "end_line": result.chunk.end_line,
            "language": result.chunk.language,
        },
    }
