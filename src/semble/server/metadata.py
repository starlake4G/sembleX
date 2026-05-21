from __future__ import annotations

import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from semble.types import Chunk, SearchResult


class RepoNotIndexedError(LookupError):
    """Raised by the remote indexer when a repo has not been indexed yet."""

    def __init__(self, repo_id: str, source: str | None = None) -> None:
        super().__init__(f"Repository not indexed: {source or repo_id}")
        self.repo_id = repo_id
        self.source = source


@dataclass(frozen=True, slots=True)
class StoredChunk:
    chunk_id: str
    chunk: Chunk
    content_hash: str


class MetadataStore:
    """SQLite metadata store for remote Semble chunks."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.executescript(
            """
            create table if not exists repos (
                repo_id text primary key,
                source text not null,
                source_path text not null,
                include_text_files integer not null,
                embedding_model text not null,
                embedding_dim integer not null,
                chunker_version text not null,
                commit_sha text,
                indexed_at text not null,
                chunk_count integer not null
            );

            create index if not exists idx_repos_source on repos(source);

            create table if not exists chunks (
                chunk_id text primary key,
                repo_id text not null,
                file_path text not null,
                start_line integer not null,
                end_line integer not null,
                language text,
                content text not null,
                content_hash text not null
            );

            create index if not exists idx_chunks_repo on chunks(repo_id);
            create index if not exists idx_chunks_location on chunks(repo_id, file_path, start_line, end_line);
            """
        )
        # Best-effort migration: add commit_sha if the column is missing on older DBs.
        cols = {row["name"] for row in self._conn.execute("pragma table_info(repos)").fetchall()}
        if "commit_sha" not in cols:
            self._conn.execute("alter table repos add column commit_sha text")
        self._conn.commit()

    def has_repo(self, repo_id: str) -> bool:
        row = self._conn.execute("select 1 from repos where repo_id = ?", (repo_id,)).fetchone()
        return row is not None

    def repo_chunk_count(self, repo_id: str) -> int:
        row = self._conn.execute("select chunk_count from repos where repo_id = ?", (repo_id,)).fetchone()
        return int(row["chunk_count"]) if row is not None else 0

    def repo_ids_for_source(self, source: str) -> list[str]:
        return [
            str(row["repo_id"])
            for row in self._conn.execute("select repo_id from repos where source = ?", (source,)).fetchall()
        ]

    def start_repo_replace(self, repo_id: str) -> None:
        with self._conn:
            self._conn.execute("delete from chunks where repo_id = ?", (repo_id,))
            self._conn.execute("delete from repos where repo_id = ?", (repo_id,))

    def finish_repo_replace(
        self,
        *,
        repo_id: str,
        source: str,
        source_path: str,
        include_text_files: bool,
        embedding_model: str,
        embedding_dim: int,
        chunker_version: str,
        commit_sha: str | None,
        indexed_at: str,
        chunk_count: int,
    ) -> None:
        with self._conn:
            self._conn.execute(
                """
                insert or replace into repos (
                    repo_id, source, source_path, include_text_files, embedding_model,
                    embedding_dim, chunker_version, commit_sha, indexed_at, chunk_count
                ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    repo_id,
                    source,
                    source_path,
                    int(include_text_files),
                    embedding_model,
                    embedding_dim,
                    chunker_version,
                    commit_sha,
                    indexed_at,
                    chunk_count,
                ),
            )

    def insert_chunks(self, repo_id: str, chunks: Iterable[StoredChunk]) -> None:
        rows = [
            (
                stored.chunk_id,
                repo_id,
                stored.chunk.file_path,
                stored.chunk.start_line,
                stored.chunk.end_line,
                stored.chunk.language,
                stored.chunk.content,
                stored.content_hash,
            )
            for stored in chunks
        ]
        if not rows:
            return
        with self._conn:
            self._conn.executemany(
                """
                insert or replace into chunks (
                    chunk_id, repo_id, file_path, start_line, end_line, language, content, content_hash
                ) values (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )

    def get_chunks(self, chunk_ids: list[str]) -> dict[str, Chunk]:
        if not chunk_ids:
            return {}
        placeholders = ",".join("?" for _ in chunk_ids)
        rows = self._conn.execute(
            f"select * from chunks where chunk_id in ({placeholders})",
            chunk_ids,
        ).fetchall()
        return {str(row["chunk_id"]): self._chunk_from_row(row) for row in rows}

    def all_chunks(self, repo_id: str) -> list[tuple[str, Chunk]]:
        """Return ``[(chunk_id, Chunk)]`` for *repo_id* in deterministic insertion order."""
        rows = self._conn.execute(
            "select * from chunks where repo_id = ? order by file_path, start_line, end_line, chunk_id",
            (repo_id,),
        ).fetchall()
        return [(str(row["chunk_id"]), self._chunk_from_row(row)) for row in rows]

    def find_chunk(self, repo_id: str, file_path: str, line: int) -> tuple[str, Chunk] | None:
        row = self._conn.execute(
            """
            select * from chunks
            where repo_id = ? and file_path = ? and start_line <= ? and end_line >= ?
            order by end_line asc
            limit 1
            """,
            (repo_id, file_path, line, line),
        ).fetchone()
        if row is None:
            return None
        return str(row["chunk_id"]), self._chunk_from_row(row)

    def results_from_hits(self, hits: list[tuple[str, float]]) -> list[SearchResult]:
        chunks = self.get_chunks([chunk_id for chunk_id, _ in hits])
        results: list[SearchResult] = []
        for cid, score in hits:
            chunk = chunks.get(cid)
            if chunk is not None:
                results.append(SearchResult(chunk=chunk, score=score))
        return results

    def _chunk_from_row(self, row: sqlite3.Row) -> Chunk:
        return Chunk(
            content=str(row["content"]),
            file_path=str(row["file_path"]),
            start_line=int(row["start_line"]),
            end_line=int(row["end_line"]),
            language=str(row["language"]) if row["language"] is not None else None,
        )
