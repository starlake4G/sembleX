from __future__ import annotations

import hashlib
from pathlib import Path

from semble.index.file_walker import walk_files


class HashTracker:
    @staticmethod
    def file_hash(path: Path) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    @staticmethod
    def compute_hashes(root: Path, extensions: frozenset[str] | set[str]) -> dict[str, str]:
        hashes: dict[str, str] = {}
        for f in walk_files(root, list(extensions)):
            try:
                rel = str(f.relative_to(root)).replace("\\", "/")
                hashes[rel] = HashTracker.file_hash(f)
            except (OSError, ValueError):
                continue
        return hashes

    @staticmethod
    def diff(
        old: dict[str, str], new: dict[str, str]
    ) -> tuple[set[str], set[str], set[str]]:
        added = set(new) - set(old)
        removed = set(old) - set(new)
        modified = {p for p in set(old) & set(new) if old[p] != new[p]}
        return added, removed, modified
