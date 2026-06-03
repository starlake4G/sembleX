from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from semble.utils import _is_git_url

_PROJECTS_ENV = "SEMBLE_PROJECTS_FILE"
_PROJECTS_FILE = "projects.json"


@dataclass(frozen=True, slots=True)
class ProjectEntry:
    """A registered repository that can participate in workspace search."""

    source: str
    name: str | None = None
    ref: str | None = None
    include_text_files: bool = False

    @property
    def display_name(self) -> str:
        """Human-friendly project name."""
        if self.name:
            return self.name
        if _is_git_url(self.source):
            return self.source.rstrip("/").rsplit("/", 1)[-1].removesuffix(".git") or self.source
        return Path(self.source).name or self.source


@dataclass(slots=True)
class ProjectsConfig:
    """List of registered repositories."""

    projects: list[ProjectEntry] = field(default_factory=list)


def projects_config_path() -> Path:
    """Return the path to the workspace project registry."""
    configured = os.environ.get(_PROJECTS_ENV)
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".semble" / _PROJECTS_FILE


def normalize_project_source(source: str) -> str:
    """Normalize a local path or git URL for stable comparisons."""
    stripped = source.strip()
    if not stripped:
        raise ValueError("Project source cannot be empty")
    if _is_git_url(stripped):
        return stripped
    return str(Path(stripped).expanduser().resolve())


def _entry_from_dict(data: dict[str, Any]) -> ProjectEntry:
    return ProjectEntry(
        source=normalize_project_source(str(data["source"])),
        name=str(data["name"]) if data.get("name") else None,
        ref=str(data["ref"]) if data.get("ref") else None,
        include_text_files=bool(data.get("include_text_files", False)),
    )


def _entry_to_dict(entry: ProjectEntry) -> dict[str, Any]:
    data: dict[str, Any] = {"source": entry.source}
    if entry.name:
        data["name"] = entry.name
    if entry.ref:
        data["ref"] = entry.ref
    if entry.include_text_files:
        data["include_text_files"] = True
    return data


def load_projects() -> ProjectsConfig:
    """Load registered projects. Missing registry means no projects."""
    path = projects_config_path()
    if not path.is_file():
        return ProjectsConfig()
    data = json.loads(path.read_text(encoding="utf-8") or "{}")
    raw_projects = data.get("projects", []) if isinstance(data, dict) else []
    projects = [_entry_from_dict(item) for item in raw_projects if isinstance(item, dict)]
    return ProjectsConfig(projects=projects)


def save_projects(config: ProjectsConfig) -> Path:
    """Persist the project registry and return the written path."""
    path = projects_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"projects": [_entry_to_dict(entry) for entry in config.projects]}
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def add_project(
    source: str,
    *,
    name: str | None = None,
    ref: str | None = None,
    include_text_files: bool = False,
) -> ProjectEntry:
    """Add or update a project in the registry."""
    normalized = normalize_project_source(source)
    config = load_projects()
    new_entry = ProjectEntry(
        source=normalized,
        name=name,
        ref=ref,
        include_text_files=include_text_files,
    )
    for idx, entry in enumerate(config.projects):
        if normalize_project_source(entry.source) == normalized:
            config.projects[idx] = new_entry
            save_projects(config)
            return new_entry
    config.projects.append(new_entry)
    save_projects(config)
    return new_entry


def remove_project(source: str) -> bool:
    """Remove a project from the registry. Returns True if it existed."""
    normalized = normalize_project_source(source)
    config = load_projects()
    kept = [entry for entry in config.projects if normalize_project_source(entry.source) != normalized]
    if len(kept) == len(config.projects):
        return False
    config.projects = kept
    save_projects(config)
    return True
