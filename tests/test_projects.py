from pathlib import Path

from semble.projects import add_project, load_projects, remove_project


def test_projects_add_list_remove(tmp_path: Path, monkeypatch) -> None:
    registry = tmp_path / "projects.json"
    project = tmp_path / "repo"
    project.mkdir()
    monkeypatch.setenv("SEMBLE_PROJECTS_FILE", str(registry))

    entry = add_project(str(project), name="api")

    assert entry.source == str(project.resolve())
    assert entry.display_name == "api"
    assert load_projects().projects == [entry]

    assert remove_project(str(project)) is True
    assert remove_project(str(project)) is False
    assert load_projects().projects == []


def test_projects_update_existing(tmp_path: Path, monkeypatch) -> None:
    registry = tmp_path / "projects.json"
    project = tmp_path / "repo"
    project.mkdir()
    monkeypatch.setenv("SEMBLE_PROJECTS_FILE", str(registry))

    add_project(str(project), name="old")
    add_project(str(project), name="new", include_text_files=True)

    projects = load_projects().projects
    assert len(projects) == 1
    assert projects[0].display_name == "new"
    assert projects[0].include_text_files is True
