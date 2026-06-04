from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

from pathspec import GitIgnoreSpec


@dataclass(frozen=True)
class IgnoreSpec:
    base: Path
    spec: GitIgnoreSpec


_DEFAULT_IGNORED_DIRS: frozenset[str] = frozenset(
    {
        ".git/",
        ".hg/",
        ".svn/",
        "__pycache__/",
        "node_modules/",
        ".venv/",
        "venv/",
        ".tox/",
        ".mypy_cache/",
        ".pytest_cache/",
        ".ruff_cache/",
        ".cache/",
        ".semble/",
        ".next/",
        "dist/",
        "build/",
        ".eggs/",
        # Vendored / third-party code: their internals pollute cross-repo search
        # results and waste embeddings (we want *your* code, not bundled libs).
        "vendor/",
        "third_party/",
        "libs/",
        "bower_components/",
    }
)

# Binary/minified detection thresholds (content-level skips applied by the
# indexer on already-read file content; kept here so all skip rules live
# together). These catch noise that the extension/dir filters can't: binary
# files dragged in by a .gitignore negation (e.g. an image under data/), and
# minified/bundled assets (xlsx.full.min.js, swagger-ui-bundle.js, ...) whose
# byte soup adds hundreds of meaningless chunks.
_BINARY_SNIFF_BYTES = 8192
_BINARY_REPLACEMENT_RATIO = 0.1
_MINIFIED_SAMPLE_CHARS = 65536
_MINIFIED_AVG_LINE_LEN = 2000


def looks_binary(text: str) -> bool:
    r"""Heuristically decide whether decoded file content is really binary.

    ``text`` is read with ``errors="replace"``, so invalid bytes survive as the
    U+FFFD replacement char and embedded NULs survive as ``\\x00``. A NUL byte is
    the classic git binary signal; a high replacement-char ratio means the bytes
    were not valid UTF-8 (image, pickle, archive, ...).
    """
    sample = text[:_BINARY_SNIFF_BYTES]
    if not sample:
        return False
    if "\x00" in sample:
        return True
    return sample.count("�") / len(sample) > _BINARY_REPLACEMENT_RATIO


def looks_minified(file_name: str, text: str) -> bool:
    """Heuristically decide whether content is a minified/bundled asset.

    Two signals: a ``.min.`` filename, or a very large average line length —
    minified/bundled files pack everything onto a few enormous lines, while
    real source (including big register-map headers) keeps short lines.
    """
    if ".min." in file_name.lower():
        return True
    sample = text[:_MINIFIED_SAMPLE_CHARS]
    if not sample:
        return False
    lines = sample.count("\n") + 1
    return len(sample) / lines > _MINIFIED_AVG_LINE_LEN


def _load_ignore_for_dir(directory: Path) -> GitIgnoreSpec | None:
    """Loads a gitignore and sembleignore for a dir."""
    gitignore = directory / ".gitignore"
    sembleignore = directory / ".sembleignore"

    lines = []
    if gitignore.is_file():
        lines.extend(gitignore.read_text(encoding="utf-8", errors="ignore").splitlines())
    if sembleignore.is_file():
        lines.extend(sembleignore.read_text(encoding="utf-8", errors="ignore").splitlines())
    if lines:
        return GitIgnoreSpec.from_lines(lines)
    return None


def walk_files(root: Path, extensions: Sequence[str], ignore: Sequence[str] | None = None) -> Iterator[Path]:
    """Yield files under root matching extensions, skipping ignored paths.

    Directories matching DEFAULT_IGNORED_DIRS plus any names in ignore are always
    skipped. If the root contains a .gitignore, its patterns are also honoured.

    :param root: Root directory to walk.
    :param extensions: List of file extensions to match.
    :param ignore: Additional patterns to ignore.
    :yield: Path to each file under root matching the criteria.
    :ytype: Path
    """
    extensions_set = frozenset(extensions)
    dir_patterns = list(sorted(_DEFAULT_IGNORED_DIRS)) + list(ignore or [])
    base_spec = GitIgnoreSpec.from_lines(dir_patterns, backend="simple")
    s = IgnoreSpec(base=root, spec=base_spec)
    yield from _walk(root, [s], extensions_set)


def _is_ignored(path: Path, specs: list[IgnoreSpec]) -> tuple[bool, bool]:
    """Check if a path is ignored by any of the provided ignore specs."""
    try:
        is_dir = path.is_dir()
    except OSError:
        return True, False
    ignored = False
    found = False
    for ignore_spec in specs:
        try:
            # If there is no relative path, this is invalid.
            relative = path.relative_to(ignore_spec.base)
        except ValueError:
            continue

        relative_str = relative.as_posix()
        # We need to add a trailing slash. Gitignore
        # matches dirs as trailing '/'.
        if is_dir:
            relative_str += "/"

        # Loop over all the patterns
        for pattern in ignore_spec.spec.patterns:
            # This pattern doesn't do anything.
            if pattern.include is None:
                continue

            if pattern.match_file(relative_str) is not None:
                ignored = pattern.include
                # Bypass extension filter only for negation patterns with a file
                # extension suffix (e.g. !special.kjs, !*.py). Patterns without
                # a suffix (e.g. !vendor/, !.github/*) target directories or
                # broad globs and should not bypass extension filtering.
                pat = pattern.pattern
                found = not ignored and isinstance(pat, str) and bool(Path(pat.rstrip("/")).suffix)

    return ignored, found


def _walk(
    directory: Path,
    inherited_specs: list[IgnoreSpec],
    extensions: frozenset[str],
) -> Iterator[Path]:
    """Recursive function for walking files under a directory."""
    spec = _load_ignore_for_dir(directory)
    if spec is not None:
        inherited_specs = [
            *inherited_specs,
            IgnoreSpec(base=directory, spec=spec),
        ]

    for item in sorted(directory.iterdir()):
        try:
            if item.is_symlink():
                continue
            is_ignored, found = _is_ignored(item, inherited_specs)
            if is_ignored:
                continue
            if item.is_dir():
                yield from _walk(item, inherited_specs, extensions)
            elif item.is_file() and (found or item.suffix.lower() in extensions):
                yield item
        except OSError:
            continue
