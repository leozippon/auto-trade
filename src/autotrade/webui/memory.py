"""Skill-library model for the console: the read view and the curated writes.

Both halves stand on the two places the pipeline already treats as
authoritative: the curated library committed under ``configs/skill_library/``
and the experiment trees under ``experiments/``, each holding the snapshot of
the library it froze at creation, which is what its sessions mount.

The curated library is the one writable surface. It is a tracked repository
directory that every experiment snapshots read-only at creation, so a write
here is a repository edit that reaches experiments created afterwards and
never the ones already created. Every write validates a staged copy with the
same validator the mount uses before it touches the library, and lands the
entry with a single rename.
"""

from __future__ import annotations

import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path

from autotrade.common.records import read_json_object
from autotrade.environment.runtime import chmod_tree
from autotrade.pipelines.hitl_state import HITL_DIR_NAME, PARAMS_NAME
from autotrade.pipelines.skills import (
    CURATED_MEMORY_SOURCE,
    DEFAULT_SKILL_LIBRARY,
    MAX_SKILLS,
    MAX_SKILLS_BYTES,
    MAX_SKILLS_FILES,
    SKILL_FILENAME,
    SKILL_LIBRARY_PATH,
    SkillsStats,
    build_skills_index,
    read_skill_library_snapshot,
    resolve_skill_library,
    skill_library_snapshot_root,
    snapshot_memory_sources,
    validate_memory_entry_ref,
    validate_skill_name,
    validate_skill_path,
    validate_skills_tree,
)

from .feedback_page import experiment_directories
from .registry import resolve_experiment_dir

# The sandbox writes the full host-side manifest beside the Agent-visible one
# when a run's artifacts are collected (environment/runtime.py,
# environment/sandbox.py). It is the only per-session record of what was
# actually mounted, and it keeps the raw session identity this projection needs.
HOST_RUN_MANIFEST_NAME = "host_run_manifest.json"
# Errors reach the browser as a category plus a stable phrase: the underlying
# exceptions embed absolute ledger and library paths in their messages.
_UNREADABLE_LIBRARY = "curated memory library is unreadable"
_UNREADABLE_SNAPSHOT = "skill library snapshot is unreadable"


def _error(exc: Exception, phrase: str) -> str:
    return f"{type(exc).__name__}: {phrase}"


def curated_library(repo_root: Path) -> dict[str, object]:
    """The human-curated tier as the mount would read it right now."""

    library = Path(repo_root) / SKILL_LIBRARY_PATH
    payload: dict[str, object] = {
        "source": CURATED_MEMORY_SOURCE,
        "library": SKILL_LIBRARY_PATH,
        "entries": [],
    }
    if not library.is_dir():
        return payload
    try:
        # The same index the Agent gets, so the console cannot describe an
        # entry differently from the session that reads it.
        index = build_skills_index(library)
    except (OSError, ValueError) as exc:
        payload["error"] = _error(exc, _UNREADABLE_LIBRARY)
        return payload
    entries = index["skills"]
    payload["entries"] = [
        {
            "name": str(entry["name"]),
            "title": str(entry["title"]),
            "summary": str(entry["summary"]),
            "bytes": int(entry["bytes"]),
            "files": len(entry["files"]),
        }
        for entry in entries  # type: ignore[union-attr]
    ]
    return payload


def curated_entry(repo_root: Path, name: str) -> dict[str, object]:
    """One curated entry's full ``SKILL.md`` text.

    The name is validated against the shared skill-name rule before it touches
    the filesystem, so the route cannot be used to read anything else.

    The body is served verbatim: this is researcher-authored repository content
    that the console also edits, and rewriting it for display would make the
    editor's round-trip lossy. Host paths are scrubbed where they can genuinely
    appear without anyone choosing to write them — error messages and projected
    experiment state — not in a file the researcher wrote and committed.
    """

    entry_name = validate_skill_name(name)
    path = Path(repo_root) / SKILL_LIBRARY_PATH / entry_name / SKILL_FILENAME
    if not path.is_file():
        raise KeyError(f"unknown curated memory entry: {entry_name}")
    content = path.read_text(encoding="utf-8")
    # The listing row this body belongs to, so the reader sees one description.
    # A library too malformed to index still serves the body it asked for.
    listed = next(
        (
            item
            for item in curated_library(repo_root)["entries"]  # type: ignore[union-attr]
            if item["name"] == entry_name
        ),
        {"name": entry_name, "title": "", "summary": "", "bytes": len(content.encode())},
    )
    return {**listed, "content": content}


def experiment_memory(experiments_root: Path, experiment_id: str) -> dict[str, object]:
    """This experiment's skill library: one list, frozen when it was created.

    Nothing here is per session. The experiment resolved the curated library
    once, at creation, and every one of its sessions mounts that same snapshot
    -- so the question "what did this session get" has one answer for the
    whole experiment, and a later library change belongs to the next
    experiment rather than to this one's middle.
    """

    directory = resolve_experiment_dir(Path(experiments_root), experiment_id)
    params = read_json_object(directory / HITL_DIR_NAME / PARAMS_NAME, missing_ok=True)
    payload: dict[str, object] = {
        "experiment_id": experiment_id,
        "mode": resolve_skill_library(params.get("skill_library")),
        "default_mode": DEFAULT_SKILL_LIBRARY,
        # A count, not a projection: the collected runs no longer answer what was
        # mounted, they only say how many sessions have run against the snapshot.
        "sessions_seen": len(
            list((directory / "artifacts").glob(f"*/{HOST_RUN_MANIFEST_NAME}"))
        ),
        "snapshot": None,
    }
    try:
        record = read_skill_library_snapshot(directory)
        if record is not None:
            payload["snapshot"] = {
                "created_at": str(record.get("created_at") or ""),
                "created_from": str(record.get("created_from") or ""),
                "mode": str(record.get("mode") or ""),
                "sources": [
                    {
                        "source": source.source,
                        "origin": source.origin,
                        "entries": list(source.entries),
                    }
                    for source in snapshot_memory_sources(record, directory)
                ],
            }
    except (OSError, TypeError, ValueError) as exc:
        payload["error"] = _error(exc, _UNREADABLE_SNAPSHOT)
    return payload


def experiment_memory_entry(
    experiments_root: Path, experiment_id: str, source: str, name: str
) -> dict[str, object]:
    """One entry's body as THIS experiment holds it, read from its snapshot.

    Deliberately not the library's current text: the snapshot is what the
    experiment's sessions actually read, and the library may have moved since.
    """

    directory = resolve_experiment_dir(Path(experiments_root), experiment_id)
    mount_source, entry_name = validate_memory_entry_ref(f"{source}/{name}")
    record = read_skill_library_snapshot(directory)
    if record is None:
        raise KeyError(f"{experiment_id} has no skill library snapshot")
    item = skill_library_snapshot_root(directory) / mount_source / entry_name
    if not (item / SKILL_FILENAME).is_file():
        raise KeyError(f"{experiment_id} did not mount {mount_source}/{entry_name}")
    # The generation's own index, so the snapshot is described exactly as the
    # session that mounted it described it.
    listed = next(
        (
            entry
            for entry in build_skills_index(item.parent)["skills"]  # type: ignore[union-attr]
            if entry["name"] == entry_name
        ),
        None,
    )
    if listed is None:
        raise KeyError(f"{experiment_id} did not mount {mount_source}/{entry_name}")
    return {
        "experiment_id": experiment_id,
        "source": mount_source,
        "origin": "curated",
        "name": entry_name,
        "title": str(listed["title"]),
        "summary": str(listed["summary"]),
        "bytes": int(listed["bytes"]),
        "files": len(listed["files"]),  # type: ignore[arg-type]
        "content": (item / SKILL_FILENAME).read_text(encoding="utf-8"),
    }


# ---- curated library writes ----------------------------------------------


def _library_dir(repo_root: Path) -> Path:
    return Path(repo_root) / SKILL_LIBRARY_PATH


def _entry_name(name: str) -> str:
    """The shared skill-name rule, plus the one name the mount layout owns."""

    entry = validate_skill_name(name)
    if entry == CURATED_MEMORY_SOURCE:
        raise ValueError(
            f"{CURATED_MEMORY_SOURCE!r} is the mount name of the curated tier "
            "and cannot be an entry name"
        )
    return entry


def _entry_usage(item: Path) -> tuple[int, int]:
    files = [path for path in item.rglob("*") if path.is_file()]
    return len(files), sum(path.stat().st_size for path in files)


def _assert_library_budget(
    library: Path, name: str, staged: SkillsStats, *, replacing: bool
) -> None:
    """The library the write would produce must stay inside the shared caps.

    ``validate_skills_tree`` bounds a tree as a whole and runs on the library at
    every mount, so a per-entry check would not answer the question. Validating
    the current library first is deliberate: an entry cannot be added to a
    library the mount already refuses — deleting the offending entry is how that
    is repaired.
    """

    current = validate_skills_tree(library, require_writable=False)
    replaced_files, replaced_bytes = (
        _entry_usage(library / name) if replacing else (0, 0)
    )
    if current.count + (0 if replacing else 1) > MAX_SKILLS:
        raise ValueError(f"curated library exceeds {MAX_SKILLS} entries")
    if current.files - replaced_files + staged.files > MAX_SKILLS_FILES:
        raise ValueError(f"curated library exceeds {MAX_SKILLS_FILES} files")
    if current.bytes - replaced_bytes + staged.bytes > MAX_SKILLS_BYTES:
        raise ValueError(f"curated library exceeds {MAX_SKILLS_BYTES} bytes")


def _staging_dir(library: Path) -> Path:
    """A sibling of the library: same filesystem, so the swap is one rename, and
    outside the library, so a concurrent mount never sees a partial entry or the
    hidden path ``validate_skills_tree`` would refuse."""

    library.parent.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(dir=library.parent, prefix=f".{library.name}.tmp-"))


def _install_entry(
    library: Path, name: str, files: Mapping[str, bytes], *, replace: bool
) -> None:
    """Stage one complete entry, validate it, then swap it into the library."""

    destination = library / name
    if destination.exists() and not replace:
        raise FileExistsError(f"curated memory entry already exists: {name}")
    if replace and not destination.is_dir():
        raise KeyError(f"unknown curated memory entry: {name}")
    library.mkdir(parents=True, exist_ok=True)
    staging = _staging_dir(library)
    try:
        tree = staging / "library"
        item = tree / name
        item.mkdir(parents=True)
        for relative, payload in files.items():
            # The shared path rule refuses '..', absolute and hidden paths, so a
            # file name cannot escape the entry it belongs to.
            target = item.joinpath(*validate_skill_path(relative).parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
        chmod_tree(tree, file_mode=0o644, dir_mode=0o755)
        # The mount's own validator on the staged copy: an entry a session would
        # refuse never reaches the library.
        staged = validate_skills_tree(tree, require_writable=True)
        _assert_library_budget(library, name, staged, replacing=destination.is_dir())
        # Both halves are validated, so the swap only has to be atomic: the old
        # entry moves out of the library before the new one moves in.
        replaced = staging / "replaced"
        if destination.is_dir():
            destination.replace(replaced)
        try:
            item.replace(destination)
        except OSError:
            if replaced.exists():
                replaced.replace(destination)
            raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _write_result(repo_root: Path, name: str, action: str) -> dict[str, object]:
    return {
        "name": name,
        "action": action,
        "curated": curated_library(repo_root),
    }


def create_curated_entry(repo_root: Path, name: str, content: str) -> dict[str, object]:
    """Add one entry written by the researcher.

    A name shared with a running experiment's own skill shadows nothing there:
    that experiment mounts the snapshot it froze at creation.
    """

    entry = _entry_name(name)
    _install_entry(
        _library_dir(repo_root),
        entry,
        {SKILL_FILENAME: str(content).encode("utf-8")},
        replace=False,
    )
    return _write_result(repo_root, entry, "created")


def update_curated_entry(repo_root: Path, name: str, content: str) -> dict[str, object]:
    """Replace one entry's ``SKILL.md``; its other files are carried over.

    Running experiments are unaffected: each mounts its own frozen snapshot.
    """

    entry = _entry_name(name)
    library = _library_dir(repo_root)
    item = library / entry
    if item.is_symlink() or not item.is_dir():
        raise KeyError(f"unknown curated memory entry: {entry}")
    files = {
        path.relative_to(item).as_posix(): path.read_bytes()
        for path in sorted(item.rglob("*"))
        if path.is_file()
    }
    files[SKILL_FILENAME] = str(content).encode("utf-8")
    _install_entry(library, entry, files, replace=True)
    return _write_result(repo_root, entry, "updated")


def delete_curated_entry(repo_root: Path, name: str) -> dict[str, object]:
    """Remove one entry. Running sessions keep the copy they already mounted.

    Deliberately not validated against the library first: removing the offending
    entry is how a library the mount refuses is repaired from the console.
    """

    entry = validate_skill_name(name)
    library = _library_dir(repo_root)
    item = library / entry
    if item.is_symlink() or not item.is_dir():
        raise KeyError(f"unknown curated memory entry: {entry}")
    staging = _staging_dir(library)
    try:
        item.replace(staging / entry)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return _write_result(repo_root, entry, "deleted")


def memory_overview(repo_root: Path, experiments_root: Path) -> dict[str, object]:
    """The whole skill-library page in one read: the curated library, and
    the experiments the page's two inboxes can be narrowed to."""

    return {
        "default_mode": DEFAULT_SKILL_LIBRARY,
        "curated": curated_library(repo_root),
        "experiments": [directory.name for directory in experiment_directories(experiments_root)],
    }


__all__ = [
    "create_curated_entry",
    "curated_entry",
    "curated_library",
    "delete_curated_entry",
    "experiment_memory",
    "experiment_memory_entry",
    "memory_overview",
    "update_curated_entry",
]
