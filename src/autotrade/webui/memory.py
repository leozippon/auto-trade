"""Operating-memory model for the console: the read view and the curated writes.

Both halves stand on the two places the pipeline already treats as
authoritative: the curated library committed under ``configs/operating_memory/``
and the experiment trees under ``experiments/``. Admission to the graduated tier
is never recomputed here — ``pipelines.skills`` answers it — so the console can
only show what a session starting now would really mount, and what past sessions
did mount, according to their run manifests.

The curated tier is the one writable surface. It is a tracked repository
directory that every session copies read-only into its workspace at session
start, so a write here is a repository edit that reaches sessions started
afterwards and never the sessions already running. Every write validates a
staged copy with the same validator the mount uses before it touches the
library, and lands the entry with a single rename.
"""

from __future__ import annotations

import shutil
import tempfile
import threading
from collections.abc import Mapping
from pathlib import Path

from autotrade.environment.runtime import chmod_tree
from autotrade.pipelines.hitl_state import HITL_DIR_NAME, PARAMS_NAME, read_json
from autotrade.pipelines.ledger import experiment_verdict
from autotrade.pipelines.skills import (
    CURATED_MEMORY_SOURCE,
    DEFAULT_OPERATING_MEMORY,
    MAX_SKILLS,
    MAX_SKILLS_BYTES,
    MAX_SKILLS_FILES,
    OPERATING_MEMORY_LIBRARY,
    SKILL_FILENAME,
    SkillsStats,
    build_skills_index,
    graduated_memory_sources,
    latest_skills_snapshot,
    operating_memory_snapshot_root,
    read_operating_memory_snapshot,
    resolve_operating_memory,
    snapshot_memory_sources,
    validate_memory_entry_ref,
    validate_skill_name,
    validate_skill_path,
    validate_skills_tree,
)

from .registry import read_ledger_records, resolve_experiment_dir

# The sandbox writes the full host-side manifest beside the Agent-visible one
# when a run's artifacts are collected (environment/runtime.py,
# environment/sandbox.py). It is the only per-session record of what was
# actually mounted, and it keeps the raw session identity this projection needs.
HOST_RUN_MANIFEST_NAME = "host_run_manifest.json"
# Errors reach the browser as a category plus a stable phrase: the underlying
# exceptions embed absolute ledger and library paths in their messages.
_UNREADABLE_LIBRARY = "curated memory library is unreadable"
_UNREADABLE_TIER = "graduated memory cannot be resolved"
_UNREADABLE_EXPERIMENT = "experiment state is unreadable"
_UNREADABLE_SNAPSHOT = "operating memory snapshot is unreadable"


def _error(exc: Exception, phrase: str) -> str:
    return f"{type(exc).__name__}: {phrase}"


def curated_library(repo_root: Path) -> dict[str, object]:
    """The human-curated tier as the mount would read it right now."""

    library = Path(repo_root) / OPERATING_MEMORY_LIBRARY
    payload: dict[str, object] = {
        "source": CURATED_MEMORY_SOURCE,
        "library": OPERATING_MEMORY_LIBRARY,
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
    path = Path(repo_root) / OPERATING_MEMORY_LIBRARY / entry_name / SKILL_FILENAME
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


# The tier reads every experiment's ledger and validates every published skills
# tree, which is most of a second on a full experiments root and several
# seconds per visitor when visits overlap. Everything a row depends on hangs off
# its experiment's ledger: the ledger is append-only (or atomically rewritten),
# and the skills generation a row points at is published immutably before the
# row names it. So the payload is kept while the experiment set and every
# ledger's size and mtime are unchanged — one slot, the current state; the lock
# makes overlapping misses compute it once.
_TIER_LOCK = threading.Lock()
_TIER_CACHE: dict[tuple[object, ...], dict[str, object]] = {}


def _tier_key(root: Path) -> tuple[object, ...]:
    key: list[object] = [str(root)]
    for directory in sorted(root.iterdir(), key=lambda path: path.name):
        if not directory.is_dir():
            continue
        try:
            stat = (directory / "ledgers" / "experiment_ledger.jsonl").stat()
        except OSError:
            key.append((directory.name,))
        else:
            key.append((directory.name, stat.st_size, stat.st_mtime_ns))
    return tuple(key)


def graduated_tier(experiments_root: Path) -> dict[str, object]:
    """Every experiment's verdict, what the tier admits now, and what it holds.

    Admission is whatever ``skills.graduated_memory_sources`` returns, never a
    second rule. ``published`` is the size of the experiment's current skills
    generation — the page needs it to tell an experiment the tier declines to
    offer from one that has nothing to offer yet.
    """

    root = Path(experiments_root)
    if not root.is_dir():
        return {"experiments": []}
    with _TIER_LOCK:
        key = _tier_key(root)
        if key in _TIER_CACHE:
            return _TIER_CACHE[key]
        payload = _graduated_tier(root)
        rows: list[dict[str, object]] = payload["experiments"]  # type: ignore[assignment]
        _TIER_CACHE.clear()
        # An error may be a transient I/O failure the key cannot see, so it is
        # read afresh on every request rather than kept.
        if "error" not in payload and not any("error" in row for row in rows):
            _TIER_CACHE[key] = payload
        return payload


def _graduated_tier(root: Path) -> dict[str, object]:
    payload: dict[str, object] = {"experiments": []}
    admitted: dict[str, list[str]] | None
    try:
        admitted = {
            source.source: list(source.entries)
            for source in graduated_memory_sources(root)
        }
    except (OSError, ValueError) as exc:
        # A tier that cannot be resolved is what a session starting now would
        # also hit. Report it, and leave every row's admission unknown rather
        # than printing a "not admitted" the read model cannot stand behind.
        payload["error"] = _error(exc, _UNREADABLE_TIER)
        admitted = None
    payload["experiments"] = [
        _tier_row(directory, admitted)
        for directory in sorted(root.iterdir(), key=lambda path: path.name)
        if directory.is_dir() and not directory.name.startswith(".")
    ]
    return payload


def _tier_row(
    directory: Path, admitted: Mapping[str, list[str]] | None
) -> dict[str, object]:
    row: dict[str, object] = {
        "experiment_id": directory.name,
        "verdict": None,
        "admitted": False,
        "published": 0,
        "entries": [],
    }
    try:
        records = read_ledger_records(directory)
        verdict = experiment_verdict(records)
        published = latest_skills_snapshot(records, experiment_dir=directory)
    except (OSError, TypeError, ValueError) as exc:
        row["error"] = _error(exc, _UNREADABLE_EXPERIMENT)
        # Unknown, not zero: an unreadable ledger answers neither question.
        row["published"] = None
        return row
    row["verdict"] = str(verdict["status"]) if isinstance(verdict, Mapping) else None
    row["published"] = published.stats.count
    if admitted is None:
        row["admitted"] = None
        return row
    row["admitted"] = directory.name in admitted
    row["entries"] = list(admitted.get(directory.name, ()))
    return row


def experiment_memory(experiments_root: Path, experiment_id: str) -> dict[str, object]:
    """This experiment's operating memory: one list, frozen when it was created.

    Nothing here is per session any more. The experiment resolved the curated
    library and the graduated tier once, at creation, and every one of its
    sessions mounts that same snapshot — so the question "what did this session
    get" has one answer for the whole experiment, and a later library change
    belongs to the next experiment rather than to this one's middle.
    """

    directory = resolve_experiment_dir(Path(experiments_root), experiment_id)
    params = read_json(directory / HITL_DIR_NAME / PARAMS_NAME)
    payload: dict[str, object] = {
        "experiment_id": experiment_id,
        "mode": resolve_operating_memory(params.get("operating_memory")),
        "default_mode": DEFAULT_OPERATING_MEMORY,
        # A count, not a projection: the collected runs no longer answer what was
        # mounted, they only say how many sessions have run against the snapshot.
        "sessions_seen": len(
            list((directory / "artifacts").glob(f"*/{HOST_RUN_MANIFEST_NAME}"))
        ),
        "snapshot": None,
    }
    try:
        record = read_operating_memory_snapshot(directory)
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
    record = read_operating_memory_snapshot(directory)
    if record is None:
        raise KeyError(f"{experiment_id} has no operating memory snapshot")
    item = operating_memory_snapshot_root(directory) / mount_source / entry_name
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
    origin = next(
        (
            str(entry.get("origin") or "")
            for entry in record.get("entries") or ()  # type: ignore[union-attr]
            if isinstance(entry, Mapping)
            and str(entry.get("source")) == mount_source
            and str(entry.get("name")) == entry_name
        ),
        "",
    )
    return {
        "experiment_id": experiment_id,
        "source": mount_source,
        "origin": origin,
        "name": entry_name,
        "title": str(listed["title"]),
        "summary": str(listed["summary"]),
        "bytes": int(listed["bytes"]),
        "files": len(listed["files"]),  # type: ignore[arg-type]
        "content": (item / SKILL_FILENAME).read_text(encoding="utf-8"),
    }


# ---- curated library writes ----------------------------------------------


def _library_dir(repo_root: Path) -> Path:
    return Path(repo_root) / OPERATING_MEMORY_LIBRARY


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
            # promoted file name cannot escape the entry it belongs to.
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


def _admitted_skill_dir(experiments_root: Path, experiment_id: str, skill: str) -> Path:
    """The only place a promotion may copy from: an admitted graduated skill.

    Admission is ``skills.graduated_memory_sources``, exactly as
    :func:`graduated_tier` shows it, so the console cannot promote from a
    candidate the page does not offer.
    """

    root = Path(experiments_root)
    resolve_experiment_dir(root, experiment_id)
    source = next(
        (
            item
            for item in graduated_memory_sources(root)
            if item.source == experiment_id
        ),
        None,
    )
    if source is None or skill not in source.entries:
        raise KeyError(f"{experiment_id} admits no skill named {skill}")
    return source.root / skill


def graduated_entry(
    experiments_root: Path, experiment_id: str, skill: str
) -> dict[str, object]:
    """One admitted graduated skill's body, read where it already lives.

    The same admission gate as the promotion it precedes, so the
    console can never show a candidate it could not copy — and a skill whose
    experiment is no longer admitted reads as unknown rather than as a body the
    tier would not mount.
    """

    name = validate_skill_name(skill)
    item = _admitted_skill_dir(Path(experiments_root), experiment_id, name)
    # The generation's own index, so a candidate is described exactly as the
    # session that mounts it would describe it.
    listed = next(
        (
            entry
            for entry in build_skills_index(item.parent)["skills"]  # type: ignore[union-attr]
            if entry["name"] == name
        ),
        None,
    )
    if listed is None:
        raise KeyError(f"{experiment_id} admits no skill named {name}")
    return {
        "experiment_id": experiment_id,
        "name": name,
        "title": str(listed["title"]),
        "summary": str(listed["summary"]),
        "bytes": int(listed["bytes"]),
        "files": len(listed["files"]),  # type: ignore[arg-type]
        "content": (item / SKILL_FILENAME).read_text(encoding="utf-8"),
    }


def promote_curated_entry(
    repo_root: Path,
    experiments_root: Path,
    *,
    name: str,
    experiment_id: str,
    skill: str,
) -> dict[str, object]:
    """Copy one admitted graduated skill into the curated library verbatim.

    The whole item is copied, ``scripts/`` and ``references/`` included, because
    that is what the mount would have carried and what the researcher then edits
    down in place.
    """

    entry = _entry_name(name or skill)
    source = _admitted_skill_dir(
        experiments_root, experiment_id, validate_skill_name(skill)
    )
    files: dict[str, bytes] = {}
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source).as_posix()
        if path.is_symlink():
            raise ValueError(f"skill symlink is forbidden: {relative}")
        if path.is_file():
            files[relative] = path.read_bytes()
    _install_entry(_library_dir(repo_root), entry, files, replace=False)
    return _write_result(repo_root, entry, "promoted")


def memory_overview(repo_root: Path, experiments_root: Path) -> dict[str, object]:
    """The whole operating-memory page in one read."""

    return {
        "default_mode": DEFAULT_OPERATING_MEMORY,
        "curated": curated_library(repo_root),
        "graduated": graduated_tier(experiments_root),
    }


__all__ = [
    "create_curated_entry",
    "curated_entry",
    "curated_library",
    "delete_curated_entry",
    "experiment_memory",
    "experiment_memory_entry",
    "graduated_entry",
    "graduated_tier",
    "memory_overview",
    "promote_curated_entry",
    "update_curated_entry",
]
