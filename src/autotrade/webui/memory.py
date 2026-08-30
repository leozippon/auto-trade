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
from collections.abc import Iterable, Mapping
from pathlib import Path

from autotrade.environment.runtime import chmod_tree
from autotrade.pipelines.hitl_state import HITL_DIR_NAME, PARAMS_NAME, read_json
from autotrade.pipelines.ledger import experiment_verdict
from autotrade.pipelines.skills import (
    CURATED_MEMORY_SOURCE,
    DEFAULT_OPERATING_MEMORY,
    GRADUATED_EXCLUSIONS_PATH,
    MAX_SKILLS,
    MAX_SKILLS_BYTES,
    MAX_SKILLS_FILES,
    OPERATING_MEMORY_LIBRARY,
    SKILL_FILENAME,
    SkillsStats,
    build_skills_index,
    graduated_exclusion_record,
    graduated_memory_sources,
    latest_skills_snapshot,
    read_graduated_exclusions,
    resolve_operating_memory,
    validate_skill_name,
    validate_skill_path,
    validate_skills_tree,
    write_graduated_exclusions,
)

from .public_identity import PublicIdentity
from .registry import read_ledger_records, resolve_experiment_dir, test_results_revealed

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
_UNREADABLE_MANIFEST = "run manifest is unreadable"


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
    """One curated entry's full ``SKILL.md`` text, read on demand.

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


def graduated_tier(repo_root: Path, experiments_root: Path) -> dict[str, object]:
    """Every experiment's held-out verdict, and what the tier would admit now.

    The verdict follows the console's own reveal gate: an experiment that has
    not revealed its held-out results publishes none here either, or this page
    would hand back exactly the evidence the gate seals. Admission is whatever
    ``skills.graduated_memory_sources`` returns, never a second rule — including
    the researcher's exclusion list, which is why the repository root is read
    here too and why each row also carries what was withdrawn from it.
    """

    root = Path(experiments_root)
    payload: dict[str, object] = {
        "exclusions": GRADUATED_EXCLUSIONS_PATH,
        "experiments": [],
    }
    if not root.is_dir():
        return payload
    admitted: dict[str, list[str]] | None
    withdrawn: dict[str, list[dict[str, str]]] = {}
    try:
        admitted = {
            source.source: list(source.entries)
            for source in graduated_memory_sources(root, repo_root=repo_root)
        }
        for item in read_graduated_exclusions(repo_root):
            withdrawn.setdefault(item.experiment_id, []).append(
                {
                    "skill": item.skill,
                    "reason": item.reason,
                    "excluded_at": item.excluded_at,
                }
            )
    except (OSError, ValueError) as exc:
        # A tier that cannot be resolved is what a session starting now would
        # also hit. Report it, and leave every row's admission unknown rather
        # than printing a "not admitted" the read model cannot stand behind.
        payload["error"] = _error(exc, _UNREADABLE_TIER)
        admitted = None
        withdrawn = {}
    payload["experiments"] = [
        _tier_row(directory, admitted, withdrawn.get(directory.name, []))
        for directory in sorted(root.iterdir(), key=lambda path: path.name)
        if directory.is_dir() and not directory.name.startswith(".")
    ]
    return payload


def _tier_row(
    directory: Path,
    admitted: Mapping[str, list[str]] | None,
    excluded: list[dict[str, str]],
) -> dict[str, object]:
    row: dict[str, object] = {
        "experiment_id": directory.name,
        "revealed": False,
        "verdict": None,
        "admitted": False,
        "entries": [],
        # Researcher-authored metadata, not held-out evidence: it stays visible
        # before the reveal so a withdrawal can always be undone.
        "excluded": excluded,
    }
    try:
        records = read_ledger_records(directory)
        revealed = test_results_revealed(directory, records)
    except (OSError, TypeError, ValueError) as exc:
        row["error"] = _error(exc, _UNREADABLE_EXPERIMENT)
        return row
    row["revealed"] = revealed
    if not revealed:
        return row
    verdict = experiment_verdict(records, strict=False)
    row["verdict"] = str(verdict["status"]) if isinstance(verdict, Mapping) else None
    if admitted is None:
        row["admitted"] = None
        return row
    row["admitted"] = directory.name in admitted
    row["entries"] = list(admitted.get(directory.name, ()))
    return row


def experiment_memory(experiments_root: Path, experiment_id: str) -> dict[str, object]:
    """What each of one experiment's collected sessions actually mounted."""

    directory = resolve_experiment_dir(Path(experiments_root), experiment_id)
    identity = PublicIdentity(directory)
    params = read_json(directory / HITL_DIR_NAME / PARAMS_NAME)
    # Collected runs are named by run id, so their directory order is arbitrary.
    # The plan is the only order a reader can follow, and the page lists one row
    # per session, so the sessions come back in it.
    order = {
        str(entry.get("_raw_key") or ""): index
        for index, entry in enumerate(identity.sessions)
    }
    collected = [
        _mounted_session(identity, path, order)
        for path in sorted(
            (directory / "artifacts").glob(f"*/{HOST_RUN_MANIFEST_NAME}")
        )
    ]
    collected.sort(key=lambda item: item[0])
    return {
        "experiment_id": experiment_id,
        "mode": resolve_operating_memory(params.get("operating_memory")),
        "default_mode": DEFAULT_OPERATING_MEMORY,
        "sessions": [entry for _index, entry in collected],
    }


def _mounted_session(
    identity: PublicIdentity, manifest_path: Path, order: Mapping[str, int]
) -> tuple[int, dict[str, object]]:
    unplanned = len(order)
    try:
        manifest = read_json(manifest_path)
    except (OSError, TypeError, ValueError) as exc:
        return unplanned, {
            "run_ref": identity.run_ref(manifest_path.parent.name),
            "error": _error(exc, _UNREADABLE_MANIFEST),
        }
    entry: dict[str, object] = {
        "run_ref": identity.run_ref(
            str(manifest.get("run_id") or "") or manifest_path.parent.name
        ),
        "kind": str(manifest.get("kind") or ""),
    }
    raw_session = str(manifest.get("session_key") or "")
    index = order.get(raw_session.split("#", 1)[0], unplanned)
    if raw_session:
        try:
            entry["session_key"] = identity.public_session_key(raw_session)
            entry["session_label"] = identity.session_display_key(raw_session)
        except (KeyError, ValueError):
            # A rollback or rerun can leave a collected run whose session the
            # current plan no longer names; the run reference still identifies it.
            pass
    record = manifest.get("operating_memory")
    if not isinstance(record, Mapping):
        return index, {**entry, "mode": None, "sources": []}
    sources = identity.public_value(record.get("sources"))
    return index, {
        **entry,
        "mode": str(record.get("mode") or ""),
        "sources": sources if isinstance(sources, list) else [],
    }


# ---- curated library writes ----------------------------------------------

# Every write says this, because it is the one thing the researcher cannot see
# in the resulting listing: the mount happens at session start.
CURATED_WRITE_NOTE = (
    "the library is mounted at session start, so this applies to sessions "
    "started afterwards; running sessions keep the read-only copy they mounted"
)


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


def _live_conflict(
    experiments_root: Path, live_experiments: Iterable[str], name: str
) -> str:
    """The running experiment whose own skill this entry name would shadow.

    Mounted memory is read-only and a session may not shadow it under its own
    name (``skills._reject_operating_memory_name``), so a curated entry named
    after a live experiment's own skill would stop that experiment's next
    session from maintaining it.
    """

    for experiment_id in live_experiments:
        directory = Path(experiments_root) / experiment_id
        try:
            snapshot = latest_skills_snapshot(
                read_ledger_records(directory), experiment_dir=directory
            )
        except (OSError, ValueError):
            # An experiment whose own skills pointer cannot be resolved cannot
            # start the session that would mount this library either.
            continue
        if snapshot.root is not None and (snapshot.root / name).is_dir():
            return experiment_id
    return ""


def _reject_live_conflict(
    experiments_root: Path | None, live_experiments: Iterable[str], name: str
) -> None:
    """An empty roster is the only no-op: nothing runs, so nothing is shadowed.

    A roster without a root is a caller mistake, not a check to skip quietly.
    """

    running = list(live_experiments)
    if not running:
        return
    if experiments_root is None:
        raise ValueError("checking running experiments needs an experiments root")
    conflict = _live_conflict(experiments_root, running, name)
    if conflict:
        raise ValueError(
            f"{name} is also a skill of the running experiment {conflict}, whose "
            "next session could no longer maintain its own skill under that "
            "name; use a different entry name"
        )


def _write_result(repo_root: Path, name: str, action: str) -> dict[str, object]:
    return {
        "name": name,
        "action": action,
        "note": CURATED_WRITE_NOTE,
        "curated": curated_library(repo_root),
    }


def create_curated_entry(
    repo_root: Path,
    name: str,
    content: str,
    *,
    experiments_root: Path | None = None,
    live_experiments: Iterable[str] = (),
) -> dict[str, object]:
    """Add one entry written by the researcher."""

    entry = _entry_name(name)
    _reject_live_conflict(experiments_root, live_experiments, entry)
    _install_entry(
        _library_dir(repo_root),
        entry,
        {SKILL_FILENAME: str(content).encode("utf-8")},
        replace=False,
    )
    return _write_result(repo_root, entry, "created")


def update_curated_entry(repo_root: Path, name: str, content: str) -> dict[str, object]:
    """Replace one entry's ``SKILL.md``; its other files are carried over.

    No name conflict can appear here: the entry is already mounted under this
    name, so whatever it shadows it shadowed before the edit.
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


def _admitted_skill_dir(
    repo_root: Path, experiments_root: Path, experiment_id: str, skill: str
) -> Path:
    """The only place a promotion may copy from: an admitted graduated skill.

    Admission is ``skills.graduated_memory_sources`` and the reveal gate is the
    console's own, exactly as :func:`graduated_tier` shows them, so the console
    cannot promote from a candidate the page does not offer — nor turn this
    route into a side channel on a held-out verdict that is still sealed.
    """

    root = Path(experiments_root)
    directory = resolve_experiment_dir(root, experiment_id)
    if not test_results_revealed(directory, read_ledger_records(directory)):
        raise KeyError(f"{experiment_id} has not revealed its held-out results")
    source = next(
        (
            item
            for item in graduated_memory_sources(root, repo_root=repo_root)
            if item.source == experiment_id
        ),
        None,
    )
    if source is None or skill not in source.entries:
        raise KeyError(f"{experiment_id} admits no skill named {skill}")
    return source.root / skill


def graduated_entry(
    repo_root: Path, experiments_root: Path, experiment_id: str, skill: str
) -> dict[str, object]:
    """One admitted graduated skill's body, read where it already lives.

    The same admission and reveal gate as the promotion it precedes, so the
    console can never show a candidate it could not copy — and a skill whose
    experiment is no longer admitted reads as unknown rather than as a body the
    tier would not mount.
    """

    name = validate_skill_name(skill)
    item = _admitted_skill_dir(repo_root, Path(experiments_root), experiment_id, name)
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
    live_experiments: Iterable[str] = (),
) -> dict[str, object]:
    """Copy one admitted graduated skill into the curated library verbatim.

    The whole item is copied, ``scripts/`` and ``references/`` included, because
    that is what the mount would have carried and what the researcher then edits
    down in place.
    """

    entry = _entry_name(name or skill)
    source = _admitted_skill_dir(
        repo_root, experiments_root, experiment_id, validate_skill_name(skill)
    )
    _reject_live_conflict(experiments_root, live_experiments, entry)
    files: dict[str, bytes] = {}
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source).as_posix()
        if path.is_symlink():
            raise ValueError(f"skill symlink is forbidden: {relative}")
        if path.is_file():
            files[relative] = path.read_bytes()
    _install_entry(_library_dir(repo_root), entry, files, replace=False)
    return _write_result(repo_root, entry, "promoted")


# ---- graduated exclusions -------------------------------------------------

# Graduated skills are another experiment's immutable artifacts: the console
# never edits one, it only stops mounting it.
GRADUATED_EXCLUSION_NOTE = (
    "the withdrawn skill is left untouched where it was published; this changes "
    "what sessions started afterwards mount, and the exclusion list is a tracked "
    "repository file the researcher commits"
)


def _tier_result(
    repo_root: Path,
    experiments_root: Path,
    experiment_id: str,
    skill: str,
    action: str,
) -> dict[str, object]:
    return {
        "experiment_id": experiment_id,
        "skill": skill,
        "action": action,
        "note": GRADUATED_EXCLUSION_NOTE,
        "graduated": graduated_tier(repo_root, experiments_root),
    }


def exclude_graduated_skill(
    repo_root: Path,
    experiments_root: Path,
    *,
    experiment_id: str,
    skill: str,
    reason: str = "",
) -> dict[str, object]:
    """Keep one currently admitted graduated skill out of every future mount."""

    entry = graduated_exclusion_record(experiment_id, skill, reason)
    current = list(read_graduated_exclusions(repo_root))
    # Asked before admission: an already withdrawn skill is no longer admitted,
    # and "already excluded" is the answer that says what happened.
    if any(item.key == entry.key for item in current):
        raise FileExistsError(
            f"{entry.skill} is already excluded for {entry.experiment_id}"
        )
    # Otherwise only a skill the tier admits right now: the console cannot
    # withdraw something it would not have mounted, nor name one that is absent.
    _admitted_skill_dir(repo_root, experiments_root, entry.experiment_id, entry.skill)
    write_graduated_exclusions(repo_root, [*current, entry])
    return _tier_result(
        repo_root, experiments_root, entry.experiment_id, entry.skill, "excluded"
    )


def restore_graduated_skill(
    repo_root: Path, experiments_root: Path, *, experiment_id: str, skill: str
) -> dict[str, object]:
    """Put one withdrawn skill back into the tier."""

    name = validate_skill_name(skill)
    current = list(read_graduated_exclusions(repo_root))
    remaining = [item for item in current if item.key != (experiment_id, name)]
    if len(remaining) == len(current):
        raise KeyError(f"{name} is not excluded for {experiment_id}")
    write_graduated_exclusions(repo_root, remaining)
    return _tier_result(repo_root, experiments_root, experiment_id, name, "restored")


def memory_overview(repo_root: Path, experiments_root: Path) -> dict[str, object]:
    """The whole operating-memory page in one read."""

    return {
        "default_mode": DEFAULT_OPERATING_MEMORY,
        "curated": curated_library(repo_root),
        "graduated": graduated_tier(repo_root, experiments_root),
    }


__all__ = [
    "CURATED_WRITE_NOTE",
    "GRADUATED_EXCLUSION_NOTE",
    "create_curated_entry",
    "curated_entry",
    "curated_library",
    "delete_curated_entry",
    "exclude_graduated_skill",
    "experiment_memory",
    "graduated_entry",
    "graduated_tier",
    "memory_overview",
    "promote_curated_entry",
    "restore_graduated_skill",
    "update_curated_entry",
]
