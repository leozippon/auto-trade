"""Operating-memory read model for the console.

Read-only over the two places the pipeline already treats as authoritative: the
curated library committed under ``configs/operating_memory/`` and the experiment
trees under ``experiments/``. Admission to the graduated tier is never
recomputed here — ``pipelines.skills`` answers it — so the console can only show
what a session starting now would really mount, and what past sessions did
mount, according to their run manifests.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from autotrade.pipelines.hitl_state import HITL_DIR_NAME, PARAMS_NAME, read_json
from autotrade.pipelines.ledger import experiment_verdict
from autotrade.pipelines.skills import (
    CURATED_MEMORY_SOURCE,
    DEFAULT_OPERATING_MEMORY,
    OPERATING_MEMORY_LIBRARY,
    SKILL_FILENAME,
    build_skills_index,
    graduated_memory_sources,
    resolve_operating_memory,
    validate_skill_name,
)

from .public_identity import PublicIdentity, redact_host_paths
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
    return {**listed, "content": redact_host_paths(content)}


def graduated_tier(experiments_root: Path) -> dict[str, object]:
    """Every experiment's held-out verdict, and what the tier would admit now.

    The verdict follows the console's own reveal gate: an experiment that has
    not revealed its held-out results publishes none here either, or this page
    would hand back exactly the evidence the gate seals. Admission is whatever
    ``skills.graduated_memory_sources`` returns, never a second rule.
    """

    root = Path(experiments_root)
    payload: dict[str, object] = {"experiments": []}
    if not root.is_dir():
        return payload
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
        "revealed": False,
        "verdict": None,
        "admitted": False,
        "entries": [],
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
    return {
        "experiment_id": experiment_id,
        "mode": resolve_operating_memory(params.get("operating_memory")),
        "default_mode": DEFAULT_OPERATING_MEMORY,
        "sessions": [
            _mounted_session(identity, path)
            for path in sorted(
                (directory / "artifacts").glob(f"*/{HOST_RUN_MANIFEST_NAME}")
            )
        ],
    }


def _mounted_session(identity: PublicIdentity, manifest_path: Path) -> dict[str, object]:
    try:
        manifest = read_json(manifest_path)
    except (OSError, TypeError, ValueError) as exc:
        return {
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
        return {**entry, "mode": None, "sources": []}
    sources = identity.public_value(record.get("sources"))
    return {
        **entry,
        "mode": str(record.get("mode") or ""),
        "sources": sources if isinstance(sources, list) else [],
    }


def memory_overview(repo_root: Path, experiments_root: Path) -> dict[str, object]:
    """The whole operating-memory page in one read."""

    return {
        "default_mode": DEFAULT_OPERATING_MEMORY,
        "curated": curated_library(repo_root),
        "graduated": graduated_tier(experiments_root),
    }


__all__ = [
    "curated_entry",
    "curated_library",
    "experiment_memory",
    "graduated_tier",
    "memory_overview",
]
