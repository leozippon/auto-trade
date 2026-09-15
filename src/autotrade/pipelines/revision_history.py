"""Host-side read view over one experiment's retained strategy revisions.

Every Validation's revision is kept, so ``artifacts/strategy`` holds the arm's
whole artifact lineage rather than a working set. This is what the research
console reads: the revisions with their parent pointers, joined to the Step node
each was validated as, plus the diff between any two of them. Host-only — raw
revision ids and node ids are resolved here and projected at the Web boundary,
never handed to the Agent.
"""

from __future__ import annotations

import json
from pathlib import Path

from autotrade.environment.artifacts import FilesystemArtifactStore

from .session_resume import STEP_SIDECAR_DIR


def revision_store(experiment_dir: str | Path) -> FilesystemArtifactStore:
    """The experiment's strategy artifact store."""

    return FilesystemArtifactStore(Path(experiment_dir) / "artifacts" / "strategy")


def revision_history(experiment_dir: str | Path) -> dict[str, object]:
    """Every retained revision of one experiment, oldest first, with lineage.

    ``node_id`` is the Step node the revision was recorded as, or ``None`` for
    a revision whose Validation replay failed before a node existed. A revision
    written before the store was content-addressed reports ``layout: legacy``
    and no parent, because its lineage was never recorded.
    """

    store = revision_store(experiment_dir)
    nodes = _nodes_by_revision(experiment_dir)
    revisions: list[dict[str, object]] = []
    for revision_id in store.revision_ids():
        manifest = store.revision_manifest(revision_id)
        files: list[dict[str, object]] = manifest["files"]  # type: ignore[assignment]
        revisions.append(
            {
                "revision_id": revision_id,
                "parent_revision_id": manifest["parent_revision_id"],
                "created_at": manifest["created_at"],
                "fingerprint": manifest["fingerprint"],
                "layout": manifest["layout"],
                "node_id": nodes.get(revision_id),
                "file_count": len(files),
                "total_bytes": sum(int(entry["size"]) for entry in files),
            }
        )
    revisions.sort(key=lambda row: (str(row["created_at"] or ""), str(row["revision_id"])))
    return {"revisions": revisions}


def revision_diff(
    experiment_dir: str | Path, revision_a: str, revision_b: str
) -> dict[str, object]:
    """Changed, added and removed paths between two of the arm's revisions."""

    return revision_store(experiment_dir).diff_revisions(revision_a, revision_b)


def _nodes_by_revision(experiment_dir: str | Path) -> dict[str, str]:
    """Which Step node each revision was recorded as, from the host sidecars.

    The Step tree carries only the opaque strategy ref, so the raw mapping lives
    in the host-only sidecar the Pipeline already writes per recorded Validation.
    """

    directory = Path(experiment_dir) / STEP_SIDECAR_DIR
    if not directory.is_dir():
        return {}
    mapping: dict[str, str] = {}
    for path in sorted(directory.glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        mapping[str(record["revision_id"])] = str(record["step_id"])
    return mapping


__all__ = ["revision_diff", "revision_history", "revision_store"]
