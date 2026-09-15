"""Step-tree console view: lineage, node metrics, source export, and revisions.

The Agent-visible tree stores the session as an opaque ``session_ref_*`` token.
The console is the researcher's trusted surface, so it resolves the token back
to the plan key (``research``) for display, and marks the node the arm froze
from the ledger's frozen record.

The same panel reads the arm's retained strategy revisions, which are the
artifact's own lineage beside the tree of validated nodes. Raw revision ids stay
on the host: they are projected as ``strategy_ref``/``parent_strategy_ref`` here
and resolved back here when a diff names two of them.
"""

from __future__ import annotations

from pathlib import Path

from autotrade.environment.identity import LegacyExperimentError
from autotrade.environment.step_tree import NODE_OUTPUT_DIR, StepTree
from autotrade.pipelines.ledger import frozen_record
from autotrade.pipelines.revision_history import revision_diff, revision_history

from .public_identity import PublicIdentity
from .registry import read_ledger_records


def _has_snapshot(steps_root: Path, node_id: str) -> bool:
    """Whether the node dir holds a strategy snapshot (output/ tree + optional
    models/); failed attempts record no snapshot."""
    return (steps_root / node_id / NODE_OUTPUT_DIR).is_dir()


def public_step_node(
    node: dict[str, object], *, identity: PublicIdentity | None = None
) -> dict[str, object]:
    """Project one Step node; modern identities use the central boundary."""

    if identity is not None:
        return identity.public_record(node)
    public = dict(node)
    session_ref = public.pop("session_ref", None)
    run_ref = public.pop("run_id", None)
    revision_ref = public.pop("revision_id", None)
    if session_ref:
        public["session_ref"] = session_ref
    if run_ref:
        public["run_ref"] = run_ref
    if revision_ref:
        public["strategy_ref"] = revision_ref
    return public


def step_tree_view(experiment_dir: Path) -> dict[str, object]:
    experiment_dir = Path(experiment_dir)
    tree = StepTree(experiment_dir / "steps")
    try:
        identity: PublicIdentity | None = PublicIdentity(experiment_dir)
    except LegacyExperimentError:
        identity = None
    frozen_step = None
    if identity is not None:
        frozen = frozen_record(read_ledger_records(experiment_dir))
        if frozen is not None:
            frozen_step = frozen["frozen"].get("source_step_id")  # type: ignore[union-attr]

    nodes: list[dict[str, object]] = []
    for node in tree.nodes():
        node_id = str(node["node_id"])
        session_ref = str(node.get("session_ref") or "")
        raw_metrics = node.get("metrics")
        raw_attachments = node.get("attachments")
        public = public_step_node(
            {
                "node_id": node_id,
                "parent_node_id": node.get("parent_node_id"),
                "session_ref": session_ref,
                "run_id": node.get("run_id"),
                "result_name": node.get("result_name"),
                "complete_validation": bool(node.get("complete_validation")),
                "status": node.get("status"),
                "error": node.get("error"),
                "metrics": dict(raw_metrics) if isinstance(raw_metrics, dict) else {},
                "revision_id": node.get("revision_id"),
                "created_at": node.get("created_at"),
                "attachments": (
                    sorted(raw_attachments) if isinstance(raw_attachments, dict) else []
                ),
                "has_snapshot": _has_snapshot(tree.root, node_id),
                "frozen": node_id == frozen_step,
                "is_current": node_id == tree.current_node_id,
            },
            identity=identity,
        )
        if identity is not None and session_ref:
            try:
                public["session_key"] = identity.store.resolve("session", session_ref)
            except (KeyError, ValueError):
                public["session_key"] = None
        nodes.append(public)
    return {"current_node_id": tree.current_node_id, "nodes": nodes}


def revision_lineage_view(
    experiment_dir: Path, identity: PublicIdentity
) -> dict[str, object]:
    """Every retained revision of the arm, oldest first, past the host boundary.

    ``node_id`` is a Step node id, which the tree already publishes as it is;
    the revision ids are not, so they and the parent pointer become opaque
    strategy refs. A revision the older layout left behind carries ``legacy``
    and no parent, because its lineage was never recorded.
    """

    revisions = []
    for row in revision_history(experiment_dir)["revisions"]:
        parent = row["parent_revision_id"]
        revisions.append(
            {
                "strategy_ref": identity.strategy_ref(row["revision_id"]),
                "parent_strategy_ref": (
                    identity.strategy_ref(parent) if parent else None
                ),
                "created_at": row["created_at"],
                "fingerprint": row["fingerprint"],
                "layout": row["layout"],
                "node_id": row["node_id"],
                "file_count": row["file_count"],
                "total_bytes": row["total_bytes"],
            }
        )
    return {"revisions": revisions}


def revision_diff_view(
    experiment_dir: Path, identity: PublicIdentity, ref_a: str, ref_b: str
) -> dict[str, object]:
    """The diff between two revisions named by their public strategy refs.

    The bodies are the Agent's own strategy files, so nothing in them is
    redacted; the manifests carry relative paths only, so the file entries pass
    through as the store wrote them.
    """

    diff = revision_diff(
        experiment_dir,
        identity.raw_strategy_id(ref_a),
        identity.raw_strategy_id(ref_b),
    )
    return {
        "strategy_ref_a": ref_a,
        "strategy_ref_b": ref_b,
        "added": diff["added"],
        "removed": diff["removed"],
        "modified": diff["modified"],
        "files": diff["files"],
    }


def node_export_dir(experiment_dir: Path, node_id: str) -> Path:
    """Validated node directory for the source.zip download (never a raw path join)."""
    tree = StepTree(Path(experiment_dir) / "steps")
    node = tree.get_node(node_id)  # raises ValueError for unknown ids
    if node.get("status") == "failed" or not node.get("complete_validation"):
        raise ValueError(f"step node {node_id} is a failed attempt without a snapshot")
    if not _has_snapshot(tree.root, str(node["node_id"])):
        raise ValueError(f"step node snapshot is missing on disk: {node_id}")
    return tree.root / str(node["node_id"])

