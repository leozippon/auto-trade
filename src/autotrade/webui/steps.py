"""Step-tree console view: lineage, node metrics and source export.

The Agent-visible tree stores the session as an opaque ``session_ref_*`` token.
The console is the researcher's trusted surface, so it resolves the token back
to the plan key (``research``) for display, and marks the node the experiment registered
from the ledger's registered record.
"""

from __future__ import annotations

from pathlib import Path

from autotrade.environment.step_tree import NODE_OUTPUT_DIR, StepTree
from autotrade.pipelines.ledger import registered_record

from .public_identity import PublicIdentity
from .registry import read_ledger_records


def _has_snapshot(steps_root: Path, node_id: str) -> bool:
    """Whether the node dir holds a strategy snapshot (output/ tree + optional
    models/); failed attempts record no snapshot."""
    return (steps_root / node_id / NODE_OUTPUT_DIR).is_dir()


def step_tree_view(experiment_dir: Path) -> dict[str, object]:
    experiment_dir = Path(experiment_dir)
    tree = StepTree(experiment_dir / "steps")
    identity = PublicIdentity(experiment_dir)
    registered_step = None
    row = registered_record(read_ledger_records(experiment_dir))
    if row is not None:
        registered_step = row["registered"].get("source_step_id")  # type: ignore[union-attr]

    nodes: list[dict[str, object]] = []
    for node in tree.nodes():
        node_id = str(node["node_id"])
        session_ref = str(node.get("session_ref") or "")
        raw_metrics = node.get("metrics")
        raw_attachments = node.get("attachments")
        public = identity.public_record(
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
                "registered": node_id == registered_step,
                "is_current": node_id == tree.current_node_id,
            }
        )
        if session_ref:
            try:
                public["session_key"] = identity.store.resolve("session", session_ref)
            except (KeyError, ValueError):
                public["session_key"] = None
        nodes.append(public)
    return {"current_node_id": tree.current_node_id, "nodes": nodes}


def node_export_dir(experiment_dir: Path, node_id: str) -> Path:
    """Validated node directory for the source.zip download (never a raw path join)."""
    tree = StepTree(Path(experiment_dir) / "steps")
    node = tree.get_node(node_id)  # raises ValueError for unknown ids
    if node.get("status") == "failed" or not node.get("complete_validation"):
        raise ValueError(f"step node {node_id} is a failed attempt without a snapshot")
    if not _has_snapshot(tree.root, str(node["node_id"])):
        raise ValueError(f"step node snapshot is missing on disk: {node_id}")
    return tree.root / str(node["node_id"])

