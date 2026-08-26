"""Step-tree console view: de-opaqued lineage, node metrics, and source export.

The agent-visible tree stores fold ids as opaque ``fold_ref_*`` tokens (the raw
label encodes the calendar period). The console is the researcher's trusted
surface, so this module recomputes the ref for every known fold id (schedule +
ledger) and maps the tokens back for display. Frozen markers come from the
ledger's fold records: the node a fold selected is the artifact that fold
shipped.
"""

from __future__ import annotations

from pathlib import Path

from autotrade.environment.identity import LegacyExperimentError
from autotrade.environment.step_tree import NODE_OUTPUT_DIR, StepTree

from .public_identity import PublicIdentity, schedule_period_label
from .registry import latest_fold_records, read_ledger_records


def _has_snapshot(steps_root: Path, node_id: str) -> bool:
    """Whether the node dir holds a strategy snapshot (output/ tree + optional
    models/); failed attempts record no snapshot."""
    return (steps_root / node_id / NODE_OUTPUT_DIR).is_dir()


def fold_sessions(identity: PublicIdentity) -> list[dict[str, object]]:
    """Ordered public Fold sessions from the validated experiment plan."""

    return [
        identity.public_session(session, heldout_revealed=False)
        for session in identity.sessions
        if session.get("kind") == "fold"
    ]


def public_step_node(
    node: dict[str, object], *, identity: PublicIdentity | None = None
) -> dict[str, object]:
    """Project one Step node; modern identities use the central boundary."""

    if identity is not None:
        return identity.public_record(node, heldout_revealed=False)
    public = dict(node)
    fold_ref = public.pop("fold_id", None)
    run_ref = public.pop("run_id", None)
    revision_ref = public.pop("revision_id", None)
    if fold_ref:
        public["fold_ref"] = fold_ref
    if run_ref:
        public["run_ref"] = run_ref
    if revision_ref:
        public["strategy_ref"] = revision_ref
    return public


def step_tree_view(experiment_dir: Path) -> dict[str, object]:
    experiment_dir = Path(experiment_dir)
    tree = StepTree(experiment_dir / "steps")
    records = read_ledger_records(experiment_dir)
    tree_nodes = tree.nodes()
    try:
        identity: PublicIdentity | None = PublicIdentity(experiment_dir)
    except LegacyExperimentError:
        identity = None
    # The selected Step of a fold record IS the node that fold froze. Legacy
    # trees remain auditable but never guess a raw schedule identity.
    frozen_for: dict[str, list[str]] = {}
    if identity is not None:
        for (epoch_id, fold_id), record in latest_fold_records(records).items():
            selected = record.get("selected_step_id")
            if selected:
                frozen_for.setdefault(str(selected), []).append(
                    f"{epoch_id}/{schedule_period_label(fold_id)}"
                )

    nodes: list[dict[str, object]] = []
    for node in tree_nodes:
        node_id = str(node["node_id"])
        fold_ref = str(node.get("fold_id") or "")
        raw_metrics = node.get("metrics")
        raw_attachments = node.get("attachments")
        nodes.append(
            public_step_node(
                {
                    "node_id": node_id,
                    "parent_node_id": node.get("parent_node_id"),
                    "epoch_id": node.get("epoch_id"),
                    "fold_id": fold_ref,
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
                    "frozen_for": sorted(frozen_for.get(node_id, [])),
                    "is_current": node_id == tree.current_node_id,
                },
                identity=identity,
            )
        )
    return {
        "current_node_id": tree.current_node_id,
        "nodes": nodes,
        "fold_sessions": fold_sessions(identity) if identity is not None else [],
        "identity_status": "modern" if identity is not None else "legacy_read_only",
        "identity_available": identity is not None,
    }


def node_export_dir(experiment_dir: Path, node_id: str) -> Path:
    """Validated node directory for the source.zip download (never a raw path join)."""
    return node_export_dir_from_root(Path(experiment_dir) / "steps", node_id)


def node_export_dir_from_root(steps_root: Path, node_id: str) -> Path:
    """Validated export directory for either a collected or live step tree."""
    tree = StepTree(steps_root)
    node = tree.get_node(node_id)  # raises ValueError for unknown ids
    if node.get("status") == "failed" or not node.get("complete_validation"):
        raise ValueError(f"step node {node_id} is a failed attempt without a snapshot")
    if not _has_snapshot(tree.root, str(node["node_id"])):
        raise ValueError(f"step node snapshot is missing on disk: {node_id}")
    return tree.root / str(node["node_id"])


def current_node_export_dir(steps_root: Path) -> tuple[str, Path]:
    """Return the current validated Step snapshot from a live tree."""
    tree = StepTree(steps_root)
    node_id = tree.current_node_id
    if not node_id:
        raise ValueError("live fold has no current validated step snapshot")
    return node_id, node_export_dir_from_root(tree.root, node_id)
