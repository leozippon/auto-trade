"""Step artifact tree: lineage of successfully validated Step artifacts.

Every successful formal validation backtest snapshots the current ``output/``
directory plus optional ``models/`` directory into
``/mnt/artifacts/steps/<node_id>/{output,models}/`` (validation attachments
such as ``detailed_return.json`` sit at the node root) and appends a node
(with a parent pointer) to ``steps/tree.json``. The tree accumulates across
Folds of one Experiment: the Pipeline hands it to the next Fold's sandbox and
positions ``current_node_id`` at the parent artifact, so the Agent can read
where it stands in the search history and branch from any validated node via
the ``step_rollback`` tool. The feature is toggleable for ablations
(``step_tree_enabled``).

One ``daily_backtest`` appends one node under the current position;
``batch_validate`` repositions the tree between records so its candidates
become siblings of one parent rather than a chain, which is what makes their
numbers comparable.
"""

from __future__ import annotations

import json
import os
import shutil
import uuid
from collections.abc import Mapping
from pathlib import Path

from autotrade.environment.artifacts import copy_artifact, copy_model_artifacts
from autotrade.environment.runtime import chmod_tree, sanitize_for_log, utc_now_iso, write_json_atomic

TREE_FILE = "tree.json"
# Node subdirectories reserved for the snapshot itself; attachments must not
# shadow them (attachment files live at the node root next to these).
NODE_OUTPUT_DIR = "output"
NODE_MODELS_DIR = "models"


def node_in_session(node: Mapping[str, object], *, fold_id: str, run_id: str) -> bool:
    """Whether a node was produced by this Fold session (fold plus run identity).

    Single source for the Agent-facing rule that only the current Fold's,
    current run's nodes can be restored or submitted; the nodes this tree
    carries in from earlier Folds and earlier runs are read-only evidence.
    """

    return node.get("fold_id") == fold_id and node.get("run_id") == run_id


class StepTree:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.tree_path = self.root / TREE_FILE
        if self.tree_path.exists():
            self.data = json.loads(self.tree_path.read_text(encoding="utf-8"))
        else:
            self.data = {"current_node_id": None, "nodes": []}

    # ---- mutation (Environment/Pipeline only; the Agent reads) ----

    def record_step(
        self,
        output_root: str | Path,
        *,
        epoch_id: str,
        fold_id: str,
        run_id: str,
        result_name: str,
        revision_id: str,
        metrics: dict[str, object],
        models_root: str | Path | None = None,
        attachments: dict[str, str | Path] | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> str:
        # result_name (valid_NNN) is only unique within one run's results dir;
        # the same fold re-executed (rerun_fold / post-rollback) starts again at
        # valid_000, so the run id must be part of the node identity.
        node_id = f"{epoch_id}__{fold_id}__{run_id}__{result_name}"
        if any(node["node_id"] == node_id for node in self.data["nodes"]):
            raise ValueError(f"step tree node already exists: {node_id}")
        node_dir = self.root / node_id
        copy_artifact(output_root, node_dir / NODE_OUTPUT_DIR)
        if models_root is not None and Path(models_root).exists():
            copy_model_artifacts(models_root, node_dir / NODE_MODELS_DIR)
        copied_attachments: dict[str, str] = {}
        for relpath, source in (attachments or {}).items():
            rel = Path(relpath)
            if rel.is_absolute() or ".." in rel.parts or not rel.parts:
                raise ValueError(f"invalid step attachment path: {relpath}")
            if rel.parts[0] in (NODE_OUTPUT_DIR, NODE_MODELS_DIR):
                raise ValueError(f"step attachment must not shadow the snapshot dirs: {relpath}")
            target = node_dir / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(Path(source), target)
            # tree.json is Agent-readable; store a node-relative reference,
            # never the host's absolute workspace path.
            copied_attachments[str(rel)] = str(Path(node_id) / rel)
        # A recorded node is immutable evidence: lock the snapshot directories so
        # a later Fold cannot silently edit the artifact it branched from.
        chmod_tree(node_dir / NODE_OUTPUT_DIR, file_mode=0o444, dir_mode=0o555)
        if (node_dir / NODE_MODELS_DIR).is_dir():
            chmod_tree(node_dir / NODE_MODELS_DIR, file_mode=0o444, dir_mode=0o555)
        self.data["nodes"].append(
            {
                "node_id": node_id,
                "parent_node_id": self.data.get("current_node_id"),
                "epoch_id": epoch_id,
                "fold_id": fold_id,
                "run_id": run_id,
                "result_name": result_name,
                "revision_id": revision_id,
                # A recorded Step is by construction one completed full-window
                # Validation; record_failed_attempt writes the False rows.
                "complete_validation": True,
                "metrics": metrics,
                "attachments": copied_attachments,
                "created_at": utc_now_iso(),
                # Why this node exists, when the caller knows: a batch records
                # the pre-registered hypothesis and its batch id here so the
                # fan-out stays readable long after the conversation is gone.
                **({"metadata": dict(metadata)} if metadata else {}),
            }
        )
        self.data["current_node_id"] = node_id
        self.save()
        return node_id

    def record_failed_attempt(
        self,
        *,
        epoch_id: str,
        fold_id: str,
        run_id: str,
        result_name: str,
        error: str,
        metrics: dict[str, object] | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> str:
        """Append a lightweight failed-attempt node so later folds see dead-ends.

        Unlike ``record_step`` this copies no ``output/`` snapshot and, crucially,
        leaves ``current_node_id`` unchanged: a failed attempt must never become
        the working position or a parent candidate.
        """
        node_id = f"{epoch_id}__{fold_id}__{run_id}__{result_name}"
        if any(node["node_id"] == node_id for node in self.data["nodes"]):
            raise ValueError(f"step tree node already exists: {node_id}")
        self.data["nodes"].append(
            {
                "node_id": node_id,
                "parent_node_id": self.data.get("current_node_id"),
                "epoch_id": epoch_id,
                "fold_id": fold_id,
                "run_id": run_id,
                "result_name": result_name,
                "revision_id": None,
                "complete_validation": False,
                "status": "failed",
                "error": error,
                "metrics": metrics or {},
                "attachments": {},
                "created_at": utc_now_iso(),
                # Same shape as a recorded Step: a batch's dead end keeps its
                # batch id and hypothesis next to its siblings.
                **({"metadata": dict(metadata)} if metadata else {}),
            }
        )
        self.save()
        return node_id

    def set_position(self, node_id: str | None) -> None:
        if node_id is not None and not any(n["node_id"] == node_id for n in self.data["nodes"]):
            raise ValueError(f"unknown step tree node: {node_id}")
        self.data["current_node_id"] = node_id
        self.save()

    def position_for_step(self, step_id: str) -> str | None:
        """Most recent validated node carrying the given step id.

        A frozen artifact records the node it was frozen from, so the branch
        point is resolved by node identity. Returns ``None`` when the node is
        not in this tree — an inherited seed, or a node a rollback pruned —
        so such a parent starts a new root instead of failing the fold.
        """
        for node in reversed(self.data["nodes"]):
            if not node.get("complete_validation"):
                continue
            if str(node.get("node_id")) == step_id:
                return str(node["node_id"])
        return None

    def save(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        safe_data = sanitize_for_log(self.data)
        # Write through a NEW inode (tmp + rename): the fold-level tree is a
        # hardlinked copy of the experiment-level tree, so an in-place write
        # would mutate the experiment copy mid-fold and an aborted fold could
        # leave it referencing node snapshots that were never copied back.
        write_json_atomic(self.tree_path, safe_data)
        # Always refresh the human/Agent-readable rendering alongside the JSON.
        self._write_text_atomic(self.root / "tree.txt", self.render_ascii() + "\n")

    @staticmethod
    def _write_text_atomic(path: Path, content: str) -> None:
        tmp = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
        try:
            tmp.write_text(content, encoding="utf-8")
            tmp.replace(path)
        finally:
            tmp.unlink(missing_ok=True)

    # ---- read views ----

    @property
    def current_node_id(self) -> str | None:
        return self.data.get("current_node_id")

    def nodes(self) -> list[dict[str, object]]:
        return list(self.data["nodes"])

    def get_node(self, node_id: str) -> dict[str, object]:
        for node in self.data["nodes"]:
            if node["node_id"] == node_id:
                return node
        raise ValueError(f"unknown step tree node: {node_id}")

    def node_output_dir(self, node_id: str) -> Path:
        return self.root / node_id / NODE_OUTPUT_DIR

    def node_models_dir(self, node_id: str) -> Path:
        return self.root / node_id / NODE_MODELS_DIR

    def render_ascii(self) -> str:
        """Human/Agent-readable tree with the current position marked."""
        children: dict[str | None, list[dict[str, object]]] = {}
        for node in self.data["nodes"]:
            children.setdefault(node.get("parent_node_id"), []).append(node)
        lines: list[str] = []

        def walk(parent_id: str | None, depth: int) -> None:
            for node in children.get(parent_id, []):
                marker = "  <- current" if node["node_id"] == self.data.get("current_node_id") else ""
                # Mark only genuine failed attempts (record_failed_attempt sets
                # status="failed"); a partial/debug validation node is not a failure.
                failed_text = " [failed]" if node.get("status") == "failed" else ""
                metrics = node.get("metrics", {})
                parts = []
                for key, label in (("total_return", "ret"), ("sharpe", "sharpe")):
                    value = metrics.get(key)
                    if isinstance(value, (int, float)):
                        parts.append(f"{label}={value:.4f}")
                metrics_text = f" {' '.join(parts)}" if parts else ""
                lines.append(f"{'  ' * depth}- {node['node_id']}{metrics_text}{failed_text}{marker}")
                walk(str(node["node_id"]), depth + 1)

        walk(None, 0)
        return "\n".join(lines) if lines else "(empty step tree)"
