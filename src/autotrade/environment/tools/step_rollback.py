"""Restore a fully evaluated Step revision into the Agent work copy."""

from __future__ import annotations

import shutil
from collections.abc import Mapping
from pathlib import Path

from autotrade.environment.artifacts import restore_working_artifacts_writable
from autotrade.environment.step_tree import StepTree, node_in_session

from .base import ToolError, ToolResult, ToolSpec


class StepRollbackTool:
    spec = ToolSpec(
        "step_rollback",
        "Restore one fully evaluated Step revision and branch from it.",
        {
            "type": "object",
            "properties": {"node_id": {"type": "string", "minLength": 1, "maxLength": 500}},
            "required": ["node_id"],
            "additionalProperties": False,
        },
        mutating=True,
    )

    def __init__(
        self,
        tree: StepTree,
        output_dir: str | Path,
        models_dir: str | Path | None = None,
        *,
        fold_id: str,
        run_id: str,
    ) -> None:
        self.tree = tree
        self.output_dir = Path(output_dir)
        self.models_dir = Path(models_dir) if models_dir is not None else None
        self.fold_id = fold_id
        self.run_id = run_id

    def invoke(self, arguments: Mapping[str, object]) -> ToolResult:
        node_id = str(arguments["node_id"])
        try:
            node = self.tree.get_node(node_id)
        except ValueError as exc:
            # Same shaping as finish_fold: a typed tool error carries an
            # error_type the model can act on, an escaping ValueError does not.
            raise ToolError("step_rollback cannot restore an absent Step") from exc
        # Same session rule as finish_fold: the tree carries earlier Folds'
        # nodes as read-only evidence, and restoring one would rebase this
        # Fold's work copy and lineage onto an artifact it may not select.
        if not node_in_session(node, fold_id=self.fold_id, run_id=self.run_id):
            raise ToolError(
                "step_rollback can restore only a Step from the current Fold "
                "session; another Fold's or run's node is evidence only"
            )
        if not node.get("complete_validation") or not node.get("revision_id"):
            raise ToolError("only a fully evaluated Step revision can be restored")
        _replace_tree(self.tree.node_output_dir(node_id), self.output_dir)
        if self.models_dir is not None:
            source_models = self.tree.node_models_dir(node_id)
            if source_models.exists():
                _replace_tree(source_models, self.models_dir)
            else:
                _empty_tree(self.models_dir)
        restore_working_artifacts_writable(self.output_dir, self.models_dir)
        self.tree.set_position(node_id)
        return ToolResult(True, value={"node_id": node_id, "revision_id": str(node["revision_id"])})


def _empty_tree(target: Path) -> None:
    if target.exists():
        for child in target.iterdir():
            shutil.rmtree(child) if child.is_dir() and not child.is_symlink() else child.unlink()
    else:
        target.mkdir(parents=True)


def _replace_tree(source: Path, target: Path) -> None:
    if not source.is_dir():
        raise ToolError(f"missing Step revision directory: {source.name}")
    _empty_tree(target)
    for child in source.iterdir():
        destination = target / child.name
        if child.is_dir() and not child.is_symlink():
            shutil.copytree(child, destination)
        else:
            shutil.copy2(child, destination)


__all__ = ["StepRollbackTool"]
