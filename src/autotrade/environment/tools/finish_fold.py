"""Select one immutable, fully evaluated Step as the Fold result."""

from __future__ import annotations

import ast
from collections.abc import Mapping
from pathlib import Path

from autotrade.environment.step_tree import StepTree

from .base import ToolError, ToolResult, ToolSpec


def executable_source_structure(source: str) -> str:
    """Return the directly comparable executable structure of ``main.py``.

    Comments, module/function docstrings, and whitespace are ignored so a
    comment-only harvest has the parent's structure, while a logic or signal
    change does not.
    """

    tree = ast.parse(source)
    _strip_docstrings(tree)
    return ast.dump(tree, annotate_fields=True, include_attributes=False)


def _tree_bytes(root: Path | None) -> dict[str, bytes]:
    if root is None or not root.is_dir():
        return {}
    files: dict[str, bytes] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            files[path.relative_to(root).as_posix()] = path.read_bytes()
    return files


def _strip_docstrings(tree: ast.AST) -> None:
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        body = node.body
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            node.body = body[1:] or [ast.Pass()]


class FinishFoldTool:
    spec = ToolSpec(
        "finish_fold",
        "Finish this Fold with a fully evaluated Step revision.",
        {
            "type": "object",
            "properties": {"node_id": {"type": "string", "minLength": 1, "maxLength": 500}},
            "required": [],
            "additionalProperties": False,
        },
    )

    def __init__(
        self,
        tree: StepTree,
        *,
        fold_id: str,
        run_id: str,
        parent_main_py: str | Path | None = None,
        current_output: str | Path | None = None,
        current_models: str | Path | None = None,
    ) -> None:
        self.tree = tree
        self.fold_id = fold_id
        self.run_id = run_id
        self._current_output = Path(current_output) if current_output is not None else None
        self._current_models = Path(current_models) if current_models is not None else None
        self._parent_structure: str | None = None
        if parent_main_py is not None:
            path = Path(parent_main_py)
            try:
                self._parent_structure = executable_source_structure(
                    path.read_text(encoding="utf-8")
                )
            except (OSError, SyntaxError) as exc:
                raise ValueError(f"parent strategy structure is invalid: {exc}") from exc

    def invoke(self, arguments: Mapping[str, object]) -> ToolResult:
        node_id = str(arguments.get("node_id") or self.tree.current_node_id or "")
        if not node_id:
            raise ToolError("finish_fold requires a fully evaluated Step")
        try:
            node = self.tree.get_node(node_id)
        except ValueError as exc:
            raise ToolError("finish_fold cannot select an absent Step") from exc
        if node.get("fold_id") != self.fold_id or node.get("run_id") != self.run_id:
            raise ToolError("finish_fold can select only a Step from the current Fold session")
        if not node.get("complete_validation") or not node.get("revision_id"):
            raise ToolError("finish_fold requires successful complete validation")
        nominated_structure = self._node_structure(node_id)
        if self._parent_structure is not None:
            self._require_different_hypothesis(node_id, nominated_structure)
        self._require_current_matches_revision(node_id)
        self.tree.set_position(node_id)
        return ToolResult(
            True,
            value={
                "node_id": node_id,
                "revision_id": str(node["revision_id"]),
                "status": "fold_finished",
                # Candidate selection, AcceptanceRules and the final freeze are
                # the Pipeline's, not the Agent's: finishing only nominates.
                "fold_status": "pending_pipeline_review",
                "write_locked": True,
            },
            finish=True,
        )

    def _require_different_hypothesis(self, node_id: str, nominated_structure: str) -> None:
        parent_structure = self._parent_structure
        if parent_structure is None:
            return
        different_ids = {
            candidate_id
            for candidate_id, structure in self._session_complete_structures()
            if structure != parent_structure
        }
        if not different_ids:
            raise ToolError(
                "finish_fold requires a complete Validation whose executable "
                "strategy logic differs from the parent; comment-only changes do not count"
            )
        if nominated_structure == parent_structure or node_id in different_ids:
            return
        raise ToolError(
            "finish_fold can select only a different-hypothesis Validation "
            "or an explicit keep-parent after one existed"
        )

    def _session_complete_structures(self) -> list[tuple[str, str]]:
        found: list[tuple[str, str]] = []
        for node in self.tree.nodes():
            if (
                node.get("fold_id") != self.fold_id
                or node.get("run_id") != self.run_id
                or not node.get("complete_validation")
                or not node.get("revision_id")
            ):
                continue
            candidate_id = str(node["node_id"])
            main_py = self.tree.node_output_dir(candidate_id) / "main.py"
            if not main_py.is_file():
                continue
            try:
                found.append(
                    (
                        candidate_id,
                        executable_source_structure(main_py.read_text(encoding="utf-8")),
                    )
                )
            except (OSError, SyntaxError):
                continue
        return found

    def _require_current_matches_revision(self, node_id: str) -> None:
        if self._current_output is None:
            return
        nominated = self.tree.node_output_dir(node_id)
        if _tree_bytes(nominated) != _tree_bytes(self._current_output):
            raise ToolError(
                "finish_fold requires the current output to match the selected "
                "Validation revision; restore that Step or run a new complete "
                "daily_backtest"
            )
        nominated_models = self.tree.node_models_dir(node_id)
        current_models = self._current_models
        if _tree_bytes(nominated_models) != _tree_bytes(current_models):
            raise ToolError(
                "finish_fold requires the current models to match the selected "
                "Validation revision; restore that Step or run a new complete "
                "daily_backtest"
            )

    def _node_structure(self, node_id: str) -> str:
        main_py = self.tree.node_output_dir(node_id) / "main.py"
        if not main_py.is_file():
            raise ToolError(
                "finish_fold cannot select a Step whose strategy snapshot is absent"
            )
        try:
            return executable_source_structure(main_py.read_text(encoding="utf-8"))
        except (OSError, SyntaxError) as exc:
            raise ToolError(f"finish_fold cannot compare {node_id}: {exc}") from exc


__all__ = ["FinishFoldTool", "executable_source_structure"]
