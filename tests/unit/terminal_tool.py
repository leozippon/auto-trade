"""A terminal tool double for the session-runner tests.

The research session ends through ``test`` (``pipelines/session_tools``),
whose real body replays a candidate; these tests are about the runner, so a
stub named ``test`` stands in: it ends the session with the node it is given
and refuses nothing, exactly the surface the runner's terminal rules read.
"""

from __future__ import annotations

from collections.abc import Mapping

from autotrade.environment.step_tree import StepTree
from autotrade.environment.tools.base import ToolResult, ToolSpec


class TerminalTool:
    spec = ToolSpec(
        "test",
        "terminal tool double: ends the session with the candidate it is given",
        {"type": "object", "properties": {}, "required": [], "additionalProperties": True},
        mutating=True,
    )

    def __init__(self, tree: StepTree | None = None, *, session_ref: str = "", registration=None) -> None:
        self.tree = tree
        self.session_ref = session_ref
        self.registration = registration
        self.calls: list[Mapping[str, object]] = []

    def invoke(self, arguments: Mapping[str, object]) -> ToolResult:
        self.calls.append(dict(arguments))
        node_id = str(arguments.get("candidate") or arguments.get("node_id") or "")
        if self.tree is not None and node_id:
            self.tree.set_position(node_id)
        return ToolResult(
            True,
            value={
                "status": "step_recorded",
                "outcome": "completed",
                "candidate": {"node_id": node_id, "deterministic": arguments.get("deterministic") is True},
            },
            finish=True,
        )


__all__ = ["TerminalTool"]
