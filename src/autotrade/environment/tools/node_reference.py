"""How a session tool names a Step node: its full id or its short handle.

A node id joins four opaque parts and runs past a hundred characters, which
the session's model mistypes (13 ``run_null_control`` refusals in one round
were spliced ids). Every tool that takes a node therefore also accepts the
node's short handle, the result name its id ends with, wherever no other node
of the session shares it; the rule itself lives in ``step_tree``.
"""

from __future__ import annotations

from collections.abc import Sequence

from autotrade.environment.step_tree import (
    NodeReferenceError,
    StepTree,
    resolve_node_reference,
    unique_handles,
)

from .base import ToolError

NODE_REFERENCE_DESCRIPTION = (
    "A Step node of this session: its full node_id, or its short handle "
    "(valid_002 -- the result name that batch rows, the Step tree and these "
    "tools show beside the id) wherever no other node of the session shares it."
)
NODE_REFERENCE_EXAMPLE = "valid_002"


def session_node_reference(
    tool: str,
    tree: StepTree,
    reference: str,
    *,
    session_ref: str,
    offered: Sequence[str],
) -> str:
    """The full node id ``reference`` names, or ``tool``'s refusal listing ``offered``.

    Any node id of the tree passes through as itself, another session's
    included, so the tool's own checks say why such a node is refused rather
    than calling it unknown.
    """

    if any(str(node["node_id"]) == reference for node in tree.nodes()):
        return reference
    try:
        return resolve_node_reference(
            reference, tree.session_node_ids(session_ref), offered=offered
        )
    except NodeReferenceError as exc:
        raise ToolError(
            f"{tool}: {exc}",
            error_type="unknown_node",
            blocked_target="node_id",
            details={"nodes": exc.offered},
        ) from exc


def session_handle(tree: StepTree, node_id: str, *, session_ref: str) -> dict[str, str]:
    """``{"handle": ...}`` for a node whose handle no other node of its session shares."""

    handle = unique_handles(tree.session_node_ids(session_ref)).get(node_id)
    return {"handle": handle} if handle else {}


def session_handles(tree: StepTree, node_ids: Sequence[str], *, session_ref: str) -> list[str]:
    """``node_ids`` as the Agent should name them: the handle where unambiguous."""

    handles = unique_handles(tree.session_node_ids(session_ref))
    return [handles.get(node_id, node_id) for node_id in node_ids]


__all__ = [
    "NODE_REFERENCE_DESCRIPTION",
    "NODE_REFERENCE_EXAMPLE",
    "session_handle",
    "session_handles",
    "session_node_reference",
]
