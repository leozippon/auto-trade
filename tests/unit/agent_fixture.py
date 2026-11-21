"""Tool doubles and builders for the session-runner and sub-agent tests.

The research session ends through ``test`` (``pipelines/session_tools``),
whose real body replays a candidate; these tests are about the runner, so
``TerminalTool`` stands in: it ends the session with the node it is given and
refuses nothing, exactly the surface the runner's terminal rules read.
``make_runner`` and ``make_engine`` build a session runner and a sub-agent
engine over scripted replies, with the defaults the tests share.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from autotrade.agent.runner import AgentSessionRunner
from autotrade.agent.subagent import SubAgentEngine
from autotrade.environment.llm import LLMProxy, ProviderResponse, ScriptedLLM
from autotrade.environment.step_tree import StepTree
from autotrade.environment.tools import ToolRegistry
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
                "candidate": {"node_id": node_id},
            },
            finish=True,
        )


class DeclaredReadOnlyShell:
    """A ``shell`` that declares itself non-mutating and records its calls."""

    spec = ToolSpec(
        "shell",
        "Test-only shell declared non-mutating.",
        {
            "type": "object",
            "properties": {"argv": {"type": "array", "items": {"type": "string"}}},
            "required": ["argv"],
            "additionalProperties": False,
        },
    )

    def __init__(self) -> None:
        self.calls: list[Mapping[str, object]] = []

    def invoke(self, arguments: Mapping[str, object]) -> ToolResult:
        self.calls.append(arguments)
        return ToolResult(True, value={"stdout": "ok"})


def _scripted(responses: tuple[ProviderResponse, ...], llm: LLMProxy | None) -> LLMProxy:
    if llm is None:
        return ScriptedLLM(list(responses))
    if responses:
        raise TypeError("pass scripted responses or an llm, not both")
    return llm


def make_runner(
    *responses: ProviderResponse,
    llm: LLMProxy | None = None,
    tools: Iterable[object] = (),
    system_prompt: str = "research",
    **kwargs: object,
) -> AgentSessionRunner:
    """A session runner over ``responses`` (or ``llm``) and ``tools``; any
    other keyword goes to ``AgentSessionRunner`` unchanged."""
    return AgentSessionRunner(
        llm=_scripted(responses, llm),
        tools=ToolRegistry(list(tools)),
        system_prompt=system_prompt,
        **kwargs,  # type: ignore[arg-type]
    )


def make_engine(
    *responses: ProviderResponse,
    llm: LLMProxy | None = None,
    tools: Iterable[object] | None = None,
    **kwargs: object,
) -> SubAgentEngine:
    """A sub-agent engine over ``responses`` (or ``llm``) whose tools default
    to one ``DeclaredReadOnlyShell``; any other keyword goes to
    ``SubAgentEngine`` unchanged."""
    return SubAgentEngine(
        llm=_scripted(responses, llm),
        tools=ToolRegistry([DeclaredReadOnlyShell()] if tools is None else list(tools)),
        **kwargs,  # type: ignore[arg-type]
    )


__all__ = ["DeclaredReadOnlyShell", "TerminalTool", "make_engine", "make_runner"]
