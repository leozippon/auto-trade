"""Tool doubles and builders for the session-runner and sub-agent tests.

The research session ends through ``test`` (``pipelines/session_tools``),
whose real body replays a candidate; these tests are about the runner, so
``TerminalTool`` stands in: it ends the session with the node it is given and
refuses nothing, exactly the surface the runner's terminal rules read.
``make_runner`` and ``make_engine`` build a session runner and a sub-agent
engine over scripted replies, with the defaults the tests share.
"""

from __future__ import annotations

import json
import tempfile
import threading
from collections.abc import Iterable, Mapping
from pathlib import Path

from autotrade.agent.experiment_facts import build_experiment_facts
from autotrade.agent.runner import AgentSessionRunner
from autotrade.agent.subagent import SubAgentEngine
from autotrade.environment.broker import BrokerProfile
from autotrade.environment.data.snapshot import SnapshotConfig
from autotrade.environment.identity import AgentRefStore
from autotrade.environment.llm import LLMProxy, ProviderResponse, ToolCall
from autotrade.environment.llm.proxy import ScriptedLLM
from autotrade.environment.step_tree import StepTree
from autotrade.environment.tools.base import ToolRegistry, ToolResult, ToolSpec


class TerminalTool:
    spec = ToolSpec(
        "test",
        "terminal tool double: ends the session with the candidate it is given",
        {"type": "object", "properties": {}, "required": [], "additionalProperties": True},
        mutating=True,
    )

    def __init__(self, tree: StepTree | None = None) -> None:
        self.tree = tree
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


def shell_turns(count: int, *, prefix: str = "s") -> list[ProviderResponse]:
    return [
        ProviderResponse(tool_calls=(ToolCall(f"{prefix}{index}", "shell", {"argv": ["rg", str(index)]}),))
        for index in range(count)
    ]


def run_facts(
    *,
    data_summary: dict[str, object] | None = None,
    acceptance_rules: dict[str, object] | None = None,
    **manifest_overrides: object,
) -> dict[str, object]:
    manifest: dict[str, object] = {
        "experiment_id": "exp",
        "run_id": "run_x",
        "epoch_id": "research",
        "fold_id": "s4",
        "kind": "research",
        "research": {
            "decision_time": "2025-06-30T23:59:59+08:00",
            "input_window": "20230701..20250630",
            "research_period": "20210701..20250630",
            "years": [{"label": "Y1", "start": "20210701", "end": "20220630"}],
        },
        "schedule": {"period": "day", "inference_time": "08:30"},
        "snapshot_config": SnapshotConfig().to_record(),
        "benchmark_index": "000300.SH",
        "broker_profile": BrokerProfile().to_record(),
    }
    manifest.update(manifest_overrides)
    with tempfile.TemporaryDirectory() as tmp:
        return build_experiment_facts(
            manifest=manifest,
            ref_store=AgentRefStore(Path(tmp) / "experiment"),
            data_summary=data_summary,
            acceptance_rules=acceptance_rules,
        )


def prompt_facts(prompt: str) -> dict:
    """The run-facts JSON block the preview embeds in the system prompt."""
    body = prompt.split("```json\n", 1)[1].split("\n```", 1)[0]
    return json.loads(body)


class FakeClock:
    def __init__(self, value: float = 100.0) -> None:
        self.value = value
        self._lock = threading.Lock()

    def __call__(self) -> float:
        with self._lock:
            return self.value

    def advance(self, seconds: float) -> None:
        with self._lock:
            self.value += seconds
