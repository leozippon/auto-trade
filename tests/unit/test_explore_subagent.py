"""Explore is a one-level read-only Fold sub-agent on the parent trace and budget."""

from __future__ import annotations

import time
from collections.abc import Mapping
from pathlib import Path

import pytest

from autotrade.agent.explore import ExploreSubAgentEngine
from autotrade.agent.runner import AgentSessionConfig, AgentSessionRunner
from autotrade.environment.llm import ProviderResponse, ScriptedLLM, ToolCall
from autotrade.environment.tools import (
    SafeWorkspace,
    ToolRegistry,
    ToolResult,
    ToolSpec,
    WriteFileTool,
)
from autotrade.pipelines.local_backend import SessionBudgetLLM


class DeclaredReadOnlyShell:
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


def test_explore_events_land_on_the_parent_fold_trace() -> None:
    events: list[tuple[str, dict[str, object]]] = []
    shell = DeclaredReadOnlyShell()
    llm = ScriptedLLM(
        [
            ProviderResponse(
                tool_calls=(ToolCall("s", "shell", {"argv": ["ls"]}),)
            ),
            ProviderResponse(content="结论：可见。"),
        ]
    )
    result = ExploreSubAgentEngine(
        llm=llm,
        tools=ToolRegistry([shell]),
        event_sink=lambda event, payload: events.append((event, payload)),
    ).run("inspect snapshot schema", parent_call_id="call_parent")
    types = [event for event, _payload in events]
    assert types[0] == "explore_task"
    assert "explore_llm" in types
    assert "explore_tool" in types
    assert types[-1] == "explore"
    assert events[0][1]["parent_call_id"] == "call_parent"
    assert events[0][1]["task"] == "inspect snapshot schema"
    assert events[0][1]["task_id"] == result["task_id"]
    tool_event = next(payload for event, payload in events if event == "explore_tool")
    assert tool_event["tool"] == "shell"
    assert tool_event["parent_call_id"] == "call_parent"
    assert result["status"] == "completed"


def test_explore_rejects_nested_explore_and_backtest_specs() -> None:
    class NamedReadOnly:
        def __init__(self, name: str) -> None:
            self.spec = ToolSpec(
                name,
                "not allowed on Explore",
                {"type": "object", "properties": {}, "required": []},
            )

        def invoke(self, arguments: Mapping[str, object]) -> ToolResult:
            del arguments
            return ToolResult(True, value={})

    with pytest.raises(ValueError, match="read-only whitelist"):
        ExploreSubAgentEngine(
            llm=ScriptedLLM([]),
            tools=ToolRegistry([NamedReadOnly("explore")]),
        )
    with pytest.raises(ValueError, match="read-only whitelist"):
        ExploreSubAgentEngine(
            llm=ScriptedLLM([]),
            tools=ToolRegistry([NamedReadOnly("daily_backtest")]),
        )


def test_explore_rejects_write_tools(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="non-mutating"):
        ExploreSubAgentEngine(
            llm=ScriptedLLM([]),
            tools=ToolRegistry([WriteFileTool(SafeWorkspace(tmp_path))]),
        )


def test_explore_and_main_share_one_session_call_budget() -> None:
    scripted = ScriptedLLM(
        [
            ProviderResponse(content="sub digest"),
            ProviderResponse(content="must remain unused"),
        ]
    )
    budgeted = SessionBudgetLLM(
        scripted, max_calls=1, deadline=time.monotonic() + 10
    )
    result = ExploreSubAgentEngine(
        llm=budgeted,
        tools=ToolRegistry([DeclaredReadOnlyShell()]),
    ).run("count rows")
    assert result["status"] == "completed"
    with pytest.raises(RuntimeError, match="budget exhausted"):
        budgeted.complete([])
    assert len(scripted.calls) == 1


def test_meta_runner_cannot_receive_explore(tmp_path: Path) -> None:
    (tmp_path / "dummy").mkdir()
    explore = ExploreSubAgentEngine(
        llm=ScriptedLLM([]),
        tools=ToolRegistry([DeclaredReadOnlyShell()]),
    )
    with pytest.raises(ValueError, match="cannot provide explore"):
        AgentSessionRunner(
            llm=ScriptedLLM([]),
            tools=ToolRegistry(),
            system_prompt="meta",
            config=AgentSessionConfig(mode="meta"),
            explore=explore,
        )


def test_runner_attaches_explore_events_to_its_sink() -> None:
    events: list[str] = []
    explore = ExploreSubAgentEngine(
        llm=ScriptedLLM([ProviderResponse(content="digest")]),
        tools=ToolRegistry([DeclaredReadOnlyShell()]),
    )
    runner = AgentSessionRunner(
        llm=ScriptedLLM([]),
        tools=ToolRegistry(),
        system_prompt="fold",
        explore=explore,
        event_sink=lambda event, _payload: events.append(event),
    )
    dispatched = runner._dispatch_explore(
        ToolCall("e1", "explore", {"task": "read schema"})
    )
    assert dispatched.get("ok") is True
    assert "explore_task" in events
    assert "explore" in events
