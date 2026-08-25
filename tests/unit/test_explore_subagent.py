"""Explore is a one-level writable Fold coding sub-agent on the parent trace and budget."""

from __future__ import annotations

import time
from collections.abc import Mapping
from pathlib import Path

import pytest

from autotrade.agent.explore import ExploreSubAgentEngine
from autotrade.agent.runner import AgentSessionConfig, AgentSessionRunner
from autotrade.environment.llm import ProviderResponse, ScriptedLLM, ToolCall
from autotrade.environment.tools import (
    CommandResult,
    EditFileTool,
    ModificationCheckTool,
    ReadOnlyShellTool,
    SafeWorkspace,
    SandboxShellTool,
    SearchRoots,
    StrategyValidationTool,
    ToolRegistry,
    ToolResult,
    ToolSpec,
    WriteFileTool,
)
from autotrade.pipelines.local_backend import SessionBudgetLLM, build_fold_explore_tools

_STRATEGY = "def generate_orders(context):\n    return []\n"


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


class WritingRunner:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.calls: list[list[str]] = []

    def run(self, argv, *, cwd, timeout_seconds, max_output_chars, input_text=None):
        del timeout_seconds, max_output_chars, input_text
        self.calls.append(list(argv))
        if argv and argv[0] == "touch" and len(argv) >= 2:
            path = (self.root / cwd / argv[1]).resolve()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("from-shell\n", encoding="utf-8")
            return CommandResult(0, stdout="ok")
        return CommandResult(0, stdout="ok")


class BoomWrite:
    spec = ToolSpec(
        "write_file",
        "Exploding writer.",
        {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["path", "content"],
            "additionalProperties": False,
        },
        mutating=True,
    )

    def invoke(self, arguments: Mapping[str, object]) -> ToolResult:
        del arguments
        raise RuntimeError("disk exploded")


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


def test_explore_rejects_nested_explore_and_fold_control_specs() -> None:
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

    for name in (
        "explore",
        "daily_backtest",
        "finish_fold",
        "step_rollback",
        "ask_user",
        "unknown_tool",
    ):
        with pytest.raises(ValueError, match="not allowed"):
            ExploreSubAgentEngine(
                llm=ScriptedLLM([]),
                tools=ToolRegistry([NamedReadOnly(name)]),
            )


def test_explore_write_edit_shell_and_checks_persist(tmp_path: Path) -> None:
    workspace = tmp_path / "agent"
    workspace.mkdir()
    (workspace / "output").mkdir()
    safe = SafeWorkspace(workspace)
    runner = WritingRunner(workspace)
    llm = ScriptedLLM(
        [
            ProviderResponse(
                tool_calls=(
                    ToolCall(
                        "w",
                        "write_file",
                        {"path": "output/main.py", "content": _STRATEGY},
                    ),
                )
            ),
            ProviderResponse(
                tool_calls=(
                    ToolCall(
                        "e",
                        "edit_file",
                        {
                            "path": "output/main.py",
                            "old_text": "return []",
                            "new_text": "return []  # ok",
                        },
                    ),
                    ToolCall("s", "shell", {"argv": ["touch", "from_shell.txt"]}),
                )
            ),
            ProviderResponse(
                tool_calls=(
                    ToolCall(
                        "v", "validate_strategy", {"path": "output/main.py"}
                    ),
                    ToolCall("m", "modification_check", {}),
                )
            ),
            ProviderResponse(content="结论：已写入并可验证。"),
        ]
    )
    result = ExploreSubAgentEngine(
        llm=llm,
        tools=ToolRegistry(
            [
                WriteFileTool(safe),
                EditFileTool(safe),
                SandboxShellTool(safe, runner),
                StrategyValidationTool(safe),
                ModificationCheckTool(workspace / "output"),
            ]
        ),
    ).run("write and check the strategy")
    assert result["status"] == "completed"
    written = (workspace / "output" / "main.py").read_text(encoding="utf-8")
    assert "return []  # ok" in written
    assert (workspace / "from_shell.txt").read_text(encoding="utf-8") == "from-shell\n"
    assert runner.calls == [["touch", "from_shell.txt"]]
    assert result["tool_calls"] == 5


def test_explore_write_failure_does_not_finish_parent(tmp_path: Path) -> None:
    events: list[str] = []
    explore = ExploreSubAgentEngine(
        llm=ScriptedLLM(
            [
                ProviderResponse(
                    tool_calls=(
                        ToolCall(
                            "w",
                            "write_file",
                            {"path": "output/main.py", "content": "x"},
                        ),
                    )
                ),
                ProviderResponse(content="must not run"),
            ]
        ),
        tools=ToolRegistry([BoomWrite()]),
        event_sink=lambda event, _payload: events.append(event),
    )
    runner = AgentSessionRunner(
        llm=ScriptedLLM([]),
        tools=ToolRegistry(),
        system_prompt="fold",
        explore=explore,
    )
    dispatched = runner._dispatch_explore(
        ToolCall("e1", "explore", {"task": "write then fail"})
    )
    value = dispatched["value"]
    assert dispatched["ok"] is False
    assert isinstance(value, dict)
    assert value["status"] == "error"
    assert "disk exploded" in str(value.get("error") or "")
    assert "explore" in events
    assert not (tmp_path / "output" / "main.py").exists()


def test_explore_readonly_write_failure_stays_an_observation(tmp_path: Path) -> None:
    workspace = tmp_path / "agent"
    workspace.mkdir()
    (workspace / "output").mkdir()
    (workspace / "output" / "README.md").write_text("keep\n", encoding="utf-8")
    llm = ScriptedLLM(
        [
            ProviderResponse(
                tool_calls=(
                    ToolCall(
                        "w",
                        "write_file",
                        {"path": "output/README.md", "content": "tamper"},
                    ),
                )
            ),
            ProviderResponse(content="写入被拒绝。"),
        ]
    )
    result = ExploreSubAgentEngine(
        llm=llm,
        tools=ToolRegistry([WriteFileTool(SafeWorkspace(workspace))]),
    ).run("overwrite readme")
    assert result["status"] == "completed"
    assert (workspace / "output" / "README.md").read_text(encoding="utf-8") == "keep\n"


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


def test_explore_unknown_tool_call_is_rejected_without_invoke() -> None:
    shell = DeclaredReadOnlyShell()
    llm = ScriptedLLM(
        [
            ProviderResponse(
                tool_calls=(ToolCall("b", "daily_backtest", {}),)
            ),
            ProviderResponse(content="unknown tool blocked"),
        ]
    )
    result = ExploreSubAgentEngine(
        llm=llm,
        tools=ToolRegistry([shell]),
    ).run("do not backtest")
    assert result["status"] == "completed"
    assert result["digest"] == "unknown tool blocked"
    assert shell.calls == []
    assert result["tool_calls"] == 0


class _UnusedRunner:
    def run(self, argv, *, cwd, timeout_seconds, max_output_chars, input_text=None):
        raise AssertionError("runner is unused")


def test_fold_explore_tools_are_writable_shell_contract(tmp_path: Path) -> None:
    workspace = tmp_path / "agent"
    workspace.mkdir()
    (workspace / "output").mkdir()
    safe = SafeWorkspace(workspace)
    tools = build_fold_explore_tools(
        SearchRoots(safe),
        safe,
        _UnusedRunner(),
        ModificationCheckTool(workspace / "output"),
    )
    assert [tool.spec.name for tool in tools] == [
        "read_file",
        "grep",
        "glob",
        "write_file",
        "edit_file",
        "shell",
        "validate_strategy",
        "modification_check",
    ]
    by_name = {tool.spec.name: tool for tool in tools}
    assert type(by_name["shell"]) is SandboxShellTool
    assert not isinstance(by_name["shell"], ReadOnlyShellTool)
