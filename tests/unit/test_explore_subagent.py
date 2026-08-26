"""Explore is a one-level writable Fold coding sub-agent on the parent trace and budget."""

from __future__ import annotations

import time
from collections.abc import Mapping
from pathlib import Path

import pytest

from autotrade.agent.explore import (
    EXPLORE_ROLES,
    EXPLORE_SYSTEM_PROMPT,
    FOLD_REQUIRED_EXPLORE_ROLES,
    META_EXPLORE_SYSTEM_PROMPT,
    META_REQUIRED_EXPLORE_ROLES,
    OPTIONAL_EXPLORE_ROLES,
    allowed_explore_tools,
    ExploreSubAgentEngine,
    explore_system_prompt,
    session_explore_roles,
)
from autotrade.agent.prompts import build_system_prompt
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
from autotrade.pipelines.local_backend import (
    SessionBudgetLLM,
    build_fold_explore_tools,
    build_meta_explore_tools,
)

_STRATEGY = "def generate_orders(context):\n    return []\n"


def _fold_config(*roles: str, **kwargs: object) -> AgentSessionConfig:
    chosen = roles or ("auditor",)
    return AgentSessionConfig(
        mode="fold",
        required_explore_roles=chosen,
        **kwargs,  # type: ignore[arg-type]
    )


def _meta_config(*roles: str, **kwargs: object) -> AgentSessionConfig:
    chosen = roles or META_REQUIRED_EXPLORE_ROLES
    return AgentSessionConfig(
        mode="meta",
        required_explore_roles=chosen,
        **kwargs,  # type: ignore[arg-type]
    )


def _function_name(record: object) -> str:
    if not isinstance(record, dict):
        raise TypeError("provider tool must be an object")
    function = record.get("function")
    if not isinstance(function, dict):
        raise TypeError("provider tool function must be an object")
    return str(function.get("name") or "")


def _function_payload(record: object) -> dict[str, object]:
    if not isinstance(record, dict):
        raise TypeError("provider tool must be an object")
    function = record.get("function")
    if not isinstance(function, dict):
        raise TypeError("provider tool function must be an object")
    return function


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
    ).run(
        "inspect snapshot schema",
        role="auditor",
        parent_call_id="call_parent",
    )
    types = [event for event, _payload in events]
    assert types[0] == "explore_task"
    assert "explore_llm" in types
    assert "explore_tool" in types
    assert types[-1] == "explore"
    assert events[0][1]["parent_call_id"] == "call_parent"
    assert events[0][1]["role"] == "auditor"
    assert "task" not in events[0][1]
    assert events[0][1]["task_id"] == result["task_id"]
    assert result["role"] == "auditor"
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
    ).run("write and check the strategy", role="developer")
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
        config=_fold_config("developer"),
        explore=explore,
    )
    dispatched = runner._dispatch_explore(
        ToolCall(
            "e1",
            "explore",
            {"role": "developer", "task": "write then fail"},
        )
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
    ).run("overwrite readme", role="developer")
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
    ).run("count rows", role="auditor")
    assert result["status"] == "completed"
    with pytest.raises(RuntimeError, match="budget exhausted"):
        budgeted.complete([])
    assert len(scripted.calls) == 1


def test_meta_runner_rejects_fold_mode_explore() -> None:
    explore = ExploreSubAgentEngine(
        llm=ScriptedLLM([]),
        tools=ToolRegistry([DeclaredReadOnlyShell()]),
        mode="fold",
    )
    with pytest.raises(ValueError, match="mode='meta'"):
        AgentSessionRunner(
            llm=ScriptedLLM([]),
            tools=ToolRegistry(),
            system_prompt="meta",
            config=_meta_config(),
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
        config=_fold_config("auditor"),
        explore=explore,
        event_sink=lambda event, _payload: events.append(event),
    )
    dispatched = runner._dispatch_explore(
        ToolCall("e1", "explore", {"role": "auditor", "task": "read schema"})
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
    ).run("do not backtest", role="auditor")
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
        "todo",
        "modification_check",
    ]
    by_name = {tool.spec.name: tool for tool in tools}
    assert type(by_name["shell"]) is SandboxShellTool
    assert not isinstance(by_name["shell"], ReadOnlyShellTool)


class _NamedTool:
    def __init__(self, name: str) -> None:
        self.spec = ToolSpec(
            name,
            "named stub",
            {"type": "object", "properties": {}, "required": []},
        )

    def invoke(self, arguments: Mapping[str, object]) -> ToolResult:
        del arguments
        return ToolResult(True, value={})


class _FinishStub:
    def __init__(self, name: str) -> None:
        self.spec = ToolSpec(
            name,
            "terminal stub",
            {"type": "object", "properties": {}, "required": []},
        )
        self.invoked = 0

    def invoke(self, arguments: Mapping[str, object]) -> ToolResult:
        del arguments
        self.invoked += 1
        return ToolResult(True, value={"status": "done"}, finish=True)


def test_finish_fold_rejected_until_required_roles_attempted() -> None:
    finish = _FinishStub("finish_fold")
    explore = ExploreSubAgentEngine(
        llm=ScriptedLLM(
            [ProviderResponse(content="digest"), ProviderResponse(content="digest")]
        ),
        tools=ToolRegistry([DeclaredReadOnlyShell()]),
    )
    events: list[tuple[str, dict[str, object]]] = []
    runner = AgentSessionRunner(
        llm=ScriptedLLM(
            [
                ProviderResponse(tool_calls=(ToolCall("f1", "finish_fold", {}),)),
                ProviderResponse(
                    tool_calls=(ToolCall("e0", "explore", {"task": "  "}),)
                ),
                ProviderResponse(tool_calls=(ToolCall("f2", "finish_fold", {}),)),
                ProviderResponse(
                    tool_calls=(
                        ToolCall(
                            "e1",
                            "explore",
                            {"role": "auditor", "task": "check data"},
                        ),
                    )
                ),
                ProviderResponse(tool_calls=(ToolCall("f3", "finish_fold", {}),)),
                ProviderResponse(
                    tool_calls=(
                        ToolCall(
                            "e2",
                            "explore",
                            {"role": "developer", "task": "check strategy"},
                        ),
                    )
                ),
                ProviderResponse(tool_calls=(ToolCall("f4", "finish_fold", {}),)),
            ]
        ),
        tools=ToolRegistry([finish]),
        system_prompt="fold",
        config=_fold_config("auditor", "developer"),
        explore=explore,
        event_sink=lambda event, payload: events.append((event, payload)),
    )
    result = runner.run("must cover both roles")
    assert result.status == "finished"
    assert finish.invoked == 1
    assert runner._explore_attempts == 2
    assert runner._explored_roles == {"auditor", "developer"}
    attempt_events = [payload for event, payload in events if event == "explore_attempt"]
    assert attempt_events[0]["ok"] is False
    assert "task" not in attempt_events[0]
    assert attempt_events[1]["role"] == "auditor"
    assert attempt_events[2]["role"] == "developer"
    ended = next(payload for event, payload in events if event == "session_end")
    assert ended["explore_attempts"] == 2
    assert ended["explored_roles"] == ["auditor", "developer"]
    assert "task" not in ended
    finish_errors = []
    for event, payload in events:
        result = payload.get("result")
        if (
            event == "tool_call"
            and payload.get("tool") == "finish_fold"
            and isinstance(result, dict)
            and result.get("ok") is False
        ):
            finish_errors.append(str(result.get("error") or ""))
    assert any("auditor" in error and "developer" in error for error in finish_errors)
    assert any("missing roles: developer" in error for error in finish_errors)


def test_failed_explore_attempt_counts_for_its_role() -> None:
    finish = _FinishStub("finish_fold")
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
                )
            ]
        ),
        tools=ToolRegistry([BoomWrite()]),
    )
    runner = AgentSessionRunner(
        llm=ScriptedLLM(
            [
                ProviderResponse(
                    tool_calls=(
                        ToolCall(
                            "e1",
                            "explore",
                            {"role": "developer", "task": "write then fail"},
                        ),
                    )
                ),
                ProviderResponse(tool_calls=(ToolCall("f1", "finish_fold", {}),)),
            ]
        ),
        tools=ToolRegistry([finish]),
        system_prompt="fold",
        config=_fold_config("developer"),
        explore=explore,
    )
    assert runner.run("failed attempt still counts").status == "finished"
    assert runner._explore_attempts == 1
    assert runner._explored_roles == {"developer"}
    assert finish.invoked == 1


def test_explore_attempt_counter_resets_on_new_run() -> None:
    finish = _FinishStub("finish_fold")
    explore = ExploreSubAgentEngine(
        llm=ScriptedLLM([ProviderResponse(content="unused")]),
        tools=ToolRegistry([DeclaredReadOnlyShell()]),
    )
    runner = AgentSessionRunner(
        llm=ScriptedLLM(
            [ProviderResponse(tool_calls=(ToolCall("f1", "finish_fold", {}),))]
        ),
        tools=ToolRegistry([finish]),
        system_prompt="fold",
        config=_fold_config(max_llm_calls=2),
        explore=explore,
    )
    runner._explore_attempts = 4
    runner._explored_roles = {"auditor"}
    with pytest.raises(RuntimeError, match="call budget"):
        runner.run("new session without explore")
    assert finish.invoked == 0
    assert runner._explore_attempts == 0
    assert runner._explored_roles == set()


def test_missing_roles_do_not_enter_hard_finalization() -> None:
    explore = ExploreSubAgentEngine(
        llm=ScriptedLLM([ProviderResponse(content="digest")]),
        tools=ToolRegistry([DeclaredReadOnlyShell()]),
    )
    runner = AgentSessionRunner(
        llm=ScriptedLLM([]),
        tools=ToolRegistry([_FinishStub("finish_fold")]),
        system_prompt="fold",
        config=_fold_config(
            "auditor",
            "developer",
            finalize_before_deadline_seconds=300.0,
            deadline_grace_seconds=0.0,
        ),
        explore=explore,
    )
    runner._complete_validation_nodes = [
        {"node_id": "node_a", "revision_id": "rev_a"}
    ]
    assert runner._activate_hard_finalization_if_ready(10.0) is False
    runner._explore_attempts = 1
    runner._explored_roles = {"auditor"}
    assert runner._activate_hard_finalization_if_ready(10.0) is False
    runner._explored_roles = {"auditor", "developer"}
    assert runner._activate_hard_finalization_if_ready(10.0) is True


def test_sessions_without_explore_still_finish() -> None:
    finish = _FinishStub("finish_fold")
    runner = AgentSessionRunner(
        llm=ScriptedLLM(
            [ProviderResponse(tool_calls=(ToolCall("f1", "finish_fold", {}),))]
        ),
        tools=ToolRegistry([finish]),
        system_prompt="fold",
    )
    assert runner.run("no explore configured").status == "finished"
    assert finish.invoked == 1


def test_meta_explore_is_readonly_and_cannot_nest(tmp_path: Path) -> None:
    workspace = tmp_path / "agent"
    workspace.mkdir()
    (workspace / "PRIOR.md").write_text("keep\n", encoding="utf-8")
    safe = SafeWorkspace(workspace)
    tools = build_meta_explore_tools(SearchRoots(safe), safe)
    assert [tool.spec.name for tool in tools] == ["read_file", "grep", "glob", "todo"]
    engine = ExploreSubAgentEngine(
        llm=ScriptedLLM(
            [
                ProviderResponse(
                    tool_calls=(
                        ToolCall(
                            "w",
                            "write_file",
                            {"path": "PRIOR.md", "content": "tamper"},
                        ),
                    )
                ),
                ProviderResponse(content="写入被拒绝。"),
            ]
        ),
        tools=ToolRegistry(tools),
        mode="meta",
    )
    result = engine.run("do not write prior", role="auditor")
    assert result["status"] == "completed"
    assert (workspace / "PRIOR.md").read_text(encoding="utf-8") == "keep\n"
    assert "sub-agent" in META_EXPLORE_SYSTEM_PROMPT
    assert "pyright" not in META_EXPLORE_SYSTEM_PROMPT
    with pytest.raises(ValueError, match="not allowed"):
        ExploreSubAgentEngine(
            llm=ScriptedLLM([]),
            tools=ToolRegistry([WriteFileTool(safe)]),
            mode="meta",
        )
    with pytest.raises(ValueError, match="not allowed"):
        ExploreSubAgentEngine(
            llm=ScriptedLLM([]),
            tools=ToolRegistry([_NamedTool("explore")]),
            mode="meta",
        )


def test_meta_runner_accepts_meta_mode_explore_and_gates_finish_meta() -> None:
    finish = _FinishStub("finish_meta")
    explore = ExploreSubAgentEngine(
        llm=ScriptedLLM([ProviderResponse(content="trace reviewed")]),
        tools=ToolRegistry([_NamedTool("read_file")]),
        mode="meta",
    )
    runner = AgentSessionRunner(
        llm=ScriptedLLM(
            [
                ProviderResponse(tool_calls=(ToolCall("f1", "finish_meta", {}),)),
                ProviderResponse(
                    tool_calls=(
                        ToolCall(
                            "e1",
                            "explore",
                            {"role": "auditor", "task": "review traces"},
                        ),
                    )
                ),
                ProviderResponse(tool_calls=(ToolCall("f2", "finish_meta", {}),)),
            ]
        ),
        tools=ToolRegistry([finish]),
        system_prompt="meta",
        config=_meta_config(),
        explore=explore,
    )
    result = runner.run("meta must explore")
    assert result.status == "finished"
    assert finish.invoked == 1


def test_fold_and_explore_prompts_name_pyright_meta_does_not() -> None:
    fold = build_system_prompt(mode="fold", experiment_facts={})
    meta = build_system_prompt(mode="meta", experiment_facts={})
    command = (
        "pyright --project /opt/autotrade/pyrightconfig.json "
        "/mnt/agent/workspace /mnt/agent/output"
    )
    assert command in fold
    assert command in EXPLORE_SYSTEM_PROMPT
    assert "debug 顾问" in fold
    assert command not in meta
    assert "pyright" not in meta
    assert command not in META_EXPLORE_SYSTEM_PROMPT
    assert "`auditor`" in fold
    assert "`developer`" in fold
    assert "`general-purpose`" in fold
    assert "`Explore`" in fold
    assert "不能替代" in fold
    assert "至少一个具体" not in fold
    assert "`auditor`" in meta
    assert "`Explore`" in meta
    assert "`general-purpose`" in meta
    for stale in (
        "data_audit",
        "strategy_audit",
        "trace_audit",
        "strategy_performance_audit",
        "context_audit",
    ):
        assert stale not in fold
        assert stale not in meta


def test_explore_schema_uses_session_role_enum() -> None:
    explore = ExploreSubAgentEngine(
        llm=ScriptedLLM([]),
        tools=ToolRegistry([DeclaredReadOnlyShell()]),
    )
    runner = AgentSessionRunner(
        llm=ScriptedLLM([]),
        tools=ToolRegistry(),
        system_prompt="fold",
        config=_fold_config(*FOLD_REQUIRED_EXPLORE_ROLES),
        explore=explore,
    )
    schema = next(
        _function_payload(tool)
        for tool in runner._provider_tools()
        if _function_name(tool) == "explore"
    )
    parameters = schema["parameters"]
    assert isinstance(parameters, dict)
    assert parameters["required"] == ["role", "task"]
    properties = parameters["properties"]
    assert isinstance(properties, dict)
    role_schema = properties["role"]
    assert isinstance(role_schema, dict)
    assert role_schema["enum"] == list(EXPLORE_ROLES)
    assert role_schema["enum"] == ["auditor", "developer", "general-purpose", "Explore"]
    meta_runner = AgentSessionRunner(
        llm=ScriptedLLM([]),
        tools=ToolRegistry(),
        system_prompt="meta",
        config=_meta_config(*META_REQUIRED_EXPLORE_ROLES),
        explore=ExploreSubAgentEngine(
            llm=ScriptedLLM([]),
            tools=ToolRegistry([_NamedTool("read_file")]),
            mode="meta",
        ),
    )
    meta_schema = next(
        _function_payload(tool)
        for tool in meta_runner._provider_tools()
        if _function_name(tool) == "explore"
    )
    meta_parameters = meta_schema["parameters"]
    assert isinstance(meta_parameters, dict)
    meta_properties = meta_parameters["properties"]
    assert isinstance(meta_properties, dict)
    meta_role = meta_properties["role"]
    assert isinstance(meta_role, dict)
    assert meta_role["enum"] == list(EXPLORE_ROLES)


def test_session_explore_roles_are_stable_required_sets() -> None:
    assert session_explore_roles("fold") == FOLD_REQUIRED_EXPLORE_ROLES
    assert session_explore_roles("meta") == META_REQUIRED_EXPLORE_ROLES
    assert session_explore_roles("meta_learning") == META_REQUIRED_EXPLORE_ROLES
    assert OPTIONAL_EXPLORE_ROLES.isdisjoint(session_explore_roles("fold"))
    assert OPTIONAL_EXPLORE_ROLES.isdisjoint(session_explore_roles("meta"))
    with pytest.raises(ValueError, match="cannot be a required"):
        AgentSessionConfig(
            mode="fold",
            required_explore_roles=(*FOLD_REQUIRED_EXPLORE_ROLES, "general-purpose"),
        )
    with pytest.raises(ValueError, match="cannot be a required"):
        AgentSessionConfig(
            mode="fold",
            required_explore_roles=("auditor", "Explore"),
        )
    with pytest.raises(ValueError, match="unknown explore role for meta"):
        AgentSessionConfig(
            mode="meta",
            required_explore_roles=("developer",),
        )


def test_role_tool_visibility_hides_writes_from_audits(tmp_path: Path) -> None:
    from autotrade.agent.runner import _FOLD_TOOLS

    assert "write_file" not in _FOLD_TOOLS
    assert "edit_file" not in _FOLD_TOOLS
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
    engine = ExploreSubAgentEngine(llm=ScriptedLLM([]), tools=ToolRegistry(tools))
    impl = {
        _function_name(record)
        for record in engine._provider_tools(allowed_explore_tools("fold", "developer"))
    }
    audit = {
        _function_name(record)
        for record in engine._provider_tools(allowed_explore_tools("fold", "auditor"))
    }
    assert {"write_file", "edit_file", "shell", "todo"} <= impl
    assert {"read_file", "grep", "glob", "shell", "todo"} <= audit
    assert "write_file" not in audit
    assert "edit_file" not in audit
    fold_general = {
        _function_name(record)
        for record in engine._provider_tools(
            allowed_explore_tools("fold", "general-purpose")
        )
    }
    assert {"write_file", "edit_file"} <= fold_general
    assert fold_general == impl
    meta_engine = ExploreSubAgentEngine(
        llm=ScriptedLLM([]),
        tools=ToolRegistry(build_meta_explore_tools(SearchRoots(safe), safe)),
        mode="meta",
    )
    meta_names = {
        _function_name(record)
        for record in meta_engine._provider_tools(
            allowed_explore_tools("meta", "auditor")
        )
    }
    meta_general = {
        _function_name(record)
        for record in meta_engine._provider_tools(
            allowed_explore_tools("meta", "general-purpose")
        )
    }
    fold_explore = {
        _function_name(record)
        for record in engine._provider_tools(allowed_explore_tools("fold", "Explore"))
    }
    assert fold_explore == {"read_file", "grep", "glob", "todo"}
    assert "shell" not in fold_explore
    assert "write_file" not in fold_explore
    meta_developer = {
        _function_name(record)
        for record in meta_engine._provider_tools(
            allowed_explore_tools("meta", "developer")
        )
    }
    meta_explore = {
        _function_name(record)
        for record in meta_engine._provider_tools(
            allowed_explore_tools("meta", "Explore")
        )
    }
    assert meta_names == {"read_file", "grep", "glob", "todo"}
    assert meta_general == meta_names
    assert meta_developer == meta_names
    assert meta_explore == meta_names
    assert "write_file" not in meta_general
    assert "edit_file" not in meta_general


def test_only_general_cannot_finish_required_roles() -> None:
    finish = _FinishStub("finish_fold")
    explore = ExploreSubAgentEngine(
        llm=ScriptedLLM(
            [
                ProviderResponse(content="general digest"),
                ProviderResponse(content="data digest"),
                ProviderResponse(content="strategy digest"),
            ]
        ),
        tools=ToolRegistry([DeclaredReadOnlyShell()]),
    )
    events: list[tuple[str, dict[str, object]]] = []
    runner = AgentSessionRunner(
        llm=ScriptedLLM(
            [
                ProviderResponse(
                    tool_calls=(
                        ToolCall(
                            "g1",
                            "explore",
                            {"role": "general-purpose", "task": "cross-cut"},
                        ),
                    )
                ),
                ProviderResponse(tool_calls=(ToolCall("f1", "finish_fold", {}),)),
                ProviderResponse(
                    tool_calls=(
                        ToolCall(
                            "e1",
                            "explore",
                            {"role": "auditor", "task": "check data"},
                        ),
                    )
                ),
                ProviderResponse(
                    tool_calls=(
                        ToolCall(
                            "e2",
                            "explore",
                            {"role": "developer", "task": "check strategy"},
                        ),
                    )
                ),
                ProviderResponse(tool_calls=(ToolCall("f2", "finish_fold", {}),)),
            ]
        ),
        tools=ToolRegistry([finish]),
        system_prompt="fold",
        config=_fold_config("auditor", "developer"),
        explore=explore,
        event_sink=lambda event, payload: events.append((event, payload)),
    )
    result = runner.run("general is optional")
    assert result.status == "finished"
    assert finish.invoked == 1
    assert runner._explore_attempts == 3
    assert runner._explored_roles == {
        "general-purpose",
        "auditor",
        "developer",
    }
    finish_errors = []
    for event, payload in events:
        record = payload.get("result")
        if (
            event == "tool_call"
            and payload.get("tool") == "finish_fold"
            and isinstance(record, dict)
            and record.get("ok") is False
        ):
            finish_errors.append(str(record.get("error") or ""))
    assert any("missing roles: auditor, developer" in error for error in finish_errors)
    attempt_events = [payload for event, payload in events if event == "explore_attempt"]
    assert attempt_events[0]["role"] == "general-purpose"
    assert "task" not in attempt_events[0]
    ended = next(payload for event, payload in events if event == "session_end")
    assert ended["explored_roles"] == [
        "auditor",
        "developer",
        "general-purpose",
    ]
    assert "task" not in ended


def test_required_roles_can_finish_without_general() -> None:
    finish = _FinishStub("finish_fold")
    explore = ExploreSubAgentEngine(
        llm=ScriptedLLM([ProviderResponse(content="digest")]),
        tools=ToolRegistry([DeclaredReadOnlyShell()]),
    )
    runner = AgentSessionRunner(
        llm=ScriptedLLM(
            [
                ProviderResponse(
                    tool_calls=(
                        ToolCall(
                            "e1",
                            "explore",
                            {"role": "auditor", "task": "check data"},
                        ),
                    )
                ),
                ProviderResponse(tool_calls=(ToolCall("f1", "finish_fold", {}),)),
            ]
        ),
        tools=ToolRegistry([finish]),
        system_prompt="fold",
        config=_fold_config("auditor"),
        explore=explore,
    )
    assert runner.run("specialized roles suffice").status == "finished"
    assert finish.invoked == 1
    assert runner._explored_roles == {"auditor"}


def test_general_prompts_explain_mode_and_role() -> None:
    fold = explore_system_prompt("fold", "general-purpose")
    meta = explore_system_prompt("meta", "general-purpose")
    assert "角色 general-purpose" in fold
    assert "write_file/edit_file" in fold
    assert "不能替代 auditor 或 developer" in fold
    assert "`general-purpose`" in meta
    assert "只读" in meta
    assert "write_file" in meta
    assert "没有 write_file" in meta
