"""Explore is a one-level writable Fold coding sub-agent on the parent trace and budget."""

from __future__ import annotations

import threading
import time
from collections.abc import Mapping
from pathlib import Path

import pytest

from autotrade.agent.explore import (
    DEFAULT_EXPLORE_MAX_CONCURRENT,
    DEFAULT_EXPLORE_THINKING,
    EXPLORE_ROLES,
    EXPLORE_SYSTEM_PROMPT,
    EXPLORE_THINKING_LEVELS,
    META_EXPLORE_SYSTEM_PROMPT,
    ExploreSubAgentConfig,
    allowed_explore_tools,
    ExploreSubAgentEngine,
    explore_system_prompt,
    normalize_explore_thinking,
    parent_context_digest,
)
from autotrade.agent.prompts import HOST_GUIDELINES_ZH, build_system_prompt
from autotrade.environment.tools.base import SessionInterrupt
from autotrade.agent.runner import (
    AgentSessionConfig,
    AgentSessionDeadlineExceeded,
    AgentSessionRunner,
)
from autotrade.environment.llm import ChatMessage, ProviderResponse, ScriptedLLM, ToolCall
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
from autotrade.environment.time_budget import InferenceTimeBudget
from autotrade.pipelines.local_backend import (
    SessionBudgetLLM,
    build_fold_explore_tools,
    build_meta_explore_tools,
)

_STRATEGY = "def generate_orders(context):\n    return []\n"


def _fold_config(**kwargs: object) -> AgentSessionConfig:
    return AgentSessionConfig(mode="fold", **kwargs)  # type: ignore[arg-type]


def _meta_config(**kwargs: object) -> AgentSessionConfig:
    return AgentSessionConfig(mode="meta", **kwargs)  # type: ignore[arg-type]


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
        role="developer",
        parent_call_id="call_parent",
    )
    types = [event for event, _payload in events]
    assert types[0] == "explore_task"
    assert "explore_llm" in types
    assert "explore_tool" in types
    assert types[-1] == "explore"
    assert events[0][1]["parent_call_id"] == "call_parent"
    assert events[0][1]["role"] == "developer"
    assert "task" not in events[0][1]
    assert events[0][1]["task_id"] == result["task_id"]
    assert result["role"] == "developer"
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
        config=_fold_config(),
        explore=explore,
    )
    dispatched = runner._dispatch_explore(
        ToolCall(
            "e1",
            "explore",
            {"role": "developer", "task": "write then fail"},
        )
    )
    assert dispatched.get("status") == "started"
    finished = runner._wait_explore_jobs()
    assert finished
    dispatched = finished[-1]
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
            ProviderResponse(content="sub summary"),
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
        llm=ScriptedLLM([ProviderResponse(content="summary")]),
        tools=ToolRegistry([DeclaredReadOnlyShell()]),
    )
    runner = AgentSessionRunner(
        llm=ScriptedLLM([]),
        tools=ToolRegistry(),
        system_prompt="fold",
        config=_fold_config(),
        explore=explore,
        event_sink=lambda event, _payload: events.append(event),
    )
    dispatched = runner._dispatch_explore(
        ToolCall("e1", "explore", {"role": "auditor", "task": "read schema"})
    )
    assert dispatched.get("status") == "started"
    finished = runner._wait_explore_jobs()
    assert finished and finished[-1]["ok"] is True
    assert "explore_task" in events
    assert "explore" in events


def test_fold_auditor_cannot_invoke_the_registered_shell() -> None:
    shell = DeclaredReadOnlyShell()
    llm = ScriptedLLM(
        [
            ProviderResponse(tool_calls=(ToolCall("s", "shell", {"argv": ["ls"]}),)),
            ProviderResponse(content="shell blocked"),
        ]
    )
    result = ExploreSubAgentEngine(
        llm=llm,
        tools=ToolRegistry([shell]),
    ).run("inspect without shell", role="auditor")
    assert result["status"] == "completed"
    assert result["summary"] == "shell blocked"
    assert shell.calls == []
    assert result["tool_calls"] == 0


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
    assert result["summary"] == "unknown tool blocked"
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
        "write_skill",
        "delete_skill",
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


def test_finish_fold_allows_zero_explore_and_emits_empty_trace_stats() -> None:
    finish = _FinishStub("finish_fold")
    events: list[tuple[str, dict[str, object]]] = []
    runner = AgentSessionRunner(
        llm=ScriptedLLM(
            [ProviderResponse(tool_calls=(ToolCall("f1", "finish_fold", {}),))]
        ),
        tools=ToolRegistry([finish]),
        system_prompt="fold",
        config=_fold_config(),
        explore=ExploreSubAgentEngine(
            llm=ScriptedLLM([]),
            tools=ToolRegistry([DeclaredReadOnlyShell()]),
        ),
        event_sink=lambda event, payload: events.append((event, payload)),
    )

    result = runner.run("finish without delegation")

    assert result.status == "finished"
    assert finish.invoked == 1
    assert runner._explore_attempts == 0
    assert runner._explored_roles == set()
    assert not [payload for event, payload in events if event == "explore_attempt"]
    ended = next(payload for event, payload in events if event == "session_end")
    assert ended["explore_attempts"] == 0
    assert ended["explored_roles"] == []


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
        config=_fold_config(),
        explore=explore,
    )
    assert runner.run("failed explore is still traced").status == "finished"
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
    assert runner.run("new session without explore").status == "finished"
    assert finish.invoked == 1
    assert runner._explore_attempts == 0
    assert runner._explored_roles == set()


def test_zero_explore_enters_hard_finalization() -> None:
    explore = ExploreSubAgentEngine(
        llm=ScriptedLLM([ProviderResponse(content="summary")]),
        tools=ToolRegistry([DeclaredReadOnlyShell()]),
    )
    runner = AgentSessionRunner(
        llm=ScriptedLLM([]),
        tools=ToolRegistry([_FinishStub("finish_fold")]),
        system_prompt="fold",
        config=_fold_config(
            finalize_before_deadline_seconds=300.0,
            deadline_grace_seconds=0.0,
        ),
        explore=explore,
    )
    runner._complete_validation_nodes = [
        {"node_id": "node_a", "revision_id": "rev_a"}
    ]
    assert runner._explore_attempts == 0
    assert runner._explored_roles == set()
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


def test_meta_runner_allows_finish_without_explore_attempt() -> None:
    finish = _FinishStub("finish_meta")
    explore = ExploreSubAgentEngine(
        llm=ScriptedLLM([ProviderResponse(content="trace reviewed")]),
        tools=ToolRegistry([_NamedTool("read_file")]),
        mode="meta",
    )
    runner = AgentSessionRunner(
        llm=ScriptedLLM(
            [ProviderResponse(tool_calls=(ToolCall("f1", "finish_meta", {}),))]
        ),
        tools=ToolRegistry([finish]),
        system_prompt="meta",
        config=_meta_config(),
        explore=explore,
    )
    result = runner.run("meta can finish directly")
    assert result.status == "finished"
    assert finish.invoked == 1
    assert runner._explore_attempts == 0
    assert runner._explored_roles == set()


def test_fold_and_explore_prompts_keep_roles_without_pyright_how_to() -> None:
    fold = build_system_prompt(mode="fold", experiment_facts={})
    meta = build_system_prompt(mode="meta", experiment_facts={})
    command = (
        "pyright --project /opt/autotrade/pyrightconfig.json "
        "/mnt/agent/workspace /mnt/agent/output"
    )
    assert command not in fold
    assert command not in EXPLORE_SYSTEM_PROMPT
    assert "pyright" not in meta
    assert command not in META_EXPLORE_SYSTEM_PROMPT
    for role in ("`auditor`", "`developer`", "`general-purpose`", "`Explore`"):
        assert role in fold
        assert role in meta
    assert "除非任务非常简单" in fold
    assert "`write_file`" in fold
    assert "`finish_fold`" in fold
    assert "`finish_meta`" in meta
    assert "通常优先" not in fold
    assert "逐个读取" not in meta
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
        config=_fold_config(),
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
    thinking_schema = properties["thinking"]
    assert isinstance(thinking_schema, dict)
    assert thinking_schema["enum"] == list(EXPLORE_THINKING_LEVELS)
    assert properties["inherit_context"]["type"] == "boolean"
    assert properties["max_turns"]["type"] == "integer"
    assert "maximum" not in properties["max_turns"]
    assert "maxLength" not in properties["task"]
    meta_runner = AgentSessionRunner(
        llm=ScriptedLLM([]),
        tools=ToolRegistry(),
        system_prompt="meta",
        config=_meta_config(),
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


def test_session_config_has_no_required_explore_role_gate() -> None:
    config = AgentSessionConfig(mode="fold")
    assert not hasattr(config, "required_explore_roles")
    assert "required_explore_roles" not in AgentSessionConfig.__dataclass_fields__


def test_role_tool_visibility_hides_writes_from_audits(tmp_path: Path) -> None:
    from autotrade.agent.runner import _FOLD_TOOLS

    assert "write_file" in _FOLD_TOOLS
    assert "edit_file" in _FOLD_TOOLS
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
    assert impl == {
        "delete_skill",
        "edit_file",
        "glob",
        "grep",
        "modification_check",
        "read_file",
        "shell",
        "todo",
        "validate_strategy",
        "write_file",
        "write_skill",
    }
    assert audit == {"glob", "grep", "read_file", "todo"}
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


def test_explore_calls_still_track_attempts_and_roles() -> None:
    finish = _FinishStub("finish_fold")
    explore = ExploreSubAgentEngine(
        llm=ScriptedLLM(
            [
                ProviderResponse(content="general summary"),
                ProviderResponse(content="data summary"),
                ProviderResponse(content="strategy summary"),
            ]
        ),
        tools=ToolRegistry([DeclaredReadOnlyShell()]),
        config=ExploreSubAgentConfig(max_concurrent=3),
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
                        ToolCall(
                            "e1",
                            "explore",
                            {"role": "auditor", "task": "check data"},
                        ),
                        ToolCall(
                            "e2",
                            "explore",
                            {"role": "developer", "task": "check strategy"},
                        ),
                    )
                ),
                ProviderResponse(tool_calls=(ToolCall("f1", "finish_fold", {}),)),
            ]
        ),
        tools=ToolRegistry([finish]),
        system_prompt="fold",
        config=_fold_config(),
        explore=explore,
        event_sink=lambda event, payload: events.append((event, payload)),
    )

    result = runner.run("delegate when useful")

    assert result.status == "finished"
    assert finish.invoked == 1
    assert runner._explore_attempts == 3
    assert runner._explored_roles == {
        "general-purpose",
        "auditor",
        "developer",
    }
    attempt_events = [payload for event, payload in events if event == "explore_attempt"]
    assert [payload["role"] for payload in attempt_events] == [
        "general-purpose",
        "auditor",
        "developer",
    ]
    assert all("task" not in payload for payload in attempt_events)
    ended = next(payload for event, payload in events if event == "session_end")
    assert ended["explore_attempts"] == 3
    assert ended["explored_roles"] == [
        "auditor",
        "developer",
        "general-purpose",
    ]
    assert "task" not in ended


def test_single_explore_role_can_finish() -> None:
    finish = _FinishStub("finish_fold")
    explore = ExploreSubAgentEngine(
        llm=ScriptedLLM([ProviderResponse(content="summary")]),
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
        config=_fold_config(),
        explore=explore,
    )
    assert runner.run("one delegated review is enough").status == "finished"
    assert finish.invoked == 1
    assert runner._explored_roles == {"auditor"}


def test_general_prompts_explain_mode_and_role() -> None:
    fold = explore_system_prompt("fold", "general-purpose")
    meta = explore_system_prompt("meta", "general-purpose")
    assert "一级 `general-purpose`" in fold
    assert "修改共享策略、模型或 skills" in fold
    assert "有界的跨域实现任务" in fold
    assert "`general-purpose`" in meta
    assert "只读" in meta
    assert "不能写策略、models、skills 或 PRIOR" in meta


def test_normalize_explore_thinking_accepts_aliases() -> None:
    assert DEFAULT_EXPLORE_THINKING == "medium"
    assert normalize_explore_thinking(None) == "medium"
    assert normalize_explore_thinking("inherit") == "medium"
    assert normalize_explore_thinking("minimal") == "low"
    assert normalize_explore_thinking("xhigh") == "high"
    assert normalize_explore_thinking("max") == "max"
    with pytest.raises(ValueError, match="explore.thinking"):
        normalize_explore_thinking("turbo")


def test_explore_defaults_are_medium_thinking_and_two_concurrent() -> None:
    assert DEFAULT_EXPLORE_MAX_CONCURRENT == 2
    assert ExploreSubAgentConfig().max_concurrent == 2
    result = ExploreSubAgentEngine(
        llm=ScriptedLLM([ProviderResponse(content="ok")]),
        tools=ToolRegistry([DeclaredReadOnlyShell()]),
    ).run("summarize", role="auditor")
    assert result["thinking"] == "medium"
    assert "没有 Sleep" in HOST_GUIDELINES_ZH
    assert "应让它 Sleep" not in HOST_GUIDELINES_ZH
    assert "Runner 会让出" in HOST_GUIDELINES_ZH
    assert "后台完成由运行时注入" in HOST_GUIDELINES_ZH


def test_parent_context_digest_skips_system_and_keeps_recent() -> None:
    digest = parent_context_digest(
        [
            ChatMessage("system", "hidden contract"),
            ChatMessage("user", "look at daily schema"),
            ChatMessage("assistant", "I will delegate"),
        ]
    )
    assert "父会话摘录" in digest
    assert "look at daily schema" in digest
    assert "I will delegate" in digest
    assert "hidden contract" not in digest


def test_explore_inherit_context_prepends_parent_digest() -> None:
    llm = ScriptedLLM([ProviderResponse(content="used parent excerpt")])
    result = ExploreSubAgentEngine(
        llm=llm,
        tools=ToolRegistry([DeclaredReadOnlyShell()]),
    ).run(
        "summarize",
        role="auditor",
        inherit_context=True,
        parent_messages=[
            ChatMessage("system", "secret"),
            ChatMessage("user", "visible parent note"),
        ],
        thinking="low",
        description="schema audit",
    )
    assert result["status"] == "completed"
    assert result["thinking"] == "low"
    assert result["inherit_context"] is True
    assert llm.calls
    first = llm.calls[0]
    assert isinstance(first, dict)
    recorded = first["messages"]
    assert isinstance(recorded, tuple)
    contents = [str(msg.content) for msg in recorded]
    assert any("visible parent note" in text for text in contents)
    assert all("secret" not in text for text in contents)
    assert any(msg.role == "user" for msg in recorded)


def test_dispatch_explore_rejects_bad_thinking() -> None:
    explore = ExploreSubAgentEngine(
        llm=ScriptedLLM([ProviderResponse(content="ok")]),
        tools=ToolRegistry([DeclaredReadOnlyShell()]),
    )
    runner = AgentSessionRunner(
        llm=ScriptedLLM([]),
        tools=ToolRegistry(),
        system_prompt="fold",
        config=_fold_config(),
        explore=explore,
    )
    record = runner._dispatch_explore(
        ToolCall("e1", "explore", {"role": "auditor", "task": "x", "thinking": "turbo"})
    )
    assert record["ok"] is False
    assert "thinking" in str(record["error"])


def test_explore_thinking_only_reply_does_not_finish_the_child() -> None:
    llm = ScriptedLLM(
        [
            ProviderResponse(content="", reasoning_content="internal plan"),
            ProviderResponse(content="done after thinking"),
        ]
    )
    result = ExploreSubAgentEngine(
        llm=llm,
        tools=ToolRegistry([DeclaredReadOnlyShell()]),
    ).run("summarize", role="auditor")
    assert result["status"] == "completed"
    assert result["summary"] == "done after thinking"
    assert result["llm_calls"] == 2


class _GateLLM:
    model = "child"
    provider = "test"
    context_window_tokens = 128000

    def __init__(self, started: threading.Event, release: threading.Event) -> None:
        self.started = started
        self.release = release

    def complete(self, messages, **kwargs):
        self.started.set()
        if not self.release.wait(3):
            raise TimeoutError("explore gate")
        return ProviderResponse(content="child done")


class _TaskGatedLLM:
    model = "child"
    provider = "test"
    context_window_tokens = 128000

    def __init__(
        self,
        gates: dict[str, tuple[threading.Event, threading.Event, str]],
    ) -> None:
        self.gates = gates

    def complete(self, messages, **kwargs):
        del kwargs
        blob = " ".join(str(message.content or "") for message in messages)
        for needle, (started, release, summary) in self.gates.items():
            if needle in blob:
                started.set()
                if not release.wait(5):
                    raise TimeoutError(needle)
                return ProviderResponse(content=summary)
        raise AssertionError(blob)


def test_parent_session_continues_before_explore_finishes() -> None:
    started = threading.Event()
    release = threading.Event()
    finish = _FinishStub("finish_fold")
    inner = ScriptedLLM(
        [
            ProviderResponse(
                tool_calls=(
                    ToolCall(
                        "e1",
                        "explore",
                        {"role": "auditor", "task": "slow look"},
                    ),
                )
            ),
            ProviderResponse(tool_calls=(ToolCall("f1", "finish_fold", {}),)),
        ]
    )
    parent_calls = {"n": 0}

    class _ParentLLM:
        model = "parent"
        provider = "test"
        context_window_tokens = 128000

        def complete(self, messages, **kwargs):
            parent_calls["n"] += 1
            if parent_calls["n"] == 2:
                assert started.wait(3)
                assert not release.is_set()
                release.set()
            return inner.complete(messages, **kwargs)

    runner = AgentSessionRunner(
        llm=_ParentLLM(),
        tools=ToolRegistry([finish]),
        system_prompt="fold",
        config=_fold_config(),
        explore=ExploreSubAgentEngine(
            llm=_GateLLM(started, release),
            tools=ToolRegistry([DeclaredReadOnlyShell()]),
        ),
    )
    result = runner.run("go")
    assert result.status == "finished"
    assert finish.invoked == 1
    assert parent_calls["n"] == 2


def test_parent_text_only_waits_for_pending_explore_then_resumes() -> None:
    started = threading.Event()
    release = threading.Event()
    call2_returned = threading.Event()
    finish = _FinishStub("finish_fold")
    inner = ScriptedLLM(
        [
            ProviderResponse(
                tool_calls=(
                    ToolCall(
                        "e1",
                        "explore",
                        {"role": "auditor", "task": "slow look"},
                    ),
                )
            ),
            ProviderResponse(content="waiting for explore"),
            ProviderResponse(tool_calls=(ToolCall("f1", "finish_fold", {}),)),
        ]
    )
    parent_calls = {"n": 0}

    class _ParentLLM:
        model = "parent"
        provider = "test"
        context_window_tokens = 128000

        def complete(self, messages, **kwargs):
            parent_calls["n"] += 1
            n = parent_calls["n"]
            if n == 2:
                assert started.wait(3)
                assert not release.is_set()

                def _release_after_parent_turn() -> None:
                    assert call2_returned.wait(3)
                    release.set()

                threading.Thread(
                    target=_release_after_parent_turn, daemon=True
                ).start()
            elif n == 3:
                assert release.is_set()
                blob = "\n".join(str(message.content or "") for message in messages)
                assert '"observation": "explore_completed"' in blob
                assert '"observation": "no_tool_call"' not in blob
            try:
                return inner.complete(messages, **kwargs)
            finally:
                if n == 2:
                    call2_returned.set()

    runner = AgentSessionRunner(
        llm=_ParentLLM(),
        tools=ToolRegistry([finish]),
        system_prompt="fold",
        config=_fold_config(),
        explore=ExploreSubAgentEngine(
            llm=_GateLLM(started, release),
            tools=ToolRegistry([DeclaredReadOnlyShell()]),
        ),
    )
    result = runner.run("go")
    assert result.status == "finished"
    assert finish.invoked == 1
    assert parent_calls["n"] == 3
    assert len(inner.calls) == 3


def test_parent_text_only_without_pending_explore_still_nudges() -> None:
    finish = _FinishStub("finish_fold")
    llm = ScriptedLLM(
        [
            ProviderResponse(content="I should act next."),
            ProviderResponse(tool_calls=(ToolCall("f1", "finish_fold", {}),)),
        ]
    )
    runner = AgentSessionRunner(
        llm=llm,
        tools=ToolRegistry([finish]),
        system_prompt="fold",
        config=_fold_config(),
        explore=ExploreSubAgentEngine(
            llm=ScriptedLLM([ProviderResponse(content="unused")]),
            tools=ToolRegistry([DeclaredReadOnlyShell()]),
        ),
    )
    result = runner.run("go")
    assert result.status == "finished"
    assert result.llm_calls == 2
    second = llm.calls[1]["messages"]
    assert isinstance(second, tuple)
    assert any(
        '"observation": "no_tool_call"' in (message.content or "") for message in second
    )


def test_parent_text_only_wakes_on_first_completed_explore() -> None:
    fast_started = threading.Event()
    fast_release = threading.Event()
    slow_started = threading.Event()
    slow_release = threading.Event()
    call2_returned = threading.Event()
    finish = _FinishStub("finish_fold")
    inner = ScriptedLLM(
        [
            ProviderResponse(
                tool_calls=(
                    ToolCall(
                        "e1",
                        "explore",
                        {"role": "auditor", "task": "fast-task"},
                    ),
                    ToolCall(
                        "e2",
                        "explore",
                        {"role": "developer", "task": "slow-task"},
                    ),
                )
            ),
            ProviderResponse(content="waiting for first explore"),
            ProviderResponse(tool_calls=(ToolCall("f1", "finish_fold", {}),)),
        ]
    )
    parent_calls = {"n": 0}

    class _ParentLLM:
        model = "parent"
        provider = "test"
        context_window_tokens = 128000

        def complete(self, messages, **kwargs):
            parent_calls["n"] += 1
            n = parent_calls["n"]
            if n == 2:
                assert fast_started.wait(3)
                assert slow_started.wait(3)
                assert not fast_release.is_set()
                assert not slow_release.is_set()

                def _release_fast() -> None:
                    assert call2_returned.wait(3)
                    fast_release.set()

                threading.Thread(target=_release_fast, daemon=True).start()
            elif n == 3:
                blob = "\n".join(str(message.content or "") for message in messages)
                assert blob.count('"observation": "explore_completed"') == 1
                assert "fast summary" in blob
                assert "slow summary" not in blob
                assert '"observation": "no_tool_call"' not in blob
                slow_release.set()
            try:
                return inner.complete(messages, **kwargs)
            finally:
                if n == 2:
                    call2_returned.set()

    runner = AgentSessionRunner(
        llm=_ParentLLM(),
        tools=ToolRegistry([finish]),
        system_prompt="fold",
        config=_fold_config(),
        explore=ExploreSubAgentEngine(
            llm=_TaskGatedLLM(
                {
                    "fast-task": (fast_started, fast_release, "fast summary"),
                    "slow-task": (slow_started, slow_release, "slow summary"),
                }
            ),
            tools=ToolRegistry([DeclaredReadOnlyShell()]),
            config=ExploreSubAgentConfig(max_concurrent=2),
        ),
    )
    result = runner.run("go")
    assert result.status == "finished"
    assert finish.invoked == 1
    assert parent_calls["n"] == 3


def test_parent_text_only_pending_explore_deadline_does_not_deadlock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from autotrade.agent import runner as runner_module

    monkeypatch.setattr(runner_module, "EXPLORE_TEARDOWN_WAIT_SECONDS", 0.2)
    monkeypatch.setattr(runner_module, "EXPLORE_TEARDOWN_WAIT_REMAINDER_SECONDS", 0.05)
    started = threading.Event()
    release = threading.Event()
    finish = _FinishStub("finish_fold")
    inner = ScriptedLLM(
        [
            ProviderResponse(
                tool_calls=(
                    ToolCall(
                        "e1",
                        "explore",
                        {"role": "auditor", "task": "hang"},
                    ),
                )
            ),
            ProviderResponse(content="waiting for explore"),
        ]
    )
    parent_calls = {"n": 0}

    class _ParentLLM:
        model = "parent"
        provider = "test"
        context_window_tokens = 128000

        def complete(self, messages, **kwargs):
            parent_calls["n"] += 1
            if parent_calls["n"] == 2:
                assert started.wait(3)
            return inner.complete(messages, **kwargs)

    t0 = time.monotonic()
    try:
        with pytest.raises(AgentSessionDeadlineExceeded):
            AgentSessionRunner(
                llm=_ParentLLM(),
                tools=ToolRegistry([finish]),
                system_prompt="fold",
                config=_fold_config(
                    deadline_seconds=0.4,
                    deadline_grace_seconds=0.0,
                    finalize_before_deadline_seconds=0.0,
                ),
                explore=ExploreSubAgentEngine(
                    llm=_GateLLM(started, release),
                    tools=ToolRegistry([DeclaredReadOnlyShell()]),
                ),
            ).run("go")
        elapsed = time.monotonic() - t0
        assert parent_calls["n"] == 2
        assert elapsed < 2.5
    finally:
        release.set()
        time.sleep(0.2)


def test_wait_first_pending_explore_returns_on_cancel() -> None:
    started = threading.Event()
    release = threading.Event()
    runner = AgentSessionRunner(
        llm=ScriptedLLM([]),
        tools=ToolRegistry(),
        system_prompt="fold",
        config=_fold_config(),
        explore=ExploreSubAgentEngine(
            llm=_GateLLM(started, release),
            tools=ToolRegistry([DeclaredReadOnlyShell()]),
        ),
    )
    dispatched = runner._dispatch_explore(
        ToolCall("e1", "explore", {"role": "auditor", "task": "hang"})
    )
    assert dispatched.get("status") == "started"
    assert started.wait(3)
    threading.Timer(0.1, runner._cancelled.set).start()
    t0 = time.monotonic()
    try:
        runner._wait_first_pending_explore(InferenceTimeBudget(duration_seconds=20))
        assert time.monotonic() - t0 < 2.0
    finally:
        release.set()


def test_explore_stops_retry_on_call_budget_and_interrupt() -> None:
    class Exhausted:
        model = "child"
        provider = "test"
        context_window_tokens = 128000

        def __init__(self) -> None:
            self.calls = 0

        def complete(self, messages, **kwargs):
            del messages, kwargs
            self.calls += 1
            raise RuntimeError("Agent session LLM call budget exhausted")

    exhausted = Exhausted()
    result = ExploreSubAgentEngine(
        llm=exhausted,
        tools=ToolRegistry([DeclaredReadOnlyShell()]),
        config=ExploreSubAgentConfig(max_rounds=5),
    ).run("look", role="auditor")
    assert result["status"] == "error"
    assert result["llm_calls"] == 1
    assert exhausted.calls == 1

    class Interrupted:
        model = "child"
        provider = "test"
        context_window_tokens = 128000

        def __init__(self) -> None:
            self.calls = 0

        def complete(self, messages, **kwargs):
            del messages, kwargs
            self.calls += 1
            raise SessionInterrupt("researcher stop")

    interrupted = Interrupted()
    result = ExploreSubAgentEngine(
        llm=interrupted,
        tools=ToolRegistry([DeclaredReadOnlyShell()]),
        config=ExploreSubAgentConfig(max_rounds=5),
    ).run("look", role="auditor")
    assert result["status"] == "error"
    assert interrupted.calls == 1


def test_explore_skips_tools_when_cancelled_after_llm() -> None:
    started = threading.Event()
    release = threading.Event()
    shell = DeclaredReadOnlyShell()

    class LateTools:
        model = "child"
        provider = "test"
        context_window_tokens = 128000

        def complete(self, messages, **kwargs):
            del messages, kwargs
            started.set()
            assert release.wait(3)
            return ProviderResponse(
                tool_calls=(
                    ToolCall("s1", "shell", {"argv": ["echo", "late"]}),
                )
            )

    engine = ExploreSubAgentEngine(
        llm=LateTools(),
        tools=ToolRegistry([shell]),
    )
    outcome: dict[str, object] = {}

    def run() -> None:
        outcome.update(engine.run("look", role="developer"))

    worker = threading.Thread(target=run)
    worker.start()
    assert started.wait(3)
    engine.cancel()
    release.set()
    worker.join(3)
    assert not worker.is_alive()
    assert outcome.get("status") == "cancelled"
    assert shell.calls == []


def test_runner_close_cancels_explore_without_infinite_wait(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from autotrade.agent import runner as runner_module

    monkeypatch.setattr(runner_module, "EXPLORE_TEARDOWN_WAIT_SECONDS", 0.2)
    monkeypatch.setattr(runner_module, "EXPLORE_TEARDOWN_WAIT_REMAINDER_SECONDS", 0.05)
    started = threading.Event()
    release = threading.Event()
    shell = DeclaredReadOnlyShell()
    finish = _FinishStub("finish_fold")

    class BlockingChild:
        model = "child"
        provider = "test"
        context_window_tokens = 128000

        def complete(self, messages, **kwargs):
            del messages, kwargs
            started.set()
            release.wait(5)
            return ProviderResponse(
                tool_calls=(
                    ToolCall("s1", "shell", {"argv": ["echo", "late"]}),
                )
            )

    parent_calls = {"n": 0}
    inner = ScriptedLLM(
        [
            ProviderResponse(
                tool_calls=(
                    ToolCall(
                        "e1",
                        "explore",
                        {"role": "developer", "task": "slow"},
                    ),
                )
            ),
            ProviderResponse(tool_calls=(ToolCall("f1", "finish_fold", {}),)),
        ]
    )

    class Parent:
        model = "parent"
        provider = "test"
        context_window_tokens = 128000

        def complete(self, messages, **kwargs):
            parent_calls["n"] += 1
            if parent_calls["n"] == 2:
                assert started.wait(3)
            return inner.complete(messages, **kwargs)

    t0 = time.monotonic()
    result = AgentSessionRunner(
        llm=Parent(),
        tools=ToolRegistry([finish]),
        system_prompt="fold",
        config=_fold_config(),
        explore=ExploreSubAgentEngine(
            llm=BlockingChild(),
            tools=ToolRegistry([shell]),
        ),
    ).run("go")
    elapsed = time.monotonic() - t0
    assert result.status == "finished"
    assert elapsed < 2.0
    release.set()
    time.sleep(0.2)
    assert shell.calls == []
