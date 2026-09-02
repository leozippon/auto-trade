"""The agent tool: a one-level background sub-agent registered as a normal tool on the parent trace and budget."""

from __future__ import annotations

import json
import threading
import time
from collections import deque
from collections.abc import Mapping
from pathlib import Path

import pytest

from autotrade.agent.subagent import (
    AGENT_TOOL_DESCRIPTION,
    AGENT_TOOL_SPEC,
    DEFAULT_SUBAGENT_MAX_CONCURRENT,
    DEFAULT_SUBAGENT_THINKING,
    STEER_MESSAGE_LABEL,
    SUBAGENT_DESCRIPTION_MAX_CHARS,
    SUBAGENT_ROLES,
    SUBAGENT_STEER_MAX_CHARS,
    SUBAGENT_THINKING_LEVELS,
    META_SUBAGENT_SYSTEM_PROMPT,
    SubAgentConfig,
    allowed_subagent_tools,
    SubAgentEngine,
    subagent_system_prompt,
    normalize_subagent_thinking,
)
from autotrade.agent import subagent as subagent_module
from autotrade.agent.prompts import FOLD_WORKFLOW_SECTION, build_system_prompt
from autotrade.environment.tools.base import SessionInterrupt
from autotrade.agent.runner import (
    AgentSessionConfig,
    AgentSessionDeadlineExceeded,
    AgentSessionRunner,
)
from autotrade.environment.llm import (
    ChatMessage,
    MalformedToolCallError,
    ProviderResponse,
    ScriptedLLM,
    ToolCall,
)
from autotrade.environment.tools import (
    CommandResult,
    EditFileTool,
    ModificationCheckTool,
    SafeWorkspace,
    SandboxShellTool,
    SearchRoots,
    ToolRegistry,
    ToolResult,
    ToolSpec,
    WriteFileTool,
)
from autotrade.environment.time_budget import InferenceTimeBudget
from autotrade.pipelines.local_backend import (
    SessionBudgetLLM,
    SessionCallBudget,
    build_fold_subagent_tools,
    build_meta_subagent_tools,
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


def test_subagent_events_land_on_the_parent_fold_trace() -> None:
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
    result = SubAgentEngine(
        llm=llm,
        tools=ToolRegistry([shell]),
        event_sink=lambda event, payload: events.append((event, payload)),
    ).run(
        "inspect snapshot schema",
        role="developer",
        parent_call_id="call_parent",
    )
    types = [event for event, _payload in events]
    assert types[0] == "subagent_task"
    assert "subagent_llm" in types
    assert "subagent_tool" in types
    assert types[-1] == "subagent"
    assert events[0][1]["parent_call_id"] == "call_parent"
    assert events[0][1]["role"] == "developer"
    # The brief is traced (clipped like tool arguments); a scripted double
    # cannot carry a thinking level, and the trace says so.
    assert events[0][1]["task"] == "inspect snapshot schema"
    assert events[0][1]["thinking_applied"] is False
    assert events[0][1]["task_id"] == result["task_id"]
    assert result["role"] == "developer"
    tool_event = next(payload for event, payload in events if event == "subagent_tool")
    assert tool_event["tool"] == "shell"
    assert tool_event["parent_call_id"] == "call_parent"
    assert result["status"] == "completed"


def test_subagent_rejects_nested_agent_and_fold_control_specs() -> None:
    class NamedReadOnly:
        def __init__(self, name: str) -> None:
            self.spec = ToolSpec(
                name,
                "not allowed on a sub-agent",
                {"type": "object", "properties": {}, "required": []},
            )

        def invoke(self, arguments: Mapping[str, object]) -> ToolResult:
            del arguments
            return ToolResult(True, value={})

    for name in (
        "agent",
        "daily_backtest",
        "finish_fold",
        "step_rollback",
        "ask_user",
        "unknown_tool",
    ):
        with pytest.raises(ValueError, match="not allowed"):
            SubAgentEngine(
                llm=ScriptedLLM([]),
                tools=ToolRegistry([NamedReadOnly(name)]),
            )


def test_subagent_write_edit_shell_and_checks_persist(tmp_path: Path) -> None:
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
                tool_calls=(ToolCall("m", "modification_check", {}),)
            ),
            ProviderResponse(content="结论：已写入并可验证。"),
        ]
    )
    result = SubAgentEngine(
        llm=llm,
        tools=ToolRegistry(
            [
                WriteFileTool(safe),
                EditFileTool(safe),
                SandboxShellTool(safe, runner),
                ModificationCheckTool(workspace / "output"),
            ]
        ),
    ).run("write and check the strategy", role="developer")
    assert result["status"] == "completed"
    written = (workspace / "output" / "main.py").read_text(encoding="utf-8")
    assert "return []  # ok" in written
    assert (workspace / "from_shell.txt").read_text(encoding="utf-8") == "from-shell\n"
    assert runner.calls == [["touch", "from_shell.txt"]]
    assert result["tool_calls"] == 4


class LockstepLLM:
    """Scripted replies gated on a shared barrier, so two children interleave
    their tool rounds instead of running back to back."""

    provider = "scripted"
    model = "scripted"
    context_window_tokens = None

    def __init__(self, responses, barrier: threading.Barrier) -> None:
        self._responses = deque(responses)
        self._barrier = barrier

    def complete(
        self,
        messages,
        *,
        tools=(),
        tool_choice="auto",
        max_tokens=None,
    ) -> ProviderResponse:
        del messages, tools, tool_choice, max_tokens
        self._barrier.wait(timeout=30)
        if not self._responses:
            raise RuntimeError("LockstepLLM has no response remaining")
        return self._responses.popleft()


def _candidate_writer(name: str, barrier: threading.Barrier) -> LockstepLLM:
    return LockstepLLM(
        [
            ProviderResponse(
                tool_calls=(
                    ToolCall(
                        f"{name}-main",
                        "write_file",
                        {
                            "path": f"candidates/{name}/main.py",
                            "content": f"# {name}\n{_STRATEGY}",
                        },
                    ),
                )
            ),
            ProviderResponse(
                tool_calls=(
                    ToolCall(
                        f"{name}-lib",
                        "write_file",
                        {
                            "path": f"candidates/{name}/lib/fit.py",
                            "content": f"NAME = {name!r}\n",
                        },
                    ),
                )
            ),
            ProviderResponse(content=f"结论：{name} 已写入。"),
        ],
        barrier,
    )


def test_concurrent_subagents_do_not_drop_each_others_workspace_writes(
    tmp_path: Path,
) -> None:
    """Concurrent children write one live workspace: no private copy is taken
    at launch and nothing is merged back at completion, so disjoint paths must
    all survive whichever child finishes last."""

    workspace = tmp_path / "agent"
    workspace.mkdir()
    safe = SafeWorkspace(workspace)
    # One registry over one SafeWorkspace serves every child, as the Fold
    # session builds it.
    tools = ToolRegistry([WriteFileTool(safe), EditFileTool(safe)])
    barrier = threading.Barrier(2)
    results: dict[str, dict[str, object]] = {}

    def run(name: str) -> None:
        results[name] = SubAgentEngine(
            llm=_candidate_writer(name, barrier), tools=tools
        ).run(f"实现候选 {name}", role="developer")

    threads = [threading.Thread(target=run, args=(name,)) for name in ("g20", "h1")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
    assert not [thread for thread in threads if thread.is_alive()]
    assert {name: record["status"] for name, record in results.items()} == {
        "g20": "completed",
        "h1": "completed",
    }
    assert {
        item.relative_to(workspace).as_posix()
        for item in workspace.rglob("*")
        if item.is_file()
    } == {
        "candidates/g20/main.py",
        "candidates/g20/lib/fit.py",
        "candidates/h1/main.py",
        "candidates/h1/lib/fit.py",
    }
    for name in ("g20", "h1"):
        candidate = workspace / "candidates" / name
        assert candidate.joinpath("main.py").read_text(encoding="utf-8").startswith(
            f"# {name}"
        )
        assert candidate.joinpath("lib", "fit.py").read_text(
            encoding="utf-8"
        ) == f"NAME = {name!r}\n"


def test_later_subagent_write_to_one_path_wins_and_leaves_the_rest(
    tmp_path: Path,
) -> None:
    """Same shared tree, same path: the later write replaces that file only.

    There is no merge and no rollback of the earlier child, so the first
    child's other file stays exactly as it left it."""

    workspace = tmp_path / "agent"
    workspace.mkdir()
    safe = SafeWorkspace(workspace)
    tools = ToolRegistry([WriteFileTool(safe), EditFileTool(safe)])
    for name in ("first", "second"):
        result = SubAgentEngine(
            llm=ScriptedLLM(
                [
                    ProviderResponse(
                        tool_calls=(
                            ToolCall(
                                f"{name}-shared",
                                "write_file",
                                {
                                    "path": "candidates/shared.py",
                                    "content": f"OWNER = {name!r}\n",
                                },
                            ),
                            ToolCall(
                                f"{name}-own",
                                "write_file",
                                {
                                    "path": f"candidates/{name}.py",
                                    "content": f"OWNER = {name!r}\n",
                                },
                            ),
                        )
                    ),
                    ProviderResponse(content=f"结论：{name} 已写入。"),
                ]
            ),
            tools=tools,
        ).run(f"写入 {name}", role="developer")
        assert result["status"] == "completed"
    candidates = workspace / "candidates"
    assert candidates.joinpath("shared.py").read_text(encoding="utf-8") == (
        "OWNER = 'second'\n"
    )
    assert candidates.joinpath("first.py").read_text(encoding="utf-8") == (
        "OWNER = 'first'\n"
    )
    assert candidates.joinpath("second.py").exists()


def test_subagent_write_failure_does_not_finish_parent(tmp_path: Path) -> None:
    events: list[str] = []
    subagent = SubAgentEngine(
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
    # ScriptedLLM records every request, so the child's view is inspectable.
    runner = AgentSessionRunner(
        llm=ScriptedLLM([]),
        tools=ToolRegistry(),
        system_prompt="fold",
        config=_fold_config(),
        subagent=subagent,
    )
    dispatched = runner.tools.invoke(
        "agent", {"agent": "developer", "task": "write then fail"}
    )
    assert dispatched.ok and dispatched.value["status"] == "started"
    finished = runner._wait_subagent_jobs()
    assert finished
    dispatched = finished[-1]
    value = dispatched["value"]
    # A tool bug fails that call, not the child: the exception rides back as
    # an error observation and the child continues to its summary.
    assert dispatched["ok"] is True
    assert isinstance(value, dict)
    assert value["status"] == "completed"
    assert value["summary"] == "must not run"
    tool_results = [
        message.content
        for message in subagent.llm.calls[1]["messages"]
        if message.role == "tool"
    ]
    assert tool_results and "disk exploded" in tool_results[0]
    assert '"error_type": "tool_exception"' in tool_results[0]
    assert "subagent" in events
    assert not (tmp_path / "output" / "main.py").exists()


def test_subagent_readonly_write_failure_stays_an_observation(tmp_path: Path) -> None:
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
    result = SubAgentEngine(
        llm=llm,
        tools=ToolRegistry([WriteFileTool(SafeWorkspace(workspace))]),
    ).run("overwrite readme", role="developer")
    assert result["status"] == "completed"
    assert (workspace / "output" / "README.md").read_text(encoding="utf-8") == "keep\n"


def test_subagent_and_main_share_one_session_call_budget() -> None:
    scripted = ScriptedLLM(
        [
            ProviderResponse(content="sub summary"),
            ProviderResponse(content="must remain unused"),
        ]
    )
    budgeted = SessionBudgetLLM(
        scripted, max_calls=1, deadline=time.monotonic() + 10
    )
    result = SubAgentEngine(
        llm=budgeted,
        tools=ToolRegistry([DeclaredReadOnlyShell()]),
    ).run("count rows", role="auditor")
    assert result["status"] == "completed"
    with pytest.raises(RuntimeError, match="budget exhausted"):
        budgeted.complete([])
    assert len(scripted.calls) == 1


def test_meta_runner_rejects_fold_mode_subagent() -> None:
    subagent = SubAgentEngine(
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
            subagent=subagent,
        )


def test_runner_attaches_subagent_events_to_its_sink() -> None:
    events: list[str] = []
    subagent = SubAgentEngine(
        llm=ScriptedLLM([ProviderResponse(content="summary")]),
        tools=ToolRegistry([DeclaredReadOnlyShell()]),
    )
    runner = AgentSessionRunner(
        llm=ScriptedLLM([]),
        tools=ToolRegistry(),
        system_prompt="fold",
        config=_fold_config(),
        subagent=subagent,
        event_sink=lambda event, _payload: events.append(event),
    )
    dispatched = runner.tools.invoke(
        "agent", {"agent": "auditor", "task": "read schema"}
    )
    assert dispatched.ok and dispatched.value["status"] == "started"
    finished = runner._wait_subagent_jobs()
    assert finished and finished[-1]["ok"] is True
    assert "subagent_task" in events
    assert "subagent" in events


def test_fold_auditor_cannot_invoke_the_registered_shell() -> None:
    shell = DeclaredReadOnlyShell()
    llm = ScriptedLLM(
        [
            ProviderResponse(tool_calls=(ToolCall("s", "shell", {"argv": ["ls"]}),)),
            ProviderResponse(content="shell blocked"),
        ]
    )
    result = SubAgentEngine(
        llm=llm,
        tools=ToolRegistry([shell]),
    ).run("inspect without shell", role="auditor")
    assert result["status"] == "completed"
    assert result["summary"] == "shell blocked"
    assert shell.calls == []
    assert result["tool_calls"] == 0


def test_subagent_unknown_tool_call_is_rejected_without_invoke() -> None:
    shell = DeclaredReadOnlyShell()
    llm = ScriptedLLM(
        [
            ProviderResponse(
                tool_calls=(ToolCall("b", "daily_backtest", {}),)
            ),
            ProviderResponse(content="unknown tool blocked"),
        ]
    )
    result = SubAgentEngine(
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


def test_fold_subagent_tools_are_writable_shell_contract(tmp_path: Path) -> None:
    workspace = tmp_path / "agent"
    workspace.mkdir()
    (workspace / "output").mkdir()
    safe = SafeWorkspace(workspace)
    tools = build_fold_subagent_tools(
        SearchRoots(safe),
        safe,
        _UnusedRunner(),
        ModificationCheckTool(workspace / "output"),
        _NamedTool("smoke_backtest"),
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
        "modification_check",
        "smoke_backtest",
    ]
    by_name = {tool.spec.name: tool for tool in tools}
    assert type(by_name["shell"]) is SandboxShellTool


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


def test_finish_fold_allows_zero_subagent_and_emits_empty_trace_stats() -> None:
    finish = _FinishStub("finish_fold")
    events: list[tuple[str, dict[str, object]]] = []
    runner = AgentSessionRunner(
        llm=ScriptedLLM(
            [ProviderResponse(tool_calls=(ToolCall("f1", "finish_fold", {}),))]
        ),
        tools=ToolRegistry([finish]),
        system_prompt="fold",
        config=_fold_config(),
        subagent=SubAgentEngine(
            llm=ScriptedLLM([]),
            tools=ToolRegistry([DeclaredReadOnlyShell()]),
        ),
        event_sink=lambda event, payload: events.append((event, payload)),
    )

    result = runner.run("finish without delegation")

    assert result.status == "finished"
    assert finish.invoked == 1
    assert runner._subagent_attempts == 0
    assert runner._subagent_roles == set()
    assert not [payload for event, payload in events if event == "subagent_attempt"]
    ended = next(payload for event, payload in events if event == "session_end")
    assert ended["subagent_attempts"] == 0
    assert ended["subagent_roles"] == []


def test_failed_subagent_attempt_counts_for_its_role() -> None:
    finish = _FinishStub("finish_fold")
    subagent = SubAgentEngine(
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
                            "agent",
                            {"agent": "developer", "task": "write then fail"},
                        ),
                    )
                ),
                ProviderResponse(tool_calls=(ToolCall("f1", "finish_fold", {}),)),
            ]
        ),
        tools=ToolRegistry([finish]),
        system_prompt="fold",
        config=_fold_config(),
        subagent=subagent,
    )
    assert runner.run("failed subagent is still traced").status == "finished"
    assert runner._subagent_attempts == 1
    assert runner._subagent_roles == {"developer"}
    assert finish.invoked == 1


def test_subagent_attempt_counter_resets_on_new_run() -> None:
    finish = _FinishStub("finish_fold")
    subagent = SubAgentEngine(
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
        subagent=subagent,
    )
    runner._subagent_attempts = 4
    runner._subagent_roles = {"auditor"}
    assert runner.run("new session without subagent").status == "finished"
    assert finish.invoked == 1
    assert runner._subagent_attempts == 0
    assert runner._subagent_roles == set()


def test_zero_subagent_enters_hard_finalization() -> None:
    subagent = SubAgentEngine(
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
        subagent=subagent,
    )
    runner._complete_validation_nodes = [
        {"node_id": "node_a", "revision_id": "rev_a"}
    ]
    assert runner._subagent_attempts == 0
    assert runner._subagent_roles == set()
    assert runner._activate_hard_finalization_if_ready(10.0) is True


def test_sessions_without_subagent_still_finish() -> None:
    finish = _FinishStub("finish_fold")
    runner = AgentSessionRunner(
        llm=ScriptedLLM(
            [ProviderResponse(tool_calls=(ToolCall("f1", "finish_fold", {}),))]
        ),
        tools=ToolRegistry([finish]),
        system_prompt="fold",
    )
    assert runner.run("no subagent configured").status == "finished"
    assert finish.invoked == 1


def test_meta_subagent_is_readonly_and_cannot_nest(tmp_path: Path) -> None:
    workspace = tmp_path / "agent"
    workspace.mkdir()
    (workspace / "PRIOR.md").write_text("keep\n", encoding="utf-8")
    safe = SafeWorkspace(workspace)
    tools = build_meta_subagent_tools(SearchRoots(safe))
    assert [tool.spec.name for tool in tools] == ["read_file", "grep", "glob"]
    engine = SubAgentEngine(
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
    assert "sub-agent" in META_SUBAGENT_SYSTEM_PROMPT
    assert "pyright" not in META_SUBAGENT_SYSTEM_PROMPT
    with pytest.raises(ValueError, match="not allowed"):
        SubAgentEngine(
            llm=ScriptedLLM([]),
            tools=ToolRegistry([WriteFileTool(safe)]),
            mode="meta",
        )
    with pytest.raises(ValueError, match="not allowed"):
        SubAgentEngine(
            llm=ScriptedLLM([]),
            tools=ToolRegistry([_NamedTool("agent")]),
            mode="meta",
        )


def test_meta_runner_allows_finish_without_subagent_attempt() -> None:
    finish = _FinishStub("finish_meta")
    subagent = SubAgentEngine(
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
        subagent=subagent,
    )
    result = runner.run("meta can finish directly")
    assert result.status == "finished"
    assert finish.invoked == 1
    assert runner._subagent_attempts == 0
    assert runner._subagent_roles == set()


def test_fold_and_subagent_prompts_keep_roles() -> None:
    # The pyright how-to assertion lives in test_sandbox_pyright.py.
    fold = build_system_prompt(mode="fold", experiment_facts={})
    meta = build_system_prompt(mode="meta", experiment_facts={})
    for role in ("`auditor`", "`developer`", "`general-purpose`", "`Explore`"):
        assert role in fold
        assert role in meta
    assert "保持自己的上下文精简" in fold
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


def test_subagent_schema_uses_session_role_enum() -> None:
    subagent = SubAgentEngine(
        llm=ScriptedLLM([]),
        tools=ToolRegistry([DeclaredReadOnlyShell()]),
    )
    runner = AgentSessionRunner(
        llm=ScriptedLLM([]),
        tools=ToolRegistry(),
        system_prompt="fold",
        config=_fold_config(),
        subagent=subagent,
    )
    schema = next(
        _function_payload(tool)
        for tool in runner._provider_tools()
        if _function_name(tool) == "agent"
    )
    parameters = schema["parameters"]
    assert isinstance(parameters, dict)
    # launch needs agent+task and message needs task_id+text: the dispatcher
    # enforces the pair, so the schema itself requires neither.
    assert parameters["required"] == []
    properties = parameters["properties"]
    assert isinstance(properties, dict)
    assert properties["action"]["enum"] == ["launch", "message"]
    assert properties["text"]["maxLength"] == SUBAGENT_STEER_MAX_CHARS == 2_000
    role_schema = properties["agent"]
    assert isinstance(role_schema, dict)
    assert role_schema["enum"] == list(SUBAGENT_ROLES)
    assert role_schema["enum"] == ["auditor", "developer", "general-purpose", "Explore"]
    thinking_schema = properties["thinking"]
    assert isinstance(thinking_schema, dict)
    assert thinking_schema["enum"] == list(SUBAGENT_THINKING_LEVELS)
    assert properties["inherit_context"]["type"] == "boolean"
    assert properties["max_turns"]["type"] == "integer"
    assert "maximum" not in properties["max_turns"]
    assert "maxLength" not in properties["task"]
    meta_runner = AgentSessionRunner(
        llm=ScriptedLLM([]),
        tools=ToolRegistry(),
        system_prompt="meta",
        config=_meta_config(),
        subagent=SubAgentEngine(
            llm=ScriptedLLM([]),
            tools=ToolRegistry([_NamedTool("read_file")]),
            mode="meta",
        ),
    )
    meta_schema = next(
        _function_payload(tool)
        for tool in meta_runner._provider_tools()
        if _function_name(tool) == "agent"
    )
    meta_parameters = meta_schema["parameters"]
    assert isinstance(meta_parameters, dict)
    meta_properties = meta_parameters["properties"]
    assert isinstance(meta_properties, dict)
    meta_role = meta_properties["agent"]
    assert isinstance(meta_role, dict)
    assert meta_role["enum"] == list(SUBAGENT_ROLES)


def test_session_config_has_no_required_subagent_role_gate() -> None:
    config = AgentSessionConfig(mode="fold")
    assert not hasattr(config, "required_subagent_roles")
    assert "required_subagent_roles" not in AgentSessionConfig.__dataclass_fields__


def test_role_tool_visibility_hides_writes_from_audits(tmp_path: Path) -> None:
    from autotrade.agent.runner import _FOLD_TOOLS

    assert "write_file" in _FOLD_TOOLS
    assert "edit_file" in _FOLD_TOOLS
    workspace = tmp_path / "agent"
    workspace.mkdir()
    (workspace / "output").mkdir()
    safe = SafeWorkspace(workspace)
    tools = build_fold_subagent_tools(
        SearchRoots(safe),
        safe,
        _UnusedRunner(),
        ModificationCheckTool(workspace / "output"),
        _NamedTool("smoke_backtest"),
    )
    engine = SubAgentEngine(llm=ScriptedLLM([]), tools=ToolRegistry(tools))
    impl = {
        _function_name(record)
        for record in engine._provider_tools(allowed_subagent_tools("fold", "developer"))
    }
    audit = {
        _function_name(record)
        for record in engine._provider_tools(allowed_subagent_tools("fold", "auditor"))
    }
    # The parent's Fold surface minus what it keeps by design (both formal
    # validation tools, finish, rollback, ask_user, agent): the unofficial
    # smoke run is a child's tool too, so it verifies its own implementation
    # on the real replay path instead of hand-rolling a shell smoke test.
    assert impl == {
        "delete_skill",
        "edit_file",
        "glob",
        "grep",
        "modification_check",
        "read_file",
        "shell",
        "smoke_backtest",
        "write_file",
        "write_skill",
    }
    assert impl == _FOLD_TOOLS - {
        "agent",
        "ask_user",
        "batch_validate",
        "daily_backtest",
        "finish_fold",
        # A child gathers evidence; the parent session is what concludes that a
        # mounted memory entry held up or did not, so the verdict is its call.
        "memory_feedback",
        # Same shape for defect reports: children report findings to the
        # parent, the parent files them with the operators.
        "report_issue",
        "step_rollback",
    }
    assert audit == {"glob", "grep", "read_file"}
    fold_general = {
        _function_name(record)
        for record in engine._provider_tools(
            allowed_subagent_tools("fold", "general-purpose")
        )
    }
    assert {"write_file", "edit_file"} <= fold_general
    assert fold_general == impl
    meta_engine = SubAgentEngine(
        llm=ScriptedLLM([]),
        tools=ToolRegistry(build_meta_subagent_tools(SearchRoots(safe))),
        mode="meta",
    )
    meta_names = {
        _function_name(record)
        for record in meta_engine._provider_tools(
            allowed_subagent_tools("meta", "auditor")
        )
    }
    meta_general = {
        _function_name(record)
        for record in meta_engine._provider_tools(
            allowed_subagent_tools("meta", "general-purpose")
        )
    }
    fold_explore_role = {
        _function_name(record)
        for record in engine._provider_tools(allowed_subagent_tools("fold", "Explore"))
    }
    assert fold_explore_role == {"read_file", "grep", "glob"}
    assert "shell" not in fold_explore_role
    assert "write_file" not in fold_explore_role
    meta_developer = {
        _function_name(record)
        for record in meta_engine._provider_tools(
            allowed_subagent_tools("meta", "developer")
        )
    }
    meta_explore_role = {
        _function_name(record)
        for record in meta_engine._provider_tools(
            allowed_subagent_tools("meta", "Explore")
        )
    }
    assert meta_names == {"read_file", "grep", "glob"}
    assert meta_general == meta_names
    assert meta_developer == meta_names
    assert meta_explore_role == meta_names
    assert "write_file" not in meta_general
    assert "edit_file" not in meta_general


def test_subagent_calls_still_track_attempts_and_roles() -> None:
    finish = _FinishStub("finish_fold")
    subagent = SubAgentEngine(
        llm=ScriptedLLM(
            [
                ProviderResponse(content="general summary"),
                ProviderResponse(content="data summary"),
                ProviderResponse(content="strategy summary"),
            ]
        ),
        tools=ToolRegistry([DeclaredReadOnlyShell()]),
        config=SubAgentConfig(max_concurrent=3),
    )
    events: list[tuple[str, dict[str, object]]] = []
    runner = AgentSessionRunner(
        llm=ScriptedLLM(
            [
                ProviderResponse(
                    tool_calls=(
                        ToolCall(
                            "g1",
                            "agent",
                            {"agent": "general-purpose", "task": "cross-cut"},
                        ),
                        ToolCall(
                            "e1",
                            "agent",
                            {"agent": "auditor", "task": "check data"},
                        ),
                        ToolCall(
                            "e2",
                            "agent",
                            {"agent": "developer", "task": "check strategy"},
                        ),
                    )
                ),
                ProviderResponse(tool_calls=(ToolCall("f1", "finish_fold", {}),)),
            ]
        ),
        tools=ToolRegistry([finish]),
        system_prompt="fold",
        config=_fold_config(),
        subagent=subagent,
        event_sink=lambda event, payload: events.append((event, payload)),
    )

    result = runner.run("delegate when useful")

    assert result.status == "finished"
    assert finish.invoked == 1
    assert runner._subagent_attempts == 3
    assert runner._subagent_roles == {
        "general-purpose",
        "auditor",
        "developer",
    }
    attempt_events = [payload for event, payload in events if event == "subagent_attempt"]
    # The three launches ran concurrently, so completion order is not fixed.
    assert sorted(payload["role"] for payload in attempt_events) == [
        "auditor",
        "developer",
        "general-purpose",
    ]
    assert all("task" not in payload for payload in attempt_events)
    ended = next(payload for event, payload in events if event == "session_end")
    assert ended["subagent_attempts"] == 3
    assert ended["subagent_roles"] == [
        "auditor",
        "developer",
        "general-purpose",
    ]
    assert "task" not in ended
    # In-flight children collected at finish still bill the session.
    assert ended["token_usage"]["subagent"]["llm_calls"] == 3
    assert result.usage["subagent"]["llm_calls"] == 3


def test_single_subagent_role_can_finish() -> None:
    finish = _FinishStub("finish_fold")
    subagent = SubAgentEngine(
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
                            "agent",
                            {"agent": "auditor", "task": "check data"},
                        ),
                    )
                ),
                ProviderResponse(tool_calls=(ToolCall("f1", "finish_fold", {}),)),
            ]
        ),
        tools=ToolRegistry([finish]),
        system_prompt="fold",
        config=_fold_config(),
        subagent=subagent,
    )
    assert runner.run("one delegated review is enough").status == "finished"
    assert finish.invoked == 1
    assert runner._subagent_roles == {"auditor"}


def test_general_prompts_explain_mode_and_role() -> None:
    fold = subagent_system_prompt("fold", "general-purpose")
    meta = subagent_system_prompt("meta", "general-purpose")
    assert "一级 `general-purpose`" in fold
    assert "修改共享策略、模型或 skills" in fold
    assert "有界的跨域实现任务" in fold
    # Writers share one live tree with the parent and sibling children: no
    # private copy, no merge-back, so writes stay inside the task's paths.
    for clause in (
        "共用的同一棵实时目录树",
        "只在 task 给定的路径下创建、修改与删除",
        "删除目录要在汇报里写明删了什么",
    ):
        assert clause in fold
        assert clause not in subagent_system_prompt("fold", "auditor")
    assert "`general-purpose`" in meta
    assert "只读" in meta
    assert "不能写策略、models、skills 或 PRIOR" in meta


def test_normalize_subagent_thinking_accepts_aliases() -> None:
    assert DEFAULT_SUBAGENT_THINKING == "xhigh"
    assert normalize_subagent_thinking(None) == "xhigh"
    assert normalize_subagent_thinking("inherit") == "xhigh"
    assert normalize_subagent_thinking("minimal") == "low"
    assert normalize_subagent_thinking("xhigh") == "xhigh"
    assert normalize_subagent_thinking("high") == "xhigh"
    assert normalize_subagent_thinking("max") == "xhigh"
    assert SUBAGENT_THINKING_LEVELS == ("off", "low", "medium", "xhigh")
    with pytest.raises(ValueError, match="agent.thinking"):
        normalize_subagent_thinking("turbo")


def test_subagent_defaults_are_xhigh_thinking_and_four_concurrent() -> None:
    assert DEFAULT_SUBAGENT_MAX_CONCURRENT == 4
    assert SubAgentConfig().max_concurrent == 4
    result = SubAgentEngine(
        llm=ScriptedLLM([ProviderResponse(content="ok")]),
        tools=ToolRegistry([DeclaredReadOnlyShell()]),
    ).run("summarize", role="auditor")
    assert result["thinking"] == "xhigh"
    assert "默认同时运行 4 个，超出排队" in AGENT_TOOL_DESCRIPTION
    assert "subagent_completed" in FOLD_WORKFLOW_SECTION
    assert "不要用工具轮询" in FOLD_WORKFLOW_SECTION
    assert "Sleep" not in FOLD_WORKFLOW_SECTION


def test_subagent_inherit_context_prepends_parent_digest() -> None:
    llm = ScriptedLLM([ProviderResponse(content="used parent excerpt")])
    result = SubAgentEngine(
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


def test_subagent_arguments_are_validated_by_the_registry() -> None:
    subagent = SubAgentEngine(
        llm=ScriptedLLM([ProviderResponse(content="ok")]),
        tools=ToolRegistry([DeclaredReadOnlyShell()]),
    )
    runner = AgentSessionRunner(
        llm=ScriptedLLM([]),
        tools=ToolRegistry(),
        system_prompt="fold",
        config=_fold_config(),
        subagent=subagent,
    )
    # The runner registered subagent like any other tool.
    assert runner.tools.spec("agent") is not None
    assert "agent" in {
        _function_name(tool) for tool in runner._provider_tools()
    }
    bad_thinking = runner.tools.invoke(
        "agent", {"agent": "auditor", "task": "x", "thinking": "turbo"}
    )
    assert bad_thinking.ok is False
    assert "thinking" in bad_thinking.error
    assert bad_thinking.value["error_type"] == "schema_error"
    bad_role = runner.tools.invoke("agent", {"agent": "reader", "task": "x"})
    assert bad_role.ok is False and "agent must be one of" in bad_role.error
    unknown = runner.tools.invoke(
        "agent", {"agent": "auditor", "task": "x", "max_rounds": 3}
    )
    assert unknown.ok is False and "max_rounds" in unknown.error
    too_long = runner.tools.invoke(
        "agent",
        {
            "agent": "auditor",
            "task": "x",
            "description": "d" * (SUBAGENT_DESCRIPTION_MAX_CHARS + 1),
        },
    )
    assert too_long.ok is False and "description" in too_long.error
    assert runner._subagent_attempts == 0
    # An integral JSON number is accepted as max_turns and the launch starts.
    started = runner.tools.invoke(
        "agent", {"agent": "auditor", "task": "x", "max_turns": 2.0}
    )
    assert started.ok and started.value["status"] == "started"
    assert runner._wait_subagent_jobs()[-1]["ok"] is True


def test_legacy_thinking_values_launch_at_xhigh_through_the_registry() -> None:
    """The documented ``high``/``max`` aliases must survive the schema enum.

    The advertised enum is the four canonical levels, so the alias mapping has
    to run before validation; an unknown level still fails with the enum."""

    runner = AgentSessionRunner(
        llm=ScriptedLLM([]),
        tools=ToolRegistry(),
        system_prompt="fold",
        config=_fold_config(),
        subagent=SubAgentEngine(
            llm=ScriptedLLM(
                [ProviderResponse(content="ok"), ProviderResponse(content="ok")]
            ),
            tools=ToolRegistry([DeclaredReadOnlyShell()]),
        ),
    )
    for legacy in ("high", "max"):
        started = runner.tools.invoke(
            "agent", {"agent": "auditor", "task": "x", "thinking": legacy}
        )
        assert started.ok and started.value["status"] == "started"
        record = runner._wait_subagent_jobs()[-1]
        assert record["ok"] is True
        child = record["value"]
        assert isinstance(child, dict) and child["thinking"] == "xhigh"
    unknown = runner.tools.invoke(
        "agent", {"agent": "auditor", "task": "x", "thinking": "turbo"}
    )
    assert unknown.ok is False
    assert unknown.value["error_type"] == "schema_error"
    assert f"thinking must be one of {list(SUBAGENT_THINKING_LEVELS)}" in unknown.error
    # A rejected launch never reached the pool.
    assert runner._subagent_attempts == 2


def test_agent_action_resume_is_told_the_resume_parameter() -> None:
    """``action="resume"`` is a recurring parent mistake; the bare enum list
    does not say where resuming actually lives, so the rejection does."""

    runner = AgentSessionRunner(
        llm=ScriptedLLM([]),
        tools=ToolRegistry(),
        system_prompt="fold",
        config=_fold_config(),
        subagent=SubAgentEngine(
            llm=ScriptedLLM([]), tools=ToolRegistry([DeclaredReadOnlyShell()])
        ),
    )
    rejected = runner.tools.invoke(
        "agent",
        {"action": "resume", "agent": "auditor", "task": "x", "task_id": "agent_1"},
    )
    assert rejected.ok is False
    assert rejected.value["error_type"] == "schema_error"
    assert "resume is not an action" in rejected.error
    assert "resume=<task_id>" in rejected.error
    assert '"agent": "auditor"' in rejected.error
    assert runner._subagent_attempts == 0
    # Any other unknown action still gets the plain enum rejection.
    other = runner.tools.invoke("agent", {"action": "steer", "task_id": "agent_1"})
    assert other.ok is False and "action must be one of" in other.error


def test_subagent_thinking_only_reply_does_not_finish_the_child() -> None:
    llm = ScriptedLLM(
        [
            ProviderResponse(content="", reasoning_content="internal plan"),
            ProviderResponse(content="done after thinking"),
        ]
    )
    result = SubAgentEngine(
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
            raise TimeoutError("subagent gate")
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


def test_parent_session_continues_before_subagent_finishes() -> None:
    started = threading.Event()
    release = threading.Event()
    finish = _FinishStub("finish_fold")
    inner = ScriptedLLM(
        [
            ProviderResponse(
                tool_calls=(
                    ToolCall(
                        "e1",
                        "agent",
                        {"agent": "auditor", "task": "slow look"},
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
        subagent=SubAgentEngine(
            llm=_GateLLM(started, release),
            tools=ToolRegistry([DeclaredReadOnlyShell()]),
        ),
    )
    result = runner.run("go")
    assert result.status == "finished"
    assert finish.invoked == 1
    assert parent_calls["n"] == 2


def test_parent_text_only_waits_for_pending_subagent_then_resumes() -> None:
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
                        "agent",
                        {"agent": "auditor", "task": "slow look"},
                    ),
                )
            ),
            ProviderResponse(content="waiting for subagent"),
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
                assert '"observation": "subagent_completed"' in blob
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
        subagent=SubAgentEngine(
            llm=_GateLLM(started, release),
            tools=ToolRegistry([DeclaredReadOnlyShell()]),
        ),
    )
    result = runner.run("go")
    assert result.status == "finished"
    assert finish.invoked == 1
    assert parent_calls["n"] == 3
    assert len(inner.calls) == 3


def test_parent_text_only_without_pending_subagent_still_nudges() -> None:
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
        subagent=SubAgentEngine(
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


def test_parent_text_only_wakes_on_first_completed_subagent() -> None:
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
                        "agent",
                        {"agent": "auditor", "task": "fast-task"},
                    ),
                    ToolCall(
                        "e2",
                        "agent",
                        {"agent": "developer", "task": "slow-task"},
                    ),
                )
            ),
            ProviderResponse(content="waiting for first subagent"),
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
                assert blob.count('"observation": "subagent_completed"') == 1
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
        subagent=SubAgentEngine(
            llm=_TaskGatedLLM(
                {
                    "fast-task": (fast_started, fast_release, "fast summary"),
                    "slow-task": (slow_started, slow_release, "slow summary"),
                }
            ),
            tools=ToolRegistry([DeclaredReadOnlyShell()]),
            config=SubAgentConfig(max_concurrent=2),
        ),
    )
    result = runner.run("go")
    assert result.status == "finished"
    assert finish.invoked == 1
    assert parent_calls["n"] == 3


def test_parent_text_only_pending_subagent_deadline_does_not_deadlock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from autotrade.agent import runner as runner_module

    monkeypatch.setattr(runner_module, "SUBAGENT_TEARDOWN_WAIT_SECONDS", 0.2)
    started = threading.Event()
    release = threading.Event()
    finish = _FinishStub("finish_fold")
    inner = ScriptedLLM(
        [
            ProviderResponse(
                tool_calls=(
                    ToolCall(
                        "e1",
                        "agent",
                        {"agent": "auditor", "task": "hang"},
                    ),
                )
            ),
            ProviderResponse(content="waiting for subagent"),
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
                subagent=SubAgentEngine(
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


def test_wait_first_pending_subagent_returns_on_cancel() -> None:
    started = threading.Event()
    release = threading.Event()
    runner = AgentSessionRunner(
        llm=ScriptedLLM([]),
        tools=ToolRegistry(),
        system_prompt="fold",
        config=_fold_config(),
        subagent=SubAgentEngine(
            llm=_GateLLM(started, release),
            tools=ToolRegistry([DeclaredReadOnlyShell()]),
        ),
    )
    dispatched = runner.tools.invoke(
        "agent", {"agent": "auditor", "task": "hang"}
    )
    assert dispatched.ok and dispatched.value["status"] == "started"
    assert started.wait(3)
    threading.Timer(0.1, runner._cancelled.set).start()
    t0 = time.monotonic()
    try:
        runner._wait_first_pending_subagent(InferenceTimeBudget(duration_seconds=20))
        assert time.monotonic() - t0 < 2.0
    finally:
        release.set()


def test_subagent_stops_retry_on_call_budget_and_interrupt() -> None:
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
    result = SubAgentEngine(
        llm=exhausted,
        tools=ToolRegistry([DeclaredReadOnlyShell()]),
        config=SubAgentConfig(max_rounds=5),
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
    result = SubAgentEngine(
        llm=interrupted,
        tools=ToolRegistry([DeclaredReadOnlyShell()]),
        config=SubAgentConfig(max_rounds=5),
    ).run("look", role="auditor")
    assert result["status"] == "error"
    assert interrupted.calls == 1


def test_subagent_skips_tools_when_cancelled_after_llm() -> None:
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

    engine = SubAgentEngine(
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


def test_runner_close_cancels_subagent_without_infinite_wait(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from autotrade.agent import runner as runner_module

    monkeypatch.setattr(runner_module, "SUBAGENT_TEARDOWN_WAIT_SECONDS", 0.2)
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
                        "agent",
                        {"agent": "developer", "task": "slow"},
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
        subagent=SubAgentEngine(
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


class _OverlapProbe:
    """Shared record of how many tool calls overlap in time, and their order."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.active = 0
        self.max_active = 0
        self.order: list[str] = []
        self.max_active_by_turn: list[int] = []

    def new_turn(self) -> None:
        with self.lock:
            self.max_active_by_turn.append(0)

    def enter(self, name: str) -> None:
        with self.lock:
            self.active += 1
            self.max_active_by_turn[-1] = max(self.max_active_by_turn[-1], self.active)
            self.order.append(name)

    def leave(self) -> None:
        with self.lock:
            self.active -= 1


class _ProbeTool:
    def __init__(
        self, name: str, probe: _OverlapProbe, *, mutating: bool = False, hold: float = 0.15
    ) -> None:
        self.spec = ToolSpec(
            name,
            "overlap probe",
            {"type": "object", "properties": {}, "required": []},
            mutating=mutating,
        )
        self.probe = probe
        self.hold = hold

    def invoke(self, arguments: Mapping[str, object]) -> ToolResult:
        del arguments
        self.probe.enter(self.spec.name)
        time.sleep(self.hold)
        self.probe.leave()
        return ToolResult(True, value={"tool": self.spec.name})


def test_parallel_safe_batch_runs_concurrently_and_mutating_batch_in_order() -> None:
    probe = _OverlapProbe()
    read = _ProbeTool("read_file", probe)
    grep = _ProbeTool("grep", probe)
    write = _ProbeTool("write_file", probe, mutating=True)
    finish = _FinishStub("finish_fold")
    llm = ScriptedLLM(
        [
            ProviderResponse(
                tool_calls=(
                    ToolCall("r1", "read_file", {}),
                    ToolCall("g1", "grep", {}),
                    ToolCall("r2", "read_file", {}),
                )
            ),
            ProviderResponse(
                tool_calls=(
                    ToolCall("g2", "grep", {}),
                    ToolCall("w1", "write_file", {}),
                    ToolCall("r3", "read_file", {}),
                )
            ),
            ProviderResponse(tool_calls=(ToolCall("f1", "finish_fold", {}),)),
        ]
    )

    class _TurnLLM:
        model = "parent"
        provider = "test"
        context_window_tokens = 128000

        def complete(self, messages, **kwargs):
            probe.new_turn()
            return llm.complete(messages, **kwargs)

    runner = AgentSessionRunner(
        llm=_TurnLLM(),
        tools=ToolRegistry([read, grep, write, finish]),
        system_prompt="fold",
        config=_fold_config(),
    )
    assert runner.run("go").status == "finished"
    # Turn 1: three parallel-safe calls overlapped.
    assert probe.max_active_by_turn[0] >= 2
    # Turn 2: one mutating call demoted the whole batch to in-order execution.
    assert probe.max_active_by_turn[1] == 1
    assert probe.order[3:] == ["grep", "write_file", "read_file"]
    # Tool results are paired to their calls in call order regardless of finish order.
    second_turn = llm.calls[1]["messages"]
    tool_ids = [message.tool_call_id for message in second_turn if message.role == "tool"]
    assert tool_ids == ["r1", "g1", "r2"]


def test_subagent_launches_beyond_the_cap_queue_instead_of_failing() -> None:
    gates = {
        f"task-{index}": (threading.Event(), threading.Event(), f"summary-{index}")
        for index in range(3)
    }
    finish = _FinishStub("finish_fold")
    inner = ScriptedLLM(
        [
            ProviderResponse(
                tool_calls=tuple(
                    ToolCall(f"e{index}", "agent", {"agent": "auditor", "task": needle})
                    for index, needle in enumerate(gates)
                )
            ),
            ProviderResponse(content="waiting"),
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
                # Only one child may run at a time; every launch was accepted.
                tool_records = [
                    message.content
                    for message in messages
                    if message.role == "tool"
                ]
                assert len(tool_records) == 3
                assert all('"status": "started"' in text for text in tool_records)
                assert sum('"queued": true' in text for text in tool_records) == 2
                running = [needle for needle, (s, _r, _x) in gates.items() if s.is_set()]
                assert len(running) == 1
                for _needle, (_s, release, _x) in gates.items():
                    release.set()
            elif n == 3:
                blob = "\n".join(str(message.content or "") for message in messages)
                assert blob.count('"observation": "subagent_completed"') >= 1
            return inner.complete(messages, **kwargs)

    runner = AgentSessionRunner(
        llm=_ParentLLM(),
        tools=ToolRegistry([finish]),
        system_prompt="fold",
        config=_fold_config(),
        subagent=SubAgentEngine(
            llm=_TaskGatedLLM(gates),
            tools=ToolRegistry([DeclaredReadOnlyShell()]),
            config=SubAgentConfig(max_concurrent=1),
        ),
    )
    result = runner.run("go")
    assert result.status == "finished"
    assert runner._subagent_attempts == 3
    assert all(job.record is not None for job in runner._subagent_jobs)
    assert sorted(
        str(job.record["value"].get("summary")) for job in runner._subagent_jobs
    ) == ["summary-0", "summary-1", "summary-2"]


def test_backtest_gate_keeps_its_batch_in_order_regardless_of_spec() -> None:
    """A completed Validation may enter hard finalization; the remaining research
    calls of the same turn must then be refused, which needs an ordered batch."""
    backtest = _NamedTool("daily_backtest")
    read = _NamedTool("read_file")
    runner = AgentSessionRunner(
        llm=ScriptedLLM([]),
        tools=ToolRegistry([backtest, read, _FinishStub("finish_fold")]),
        system_prompt="fold",
        config=_fold_config(),
    )
    assert backtest.spec.mutating is False
    assert runner._is_parallel_batch(
        (ToolCall("b", "daily_backtest", {}), ToolCall("r", "read_file", {}))
    ) is False
    assert runner._is_parallel_batch(
        (ToolCall("r1", "read_file", {}), ToolCall("r2", "read_file", {}))
    ) is True


def test_unfinished_session_end_still_reports_token_usage() -> None:
    runner = AgentSessionRunner(
        llm=ScriptedLLM([ProviderResponse(content="thinking aloud")]),
        tools=ToolRegistry([_FinishStub("finish_fold")]),
        system_prompt="fold",
        config=_fold_config(max_llm_calls=1),
    )
    events: list[tuple[str, dict[str, object]]] = []
    runner.event_sink = lambda event, payload: events.append((event, payload))
    with pytest.raises(RuntimeError, match="call budget"):
        runner.run("go")
    ended = next(payload for event, payload in events if event == "session_end")
    assert ended["status"] == "call_budget_exhausted"
    assert ended["token_usage"]["llm_calls_with_usage"] == 1


def test_inherit_context_fork_drops_the_unanswered_tool_calls() -> None:
    child = ScriptedLLM([ProviderResponse(content="forked")])
    runner = AgentSessionRunner(
        llm=ScriptedLLM([]),
        tools=ToolRegistry(),
        system_prompt="fold",
        config=_fold_config(),
        subagent=SubAgentEngine(
            llm=child, tools=ToolRegistry([DeclaredReadOnlyShell()])
        ),
    )
    # The realistic snapshot: the batch holding this subagent call is running,
    # so the last assistant turn has no tool results yet.
    runner._live_messages = [
        ChatMessage("system", "secret contract"),
        ChatMessage("user", "look at daily schema"),
        ChatMessage("assistant", None, (ToolCall("r1", "read_file", {"path": "a"}),)),
        ChatMessage("tool", '{"ok": true}', tool_call_id="r1"),
        ChatMessage(
            "assistant",
            None,
            (ToolCall("e1", "agent", {"agent": "auditor", "task": "fork"}),),
        ),
    ]
    started = runner.tools.invoke(
        "agent", {"agent": "auditor", "task": "fork", "inherit_context": True}
    )
    assert started.ok
    assert runner._wait_subagent_jobs()[-1]["ok"] is True
    request = child.calls[0]["messages"]
    assert request[-1].role == "user" and request[-1].content == "fork"
    assert all("secret contract" not in (m.content or "") for m in request)
    answered = {m.tool_call_id for m in request if m.role == "tool"}
    for message in request:
        if message.role == "assistant":
            assert {call.id for call in message.tool_calls} <= answered
    assert any(m.tool_call_id == "r1" for m in request)


class _BoomTool:
    spec = ToolSpec("grep", "raises", {"type": "object", "properties": {}, "required": []})

    def invoke(self, arguments: Mapping[str, object]) -> ToolResult:
        del arguments
        raise RuntimeError("regex engine exploded")


def test_tool_exception_in_a_parallel_batch_keeps_sibling_results() -> None:
    finish = _FinishStub("finish_fold")
    llm = ScriptedLLM(
        [
            ProviderResponse(
                tool_calls=(
                    ToolCall("g", "grep", {}),
                    ToolCall("r", "read_file", {}),
                )
            ),
            ProviderResponse(tool_calls=(ToolCall("f1", "finish_fold", {}),)),
        ]
    )
    runner = AgentSessionRunner(
        llm=llm,
        tools=ToolRegistry([_BoomTool(), _NamedTool("read_file"), finish]),
        system_prompt="fold",
        config=_fold_config(),
    )
    assert runner.run("go").status == "finished"
    second = llm.calls[1]["messages"]
    results = {m.tool_call_id: json.loads(m.content) for m in second if m.role == "tool"}
    assert results["r"]["ok"] is True
    assert results["g"]["ok"] is False
    assert "RuntimeError" in results["g"]["error"]
    assert results["g"]["value"]["error_type"] == "tool_exception"


def test_daily_backtest_waits_for_running_subagent() -> None:
    class _SlowChild:
        model = "child"
        provider = "test"
        context_window_tokens = 128000

        def complete(self, messages, **kwargs):
            del messages, kwargs
            time.sleep(0.3)
            return ProviderResponse(content="child done")

    seen: dict[str, bool] = {}

    class _Backtest:
        spec = ToolSpec(
            "daily_backtest", "gate", {"type": "object", "properties": {}, "required": []}
        )

        def invoke(self, arguments: Mapping[str, object]) -> ToolResult:
            del arguments
            seen["children_done"] = all(
                job.future.done() for job in runner._subagent_jobs
            )
            return ToolResult(True, value={"status": "probe"})

    finish = _FinishStub("finish_fold")
    llm = ScriptedLLM(
        [
            ProviderResponse(
                tool_calls=(ToolCall("e1", "agent", {"agent": "developer", "task": "slow"}),)
            ),
            ProviderResponse(tool_calls=(ToolCall("b1", "daily_backtest", {}),)),
            ProviderResponse(tool_calls=(ToolCall("f1", "finish_fold", {}),)),
        ]
    )
    runner = AgentSessionRunner(
        llm=llm,
        tools=ToolRegistry([_Backtest(), finish]),
        system_prompt="fold",
        config=_fold_config(),
        subagent=SubAgentEngine(
            llm=_SlowChild(), tools=ToolRegistry([DeclaredReadOnlyShell()])
        ),
    )
    assert runner.run("go").status == "finished"
    assert seen["children_done"] is True
    # The child's result was delivered to the conversation after the barrier.
    third = llm.calls[2]["messages"]
    assert any('"observation": "subagent_completed"' in (m.content or "") for m in third)


def test_agent_tool_schema_through_the_registry() -> None:
    runner = AgentSessionRunner(
        llm=ScriptedLLM([]),
        tools=ToolRegistry(),
        system_prompt="fold",
        config=_fold_config(),
        subagent=SubAgentEngine(
            llm=ScriptedLLM([]), tools=ToolRegistry([DeclaredReadOnlyShell()])
        ),
    )
    spec = runner.tools.spec("agent")
    assert spec is not None and spec.name == "agent"
    schema = spec.input_schema
    assert schema["required"] == []
    assert set(schema["properties"]) == {
        "action", "agent", "task", "task_id", "text",
        "description", "max_turns", "thinking", "inherit_context", "resume",
    }
    assert schema["properties"]["agent"]["enum"] == list(SUBAGENT_ROLES)
    for phrase in ("subagent_completed", "resume", "不能嵌套", "不要轮询", "action=message"):
        assert phrase in spec.description
    # The old parameter name is a schema error, not a silent fallback.
    stale = runner.tools.invoke("agent", {"role": "auditor", "task": "x"})
    assert stale.ok is False and "role" in stale.error
    # Each action still fails fast on its own required pair.
    for arguments in ({"task": "x"}, {"agent": "auditor"}, {"action": "launch"}):
        bare = runner.tools.invoke("agent", arguments)
        assert bare.ok is False and bare.value["error_type"] == "schema_error"
        assert "launch requires agent and task" in bare.error
    for arguments in ({"action": "message", "text": "x"}, {"action": "message", "task_id": "agent_1"}):
        bare = runner.tools.invoke("agent", arguments)
        assert bare.ok is False and bare.value["error_type"] == "schema_error"
        assert "message requires task_id and text" in bare.error
    assert runner.tools.spec("explore") is None


def test_resume_continues_a_finished_child_transcript() -> None:
    child = ScriptedLLM(
        [ProviderResponse(content="first summary"), ProviderResponse(content="second summary")]
    )
    events: list[tuple[str, dict[str, object]]] = []
    runner = AgentSessionRunner(
        llm=ScriptedLLM([]),
        tools=ToolRegistry(),
        system_prompt="fold",
        config=_fold_config(),
        subagent=SubAgentEngine(llm=child, tools=ToolRegistry([DeclaredReadOnlyShell()])),
        event_sink=lambda event, payload: events.append((event, payload)),
    )
    first = runner.tools.invoke("agent", {"agent": "auditor", "task": "look at daily"})
    assert first.ok
    first_id = str(first.value["task_id"])
    assert first_id.startswith("agent_")
    assert runner._wait_subagent_jobs()[-1]["ok"] is True
    follow = runner.tools.invoke(
        "agent", {"agent": "auditor", "task": "now check minutes", "resume": first_id}
    )
    assert follow.ok and follow.value["resumed_from"] == first_id
    assert follow.value["task_id"] != first_id
    assert runner._wait_subagent_jobs()[-1]["ok"] is True
    request = child.calls[1]["messages"]
    texts = [str(message.content or "") for message in request]
    assert request[0].role == "system"
    assert "look at daily" in texts and "first summary" in texts
    assert request[-1].role == "user" and request[-1].content == "now check minutes"
    started = [payload for event, payload in events if event == "subagent_task"]
    assert started[1]["resumed_from"] == first_id
    finished = [payload for event, payload in events if event == "subagent"]
    assert finished[1]["resumed_from"] == first_id
    assert finished[1]["summary"] == "second summary"
    assert runner._subagent_attempts == 2


def test_resume_refuses_unknown_running_or_mismatched_children() -> None:
    started = threading.Event()
    release = threading.Event()
    runner = AgentSessionRunner(
        llm=ScriptedLLM([]),
        tools=ToolRegistry(),
        system_prompt="fold",
        config=_fold_config(),
        subagent=SubAgentEngine(
            llm=_GateLLM(started, release), tools=ToolRegistry([DeclaredReadOnlyShell()])
        ),
    )
    unknown = runner.tools.invoke(
        "agent", {"agent": "auditor", "task": "x", "resume": "agent_nope"}
    )
    assert unknown.ok is False and "unknown" in unknown.error
    assert unknown.value["error_type"] == "unknown_subagent"
    launched = runner.tools.invoke("agent", {"agent": "auditor", "task": "hang"})
    assert launched.ok
    task_id = str(launched.value["task_id"])
    assert started.wait(3)
    try:
        running = runner.tools.invoke(
            "agent", {"agent": "auditor", "task": "more", "resume": task_id}
        )
        assert running.ok is False and "still running" in running.error
        assert running.value["error_type"] == "subagent_running"
    finally:
        release.set()
    assert runner._wait_subagent_jobs()[-1]["ok"] is True
    mismatch = runner.tools.invoke(
        "agent", {"agent": "developer", "task": "more", "resume": task_id}
    )
    assert mismatch.ok is False and "auditor" in mismatch.error
    assert mismatch.value["error_type"] == "subagent_role_mismatch"
    assert runner._subagent_attempts == 1


def test_delegation_reminder_fires_once_after_eight_own_calls() -> None:
    read = _NamedTool("read_file")
    finish = _FinishStub("finish_fold")
    llm = ScriptedLLM(
        [
            *(
                ProviderResponse(tool_calls=(ToolCall(f"r{index}", "read_file", {}),))
                for index in range(9)
            ),
            ProviderResponse(tool_calls=(ToolCall("f1", "finish_fold", {}),)),
        ]
    )
    events: list[tuple[str, dict[str, object]]] = []
    runner = AgentSessionRunner(
        llm=llm,
        tools=ToolRegistry([read, finish]),
        system_prompt="fold",
        config=_fold_config(),
        subagent=SubAgentEngine(
            llm=ScriptedLLM([]), tools=ToolRegistry([DeclaredReadOnlyShell()])
        ),
        event_sink=lambda event, payload: events.append((event, payload)),
    )
    assert runner.run("go").status == "finished"
    reminders = [payload for event, payload in events if event == "delegation_reminder"]
    assert len(reminders) == 1 and reminders[0]["own_work_calls"] == 8
    ninth = llm.calls[8]["messages"]
    assert sum('"observation": "delegation_reminder"' in (m.content or "") for m in ninth) == 1
    last = llm.calls[-1]["messages"]
    assert sum('"observation": "delegation_reminder"' in (m.content or "") for m in last) == 1


def test_delegation_reminder_rearms_per_streak_and_counts_writes() -> None:
    """The reminder is not a one-shot latch: every further streak of eight
    own-work calls (reads or writes) with no child running fires it again,
    whether or not a launch happened in between."""
    read = _NamedTool("read_file")
    write = _NamedTool("write_file")
    finish = _FinishStub("finish_fold")
    llm = ScriptedLLM(
        [
            ProviderResponse(
                tool_calls=(ToolCall("a1", "agent", {"agent": "auditor", "task": "look"}),)
            ),
            # Text only while the child runs: the parent yields until it ends.
            ProviderResponse(content="waiting"),
            # Streak one and two: the eighth and the sixteenth call fire.
            *(
                ProviderResponse(tool_calls=(ToolCall(f"r{index}", "read_file", {}),))
                for index in range(18)
            ),
            ProviderResponse(
                tool_calls=(ToolCall("a2", "agent", {"agent": "auditor", "task": "again"}),)
            ),
            ProviderResponse(content="waiting"),
            # Streak three: self-implementation counts as own work too.
            *(
                ProviderResponse(tool_calls=(ToolCall(f"w{index}", "write_file", {}),))
                for index in range(8)
            ),
            ProviderResponse(tool_calls=(ToolCall("f1", "finish_fold", {}),)),
        ]
    )
    events: list[tuple[str, dict[str, object]]] = []
    runner = AgentSessionRunner(
        llm=llm,
        tools=ToolRegistry([read, write, finish]),
        system_prompt="fold",
        config=_fold_config(),
        subagent=SubAgentEngine(
            llm=ScriptedLLM([ProviderResponse(content="seen")] * 2),
            tools=ToolRegistry([DeclaredReadOnlyShell()]),
        ),
        event_sink=lambda event, payload: events.append((event, payload)),
    )
    assert runner.run("go").status == "finished"
    reminders = [payload for event, payload in events if event == "delegation_reminder"]
    assert [payload["own_work_calls"] for payload in reminders] == [8, 8, 8]
    assert all(payload["running_children"] == [] for payload in reminders)
    delivered = sum(
        '"observation": "delegation_reminder"' in str(message.content or "")
        for message in llm.calls[-1]["messages"]
    )
    assert delivered == 3


def test_delegation_reminder_waits_for_a_running_child_to_finish() -> None:
    """Own work beside a running child is parallel work, not a reason to
    nag; the streak fires once the parent is alone again."""
    started, release = threading.Event(), threading.Event()

    class ReleasingRead(_NamedTool):
        def __init__(self) -> None:
            super().__init__("read_file")
            self.calls = 0

        def invoke(self, arguments: Mapping[str, object]) -> ToolResult:
            self.calls += 1
            if self.calls == 9:
                release.set()
            return super().invoke(arguments)

    read = ReleasingRead()
    finish = _FinishStub("finish_fold")
    llm = ScriptedLLM(
        [
            ProviderResponse(
                tool_calls=(ToolCall("a1", "agent", {"agent": "auditor", "task": "slow look"}),)
            ),
            *(
                ProviderResponse(tool_calls=(ToolCall(f"r{index}", "read_file", {}),))
                for index in range(9)
            ),
            ProviderResponse(content="waiting"),
            ProviderResponse(tool_calls=(ToolCall("r9", "read_file", {}),)),
            ProviderResponse(tool_calls=(ToolCall("f1", "finish_fold", {}),)),
        ]
    )
    events: list[tuple[str, dict[str, object]]] = []
    runner = AgentSessionRunner(
        llm=llm,
        tools=ToolRegistry([read, finish]),
        system_prompt="fold",
        config=_fold_config(),
        subagent=SubAgentEngine(
            llm=_GateLLM(started, release),
            tools=ToolRegistry([DeclaredReadOnlyShell()]),
        ),
        event_sink=lambda event, payload: events.append((event, payload)),
    )
    try:
        assert runner.run("go").status == "finished"
    finally:
        release.set()
    reminders = [payload for event, payload in events if event == "delegation_reminder"]
    assert [payload["own_work_calls"] for payload in reminders] == [10]
    assert reminders[0]["running_children"] == []


def test_agent_description_states_role_capabilities_and_thinking_tiers() -> None:
    from autotrade.agent.subagent import SUBAGENT_MAX_TRUNCATION_CONTINUATIONS
    from autotrade.environment.llm import AGENT_MAX_OUTPUT_TOKENS

    continuations = f"最多 {SUBAGENT_MAX_TRUNCATION_CONTINUATIONS} 次强制简洁续写"
    for phrase in (
        "shell",
        "smoke_backtest",
        "不能执行",
        "general-purpose 或 developer",
        "thinking 默认 xhigh",
        "机械工作",
        "low/medium",
        f"{AGENT_MAX_OUTPUT_TOKENS} token",
        continuations,
        "不要串成 resume 链",
        "action=message",
        "不为催促而发",
    ):
        assert phrase in AGENT_TOOL_DESCRIPTION
    thinking_field = AGENT_TOOL_SPEC.input_schema["properties"]["thinking"]["description"]
    assert "均为 xhigh" in thinking_field and continuations in thinking_field
    for prompt in (FOLD_WORKFLOW_SECTION, build_system_prompt(mode="meta", experiment_facts={})):
        assert "low/medium" in prompt and "action=message" in prompt
        assert "xhigh 只给纯文本" not in prompt
        assert "优先 `resume`" not in prompt
    agent_field = AGENT_TOOL_SPEC.input_schema["properties"]["agent"]
    assert "不能执行" in agent_field["description"] and "shell" in agent_field["description"]
    # The three call shapes are separate, labelled parts carrying their exact
    # argument names; resume is a launch parameter, never an action.
    for shape in (
        "1. launch（省略 action 或 action=launch）",
        "2. resume（不是 action",
        '"resume": <已完成子代理的 task_id>',
        "action=resume、只给 task_id",
        "3. message（action=message）",
    ):
        assert shape in AGENT_TOOL_DESCRIPTION
    assert "没有 resume 这个 action" in AGENT_TOOL_SPEC.input_schema["properties"]["action"]["description"]
    assert "不写 task_id" in AGENT_TOOL_SPEC.input_schema["properties"]["resume"]["description"]
    assert "high 与 max 是 xhigh 的别名" in thinking_field


def test_agent_result_echoes_running_and_queued_children_with_descriptions() -> None:
    """The launch result is the parent's live picture: which scopes are already
    running or waiting for a slot, by the parent's own description, so a
    duplicate fan-out is visible before it is issued again."""
    gates = {
        needle: (threading.Event(), threading.Event(), f"done-{needle}")
        for needle in ("scope-alpha", "scope-beta")
    }
    runner = AgentSessionRunner(
        llm=ScriptedLLM([]),
        tools=ToolRegistry(),
        system_prompt="fold",
        config=_fold_config(),
        subagent=SubAgentEngine(
            llm=_TaskGatedLLM(gates),
            tools=ToolRegistry([DeclaredReadOnlyShell()]),
            config=SubAgentConfig(max_concurrent=1),
        ),
    )
    try:
        first = runner.tools.invoke(
            "agent",
            {"agent": "auditor", "task": "scope-alpha", "description": "read data summary"},
        )
        assert first.ok
        assert first.value["running_children"] == [
            {
                "task_id": first.value["task_id"],
                "role": "auditor",
                "description": "read data summary",
            }
        ]
        assert first.value["queued_children"] == []
        assert "queued" not in first.value
        assert gates["scope-alpha"][0].wait(3)
        second = runner.tools.invoke(
            "agent",
            {"agent": "Explore", "task": "scope-beta", "description": "read parent strategy"},
        )
        assert second.ok and second.value["queued"] is True
        assert [child["description"] for child in second.value["running_children"]] == [
            "read data summary"
        ]
        assert second.value["queued_children"] == [
            {
                "task_id": second.value["task_id"],
                "role": "Explore",
                "description": "read parent strategy",
            }
        ]
    finally:
        for _started, release, _summary in gates.values():
            release.set()
    assert all(record["ok"] for record in runner._wait_subagent_jobs())
    assert runner._subagent_live_picture() == {
        "running_children": [],
        "queued_children": [],
    }


def test_delegation_reminder_carries_the_live_picture() -> None:
    read = _NamedTool("read_file")
    finish = _FinishStub("finish_fold")
    llm = ScriptedLLM(
        [
            *(
                ProviderResponse(tool_calls=(ToolCall(f"r{index}", "read_file", {}),))
                for index in range(9)
            ),
            ProviderResponse(tool_calls=(ToolCall("f1", "finish_fold", {}),)),
        ]
    )
    runner = AgentSessionRunner(
        llm=llm,
        tools=ToolRegistry([read, finish]),
        system_prompt="fold",
        config=_fold_config(),
        subagent=SubAgentEngine(
            llm=ScriptedLLM([]), tools=ToolRegistry([DeclaredReadOnlyShell()])
        ),
    )
    assert runner.run("go").status == "finished"
    reminder = next(
        json.loads(str(message.content))
        for message in llm.calls[8]["messages"]
        if message.role == "user"
        and '"observation": "delegation_reminder"' in str(message.content or "")
    )
    assert reminder["running_children"] == [] and reminder["queued_children"] == []


def test_parent_and_child_output_budgets_share_the_safety_ceiling() -> None:
    from autotrade.environment.llm import AGENT_MAX_OUTPUT_TOKENS

    assert AGENT_MAX_OUTPUT_TOKENS == 32_768
    assert AgentSessionConfig().max_response_tokens == AGENT_MAX_OUTPUT_TOKENS
    assert SubAgentConfig().max_tokens == AGENT_MAX_OUTPUT_TOKENS
    llm = _GateLLM(threading.Event(), threading.Event())  # 128k window
    messages = [ChatMessage("user", "x")]
    shared = SubAgentEngine(llm=llm, tools=ToolRegistry([DeclaredReadOnlyShell()]))
    assert shared._output_tokens(llm, messages, ()) == AGENT_MAX_OUTPUT_TOKENS
    capped = SubAgentEngine(
        llm=llm,
        tools=ToolRegistry([DeclaredReadOnlyShell()]),
        config=SubAgentConfig(max_tokens=500),
    )
    assert capped._output_tokens(llm, messages, ()) == 500


def test_child_output_truncation_is_marked_and_never_silent() -> None:
    from autotrade.agent.subagent import OUTPUT_TRUNCATED_MARKER

    cut = ScriptedLLM(
        [
            ProviderResponse(
                content="结论：因子 A 在 2019 年后",
                usage={"prompt_tokens": 10, "completion_tokens": 500, "total_tokens": 510},
            )
        ],
        context_window_tokens=128_000,
    )
    result = SubAgentEngine(
        llm=cut,
        tools=ToolRegistry([DeclaredReadOnlyShell()]),
        config=SubAgentConfig(max_tokens=500),
    ).run("audit", role="auditor")
    assert result["status"] == "completed" and result["truncated"] is True
    assert result["summary"].endswith(OUTPUT_TRUNCATED_MARKER.format(limit=500))
    assert result["summary"].startswith("结论：因子 A 在 2019 年后")

    # The whole budget went into thinking: the child gets the same forced
    # concise continuation as the parent and carries on with a tool call.
    from autotrade.agent.subagent import SUBAGENT_MAX_TRUNCATION_CONTINUATIONS

    cut_reply = ProviderResponse(
        content="",
        reasoning_content="thinking...",
        usage={"prompt_tokens": 10, "completion_tokens": 500, "total_tokens": 510},
    )
    empty = ScriptedLLM(
        [
            cut_reply,
            ProviderResponse(tool_calls=(ToolCall("s", "shell", {"argv": ["ls"]}),)),
            ProviderResponse(content="简洁结论"),
        ],
        context_window_tokens=128_000,
    )
    events: list[tuple[str, dict[str, object]]] = []
    result = SubAgentEngine(
        llm=empty,
        tools=ToolRegistry([DeclaredReadOnlyShell()]),
        config=SubAgentConfig(max_tokens=500),
        event_sink=lambda event, payload: events.append((event, payload)),
    ).run("audit", role="developer")
    assert result["status"] == "completed" and result["llm_calls"] == 3
    assert result["summary"] == "简洁结论" and result["tool_calls"] == 1
    assert result["truncated"] is True and result["truncated_rounds"] == 1
    continued = empty.calls[1]["messages"]
    assert continued[-1].role == "user"
    observation = json.loads(str(continued[-1].content))
    assert observation["observation"] == "output_truncated"
    assert observation["max_tokens"] == 500 and "被截断" in observation["message"]
    cut_events = [payload for event, payload in events if event == "subagent_output_truncated"]
    assert [(e["round"], e["continuation"], e["max_tokens"]) for e in cut_events] == [(1, 1, 500)]

    # Every continuation exhausted on reasoning too: the launch failed, and
    # it says so instead of paying for one more apology round.
    exhausted = ScriptedLLM(
        [cut_reply] * (SUBAGENT_MAX_TRUNCATION_CONTINUATIONS + 1)
        + [ProviderResponse(content="must not be requested")],
        context_window_tokens=128_000,
    )
    result = SubAgentEngine(
        llm=exhausted,
        tools=ToolRegistry([DeclaredReadOnlyShell()]),
        config=SubAgentConfig(max_tokens=500),
    ).run("audit", role="auditor")
    assert result["status"] == "error" and result["summary"] == ""
    assert result["llm_calls"] == SUBAGENT_MAX_TRUNCATION_CONTINUATIONS + 1
    assert result["truncated_rounds"] == SUBAGENT_MAX_TRUNCATION_CONTINUATIONS + 1
    assert "output budget exhausted" in result["error"] and result["tool_calls"] == 0


class _OverflowingLLM:
    """A gateway whose scripted items are responses or exceptions to raise."""

    model = "child"
    provider = "test"
    context_window_tokens = 128_000

    def __init__(self, items: list[object]) -> None:
        self.items = list(items)
        self.calls: list[dict[str, object]] = []

    def complete(self, messages, *, tools=(), tool_choice="auto", max_tokens=None):
        self.calls.append({"messages": tuple(messages), "max_tokens": max_tokens})
        item = self.items.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class _LongResultTool:
    def __init__(self) -> None:
        self.spec = ToolSpec(
            "read_file", "long", {"type": "object", "properties": {}, "required": []}
        )

    def invoke(self, arguments: Mapping[str, object]) -> ToolResult:
        del arguments
        return ToolResult(True, value={"rows": 9, "text": "x" * 4_000})


def test_child_gets_the_parents_single_post_provider_overflow_recovery() -> None:
    """The provider is authoritative: when it refuses a request the local
    estimate called fitting, the child recovers once exactly like the parent
    (forced compaction, then a guaranteed tool-result edit) and retries; a
    second refusal ends the child with an explicit error."""
    from autotrade.agent.subagent import SUBAGENT_MAX_TRUNCATION_CONTINUATIONS
    from autotrade.environment.llm.proxy import LLMProxyError

    assert SUBAGENT_MAX_TRUNCATION_CONTINUATIONS == 1
    overflow = LLMProxyError(
        "HTTP 400: This model's maximum context length is 128000 tokens",
        retryable=False,
    )
    llm = _OverflowingLLM(
        [
            ProviderResponse(tool_calls=(ToolCall("r1", "read_file", {}),)),
            overflow,
            ProviderResponse(content="recovered report"),
        ]
    )
    events: list[tuple[str, dict[str, object]]] = []
    result = SubAgentEngine(
        llm=llm,  # type: ignore[arg-type]
        tools=ToolRegistry([_LongResultTool()]),
        event_sink=lambda event, payload: events.append((event, payload)),
    ).run("dig", role="auditor")
    assert result["status"] == "completed" and result["summary"] == "recovered report"
    assert result["llm_errors"] == 1 and result["llm_calls"] == 3
    edits = [payload for event, payload in events if event == "subagent_context_edit"]
    assert len(edits) == 1
    assert edits[0]["context_edit"]["reason"] == "provider_context_overflow_recovery"
    assert edits[0]["context_edit"]["summarized_tool_results"] == 1
    # The retried request carries the summarized tool result, and no
    # ``llm_error`` observation was spent on the recovery.
    retried = llm.calls[2]["messages"]
    tool_messages = [m for m in retried if m.role == "tool"]
    assert len(tool_messages) == 1
    assert '"context_tool_result_summary"' in str(tool_messages[0].content)
    assert not any('"llm_error"' in str(m.content) for m in retried if m.role == "user")

    twice = _OverflowingLLM(
        [
            ProviderResponse(tool_calls=(ToolCall("r1", "read_file", {}),)),
            overflow,
            overflow,
            ProviderResponse(content="must not be requested"),
        ]
    )
    result = SubAgentEngine(
        llm=twice,  # type: ignore[arg-type]
        tools=ToolRegistry([_LongResultTool()]),
    ).run("dig", role="auditor")
    assert result["status"] == "error" and result["summary"] == ""
    assert result["llm_errors"] == 2 and len(twice.items) == 1
    assert "maximum context length" in result["error"]


def test_child_turns_default_to_48_with_grace_wrap_up() -> None:
    from autotrade.agent.subagent import DEFAULT_SUBAGENT_MAX_ROUNDS, SUBAGENT_GRACE_ROUNDS

    assert DEFAULT_SUBAGENT_MAX_ROUNDS == 48 and SUBAGENT_GRACE_ROUNDS == 2
    assert SubAgentConfig().max_rounds == 48
    assert "48 轮" in AGENT_TOOL_DESCRIPTION and "24 轮" not in AGENT_TOOL_DESCRIPTION
    max_turns_field = AGENT_TOOL_SPEC.input_schema["properties"]["max_turns"]["description"]
    assert "48 轮" in max_turns_field and "自动压缩" in max_turns_field
    # Parents learn that a child has their context window, and that several
    # bounded parallel children still beat one long serial child.
    assert "相同的上下文窗口" in AGENT_TOOL_DESCRIPTION
    assert "并行的有界子代理仍好过一个很长的串行子代理" in AGENT_TOOL_DESCRIPTION
    assert "并行的有界子代理仍好过一个很长的串行子代理" in FOLD_WORKFLOW_SECTION
    assert "并行的有界子代理仍好过一个很长的串行子代理" in build_system_prompt(
        mode="meta", experiment_facts={}
    )

    busy = ScriptedLLM(
        [
            *(
                ProviderResponse(tool_calls=(ToolCall(f"s{index}", "shell", {"argv": ["ls"]}),))
                for index in range(4)
            ),
            ProviderResponse(content="final"),
        ],
        context_window_tokens=128_000,
    )
    events: list[tuple[str, dict[str, object]]] = []
    engine = SubAgentEngine(
        llm=busy,
        tools=ToolRegistry([DeclaredReadOnlyShell()]),
        config=SubAgentConfig(max_rounds=4),
        event_sink=lambda event, payload: events.append((event, payload)),
    )
    result = engine.run("dig", role="auditor")
    assert result["rounds"] == 4 and result["summary"] == "final"
    # Round 2 of 4 opens with the wrap-up notice; rounds 1 and 3 do not.
    second = busy.calls[1]["messages"]
    assert second[-1].role == "user" and "还剩 3 轮模型调用（上限 4）" in str(second[-1].content)
    assert "还剩" not in str(busy.calls[0]["messages"][-1].content)
    assert "还剩" not in str(busy.calls[2]["messages"][-1].content)
    wrap = [payload for event, payload in events if event == "subagent_wrap_up"]
    assert wrap == [
        {
            "task_id": result["task_id"],
            "role": "auditor",
            "round": 2,
            "rounds_limit": 4,
            "parent_call_id": None,
        }
    ]
    assert all(
        payload["role"] == "auditor"
        for event, payload in events
        if event in {"subagent_llm", "subagent_tool"}
    )

    # An explicit small budget is honoured as given; no grace below three turns.
    short = ScriptedLLM(
        [
            *(
                ProviderResponse(tool_calls=(ToolCall(f"s{index}", "shell", {"argv": ["ls"]}),))
                for index in range(2)
            ),
            ProviderResponse(content="done"),
        ],
        context_window_tokens=128_000,
    )
    result = SubAgentEngine(
        llm=short, tools=ToolRegistry([DeclaredReadOnlyShell()])
    ).run("dig", role="auditor", max_rounds=2)
    assert result["rounds"] == 2 and result["summary"] == "done"
    assert not any("还剩" in str(m.content) for call in short.calls for m in call["messages"])


def _tool_round(index: int) -> ProviderResponse:
    return ProviderResponse(tool_calls=(ToolCall(f"s{index}", "shell", {"argv": ["ls"]}),))


def test_child_compacts_at_the_shared_threshold_with_fresh_counters_per_launch() -> None:
    """A child has the parent's context window: at the parent's threshold it
    compacts with the parent's gateway and config instead of failing when the
    window fills, every launch gets its own counters, and the record reaches
    the trace nested so its status never reads as the child's outcome."""
    from autotrade.agent.compact import (
        ContextCompactionConfig,
        ContextCompactor,
        is_compaction_message,
    )
    from autotrade.pipelines.local_backend import _safe_meta_trace_payload

    # Production shape: every model role shares one session budget, and the
    # engine refuses a compactor bound to another budget.
    shared = SessionCallBudget(
        max_calls=40, time_budget=InferenceTimeBudget(duration_seconds=600)
    )
    summary = "## 目标\ncontinue\n\n## 下一步\n- finish"
    compact_llm = ScriptedLLM([ProviderResponse(content=summary)] * 2)
    parent_compactor = ContextCompactor(
        SessionBudgetLLM(compact_llm, budget=shared, role="compact"),
        ContextCompactionConfig(
            token_threshold=1, min_messages=4, keep_recent_messages=2, max_calls=1
        ),
    )
    child_llm = ScriptedLLM(
        [*(_tool_round(index) for index in range(3)), ProviderResponse(content="first")]
        + [*(_tool_round(index) for index in range(3)), ProviderResponse(content="second")],
        context_window_tokens=128_000,
    )
    events: list[tuple[str, dict[str, object]]] = []
    with pytest.raises(ValueError, match="subagent_compactor is unbound"):
        SubAgentEngine(
            llm=child_llm,
            tools=ToolRegistry([DeclaredReadOnlyShell()]),
            compactor=ContextCompactor(compact_llm),
            time_budget=shared.time_budget,
        )
    engine = SubAgentEngine(
        llm=SessionBudgetLLM(child_llm, budget=shared, role="subagent"),
        tools=ToolRegistry([DeclaredReadOnlyShell()]),
        compactor=parent_compactor,
        event_sink=lambda event, payload: events.append((event, payload)),
    )
    assert engine.time_budget is shared.time_budget
    first = engine.run("dig", role="auditor")
    second = engine.run("dig again", role="auditor")
    assert first["summary"] == "first" and second["summary"] == "second"
    # Round 2 crossed the threshold: the request the model saw carries the
    # compaction summary in place of the older turns.
    round_two = child_llm.calls[1]["messages"]
    assert is_compaction_message(round_two[1]) and len(round_two) == 4
    assert not any(is_compaction_message(m) for m in child_llm.calls[0]["messages"])
    compactions = [payload for event, payload in events if event == "subagent_context_compaction"]
    assert [payload["task_id"] for payload in compactions] == [first["task_id"], second["task_id"]]
    record = compactions[0]
    assert record["role"] == "auditor" and record["round"] == 2
    assert record["compaction"]["status"] == "ok" and record["compaction"]["messages_before"] == 4
    assert "status" not in record and "summary" not in record
    # ``max_calls=1`` per conversation: the second child compacted too, so
    # each launch ran a fresh compactor while the parent's stayed untouched.
    assert len(compact_llm.calls) == 2 and parent_compactor.compaction_count == 0
    # A Meta trace keeps the shape of the compaction, never the summary text.
    meta = _safe_meta_trace_payload("subagent_context_compaction", record)
    assert meta["compaction"]["status"] == "ok" and meta["compaction"]["messages_before"] == 4
    assert "summary" not in meta["compaction"] and meta["task_id"] == first["task_id"]


def test_runner_hands_its_compactor_to_children_and_honours_max_turns() -> None:
    from autotrade.agent.compact import ContextCompactionConfig, ContextCompactor

    shared = SessionCallBudget(
        max_calls=40, time_budget=InferenceTimeBudget(duration_seconds=600)
    )
    parent_compactor = ContextCompactor(
        SessionBudgetLLM(ScriptedLLM([]), budget=shared, role="compact"),
        ContextCompactionConfig(),
    )
    engine = SubAgentEngine(
        llm=SessionBudgetLLM(
            ScriptedLLM(
                [_tool_round(0), _tool_round(1), ProviderResponse(content="done")],
                context_window_tokens=128_000,
            ),
            budget=shared,
            role="subagent",
        ),
        tools=ToolRegistry([DeclaredReadOnlyShell()]),
    )
    events: list[tuple[str, dict[str, object]]] = []
    runner = AgentSessionRunner(
        llm=SessionBudgetLLM(ScriptedLLM([]), budget=shared, role="main"),
        tools=ToolRegistry(),
        system_prompt="fold",
        config=_fold_config(),
        compactor=parent_compactor,
        subagent=engine,
        event_sink=lambda event, payload: events.append((event, payload)),
    )
    assert engine.compactor is parent_compactor
    started = runner.tools.invoke("agent", {"agent": "auditor", "task": "dig", "max_turns": 3})
    assert started.ok is True
    assert runner._wait_subagent_jobs()[-1]["ok"] is True
    wrap = [payload for event, payload in events if event == "subagent_wrap_up"]
    assert [(payload["round"], payload["rounds_limit"]) for payload in wrap] == [(1, 3)]
    ended = [payload for event, payload in events if event == "subagent"]
    assert ended[-1]["rounds"] == 3 and ended[-1]["summary"] == "done"


def test_child_ends_on_context_overflow_instead_of_retrying() -> None:
    from autotrade.environment.llm import context_overflow_error

    class Overflowing:
        model = "child"
        provider = "test"
        context_window_tokens = 128_000

        def __init__(self) -> None:
            self.calls = 0

        def complete(self, messages, **kwargs):
            del messages, kwargs
            self.calls += 1
            raise context_overflow_error(
                estimated_prompt_tokens=130_000, max_tokens=12_000, context_window=128_000
            )

    llm = Overflowing()
    result = SubAgentEngine(
        llm=llm,
        tools=ToolRegistry([DeclaredReadOnlyShell()]),
        config=SubAgentConfig(max_rounds=5),
    ).run("look", role="auditor")
    assert result["status"] == "error" and llm.calls == 1
    assert "context window" in result["error"] and result["llm_errors"] == 1


def test_parent_thinking_only_truncated_turn_gets_a_forced_continuation() -> None:
    finish = _FinishStub("finish_fold")
    llm = ScriptedLLM(
        [
            ProviderResponse(
                content="",
                reasoning_content="12k tokens of thinking",
                usage={"prompt_tokens": 10, "completion_tokens": 500, "total_tokens": 510},
            ),
            ProviderResponse(tool_calls=(ToolCall("f1", "finish_fold", {}),)),
        ]
    )
    events: list[tuple[str, dict[str, object]]] = []
    runner = AgentSessionRunner(
        llm=llm,
        tools=ToolRegistry([finish]),
        system_prompt="fold",
        config=_fold_config(max_response_tokens=500),
        event_sink=lambda event, payload: events.append((event, payload)),
    )
    assert runner.run("go").status == "finished"
    truncated = [payload for event, payload in events if event == "output_truncated"]
    assert truncated == [{"call_index": 1, "completion_tokens": 500, "max_tokens": 500}]
    second = llm.calls[1]["messages"]
    assert second[-1].role == "user"
    observation = json.loads(str(second[-1].content))
    assert observation["observation"] == "output_truncated"
    assert "被截断" in observation["message"]
    assert not any('"no_tool_call"' in str(message.content or "") for message in second)


def test_subagent_completed_surfaces_truncation_and_rounds() -> None:
    from autotrade.agent.subagent import OUTPUT_TRUNCATED_MARKER

    finish = _FinishStub("finish_fold")
    child = ScriptedLLM(
        [
            ProviderResponse(
                content="长报告" * 100,
                usage={"prompt_tokens": 10, "completion_tokens": 500, "total_tokens": 510},
            )
        ],
        context_window_tokens=128_000,
    )
    llm = ScriptedLLM(
        [
            ProviderResponse(tool_calls=(ToolCall("a1", "agent", {"agent": "auditor", "task": "audit"}),)),
            ProviderResponse(content="waiting"),
            ProviderResponse(tool_calls=(ToolCall("f1", "finish_fold", {}),)),
        ]
    )
    runner = AgentSessionRunner(
        llm=llm,
        tools=ToolRegistry([finish]),
        system_prompt="fold",
        config=_fold_config(),
        subagent=SubAgentEngine(
            llm=child,
            tools=ToolRegistry([DeclaredReadOnlyShell()]),
            config=SubAgentConfig(max_tokens=500),
        ),
    )
    assert runner.run("go").status == "finished"
    completed = next(
        json.loads(str(message.content))
        for message in llm.calls[-1]["messages"]
        if '"subagent_completed"' in str(message.content or "")
    )
    assert completed["truncated"] is True and completed["truncated_rounds"] == 1
    assert completed["rounds"] == 1 and completed["tool_calls"] == 0
    assert completed["summary"].endswith(OUTPUT_TRUNCATED_MARKER.format(limit=500))


def test_child_malformed_tool_call_keeps_its_analysis_and_re_issues_once() -> None:
    """A call the child wrote unparseably costs a round, not a generation.

    The text that did arrive is replayed as the child's own turn and only the
    call is asked for again; a second failure in the same streak falls back to
    the generic llm_error observation instead of replaying the analysis again.
    """

    class Malformed:
        model = "child"
        provider = "test"
        context_window_tokens = 128000

        def __init__(self) -> None:
            self.calls: list[tuple] = []

        def complete(self, messages, **kwargs):
            del kwargs
            self.calls.append(tuple(messages))
            if len(self.calls) <= 2:
                raise MalformedToolCallError(
                    "provider returned a malformed tool call (tool=shell: "
                    "Expecting value: line 1 column 9 (char 8)); no call from "
                    "this response was executed",
                    content="已经读完三个文件",
                    reasoning_content="逐个核对",
                )
            return ProviderResponse(content="recovered report")

    child = Malformed()
    events: list[tuple[str, dict[str, object]]] = []
    result = SubAgentEngine(
        llm=child,
        tools=ToolRegistry([DeclaredReadOnlyShell()]),
        event_sink=lambda event, payload: events.append((event, payload)),
    ).run("look", role="auditor")

    assert result["status"] == "completed" and result["summary"] == "recovered report"
    assert result["llm_errors"] == 2
    second = child.calls[1]
    assert second[-2].role == "assistant"
    assert second[-2].content == "已经读完三个文件"
    assert second[-2].reasoning_content == "逐个核对"
    assert json.loads(str(second[-1].content))["observation"] == "malformed_tool_call"
    third = child.calls[2]
    assert json.loads(str(third[-1].content))["observation"] == "llm_error"
    assert sum(1 for message in third if message.role == "assistant") == 1
    traced = [payload for event, payload in events if event == "subagent_llm_error"]
    assert [payload.get("error_type") for payload in traced] == [
        "malformed_tool_call",
        "malformed_tool_call",
    ]


def test_child_llm_error_is_traced_and_a_recovered_child_is_not_an_error() -> None:
    """A transient provider failure leaves a trace event and a counter; the
    child's outcome is what the surviving rounds produced."""

    class Flaky:
        model = "child"
        provider = "test"
        context_window_tokens = 128000

        def __init__(self) -> None:
            self.calls = 0

        def complete(self, messages, **kwargs):
            del messages, kwargs
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("provider returned an invalid stream")
            return ProviderResponse(content="recovered report")

    events: list[tuple[str, dict[str, object]]] = []
    result = SubAgentEngine(
        llm=Flaky(),
        tools=ToolRegistry([DeclaredReadOnlyShell()]),
        event_sink=lambda event, payload: events.append((event, payload)),
    ).run("look", role="auditor", parent_call_id="call_p")
    assert result["status"] == "completed" and result["summary"] == "recovered report"
    assert result["llm_errors"] == 1 and result["llm_calls"] == 2 and "error" not in result
    traced = [payload for event, payload in events if event == "subagent_llm_error"]
    assert len(traced) == 1
    assert traced[0]["round"] == 1 and traced[0]["parent_call_id"] == "call_p"
    assert "invalid stream" in traced[0]["llm_error"]
    assert "error_type" not in traced[0]
    # The runner's observation forwards the counter so the parent sees it.
    finish = _FinishStub("finish_fold")
    llm = ScriptedLLM(
        [
            ProviderResponse(tool_calls=(ToolCall("a1", "agent", {"agent": "auditor", "task": "look"}),)),
            ProviderResponse(content="waiting"),
            ProviderResponse(tool_calls=(ToolCall("f1", "finish_fold", {}),)),
        ]
    )
    runner = AgentSessionRunner(
        llm=llm,
        tools=ToolRegistry([finish]),
        system_prompt="fold",
        config=_fold_config(),
        subagent=SubAgentEngine(llm=Flaky(), tools=ToolRegistry([DeclaredReadOnlyShell()])),
    )
    assert runner.run("go").status == "finished"
    completed = next(
        json.loads(str(message.content))
        for message in llm.calls[-1]["messages"]
        if '"subagent_completed"' in str(message.content or "")
    )
    assert completed["ok"] is True and completed["llm_errors"] == 1
    assert "error" not in completed


def test_child_cut_short_by_worker_shutdown_is_cancelled_not_failed(monkeypatch) -> None:
    """A restart tears the thread pool down under a still-running child.

    From then on ``ThreadPoolExecutor.submit`` raises for the rest of the
    interpreter's life. That is the worker exiting, not the child failing, so
    the terminal record has to say cancelled and name the reason — otherwise
    every restart writes a child failure into the trace and the ledger."""

    class _ShutdownPool:
        def __init__(self, *args, **kwargs) -> None:
            del args, kwargs

        def __enter__(self):
            return self

        def __exit__(self, *exc_info) -> bool:
            return False

        def submit(self, *args, **kwargs):
            del args, kwargs
            raise RuntimeError(
                "cannot schedule new futures after interpreter shutdown"
            )

    monkeypatch.setattr(subagent_module, "ThreadPoolExecutor", _ShutdownPool)
    events: list[tuple[str, dict[str, object]]] = []
    child = ScriptedLLM(
        [
            ProviderResponse(
                tool_calls=(
                    ToolCall("g", "grep", {}),
                    ToolCall("r", "read_file", {}),
                )
            )
        ],
        context_window_tokens=128_000,
    )
    result = SubAgentEngine(
        llm=child,
        tools=ToolRegistry([_NamedTool("grep"), _NamedTool("read_file")]),
        event_sink=lambda event, payload: events.append((event, payload)),
    ).run("look", role="auditor")

    assert result["status"] == "cancelled"
    assert result["error"] == "Sub-agent cancelled by worker shutdown"
    ended = next(payload for event, payload in events if event == "subagent")
    assert ended["status"] == "cancelled"

    # What the parent, the trace projection and the console see.
    finish = _FinishStub("finish_fold")
    llm = ScriptedLLM(
        [
            ProviderResponse(
                tool_calls=(
                    ToolCall("a1", "agent", {"agent": "auditor", "task": "look"}),
                )
            ),
            ProviderResponse(content="waiting"),
            ProviderResponse(tool_calls=(ToolCall("f1", "finish_fold", {}),)),
        ]
    )
    attempts: list[dict[str, object]] = []
    runner = AgentSessionRunner(
        llm=llm,
        tools=ToolRegistry([finish]),
        system_prompt="fold",
        config=_fold_config(),
        subagent=SubAgentEngine(
            llm=ScriptedLLM(
                [
                    ProviderResponse(
                        tool_calls=(
                            ToolCall("g", "grep", {}),
                            ToolCall("r", "read_file", {}),
                        )
                    )
                ],
                context_window_tokens=128_000,
            ),
            tools=ToolRegistry([_NamedTool("grep"), _NamedTool("read_file")]),
        ),
        event_sink=lambda event, payload: attempts.append(payload)
        if event == "subagent_attempt"
        else None,
    )
    assert runner.run("go").status == "finished"
    assert [payload["status"] for payload in attempts] == ["cancelled"]
    assert attempts[0]["ok"] is False
    observation = next(
        json.loads(str(message.content))
        for message in llm.calls[-1]["messages"]
        if '"subagent_completed"' in str(message.content or "")
    )
    assert observation["status"] == "cancelled"
    assert observation["error"] == "Sub-agent cancelled by worker shutdown"


def test_child_without_a_report_is_not_completed() -> None:
    """``completed`` means a report reached the parent: a child whose rounds
    ran out and whose forced summary came back empty is an error."""
    silent = ScriptedLLM(
        [
            ProviderResponse(tool_calls=(ToolCall("s", "shell", {"argv": ["ls"]}),)),
            ProviderResponse(content="", reasoning_content="nothing to report"),
        ],
        context_window_tokens=128_000,
    )
    result = SubAgentEngine(
        llm=silent,
        tools=ToolRegistry([DeclaredReadOnlyShell()]),
        config=SubAgentConfig(max_rounds=1),
    ).run("dig", role="developer")
    assert result["status"] == "error" and result["summary"] == ""
    assert result["error"] == "Sub-agent ended without a report"
    assert result["tool_calls"] == 1 and result["llm_calls"] == 2


def test_child_thinking_level_reaches_the_budget_wrapped_gateway() -> None:
    """Production hands the child a budget wrapper around the gateway; the
    per-child ``thinking`` must still change the request that goes out."""
    from autotrade.environment.llm.deepseek import (
        OpenAICompatibleConfig,
        OpenAICompatibleProxy,
    )
    from autotrade.environment.llm.model_profiles import LOCAL_QWEN_MODEL

    class Transport:
        def __init__(self) -> None:
            self.bodies: list[dict[str, object]] = []

        def post(self, url, headers, body, timeout):
            del url, headers, timeout
            self.bodies.append(json.loads(body))
            return json.dumps(
                {
                    "model": LOCAL_QWEN_MODEL,
                    "choices": [{"message": {"content": "child report"}}],
                    "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
                }
            ).encode()

    transport = Transport()
    gateway = OpenAICompatibleProxy(
        OpenAICompatibleConfig(
            api_key="local-secret",
            provider="vllm",
            model=LOCAL_QWEN_MODEL,
            base_url="http://127.0.0.1:8011/v1",
            request_dialect="vllm-qwen",
            thinking_enabled=True,
            reasoning_effort="xhigh",
            stream_tool_calls=False,
            conversation_log_dir=None,
            context_window_tokens=262_144,
        ),
        transport=transport,
    )
    budgeted = SessionBudgetLLM(gateway, max_calls=4, deadline=time.monotonic() + 10, role="subagent")
    events: list[tuple[str, dict[str, object]]] = []
    engine = SubAgentEngine(
        llm=budgeted,
        tools=ToolRegistry([DeclaredReadOnlyShell()]),
        event_sink=lambda event, payload: events.append((event, payload)),
    )
    for level, expected in (
        ("low", {"enable_thinking": True, "reasoning_effort": "low"}),
        ("off", {"enable_thinking": False}),
    ):
        result = engine.run("look", role="auditor", thinking=level)
        assert result["status"] == "completed" and result["thinking_applied"] is True
        assert transport.bodies[-1]["chat_template_kwargs"] == expected
    # The session's own gateway and call budget are untouched by the clones.
    assert budgeted.calls == 2 and gateway.config.reasoning_effort == "xhigh"
    started = [payload for event, payload in events if event == "subagent_task"]
    assert [payload["thinking_applied"] for payload in started] == [True, True]


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class _ClockedLLM(ScriptedLLM):
    """Scripted replies that move a fake clock to a given time after each call."""

    def __init__(self, responses, clock: _Clock, times: list[float]) -> None:
        super().__init__(responses)
        self.clock = clock
        self.times = list(times)

    def complete(self, messages, **kwargs):
        response = super().complete(messages, **kwargs)
        if self.times:
            self.clock.now = self.times.pop(0)
        return response


def test_time_budget_notice_states_remaining_minutes_and_backtests() -> None:
    from autotrade.agent.runner import TIME_BUDGET_NOTICE_FRACTIONS

    assert TIME_BUDGET_NOTICE_FRACTIONS == (0.5, 0.75, 0.9)
    clock = _Clock()
    budget = InferenceTimeBudget(duration_seconds=100_000.0, clock=clock)
    llm = _ClockedLLM(
        [
            ProviderResponse(tool_calls=(ToolCall("r1", "read_file", {}),)),
            ProviderResponse(tool_calls=(ToolCall("s1", "smoke_backtest", {}),)),
            ProviderResponse(tool_calls=(ToolCall("r2", "read_file", {}),)),
            ProviderResponse(tool_calls=(ToolCall("f1", "finish_fold", {}),)),
        ],
        clock,
        [55_000.0, 80_000.0, 95_000.0],
    )
    events: list[tuple[str, dict[str, object]]] = []
    runner = AgentSessionRunner(
        llm=llm,
        tools=ToolRegistry([_NamedTool("read_file"), _NamedTool("smoke_backtest"), _FinishStub("finish_fold")]),
        system_prompt="fold",
        config=_fold_config(),
        time_budget=budget,
        event_sink=lambda event, payload: events.append((event, payload)),
    )
    assert runner.run("go").status == "finished"
    notices = [payload for event, payload in events if event == "time_budget_notice"]
    assert [n["elapsed_fraction"] for n in notices] == [0.5, 0.75, 0.9]
    assert [n["remaining_minutes"] for n in notices] == [750.0, 333.3, 83.3]
    assert [n["smoke_backtests"] for n in notices] == [0, 1, 1]
    assert all(n["daily_backtests"] == 0 and n["complete_validations"] == 0 for n in notices)
    delivered = [
        json.loads(str(message.content))
        for message in llm.calls[-1]["messages"]
        if '"time_budget_notice"' in str(message.content or "")
    ]
    assert len(delivered) == 3
    assert "smoke_backtest 1 次" in delivered[1]["message"] and "finish_fold" in delivered[1]["message"]

    # Crossing several fractions at once yields one notice, and Meta has no
    # backtests to report.
    clock = _Clock()
    budget = InferenceTimeBudget(duration_seconds=100_000.0, clock=clock)
    llm = _ClockedLLM(
        [
            ProviderResponse(tool_calls=(ToolCall("r1", "read_file", {}),)),
            ProviderResponse(tool_calls=(ToolCall("f1", "finish_meta", {}),)),
        ],
        clock,
        [80_000.0],
    )
    events = []
    runner = AgentSessionRunner(
        llm=llm,
        tools=ToolRegistry([_NamedTool("read_file"), _FinishStub("finish_meta")]),
        system_prompt="meta",
        config=_meta_config(),
        time_budget=budget,
        event_sink=lambda event, payload: events.append((event, payload)),
    )
    assert runner.run("go").status == "finished"
    notices = [payload for event, payload in events if event == "time_budget_notice"]
    assert notices == [{"elapsed_fraction": 0.75, "remaining_minutes": 333.3}]
    message = next(
        json.loads(str(m.content))["message"]
        for m in llm.calls[-1]["messages"]
        if '"time_budget_notice"' in str(m.content or "")
    )
    assert "finish_meta" in message


def test_role_table_is_the_single_source_for_roles_and_launch_defaults() -> None:
    from autotrade.agent.subagent import (
        DEFAULT_SUBAGENT_MAX_ROUNDS,
        SUBAGENT_ROLE_TABLE,
        SubAgentRole,
        resolve_subagent_max_turns,
        subagent_role,
    )

    assert tuple(role.name for role in SUBAGENT_ROLE_TABLE) == SUBAGENT_ROLES
    for role in SUBAGENT_ROLE_TABLE:
        assert allowed_subagent_tools("fold", role.name) == role.fold_tools
        assert role.shell == ("shell" in role.fold_tools)
        # No shipped role pins a level or a budget: both defer to the globals.
        assert role.thinking is None and role.max_turns is None
        assert role.default_thinking == DEFAULT_SUBAGENT_THINKING
        assert role.default_max_turns(DEFAULT_SUBAGENT_MAX_ROUNDS) == DEFAULT_SUBAGENT_MAX_ROUNDS
        assert subagent_role(role.name) is role
        line = (
            f"{role.name}：{role.description}（"
            f"{'有 Sandbox shell 与 smoke_backtest、可写' if role.shell else '只读 glob/grep/read_file，不能执行'}；"
            f"默认 thinking {DEFAULT_SUBAGENT_THINKING}、max_turns {DEFAULT_SUBAGENT_MAX_ROUNDS}）"
        )
        assert line in AGENT_TOOL_SPEC.input_schema["properties"]["agent"]["description"]
    assert {role.name for role in SUBAGENT_ROLE_TABLE if role.shell} == {"developer", "general-purpose"}
    with pytest.raises(ValueError, match="not allowed"):
        subagent_role("reader")
    with pytest.raises(ValueError, match="thinking"):
        SubAgentRole("x", "d", frozenset(), fold_mission="m", meta_mission="m", thinking="turbo")
    with pytest.raises(ValueError, match="max_turns"):
        SubAgentRole("x", "d", frozenset(), fold_mission="m", meta_mission="m", max_turns=0)

    # Precedence: call argument > role default > global default.
    pinned = SubAgentRole("x", "d", frozenset(), fold_mission="m", meta_mission="m", thinking="low", max_turns=12)
    assert pinned.default_thinking == "low" and pinned.default_max_turns(48) == 12
    assert resolve_subagent_max_turns(None, "auditor", 48) == 48
    assert resolve_subagent_max_turns(None, "auditor", None) is None
    assert resolve_subagent_max_turns(7, "auditor", 48) == 7
    assert normalize_subagent_thinking(None, "auditor") == DEFAULT_SUBAGENT_THINKING
    assert normalize_subagent_thinking("xhigh", "auditor") == "xhigh"
    for bad in (0, -1, True, "3"):
        with pytest.raises(ValueError, match="max_turns"):
            resolve_subagent_max_turns(bad, "auditor", 48)
    # The tool text states the precedence and the ranges.
    assert "本次调用参数 > 角色默认" in AGENT_TOOL_DESCRIPTION
    max_turns_field = AGENT_TOOL_SPEC.input_schema["properties"]["max_turns"]
    assert max_turns_field["minimum"] == 1 and "max_llm_calls 的一半" in max_turns_field["description"]
    assert "off/low/medium/xhigh" in AGENT_TOOL_SPEC.input_schema["properties"]["thinking"]["description"]


def test_launch_precedence_reaches_the_child_and_its_trace(monkeypatch: pytest.MonkeyPatch) -> None:
    """A pinned role tier beats the global default and loses to the call argument;
    the effective values are what ``subagent_task``/``subagent`` record."""

    import dataclasses

    from autotrade.agent import subagent as module

    pinned = dataclasses.replace(module.subagent_role("auditor"), thinking="low", max_turns=3)
    monkeypatch.setitem(module._ROLES_BY_NAME, "auditor", pinned)

    def _run(arguments: dict[str, object]) -> tuple[dict[str, object], dict[str, object]]:
        child = ScriptedLLM([ProviderResponse(content="report")], context_window_tokens=128_000)
        events: list[tuple[str, dict[str, object]]] = []
        runner = AgentSessionRunner(
            llm=ScriptedLLM([]),
            tools=ToolRegistry(),
            system_prompt="fold",
            config=_fold_config(),
            subagent=SubAgentEngine(llm=child, tools=ToolRegistry([DeclaredReadOnlyShell()])),
            event_sink=lambda event, payload: events.append((event, payload)),
        )
        assert runner.tools.invoke("agent", {"agent": "auditor", "task": "dig", **arguments}).ok
        assert runner._wait_subagent_jobs()[-1]["ok"] is True
        started = next(payload for event, payload in events if event == "subagent_task")
        ended = next(payload for event, payload in events if event == "subagent")
        return started, ended

    started, ended = _run({})
    assert (started["thinking"], started["rounds_limit"]) == ("low", 3)
    assert (ended["thinking"], ended["rounds_limit"]) == ("low", 3)
    assert isinstance(started["thinking_applied"], bool)
    started, ended = _run({"thinking": "xhigh", "max_turns": 5})
    assert (started["thinking"], started["rounds_limit"]) == ("xhigh", 5)
    assert (ended["thinking"], ended["rounds_limit"]) == ("xhigh", 5)

    # Untouched roles still resolve to the global defaults.
    monkeypatch.undo()
    started, _ended = _run({})
    assert (started["thinking"], started["rounds_limit"]) == (
        module.DEFAULT_SUBAGENT_THINKING,
        module.DEFAULT_SUBAGENT_MAX_ROUNDS,
    )


def test_long_child_report_is_clipped_inline_and_spilled_for_read_back(tmp_path: Path) -> None:
    from autotrade.agent.subagent import SUBAGENT_REPORT_MAX_CHARS
    from autotrade.environment.tools import ReadFileTool

    assert SUBAGENT_REPORT_MAX_CHARS == 6_000
    assert f"{SUBAGENT_REPORT_MAX_CHARS} 字符" in AGENT_TOOL_DESCRIPTION
    report = "\n".join(f"第{index}行 证据与结论" for index in range(700))
    assert len(report) > SUBAGENT_REPORT_MAX_CHARS
    roots = SearchRoots(SafeWorkspace(tmp_path))
    finish = _FinishStub("finish_fold")
    llm = ScriptedLLM(
        [
            ProviderResponse(tool_calls=(ToolCall("a1", "agent", {"agent": "auditor", "task": "audit"}),)),
            ProviderResponse(content="waiting"),
            ProviderResponse(tool_calls=(ToolCall("f1", "finish_fold", {}),)),
        ]
    )
    events: list[tuple[str, dict[str, object]]] = []
    runner = AgentSessionRunner(
        llm=llm,
        tools=ToolRegistry([ReadFileTool(roots), finish]),
        system_prompt="fold",
        config=_fold_config(),
        subagent=SubAgentEngine(
            llm=ScriptedLLM([ProviderResponse(content=report)], context_window_tokens=128_000),
            tools=ToolRegistry([DeclaredReadOnlyShell()]),
        ),
        event_sink=lambda event, payload: events.append((event, payload)),
    )
    assert runner.tools.result_store() is roots
    assert runner.run("go").status == "finished"
    completed = next(
        json.loads(str(message.content))
        for message in llm.calls[-1]["messages"]
        if '"subagent_completed"' in str(message.content or "")
    )
    assert completed["summary"] == report[:SUBAGENT_REPORT_MAX_CHARS]
    assert completed["summary_chars"] == len(report)
    assert completed["summary_delivered_chars"] == SUBAGENT_REPORT_MAX_CHARS
    assert completed["summary_truncated"] is True
    assert completed["result_root"] == "workspace" and "read_file" in completed["result_hint"]
    # The parent reads the full report back through the same path as any
    # other spilled tool result; the reference carries no host path.
    ref = completed["result_ref"]
    assert ref.startswith("logs/tool_results/agent_report_") and str(tmp_path) not in ref
    assert (tmp_path / ref).read_text(encoding="utf-8") == report
    page = runner.tools.invoke("read_file", {"root": "workspace", "path": ref, "limit": 2})
    assert page.ok and "1\t第0行 证据与结论" in str(page.value["content"])

    # read_file pages by LINE, so the clip is reported in lines too: a parent
    # told only how many characters it received extrapolates the resume point
    # in characters and reads past the end of the file for nothing.
    lines = report.splitlines()
    assert completed["summary_lines"] == len(lines)
    resume = completed["resume_line"]
    assert f"offset={resume}" in completed["result_hint"]
    blind = runner.tools.invoke(
        "read_file",
        {"root": "workspace", "path": ref, "offset": completed["summary_delivered_chars"]},
    )
    assert blind.ok and blind.value["returned"] == 0
    rest = runner.tools.invoke(
        "read_file", {"root": "workspace", "path": ref, "offset": resume, "limit": 5}
    )
    assert rest.ok and rest.value["returned"] == 5
    # The line the clip fell inside comes back whole rather than being lost.
    assert str(rest.value["content"]).splitlines()[0] == f"{resume + 1}\t{lines[resume]}"
    assert lines[resume].startswith(str(completed["summary"]).rsplit("\n", 1)[-1])

    attempt = next(payload for event, payload in events if event == "subagent_attempt")
    assert attempt["summary_chars"] == len(report)
    assert attempt["summary_delivered_chars"] == SUBAGENT_REPORT_MAX_CHARS
    assert attempt["summary_lines"] == len(lines) and attempt["resume_line"] == resume
    assert attempt["summary_truncated"] is True and attempt["result_ref"] == ref


def test_short_child_report_is_delivered_whole_and_no_store_is_explicit() -> None:
    from autotrade.agent.subagent import SUBAGENT_REPORT_MAX_CHARS, deliver_subagent_report

    assert ToolRegistry([DeclaredReadOnlyShell()]).result_store() is None
    whole = deliver_subagent_report("短汇报", None)
    assert whole == {"summary": "短汇报", "summary_chars": 3}
    long = "x" * (SUBAGENT_REPORT_MAX_CHARS + 1)
    clipped = deliver_subagent_report(long, None)
    assert clipped["summary"] == long[:SUBAGENT_REPORT_MAX_CHARS]
    assert clipped["summary_truncated"] is True and "result_ref" not in clipped
    assert "not persisted" in clipped["result_hint"]

    # Through the runner without search tools: the clip is still visible.
    finish = _FinishStub("finish_fold")
    llm = ScriptedLLM(
        [
            ProviderResponse(tool_calls=(ToolCall("a1", "agent", {"agent": "auditor", "task": "audit"}),)),
            ProviderResponse(content="waiting"),
            ProviderResponse(tool_calls=(ToolCall("f1", "finish_fold", {}),)),
        ]
    )
    runner = AgentSessionRunner(
        llm=llm,
        tools=ToolRegistry([finish]),
        system_prompt="fold",
        config=_fold_config(),
        subagent=SubAgentEngine(
            llm=ScriptedLLM([ProviderResponse(content=long)], context_window_tokens=128_000),
            tools=ToolRegistry([DeclaredReadOnlyShell()]),
        ),
    )
    assert runner.run("go").status == "finished"
    completed = next(
        json.loads(str(message.content))
        for message in llm.calls[-1]["messages"]
        if '"subagent_completed"' in str(message.content or "")
    )
    assert completed["summary_truncated"] is True and "result_ref" not in completed
    assert len(completed["summary"]) == SUBAGENT_REPORT_MAX_CHARS


def test_prompts_carry_the_todo_convention_and_per_launch_knobs() -> None:
    fold = build_system_prompt(mode="fold", experiment_facts={})
    meta = build_system_prompt(mode="meta", experiment_facts={})
    for prompt, finish in ((fold, "finish_fold"), (meta, "finish_meta")):
        assert "`TODO.md`" in prompt and "不需要任何人工参与" in prompt
        assert "每个任务一行，写明负责方、状态和一句话结果" in prompt
        assert f"`{finish}` 前核对全部条目" in prompt
    assert "`thinking` 与 `max_turns` 由你按次决定" in FOLD_WORKFLOW_SECTION


class _RecordingLLM:
    """Scripted child model that keeps the messages of every request."""

    model = "child"
    provider = "test"
    context_window_tokens = 128_000

    def __init__(self, responses: list[ProviderResponse]) -> None:
        self._responses = list(responses)
        self.seen: list[list[tuple[str, str]]] = []

    def complete(self, messages, **kwargs):
        del kwargs
        self.seen.append([(message.role, str(message.content or "")) for message in messages])
        return self._responses.pop(0)


def test_steer_is_delivered_before_the_childs_next_round() -> None:
    steer: deque[str] = deque()
    text = "范围缩小到 output/main.py，然后立即汇报"

    class _SteeringShell(DeclaredReadOnlyShell):
        # The parent steers while the child's tool is running.
        def invoke(self, arguments: Mapping[str, object]) -> ToolResult:
            steer.append(text)
            return super().invoke(arguments)

    llm = _RecordingLLM(
        [
            ProviderResponse(tool_calls=(ToolCall("s", "shell", {"argv": ["ls"]}),)),
            ProviderResponse(content="done"),
        ]
    )
    events: list[tuple[str, dict[str, object]]] = []
    result, transcript = SubAgentEngine(
        llm=llm,
        tools=ToolRegistry([_SteeringShell()]),
        event_sink=lambda event, payload: events.append((event, payload)),
    ).run_with_transcript("look", role="developer", parent_call_id="call_p", steer_queue=steer)
    assert result["status"] == "completed" and result["steers"] == 1 and not steer
    # Round 1 never saw it; round 2 got it after the tool result, before the model call.
    assert not any(
        role == "user" and content.startswith(STEER_MESSAGE_LABEL) for role, content in llm.seen[0]
    )
    assert llm.seen[1][-1] == ("user", f"{STEER_MESSAGE_LABEL} {text}")
    assert llm.seen[1][-2][0] == "tool"
    assert [payload for event, payload in events if event == "subagent_steer"] == [
        {
            "task_id": result["task_id"],
            "role": "developer",
            "round": 2,
            "chars": len(text),
            "delivery": "delivered",
            "parent_call_id": "call_p",
        }
    ]
    assert any(STEER_MESSAGE_LABEL in str(message.content) for message in transcript)
    # A queued child (instruction sent before its first round) reads it right after the task.
    early = _RecordingLLM([ProviderResponse(content="ok")])
    result = SubAgentEngine(llm=early, tools=ToolRegistry([DeclaredReadOnlyShell()])).run(
        "queued task", role="auditor", steer_queue=deque(["先看 README"])
    )
    assert result["steers"] == 1
    assert early.seen[0][-2:] == [("user", "queued task"), ("user", f"{STEER_MESSAGE_LABEL} 先看 README")]
    # No instruction: no message, no counter.
    plain = SubAgentEngine(
        llm=ScriptedLLM([ProviderResponse(content="ok")]), tools=ToolRegistry([DeclaredReadOnlyShell()])
    ).run("plain", role="auditor")
    assert "steers" not in plain


class _SteerProbeLLM:
    """Child model shared by two children: the ``slow`` one blocks in its first
    round behind a gate and then makes a tool call; every other request answers."""

    model = "child"
    provider = "test"
    context_window_tokens = 128_000

    def __init__(self, started: threading.Event, release: threading.Event) -> None:
        self.started = started
        self.release = release
        self.seen: dict[str, list[list[tuple[str, str]]]] = {}

    def complete(self, messages, **kwargs):
        del kwargs
        blob = [(message.role, str(message.content or "")) for message in messages]
        task = "slow" if any("slow task" in content for _role, content in blob) else "queued"
        self.seen.setdefault(task, []).append(blob)
        if task == "slow" and len(self.seen["slow"]) == 1:
            self.started.set()
            if not self.release.wait(5):
                raise TimeoutError("steer gate")
            return ProviderResponse(tool_calls=(ToolCall("s", "shell", {"argv": ["ls"]}),))
        return ProviderResponse(content=f"{task} done")


def test_agent_message_action_steers_running_and_queued_children() -> None:
    started = threading.Event()
    release = threading.Event()
    child = _SteerProbeLLM(started, release)
    events: list[tuple[str, dict[str, object]]] = []
    runner = AgentSessionRunner(
        llm=ScriptedLLM([]),
        tools=ToolRegistry(),
        system_prompt="fold",
        config=_fold_config(),
        subagent=SubAgentEngine(
            llm=child,
            tools=ToolRegistry([DeclaredReadOnlyShell()]),
            config=SubAgentConfig(max_concurrent=1),
        ),
        event_sink=lambda event, payload: events.append((event, payload)),
    )
    slow = runner.tools.invoke("agent", {"agent": "developer", "task": "slow task"})
    queued = runner.tools.invoke("agent", {"agent": "auditor", "task": "queued task"})
    assert slow.ok and queued.ok and queued.value["queued"] is True
    slow_id, queued_id = slow.value["task_id"], queued.value["task_id"]
    assert started.wait(3)
    try:
        ack = runner.tools.invoke("agent", {"action": "message", "task_id": slow_id, "text": " 提前收尾 "})
        assert ack.ok and ack.value == {"status": "queued", "task_id": slow_id, "delivered_at_round": None}
        ack = runner.tools.invoke("agent", {"action": "message", "task_id": queued_id, "text": "先看 README"})
        assert ack.ok and ack.value["child_queued"] is True
        # Bounded and sanitised like a brief; unknown ids are typed errors.
        too_long = runner.tools.invoke(
            "agent", {"action": "message", "task_id": slow_id, "text": "x" * (SUBAGENT_STEER_MAX_CHARS + 1)}
        )
        assert too_long.ok is False and too_long.value["error_type"] == "schema_error"
        blank = runner.tools.invoke("agent", {"action": "message", "task_id": slow_id, "text": "   "})
        assert blank.ok is False and "non-empty" in blank.error
        unknown = runner.tools.invoke("agent", {"action": "message", "task_id": "agent_nope", "text": "hi"})
        assert unknown.ok is False and unknown.value["error_type"] == "unknown_subagent"
    finally:
        release.set()
    assert [record["ok"] for record in runner._wait_subagent_jobs()] == [True, True]
    # The running child read it after its tool result, before round 2; the
    # queued child read it right after its task, before round 1.
    assert child.seen["slow"][1][-1] == ("user", f"{STEER_MESSAGE_LABEL} 提前收尾")
    assert child.seen["slow"][1][-2][0] == "tool"
    assert child.seen["queued"][0][-2:] == [
        ("user", "queued task"),
        ("user", f"{STEER_MESSAGE_LABEL} 先看 README"),
    ]
    steers = [payload for event, payload in events if event == "subagent_steer"]
    assert [(p["task_id"], p["delivery"], p.get("round")) for p in steers] == [
        (slow_id, "queued", None),
        (queued_id, "queued", None),
        (slow_id, "delivered", 2),
        (queued_id, "delivered", 1),
    ]
    assert all(p["chars"] > 0 and "text" not in p for p in steers)
    completed = {
        payload["task_id"]: payload
        for payload in (
            json.loads(str(message.content))
            for message in runner._append_subagent_observations([])
        )
    }
    assert completed[slow_id]["steers"] == 1 and completed[queued_id]["steers"] == 1
    assert "steers_undelivered" not in completed[slow_id]
    # A finished child takes follow-ups through resume, not message.
    finished = runner.tools.invoke("agent", {"action": "message", "task_id": slow_id, "text": "more"})
    assert finished.ok is False and finished.value["error_type"] == "subagent_finished"
    assert "resume" in finished.error


def test_steer_the_child_never_read_is_reported_undelivered() -> None:
    started = threading.Event()
    release = threading.Event()
    runner = AgentSessionRunner(
        llm=ScriptedLLM([]),
        tools=ToolRegistry(),
        system_prompt="fold",
        config=_fold_config(),
        subagent=SubAgentEngine(
            llm=_GateLLM(started, release), tools=ToolRegistry([DeclaredReadOnlyShell()])
        ),
    )
    launched = runner.tools.invoke("agent", {"agent": "auditor", "task": "one round"})
    assert launched.ok and started.wait(3)
    # The child is inside its only model call; it reports without another round.
    assert runner.tools.invoke(
        "agent", {"action": "message", "task_id": launched.value["task_id"], "text": "stop"}
    ).ok
    release.set()
    assert runner._wait_subagent_jobs()[0]["ok"] is True
    completed = json.loads(str(runner._append_subagent_observations([])[0].content))
    assert completed["status"] == "completed" and completed["steers_undelivered"] == 1
    assert "steers" not in completed
