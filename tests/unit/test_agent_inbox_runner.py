"""Phase 2B: Runner inbox safe points, interrupt pairing, session isolation."""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from pathlib import Path

from autotrade.agent.explore import ExploreSubAgentEngine
from autotrade.agent.runner import (
    AgentSessionConfig,
    AgentSessionRunner,
    INBOX_SAFE_AFTER_LLM_BEFORE_TOOLS,
    INBOX_SAFE_AFTER_PARALLEL_READONLY,
    INBOX_SAFE_AFTER_TOOLS_BEFORE_LLM,
    INBOX_SAFE_BEFORE_LLM,
    INBOX_SAFE_BETWEEN_SERIAL_TOOLS,
)
from autotrade.environment.artifacts import new_revision_id
from autotrade.environment.llm import ChatMessage, ProviderResponse, ScriptedLLM, ToolCall
from autotrade.environment.step_tree import StepTree
from autotrade.environment.tools import (
    FinishFoldTool,
    ToolRegistry,
    ToolResult,
    ToolSpec,
)
from autotrade.pipelines.agent_inbox import (
    bind_session_inbox,
    enqueue_inbox_message,
    inbox_path,
    list_unconsumed_messages,
)
from autotrade.pipelines.hitl_state import ControlState, write_control
from autotrade.pipelines.interactive import InteractiveExperimentRunner
from autotrade.pipelines.ledger import ExperimentLedger
from autotrade.pipelines.meta_inputs import compact_agent_trace
from autotrade.webui.traces import project_trace_blocks
from tests.unit.test_interactive_runner import RecordingExecutor, sessions_for

SESSION_A = "epoch_001/fold_2022Q2"
SESSION_B = "epoch_001/fold_2022Q1"


@dataclass
class FakeNotice:
    message_id: str
    text: str
    interrupt: bool = False


@dataclass
class FakeInbox:
    items: list[FakeNotice] = field(default_factory=list)
    consumed: list[str] = field(default_factory=list)

    def pending(self) -> tuple[FakeNotice, ...]:
        return tuple(self.items)

    def consume(self, message_id: str) -> str:
        self.items = [item for item in self.items if item.message_id != message_id]
        self.consumed.append(message_id)
        return "consumed"

    def push(self, notice: FakeNotice) -> None:
        self.items.append(notice)


class RecordingTool:
    def __init__(self, name: str, *, mutating: bool = False, on_invoke=None):
        self.spec = ToolSpec(
            name,
            "test tool",
            {"type": "object", "properties": {}, "required": []},
            mutating=mutating,
        )
        self.calls: list[object] = []
        self.on_invoke = on_invoke

    def invoke(self, arguments):
        self.calls.append(arguments)
        if self.on_invoke is not None:
            self.on_invoke(self, arguments)
        return ToolResult(True, value={"name": self.spec.name})


def _finish_tool(root: Path) -> tuple[FinishFoldTool, str]:
    output = root / "output"
    output.mkdir(parents=True, exist_ok=True)
    (output / "main.py").write_text(
        "def generate_orders(context):\n    return []\n", encoding="utf-8"
    )
    tree = StepTree(root / "steps")
    node_id = tree.record_step(
        output,
        epoch_id="epoch_001",
        fold_id="fold_ref_ab",
        run_id="run_x",
        result_name="valid_000",
        revision_id=new_revision_id("revision"),
        metrics={"total_return": 0.01},
        complete_validation=True,
    )
    return FinishFoldTool(tree, fold_id="fold_ref_ab", run_id="run_x"), node_id


def _user_texts(messages: list[ChatMessage]) -> list[str]:
    return [
        message.content or ""
        for message in messages
        if message.role == "user" and message.content
    ]


def test_before_llm_applies_guidance_without_system_or_manifest(tmp_path: Path) -> None:
    finish, node_id = _finish_tool(tmp_path)
    inbox = FakeInbox([FakeNotice("m1", "先看回撤", interrupt=False)])
    events: list[tuple[str, dict[str, object]]] = []
    llm = ScriptedLLM(
        [
            ProviderResponse(
                tool_calls=(ToolCall("f", "finish_fold", {"node_id": node_id}),)
            )
        ]
    )
    runner = AgentSessionRunner(
        llm=llm,
        tools=ToolRegistry([finish]),
        system_prompt="daily JSON only",
        config=AgentSessionConfig(max_llm_calls=2),
        event_sink=lambda event, payload: events.append((event, payload)),
        inbox=inbox,
    )
    assert runner.run("finish the validated strategy").status == "finished"
    first = llm.calls[0]["messages"]
    assert first[0].role == "system"
    assert first[0].content == "daily JSON only"
    assert "先看回撤" in _user_texts(first)
    assert inbox.consumed == ["m1"]
    user_events = [payload for event, payload in events if event == "user_message"]
    assert user_events[0]["message_id"] == "m1"
    assert user_events[0]["interrupt"] is False
    assert user_events[0]["safe_point"] == INBOX_SAFE_BEFORE_LLM
    assert user_events[0]["content"] == "先看回撤"
    assert "applied_at" in user_events[0]


def test_interrupt_after_llm_skips_unstarted_tools_with_pairing(tmp_path: Path) -> None:
    finish, node_id = _finish_tool(tmp_path)
    write = RecordingTool("write_file", mutating=True)
    inbox = FakeInbox()
    events: list[tuple[str, dict[str, object]]] = []

    class InjectingLLM(ScriptedLLM):
        def complete(self, messages, **kwargs):
            if not self.calls:
                inbox.push(FakeNotice("int1", "立刻停手", interrupt=True))
            return super().complete(messages, **kwargs)

    llm = InjectingLLM(
        [
            ProviderResponse(
                tool_calls=(
                    ToolCall("w", "write_file", {"path": "output/main.py"}),
                    ToolCall("f", "finish_fold", {"node_id": node_id}),
                )
            ),
            ProviderResponse(
                tool_calls=(ToolCall("done", "finish_fold", {"node_id": node_id}),)
            ),
        ]
    )
    runner = AgentSessionRunner(
        llm=llm,
        tools=ToolRegistry([write, finish]),
        system_prompt="daily JSON only",
        config=AgentSessionConfig(max_llm_calls=3),
        event_sink=lambda event, payload: events.append((event, payload)),
        inbox=inbox,
    )
    result = runner.run("write then finish")
    assert result.status == "finished"
    assert write.calls == []
    second = llm.calls[1]["messages"]
    tools = [message for message in second if message.role == "tool"]
    assert {message.tool_call_id for message in tools} == {"w", "f"}
    for message in tools:
        payload = json.loads(message.content or "{}")
        assert payload["ok"] is False
        assert payload["observation"] == "interrupted_by_user"
        assert payload["error"] == "interrupted_by_user"
    skipped = [payload for event, payload in events if event == "tool_skipped"]
    assert [item["tool"] for item in skipped] == ["write_file", "finish_fold"]
    assert all(item["reason"] == "interrupted_by_user" for item in skipped)
    assert all(item["safe_point"] == INBOX_SAFE_AFTER_LLM_BEFORE_TOOLS for item in skipped)
    user_events = [payload for event, payload in events if event == "user_message"]
    assert user_events[0]["interrupt"] is True
    assert user_events[0]["safe_point"] == INBOX_SAFE_AFTER_LLM_BEFORE_TOOLS
    assert "立刻停手" in _user_texts(second)


def test_started_mutating_tool_is_not_killed(tmp_path: Path) -> None:
    finish, node_id = _finish_tool(tmp_path)
    inbox = FakeInbox()
    started = threading.Event()
    release = threading.Event()
    later = RecordingTool("grep")

    def _write(_tool, _arguments) -> None:
        started.set()
        assert release.wait(timeout=2)

    write = RecordingTool("write_file", mutating=True, on_invoke=_write)

    def _pump() -> None:
        assert started.wait(timeout=2)
        inbox.push(FakeNotice("late", "打断后续", interrupt=True))
        release.set()

    pump = threading.Thread(target=_pump)
    pump.start()
    llm = ScriptedLLM(
        [
            ProviderResponse(
                tool_calls=(
                    ToolCall("w", "write_file", {}),
                    ToolCall("g", "grep", {}),
                    ToolCall("f", "finish_fold", {"node_id": node_id}),
                )
            ),
            ProviderResponse(
                tool_calls=(ToolCall("done", "finish_fold", {"node_id": node_id}),)
            ),
        ]
    )
    events: list[tuple[str, dict[str, object]]] = []
    runner = AgentSessionRunner(
        llm=llm,
        tools=ToolRegistry([write, later, finish]),
        system_prompt="daily JSON only",
        config=AgentSessionConfig(max_llm_calls=3),
        event_sink=lambda event, payload: events.append((event, payload)),
        inbox=inbox,
    )
    assert runner.run("mutate then search").status == "finished"
    pump.join(timeout=2)
    assert write.calls
    assert later.calls == []
    second = llm.calls[1]["messages"]
    tools = [message for message in second if message.role == "tool"]
    by_id = {message.tool_call_id: json.loads(message.content or "{}") for message in tools}
    assert by_id["w"]["ok"] is True
    assert by_id["g"]["ok"] is False
    assert by_id["g"]["error"] == "interrupted_by_user"
    assert by_id["f"]["ok"] is False
    skipped = [payload for event, payload in events if event == "tool_skipped"]
    assert [item["tool"] for item in skipped] == ["grep", "finish_fold"]
    assert skipped[0]["safe_point"] == INBOX_SAFE_BETWEEN_SERIAL_TOOLS


def test_non_interrupt_after_llm_waits_for_tool_pairing(tmp_path: Path) -> None:
    finish, node_id = _finish_tool(tmp_path)
    inbox = FakeInbox()
    grep = RecordingTool("grep")

    class InjectingLLM(ScriptedLLM):
        def complete(self, messages, **kwargs):
            if not self.calls:
                inbox.push(FakeNotice("g1", "工具后再看", interrupt=False))
            return super().complete(messages, **kwargs)

    llm = InjectingLLM(
        [
            ProviderResponse(tool_calls=(ToolCall("g", "grep", {}),)),
            ProviderResponse(
                tool_calls=(ToolCall("f", "finish_fold", {"node_id": node_id}),)
            ),
        ]
    )
    events: list[tuple[str, dict[str, object]]] = []
    runner = AgentSessionRunner(
        llm=llm,
        tools=ToolRegistry([grep, finish]),
        system_prompt="daily JSON only",
        config=AgentSessionConfig(max_llm_calls=3),
        event_sink=lambda event, payload: events.append((event, payload)),
        inbox=inbox,
    )
    assert runner.run("search then finish").status == "finished"
    assert grep.calls
    first_tools = [message for message in llm.calls[0]["messages"] if message.role == "user"]
    assert all("工具后再看" not in (message.content or "") for message in first_tools)
    assert "工具后再看" in _user_texts(llm.calls[1]["messages"])
    tool_then_user = [
        message.role for message in llm.calls[1]["messages"] if message.role in {"tool", "user"}
    ]
    assert tool_then_user[-2:] == ["tool", "user"] or "tool" in tool_then_user
    user_events = [payload for event, payload in events if event == "user_message"]
    assert user_events[0]["safe_point"] == INBOX_SAFE_AFTER_TOOLS_BEFORE_LLM
    assert user_events[0]["interrupt"] is False


def test_after_parallel_readonly_applies_before_next_llm(tmp_path: Path) -> None:
    finish, node_id = _finish_tool(tmp_path)
    inbox = FakeInbox()
    glob = RecordingTool("glob", on_invoke=lambda *_: inbox.push(FakeNotice("p1", "并行后看")))
    grep = RecordingTool("grep")
    llm = ScriptedLLM(
        [
            ProviderResponse(
                tool_calls=(ToolCall("a", "glob", {}), ToolCall("b", "grep", {}))
            ),
            ProviderResponse(
                tool_calls=(ToolCall("f", "finish_fold", {"node_id": node_id}),)
            ),
        ]
    )
    events: list[tuple[str, dict[str, object]]] = []
    runner = AgentSessionRunner(
        llm=llm,
        tools=ToolRegistry([glob, grep, finish]),
        system_prompt="daily JSON only",
        config=AgentSessionConfig(max_llm_calls=3),
        event_sink=lambda event, payload: events.append((event, payload)),
        inbox=inbox,
    )
    assert runner.run("read then finish").status == "finished"
    assert glob.calls and grep.calls
    second = llm.calls[1]["messages"]
    assert "并行后看" in _user_texts(second)
    user_events = [payload for event, payload in events if event == "user_message"]
    assert user_events[0]["safe_point"] == INBOX_SAFE_AFTER_PARALLEL_READONLY


def test_explore_does_not_consume_inbox_until_parent_returns(tmp_path: Path) -> None:
    finish, node_id = _finish_tool(tmp_path)
    inbox = FakeInbox()
    shell = RecordingTool("shell")

    class ExploreLLM(ScriptedLLM):
        def complete(self, messages, **kwargs):
            inbox.push(FakeNotice("from_child", "子代理期间到达"))
            return super().complete(messages, **kwargs)

    explore_llm = ExploreLLM([ProviderResponse(content="探查完成")])
    parent_llm = ScriptedLLM(
        [
            ProviderResponse(
                tool_calls=(ToolCall("e", "explore", {"task": "inspect workspace"}),)
            ),
            ProviderResponse(
                tool_calls=(ToolCall("f", "finish_fold", {"node_id": node_id}),)
            ),
        ]
    )
    runner = AgentSessionRunner(
        llm=parent_llm,
        tools=ToolRegistry([finish]),
        system_prompt="daily JSON only",
        config=AgentSessionConfig(max_llm_calls=3),
        explore=ExploreSubAgentEngine(llm=explore_llm, tools=ToolRegistry([shell])),
        inbox=inbox,
    )
    assert runner.run("delegate then finish").status == "finished"
    explore_seen = _user_texts(explore_llm.calls[0]["messages"])
    assert all("子代理期间到达" not in text for text in explore_seen)
    assert "子代理期间到达" in _user_texts(parent_llm.calls[1]["messages"])
    assert inbox.consumed == ["from_child"]


def test_session_inbox_hook_isolates_sessions_and_run_ids(tmp_path: Path) -> None:
    path = inbox_path(tmp_path)
    enqueue_inbox_message(path, session_key=SESSION_A, text="只给 A")
    enqueue_inbox_message(path, session_key=SESSION_B, text="只给 B")
    hook_a = bind_session_inbox(
        tmp_path, session_key=SESSION_A, run_id="run_a", committed_run_ids=()
    )
    hook_b = bind_session_inbox(
        tmp_path, session_key=SESSION_B, run_id="run_b", committed_run_ids=()
    )
    assert hook_a is not None and hook_b is not None
    assert [item.text for item in hook_a.pending()] == ["只给 A"]
    assert [item.text for item in hook_b.pending()] == ["只给 B"]
    finish, node_id = _finish_tool(tmp_path)
    llm = ScriptedLLM(
        [
            ProviderResponse(
                tool_calls=(ToolCall("f", "finish_fold", {"node_id": node_id}),)
            )
        ]
    )
    runner = AgentSessionRunner(
        llm=llm,
        tools=ToolRegistry([finish]),
        system_prompt="daily JSON only",
        config=AgentSessionConfig(max_llm_calls=2),
        inbox=hook_a,
    )
    assert runner.run("finish").status == "finished"
    assert "只给 A" in _user_texts(llm.calls[0]["messages"])
    assert all("只给 B" not in text for text in _user_texts(llm.calls[0]["messages"]))
    assert list_unconsumed_messages(path, SESSION_A) == ()
    assert [item.text for item in list_unconsumed_messages(path, SESSION_B)] == ["只给 B"]


def test_bind_none_without_identity_and_same_run_does_not_repeat(tmp_path: Path) -> None:
    assert bind_session_inbox(tmp_path, session_key="", run_id="run") is None
    assert bind_session_inbox(tmp_path, session_key=SESSION_A, run_id="") is None
    path = inbox_path(tmp_path)
    queued = enqueue_inbox_message(path, session_key=SESSION_A, text="一次即可")
    hook = bind_session_inbox(
        tmp_path, session_key=SESSION_A, run_id="run_1", committed_run_ids=()
    )
    assert hook is not None
    assert hook.consume(str(queued["message_id"])) == "consumed"
    assert hook.pending() == ()
    again = bind_session_inbox(
        tmp_path, session_key=SESSION_A, run_id="run_1", committed_run_ids={"run_1"}
    )
    assert again is not None
    assert again.pending() == ()


def test_user_message_trace_projects_and_meta_compact_redacts() -> None:
    events = [
        {
            "event_type": "tool_call_started",
            "tool": "write_file",
            "tool_call_id": "c1",
        },
        {
            "event_type": "user_message",
            "message_id": "m9",
            "interrupt": True,
            "applied_at": "t0",
            "safe_point": INBOX_SAFE_AFTER_TOOLS_BEFORE_LLM,
            "content": "keep /mnt/agent/workspace/main.py drop /Data2/lzp/secret",
        },
        {"event_type": "llm_call", "content": "ack"},
    ]
    blocks = project_trace_blocks(events)
    assert [block["kind"] for block in blocks] == ["tool_group", "user", "agent_output"]
    assert "keep /mnt/agent/workspace/main.py" in str(blocks[1]["text"])
    compact = compact_agent_trace(events)
    assert compact[0]["event_type"] == "user_message"
    assert compact[0]["interrupt"] is True
    assert compact[0]["safe_point"] == INBOX_SAFE_AFTER_TOOLS_BEFORE_LLM
    assert "/Data2/" not in str(compact)
    assert "/mnt/agent/workspace/main.py" in str(compact[0]["content"])
    assert "system_prompt" not in str(compact)


def test_fold_and_meta_backends_bind_inbox_hooks() -> None:
    source = Path("src/autotrade/pipelines/local_backend.py").read_text(
        encoding="utf-8"
    )
    assert source.count("inbox=bind_session_inbox(") == 2
    assert "session_key=request.session_key" in source
    assert 'session_key=str(facts.get("session_key") or "")' in source


def test_completed_interactive_session_expires_leftover(tmp_path: Path) -> None:
    hitl = tmp_path / "hitl"
    hitl.mkdir()
    control = hitl / "control.json"
    status = hitl / "status.json"
    write_control(control, ControlState(mode="auto"))
    ledger = ExperimentLedger(tmp_path / "ledgers" / "experiment_ledger.jsonl")
    path = inbox_path(tmp_path)
    enqueue_inbox_message(path, session_key="epoch_001/fold_a", text="会话结束前未消费")
    enqueue_inbox_message(path, session_key="epoch_001/fold_b", text="下一会话")
    executor = RecordingExecutor(ledger)
    InteractiveExperimentRunner(
        experiment_id="exp",
        sessions=sessions_for("fold_a"),
        execute_session=executor,
        ledger=ledger,
        control_path=control,
        status_path=status,
        poll_seconds=0.01,
    ).run()
    assert list_unconsumed_messages(path, "epoch_001/fold_a") == ()
    assert [item.text for item in list_unconsumed_messages(path, "epoch_001/fold_b")] == [
        "下一会话"
    ]
