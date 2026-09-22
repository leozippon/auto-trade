from __future__ import annotations

import json
from pathlib import Path

import pytest

from autotrade.agent import (
    AgentSessionConfig,
    AgentSessionRunner,
    ContextCompactionConfig,
    ContextCompactor,
    SubAgentEngine,
)
from autotrade.agent import compact as compact_module
from autotrade.agent.compact import (
    fit_tool_results_to_context,
    summarize_tool_result_for_context,
)
from autotrade.agent.experiment_facts import build_experiment_facts
from autotrade.agent.prompts import (
    SESSION_STATIC_SECTIONS,
    build_system_prompt,
)
from autotrade.environment.artifacts import new_revision_id
from autotrade.environment.identity import AgentRefStore
from autotrade.environment.llm import (
    CONTEXT_OUTPUT_MIN_TOKENS,
    CONTEXT_OUTPUT_TOKEN_MARGIN,
    LOCAL_QWEN_MODEL,
    ChatMessage,
    LLMProxyError,
    MalformedToolCallError,
    ProviderResponse,
    ScriptedLLM,
    ToolCall,
    clamp_requested_max_tokens,
    context_request_fits,
    estimate_chat_request_tokens,
    is_context_overflow_error,
)
from autotrade.environment.step_tree import StepTree
from autotrade.environment.time_budget import InferenceTimeBudget
from autotrade.environment.tools import (
    CompactTool,
    FinishSessionTool,
    ReadFileTool,
    SafeWorkspace,
    SearchRoots,
    ToolRegistry,
    ToolResult,
    ToolSpec,
    WriteFileTool,
)
from autotrade.pipelines.local_backend import SessionBudgetLLM, SessionCallBudget

def _passing_gate(node_id: str) -> dict[str, object]:
    """A freeze gate every node passes: these tests are about the Runner."""
    return {"passed": True, "reasons": []}



def finish_session_tool(root: Path) -> tuple[FinishSessionTool, str]:
    """A terminal tool over a real step tree carrying one validated node.

    ``finish_session`` is the Fold session's terminal tool; these are generic
    session-runner tests, so what matters is that a terminal tool ends the
    session and cancels later mutating calls in the same turn."""
    output = root / "output"
    output.mkdir(parents=True, exist_ok=True)
    (output / "main.py").write_text(
        "def generate_orders(context):\n    return []\n", encoding="utf-8"
    )
    tree = StepTree(root / "steps")
    node_id = tree.record_step(
        output,
        epoch_id="epoch_001",
        session_ref="session_ref_ab",
        run_id="run_x",
        result_name="valid_000",
        revision_id=new_revision_id("revision"),
        metrics={"total_return": 0.01},
    )
    return FinishSessionTool(tree, session_ref="session_ref_ab", freeze_gate=_passing_gate), node_id


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

    def __init__(self):
        self.calls = []

    def invoke(self, arguments):
        self.calls.append(arguments)
        return ToolResult(True, value={"stdout": "ran"})


class LongResultShell(DeclaredReadOnlyShell):
    def invoke(self, arguments):
        self.calls.append(arguments)
        return ToolResult(
            True,
            value={
                "stdout": "row\n" * 10_000,
                "stderr": "",
                "timed_out": False,
                "command_kind": "read",
            },
        )


def test_compactor_replaces_old_messages_and_keeps_recent_tool_turns():
    llm = ScriptedLLM(
        [
            ProviderResponse(
                content="## 目标\ncontinue\n\n## 下一步\n- finish"
            )
        ]
    )
    compactor = ContextCompactor(
        llm,
        ContextCompactionConfig(
            token_threshold=1, min_messages=5, keep_recent_messages=2
        ),
    )
    messages = [ChatMessage("system", "system")]
    messages.extend(ChatMessage("user", f"message {index}") for index in range(6))
    result = compactor.compact(messages)
    assert result is not None
    assert result.event["status"] == "ok"
    assert (
        json.loads(result.messages[1].content or "{}")["observation"]
        == "context_compaction"
    )
    assert len(result.messages) == 4
    assert result.messages[0] is messages[0]
    # A conversation that does not start with its system prompt would have the
    # session contract summarized away; compaction refuses it instead.
    with pytest.raises(ValueError, match="system prompt"):
        compactor.compact(messages[1:])


def test_compactor_bounds_one_huge_recent_tool_result_before_local_request():
    llm = ScriptedLLM(
        [ProviderResponse(content="## 目标\ncontinue\n\n## 下一步\n- finish")],
        context_window_tokens=3_000,
    )
    compactor = ContextCompactor(
        llm,
        ContextCompactionConfig(
            token_threshold=1,
            min_messages=5,
            keep_recent_messages=2,
            max_response_tokens=500,
        ),
    )
    messages = [
        ChatMessage("system", "system"),
        ChatMessage("user", "inspect"),
        ChatMessage(
            "assistant",
            tool_calls=(ToolCall("shell-1", "shell", {"argv": ["rg", "x"]}),),
        ),
        ChatMessage("tool", "row\n" * 10_000, tool_call_id="shell-1"),
        ChatMessage("user", "continue"),
    ]

    result = compactor.compact(messages)

    assert result is not None and result.event["status"] == "ok"
    assert result.event["request_context_edit"]["summarized_tool_results"] == 1
    request = llm.calls[0]["messages"]
    body = request[1].content or ""
    # The compactor request is Markdown; the transcript rides as a JSON block.
    recent = json.loads(body.split("## 此后的新消息（JSON 记录）\n", 1)[1])
    summarized_record = next(record for record in recent if record["role"] == "tool")
    tool_summary = json.loads(summarized_record["content"])
    assert tool_summary["observation"] == "context_tool_result_summary"
    assert tool_summary["source_omitted"] is True
    assert "Re-run a narrower paginated query" in tool_summary["note"]
    assert set(tool_summary) == {
        "observation",
        "note",
        "original_chars",
        "source_omitted",
        "head",
        "tail",
    }
    fits, _, _ = context_request_fits(llm, request, max_tokens=500)
    assert fits is True


def _spill_roots(tmp_path: Path) -> SearchRoots:
    """A real spill store over a temporary workspace.

    Without a sandbox layout the search roots spill under the workspace, so a
    spilled reference is ``root='workspace'`` plus a relative path — the same
    ``root`` + ``path`` pair a session reads back with ``read_file``."""

    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    return SearchRoots(SafeWorkspace(workspace))


def _shell_turns(count: int, *, prefix: str = "s") -> list[ProviderResponse]:
    return [
        ProviderResponse(tool_calls=(ToolCall(f"{prefix}{index}", "shell", {"argv": ["rg", str(index)]}),))
        for index in range(count)
    ]


SUMMARY = (
    "## 策略现状\noutput/ 是动量腿 v2（20 日收益排序、每周复核、15 只等权），候选 candidates/v3 加了 60 日"
    "波动过滤，待冒烟。\n## 决定\n- 否定 60 日反转：Y2 中性化超额 -1.2%，空对照分位 0.48，"
    "节点 research__session_ref_ab__run_ref_x__valid_002。\n- 保留 v2 作为当轮基准：完整研究期中性化超额 +2.1%，"
    "四个研究年里三个为正。\n## 线索\n- 下一轮预登记 v3 与对照 c1（去掉波动过滤的同一载体），span=full。\n"
    "## trace\n- 第 3 次调用的 shell 输出有分位数表；grep 'ic=' 可找回筛选读数。"
)


def _compact_call(summary: str = SUMMARY) -> ProviderResponse:
    return ProviderResponse(tool_calls=(ToolCall("c", "compact", {"summary": summary}),))


def _finish_call(node_id: str) -> ProviderResponse:
    return ProviderResponse(
        tool_calls=(ToolCall("f", "finish_session", {"outcome": "freeze", "node_id": node_id}),)
    )


def _compaction_payloads(messages) -> list[dict]:
    return [
        json.loads(message.content)
        for message in messages
        if message.role == "user" and "context_compaction" in (message.content or "")
    ]


def test_the_agent_compacts_its_own_context_and_is_pointed_at_the_trace(tmp_path: Path):
    """``compact`` rebuilds the conversation around the Agent's own summary:
    system prompt, one summary message naming the transcript, and the recent
    tail with its protocol pairs; the event locates the replaced calls."""

    finish, node_id = finish_session_tool(tmp_path)
    llm = ScriptedLLM([*_shell_turns(3), _compact_call(), *_shell_turns(1, prefix="t"), _finish_call(node_id)])
    events: list[tuple[str, dict[str, object]]] = []
    time_budget = InferenceTimeBudget(duration_seconds=120.0)
    shared = SessionCallBudget(max_calls=40, time_budget=time_budget)
    runner = AgentSessionRunner(
        llm=SessionBudgetLLM(llm, budget=shared, role="main"),
        tools=ToolRegistry([DeclaredReadOnlyShell(), CompactTool(), finish]),
        system_prompt="inspect",
        config=AgentSessionConfig(max_llm_calls=8),
        compactor=ContextCompactor(
            SessionBudgetLLM(ScriptedLLM([]), budget=shared, role="compact"),
            ContextCompactionConfig(token_threshold=10**9, keep_recent_messages=2),
        ),
        time_budget=time_budget,
        event_sink=lambda event, payload: events.append((event, payload)),
        trace_ref="run_ref_x",
    )

    assert runner.run("inspect then finish").status == "finished"

    # The request after the compaction: system prompt, the Agent's summary,
    # then the last two messages (the compact turn and its result).
    after = llm.calls[4]["messages"]
    assert after[0].role == "system" and after[0].content == "inspect"
    [payload] = _compaction_payloads(after)
    assert payload["summary_kind"] == "agent" and payload["summary"] == SUMMARY
    assert payload["trace"] == {
        "root": "trace",
        "path": "run_ref_x.txt",
        "hint": payload["trace"]["hint"],
    }
    assert "grep or read_file" in payload["trace"]["hint"]
    assert [message.role for message in after[2:]] == ["assistant", "tool"]
    assert after[2].tool_calls[0].name == "compact" and after[3].tool_call_id == "c"
    assert len(after) == 4
    [compaction] = [payload for event, payload in events if event == "context_compaction"]
    assert compaction["trigger"] == "agent" and compaction["status"] == "ok"
    assert compaction["call_index"] == 4 and compaction["replaced_call_range"] == [1, 4]
    # System prompt, instruction, the wrap-up prompt (the test budget sits
    # inside the default grace), three shell turns and the compact turn.
    assert (compaction["messages_before"], compaction["messages_after"]) == (11, 4)
    assert compaction["summary"] == SUMMARY
    # The tool result itself is a plain acknowledgement.
    tool_events = [payload for event, payload in events if event == "tool_call" and payload["tool"] == "compact"]
    assert tool_events[0]["result"]["value"] == {"status": "compacted", "summary_chars": len(SUMMARY)}
    assert not any(event == "context_notice" for event, _payload in events)


def test_a_resumed_attempt_opens_with_its_preamble_and_seeded_candidates(tmp_path: Path):
    """The runner accepts the interrupted attempt's summary as a preamble and
    the Validations it recorded as finalization candidates."""

    from autotrade.agent.compact import compaction_summary_message

    finish, node_id = finish_session_tool(tmp_path)
    llm = ScriptedLLM([_finish_call(node_id)])
    events: list[tuple[str, dict[str, object]]] = []
    runner = AgentSessionRunner(
        llm=llm,
        tools=ToolRegistry([finish]),
        system_prompt="inspect",
        config=AgentSessionConfig(max_llm_calls=4),
        event_sink=lambda event, payload: events.append((event, payload)),
        freeze_gate=_passing_gate,
    )
    checkpoint = compaction_summary_message(SUMMARY, kind="resume", trace_ref="run_ref_first")
    result = runner.run(
        "continue from the note",
        preamble=[checkpoint],
        complete_validations=[{"node_id": node_id, "revision_id": "strategy_ref_1", "stats": {"sharpe": 1.2}}],
    )
    assert result.status == "finished"
    [system, summary, note] = llm.calls[0]["messages"]
    assert (system.role, summary.role, note.role) == ("system", "user", "user")
    assert json.loads(summary.content)["summary_kind"] == "resume"
    assert note.content == "continue from the note"
    start = next(payload for event, payload in events if event == "session_start")
    assert start["preamble"] == [checkpoint.content]
    assert [candidate["node_id"] for candidate in runner._finalization_candidates()] == [node_id]


def test_a_short_summary_is_refused_and_nothing_is_compacted(tmp_path: Path):
    finish, node_id = finish_session_tool(tmp_path)
    llm = ScriptedLLM([*_shell_turns(2), _compact_call("too short"), _finish_call(node_id)])
    events: list[tuple[str, dict[str, object]]] = []
    runner = AgentSessionRunner(
        llm=llm,
        tools=ToolRegistry([DeclaredReadOnlyShell(), CompactTool(), finish]),
        system_prompt="inspect",
        config=AgentSessionConfig(max_llm_calls=8),
        event_sink=lambda event, payload: events.append((event, payload)),
    )
    assert runner.run("inspect then finish").status == "finished"
    refused = next(payload for event, payload in events if event == "tool_call" and payload["tool"] == "compact")
    assert refused["result"]["ok"] is False and "summary" in refused["result"]["error"]
    assert not any(event == "context_compaction" for event, _payload in events)
    # Every message is still there: the history was not rebuilt.
    assert len(llm.calls[-1]["messages"]) == 2 + 3 * 2


def test_the_context_notice_arrives_at_the_advisory_fraction_and_rearms_after_a_compaction(
    tmp_path: Path,
):
    finish, node_id = finish_session_tool(tmp_path)
    llm = ScriptedLLM(
        [*_shell_turns(3), _compact_call(), *_shell_turns(3, prefix="t"), _finish_call(node_id)]
    )
    events: list[tuple[str, dict[str, object]]] = []
    time_budget = InferenceTimeBudget(duration_seconds=120.0)
    shared = SessionCallBudget(max_calls=40, time_budget=time_budget)
    # A threshold the runtime compactor never reaches (its message floor is
    # high) while three shell turns cross 75% of it.
    runner = AgentSessionRunner(
        llm=SessionBudgetLLM(llm, budget=shared, role="main"),
        tools=ToolRegistry([DeclaredReadOnlyShell(), CompactTool(), finish]),
        system_prompt="inspect",
        config=AgentSessionConfig(max_llm_calls=12),
        compactor=ContextCompactor(
            SessionBudgetLLM(ScriptedLLM([]), budget=shared, role="compact"),
            ContextCompactionConfig(token_threshold=260, min_messages=100, keep_recent_messages=2),
        ),
        time_budget=time_budget,
        event_sink=lambda event, payload: events.append((event, payload)),
    )

    assert runner.run("inspect then finish").status == "finished"

    notices = [index for index, (event, _payload) in enumerate(events) if event == "context_notice"]
    compactions = [index for index, (event, _payload) in enumerate(events) if event == "context_compaction"]
    # One notice before the Agent compacted, one after the window filled again.
    assert len(notices) == 2 and len(compactions) == 1
    assert notices[0] < compactions[0] < notices[1]
    notice = events[notices[0]][1]
    assert notice["token_threshold"] == 260 and notice["estimated_tokens"] >= 195
    observation = next(
        json.loads(message.content)
        for call in llm.calls
        for message in call["messages"]
        if message.role == "user" and "context_notice" in (message.content or "")
    )
    assert observation["observation"] == "context_notice" and "compact(summary=...)" in observation["message"]


def test_a_runtime_compaction_updates_the_agents_own_summary(tmp_path: Path):
    """The safety net treats the Agent's summary as the previous summary it
    updates, so the two triggers keep one continuous checkpoint."""

    finish, node_id = finish_session_tool(tmp_path)
    compact_llm = ScriptedLLM([ProviderResponse(content="## 目标\nupdated\n\n## 下一步\n- finish")])
    llm = ScriptedLLM([*_shell_turns(2), _compact_call(), *_shell_turns(3, prefix="t"), _finish_call(node_id)])
    time_budget = InferenceTimeBudget(duration_seconds=120.0)
    shared = SessionCallBudget(max_calls=40, time_budget=time_budget)
    events: list[tuple[str, dict[str, object]]] = []
    runner = AgentSessionRunner(
        llm=SessionBudgetLLM(llm, budget=shared, role="main"),
        tools=ToolRegistry([DeclaredReadOnlyShell(), CompactTool(), finish]),
        system_prompt="inspect",
        config=AgentSessionConfig(max_llm_calls=12),
        compactor=ContextCompactor(
            SessionBudgetLLM(compact_llm, budget=shared, role="compact"),
            # The floor sits above the history before the Agent's own
            # compaction, so the safety net fires only on the window after it.
            ContextCompactionConfig(
                token_threshold=1, min_messages=10, keep_recent_messages=2, min_remaining_seconds=0
            ),
            trace_ref="run_ref_x",
        ),
        time_budget=time_budget,
        event_sink=lambda event, payload: events.append((event, payload)),
        trace_ref="run_ref_x",
    )

    assert runner.run("inspect then finish").status == "finished"

    triggers = [payload["trigger"] for event, payload in events if event == "context_compaction"]
    assert triggers == ["agent", "runtime"]
    request = compact_llm.calls[0]["messages"][1].content
    assert "## 上一份摘要" in request and SUMMARY.splitlines()[0] in request
    runtime = [payload for event, payload in events if event == "context_compaction"][1]
    assert runtime["replaced_call_range"] == [4, runtime["call_index"]]
    [payload] = _compaction_payloads(llm.calls[-1]["messages"])
    assert payload["summary_kind"] == "model" and payload["summary"].startswith("## 目标\nupdated")
    assert payload["trace"]["path"] == "run_ref_x.txt"


def test_a_read_of_the_trace_root_is_traced_as_a_stub(tmp_path: Path):
    finish, node_id = finish_session_tool(tmp_path)
    transcripts = tmp_path / "transcripts"
    transcripts.mkdir()
    (transcripts / "run_ref_x.txt").write_text("=== event\ncontent: secret numbers\n", encoding="utf-8")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "note.txt").write_text("plain note\n", encoding="utf-8")
    roots = SearchRoots(SafeWorkspace(workspace), trace_root=transcripts)
    assert "trace" in roots.names
    llm = ScriptedLLM(
        [
            ProviderResponse(
                tool_calls=(
                    ToolCall("r1", "read_file", {"root": "trace", "path": "run_ref_x.txt"}),
                    ToolCall("r2", "read_file", {"root": "workspace", "path": "note.txt"}),
                )
            ),
            _finish_call(node_id),
        ]
    )
    events: list[tuple[str, dict[str, object]]] = []
    runner = AgentSessionRunner(
        llm=llm,
        tools=ToolRegistry([ReadFileTool(roots), finish]),
        system_prompt="inspect",
        config=AgentSessionConfig(max_llm_calls=4),
        event_sink=lambda event, payload: events.append((event, payload)),
    )
    assert runner.run("read then finish").status == "finished"
    traced = {payload["tool_call_id"]: payload["result"] for event, payload in events if event == "tool_call"}
    assert "content" not in traced["r1"]["value"]
    assert traced["r1"]["value"]["trace_read"]["content_chars"] > 0
    assert traced["r1"]["value"]["line_count"] == 2
    assert "secret numbers" not in json.dumps(traced["r1"])
    assert "plain note" in traced["r2"]["value"]["content"]
    # The conversation itself still carries what was read.
    tool_messages = [message for message in llm.calls[1]["messages"] if message.role == "tool"]
    assert any("secret numbers" in (message.content or "") for message in tool_messages)


def test_emergency_tool_result_summary_spills_the_body_it_removes(tmp_path: Path):
    roots = _spill_roots(tmp_path)
    body = json.dumps({"status": "ok", "rows": ["row"] * 4_000})
    messages = [
        ChatMessage("system", "system"),
        ChatMessage(
            "assistant",
            tool_calls=(ToolCall("shell-1", "shell", {"argv": ["rg", "x"]}),),
        ),
        ChatMessage("tool", body, tool_call_id="shell-1"),
    ]

    edited, edit = fit_tool_results_to_context(
        ScriptedLLM([], context_window_tokens=3_000),
        messages,
        max_tokens=500,
        result_store=roots,
    )

    assert edit["summarized_tool_results"] == 1
    summary = json.loads(edited[2].content or "{}")
    assert summary["source_omitted"] is True
    assert "read_file" in summary["result_hint"]
    result = ReadFileTool(roots).invoke(
        {"root": summary["result_root"], "path": summary["result_ref"], "limit": 5_000}
    )
    assert result.ok
    assert body[:200] in str(result.value["content"])
    # Without a store the replacement stays purely lossy, as before.
    assert "result_ref" not in json.loads(
        summarize_tool_result_for_context(messages[2]).content or "{}"
    )


def test_emergency_summary_spills_nothing_for_a_replacement_it_discards(
    tmp_path: Path,
):
    """A result too small to shrink keeps its body and leaves no orphan file."""

    roots = _spill_roots(tmp_path)
    messages = [
        ChatMessage("system", "system"),
        ChatMessage(
            "assistant",
            tool_calls=(ToolCall("shell-1", "shell", {"argv": ["rg", "x"]}),),
        ),
        ChatMessage("tool", "x" * 600, tool_call_id="shell-1"),
    ]

    edited, edit = fit_tool_results_to_context(
        ScriptedLLM([], context_window_tokens=10),
        messages,
        max_tokens=5,
        force=True,
        result_store=roots,
    )

    assert edit == {} and edited[2].content == "x" * 600
    assert not list((tmp_path / "workspace" / "logs").rglob("tool_result.txt"))


def test_context_token_estimate_includes_reasoning_content():
    without_reasoning = [ChatMessage("assistant", "answer")]
    with_reasoning = [ChatMessage("assistant", "answer", reasoning_content="r" * 400)]

    assert estimate_chat_request_tokens(with_reasoning) > estimate_chat_request_tokens(
        without_reasoning
    )


def test_session_tracks_calls_and_finish_value(tmp_path: Path):
    finish, node_id = finish_session_tool(tmp_path)
    llm = ScriptedLLM(
        [
            ProviderResponse(
                tool_calls=(ToolCall("f", "finish_session", {"outcome": "freeze", "node_id": node_id}),)
            )
        ]
    )
    runner = AgentSessionRunner(
        llm=llm,
        tools=ToolRegistry([finish]),
        system_prompt="daily JSON only",
        config=AgentSessionConfig(max_llm_calls=2),
    )
    result = runner.run("finish the validated strategy")
    assert result.status == "finished"
    assert result.finish_value["node_id"] == node_id
    assert result.finish_value["status"] == "session_finished"
    assert result.finish_value["outcome"] == "freeze"
    assert result.llm_calls == 1


def test_session_end_counts_the_parents_failed_tool_calls(tmp_path: Path):
    """A summary-only audit reads ``session_end``; a failed tool call is
    counted there, and the finish payload rides along."""
    finish, node_id = finish_session_tool(tmp_path)
    events: list[tuple[str, dict]] = []
    llm = ScriptedLLM(
        [
            ProviderResponse(
                tool_calls=(ToolCall("bad", "finish_session", {"outcome": "freeze", "node_id": "missing"}),)
            ),
            ProviderResponse(
                tool_calls=(
                    ToolCall(
                        "f",
                        "finish_session",
                        {
                            "outcome": "freeze",
                            "node_id": node_id,
                            "reason": "H2 untestable here: the events domain is empty in this window",
                        },
                    ),
                )
            ),
        ]
    )
    runner = AgentSessionRunner(
        llm=llm,
        tools=ToolRegistry([finish]),
        system_prompt="daily JSON only",
        config=AgentSessionConfig(max_llm_calls=3),
        event_sink=lambda event, payload: events.append((event, payload)),
    )
    assert runner.run("finish").status == "finished"
    end = [payload for event, payload in events if event == "session_end"][-1]
    assert end["status"] == "finished" and end["tool_failures"] == 1
    assert end["finish"]["reason"] == "H2 untestable here: the events domain is empty in this window"


def test_session_end_tool_failures_include_the_childrens(tmp_path: Path):
    """``session_end`` must agree with the trace it summarizes.

    Almost every tool call of a long Fold is made by a sub-agent, so counting
    only the parent's made the scalar report a fraction of the failures the
    Agent Trace summary counts from the same events, and a Meta session reading
    the two side by side saw them contradict each other.
    """

    workspace = tmp_path / "agent"
    (workspace / "output").mkdir(parents=True)
    (workspace / "output" / "README.md").write_text("contract\n", encoding="utf-8")
    finish, node_id = finish_session_tool(tmp_path)
    events: list[tuple[str, dict]] = []
    subagent = SubAgentEngine(
        llm=ScriptedLLM(
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
                ProviderResponse(content="写入被拒绝，README 是只读合同。"),
            ]
        ),
        tools=ToolRegistry([WriteFileTool(SafeWorkspace(workspace))]),
    )
    runner = AgentSessionRunner(
        llm=ScriptedLLM(
            [
                ProviderResponse(
                    tool_calls=(
                        ToolCall(
                            "e1",
                            "agent",
                            {"agent": "general-purpose", "task": "rewrite the contract"},
                        ),
                    )
                ),
                ProviderResponse(content="等子代理返回。"),
                ProviderResponse(
                    tool_calls=(ToolCall("bad", "finish_session", {"outcome": "freeze", "node_id": "missing"}),)
                ),
                ProviderResponse(
                    tool_calls=(ToolCall("f", "finish_session", {"outcome": "freeze", "node_id": node_id}),)
                ),
            ]
        ),
        tools=ToolRegistry([finish]),
        system_prompt="daily JSON only",
        config=AgentSessionConfig(max_llm_calls=5),
        subagent=subagent,
        event_sink=lambda event, payload: events.append((event, payload)),
    )
    assert runner.run("delegate, then finish").status == "finished"
    failed = [
        event
        for event, payload in events
        if event in {"tool_call", "subagent_tool"}
        and payload.get("result", {}).get("ok") is False
    ]
    # One refused write by the child, one bad node id by the parent.
    assert sorted(failed) == ["subagent_tool", "tool_call"]
    end = [payload for event, payload in events if event == "session_end"][-1]
    assert end["tool_failures"] == len(failed)


def test_fold_session_nudges_text_only_turn_then_requires_finish(tmp_path: Path):
    finish, node_id = finish_session_tool(tmp_path)
    llm = ScriptedLLM(
        [
            ProviderResponse(content="I should act next."),
            ProviderResponse(
                tool_calls=(ToolCall("f", "finish_session", {"outcome": "freeze", "node_id": node_id}),)
            ),
        ]
    )
    runner = AgentSessionRunner(
        llm=llm,
        tools=ToolRegistry([finish]),
        system_prompt="daily JSON only",
        config=AgentSessionConfig(max_llm_calls=2),
    )
    result = runner.run("finish")
    assert result.status == "finished"
    assert result.llm_calls == 2
    second_messages = llm.calls[1]["messages"]
    assert any("no_tool_call" in (message.content or "") for message in second_messages)


def test_fold_session_replays_reasoning_with_tool_call_on_next_round(tmp_path: Path):
    finish, node_id = finish_session_tool(tmp_path)
    shell = DeclaredReadOnlyShell()
    llm = ScriptedLLM(
        [
            ProviderResponse(
                tool_calls=(ToolCall("s", "shell", {"argv": ["rg", "needle", "."]}),),
                reasoning_content="inspect before finishing",
            ),
            ProviderResponse(
                tool_calls=(ToolCall("f", "finish_session", {"outcome": "freeze", "node_id": node_id}),)
            ),
        ]
    )
    runner = AgentSessionRunner(
        llm=llm,
        tools=ToolRegistry([shell, finish]),
        system_prompt="daily JSON only",
        config=AgentSessionConfig(max_llm_calls=2),
    )

    assert runner.run("inspect and finish").status == "finished"
    assistant = next(
        message for message in llm.calls[1]["messages"] if message.role == "assistant"
    )
    assert assistant.content is None
    assert assistant.reasoning_content == "inspect before finishing"
    assert assistant.to_record()["reasoning_content"] == "inspect before finishing"


def test_fold_session_edits_huge_recent_tool_result_below_min_message_count(
    tmp_path: Path,
):
    finish, node_id = finish_session_tool(tmp_path)
    shell = LongResultShell()
    llm = ScriptedLLM(
        [
            ProviderResponse(
                tool_calls=(ToolCall("shell-1", "shell", {"argv": ["rg", "x"]}),)
            ),
            ProviderResponse(
                tool_calls=(ToolCall("finish-1", "finish_session", {"outcome": "freeze", "node_id": node_id}),)
            ),
        ],
        context_window_tokens=3_000,
    )
    events: list[tuple[str, dict[str, object]]] = []
    runner = AgentSessionRunner(
        llm=llm,
        tools=ToolRegistry([shell, finish]),
        system_prompt="inspect and finish",
        config=AgentSessionConfig(max_llm_calls=3, max_response_tokens=500),
        event_sink=lambda event, payload: events.append((event, payload)),
    )

    assert runner.run("inspect once").status == "finished"

    assert len(llm.calls) == 2
    second_messages = llm.calls[1]["messages"]
    tool_message = next(message for message in second_messages if message.role == "tool")
    assistant = next(
        message for message in second_messages if message.role == "assistant"
    )
    assert tool_message.tool_call_id == "shell-1"
    assert assistant.tool_calls[0].id == tool_message.tool_call_id
    summary = json.loads(tool_message.content or "{}")
    assert summary["observation"] == "context_tool_result_summary"
    assert "Re-run a narrower paginated query" in summary["note"]
    assert summary["original_chars"] > 40_000
    assert summary["source_omitted"] is True
    assert "sha256" not in summary
    assert {"head", "tail", "retained_fields"} <= summary.keys()
    assert summary["retained_fields"] == {
        "value.timed_out": False,
        "value.command_kind": "read",
    }
    assert set(summary) == {
        "observation",
        "note",
        "original_chars",
        "source_omitted",
        "head",
        "tail",
        "retained_fields",
    }
    assert any(
        event == "context_edit" and payload["summarized_tool_results"] == 1
        for event, payload in events
    )


def test_fold_session_recovers_one_provider_context_overflow_without_blind_repeat(
    tmp_path: Path,
):
    finish, _node_id = finish_session_tool(tmp_path)
    shell = LongResultShell()

    class AlwaysOverflowAfterTool:
        provider = "vllm"
        model = LOCAL_QWEN_MODEL
        context_window_tokens = None

        def __init__(self):
            self.calls = []

        def complete(self, messages, **kwargs):
            self.calls.append(tuple(messages))
            if len(self.calls) == 1:
                return ProviderResponse(
                    tool_calls=(
                        ToolCall("shell-1", "shell", {"argv": ["rg", "x"]}),
                    )
                )
            raise LLMProxyError(
                "provider HTTP error 400: maximum context length exceeded",
                retryable=False,
                status_code=400,
            )

    llm = AlwaysOverflowAfterTool()
    runner = AgentSessionRunner(
        llm=llm,
        tools=ToolRegistry([shell, finish]),
        system_prompt="inspect and finish",
        config=AgentSessionConfig(max_llm_calls=6, max_response_tokens=500),
    )

    with pytest.raises(RuntimeError, match="cannot be reduced safely"):
        runner.run("inspect once")

    assert len(llm.calls) == 3
    raw_tool = next(
        message.content or "" for message in llm.calls[1] if message.role == "tool"
    )
    assert "row\nrow" in json.loads(raw_tool)["value"]["stdout"]
    compacted_tool = next(
        message.content or "" for message in llm.calls[2] if message.role == "tool"
    )
    assert json.loads(compacted_tool)["source_omitted"] is True
    assert next(
        message.tool_call_id for message in llm.calls[2] if message.role == "tool"
    ) == "shell-1"


def test_fold_session_reissues_one_malformed_tool_call_without_repeating_the_analysis(
    tmp_path: Path,
):
    """A tool call the model wrote unparseably must not cost a generation.

    The whole reply is rejected by the gateway, so the session keeps the text
    that did arrive as its own assistant turn and asks only for the call
    again. A second failure in the same streak drops back to the generic
    llm_error handling instead of replaying the analysis a second time."""

    finish, node_id = finish_session_tool(tmp_path)

    class MalformedThenFinish:
        provider = "vllm"
        model = LOCAL_QWEN_MODEL
        context_window_tokens = None

        def __init__(self):
            self.calls = []

        def complete(self, messages, **kwargs):
            del kwargs
            self.calls.append(tuple(messages))
            if len(self.calls) <= 2:
                raise MalformedToolCallError(
                    "provider returned a malformed tool call (tool=finish_session: "
                    "Expecting value: line 1 column 9 (char 8)); no call from "
                    "this response was executed",
                    content="换手率已压到 12%，可以收官。",
                    reasoning_content="先复核 Validation",
                )
            return ProviderResponse(
                tool_calls=(ToolCall("f", "finish_session", {"outcome": "freeze", "node_id": node_id}),)
            )

    llm = MalformedThenFinish()
    events: list[tuple[str, dict[str, object]]] = []
    runner = AgentSessionRunner(
        llm=llm,
        tools=ToolRegistry([finish]),
        system_prompt="daily JSON only",
        config=AgentSessionConfig(max_llm_calls=6),
        event_sink=lambda event, payload: events.append((event, payload)),
    )

    result = runner.run("finish the validated strategy")

    assert result.status == "finished"
    assert len(llm.calls) == 3
    second = llm.calls[1]
    assert second[-2].role == "assistant"
    assert second[-2].content == "换手率已压到 12%，可以收官。"
    assert second[-2].reasoning_content == "先复核 Validation"
    assert second[-2].tool_calls == ()
    observation = json.loads(second[-1].content or "{}")
    assert observation["observation"] == "malformed_tool_call"
    assert "tool=finish_session" in observation["error"]
    third = llm.calls[2]
    assert json.loads(third[-1].content or "{}")["observation"] == "llm_error"
    assert sum(1 for message in third if message.role == "assistant") == 1
    # Countable in an audit, and separated from transport failures.
    assert [
        payload.get("error_type")
        for event, payload in events
        if event == "llm_call" and payload.get("status") == "error"
    ] == ["malformed_tool_call", "malformed_tool_call"]


def test_fold_session_keeps_long_history_without_proactive_clear_or_trim(
    tmp_path: Path,
):
    finish, node_id = finish_session_tool(tmp_path)

    class MarkedResultShell(DeclaredReadOnlyShell):
        def invoke(self, arguments):
            self.calls.append(arguments)
            marker = str((arguments.get("argv") or ["?"])[-1])
            return ToolResult(
                True,
                value={"stdout": f"{marker}:" + ("row\n" * 2_000), "marker": marker},
            )

    rounds = 12
    responses = [
        ProviderResponse(
            tool_calls=(ToolCall(f"s{index}", "shell", {"argv": ["echo", str(index)]}),)
        )
        for index in range(rounds)
    ]
    responses.append(
        ProviderResponse(
            tool_calls=(ToolCall("f", "finish_session", {"outcome": "freeze", "node_id": node_id}),)
        )
    )
    llm = ScriptedLLM(responses)
    events: list[tuple[str, dict[str, object]]] = []
    runner = AgentSessionRunner(
        llm=llm,
        tools=ToolRegistry([MarkedResultShell(), finish]),
        system_prompt="inspect many times",
        config=AgentSessionConfig(max_llm_calls=rounds + 2),
        event_sink=lambda event, payload: events.append((event, payload)),
    )

    assert runner.run("inspect repeatedly").status == "finished"
    assert not hasattr(AgentSessionRunner, "_trim")
    assert not hasattr(AgentSessionRunner, "_clear_stale_tool_results")
    assert not any(event == "context_summary" for event, _ in events)
    assert not any(
        event == "context_edit" and "cleared_tool_results" in payload
        for event, payload in events
    )
    final_messages = llm.calls[-1]["messages"]
    assert isinstance(final_messages, tuple)
    tool_payloads = [
        json.loads(message.content or "{}")
        for message in final_messages
        if message.role == "tool"
    ]
    assert len(tool_payloads) == rounds
    assert [payload["value"]["marker"] for payload in tool_payloads] == [
        str(index) for index in range(rounds)
    ]
    assert all("observation" not in payload for payload in tool_payloads)
    assert len(final_messages) == 2 + rounds * 2


def test_fold_session_triggers_semantic_compact_on_threshold(tmp_path: Path):
    finish, node_id = finish_session_tool(tmp_path)
    compact_llm = ScriptedLLM(
        [
            ProviderResponse(
                content="## 目标\ncontinue\n\n## 下一步\n- finish"
            ),
            ProviderResponse(
                content="## 目标\nfinish\n\n## 下一步\n- finish"
            ),
        ]
    )
    llm = ScriptedLLM(
        [
            ProviderResponse(
                tool_calls=(ToolCall("s1", "shell", {"argv": ["rg", "a"]}),)
            ),
            ProviderResponse(
                tool_calls=(ToolCall("s2", "shell", {"argv": ["rg", "b"]}),)
            ),
            ProviderResponse(
                tool_calls=(ToolCall("s3", "shell", {"argv": ["rg", "c"]}),)
            ),
            ProviderResponse(
                tool_calls=(ToolCall("f", "finish_session", {"outcome": "freeze", "node_id": node_id}),)
            ),
        ]
    )
    events: list[tuple[str, dict[str, object]]] = []
    time_budget = InferenceTimeBudget(duration_seconds=120.0)
    shared = SessionCallBudget(max_calls=20, time_budget=time_budget)
    runner = AgentSessionRunner(
        llm=SessionBudgetLLM(llm, budget=shared, role="main"),
        tools=ToolRegistry([DeclaredReadOnlyShell(), finish]),
        system_prompt="inspect",
        config=AgentSessionConfig(max_llm_calls=6),
        compactor=ContextCompactor(
            SessionBudgetLLM(compact_llm, budget=shared, role="compact"),
            ContextCompactionConfig(
                token_threshold=1,
                min_messages=6,
                keep_recent_messages=2,
                min_remaining_seconds=0,
            ),
        ),
        time_budget=time_budget,
        event_sink=lambda event, payload: events.append((event, payload)),
    )

    assert runner.run("inspect then finish").status == "finished"
    compact_events = [
        payload for event, payload in events if event == "context_compaction"
    ]
    assert compact_events and compact_events[0]["status"] == "ok"
    assert not any(event == "context_summary" for event, _ in events)
    later = llm.calls[-1]["messages"]
    assert isinstance(later, tuple)
    assert any(
        "context_compaction" in (message.content or "") for message in later
    )
    tool_ids = [
        message.tool_call_id for message in later if message.role == "tool"
    ]
    assistant_ids = [
        call.id
        for message in later
        if message.role == "assistant" and message.tool_calls
        for call in message.tool_calls
    ]
    assert tool_ids
    assert set(tool_ids) <= set(assistant_ids)


def test_compaction_keeps_the_session_system_prompt_byte_identical(tmp_path: Path):
    """The system prompt is composed once and reused verbatim all session.

    Compaction rewrites history; it must not re-render or re-place the system
    prompt, both because the session contract has to stay identical and
    because a byte-stable prefix is what the provider's cache keys on."""
    finish, node_id = finish_session_tool(tmp_path)
    system_prompt = build_system_prompt(
        experiment_facts={"experiment_id": "exp_x", "session_ref": "session_ref_ab"},
        session_directive="check the volume filter",
    )
    compact_llm = ScriptedLLM(
        [ProviderResponse(content="## 目标\ncontinue\n\n## 下一步\n- finish")] * 4
    )
    llm = ScriptedLLM(
        [
            ProviderResponse(
                tool_calls=(ToolCall(f"s{index}", "shell", {"argv": ["rg", "a"]}),)
            )
            for index in range(3)
        ]
        + [
            ProviderResponse(
                tool_calls=(ToolCall("f", "finish_session", {"outcome": "freeze", "node_id": node_id}),)
            )
        ]
    )
    events: list[tuple[str, dict[str, object]]] = []
    time_budget = InferenceTimeBudget(duration_seconds=120.0)
    shared = SessionCallBudget(max_calls=20, time_budget=time_budget)
    runner = AgentSessionRunner(
        llm=SessionBudgetLLM(llm, budget=shared, role="main"),
        tools=ToolRegistry([DeclaredReadOnlyShell(), finish]),
        system_prompt=system_prompt,
        config=AgentSessionConfig(max_llm_calls=6),
        compactor=ContextCompactor(
            SessionBudgetLLM(compact_llm, budget=shared, role="compact"),
            # A floor of eight: with the wrap-up prompt and the compaction
            # advisory in the history, the window fills every other turn, so
            # the request lengths below really rise and fall.
            ContextCompactionConfig(
                token_threshold=1,
                min_messages=8,
                keep_recent_messages=2,
                min_remaining_seconds=0,
            ),
        ),
        time_budget=time_budget,
        event_sink=lambda event, payload: events.append((event, payload)),
    )

    assert runner.run("inspect then finish").status == "finished"

    compactions = [
        payload
        for event, payload in events
        if event == "context_compaction" and payload["status"] == "ok"
    ]
    assert compactions, "the session must cross a compaction to prove the invariant"
    lengths = [len(call["messages"]) for call in llm.calls]
    # History really shrank, so the equality below is not a vacuous check.
    assert any(later < earlier for earlier, later in zip(lengths, lengths[1:]))
    assert any(
        "context_compaction" in (message.content or "")
        for message in llm.calls[-1]["messages"]
    )
    for call in llm.calls:
        head = call["messages"][0]
        assert head.role == "system"
        assert head.content == system_prompt
    assert events[0][0] == "session_start"
    assert events[0][1]["system_prompt"] == system_prompt


def test_disabled_compact_fails_closed_without_dropping_history(tmp_path: Path):
    finish, _node_id = finish_session_tool(tmp_path)
    events: list[tuple[str, dict[str, object]]] = []
    llm = ScriptedLLM([], context_window_tokens=200)
    runner = AgentSessionRunner(
        llm=llm,
        tools=ToolRegistry([finish]),
        system_prompt="s",
        config=AgentSessionConfig(max_llm_calls=2, max_response_tokens=50),
        event_sink=lambda event, payload: events.append((event, payload)),
    )

    with pytest.raises(RuntimeError, match="cannot be reduced safely"):
        runner.run("y" * 5_000)

    assert llm.calls == []
    assert not any(event == "context_summary" for event, _ in events)
    assert not any(
        event == "context_edit" and "cleared_tool_results" in payload
        for event, payload in events
    )
    end = next(payload for event, payload in events if event == "session_end")
    assert end["status"] == "context_window_exceeded"


def test_terminal_tool_cancels_later_mutation_in_same_turn(tmp_path: Path):
    finish, node_id = finish_session_tool(tmp_path)
    workspace = SafeWorkspace(tmp_path)
    llm = ScriptedLLM(
        [
            ProviderResponse(
                tool_calls=(
                    ToolCall("f", "finish_session", {"outcome": "freeze", "node_id": node_id}),
                    ToolCall(
                        "w",
                        "write_file",
                        {"path": "notes.md", "content": "should not persist"},
                    ),
                )
            )
        ]
    )
    runner = AgentSessionRunner(
        llm=llm,
        tools=ToolRegistry([finish, WriteFileTool(workspace)]),
        system_prompt="daily JSON only",
    )
    runner.run("finish")
    assert "generate_orders" in (tmp_path / "output/main.py").read_text(
        encoding="utf-8"
    )
    assert not (tmp_path / "notes.md").exists()


def test_subagent_accepts_write_file_tool(tmp_path: Path):
    engine = SubAgentEngine(
        llm=ScriptedLLM([]),
        tools=ToolRegistry([WriteFileTool(SafeWorkspace(tmp_path))]),
    )
    assert {spec.name for spec in engine.tools.specs()} == {"write_file"}


def test_subagent_dispatches_full_shell_commands():
    shell = DeclaredReadOnlyShell()
    llm = ScriptedLLM(
        [
            ProviderResponse(
                tool_calls=(ToolCall("s", "shell", {"argv": ["python", "-V"]}),),
                reasoning_content="check the runtime",
            ),
            ProviderResponse(
                tool_calls=(ToolCall("r", "shell", {"argv": ["rg", "needle", "."]}),)
            ),
            ProviderResponse(content="ran full shell"),
        ]
    )
    result = SubAgentEngine(
        llm=llm,
        tools=ToolRegistry([shell]),
    ).run("inspect", role="general-purpose")
    assert result["summary"] == "ran full shell"
    assert shell.calls == [
        {"argv": ["python", "-V"]},
        {"argv": ["rg", "needle", "."]},
    ]
    assert result["tool_calls"] == 2
    first_assistant = next(
        message for message in llm.calls[1]["messages"] if message.role == "assistant"
    )
    assert first_assistant.reasoning_content == "check the runtime"


def test_shared_session_llm_budget_counts_every_gateway_call():
    scripted = ScriptedLLM(
        [
            ProviderResponse(content="first"),
            ProviderResponse(content="must remain unused"),
        ]
    )
    budgeted = SessionBudgetLLM(
        scripted, max_calls=1, deadline=__import__("time").monotonic() + 10
    )
    budgeted.complete([ChatMessage("user", "one")])
    with pytest.raises(RuntimeError, match="budget exhausted"):
        budgeted.complete([ChatMessage("user", "two")])
    assert len(scripted.calls) == 1


def test_compaction_failure_keeps_history_and_opens_failure_circuit():
    class BrokenLLM:
        def complete(self, *args, **kwargs):
            raise RuntimeError("provider failed with Authorization: secret")

    compactor = ContextCompactor(
        BrokenLLM(),
        ContextCompactionConfig(
            token_threshold=1,
            min_messages=4,
            keep_recent_messages=2,
            max_failures=1,
        ),
    )
    messages = [ChatMessage("system", "system")]
    messages.extend(ChatMessage("user", f"m{index}") for index in range(5))
    failed = compactor.compact(messages)
    assert failed is not None
    assert failed.messages == tuple(messages)
    assert failed.event["status"] == "error"
    assert "secret" not in str(failed.event["error"])
    assert compactor.compact(messages) is None


def test_an_error_payload_answered_as_a_summary_fails_the_compaction():
    """A compaction replaces the conversation, so what it keeps must be the
    summary that was asked for.

    Observed live: a gateway answered an over-sized compaction request with a
    normal completion whose content was its own error envelope, and it was
    recorded as a successful compaction and became the whole retained context.
    """

    envelope = json.dumps(
        {
            "error": {
                "message": (
                    "Your request is over the maximum allowed context size. "
                    "Please minimize or reduce the prompts used."
                ),
                "type": "invalid_request_error",
            }
        },
        ensure_ascii=False,
    )
    compactor = ContextCompactor(
        ScriptedLLM([ProviderResponse(content=envelope)]),
        ContextCompactionConfig(
            token_threshold=1, min_messages=4, keep_recent_messages=2
        ),
    )
    messages = [ChatMessage("system", "system")]
    messages.extend(ChatMessage("user", f"m{index}") for index in range(5))
    result = compactor.compact(messages)
    assert result is not None
    assert result.event["status"] == "error"
    assert "summary" not in result.event
    # The failure names what came back, so the trace shows the real cause.
    assert "maximum allowed context size" in str(result.event["error"])
    # History is kept: the caller falls back to in-place tool-result fitting.
    assert result.messages == tuple(messages)
    assert compactor.compaction_count == 0


def test_sessions_reject_tools_outside_their_positive_contracts():
    class StubTool:
        def __init__(self, name):
            self.spec = ToolSpec(
                name,
                "must be rejected",
                {"type": "object", "properties": {}, "required": []},
            )

        def invoke(self, arguments):
            return ToolResult(True)

    for name in ("external_lookup", "finish_meta"):
        with pytest.raises(ValueError, match="unsupported tools"):
            AgentSessionRunner(
                llm=ScriptedLLM([]),
                tools=ToolRegistry([StubTool(name)]),
                system_prompt="session",
                config=AgentSessionConfig(),
            )

    with pytest.raises(ValueError, match="batch_validate requires finish_session"):
        AgentSessionRunner(
            llm=ScriptedLLM([]),
            tools=ToolRegistry([StubTool("batch_validate")]),
            system_prompt="session",
            config=AgentSessionConfig(),
        )


def test_prompt_and_facts_encode_daily_json_and_hidden_stage_boundaries(
    tmp_path: Path,
):
    prompt = build_system_prompt(experiment_facts={"identity": {"session_kind": "research"}})
    assert "generate_orders(context)" in prompt
    assert "严格 JSON 往返的订单数组" in prompt
    assert "策略执行时钟" in prompt
    assert "不能假定 `context.bars` 含完整历史" in prompt
    assert "实际挂载清单、schema、单位引用" in prompt
    assert "Held-out" in prompt
    # The prohibition list carries the item-6 execution-model rules.
    prohibitions = prompt[prompt.index("# 禁止事项") :]
    for rule in (
        "读取研究期末之后的任何数据、前推期或 Held-out",
        "绕过 `available_at`",
        "把历史分钟、竞价或事件时间当成策略执行时钟",
        "伪造工具结果",
    ):
        assert rule in prohibitions
    facts = build_experiment_facts(
        manifest={"kind": "research", "experiment_id": "exp"},
        ref_store=AgentRefStore(tmp_path / "experiment"),
        runtime_env={"sandbox_spec": {"network": "none"}},
    )
    assert facts["visibility_policy"]["research_period_visible"] is True
    assert "不进入任何会话" in facts["visibility_policy"]["after_research_end"]


def test_session_prompt_keeps_hard_boundaries_and_leaves_how_tos_mounted():
    prompt = build_system_prompt(experiment_facts={})
    for rule in (
        "已挂载的事实、数据、起点产物与参考材料都是待检验输入",
        "从日期、路径、元数据和模型常识推断它们的行情",
        "正式回测不能由自建回放替代",
        "不得用它修改策略产物、启动后台任务、sleep/等待包装或轮询状态",
    ):
        assert rule in prompt
    assert "pyright --project" not in prompt
    assert "pandas.read_parquet" not in prompt
    assert 'context.asof_dir + "/daily"' not in prompt

    reference = Path("configs/agent_output_template/README.md").read_text(
        encoding="utf-8"
    )
    for rule in (
        "keeps one strategy worker alive",
        "recorded `context.asof_version`",
        "must remain correct from a cold cache",
        "Do not reread the full PIT directory",
        "must not admit a row beyond `context.inference_at`",
    ):
        assert rule in reference


def _export_prompts_module():
    import importlib.util

    repo_root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location(
        "export_prompts", repo_root / "scripts" / "dev" / "export_prompts.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return repo_root, module


def test_prompt_audit_snapshot_is_byte_exact():
    """PROMPTS.md is generated, so the freshness contract is byte equality.

    A substring check over SESSION_STATIC_SECTIONS only sees the fold sections —
    which is exactly why a changed Meta prompt and a relocated NL prompt both
    landed silently. Byte equality additionally catches a reordered section, a
    dropped section, a changed heading and a hand-edit of the snapshot."""
    repo_root, module = _export_prompts_module()
    committed = (repo_root / "configs" / "prompts" / "PROMPTS.md").read_text(
        encoding="utf-8"
    )
    assert committed == module.render(), (
        "configs/prompts/PROMPTS.md is stale; run scripts/dev/export_prompts.py"
    )


def test_prompt_audit_snapshot_check_fails_on_a_hand_edit(tmp_path: Path):
    """The freshness check must be able to fail — the mutation proves it."""
    _repo_root, module = _export_prompts_module()
    rendered = module.render()
    stale = tmp_path / "PROMPTS.md"
    stale.write_text(
        rendered.replace("# Prompt 模板审计快照", "# 手工改过的标题", 1),
        encoding="utf-8",
    )
    assert stale.read_text(encoding="utf-8") != rendered
    # And every fold section still rides in the snapshot as a fenced block.
    for section in SESSION_STATIC_SECTIONS:
        assert f"```text\n{section.strip()}\n```" in rendered


def _prompt_tool_tokens(text: str) -> set[str]:
    """Backticked lowercase identifiers a prompt presents as tool names."""
    import re

    return set(re.findall(r"`([a-z][a-z0-9_]*)`", text))


def _all_registrable_tool_names() -> set[str]:
    """Every tool name any session in this tree can actually register."""
    import tempfile

    import autotrade.environment.tools as tools_pkg
    from autotrade.agent.subagent import AgentTool
    from autotrade.environment.nl.engine import TEXT_RETRIEVE_TOOL
    from autotrade.environment.tools import SafeWorkspace, SearchRoots
    from autotrade.environment.tools.search import GlobTool, GrepTool, ReadFileTool
    from autotrade.pipelines.session_tools import (
        BatchValidateTool,
        NullControlTool,
        SmokeBacktestTool,
    )

    # The backtest tools are constructed per fold in local_backend rather than
    # exported from the tools package, so read their names off the classes: a
    # rename must not silently drop one from this set.
    names = {
        TEXT_RETRIEVE_TOOL,
        BatchValidateTool.spec.name,
        NullControlTool.spec.name,
        SmokeBacktestTool.spec.name,
    }
    with tempfile.TemporaryDirectory() as tmp:
        roots = SearchRoots(SafeWorkspace(Path(tmp)))
        instances = [GlobTool(roots), GrepTool(roots), ReadFileTool(roots)]
    candidates = [*instances, AgentTool]
    candidates.extend(getattr(tools_pkg, name, None) for name in dir(tools_pkg))
    for candidate in candidates:
        spec = getattr(candidate, "spec", None)
        if spec is not None and getattr(spec, "name", None):
            names.add(spec.name)
    return names


def test_every_tool_named_in_a_prompt_is_registrable_in_that_session():
    """A prompt that names a tool the session cannot register is a live lie.

    This is the class of defect that left the Fold prompt pointing at
    `nl_query` / `finish` after the authoring stack was deleted: the prompt and
    the registry drifted apart with nothing comparing them."""
    from autotrade.agent.runner import _SESSION_TOOLS
    from autotrade.environment.nl.engine import (
        SUB_AGENT_SYSTEM_PROMPT,
        TEXT_RETRIEVE_TOOL,
    )

    registrable = _all_registrable_tool_names()
    # Every allowlisted name must correspond to a tool that exists.
    assert _SESSION_TOOLS <= registrable

    sessions = (
        ("fold", build_system_prompt(experiment_facts={}), _SESSION_TOOLS),
        ("nl_sub_agent", SUB_AGENT_SYSTEM_PROMPT, {TEXT_RETRIEVE_TOOL}),
    )
    for name, prompt, allowed in sessions:
        referenced = _prompt_tool_tokens(prompt) & registrable
        assert referenced, f"{name} prompt names no tool at all"
        unresolved = sorted(referenced - set(allowed))
        assert unresolved == [], (
            f"{name} prompt names tools it cannot register: {unresolved}"
        )


def test_the_prompt_tool_check_fails_on_a_tool_the_session_cannot_register():
    """The mutation proves the check above can fail."""
    from autotrade.agent.runner import _SESSION_TOOLS

    registrable = _all_registrable_tool_names()
    # `text_retrieve` exists, but only the NL sub-agent may register it.
    assert "text_retrieve" in registrable
    mutated = (
        build_system_prompt(experiment_facts={})
        + "\n- 用 `text_retrieve` 检索证据。"
    )
    referenced = _prompt_tool_tokens(mutated) & registrable
    assert sorted(referenced - set(_SESSION_TOOLS)) == ["text_retrieve"]


def _context_runner(llm: ScriptedLLM, **config) -> AgentSessionRunner:
    return AgentSessionRunner(
        llm=llm,
        tools=ToolRegistry([]),
        system_prompt="test",
        config=AgentSessionConfig(**config),
    )


def test_prepare_context_request_returns_messages_that_fit():
    runner = _context_runner(ScriptedLLM([], context_window_tokens=100_000))
    messages = [ChatMessage("user", "short")]
    assert (
        runner._prepare_context_request(messages, (), 30.0) is messages
    )


def test_prepare_context_request_passes_when_window_is_unknown():
    runner = _context_runner(ScriptedLLM([], context_window_tokens=None))
    messages = [ChatMessage("user", "x" * 8_000)]
    assert (
        runner._prepare_context_request(messages, (), 30.0) is messages
    )


def test_prepare_context_request_raises_context_overflow_error():
    runner = _context_runner(
        ScriptedLLM([], context_window_tokens=200),
        max_response_tokens=50,
    )
    messages = [ChatMessage("user", "x" * 5_000)]
    with pytest.raises(LLMProxyError) as excinfo:
        runner._prepare_context_request(
            messages, (), 30.0, allow_semantic_compaction=False
        )
    assert is_context_overflow_error(excinfo.value)
    assert "context window 200" in str(excinfo.value)


def test_prepare_context_request_sends_with_the_gateway_clamp_when_only_the_output_budget_overflows():
    """Parity with the gateway: a prompt that leaves output room short of the
    configured ceiling goes out with a clamped budget instead of failing the
    session, and the room reported is the gateway's own clamp arithmetic."""

    window = 10_000
    events: list[tuple[str, dict[str, object]]] = []
    llm = ScriptedLLM([], context_window_tokens=window)
    runner = AgentSessionRunner(
        llm=llm,
        tools=ToolRegistry([]),
        system_prompt="test",
        config=AgentSessionConfig(max_response_tokens=5_000),
        event_sink=lambda event, payload: events.append((event, payload)),
    )
    messages = [ChatMessage("user", "word " * 3_600)]
    fits, prompt_tokens, _window = context_request_fits(llm, messages, max_tokens=5_000)
    assert not fits
    assert prompt_tokens + CONTEXT_OUTPUT_TOKEN_MARGIN + CONTEXT_OUTPUT_MIN_TOKENS <= window

    # Untouched content: no compactor is attached and there is no tool result
    # to summarize, so the emergency fit hands back an unedited copy.
    assert (
        runner._prepare_context_request(
            messages, (), 30.0, allow_semantic_compaction=False
        )
        == messages
    )
    ((_event, payload),) = [
        (event, payload) for event, payload in events if event == "context_output_clamped"
    ]
    clamped, prompt_fits = clamp_requested_max_tokens(
        requested_max_tokens=5_000,
        estimated_prompt_tokens=prompt_tokens,
        context_window=window,
    )
    assert prompt_fits and clamped == window - prompt_tokens - CONTEXT_OUTPUT_TOKEN_MARGIN
    assert payload == {
        "estimated_prompt_tokens": prompt_tokens,
        "context_window": window,
        "requested_max_output_tokens": 5_000,
        "available_output_tokens": clamped,
    }

    # One token short of a usable minimum output: refused before the network,
    # as the gateway refuses a prompt that leaves no room.
    tight = ScriptedLLM(
        [],
        context_window_tokens=prompt_tokens
        + CONTEXT_OUTPUT_TOKEN_MARGIN
        + CONTEXT_OUTPUT_MIN_TOKENS
        - 1,
    )
    with pytest.raises(LLMProxyError) as excinfo:
        _context_runner(tight, max_response_tokens=5_000)._prepare_context_request(
            messages, (), 30.0, allow_semantic_compaction=False
        )
    assert is_context_overflow_error(excinfo.value)


def test_forced_compaction_proceeds_past_the_call_cap_and_message_floor():
    """A request that does not fit has no other way forward: the cap and the
    floor bound threshold-triggered compaction only. The structural guard —
    nothing older than the retained tail — and the failure circuit still hold
    under force, because then compaction cannot help."""

    llm = ScriptedLLM([ProviderResponse(content="## 目标\nkeep going")] * 3)
    compactor = ContextCompactor(
        llm,
        ContextCompactionConfig(
            token_threshold=10**9, min_messages=20, keep_recent_messages=2, max_calls=0
        ),
    )
    messages = [ChatMessage("system", "system")]
    messages.extend(ChatMessage("user", f"m{index}") for index in range(6))
    assert compactor.should_compact(messages)[1]["skip_reason"] == "call_limit_reached"
    forced, reason = compactor.should_compact(messages, force=True)
    assert forced and reason["trigger_reason"] == "forced_context_overflow"

    first = compactor.compact(messages, force=True)
    assert first is not None and first.event["status"] == "ok"
    assert len(first.messages) == 4
    # Attempts now exceed the cap and the conversation sits below the floor:
    # a forced compaction still proceeds while older messages remain.
    second = compactor.compact(
        [*first.messages, ChatMessage("user", "m6"), ChatMessage("user", "m7")],
        force=True,
    )
    assert second is not None and second.event["status"] == "ok"
    assert compactor.should_compact(list(second.messages), force=True)[1][
        "skip_reason"
    ] == "nothing_to_compact"
    assert compactor.compact(list(second.messages), force=True) is None

    class BrokenLLM:
        def complete(self, *args, **kwargs):
            raise RuntimeError("compaction model down")

    broken = ContextCompactor(
        BrokenLLM(),
        ContextCompactionConfig(
            token_threshold=1, min_messages=4, keep_recent_messages=2, max_failures=1
        ),
    )
    failed = broken.compact(messages)
    assert failed is not None and failed.event["status"] == "error"
    assert broken.should_compact(messages, force=True)[1]["skip_reason"] == "failure_circuit_open"


def test_prepare_context_request_rejects_missing_window_without_assert(monkeypatch):
    runner = _context_runner(ScriptedLLM([], context_window_tokens=200))
    messages = [ChatMessage("user", "short")]
    monkeypatch.setattr(
        "autotrade.agent.runner.context_request_fits",
        lambda *args, **kwargs: (False, 10, None),
    )
    with pytest.raises(RuntimeError, match="context window is unavailable"):
        runner._prepare_context_request(
            messages, (), 30.0, allow_semantic_compaction=False
        )


def test_compaction_trigger_counts_the_provider_tool_schemas():
    compactor = ContextCompactor(
        ScriptedLLM([]),
        ContextCompactionConfig(token_threshold=400, min_messages=2, keep_recent_messages=1),
    )
    messages = [ChatMessage("system", "s")] + [
        ChatMessage("user", f"turn {index} " + "x" * 60) for index in range(6)
    ]
    tools = ({"type": "function", "function": {"name": "t", "description": "d" * 3_000}},)
    without_tools, reason = compactor.should_compact(messages)
    with_tools, reason_with = compactor.should_compact(messages, tools=tools)
    assert without_tools is False and reason["skip_reason"] == "below_token_threshold"
    assert with_tools is True
    assert reason_with["estimated_tokens"] > reason["estimated_tokens"]


def test_compactor_keeps_markdown_summary_and_files_trail_across_compactions():
    llm = ScriptedLLM(
        [
            ProviderResponse(content="<think>plan</think>## 目标\nfirst\n\n## 下一步\n- more"),
            ProviderResponse(content="## 目标\nsecond\n\n## 下一步\n- finish"),
            ProviderResponse(content="   "),
        ]
    )
    compactor = ContextCompactor(
        llm, ContextCompactionConfig(token_threshold=1, min_messages=4, keep_recent_messages=1)
    )
    messages = [
        ChatMessage("system", "s"),
        ChatMessage("user", "go"),
        ChatMessage(
            "assistant", None, (ToolCall("r1", "read_file", {"root": "snapshot", "path": "a.parquet"}),)
        ),
        ChatMessage("tool", "{}", tool_call_id="r1"),
        ChatMessage("assistant", None, (ToolCall("w1", "write_file", {"path": "output/main.py", "content": "x"}),)),
        ChatMessage("tool", "{}", tool_call_id="w1"),
        ChatMessage("user", "next"),
    ]
    first = compactor.compact(messages)
    assert first is not None and first.event["status"] == "ok"
    envelope = json.loads(first.messages[1].content or "{}")
    assert envelope["summary"].startswith("## 目标\nfirst")
    assert envelope["files"] == {"read": ["[snapshot] a.parquet"], "modified": ["output/main.py"]}
    # The compactor request carried the previous summary as Markdown, not JSON.
    second_input = list(first.messages) + [
        ChatMessage("assistant", None, (ToolCall("r2", "grep", {"root": "workspace", "path": "inputs"}),)),
        ChatMessage("tool", "{}", tool_call_id="r2"),
        ChatMessage("user", "again"),
        ChatMessage("user", "and again"),
    ]
    second = compactor.compact(second_input)
    assert second is not None and second.event["status"] == "ok"
    request = llm.calls[1]["messages"][1].content
    assert "## 上一份摘要" in request and "## 目标\nfirst" in request
    envelope = json.loads(second.messages[1].content or "{}")
    assert envelope["files"]["read"] == ["[snapshot] a.parquet", "[workspace] inputs"]
    assert len(second.messages) == 3
    # An empty reply is one failed attempt: the history is kept untouched.
    third = compactor.compact(list(second.messages) + [ChatMessage("user", f"m{i}") for i in range(4)])
    assert third is not None and third.event["status"] == "error"
    assert "empty" in third.event["error"]
    assert compactor.compaction_count == 2
