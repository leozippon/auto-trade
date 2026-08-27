from __future__ import annotations

import json
from pathlib import Path

import pytest

from autotrade.agent import (
    AgentSessionConfig,
    AgentSessionRunner,
    ContextCompactionConfig,
    ContextCompactor,
    ExploreSubAgentEngine,
)
from autotrade.agent.compact import estimate_messages_tokens
from autotrade.agent.experiment_facts import build_experiment_facts
from autotrade.agent.prompts import (
    FOLD_STATIC_SECTIONS,
    META_SYSTEM_PROMPT,
    build_system_prompt,
)
from autotrade.environment.artifacts import new_revision_id
from autotrade.environment.identity import AgentRefStore
from autotrade.environment.llm import (
    LOCAL_QWEN_MODEL,
    ChatMessage,
    LLMProxyError,
    ProviderResponse,
    ScriptedLLM,
    ToolCall,
    context_request_fits,
    is_context_overflow_error,
)
from autotrade.environment.step_tree import StepTree
from autotrade.environment.tools import (
    FinishFoldTool,
    SafeWorkspace,
    ToolRegistry,
    ToolResult,
    TodoTool,
    ToolSpec,
    WriteFileTool,
)
from autotrade.pipelines.local_backend import SessionBudgetLLM


def finish_fold_tool(root: Path) -> tuple[FinishFoldTool, str]:
    """A terminal tool over a real step tree carrying one validated node.

    ``finish_fold`` is the Fold session's terminal tool; these are generic
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
        fold_id="fold_ref_ab",
        run_id="run_x",
        result_name="valid_000",
        revision_id=new_revision_id("revision"),
        metrics={"total_return": 0.01},
        complete_validation=True,
    )
    return FinishFoldTool(tree, fold_id="fold_ref_ab", run_id="run_x"), node_id


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
                content=json.dumps({"goal": "continue", "next_actions": ["validate"]})
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


def test_compactor_bounds_one_huge_recent_tool_result_before_local_request():
    llm = ScriptedLLM(
        [ProviderResponse(content=json.dumps({"goal": "continue", "next_steps": []}))],
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
    compact_input = json.loads(request[1].content or "{}")
    recent = compact_input["messages_since_previous_summary"]
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


def test_context_token_estimate_includes_reasoning_content():
    without_reasoning = [ChatMessage("assistant", "answer")]
    with_reasoning = [ChatMessage("assistant", "answer", reasoning_content="r" * 400)]

    assert estimate_messages_tokens(with_reasoning) > estimate_messages_tokens(
        without_reasoning
    )


def test_fold_session_tracks_calls_and_finish_value(tmp_path: Path):
    finish, node_id = finish_fold_tool(tmp_path)
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
    )
    result = runner.run("finish the validated strategy")
    assert result.status == "finished"
    assert result.finish_value["node_id"] == node_id
    assert result.finish_value["status"] == "fold_finished"
    # Finishing only nominates; the Pipeline freezes.
    assert result.finish_value["fold_status"] == "pending_pipeline_review"
    assert result.finish_value["write_locked"] is True
    assert result.llm_calls == 1


def test_fold_session_nudges_text_only_turn_then_requires_finish(tmp_path: Path):
    finish, node_id = finish_fold_tool(tmp_path)
    llm = ScriptedLLM(
        [
            ProviderResponse(content="I should act next."),
            ProviderResponse(
                tool_calls=(ToolCall("f", "finish_fold", {"node_id": node_id}),)
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
    finish, node_id = finish_fold_tool(tmp_path)
    shell = DeclaredReadOnlyShell()
    llm = ScriptedLLM(
        [
            ProviderResponse(
                tool_calls=(ToolCall("s", "shell", {"argv": ["rg", "needle", "."]}),),
                reasoning_content="inspect before finishing",
            ),
            ProviderResponse(
                tool_calls=(ToolCall("f", "finish_fold", {"node_id": node_id}),)
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
    finish, node_id = finish_fold_tool(tmp_path)
    shell = LongResultShell()
    llm = ScriptedLLM(
        [
            ProviderResponse(
                tool_calls=(ToolCall("shell-1", "shell", {"argv": ["rg", "x"]}),)
            ),
            ProviderResponse(
                tool_calls=(ToolCall("finish-1", "finish_fold", {"node_id": node_id}),)
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
    finish, _node_id = finish_fold_tool(tmp_path)
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


def test_terminal_tool_cancels_later_mutation_in_same_turn(tmp_path: Path):
    finish, node_id = finish_fold_tool(tmp_path)
    workspace = SafeWorkspace(tmp_path)
    llm = ScriptedLLM(
        [
            ProviderResponse(
                tool_calls=(
                    ToolCall("f", "finish_fold", {"node_id": node_id}),
                    ToolCall(
                        "todo",
                        "todo",
                        {"action": "create", "subject": "should not persist"},
                    ),
                )
            )
        ]
    )
    runner = AgentSessionRunner(
        llm=llm,
        tools=ToolRegistry([finish, TodoTool(workspace)]),
        system_prompt="daily JSON only",
    )
    runner.run("finish")
    assert "generate_orders" in (tmp_path / "output/main.py").read_text(
        encoding="utf-8"
    )
    assert not (tmp_path / "TODO.json").exists()


def test_explore_accepts_write_file_tool(tmp_path: Path):
    engine = ExploreSubAgentEngine(
        llm=ScriptedLLM([]),
        tools=ToolRegistry([WriteFileTool(SafeWorkspace(tmp_path))]),
    )
    assert {spec.name for spec in engine.tools.specs()} == {"write_file"}


def test_explore_dispatches_full_shell_commands():
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
    result = ExploreSubAgentEngine(
        llm=llm,
        tools=ToolRegistry([shell]),
    ).run("inspect", role="developer")
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

    for name in ("daily_backtest", "external_lookup"):
        with pytest.raises(ValueError):
            AgentSessionRunner(
                llm=ScriptedLLM([]),
                tools=ToolRegistry([StubTool(name)]),
                system_prompt="offline meta",
                config=AgentSessionConfig(mode="meta"),
            )

    with pytest.raises(ValueError, match="daily_backtest requires finish_fold"):
        AgentSessionRunner(
            llm=ScriptedLLM([]),
            tools=ToolRegistry([StubTool("daily_backtest")]),
            system_prompt="formal fold",
            config=AgentSessionConfig(mode="fold"),
        )


def test_prompt_and_facts_encode_daily_json_and_offline_meta_boundaries(
    tmp_path: Path,
):
    prompt = build_system_prompt(mode="fold", experiment_facts={"fold": "visible"})
    assert "generate_orders(context)" in prompt
    assert "严格 JSON 往返的订单数组" in prompt
    assert "分钟策略时钟" in prompt
    assert "不能假定 `context.bars` 含完整历史" in prompt
    assert "实际挂载清单、schema、单位引用" in prompt
    assert "Test" in prompt and "Held-out" in prompt
    # The prohibition list carries the item-6 execution-model rules.
    prohibitions = prompt[prompt.index("# 禁止事项") :]
    for rule in (
        "读取当前或未来 Test、Held-out",
        "绕过 `available_at`",
        "把历史分钟、竞价或事件时间当成策略执行时钟",
        "伪造工具结果",
    ):
        assert rule in prohibitions
    meta_prompt = build_system_prompt(mode="meta", experiment_facts={})
    assert "离线 Meta 主协调者" in META_SYSTEM_PROMPT
    assert "# 核心执行合同" not in meta_prompt
    assert "`buy`/`sell` action" in meta_prompt
    assert "不早于 `context.inference_at`" in meta_prompt
    # The Meta session is offline and evidence-bounded, may regularize under
    # the modification constraints, and may declare its own image dependencies.
    for rule in (
        "不得读取当前或未来 Test、Held-out 原始记录",
        "不得运行回测",
        "`inputs/meta_context.json`",
        "`modification_check`",
        "sandbox_environment.json",
        "焊接的日历日期",
    ):
        assert rule in META_SYSTEM_PROMPT

    facts = build_experiment_facts(
        manifest={"kind": "meta_learning", "experiment_id": "exp"},
        ref_store=AgentRefStore(tmp_path / "experiment"),
        runtime_env={"sandbox_spec": {"network": "none"}},
    )
    assert facts["visibility_policy"]["test_visible"] is False
    assert facts["visibility_policy"]["heldout_visible"] is False
    assert facts["meta_learning"]["backtest_allowed"] is False
    assert facts["meta_learning"]["sample_window_only"] is True
    assert facts["meta_learning"]["prior_output_path"].endswith("PRIOR.md")


def test_fold_prompt_keeps_hard_boundaries_and_leaves_how_tos_mounted():
    prompt = build_system_prompt(mode="fold", experiment_facts={})
    for rule in (
        "自由检查已挂载的事实",
        "不得从名称、日期或路径推断隐藏区间",
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

    A substring check over FOLD_STATIC_SECTIONS only sees the fold sections —
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
    for section in FOLD_STATIC_SECTIONS:
        assert f"```text\n{section.strip()}\n```" in rendered


def _prompt_tool_tokens(text: str) -> set[str]:
    """Backticked lowercase identifiers a prompt presents as tool names."""
    import re

    return set(re.findall(r"`([a-z][a-z0-9_]*)`", text))


def _all_registrable_tool_names() -> set[str]:
    """Every tool name any session in this tree can actually register."""
    import tempfile

    import autotrade.environment.tools as tools_pkg
    from autotrade.agent.runner import FinishMetaTool
    from autotrade.environment.nl.engine import TEXT_RETRIEVE_TOOL
    from autotrade.environment.tools import SafeWorkspace, SearchRoots
    from autotrade.environment.tools.search import GlobTool, GrepTool, ReadFileTool
    # The backtest tool is constructed per fold in local_backend, so its name
    # is named here rather than discovered.
    names = {TEXT_RETRIEVE_TOOL, "daily_backtest"}
    with tempfile.TemporaryDirectory() as tmp:
        roots = SearchRoots(SafeWorkspace(Path(tmp)))
        instances = [GlobTool(roots), GrepTool(roots), ReadFileTool(roots)]
    candidates = [*instances, FinishMetaTool]
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
    from autotrade.agent.runner import _FOLD_TOOLS, _META_TOOLS
    from autotrade.environment.nl.engine import (
        SUB_AGENT_SYSTEM_PROMPT,
        TEXT_RETRIEVE_TOOL,
    )

    registrable = _all_registrable_tool_names()
    # Every allowlisted name must correspond to a tool that exists.
    assert (_FOLD_TOOLS | _META_TOOLS) <= registrable

    sessions = (
        ("fold", build_system_prompt(mode="fold", experiment_facts={}), _FOLD_TOOLS),
        ("meta", build_system_prompt(mode="meta", experiment_facts={}), _META_TOOLS),
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
    from autotrade.agent.runner import _FOLD_TOOLS

    registrable = _all_registrable_tool_names()
    # `finish_meta` exists, but only a Meta session may register it.
    assert "finish_meta" in registrable
    mutated = (
        build_system_prompt(mode="fold", experiment_facts={})
        + "\n- 用 `finish_meta` 结束本 Fold。"
    )
    referenced = _prompt_tool_tokens(mutated) & registrable
    assert sorted(referenced - set(_FOLD_TOOLS)) == ["finish_meta"]


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
