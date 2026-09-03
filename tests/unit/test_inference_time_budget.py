from __future__ import annotations

import json
import threading
from datetime import UTC, datetime
from pathlib import Path

import pytest

from autotrade.agent.compact import ContextCompactionConfig, ContextCompactor
from autotrade.agent.prompts import WRAP_UP_PROMPT
from autotrade.agent.subagent import SubAgentEngine
from autotrade.agent.runner import AgentSessionConfig, AgentSessionRunner
from autotrade.environment.artifacts import FilesystemArtifactStore
from autotrade.environment.broker import BrokerProfile
from autotrade.environment.identity import AgentRefStore
from autotrade.environment.llm import (
    ChatMessage,
    ProviderResponse,
    ScriptedLLM,
    ToolCall,
)
from autotrade.environment.runtime import AgentTraceWriter
from autotrade.environment.step_tree import StepTree
from autotrade.environment.strategy import StrategySchedule
from autotrade.environment.time_budget import InferenceTimeBudget
from autotrade.environment.tools import (
    AskUserTool,
    FinishFoldTool,
    StepRollbackTool,
    ToolRegistry,
    ToolResult,
    ToolSpec,
)
from autotrade.pipelines.config import (
    EvaluationResult,
    FoldSessionRequest,
    SnapshotBundle,
)
from autotrade.pipelines.folds import FoldSpec
from autotrade.pipelines.local_backend import (
    FoldBacktestTool,
    SessionBudgetLLM,
    SessionCallBudget,
    session_role_quotas,
)

from .fixtures_sandbox import PassingModificationCheck


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


class TimedLLM:
    def __init__(self, delegate: ScriptedLLM, clock: FakeClock, seconds: float) -> None:
        self.delegate = delegate
        self.clock = clock
        self.seconds = seconds

    def complete(self, *args, **kwargs):
        self.clock.advance(self.seconds)
        return self.delegate.complete(*args, **kwargs)


class PassingCheck(PassingModificationCheck):
    """The shared double, carrying this suite's marker value."""

    def __init__(self, output_dir: Path, models_dir: Path) -> None:
        super().__init__(output_dir, models_dir, allowed_to_backtest=True)


class RecordingShell:
    spec = ToolSpec(
        "shell",
        "test-only research shell",
        {"type": "object", "properties": {}, "required": []},
    )

    def __init__(self) -> None:
        self.calls = 0

    def invoke(self, _arguments):
        self.calls += 1
        return ToolResult(True, value={"stdout": "research"})


class ScheduledTimedLLM:
    def __init__(
        self, delegate: ScriptedLLM, clock: FakeClock, seconds: list[float]
    ) -> None:
        self.delegate = delegate
        self.clock = clock
        self.seconds = list(seconds)

    def complete(self, *args, **kwargs):
        self.clock.advance(self.seconds.pop(0))
        return self.delegate.complete(*args, **kwargs)


def _fold_request() -> FoldSessionRequest:
    moment = datetime(2025, 12, 31, 23, 59, 59, tzinfo=UTC)
    return FoldSessionRequest(
        experiment_id="exp",
        epoch_id="epoch_001",
        fold=FoldSpec(
            fold_id="fold_2026Q1",
            input_window_start="20240101",
            input_window_end="20250930",
            validation_start="20251001",
            validation_end="20251231",
            test_start="20260101",
            test_end="20260331",
            valid_decision_time=moment,
            test_decision_time=moment,
        ),
        run_id="run_budget",
        parent=None,
        prior="",
        snapshot=SnapshotBundle("snapshot", "decision", "replay"),
        max_steps=3,
        max_backtests=3,
        max_llm_calls=3,
        deadline_seconds=2.0,
        record_failed_attempts=False,
    )


def test_backtest_failure_past_wall_deadline_keeps_llm_repair_budget(
    tmp_path: Path,
) -> None:
    clock = FakeClock()
    time_budget = InferenceTimeBudget(duration_seconds=2.0, clock=clock)
    request = _fold_request()
    output = tmp_path / "output"
    models = tmp_path / "models"
    output.mkdir()
    models.mkdir()
    (output / "main.py").write_text(
        "def generate_orders(context):\n    return []\n", encoding="utf-8"
    )
    tree = StepTree(tmp_path / "steps")
    ref_store = AgentRefStore(tmp_path / "experiment")
    fold_ref = ref_store.get_or_create("fold", request.fold.fold_id)
    run_ref = ref_store.get_or_create("run", request.run_id)

    class FailThenPassEvaluator:
        calls = 0

        def evaluate(self, _request):
            self.calls += 1
            clock.advance(10.0)
            if self.calls == 1:
                raise RuntimeError("repairable validation failure")
            result = tmp_path / "result.json"
            result.write_text("{}\n", encoding="utf-8")
            return EvaluationResult(
                summary={"total_return": 0.01}, result_ref=str(result)
            )

    evaluator = FailThenPassEvaluator()
    backtest = FoldBacktestTool(
        request=request,
        output_dir=output,
        models_dir=models,
        modification_check=PassingCheck(output, models),
        artifact_store=FilesystemArtifactStore(tmp_path / "artifacts"),
        evaluator=evaluator,
        tree=tree,
        schedule=StrategySchedule(),
        broker_profile=BrokerProfile(),
        time_budget=time_budget,
        ref_store=ref_store,
    )
    scripted = ScriptedLLM(
        [
            ProviderResponse(tool_calls=(ToolCall("bad", "daily_backtest", {}),)),
            ProviderResponse(tool_calls=(ToolCall("fixed", "daily_backtest", {}),)),
            ProviderResponse(
                tool_calls=(
                    ToolCall(
                        "done",
                        "finish_fold",
                        {
                            "node_id": (
                                "epoch_001__"
                                f"{fold_ref}__"
                                f"{run_ref}__valid_002"
                            )
                        },
                    ),
                )
            ),
        ]
    )
    call_budget = SessionCallBudget(max_calls=3, time_budget=time_budget)
    llm = SessionBudgetLLM(TimedLLM(scripted, clock, 0.5), budget=call_budget)
    trace_path = tmp_path / "agent-trace.jsonl"
    runner = AgentSessionRunner(
        llm=llm,
        tools=ToolRegistry(
            [
                backtest,
                FinishFoldTool(
                    tree,
                    fold_id=fold_ref,
                    run_id=run_ref,
                ),
            ]
        ),
        system_prompt="repair a failed validation and finish",
        config=AgentSessionConfig(
            max_llm_calls=3,
            deadline_seconds=2.0,
            finalize_before_deadline_seconds=1.0,
        ),
        time_budget=time_budget,
        event_sink=AgentTraceWriter(
            trace_path, ids={"run_ref": run_ref, "fold_ref": fold_ref}
        ).emit,
    )
    assert runner.time_budget is backtest.time_budget is llm.time_budget is time_budget

    result = runner.run("validate")

    assert result.status == "finished"
    assert evaluator.calls == 2
    assert len(scripted.calls) == 3
    repair_context = "".join(
        message.content or "" for message in scripted.calls[1]["messages"]
    )
    assert (
        "daily Validation failed: RuntimeError: repairable validation failure"
        in repair_context
    )
    assert str(tmp_path) not in repair_context
    assert time_budget.remaining() == pytest.approx(0.5)
    # The host keeps the real EvaluationResult path for ledger/freeze work, but
    # neither the Agent observation nor its audit Trace may expose that path.
    host_result_ref = backtest.steps[-1].validation.result_ref
    assert Path(host_result_ref).is_absolute()
    trace_text = trace_path.read_text(encoding="utf-8")
    assert host_result_ref not in trace_text
    events = [json.loads(line) for line in trace_text.splitlines()]
    successful = next(
        event
        for event in events
        if event.get("event_type") == "tool_call"
        and event.get("tool") == "daily_backtest"
        and event["result"]["ok"]
    )
    # The reference names the step-tree attachment that actually holds the full
    # replay record, under the ``steps`` search root the Agent can read.
    assert successful["result"]["value"]["result_root"] == "steps"
    assert successful["result"]["value"]["result_ref"] == (
        f"epoch_001__{fold_ref}__{run_ref}__valid_002/validation/result.json"
    )
    assert (tree.root / successful["result"]["value"]["result_ref"]).is_file()


def test_complete_node_enters_hard_finalization_without_compaction_or_research(
    tmp_path: Path,
) -> None:
    clock = FakeClock()
    time_budget = InferenceTimeBudget(duration_seconds=10.0, clock=clock)
    output = tmp_path / "output"
    models = tmp_path / "models"
    output.mkdir()
    models.mkdir()
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "main.py").write_text(
        "def generate_orders(context):\n    return []\n", encoding="utf-8"
    )
    (output / "main.py").write_text(
        "def generate_orders(context):\n    return []\n", encoding="utf-8"
    )
    tree = StepTree(tmp_path / "steps")
    node_id = tree.record_step(
        snapshot,
        epoch_id="epoch_001",
        fold_id="fold_ref_current",
        run_id="run_current",
        result_name="valid_001",
        revision_id="revision_current",
        metrics={"total_return": 0.03, "max_drawdown": 0.12},
    )

    class ExistingValidation:
        spec = ToolSpec(
            "daily_backtest",
            "return one completed current-run validation",
            {"type": "object", "properties": {}, "required": []},
        )

        def invoke(self, _arguments):
            (output / "main.py").write_text("workspace drift\n", encoding="utf-8")
            return ToolResult(
                True,
                value={
                    "node_id": node_id,
                    "revision_id": "revision_current",
                    "stats": {"total_return": 0.03, "max_drawdown": 0.12},
                },
            )

    shell = RecordingShell()
    scripted = ScriptedLLM(
        [
            ProviderResponse(
                tool_calls=(
                    ToolCall("valid", "daily_backtest", {}),
                    ToolCall("research", "shell", {}),
                )
            ),
            ProviderResponse(
                tool_calls=(
                    ToolCall("rollback", "step_rollback", {"node_id": node_id}),
                )
            ),
            ProviderResponse(
                tool_calls=(ToolCall("finish", "finish_fold", {"node_id": node_id}),)
            ),
        ]
    )
    compact_scripted = ScriptedLLM(
        [ProviderResponse(content="## 目标\nshould not run")]
    )
    shared = SessionCallBudget(max_calls=6, time_budget=time_budget)
    main = SessionBudgetLLM(
        ScheduledTimedLLM(scripted, clock, [5.0, 0.1, 0.1]), budget=shared
    )
    compact = SessionBudgetLLM(compact_scripted, budget=shared)
    compactor = ContextCompactor(
        compact,
        ContextCompactionConfig(
            token_threshold=1,
            min_messages=5,
            keep_recent_messages=1,
            max_response_tokens=200,
        ),
    )
    events: list[tuple[str, dict[str, object]]] = []
    runner = AgentSessionRunner(
        llm=main,
        tools=ToolRegistry(
            [
                ExistingValidation(),
                shell,
                StepRollbackTool(
                    tree,
                    output,
                    models,
                    fold_id="fold_ref_current",
                    run_id="run_current",
                ),
                FinishFoldTool(tree, fold_id="fold_ref_current", run_id="run_current"),
            ]
        ),
        system_prompt="research before finishing",
        config=AgentSessionConfig(
            max_llm_calls=3,
            deadline_seconds=10.0,
            finalize_before_deadline_seconds=6.0,
            # No trailing wrap-up grace: this 10s budget would otherwise sit
            # entirely inside the default grace window and never reach hard
            # finalization.
            deadline_grace_seconds=0.0,
            max_response_tokens=500,
        ),
        compactor=compactor,
        time_budget=time_budget,
        event_sink=lambda event, payload: events.append((event, payload)),
    )

    result = runner.run("validate and continue researching")

    assert result.status == "finished"
    assert result.finish_value["node_id"] == node_id
    assert shell.calls == 0
    assert compact_scripted.calls == []
    assert "generate_orders" in (output / "main.py").read_text(encoding="utf-8")
    final_tools = {item["function"]["name"] for item in scripted.calls[1]["tools"]}
    assert final_tools == {"step_rollback", "finish_fold"}
    assert len(scripted.calls[1]["messages"]) == 2
    final_payload = json.loads(scripted.calls[1]["messages"][1].content or "{}")
    assert final_payload["complete_validation_candidates"][0]["node_id"] == node_id
    assert any(
        event == "hard_finalization_started"
        and payload["candidate_node_ids"] == [node_id]
        for event, payload in events
    )
    assert any(
        event == "tool_call"
        and payload["tool"] == "shell"
        and "unavailable" in str(payload["result"])
        for event, payload in events
    )


def test_a_validation_completing_inside_the_grace_keeps_the_conversation(
    tmp_path: Path,
) -> None:
    """Past the main deadline the wrap-up window owns the session.

    A turn that starts outside every window can still end inside the grace,
    because the backtest it dispatched burns wall clock. Activating hard
    finalization there would replace the whole conversation with the two-message
    finalization context and skip WRAP_UP_PROMPT entirely, so the session would
    lose both its context and the last-modification autonomy the grace promises.
    """

    clock = FakeClock()
    budget_seconds = 3600.0
    grace_seconds = 600.0
    time_budget = InferenceTimeBudget(duration_seconds=budget_seconds, clock=clock)
    output = tmp_path / "output"
    output.mkdir()
    (output / "main.py").write_text(
        "def generate_orders(context):\n    return []\n", encoding="utf-8"
    )
    tree = StepTree(tmp_path / "steps")
    node_id = tree.record_step(
        output,
        epoch_id="epoch_001",
        fold_id="fold_grace",
        run_id="run_grace",
        result_name="valid_001",
        revision_id="revision_grace",
        metrics={"total_return": 0.01},
    )

    class SlowValidation:
        spec = ToolSpec(
            "daily_backtest",
            "complete Validation",
            {"type": "object", "properties": {}, "additionalProperties": False},
            mutating=True,
        )

        def invoke(self, arguments):
            # The turn started with 650 s left (outside grace) and this replay
            # ends with 500 s left, i.e. past the main deadline.
            clock.advance(150.0)
            return ToolResult(
                True,
                value={
                    "node_id": node_id,
                    "revision_id": "revision_grace",
                    "stats": {"total_return": 0.01},
                },
            )

    scripted = ScriptedLLM(
        [
            ProviderResponse(tool_calls=(ToolCall("b1", "daily_backtest", {}),)),
            ProviderResponse(tool_calls=(ToolCall("f1", "finish_fold", {"node_id": node_id}),)),
        ],
        context_window_tokens=128_000,
    )
    events: list[tuple[str, dict[str, object]]] = []
    runner = AgentSessionRunner(
        llm=scripted,
        tools=ToolRegistry(
            [
                SlowValidation(),
                FinishFoldTool(tree, fold_id="fold_grace", run_id="run_grace"),
            ]
        ),
        system_prompt="fold system prompt",
        config=AgentSessionConfig(
            mode="fold",
            max_llm_calls=4,
            deadline_seconds=budget_seconds,
            deadline_grace_seconds=grace_seconds,
            finalize_before_deadline_seconds=300.0,
        ),
        time_budget=time_budget,
        event_sink=lambda event, payload: events.append((event, payload)),
    )
    clock.advance(budget_seconds - 650.0)
    assert time_budget.remaining() == 650.0

    result = runner.run("go")

    assert result.status == "finished"
    assert not [event for event, _ in events if event == "hard_finalization_started"]
    wrap_up = [payload for event, payload in events if event == "wrap_up_started"]
    assert wrap_up and wrap_up[0]["remaining_seconds"] == 500.0
    # The conversation survives: the wrap-up prompt is appended to it, it is not
    # a fresh two-message finalization context.
    second = scripted.calls[1]["messages"]
    assert [message.role for message in second[:2]] == ["system", "user"]
    assert {message.role for message in second} == {"system", "user", "assistant", "tool"}
    assert second[-1].content == WRAP_UP_PROMPT


def test_reserve_without_complete_node_keeps_research_and_compaction_available() -> (
    None
):
    clock = FakeClock()
    time_budget = InferenceTimeBudget(duration_seconds=10.0, clock=clock)
    shell = RecordingShell()
    scripted = ScriptedLLM(
        [
            ProviderResponse(tool_calls=(ToolCall("research-1", "shell", {}),)),
            ProviderResponse(tool_calls=(ToolCall("research-2", "shell", {}),)),
        ]
    )
    compact_scripted = ScriptedLLM(
        [ProviderResponse(content="## 目标\ncontinue\n\n## 下一步\n- research")]
    )
    shared = SessionCallBudget(max_calls=4, time_budget=time_budget)
    main = SessionBudgetLLM(
        ScheduledTimedLLM(scripted, clock, [5.0, 0.1]), budget=shared
    )
    compactor = ContextCompactor(
        SessionBudgetLLM(compact_scripted, budget=shared),
        ContextCompactionConfig(
            token_threshold=1,
            min_messages=4,
            keep_recent_messages=1,
            max_response_tokens=200,
            min_remaining_seconds=0.0,
        ),
    )
    runner = AgentSessionRunner(
        llm=main,
        tools=ToolRegistry([shell]),
        system_prompt="research until a validation exists",
        config=AgentSessionConfig(
            max_llm_calls=2,
            deadline_seconds=10.0,
            finalize_before_deadline_seconds=6.0,
            max_response_tokens=500,
        ),
        compactor=compactor,
        time_budget=time_budget,
    )

    with pytest.raises(RuntimeError, match="call budget"):
        runner.run("research")

    assert shell.calls == 2
    assert len(compact_scripted.calls) == 1
    second_tool_names = {
        item["function"]["name"] for item in scripted.calls[1]["tools"]
    }
    assert second_tool_names == {"shell"}


def test_evaluation_and_ask_user_waits_are_refunded_on_success_and_failure() -> None:
    clock = FakeClock()
    budget = InferenceTimeBudget(duration_seconds=5.0, clock=clock)

    before = budget.remaining()
    with budget.pause():
        clock.advance(20.0)
        with pytest.raises(RuntimeError, match="evaluation failed"), budget.pause():
            clock.advance(10.0)
            raise RuntimeError("evaluation failed")
    assert budget.remaining() == pytest.approx(before)

    ask = AskUserTool(
        lambda _question, _summary: (clock.advance(30.0), "continue")[1],
        time_budget=budget,
    )
    assert ask.invoke({"question": "continue?"}).value["reply"] == "continue"
    assert budget.remaining() == pytest.approx(before)


def test_non_exempt_llm_and_concurrent_read_tools_consume_effective_time() -> None:
    clock = FakeClock()
    budget = InferenceTimeBudget(duration_seconds=10.0, clock=clock)
    shared = SessionCallBudget(max_calls=2, time_budget=budget)
    llm = SessionBudgetLLM(
        TimedLLM(ScriptedLLM([ProviderResponse(content="ok")]), clock, 1.5),
        budget=shared,
    )
    llm.complete([ChatMessage("user", "one")])

    class TimedReadTool:
        def __init__(self, name: str) -> None:
            self.spec = ToolSpec(
                name,
                "test read",
                {"type": "object", "properties": {}, "required": []},
            )

        def invoke(self, _arguments):
            clock.advance(1.0)
            return ToolResult(True, value={"read": True})

    registry = ToolRegistry([TimedReadTool("glob"), TimedReadTool("grep")])
    runner = AgentSessionRunner(
        llm=ScriptedLLM([]),
        tools=registry,
        system_prompt="read",
        time_budget=budget,
    )
    records, skipped = runner._dispatch_tool_calls(
        (ToolCall("a", "glob", {}), ToolCall("b", "grep", {})), budget
    )

    assert skipped is None
    assert all(record[1]["ok"] for record in records)
    assert budget.remaining() == pytest.approx(6.5)


def test_all_session_components_hold_one_budget_and_expire_at_boundaries() -> None:
    clock = FakeClock()
    time_budget = InferenceTimeBudget(duration_seconds=1.0, clock=clock)
    calls = SessionCallBudget(max_calls=4, time_budget=time_budget)
    main = SessionBudgetLLM(
        ScriptedLLM([ProviderResponse(content="main")]), budget=calls
    )
    subagent = SessionBudgetLLM(
        ScriptedLLM([ProviderResponse(content="agent")]), budget=calls
    )
    compact = SessionBudgetLLM(
        ScriptedLLM([ProviderResponse(content="compact")]), budget=calls
    )

    assert main.budget is subagent.budget is compact.budget is calls
    assert calls.time_budget is time_budget
    with pytest.raises(ValueError, match="share one inference time budget"):
        AgentSessionRunner(
            llm=main,
            tools=ToolRegistry(),
            system_prompt="mismatched budget",
            time_budget=InferenceTimeBudget(duration_seconds=1.0, clock=clock),
        )
    clock.advance(1.0)
    with pytest.raises(TimeoutError, match="deadline exceeded"):
        main.complete([ChatMessage("user", "too late")])
    assert calls.calls == 0


def test_runner_rejects_mismatched_subagent_budget() -> None:
    clock = FakeClock()
    main_budget = InferenceTimeBudget(duration_seconds=5.0, clock=clock)
    subagent_budget = InferenceTimeBudget(duration_seconds=5.0, clock=clock)
    main = SessionBudgetLLM(
        ScriptedLLM([]),
        budget=SessionCallBudget(max_calls=2, time_budget=main_budget),
    )
    subagent_llm = SessionBudgetLLM(
        ScriptedLLM([]),
        budget=SessionCallBudget(max_calls=2, time_budget=subagent_budget),
    )
    subagent = SubAgentEngine(
        llm=subagent_llm,
        tools=ToolRegistry(),
        time_budget=subagent_budget,
    )

    with pytest.raises(ValueError, match="subagent is bound to another budget"):
        AgentSessionRunner(
            llm=main,
            tools=ToolRegistry(),
            system_prompt="mismatch",
            subagent=subagent,
            time_budget=main_budget,
        )


def test_runner_rejects_mismatched_compactor_budget() -> None:
    clock = FakeClock()
    main_budget = InferenceTimeBudget(duration_seconds=5.0, clock=clock)
    compact_budget = InferenceTimeBudget(duration_seconds=5.0, clock=clock)
    main = SessionBudgetLLM(
        ScriptedLLM([]),
        budget=SessionCallBudget(max_calls=2, time_budget=main_budget),
    )
    compact_llm = SessionBudgetLLM(
        ScriptedLLM([]),
        budget=SessionCallBudget(max_calls=2, time_budget=compact_budget),
    )

    with pytest.raises(ValueError, match="compactor is bound to another budget"):
        AgentSessionRunner(
            llm=main,
            tools=ToolRegistry(),
            system_prompt="mismatch",
            compactor=ContextCompactor(compact_llm),
            time_budget=main_budget,
        )


def test_runner_rejects_mismatched_ask_user_budget() -> None:
    clock = FakeClock()
    main_budget = InferenceTimeBudget(duration_seconds=5.0, clock=clock)
    ask_budget = InferenceTimeBudget(duration_seconds=5.0, clock=clock)
    main = SessionBudgetLLM(
        ScriptedLLM([]),
        budget=SessionCallBudget(max_calls=2, time_budget=main_budget),
    )

    with pytest.raises(ValueError, match="tool:ask_user is bound to another budget"):
        AgentSessionRunner(
            llm=main,
            tools=ToolRegistry(
                [AskUserTool(lambda _question, _summary: "ok", time_budget=ask_budget)]
            ),
            system_prompt="mismatch",
            time_budget=main_budget,
        )


def test_runner_rejects_mismatched_backtest_budget(tmp_path: Path) -> None:
    clock = FakeClock()
    main_budget = InferenceTimeBudget(duration_seconds=5.0, clock=clock)
    backtest_budget = InferenceTimeBudget(duration_seconds=5.0, clock=clock)
    output = tmp_path / "output"
    models = tmp_path / "models"
    output.mkdir()
    models.mkdir()
    (output / "main.py").write_text(
        "def generate_orders(context):\n    return []\n", encoding="utf-8"
    )

    class UnusedEvaluator:
        def evaluate(self, _request):
            raise AssertionError("budget mismatch must fail before evaluation")

    backtest = FoldBacktestTool(
        request=_fold_request(),
        output_dir=output,
        models_dir=models,
        modification_check=PassingCheck(output, models),
        artifact_store=FilesystemArtifactStore(tmp_path / "artifacts"),
        evaluator=UnusedEvaluator(),
        tree=StepTree(tmp_path / "steps"),
        schedule=StrategySchedule(),
        broker_profile=BrokerProfile(),
        time_budget=backtest_budget,
        ref_store=AgentRefStore(tmp_path / "experiment"),
    )
    main = SessionBudgetLLM(
        ScriptedLLM([]),
        budget=SessionCallBudget(max_calls=2, time_budget=main_budget),
    )

    with pytest.raises(
        ValueError, match="tool:daily_backtest is bound to another budget"
    ):
        AgentSessionRunner(
            llm=main,
            tools=ToolRegistry([backtest]),
            system_prompt="mismatch",
            time_budget=main_budget,
        )


def test_session_call_budget_role_quotas_scale_and_keep_subagent_usable() -> None:
    assert session_role_quotas(400) == (200, 50)
    assert session_role_quotas(2) == (1, 0)
    assert session_role_quotas(1) == (0, 0)
    assert session_role_quotas(8) == (4, 1)
    clock = FakeClock()
    budget = SessionCallBudget(
        max_calls=8, time_budget=InferenceTimeBudget(duration_seconds=10.0, clock=clock)
    )
    assert budget.subagent_cap == 4
    assert budget.parent_reserve == 1
    compact = SessionBudgetLLM(
        ScriptedLLM([ProviderResponse(content="c")] * 8),
        budget=budget,
        role="compact",
    )
    for _ in range(7):
        compact.complete([ChatMessage("user", "c")])
    with pytest.raises(RuntimeError, match="budget exhausted"):
        compact.complete([ChatMessage("user", "c")])
    main = SessionBudgetLLM(
        ScriptedLLM([ProviderResponse(content="m")]),
        budget=budget,
        role="main",
    )
    main.complete([ChatMessage("user", "m")])
    assert budget.calls == 8
    assert budget.main_calls == 1

    small = SessionCallBudget(
        max_calls=2, time_budget=InferenceTimeBudget(duration_seconds=10.0, clock=clock)
    )
    subagent = SessionBudgetLLM(
        ScriptedLLM([ProviderResponse(content="e")] * 2),
        budget=small,
        role="subagent",
    )
    subagent.complete([ChatMessage("user", "e")])
    with pytest.raises(RuntimeError, match="budget exhausted"):
        subagent.complete([ChatMessage("user", "e")])
    SessionBudgetLLM(
        ScriptedLLM([ProviderResponse(content="m")]),
        budget=small,
        role="main",
    ).complete([ChatMessage("user", "m")])
    assert small.calls == 2


def test_session_call_budget_claims_are_serialized() -> None:
    clock = FakeClock()
    budget = SessionCallBudget(
        max_calls=20, time_budget=InferenceTimeBudget(duration_seconds=10.0, clock=clock)
    )
    errors: list[str] = []

    def claim_many(role: str) -> None:
        for _ in range(20):
            try:
                budget.claim(role)
            except RuntimeError as exc:
                errors.append(str(exc))

    workers = [
        threading.Thread(target=claim_many, args=("subagent",)),
        threading.Thread(target=claim_many, args=("compact",)),
        threading.Thread(target=claim_many, args=("main",)),
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()
    assert budget.calls == 20
    assert budget.subagent_calls <= budget.subagent_cap
    assert budget.main_calls >= 0
    assert len(errors) == 40
