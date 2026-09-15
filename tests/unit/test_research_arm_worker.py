"""The research arm end to end through the real worker and the real PIT replay.

Only the snapshot provider is synthetic (``tests/unit/synthetic_arm.py``): the
params loader, the interactive runner, the pipeline, the deterministic
developer, the span replay, the Broker, the verdict and the terminal status
are the production code.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from autotrade.pipelines import experiment as experiment_module
from autotrade.pipelines import worker
from autotrade.pipelines.hitl_state import read_status
from autotrade.pipelines.ledger import (
    ExperimentLedger,
    experiment_verdict,
    forward_record,
    frozen_record,
    paper_candidate,
    research_records,
)
from autotrade.pipelines.worker import load_worker_options, run_local_interactive_worker
from tests.unit.synthetic_arm import (
    GEOMETRY,
    RELEASE_END,
    SyntheticPITProvider,
    decision_anchor,
    make_arm,
)
from tests.unit.test_research_session_prompt import retired_vocabulary


@pytest.fixture(autouse=True)
def synthetic_provider(monkeypatch: pytest.MonkeyPatch):
    SyntheticPITProvider.requests = []
    monkeypatch.setattr(worker, "ResearchPITSnapshotProvider", SyntheticPITProvider)
    yield SyntheticPITProvider


def _rss_mib() -> float:
    for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) / 1024
    raise RuntimeError("no VmRSS")


class _PeakRss:
    """Peak resident memory above the start while the block runs."""

    def __enter__(self):
        self.start = _rss_mib()
        self.peak = self.start
        self._stop = threading.Event()

        def sample() -> None:
            while not self._stop.wait(0.02):
                self.peak = max(self.peak, _rss_mib())

        self._thread = threading.Thread(target=sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_exc) -> None:
        self._stop.set()
        self._thread.join()
        self.peak = max(self.peak, _rss_mib())

    @property
    def growth(self) -> float:
        return self.peak - self.start


def test_an_arm_runs_research_freezes_once_and_replays_forward_and_heldout_in_one_span(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, synthetic_provider
):
    repo, experiment = make_arm(tmp_path)
    options = load_worker_options(experiment, repo_root=repo)
    measured: dict[str, float] = {}
    real_research, real_forward = (
        experiment_module.RollingExperimentPipeline.run_research_session,
        experiment_module.RollingExperimentPipeline.run_forward,
    )

    def research(self, index, **kwargs):
        with _PeakRss() as peak:
            record = real_research(self, index, **kwargs)
        measured[f"s{index}"] = peak.growth
        return record

    def forward(self, **kwargs):
        with _PeakRss() as peak:
            record = real_forward(self, **kwargs)
        measured["forward"] = peak.growth
        return record

    monkeypatch.setattr(experiment_module.RollingExperimentPipeline, "run_research_session", research)
    monkeypatch.setattr(experiment_module.RollingExperimentPipeline, "run_forward", forward)

    result = run_local_interactive_worker(options)

    records = ExperimentLedger(options.rolling.ledger_path).read()
    assert [row["record_type"] for row in records] == [
        "research_session",
        "research_session",
        "forward",
    ]
    first, second, replay = records
    assert (first["outcome"], second["outcome"]) == ("continue", "freeze")
    assert first["frozen"] is None and second["freeze_gate"]["passed"] is True
    frozen = frozen_record(records)["frozen"]
    assert Path(frozen["output_path"], "main.py").is_file()

    # Research never asked for a slot or view after research end; the forward
    # stage asked for exactly the forward and the clipped Held-out slot.
    research_end = decision_anchor(GEOMETRY["research_end"])
    requests = synthetic_provider.requests
    forward_start = next(index for index, item in enumerate(requests) if item[1] == "20240701")
    assert all(
        decision <= research_end and (not end or end <= GEOMETRY["research_end"])
        for _phase, _start, end, decision in requests[:forward_start]
    )
    assert [item[:3] for item in requests[forward_start:]] == [
        ("heldout", "20240701", "20250630"),
        ("heldout", "20250701", RELEASE_END),
    ]

    # One continuous book over both slices.
    assert replay["status"] == "ok"
    result_record = json.loads(Path(replay["result_ref"]).read_text(encoding="utf-8"))
    assert result_record["pit"]["replay_slots"] == [
        "20240701_20250630_20240630T235959+0800",
        f"20250701_{RELEASE_END}_20250630T235959+0800",
    ]
    assert result_record["equity_curve"][0]["trade_date"] == "20240701"
    assert result_record["equity_curve"][-1]["trade_date"] == RELEASE_END
    forward_slice = replay["slices"]["forward"]
    heldout_slice = replay["slices"]["heldout"]
    assert forward_slice["activity"]["round_trips"] >= 12
    assert heldout_slice["activity"]["trade_days"] < forward_slice["activity"]["trade_days"]
    assert replay["verdict"]["status"] == "graduated", replay["verdict"]["reasons"]

    verdict = experiment_verdict(records)
    assert verdict == {"status": "graduated", "reasons": []}
    assert paper_candidate(records)["artifact_id"] == frozen["artifact_id"]
    assert result["state"] == "completed"
    assert result["verdict"] == verdict
    assert result["final_strategy_artifact"] == frozen["artifact_id"]
    assert read_status(experiment / "hitl" / "status.json")["verdict"] == verdict
    plan = json.loads((experiment / "hitl" / "schedule.json").read_text(encoding="utf-8"))
    assert [row["session_key"] for row in plan["sessions"]] == ["s1", "s2", "forward"]
    assert plan["sessions"][-1]["replay"]["replay_end"] == RELEASE_END

    # Replay memory by stage: a research-year replay and the forward span.
    print(json.dumps({"peak_rss_growth_mib": measured}))
    assert set(measured) == {"s1", "s2", "forward"}

    # A resume after the verdict republishes the terminal status and runs nothing.
    before = ExperimentLedger(options.rolling.ledger_path).read()
    synthetic_provider.requests.clear()
    again = run_local_interactive_worker(load_worker_options(experiment, repo_root=repo))
    assert again["verdict"] == verdict
    assert ExperimentLedger(options.rolling.ledger_path).read() == before
    assert synthetic_provider.requests == []


def test_a_session_that_crashes_is_resumed_without_rerunning_the_recorded_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    repo, experiment = make_arm(tmp_path, session_max_attempts=1)
    real = experiment_module.RollingExperimentPipeline.run_research_session
    calls: list[int] = []

    def crash_in_s2(self, index, **kwargs):
        calls.append(index)
        if index == 2 and calls.count(2) == 1:
            raise RuntimeError("session container died")
        return real(self, index, **kwargs)

    monkeypatch.setattr(experiment_module.RollingExperimentPipeline, "run_research_session", crash_in_s2)
    with pytest.raises(RuntimeError, match="session container died"):
        run_local_interactive_worker(load_worker_options(experiment, repo_root=repo))
    ledger = ExperimentLedger(experiment / "ledgers" / "experiment_ledger.jsonl")
    assert [row["session_key"] for row in research_records(ledger.read())] == ["s1"]

    result = run_local_interactive_worker(load_worker_options(experiment, repo_root=repo))
    assert calls == [1, 2, 2]
    assert [row["session_key"] for row in research_records(ledger.read())] == ["s1", "s2"]
    assert result["verdict"]["status"] == "graduated"


def test_a_worker_stopped_after_the_freeze_resumes_with_the_forward_replay_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    repo, experiment = make_arm(tmp_path, session_max_attempts=1)
    real_forward = experiment_module.RollingExperimentPipeline.run_forward
    state = {"forward_calls": 0}

    def forward_fails_once(self, **kwargs):
        state["forward_calls"] += 1
        if state["forward_calls"] == 1:
            raise OSError("replay host disk vanished")
        return real_forward(self, **kwargs)

    monkeypatch.setattr(experiment_module.RollingExperimentPipeline, "run_forward", forward_fails_once)
    with pytest.raises(OSError):
        run_local_interactive_worker(load_worker_options(experiment, repo_root=repo))
    ledger = ExperimentLedger(experiment / "ledgers" / "experiment_ledger.jsonl")
    assert frozen_record(ledger.read()) is not None
    assert forward_record(ledger.read()) is None

    research = experiment_module.RollingExperimentPipeline.run_research_session

    def no_more_research(self, index, **kwargs):
        raise AssertionError(f"research session {index} re-ran after the freeze")

    monkeypatch.setattr(experiment_module.RollingExperimentPipeline, "run_research_session", no_more_research)
    result = run_local_interactive_worker(load_worker_options(experiment, repo_root=repo))
    assert state["forward_calls"] == 2
    assert result["verdict"]["status"] == "graduated"
    assert research is not no_more_research


def test_a_failed_replay_attempt_is_recorded_and_retried_from_forward_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    repo, experiment = make_arm(tmp_path, session_max_attempts=3)
    options = load_worker_options(experiment, repo_root=repo)
    from autotrade.pipelines.pit_backend import PITDailyEvaluationBackend

    real_evaluate = PITDailyEvaluationBackend.evaluate
    spans: list[tuple[str, str]] = []

    def flaky_forward(self, request, **kwargs):
        if request.mode == "heldout":
            spans.append((request.start, request.end))
            if len(spans) == 1:
                raise TimeoutError("strategy inference exceeded 360s")
        return real_evaluate(self, request, **kwargs)

    monkeypatch.setattr(PITDailyEvaluationBackend, "evaluate", flaky_forward)
    result = run_local_interactive_worker(options)

    assert spans == [("20240701", RELEASE_END)] * 2
    rows = ExperimentLedger(options.rolling.ledger_path).read()
    failed = [row for row in rows if row["record_type"] == "attempt_failed"]
    assert [(row["phase"], row["error"]) for row in failed] == [
        ("forward", "TimeoutError: strategy inference exceeded 360s")
    ]
    assert forward_record(rows)["status"] == "ok"
    assert result["verdict"]["status"] == "graduated"


def test_a_forward_replay_that_keeps_failing_fails_the_experiment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    repo, experiment = make_arm(tmp_path, session_max_attempts=2)
    from autotrade.pipelines.pit_backend import PITDailyEvaluationBackend

    real_evaluate = PITDailyEvaluationBackend.evaluate

    def broken_forward(self, request, **kwargs):
        if request.mode == "heldout":
            raise TimeoutError("strategy inference exceeded 360s")
        return real_evaluate(self, request, **kwargs)

    monkeypatch.setattr(PITDailyEvaluationBackend, "evaluate", broken_forward)
    with pytest.raises(TimeoutError):
        run_local_interactive_worker(load_worker_options(experiment, repo_root=repo))
    rows = ExperimentLedger(experiment / "ledgers" / "experiment_ledger.jsonl").read()
    assert [row["phase"] for row in rows if row["record_type"] == "attempt_failed"] == [
        "forward",
        "forward",
    ]
    assert experiment_verdict(rows) is None
    assert read_status(experiment / "hitl" / "status.json")["state"] == "failed"


def test_an_arm_whose_research_ends_without_a_freeze_runs_no_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, synthetic_provider
):
    repo, experiment = make_arm(tmp_path, research_sessions=1)
    from autotrade.pipelines import local_backend

    real_call = local_backend.DeterministicBaselineDeveloper.__call__

    def ends_without_edge(self, request):
        result = real_call(self, request)
        return type(result)(
            result.conversation_id,
            result.steps,
            "no_edge",
            reason="the pack's termination rule fired",
        )

    monkeypatch.setattr(local_backend.DeterministicBaselineDeveloper, "__call__", ends_without_edge)
    result = run_local_interactive_worker(load_worker_options(experiment, repo_root=repo))
    assert result["verdict"] == {
        "status": "no_deliverable",
        "reasons": ["no_edge: the pack's termination rule fired"],
    }
    assert all(item[0] != "heldout" for item in synthetic_provider.requests)
    rows = ExperimentLedger(experiment / "ledgers" / "experiment_ledger.jsonl").read()
    assert [row["record_type"] for row in rows] == ["research_session"]


def test_a_fold_era_ledger_is_refused(tmp_path: Path):
    from autotrade.environment.identity import AgentRefStore

    repo, experiment = make_arm(tmp_path)
    AgentRefStore(experiment)
    ledger = ExperimentLedger(experiment / "ledgers" / "experiment_ledger.jsonl")
    ledger.append(
        {
            "record_type": "fold",
            "experiment_id": "arm",
            "epoch_id": "epoch_001",
            "fold_id": "fold_2024",
            "run_id": "run_old",
        }
    )
    with pytest.raises(ValueError, match="Fold-era ledger"):
        run_local_interactive_worker(load_worker_options(experiment, repo_root=repo))


def test_the_worker_refuses_a_release_that_does_not_reach_heldout(tmp_path: Path):
    repo, experiment = make_arm(tmp_path, heldout_end="20251231", forward_end="20250630")
    params_path = experiment / "hitl" / "params.json"
    params = json.loads(params_path.read_text(encoding="utf-8"))
    params.update({"research_end": "20250630", "forward_end": "20260630", "heldout_end": "20260930"})
    params_path.write_text(json.dumps(params), encoding="utf-8")
    with pytest.raises(ValueError, match="Held-out"):
        load_worker_options(experiment, repo_root=repo)


def test_llm_research_sessions_mount_only_the_research_end_view_and_hand_off_prior(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, synthetic_provider
):
    """Two scripted Agent sessions through the real Agent adapter.

    The first validates the working copy, leaves a PRIOR.md and continues;
    the second reads that PRIOR, validates again and freezes its node through
    the gate, which then runs forward. Each session's
    ``/mnt/snapshot`` is the decision view at research end, the two snapshot
    slot mounts stay empty, and no replay slot appears anywhere in the
    session's runtime tree. A graduated sibling experiment's skills are
    mounted as operating memory and named in the run manifest.
    """

    from autotrade.environment.llm import ProviderResponse, ToolCall
    from autotrade.pipelines.hitl_state import ControlState, read_control, write_control
    from tests.unit.test_interactive_worker_local import (
        LAST_WORKING_COPY_NODE,
        VALIDATE_WORKING_COPY,
        _agent_then,
        _NominatingLLM,
        _NoShellRunner,
    )
    from tests.unit.test_operating_memory import GRADUATED_SKILL, _experiment_with_skill

    repo, experiment = make_arm(tmp_path, developer_mode="llm", max_replay_years_per_session=1)
    _experiment_with_skill(repo / "experiments", "adopted")
    monkeypatch.setenv("VLLM_API_KEY", "local-test-key")
    # The console's per-session GPU allocation, one-shot, for s1 only.
    write_control(experiment / "hitl" / "control.json", ControlState(mode="auto", gpu_counts={"s1": 3}))
    options = load_worker_options(experiment, repo_root=repo)
    handoff = "动量腿在研究期稳定，下一会话复核同一机制并决定是否冻结。"
    llm = _NominatingLLM(
        [
            *_agent_then(
                ToolCall("prior", "write_file", {"path": "PRIOR.md", "content": handoff}),
                VALIDATE_WORKING_COPY,
                ToolCall(
                    "finish",
                    "finish_session",
                    {
                        "outcome": "continue",
                        "reason": "本会话只完成了一次整个研究期的验证，边际是否稳定还需要下一会话独立复核同一机制后再判断，因此不在本会话冻结。",
                    },
                ),
                roles=(),
            ),
            *_agent_then(VALIDATE_WORKING_COPY, roles=()),
            ProviderResponse(
                tool_calls=(
                    ToolCall(
                        "finish",
                        "finish_session",
                        {"outcome": "freeze", "node_id": LAST_WORKING_COPY_NODE},
                    ),
                )
            ),
        ]
    )
    result = run_local_interactive_worker(
        options, llm=llm, command_runner_factory=lambda _workspace: _NoShellRunner()
    )

    records = ExperimentLedger(options.rolling.ledger_path).read()
    first, second, replay = records
    assert (first["outcome"], second["outcome"]) == ("continue", "freeze")
    assert first["prior"] == handoff and first["prior_published"] is True
    assert second["frozen"] is not None
    assert replay["record_type"] == "forward"
    assert result["verdict"]["status"] == "graduated"
    second_prompts = [
        message.content or ""
        for call in llm.calls
        for message in call["messages"]
        if message.role == "system"
    ]
    assert any(handoff in prompt for prompt in second_prompts)
    # The console preview of s2 is the prompt s2 received, runtime facts aside.
    from autotrade.webui.prompt_preview import RUNTIME_PLACEHOLDER, build_prompt_preview
    from tests.unit.test_webui_prompt_preview import _facts

    received = _facts(next(prompt for prompt in reversed(second_prompts) if handoff in prompt))
    previewed = _facts(str(build_prompt_preview(experiment, "s2", "", repo_root=repo)["prompt"]))
    assert set(previewed) == set(received)
    for block in ("budgets", "research_scope", "visibility_policy", "broker_replay", "arm"):
        assert set(previewed[block]) == set(received[block]), block
    assert previewed["research_scope"] == received["research_scope"]
    assert previewed["research_geometry"] == received["research_geometry"]
    assert previewed["identity"]["session"] == received["identity"]["session"]
    # s1 continued without naming a node, so s2 starts from the template too.
    assert {key: value for key, value in previewed["artifact_contract"]["start"].items() if key != "model_artifacts_empty"} == {
        key: value for key, value in received["artifact_contract"]["start"].items() if key != "model_artifacts_empty"
    } == {"kind": "template", "template_ref": "agent_output_template"}
    assert previewed["earlier_sessions"] == received["earlier_sessions"]
    assert {key: value for key, value in previewed["budgets"].items() if key != "context_compaction"} == {
        key: value for key, value in received["budgets"].items() if key != "context_compaction"
    }
    assert previewed["identity"]["run_id"] == RUNTIME_PLACEHOLDER

    specs = [
        json.loads(
            (Path(record["run_manifest_ref"]).parent / "host_run_manifest.json").read_text(encoding="utf-8")
        )["sandbox_spec"]
        for record in (first, second)
    ]
    assert (specs[0]["gpu_count"], specs[0]["gpu"], specs[0]["gpu_name_filter"]) == (3, "auto", "L20")
    assert specs[1]["gpu_count"] == options.agent_sandbox.gpu_count
    memory = json.loads(Path(first["run_manifest_ref"]).read_text(encoding="utf-8"))["operating_memory"]
    assert {"source": "adopted", "origin": "graduated", "entries": [GRADUATED_SKILL]} in memory["sources"]
    assert read_control(experiment / "hitl" / "control.json").gpu_counts == {}

    research_anchor = decision_anchor(GEOMETRY["research_end"]).isoformat()
    for record in (first, second):
        root = options.work_root / options.experiment_id / str(record["run_id"])
        view = json.loads((root / "runtime" / "current_snapshot" / "manifest.json").read_text(encoding="utf-8"))
        assert (view["kind"], view["decision_time"]) == ("decision_input", research_anchor)
        assert not any((root / "snapshots" / "train").iterdir())
        assert not any((root / "snapshots" / "valid").iterdir())
        manifests = [
            json.loads(path.read_text(encoding="utf-8"))
            for path in root.rglob("manifest.json")
            if path.is_file()
        ]
        assert all(item.get("kind") != "replay_slot" for item in manifests)
    for step in first["steps"] + second["steps"]:
        replayed = json.loads(Path(step["validation_result_ref"]).read_text(encoding="utf-8"))
        assert all(
            name.split("_")[1] <= GEOMETRY["research_end"]
            for name in replayed["pit"]["replay_slots"]
        )
    # Everything the model read, tool schemas included, is research-session
    # vocabulary and names no date after research end.
    read = _model_input(llm)
    assert retired_vocabulary(read) == []
    for later in ("20240701", "2024-07-01", GEOMETRY["forward_end"], GEOMETRY["heldout_end"], RELEASE_END):
        assert later not in read


def _model_input(llm) -> str:
    """Every message and tool schema a scripted model was sent, as one text."""

    return "\n".join(
        [message.content or "" for call in llm.calls for message in call["messages"]]
        + [json.dumps(call["tools"], ensure_ascii=False) for call in llm.calls]
    )


def test_llm_sessions_validate_a_multi_year_span_meet_the_gate_continue_and_end_the_arm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, synthetic_provider
):
    """Every finish outcome except a freeze, through the real worker.

    On a two-year research period, s1 validates its working copy on Y2 and on
    the full span, which replays both research years as one book; a freeze of
    the full-span node is refused by the Pipeline's own gate with its named
    reason, so s1 continues from that node. s2 starts from the node s1 handed
    on, validates it again and ends the arm with no_edge: the arm has no
    deliverable and nothing is replayed after research end.
    """

    from autotrade.environment.llm import ProviderResponse, ToolCall
    from tests.unit.test_interactive_worker_local import (
        LAST_WORKING_COPY_NODE,
        _NominatingLLM,
        _NoShellRunner,
    )

    def validate(span: str) -> ToolCall:
        return ToolCall(
            f"valid_{span}",
            "batch_validate",
            {
                "span": span,
                "candidates": [
                    {
                        "name": "working_copy",
                        "hypothesis": f"the working copy earns a positive neutralized excess over {span}",
                        "path": "output",
                    }
                ],
            },
        )

    def finish(**arguments: object) -> ProviderResponse:
        return ProviderResponse(tool_calls=(ToolCall("finish", "finish_session", dict(arguments)),))

    reason = "两个研究年里只有一次完整研究期验证，冻结门要求至少两次，交给下一会话在同一机制上补足对照后再判断。"
    repo, experiment = make_arm(
        tmp_path,
        developer_mode="llm",
        research_start="20220701",
        max_replay_years_per_session=6,
    )
    monkeypatch.setenv("VLLM_API_KEY", "local-test-key")
    options = load_worker_options(experiment, repo_root=repo)
    llm = _NominatingLLM(
        [
            ProviderResponse(tool_calls=(validate("Y2"), validate("full"))),
            finish(outcome="freeze", node_id=LAST_WORKING_COPY_NODE),
            finish(outcome="continue", node_id=LAST_WORKING_COPY_NODE, reason=reason),
            ProviderResponse(tool_calls=(validate("full"),)),
            finish(outcome="no_edge", reason="完整研究期上复核后，本机制没有稳定的中性化超额，参考包的终止条件已满足，结束本臂。"),
        ]
    )
    result = run_local_interactive_worker(
        options, llm=llm, command_runner_factory=lambda _workspace: _NoShellRunner()
    )

    records = ExperimentLedger(options.rolling.ledger_path).read()
    assert [row["record_type"] for row in records] == ["research_session", "research_session"]
    first, second = records
    assert [step["span"] for step in first["steps"]] == ["Y2", "full"]
    full_node = first["steps"][1]["step_id"]
    assert (first["outcome"], first["next_start_node_id"], first["frozen"]) == ("continue", full_node, None)
    assert (second["start_node_id"], second["outcome"]) == (full_node, "no_edge")
    assert result["verdict"] == {"status": "no_deliverable", "reasons": [f"no_edge: {second['reason']}"]}
    assert all(item[0] != "heldout" for item in synthetic_provider.requests)

    # The refusal the model read names the gate's reason and its numbers.
    refusal = next(
        message.content
        for call in llm.calls
        for message in call["messages"]
        if "freeze_gate_refused" in (message.content or "")
    )
    assert "freeze_too_few_full_span_validations" in refusal and "full_span_validations=1" in refusal

    # Y2 replayed one slot; the full span replayed Y1 and then Y2 as one book.
    year, full = (
        json.loads(Path(step["validation_result_ref"]).read_text(encoding="utf-8")) for step in first["steps"]
    )
    assert [name.split("_")[:2] for name in year["pit"]["replay_slots"]] == [["20230701", "20240630"]]
    assert [name.split("_")[:2] for name in full["pit"]["replay_slots"]] == [
        ["20220701", "20230630"],
        ["20230701", "20240630"],
    ]
    assert full["equity_curve"][0]["trade_date"] == "20220701"
    assert full["equity_curve"][-1]["trade_date"] <= GEOMETRY["research_end"]
    assert [row["label"] for row in full["stats"]["sub_windows"]] == ["202207-202306", "202307-202406"]

    # s2's facts: the last of two sessions, started from s1's node, with s1's outcome.
    s2_system = next(
        message.content
        for call in reversed(llm.calls)
        for message in call["messages"]
        if message.role == "system" and "earlier_sessions" in (message.content or "")
    )
    facts = json.loads(s2_system.split("```json\n", 1)[1].split("\n```", 1)[0])
    assert facts["identity"]["session"] == {"index": 2, "of": 2, "last": True}
    assert facts["artifact_contract"]["start"]["node_id"] == full_node
    assert [year["label"] for year in facts["research_geometry"]["years"]] == ["Y1", "Y2"]
    assert facts["research_geometry"]["research_period"] == "20220701..20240630"
    assert facts["arm"]["trials_to_date"] == 2 and facts["arm"]["full_span_validations_to_date"] == 1
    assert [row["outcome"] for row in facts["earlier_sessions"]] == ["continue"]

    read = _model_input(llm)
    assert retired_vocabulary(read) == []
    for later in ("20240701", "2024-07-01", GEOMETRY["forward_end"], GEOMETRY["heldout_end"], RELEASE_END):
        assert later not in read
