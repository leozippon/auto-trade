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

from autotrade.environment.runtime import append_versioned_jsonl
from autotrade.pipelines import experiment as experiment_module
from autotrade.pipelines import worker
from autotrade.pipelines.hitl_state import read_status
from autotrade.pipelines.ledger import (
    LEDGER_RECORD_SCHEMA_VERSION,
    ExperimentLedger,
    experiment_verdict,
    forward_record,
    frozen_record,
    paper_candidate,
    research_records,
)
from autotrade.pipelines.revision_history import revision_history
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

    def research(self, **kwargs):
        with _PeakRss() as peak:
            record = real_research(self, **kwargs)
        measured["research"] = peak.growth
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
    assert [row["record_type"] for row in records] == ["research_session", "forward"]
    research_row, replay = records
    assert research_row["outcome"] == "freeze"
    assert research_row["freeze_gate"]["passed"] is True
    assert [step["span"] for step in research_row["steps"]] == ["full", "full"]
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
    assert [row["session_key"] for row in plan["sessions"]] == ["research", "forward"]
    assert plan["sessions"][-1]["replay"]["replay_end"] == RELEASE_END

    # Replay memory by stage: the research replays and the forward span.
    print(json.dumps({"peak_rss_growth_mib": measured}))
    assert set(measured) == {"research", "forward"}

    # A resume after the verdict republishes the terminal status and runs nothing.
    before = ExperimentLedger(options.rolling.ledger_path).read()
    synthetic_provider.requests.clear()
    again = run_local_interactive_worker(load_worker_options(experiment, repo_root=repo))
    assert again["verdict"] == verdict
    assert ExperimentLedger(options.rolling.ledger_path).read() == before
    assert synthetic_provider.requests == []


def test_a_crashed_research_attempt_is_run_again_by_the_next_worker_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    repo, experiment = make_arm(tmp_path, session_max_attempts=1)
    real = experiment_module.RollingExperimentPipeline.run_research_session
    calls: list[int] = []

    def crash_once(self, **kwargs):
        calls.append(len(calls) + 1)
        if len(calls) == 1:
            raise RuntimeError("session container died")
        return real(self, **kwargs)

    monkeypatch.setattr(experiment_module.RollingExperimentPipeline, "run_research_session", crash_once)
    with pytest.raises(RuntimeError, match="session container died"):
        run_local_interactive_worker(load_worker_options(experiment, repo_root=repo))
    ledger = ExperimentLedger(experiment / "ledgers" / "experiment_ledger.jsonl")
    assert research_records(ledger.read()) == []

    result = run_local_interactive_worker(load_worker_options(experiment, repo_root=repo))
    assert calls == [1, 2]
    assert [row["session_key"] for row in research_records(ledger.read())] == ["research"]
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

    def no_more_research(self, **kwargs):
        raise AssertionError("the research session re-ran after the freeze")

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
    repo, experiment = make_arm(tmp_path)
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
    archived = {
        "record_type": "fold",
        "experiment_id": "arm",
        "epoch_id": "epoch_001",
        "fold_id": "fold_2024",
        "run_id": "run_old",
    }
    with pytest.raises(ValueError, match="unsupported record_type"):
        ledger.append(archived)
    append_versioned_jsonl(ledger.path, archived, schema_version=LEDGER_RECORD_SCHEMA_VERSION)
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


def test_the_llm_research_session_mounts_only_the_research_end_view_and_freezes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, synthetic_provider
):
    """One scripted Agent session through the real Agent adapter.

    It validates the working copy twice (the gate needs two full-span
    validations) and freezes the second node through the gate, which then
    runs forward. The session's ``/mnt/snapshot`` is the decision view at
    research end, the two snapshot slot mounts stay empty, and no replay slot
    appears anywhere in the session's runtime tree. A graduated sibling
    experiment's skills are mounted as operating memory and named in the run
    manifest.
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

    repo, experiment = make_arm(tmp_path, developer_mode="llm", max_replay_years=2)
    _experiment_with_skill(repo / "experiments", "adopted")
    monkeypatch.setenv("VLLM_API_KEY", "local-test-key")
    # The console's GPU allocation for the research session, one-shot.
    write_control(experiment / "hitl" / "control.json", ControlState(mode="auto", gpu_counts={"research": 3}))
    options = load_worker_options(experiment, repo_root=repo)
    llm = _NominatingLLM(
        [
            *_agent_then(VALIDATE_WORKING_COPY, roles=()),
            ProviderResponse(tool_calls=(VALIDATE_WORKING_COPY,)),
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
    research_row, replay = records
    assert research_row["outcome"] == "freeze"
    assert research_row["frozen"] is not None
    assert research_row["finish_reason"] == "llm_agent_finish_session"
    assert replay["record_type"] == "forward"
    assert result["verdict"]["status"] == "graduated"
    system_prompts = [
        message.content or ""
        for call in llm.calls
        for message in call["messages"]
        if message.role == "system"
    ]
    # The console preview is the prompt the session received, runtime facts aside.
    from autotrade.webui.prompt_preview import RUNTIME_PLACEHOLDER, build_prompt_preview
    from tests.unit.test_webui_prompt_preview import _facts

    received = _facts(system_prompts[0])
    previewed = _facts(str(build_prompt_preview(experiment, "research", "", repo_root=repo)["prompt"]))
    assert set(previewed) == set(received)
    for block in ("budgets", "research_scope", "visibility_policy", "broker_replay", "arm"):
        assert set(previewed[block]) == set(received[block]), block
    assert previewed["research_scope"] == received["research_scope"]
    assert previewed["research_geometry"] == received["research_geometry"]
    assert "session" not in received["identity"]
    assert previewed["artifact_contract"]["start"] == received["artifact_contract"]["start"] == {
        "kind": "template",
        "template_ref": "agent_output_template",
        "model_artifacts_empty": True,
    }
    assert {key: value for key, value in previewed["budgets"].items() if key != "context_compaction"} == {
        key: value for key, value in received["budgets"].items() if key != "context_compaction"
    }
    assert previewed["identity"]["run_id"] == RUNTIME_PLACEHOLDER

    spec = json.loads(
        (Path(research_row["run_manifest_ref"]).parent / "host_run_manifest.json").read_text(encoding="utf-8")
    )["sandbox_spec"]
    assert (spec["gpu_count"], spec["gpu"], spec["gpu_name_filter"]) == (3, "auto", "L20")
    memory = json.loads(Path(research_row["run_manifest_ref"]).read_text(encoding="utf-8"))["operating_memory"]
    assert {"source": "adopted", "origin": "graduated", "entries": [GRADUATED_SKILL]} in memory["sources"]
    assert read_control(experiment / "hitl" / "control.json").gpu_counts == {}

    # The session's transcript, named by the run's opaque ref, carries what
    # the Agent did and nothing host-side; the JSONL trace stays under traces/.
    [transcript] = (experiment / "artifacts" / "transcripts").glob("run_ref_*.txt")
    transcript_text = transcript.read_text(encoding="utf-8")
    assert "tool_call call=" in transcript_text and "batch_validate" in transcript_text
    assert str(research_row["run_id"]) not in transcript_text
    assert "/Data2" not in transcript_text and str(options.work_root) not in transcript_text
    assert research_row["agent_trace_ref"].endswith(f"{research_row['run_id']}.jsonl")

    research_anchor = decision_anchor(GEOMETRY["research_end"]).isoformat()
    root = options.work_root / options.experiment_id / "research"
    view = json.loads((root / "runtime" / "current_snapshot" / "manifest.json").read_text(encoding="utf-8"))
    assert (view["kind"], view["decision_time"]) == ("decision_input", research_anchor)
    assert not (root / "snapshots").exists()
    manifests = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in root.rglob("manifest.json")
        if path.is_file()
    ]
    assert all(item.get("kind") != "replay_slot" for item in manifests)
    for step in research_row["steps"]:
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


def _resume_script_summary() -> str:
    return (
        "## 策略现状\noutput/ 是模板策略，已完成一次完整研究期验证（working_copy）。\n## 决定\n"
        "- 该验证的中性化超额为正，作为冻结门的第一个完整研究期验证保留；节点 id 见 trace 第 1 次调用的 "
        "batch_validate 结果。\n## 线索\n- 下一步再验证一次 working_copy 补足冻结门要求的两个完整研究期验证，"
        "然后提名冻结。\n## trace\n- grep 'working_copy' 可找回节点 id 与读数。"
    )


def test_an_interrupted_llm_session_resumes_with_its_summary_budget_and_nodes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, synthetic_provider
):
    """The real worker, twice. The first attempt validates the working copy,
    compacts its context and then loses its model (the script runs out):
    the attempt is recorded as failed, its node, revision, workspace and
    transcript stay. The second worker start resumes the same session: it
    starts from the compaction summary and a note, continues the budget,
    validates once more and freezes; the record carries both attempts' steps."""

    from autotrade.environment.llm import ProviderResponse, ToolCall
    from tests.unit.test_interactive_worker_local import (
        LAST_WORKING_COPY_NODE,
        VALIDATE_WORKING_COPY,
        _agent_then,
        _NominatingLLM,
        _NoShellRunner,
    )

    repo, experiment = make_arm(
        tmp_path, developer_mode="llm", max_replay_years=2, session_max_attempts=1, max_llm_calls=40
    )
    monkeypatch.setenv("VLLM_API_KEY", "local-test-key")
    options = load_worker_options(experiment, repo_root=repo)
    summary = _resume_script_summary()
    first_llm = _NominatingLLM(
        [
            *_agent_then(VALIDATE_WORKING_COPY, roles=()),
            ProviderResponse(tool_calls=(ToolCall("c", "compact", {"summary": summary}),)),
            # The script ends: three consecutive model failures end the attempt.
        ]
    )
    with pytest.raises(RuntimeError, match="language model unavailable"):
        run_local_interactive_worker(
            options, llm=first_llm, command_runner_factory=lambda _workspace: _NoShellRunner()
        )
    ledger = ExperimentLedger(options.rolling.ledger_path)
    [failed] = ledger.read()
    assert failed["record_type"] == "attempt_failed" and failed["session_key"] == "research"
    session_root = options.work_root / options.experiment_id / "research"
    assert (session_root / "agent" / "workspace" / "output" / "main.py").is_file()
    tree = json.loads((experiment / "steps" / "tree.json").read_text(encoding="utf-8"))
    [node] = [item for item in tree["nodes"] if item.get("complete_validation")]
    assert (experiment / ".host" / "steps" / f"{node['node_id']}.json").is_file()
    assert len(list((experiment / "artifacts" / "strategy" / "revisions").iterdir())) == 1
    transcripts = sorted((experiment / "artifacts" / "transcripts").glob("run_ref_*.txt"))
    assert len(transcripts) == 1 and "compact" in transcripts[0].read_text(encoding="utf-8")

    second_llm = _NominatingLLM(
        [
            ProviderResponse(tool_calls=(VALIDATE_WORKING_COPY,)),
            ProviderResponse(
                tool_calls=(
                    ToolCall("finish", "finish_session", {"outcome": "freeze", "node_id": LAST_WORKING_COPY_NODE}),
                )
            ),
        ]
    )
    result = run_local_interactive_worker(
        load_worker_options(experiment, repo_root=repo),
        llm=second_llm,
        command_runner_factory=lambda _workspace: _NoShellRunner(),
    )

    records = ledger.read()
    assert [row["record_type"] for row in records] == ["attempt_failed", "research_session", "forward"]
    record = records[1]
    assert record["attempts"] == 2 and record["outcome"] == "freeze"
    assert record["steps"][0]["step_id"] == node["node_id"]
    assert record["trials_to_date"] == 2 and record["freeze_gate"]["passed"] is True
    assert record["budget_used"]["replay_years"] == 2
    # The calls of both attempts: the first spent five (two scripted, three failed).
    assert record["budget_used"]["llm_calls"] == 5 + len(second_llm.calls)
    assert result["verdict"]["status"] == "graduated"
    # The resumed attempt opened with the interrupted one's summary, then the note.
    opening = second_llm.calls[0]["messages"]
    assert opening[0].role == "system"
    checkpoint = json.loads(opening[1].content)
    assert checkpoint["observation"] == "context_compaction"
    assert checkpoint["summary_kind"] == "resume" and checkpoint["summary"] == summary
    assert checkpoint["trace"]["root"] == "trace"
    note = opening[2].content
    assert "第 2 次尝试" in note and "language model unavailable" in note
    assert "回放年 1/2" in note and transcripts[0].name in note
    assert "上面是中断前最近一次压缩的摘要" in note
    facts = json.loads(opening[0].content.split("```json\n", 1)[1].split("\n```", 1)[0])
    assert facts["arm"] == {"frozen": False, "freezes_per_arm": 1, "trials_to_date": 1, "full_span_validations_to_date": 1}
    assert facts["budgets"]["used_before_this_attempt"]["replay_years"] == 1
    assert facts["budgets"]["used_before_this_attempt"]["llm_calls"] == 5
    # One session root, two transcripts, and both attempts' revisions kept as
    # the arm's artifact history: the resumed attempt's revision descends from
    # the node it continued out of, and each is joined to the Step it validated.
    assert sorted(path.name for path in (experiment / "artifacts" / "transcripts").glob("run_ref_*.txt")) != [transcripts[0].name]
    history = revision_history(experiment)["revisions"]
    assert [row["parent_revision_id"] for row in history] == [None, history[0]["revision_id"]]
    assert [row["node_id"] for row in history] == [node["node_id"], record["steps"][1]["step_id"]]


def test_an_interrupted_session_without_a_checkpoint_resumes_from_the_note_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, synthetic_provider
):
    """The first attempt loses its model before it validated or compacted
    anything. The resumed attempt has no summary to start from: it opens with
    the note alone (which says so), continues the call budget, and ends the
    arm honestly."""

    from autotrade.environment.llm import ProviderResponse, ToolCall
    from tests.unit.test_interactive_worker_local import (
        VALIDATE_WORKING_COPY,
        _NominatingLLM,
        _NoShellRunner,
    )

    repo, experiment = make_arm(
        tmp_path, developer_mode="llm", max_replay_years=2, session_max_attempts=1, max_llm_calls=40
    )
    monkeypatch.setenv("VLLM_API_KEY", "local-test-key")
    options = load_worker_options(experiment, repo_root=repo)
    with pytest.raises(RuntimeError, match="language model unavailable"):
        run_local_interactive_worker(
            options, llm=_NominatingLLM([]), command_runner_factory=lambda _workspace: _NoShellRunner()
        )
    ledger = ExperimentLedger(options.rolling.ledger_path)
    [failed] = ledger.read()
    assert failed["record_type"] == "attempt_failed"
    assert not (experiment / "steps" / "tree.json").is_file()

    reason = "neutralized excess is negative in three of four research years; the null percentile is 0.48"
    second_llm = _NominatingLLM(
        [
            ProviderResponse(tool_calls=(VALIDATE_WORKING_COPY,)),
            ProviderResponse(
                tool_calls=(ToolCall("finish", "finish_session", {"outcome": "no_edge", "reason": reason}),)
            ),
        ]
    )
    result = run_local_interactive_worker(
        load_worker_options(experiment, repo_root=repo),
        llm=second_llm,
        command_runner_factory=lambda _workspace: _NoShellRunner(),
    )

    records = ledger.read()
    assert [row["record_type"] for row in records] == ["attempt_failed", "research_session"]
    record = records[1]
    assert record["attempts"] == 2 and record["outcome"] == "no_edge"
    assert record["arm_end"] == {"status": "no_deliverable", "reason": f"no_edge: {reason}"}
    assert record["trials_to_date"] == 1
    # Three failed calls of the first attempt, then this attempt's.
    assert record["budget_used"]["llm_calls"] == 3 + len(second_llm.calls)
    assert result["verdict"]["status"] == "no_deliverable"
    opening = second_llm.calls[0]["messages"]
    assert [message.role for message in opening] == ["system", "user"]
    note = opening[1].content
    assert "第 2 次尝试" in note and "language model unavailable" in note
    assert "中断前没有压缩摘要" in note and "模型调用 3/40" in note
    facts = json.loads(opening[0].content.split("```json\n", 1)[1].split("\n```", 1)[0])
    assert facts["budgets"]["used_before_this_attempt"]["llm_calls"] == 3
    assert facts["arm"]["trials_to_date"] == 0


def test_a_resume_without_the_interrupted_workspace_fails_the_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, synthetic_provider
):
    from autotrade.environment.llm import ProviderResponse, ToolCall
    from autotrade.pipelines.local_backend import _remove_mounted_tree
    from tests.unit.test_interactive_worker_local import _NominatingLLM, _NoShellRunner

    repo, experiment = make_arm(tmp_path, developer_mode="llm", max_replay_years=2, session_max_attempts=1)
    monkeypatch.setenv("VLLM_API_KEY", "local-test-key")
    options = load_worker_options(experiment, repo_root=repo)
    with pytest.raises(RuntimeError, match="language model unavailable"):
        run_local_interactive_worker(
            options, llm=_NominatingLLM([]), command_runner_factory=lambda _workspace: _NoShellRunner()
        )
    _remove_mounted_tree(options.work_root / options.experiment_id / "research")

    with pytest.raises(RuntimeError, match="workspace of the interrupted attempt is missing"):
        run_local_interactive_worker(
            load_worker_options(experiment, repo_root=repo),
            llm=_NominatingLLM([ProviderResponse(tool_calls=(ToolCall("f", "finish_session", {"outcome": "no_edge", "reason": "x" * 60}),))]),
            command_runner_factory=lambda _workspace: _NoShellRunner(),
        )
    rows = ExperimentLedger(options.rolling.ledger_path).read()
    assert [row["record_type"] for row in rows] == ["attempt_failed", "attempt_failed"]
    assert "workspace of the interrupted attempt is missing" in rows[-1]["error"]
    assert read_status(experiment / "hitl" / "status.json")["state"] == "failed"


def _model_input(llm) -> str:
    """Every message and tool schema a scripted model was sent, as one text."""

    return "\n".join(
        [message.content or "" for call in llm.calls for message in call["messages"]]
        + [json.dumps(call["tools"], ensure_ascii=False) for call in llm.calls]
    )


def test_the_llm_session_validates_a_multi_year_span_is_refused_by_the_gate_and_ends_the_arm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, synthetic_provider
):
    """The no_edge outcome through the real worker.

    On a two-year research period the session validates its working copy on
    Y2 and on the full span, which replays both research years as one book; a
    freeze of the full-span node is refused by the Pipeline's own gate with
    its named reason (one full-span validation), the session validates the
    full span again and ends the arm with no_edge: the arm has no deliverable
    and nothing is replayed after research end.
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

    repo, experiment = make_arm(
        tmp_path,
        developer_mode="llm",
        research_start="20220701",
        max_replay_years=6,
    )
    monkeypatch.setenv("VLLM_API_KEY", "local-test-key")
    options = load_worker_options(experiment, repo_root=repo)
    llm = _NominatingLLM(
        [
            ProviderResponse(tool_calls=(validate("Y2"), validate("full"))),
            finish(outcome="freeze", node_id=LAST_WORKING_COPY_NODE),
            ProviderResponse(tool_calls=(validate("full"),)),
            finish(outcome="no_edge", reason="完整研究期上复核后，本机制没有稳定的中性化超额，参考包的终止条件已满足，结束本臂。"),
        ]
    )
    result = run_local_interactive_worker(
        options, llm=llm, command_runner_factory=lambda _workspace: _NoShellRunner()
    )

    records = ExperimentLedger(options.rolling.ledger_path).read()
    assert [row["record_type"] for row in records] == ["research_session"]
    [record] = records
    assert [step["span"] for step in record["steps"]] == ["Y2", "full", "full"]
    assert (record["outcome"], record["frozen"], record["trials_to_date"]) == ("no_edge", None, 3)
    assert result["verdict"] == {"status": "no_deliverable", "reasons": [f"no_edge: {record['reason']}"]}
    assert all(item[0] != "heldout" for item in synthetic_provider.requests)

    # The refusal the model read names the gate's reason and its numbers.
    refusal = next(
        message.content
        for call in llm.calls
        for message in call["messages"]
        if "freeze_gate_refused" in (message.content or "")
    )
    assert "freeze_too_few_full_span_validations" in refusal and "full_span_validations=1" in refusal
    assert "no_edge" in refusal and "continue" not in refusal

    # Y2 replayed one slot; the full span replayed Y1 and then Y2 as one book.
    year, full, _again = (
        json.loads(Path(step["validation_result_ref"]).read_text(encoding="utf-8")) for step in record["steps"]
    )
    assert [name.split("_")[:2] for name in year["pit"]["replay_slots"]] == [["20230701", "20240630"]]
    assert [name.split("_")[:2] for name in full["pit"]["replay_slots"]] == [
        ["20220701", "20230630"],
        ["20230701", "20240630"],
    ]
    assert full["equity_curve"][0]["trade_date"] == "20220701"
    assert full["equity_curve"][-1]["trade_date"] <= GEOMETRY["research_end"]
    assert [row["label"] for row in full["stats"]["sub_windows"]] == ["202207-202306", "202307-202406"]

    # The session's facts: one session, started from the template, no trials yet.
    system = next(
        message.content
        for call in llm.calls
        for message in call["messages"]
        if message.role == "system" and "research_geometry" in (message.content or "")
    )
    facts = json.loads(system.split("```json\n", 1)[1].split("\n```", 1)[0])
    assert "session" not in facts["identity"] and "earlier_sessions" not in facts
    assert facts["artifact_contract"]["start"]["kind"] == "template"
    assert [year["label"] for year in facts["research_geometry"]["years"]] == ["Y1", "Y2"]
    assert facts["research_geometry"]["research_period"] == "20220701..20240630"
    assert facts["arm"]["trials_to_date"] == 0 and facts["arm"]["full_span_validations_to_date"] == 0

    read = _model_input(llm)
    assert retired_vocabulary(read) == []
    for later in ("20240701", "2024-07-01", GEOMETRY["forward_end"], GEOMETRY["heldout_end"], RELEASE_END):
        assert later not in read
