"""The research experiment end to end through the real worker and the real PIT replay.

Only the snapshot provider is synthetic (``tests/unit/synthetic_experiment.py``): the
params loader, the interactive runner, the pipeline, the deterministic
developer, the span replay, the Broker, the readout and the terminal status
are the production code.
"""

from __future__ import annotations

import json
import re
import threading
from pathlib import Path

import pytest

from autotrade.environment.artifacts import (
    REVISION_MANIFEST_FILE,
    FilesystemArtifactStore,
)
from autotrade.environment.llm.proxy import ScriptedLLM
from autotrade.environment.runtime import append_versioned_jsonl
from autotrade.pipelines import experiment as experiment_module
from autotrade.pipelines import worker
from autotrade.pipelines.hitl_state import read_status
from autotrade.pipelines.ledger import (
    LEDGER_RECORD_SCHEMA_VERSION,
    ExperimentLedger,
    experiment_verdict,
    forward_record,
    paper_candidate,
    registered_record,
    research_records,
)
from autotrade.pipelines.worker import load_worker_options, run_local_interactive_worker
from tests.unit.synthetic_experiment import (
    GEOMETRY,
    RELEASE_END,
    SyntheticPITProvider,
    decision_anchor,
    make_experiment,
)
from tests.unit.test_research_session_prompt import retired_vocabulary

# A calendar day as the session's facts, tool results and views spell it.
DATE = re.compile(r"(?<!\d)(20\d{6})(?!\d)")


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


def test_an_experiment_runs_research_completes_the_record_and_replays_forward_and_heldout_in_one_span(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, synthetic_provider
):
    repo, experiment = make_experiment(tmp_path)
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
    assert [row["record_type"] for row in records] == [*(["test_step"] * 4), "research_session", "forward"]
    *tests, research_row, replay = records
    # The baseline tests nothing: its budget ends and the host registers its
    # last Validation and fills the whole record from one replay of it.
    assert (research_row["outcome"], research_row["finish_reason"]) == (
        "deadline", "deterministic_baseline_replay_no_agent_improvement"
    )
    assert research_row["freeze_gate"]["passed"] is True
    assert [step["span"] for step in research_row["steps"]] == ["full", "full"]
    assert research_row["nominated_step_id"] == research_row["steps"][-1]["step_id"]
    assert [(row["step"], row["test_horizon"], row["source"]) for row in tests] == [
        (1, "20230930", "host_completion"),
        (2, "20231231", "host_completion"),
        (3, "20240331", "host_completion"),
        (4, "20240630", "host_completion"),
    ]
    assert {row["candidate"]["step_id"] for row in tests} == {research_row["nominated_step_id"]}
    assert (research_row["test_horizon"], research_row["steps_tested"]) == ("20240630", 4)
    frozen = registered_record(records)["frozen"]
    assert Path(frozen["output_path"], "main.py").is_file()

    # Research and the host's completion never asked for a slot or view after
    # research end; the forward stage asked for exactly the forward and the
    # clipped Held-out slot.
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
    # The readout: the record the host completed, read as one series, and
    # the forward reading; nothing judges.
    readout = replay["readout"]
    assert "verdict" not in replay and "sealed" not in replay
    assert (readout["state"], readout["record"]["steps"], readout["record"]["candidate_changes"]) == (
        "registered_with_record", 4, 0
    )
    assert (readout["record"]["start"], readout["record"]["end"]) == ("20230701", "20240630")
    assert 0.0 <= readout["record"]["p_value"] <= 1.0
    assert readout["forward"]["lower_bound"] == forward_slice["lower_bound"]

    verdict = experiment_verdict(records)
    assert verdict == {"status": "registered_with_record", "reasons": []}
    assert paper_candidate(records)["artifact_id"] == frozen["artifact_id"]
    assert result["state"] == "completed"
    assert result["verdict"] == verdict
    assert result["final_strategy_artifact"] == frozen["artifact_id"]
    assert read_status(experiment / "hitl" / "status.json")["verdict"] == verdict
    # The terminal step released the PIT view cache, and only that: the durable
    # record read above and below is intact.
    assert not (experiment / "pit_views").exists()
    assert "pit_views_release_error" not in result
    assert {path.name for path in experiment.iterdir()} >= {".host", "artifacts", "hitl", "ledgers"}
    # No Paper book opens by itself: the operator opens one from the console.
    assert "paper_book" not in result and not (repo / "data/trading/paper").exists()
    plan = json.loads((experiment / "hitl" / "schedule.json").read_text(encoding="utf-8"))
    assert [row["session_key"] for row in plan["sessions"]] == ["research", "forward"]
    assert plan["sessions"][-1]["replay"]["replay_end"] == RELEASE_END

    # Replay memory by stage: the research replays and the forward span.
    print(json.dumps({"peak_rss_growth_mib": measured}))
    assert set(measured) == {"research", "forward"}

    # A resume after the readout republishes the terminal status and runs nothing.
    before = ExperimentLedger(options.rolling.ledger_path).read()
    synthetic_provider.requests.clear()
    again = run_local_interactive_worker(load_worker_options(experiment, repo_root=repo))
    assert again["verdict"] == verdict
    assert ExperimentLedger(options.rolling.ledger_path).read() == before
    assert synthetic_provider.requests == []


def test_a_crashed_research_attempt_is_run_again_by_the_next_worker_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    repo, experiment = make_experiment(tmp_path, session_max_attempts=1)
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
    assert result["verdict"]["status"] == "registered_with_record"


def test_a_worker_stopped_after_the_registration_resumes_with_the_forward_replay_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    repo, experiment = make_experiment(tmp_path, session_max_attempts=1)
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
    assert registered_record(ledger.read()) is not None
    assert forward_record(ledger.read()) is None
    # A failed run may resume: its cache stays until the verdict.
    assert (experiment / "pit_views" / "provider.json").is_file()

    research = experiment_module.RollingExperimentPipeline.run_research_session

    def no_more_research(self, **kwargs):
        raise AssertionError("the research session re-ran after the registration")

    monkeypatch.setattr(experiment_module.RollingExperimentPipeline, "run_research_session", no_more_research)
    result = run_local_interactive_worker(load_worker_options(experiment, repo_root=repo))
    assert state["forward_calls"] == 2
    assert result["verdict"]["status"] == "registered_with_record"
    assert research is not no_more_research
    assert not (experiment / "pit_views").exists()


def test_an_experiment_stopped_before_its_verdict_keeps_its_pit_views(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from autotrade.pipelines.hitl_state import read_control, write_control

    repo, experiment = make_experiment(tmp_path)
    control_path = experiment / "hitl" / "control.json"
    real = experiment_module.RollingExperimentPipeline.run_research_session

    def stop_after_research(self, **kwargs):
        record = real(self, **kwargs)
        control = read_control(control_path)
        control.request = "stop"
        write_control(control_path, control)
        return record

    monkeypatch.setattr(experiment_module.RollingExperimentPipeline, "run_research_session", stop_after_research)
    result = run_local_interactive_worker(load_worker_options(experiment, repo_root=repo))
    assert result["status"] == "stop"
    rows = ExperimentLedger(experiment / "ledgers" / "experiment_ledger.jsonl").read()
    assert registered_record(rows) is not None and experiment_verdict(rows) is None
    assert (experiment / "pit_views" / "provider.json").is_file()


def test_a_cache_that_cannot_be_released_is_recorded_and_the_verdict_stands(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    repo, experiment = make_experiment(tmp_path)

    def stuck(_experiment_dir: Path) -> None:
        raise OSError("pit_views still exists after removal")

    monkeypatch.setattr(worker, "release_pit_views", stuck)
    result = run_local_interactive_worker(load_worker_options(experiment, repo_root=repo))
    assert result["state"] == "completed"
    assert result["verdict"]["status"] == "registered_with_record"
    assert result["pit_views_release_error"] == "OSError: pit_views still exists after removal"
    assert read_status(experiment / "hitl" / "status.json") == result
    assert (experiment / "pit_views" / "provider.json").is_file()


def test_a_failed_replay_attempt_is_recorded_and_retried_from_forward_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    repo, experiment = make_experiment(tmp_path, session_max_attempts=3)
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
    assert result["verdict"]["status"] == "registered_with_record"


def test_a_forward_replay_that_keeps_failing_fails_the_experiment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    repo, experiment = make_experiment(tmp_path, session_max_attempts=2)
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


def test_an_experiment_whose_budget_ends_without_a_registrable_candidate_runs_no_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, synthetic_provider
):
    repo, experiment = make_experiment(tmp_path)
    from autotrade.pipelines import local_backend

    real_call = local_backend.DeterministicBaselineDeveloper.__call__

    def ends_without_a_candidate(self, request):
        result = real_call(self, request)
        return type(result)(
            result.conversation_id, (), "deadline", finish_reason="llm_call_budget_exhausted"
        )

    monkeypatch.setattr(local_backend.DeterministicBaselineDeveloper, "__call__", ends_without_a_candidate)
    result = run_local_interactive_worker(load_worker_options(experiment, repo_root=repo))
    assert result["verdict"] == {
        "status": "no_deliverable",
        "reasons": [
            (
                "research budget exhausted before any test, with no registrable full-span "
                "candidate (llm_call_budget_exhausted)"
            )
        ],
    }
    assert all(item[0] != "heldout" for item in synthetic_provider.requests)
    rows = ExperimentLedger(experiment / "ledgers" / "experiment_ledger.jsonl").read()
    assert [row["record_type"] for row in rows] == ["research_session"]


def test_a_fold_era_ledger_is_refused(tmp_path: Path):
    from autotrade.environment.identity import AgentRefStore

    repo, experiment = make_experiment(tmp_path)
    AgentRefStore(experiment)
    ledger = ExperimentLedger(experiment / "ledgers" / "experiment_ledger.jsonl")
    archived = {
        "record_type": "fold",
        "experiment_id": "experiment",
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
    repo, experiment = make_experiment(tmp_path, heldout_end="20251231", forward_end="20250630")
    params_path = experiment / "hitl" / "params.json"
    params = json.loads(params_path.read_text(encoding="utf-8"))
    params.update({"research_end": "20250630", "forward_end": "20260630", "heldout_end": "20260930"})
    params_path.write_text(json.dumps(params), encoding="utf-8")
    with pytest.raises(ValueError, match="Held-out"):
        load_worker_options(experiment, repo_root=repo)


def test_the_llm_research_session_tests_through_the_validation_year_step_by_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, synthetic_provider
):
    """One scripted Agent session through the real Agent adapter.

    It validates the working copy twice and tests the second node through
    the four steps of the validation year: each test replays the span from
    the validation start to the new test horizon and records the step, while the
    session's ``/mnt/snapshot`` stays the decision view at the research
    year's end; the last test ends the session with its candidate frozen, and the host
    runs forward and Held-out. The two snapshot slot mounts stay empty, no
    replay slot appears anywhere in the session's runtime tree, and the
    Validations the session replayed never saw a bar past the research year.
    A finished sibling experiment's skills stay out of the session: it mounts
    only its own snapshot of the curated library, of which this checkout has
    none.
    """

    from autotrade.environment.llm import ProviderResponse
    from autotrade.pipelines.hitl_state import ControlState, read_control, write_control
    from tests.unit.test_interactive_worker_local import (
        TEST_WORKING_COPY,
        VALIDATE_WORKING_COPY,
        _agent_then,
        _NominatingLLM,
        _NoShellRunner,
    )
    from tests.unit.test_operating_memory import _finished_experiment

    repo, experiment = make_experiment(tmp_path, developer_mode="llm")
    # The assertions below read the session tree. A finished experiment otherwise deletes it.
    monkeypatch.setattr(worker, "release_finished_sandbox", lambda _options: None)
    _finished_experiment(repo / "experiments", "adopted")
    monkeypatch.setenv("VLLM_API_KEY", "local-test-key")
    # The console's GPU allocation for the research session, one-shot.
    write_control(experiment / "hitl" / "control.json", ControlState(gpu_counts={"research": 3}))
    options = load_worker_options(experiment, repo_root=repo)
    llm = _NominatingLLM(
        [
            *_agent_then(VALIDATE_WORKING_COPY, roles=()),
            ProviderResponse(tool_calls=(VALIDATE_WORKING_COPY,)),
            *(ProviderResponse(tool_calls=(TEST_WORKING_COPY,)) for _step in range(4)),
        ]
    )
    result = run_local_interactive_worker(
        options, llm=llm, command_runner_factory=lambda _workspace: _NoShellRunner()
    )

    records = ExperimentLedger(options.rolling.ledger_path).read()
    *tests, research_row, replay = records
    assert [(row["record_type"], row["step"], row["test_horizon"], row["source"]) for row in tests] == [
        ("test_step", 1, "20230930", "test"),
        ("test_step", 2, "20231231", "test"),
        ("test_step", 3, "20240331", "test"),
        ("test_step", 4, "20240630", "test"),
    ]
    assert research_row["outcome"] == "completed"
    assert research_row["finish_reason"] == "validation_period_complete"
    assert (research_row["test_horizon"], research_row["steps_tested"]) == ("20240630", 4)
    tested = research_row["steps"][1]["step_id"]
    assert {row["candidate"]["step_id"] for row in tests} == {tested}
    assert research_row["frozen"]["source_step_id"] == research_row["nominated_step_id"] == tested
    assert research_row["budget_used"]["test_replays"] == 4
    assert replay["record_type"] == "forward"
    assert result["verdict"]["status"] == "registered_with_record"
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
    for block in ("budgets", "research_scope", "visibility_policy", "broker_replay", "experiment"):
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

    host_manifest = json.loads(
        (Path(research_row["run_manifest_ref"]).parent / "host_run_manifest.json").read_text(encoding="utf-8")
    )
    spec = host_manifest["sandbox_spec"]
    assert (spec["gpu_count"], spec["gpu"], spec["gpu_name_filter"]) == (3, "auto", "L20")
    agent_manifest_text = Path(research_row["run_manifest_ref"]).read_text(encoding="utf-8")
    assert json.loads(agent_manifest_text)["operating_memory"]["sources"] == []
    assert "operating_memory_refused" not in host_manifest
    workspace = options.work_root / options.experiment_id / "research" / "agent" / "workspace"
    assert not (workspace / "memory").exists()
    assert "adopted" not in (workspace / "inputs" / "skills_index.json").read_text(encoding="utf-8")
    assert read_control(experiment / "hitl" / "control.json").gpu_counts == {}

    # The session's transcript, named by the run's opaque ref, carries what
    # the Agent did and nothing host-side; the JSONL trace stays under traces/.
    [transcript] = (experiment / "artifacts" / "transcripts").glob("run_ref_*.txt")
    transcript_text = transcript.read_text(encoding="utf-8")
    assert "tool_call call=" in transcript_text and "validate" in transcript_text
    assert str(research_row["run_id"]) not in transcript_text
    assert "/Data2" not in transcript_text and str(options.work_root) not in transcript_text
    assert research_row["agent_trace_ref"].endswith(f"{research_row['run_id']}.jsonl")

    # The research year is Y1; Y2 is the validation year, revealed one step
    # at a time through the test results alone. The session opened on the
    # decision view at the research year's end and read it until it ended:
    # no test prepared or mounted another view.
    first_validation_day, visible_end = "20230701", "20230630"
    anchors = [
        decision.strftime("%Y%m%d")
        for phase, _start, _end, decision in synthetic_provider.requests
        if phase == "decision"
    ]
    assert anchors == [visible_end]
    root = options.work_root / options.experiment_id / "research"
    view = json.loads((root / "runtime" / "current_snapshot" / "manifest.json").read_text(encoding="utf-8"))
    assert (view["kind"], view["decision_time"]) == ("decision_input", decision_anchor(visible_end).isoformat())
    # The facts file was written once, when the attempt opened.
    facts_after = json.loads((workspace / "inputs" / "session_context.json").read_text(encoding="utf-8"))
    assert (facts_after["research_geometry"]["test_horizon"], facts_after["research_geometry"]["steps_tested"]) == (
        visible_end,
        0,
    )
    assert not (root / "snapshots").exists()
    manifests = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in root.rglob("manifest.json")
        if path.is_file()
    ]
    assert all(item.get("kind") != "replay_slot" for item in manifests)
    for step in research_row["steps"]:
        replayed = json.loads(Path(step["validation_result_ref"]).read_text(encoding="utf-8"))
        assert all(name.split("_")[1] <= visible_end for name in replayed["pit"]["replay_slots"])
        # The probe: the last bar the strategy was asked to decide on, and the
        # last day its book was marked, are both before the validation year.
        decided = max(str(day)[:10].replace("-", "") for day in replayed["inference_dates"])
        marked = max(str(row["trade_date"]) for row in replayed["equity_curve"])
        assert decided < first_validation_day and marked < first_validation_day, (decided, marked)
    # Each test replayed the validation span as one fresh book from its first
    # day to the step's test horizon, and no further.
    for row in tests:
        assert (row["replay"]["start"], row["replay"]["end"]) == (first_validation_day, row["test_horizon"])
        replayed = json.loads(Path(row["replay"]["result_ref"]).read_text(encoding="utf-8"))
        assert first_validation_day <= replayed["equity_curve"][0]["trade_date"] <= "20230703"
        assert replayed["equity_curve"][-1]["trade_date"] <= row["test_horizon"]
    # The readout reads the four tested steps as one record; the forward
    # stage replayed nothing of the validation year again.
    readout = replay["readout"]
    assert "sealed" not in replay and readout["record"]["steps"] == 4
    assert readout["record"]["candidate_changes"] == 0
    # Everything the model read, tool schemas included, is research-session
    # vocabulary, names the validation year only by its dates and no date
    # after research end, and never the forward reading.
    read = _model_input(llm)
    assert retired_vocabulary(read) == []
    assert f"{first_validation_day}..{GEOMETRY['research_end']}" in read
    for later in ("20240701", "2024-07-01", GEOMETRY["forward_end"], GEOMETRY["heldout_end"], RELEASE_END):
        assert later not in read
    forward_reading = replay["slices"]["forward"]["residual_reading"]["residual_information_ratio"]
    assert f"{forward_reading:.4f}" not in read
    # What each test returned dates no later than the test horizon it reached.
    returned = [
        message.content
        for message in llm.calls[-1]["messages"]
        if '"status": "step_recorded"' in (message.content or "")
    ]
    assert len(returned) == 3
    for row, content in zip(tests[:3], returned, strict=True):
        assert max(DATE.findall(content)) == row["test_horizon"], row["step"]


def _resume_script_summary() -> str:
    return (
        "## 策略现状\noutput/ 是模板策略，已完成一次完整研究期验证（working_copy）。\n## 决定\n"
        "- 该验证的中性化超额为正，作为可登记的完整研究期验证保留；节点 id 见 trace 第 1 次调用的 "
        "validate 结果。\n## 线索\n- 下一步再验证一次 working_copy，然后用它 test 验证期的各步。"
        "\n## trace\n- grep 'working_copy' 可找回节点 id 与读数。"
    )


def test_an_interrupted_llm_session_resumes_with_its_summary_budget_and_nodes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, synthetic_provider
):
    """The real worker, twice. The first attempt validates the working copy,
    compacts its context and then loses its model (the script runs out):
    the attempt is recorded as failed, its node, revision, workspace and
    transcript stay. The second worker start resumes the same session: it
    starts from the compaction summary and a note, continues the budget,
    validates once more and tests through the validation year; the record
    carries both attempts' steps."""

    from autotrade.environment.llm import ProviderResponse, ToolCall
    from tests.unit.test_interactive_worker_local import (
        TEST_WORKING_COPY,
        VALIDATE_WORKING_COPY,
        _agent_then,
        _NominatingLLM,
        _NoShellRunner,
    )

    repo, experiment = make_experiment(
        tmp_path, developer_mode="llm", session_max_attempts=1, max_llm_calls=40
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
            *(ProviderResponse(tool_calls=(TEST_WORKING_COPY,)) for _step in range(4)),
        ]
    )
    result = run_local_interactive_worker(
        load_worker_options(experiment, repo_root=repo),
        llm=second_llm,
        command_runner_factory=lambda _workspace: _NoShellRunner(),
    )

    records = ledger.read()
    assert [row["record_type"] for row in records] == [
        "attempt_failed", *(["test_step"] * 4), "research_session", "forward"
    ]
    record = records[-2]
    assert record["attempts"] == 2 and record["outcome"] == "completed"
    assert record["steps"][0]["step_id"] == node["node_id"]
    # Both attempts validated the same working copy: two revisions, one set of bytes.
    assert len(record["steps"]) == 2 and record["freeze_gate"]["passed"] is True
    assert len({row["revision_id"] for row in record["steps"]}) == 2
    assert len({row["fingerprint"] for row in record["steps"]}) == 1
    assert record["budget_used"]["test_replays"] == 4
    # The calls of both attempts: the first spent five (two scripted, three failed).
    assert record["budget_used"]["llm_calls"] == 5 + len(second_llm.calls)
    assert result["verdict"]["status"] == "registered_with_record"
    # The resumed attempt opened with the interrupted one's summary, then the note.
    opening = second_llm.calls[0]["messages"]
    assert opening[0].role == "system"
    checkpoint = json.loads(opening[1].content)
    assert checkpoint["observation"] == "context_compaction"
    assert checkpoint["summary_kind"] == "resume" and checkpoint["summary"] == summary
    assert checkpoint["trace"]["root"] == "trace"
    note = opening[2].content
    assert "第 2 次尝试" in note and "language model unavailable" in note
    assert "模型调用 5/" in note and transcripts[0].name in note
    assert "回测年" not in note
    assert "上面是中断前最近一次压缩的摘要" in note
    facts = json.loads(opening[0].content.split("```json\n", 1)[1].split("\n```", 1)[0])
    assert facts["experiment"] == {"validations_to_date": 1, "full_span_validations_to_date": 1}
    assert facts["budgets"]["used_before_this_attempt"]["test_replays"] == 0
    assert facts["budgets"]["used_before_this_attempt"]["llm_calls"] == 5
    # One session root, two transcripts, and both attempts' revisions kept as
    # the experiment's artifact history: the resumed attempt's revision descends from
    # the one the node it continued out of was validated as.
    assert sorted(path.name for path in (experiment / "artifacts" / "transcripts").glob("run_ref_*.txt")) != [transcripts[0].name]
    store = FilesystemArtifactStore(experiment / "artifacts" / "strategy")
    manifests = sorted(
        (
            json.loads(path.read_text(encoding="utf-8"))
            for path in store.revisions_root.glob(f"*/{REVISION_MANIFEST_FILE}")
        ),
        key=lambda manifest: str(manifest["created_at"]),
    )
    assert [manifest["parent_revision_id"] for manifest in manifests] == [
        None,
        manifests[0]["revision_id"],
    ]


def test_an_interrupted_session_resumes_from_its_test_horizon_and_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, synthetic_provider
):
    """The first attempt validates the working copy, tests it and then loses
    its model: the step it recorded stays in the ledger. The second worker
    start resumes the session at the test horizon that step reached -- on the
    same view at the research year's end, with facts that state the steps
    tested -- and when its one remaining call is spent the host completes the
    record from the tested candidate, not from the Validation the resumed
    attempt added."""

    from autotrade.environment.llm import ProviderResponse
    from tests.unit.test_interactive_worker_local import (
        TEST_WORKING_COPY,
        VALIDATE_WORKING_COPY,
        _NominatingLLM,
        _NoShellRunner,
    )

    repo, experiment = make_experiment(
        tmp_path, developer_mode="llm", session_max_attempts=1, max_llm_calls=6
    )
    monkeypatch.setenv("VLLM_API_KEY", "local-test-key")
    options = load_worker_options(experiment, repo_root=repo)
    first_llm = _NominatingLLM(
        [
            ProviderResponse(tool_calls=(VALIDATE_WORKING_COPY,)),
            ProviderResponse(tool_calls=(TEST_WORKING_COPY,)),
            # The script ends: three consecutive model failures end the attempt.
        ]
    )
    with pytest.raises(RuntimeError, match="language model unavailable"):
        run_local_interactive_worker(
            options, llm=first_llm, command_runner_factory=lambda _workspace: _NoShellRunner()
        )
    ledger = ExperimentLedger(options.rolling.ledger_path)
    tested, failed = ledger.read()
    assert (tested["record_type"], tested["step"], tested["test_horizon"], tested["source"]) == (
        "test_step", 1, "20230930", "test"
    )
    assert failed["record_type"] == "attempt_failed"
    tested_node = tested["candidate"]["step_id"]

    second_llm = _NominatingLLM([ProviderResponse(tool_calls=(VALIDATE_WORKING_COPY,))])
    result = run_local_interactive_worker(
        load_worker_options(experiment, repo_root=repo),
        llm=second_llm,
        command_runner_factory=lambda _workspace: _NoShellRunner(),
    )

    records = ledger.read()
    assert [row["record_type"] for row in records] == [
        "test_step", "attempt_failed", *(["test_step"] * 3), "research_session", "forward"
    ]
    record = records[-2]
    assert (record["attempts"], record["outcome"], record["finish_reason"]) == (
        2, "deadline", "llm_call_budget_exhausted"
    )
    # Five calls of the first attempt (two scripted, three failed), one of the second.
    assert record["budget_used"]["llm_calls"] == 6 and record["budget_used"]["test_replays"] == 1
    assert [row["step_id"] for row in record["steps"]] != [tested_node]
    assert record["frozen"]["source_step_id"] == tested_node != record["steps"][-1]["step_id"]
    completion = records[2:5]
    assert [(row["step"], row["test_horizon"], row["source"]) for row in completion] == [
        (2, "20231231", "host_completion"),
        (3, "20240331", "host_completion"),
        (4, "20240630", "host_completion"),
    ]
    assert {row["candidate"]["step_id"] for row in completion} == {tested_node}
    assert (record["test_horizon"], record["steps_tested"]) == ("20240630", 4)
    assert result["verdict"]["status"] == "registered_with_record"
    # Both attempts opened on the research year's view; the resumed one read
    # the test horizon in its facts.
    research_requests = synthetic_provider.requests[
        : next(index for index, item in enumerate(synthetic_provider.requests) if item[0] == "heldout")
    ]
    assert [
        decision.strftime("%Y%m%d") for phase, _start, _end, decision in research_requests if phase == "decision"
    ] == ["20230630", "20230630"]
    opening = second_llm.calls[0]["messages"]
    facts = json.loads(opening[0].content.split("```json\n", 1)[1].split("\n```", 1)[0])
    geometry = facts["research_geometry"]
    assert (geometry["test_horizon"], geometry["steps_tested"], geometry["steps"]) == ("20230930", 1, 4)
    assert facts["experiment"] == {"validations_to_date": 1, "full_span_validations_to_date": 1}
    assert facts["budgets"]["used_before_this_attempt"]["test_replays"] == 1
    assert "第 2 次尝试" in opening[1].content


def test_an_interrupted_session_without_a_checkpoint_resumes_from_the_note_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, synthetic_provider
):
    """The first attempt loses its model before it validated or compacted
    anything. The resumed attempt has no summary to start from: it opens with
    the note alone (which says so) and continues the call budget, which its
    one validate exhausts; the host then registers that registrable Validation
    and completes the validation record from one replay of it."""

    from autotrade.environment.llm import ProviderResponse
    from tests.unit.test_interactive_worker_local import (
        VALIDATE_WORKING_COPY,
        _NominatingLLM,
        _NoShellRunner,
    )

    repo, experiment = make_experiment(
        tmp_path, developer_mode="llm", session_max_attempts=1, max_llm_calls=4
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

    second_llm = _NominatingLLM([ProviderResponse(tool_calls=(VALIDATE_WORKING_COPY,))])
    result = run_local_interactive_worker(
        load_worker_options(experiment, repo_root=repo),
        llm=second_llm,
        command_runner_factory=lambda _workspace: _NoShellRunner(),
    )

    records = ledger.read()
    assert [row["record_type"] for row in records] == [
        "attempt_failed", *(["test_step"] * 4), "research_session", "forward"
    ]
    record = records[-2]
    assert (record["attempts"], record["outcome"], record["finish_reason"]) == (
        2, "deadline", "llm_call_budget_exhausted"
    )
    assert record["experiment_end"] is None and len(record["steps"]) == 1
    assert record["frozen"]["source_step_id"] == record["steps"][0]["step_id"]
    assert {row["source"] for row in records[1:5]} == {"host_completion"}
    # Three failed calls of the first attempt, then this attempt's one.
    assert record["budget_used"]["llm_calls"] == 3 + len(second_llm.calls) == 4
    assert result["verdict"]["status"] == "registered_with_record"
    opening = second_llm.calls[0]["messages"]
    assert [message.role for message in opening] == ["system", "user"]
    note = opening[1].content
    assert "第 2 次尝试" in note and "language model unavailable" in note
    assert "中断前没有压缩摘要" in note and "模型调用 3/4" in note
    facts = json.loads(opening[0].content.split("```json\n", 1)[1].split("\n```", 1)[0])
    assert facts["budgets"]["used_before_this_attempt"]["llm_calls"] == 3
    assert facts["experiment"]["validations_to_date"] == 0
    assert not (options.work_root / options.experiment_id).exists()


def test_a_hosted_session_that_reaches_its_api_spend_cap_is_completed_by_the_host(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, synthetic_provider
):
    """The real worker, twice, on a gateway priced as the relay's
    claude-opus-5-5: each scripted reply bills 5,000 uncached and 5,000 cached
    prompt tokens and 1,000 completion tokens, $0.041. The first attempt
    spends one reply and loses its model; the resumed attempt continues from
    that spend, its next reply reaches the $0.08 cap, and the call after it is
    never made: the session ends on the spend cap, the host registers the one
    registrable Validation and completes the record, and the ledger record
    carries both attempts' spend."""

    from autotrade.environment.llm import ProviderResponse
    from tests.unit.test_interactive_worker_local import (
        TEST_WORKING_COPY,
        VALIDATE_WORKING_COPY,
        _NominatingLLM,
        _NoShellRunner,
    )

    class HostedLLM(_NominatingLLM):
        provider = "lighting"
        model = "claude-opus-5-5"

    usage = {
        "prompt_tokens": 10_000,
        "completion_tokens": 1_000,
        "prompt_tokens_details": {"cached_tokens": 5_000},
    }
    repo, experiment = make_experiment(
        tmp_path,
        developer_mode="llm",
        session_max_attempts=1,
        max_llm_calls=40,
        max_api_spend_usd=0.08,
    )
    monkeypatch.setenv("VLLM_API_KEY", "local-test-key")
    options = load_worker_options(experiment, repo_root=repo)
    with pytest.raises(RuntimeError, match="language model unavailable"):
        run_local_interactive_worker(
            options,
            llm=HostedLLM([ProviderResponse(tool_calls=(VALIDATE_WORKING_COPY,), usage=usage)]),
            command_runner_factory=lambda _workspace: _NoShellRunner(),
        )

    second_llm = HostedLLM(
        [
            ProviderResponse(tool_calls=(VALIDATE_WORKING_COPY,), usage=usage),
            ProviderResponse(tool_calls=(TEST_WORKING_COPY,), usage=usage),
        ]
    )
    result = run_local_interactive_worker(
        load_worker_options(experiment, repo_root=repo),
        llm=second_llm,
        command_runner_factory=lambda _workspace: _NoShellRunner(),
    )

    # The test was never asked for: the cap refused the call before it.
    assert len(second_llm.calls) == 1
    records = ExperimentLedger(options.rolling.ledger_path).read()
    assert [row["record_type"] for row in records] == [
        "attempt_failed", *(["test_step"] * 4), "research_session", "forward"
    ]
    record = records[-2]
    assert (record["attempts"], record["outcome"], record["finish_reason"]) == (
        2,
        "deadline",
        "api_spend_exhausted",
    )
    assert record["experiment_end"] is None
    assert len(record["steps"]) == 2
    assert record["frozen"]["source_step_id"] == record["steps"][-1]["step_id"]
    assert record["budget_used"]["api_spend_usd"] == 0.082
    assert result["verdict"]["status"] == "registered_with_record"
    events = [
        json.loads(line)
        for line in Path(record["agent_trace_ref"]).read_text(encoding="utf-8").splitlines()
    ]
    [end] = [event for event in events if event.get("event_type") == "session_end"]
    assert end["status"] == "api_spend_exhausted"
    # The resumed attempt was told the cap and what the first one spent of it.
    opening = second_llm.calls[0]["messages"]
    assert "API 花费 $0.04/$0.08" in opening[1].content
    facts = json.loads(opening[0].content.split("```json\n", 1)[1].split("\n```", 1)[0])
    assert facts["budgets"]["max_api_spend_usd"] == 0.08
    assert facts["budgets"]["used_before_this_attempt"]["api_spend_usd"] == 0.041


def test_release_finished_sandbox_removes_only_that_experiment(tmp_path: Path) -> None:
    from types import SimpleNamespace

    from autotrade.pipelines.worker import release_finished_sandbox

    work = tmp_path / "sandboxes"
    own = work / "experiment_a" / "research"
    own.mkdir(parents=True)
    locked = own / "snapshot"
    locked.mkdir(mode=0o555)
    (own / "note.txt").write_text("x", encoding="utf-8")
    sibling = work / "experiment_b"
    sibling.mkdir()
    (sibling / "keep.txt").write_text("keep", encoding="utf-8")

    release_finished_sandbox(SimpleNamespace(work_root=work, experiment_id="experiment_a"))
    assert not (work / "experiment_a").exists()
    assert (sibling / "keep.txt").read_text(encoding="utf-8") == "keep"

    release_finished_sandbox(SimpleNamespace(work_root=work, experiment_id="experiment_a"))
    release_finished_sandbox(SimpleNamespace(work_root=work, experiment_id=""))
    assert work.is_dir() and (sibling / "keep.txt").is_file()


def test_a_resume_without_the_interrupted_workspace_fails_the_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, synthetic_provider
):
    from autotrade.environment.llm import ProviderResponse, ToolCall
    from autotrade.pipelines.research_session import _remove_mounted_tree
    from tests.unit.test_interactive_worker_local import _NominatingLLM, _NoShellRunner

    repo, experiment = make_experiment(tmp_path, developer_mode="llm", session_max_attempts=1)
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
            llm=_NominatingLLM([ProviderResponse(tool_calls=(ToolCall("f", "test", {"outcome": "no_edge", "reason": "x" * 60}),))]),
            command_runner_factory=lambda _workspace: _NoShellRunner(),
        )
    rows = ExperimentLedger(options.rolling.ledger_path).read()
    assert [row["record_type"] for row in rows] == ["attempt_failed", "attempt_failed"]
    assert "workspace of the interrupted attempt is missing" in rows[-1]["error"]
    assert read_status(experiment / "hitl" / "status.json")["state"] == "failed"


def test_attempts_that_fail_before_the_workspace_is_seeded_leave_a_fresh_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, synthetic_provider
):
    """One attempt dies preparing the PIT inputs (no runtime at all), the next
    while laying out the workspace (a runtime without a working copy). Neither
    spent or recorded anything, so the third opens the session as a first
    attempt would instead of failing for a workspace that never existed."""

    from autotrade.agent.prompts import SESSION_DEFAULT_INSTRUCTION
    from autotrade.environment.llm import ProviderResponse
    from autotrade.pipelines import research_session
    from tests.unit.test_interactive_worker_local import (
        VALIDATE_WORKING_COPY,
        _NominatingLLM,
        _NoShellRunner,
    )

    repo, experiment = make_experiment(tmp_path, developer_mode="llm", session_max_attempts=1, max_llm_calls=1)
    monkeypatch.setenv("VLLM_API_KEY", "local-test-key")
    session_root = tmp_path / "unused"

    def run(llm):
        options = load_worker_options(experiment, repo_root=repo)
        nonlocal session_root
        session_root = options.work_root / options.experiment_id / "research"
        return run_local_interactive_worker(
            options, llm=llm, command_runner_factory=lambda _workspace: _NoShellRunner()
        )

    def interrupted(*_args, **_kwargs):
        raise SystemExit("worker stopped")

    pipeline = experiment_module.RollingExperimentPipeline
    research_inputs = pipeline.research_inputs
    monkeypatch.setattr(pipeline, "research_inputs", interrupted)
    with pytest.raises(SystemExit):
        run(_NominatingLLM([]))
    assert not session_root.exists()
    monkeypatch.setattr(pipeline, "research_inputs", research_inputs)

    copy_artifact = research_session.copy_artifact
    monkeypatch.setattr(research_session, "copy_artifact", interrupted)
    with pytest.raises(SystemExit):
        run(_NominatingLLM([]))
    workspace = session_root / "agent" / "workspace"
    assert workspace.is_dir() and not (workspace / "output" / "main.py").exists()
    monkeypatch.setattr(research_session, "copy_artifact", copy_artifact)

    # One call is the whole budget: the validate spends it, and the host
    # completes the record from the Validation it recorded.
    llm = _NominatingLLM([ProviderResponse(tool_calls=(VALIDATE_WORKING_COPY,))])
    run(llm)
    rows = ExperimentLedger(experiment / "ledgers" / "experiment_ledger.jsonl").read()
    assert [row["record_type"] for row in rows] == [
        "attempt_failed", "attempt_failed", *(["test_step"] * 4), "research_session", "forward"
    ]
    record = rows[-2]
    assert record["attempts"] == 3 and record["outcome"] == "deadline"
    assert record["budget_used"]["llm_calls"] == len(llm.calls) == 1
    opening = llm.calls[0]["messages"]
    assert [message.role for message in opening] == ["system", "user"]
    assert opening[1].content == SESSION_DEFAULT_INSTRUCTION


def _model_input(llm) -> str:
    """Every message and tool schema a scripted model was sent, as one text."""

    return "\n".join(
        [message.content or "" for call in llm.calls for message in call["messages"]]
        + [json.dumps(call["tools"], ensure_ascii=False) for call in llm.calls]
    )


def test_a_control_is_refused_at_test_and_a_tested_step_is_completed_by_the_host(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, synthetic_provider
):
    """The validation year through the real worker.

    The session validates its working copy as a control; a test of that node
    is refused at registration with its named reason, which carries no
    reading of any kind and moves no test horizon. It validates the working copy
    again and tests it: the first step is recorded, the test horizon advances a
    quarter while the session's view stays at the research year's end, and
    the model reads the step. The call budget then ends the session, and the host
    completes the record's remaining steps from one replay of the tested
    candidate before the forward stage runs.
    """

    from autotrade.environment.llm import ProviderResponse, ToolCall
    from tests.unit.test_interactive_worker_local import (
        TEST_WORKING_COPY,
        _NominatingLLM,
        _NoShellRunner,
    )

    def validate(name: str, *, control: bool) -> ToolCall:
        return ToolCall(
            f"valid_{name}",
            "validate",
            {
                "candidates": [
                    {
                        "name": name,
                        "hypothesis": "the working copy earns a positive active excess",
                        "path": "output",
                        "control": control,
                    }
                ],
            },
        )

    repo, experiment = make_experiment(tmp_path, developer_mode="llm", max_llm_calls=5)
    monkeypatch.setenv("VLLM_API_KEY", "local-test-key")
    options = load_worker_options(experiment, repo_root=repo)
    llm = _NominatingLLM(
        [
            ProviderResponse(tool_calls=(validate("working_copy", control=True),)),
            ProviderResponse(tool_calls=(TEST_WORKING_COPY,)),
            ProviderResponse(tool_calls=(validate("working_copy", control=False),)),
            ProviderResponse(tool_calls=(TEST_WORKING_COPY,)),
            ProviderResponse(content="第一步已记录。"),
        ]
    )
    result = run_local_interactive_worker(
        options, llm=llm, command_runner_factory=lambda _workspace: _NoShellRunner()
    )

    assert len(llm.calls) == 5
    records = ExperimentLedger(options.rolling.ledger_path).read()
    *tests, record, replay = records
    assert [(row["record_type"], row["step"], row["test_horizon"], row["source"]) for row in tests] == [
        ("test_step", 1, "20230930", "test"),
        ("test_step", 2, "20231231", "host_completion"),
        ("test_step", 3, "20240331", "host_completion"),
        ("test_step", 4, "20240630", "host_completion"),
    ]
    assert [(step["span"], step["control"]) for step in record["steps"]] == [("full", True), ("full", False)]
    tested = record["steps"][1]["step_id"]
    assert {row["candidate"]["step_id"] for row in tests} == {tested}
    assert (record["outcome"], record["finish_reason"]) == ("deadline", "llm_call_budget_exhausted")
    assert record["frozen"]["source_step_id"] == tested and record["experiment_end"] is None
    assert (record["test_horizon"], record["steps_tested"]) == ("20240630", 4)
    assert record["budget_used"]["test_replays"] == 1
    assert replay["record_type"] == "forward" and result["verdict"]["status"] == "registered_with_record"
    # The session's view: the research year's end, the only decision view
    # the research stage prepared, whatever the tests reached.
    research_requests = synthetic_provider.requests[
        : next(index for index, item in enumerate(synthetic_provider.requests) if item[0] == "heldout")
    ]
    assert [
        decision.strftime("%Y%m%d") for phase, _start, _end, decision in research_requests if phase == "decision"
    ] == ["20230630"]
    assert all(decision.strftime("%Y%m%d") <= "20230630" for _phase, _start, _end, decision in research_requests)

    # The refusal the model read names the registration's reason and nothing measured.
    refusal = next(
        message.content
        for call in llm.calls
        for message in call["messages"]
        if "does not register" in (message.content or "")
    )
    assert "freeze_nominee_is_control" in refusal and "the test horizon did not move" in refusal
    assert "information_ratio" not in refusal and "continue" not in refusal
    # What the test returned dates no later than the test horizon it reached.
    [returned] = [
        message.content
        for message in llm.calls[-1]["messages"]
        if '"status": "step_recorded"' in (message.content or "")
    ]
    assert max(DATE.findall(returned)) == "20230930"
    assert '"step": 1' in returned and '"steps_left": 3' in returned

    # The full span is the research year, one slot, one book.
    first = json.loads(Path(record["steps"][0]["validation_result_ref"]).read_text(encoding="utf-8"))
    assert [name.split("_")[:2] for name in first["pit"]["replay_slots"]] == [["20220701", "20230630"]]
    assert first["equity_curve"][0]["trade_date"] == "20220701"
    assert first["equity_curve"][-1]["trade_date"] <= "20230630"

    # The session's facts: one session, started from the template, the
    # research year and the validation year's dates, no Validation yet.
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
    assert facts["research_geometry"]["research_period"] == "20220701..20230630"
    assert facts["research_geometry"]["validation_period"] == "20230701..20240630"
    assert facts["research_geometry"]["input_window"] == "20210701..20230630"
    assert (facts["research_geometry"]["test_horizon"], facts["research_geometry"]["steps_tested"]) == ("20230630", 0)
    assert facts["experiment"] == {"validations_to_date": 0, "full_span_validations_to_date": 0}
    significance = facts["artifact_contract"]["acceptance_rules"]["significance"]
    assert (significance["alpha"], significance["threshold_over_validation_period"]) == (0.05, 1.645)
    # The facts state the test horizon, so the prompt carries the run section
    # (prompts.build_run_section) and no sealed-gate paragraph.
    assert "## 本次运行" in system and "## 封存检验" not in system

    read = _model_input(llm)
    assert retired_vocabulary(read) == []
    for later in ("20240701", "2024-07-01", GEOMETRY["forward_end"], GEOMETRY["heldout_end"], RELEASE_END):
        assert later not in read


def test_a_span_the_host_cannot_replay_ends_the_experiment_environment_blocked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, synthetic_provider, capsys
):
    """Round 20261015 through the real worker: every replay of the research
    year fails in the host's as-of merge. The full span fails on the
    environment twice; the second ends the attempt without another model
    call. The block ends the attempt, not the research: the next worker
    attempt resumes, and only the third such attempt ends the experiment
    ``environment_blocked`` -- not ``no_deliverable`` -- with the host's
    reason in its ledger record, its terminal status, the worker log and the
    console's attention line. No replay follows."""

    import pyarrow as pa

    from autotrade.environment.llm import ProviderResponse, ToolCall
    from autotrade.pipelines import pit_backend
    from autotrade.webui.registry import summarize_experiment
    from tests.unit.test_interactive_worker_local import _NominatingLLM, _NoShellRunner

    real_evaluate = pit_backend.PITDailyEvaluationBackend.evaluate

    def merge_breaks(self, request, **kwargs):
        if request.end > "20230331":
            raise pa.ArrowTypeError(
                "Unable to merge: Field dataset has incompatible types: large_string vs string"
            )
        return real_evaluate(self, request, **kwargs)

    monkeypatch.setattr(pit_backend.PITDailyEvaluationBackend, "evaluate", merge_breaks)

    def validate(span: str) -> ProviderResponse:
        return ProviderResponse(
            tool_calls=(
                ToolCall(
                    f"valid_{span}",
                    "validate",
                    {
                        "span": span,
                        "candidates": [
                            {
                                "name": "working_copy",
                                "hypothesis": f"the working copy earns a positive active excess over {span}",
                                "path": "output",
                                "control": False,
                            }
                        ],
                    },
                ),
            )
        )

    repo, experiment = make_experiment(tmp_path, developer_mode="llm")
    monkeypatch.setenv("VLLM_API_KEY", "local-test-key")
    options = load_worker_options(experiment, repo_root=repo)
    never = ProviderResponse(tool_calls=(ToolCall("test", "test", {"candidate": "working_copy"}),))
    attempts = options.rolling.session_max_attempts
    assert attempts == 3
    llm = _NominatingLLM([*([validate("full")] * (2 * attempts)), never])
    result = run_local_interactive_worker(
        options, llm=llm, command_runner_factory=lambda _workspace: _NoShellRunner()
    )

    assert len(llm.calls) == 2 * attempts
    *failed, record = ExperimentLedger(options.rolling.ledger_path).read()
    assert [row["record_type"] for row in failed] == ["attempt_failed"] * (attempts - 1)
    assert all("EnvironmentBlockedAttempt: span full: 2 batches failed" in row["error"] for row in failed)
    assert record["record_type"] == "research_session" and record["attempts"] == attempts
    reason = record["reason"]
    assert reason.startswith("span full: 2 batches failed on the environment in every candidate")
    assert reason.endswith("ArrowTypeError: Unable to merge: Field dataset has incompatible types: large_string vs string")
    assert (record["outcome"], record["finish_reason"], record["frozen"]) == (
        "environment_blocked",
        "environment_blocked",
        None,
    )
    assert record["experiment_end"] == {"status": "environment_blocked", "reason": reason}
    assert record["steps"] == []
    # The failed batches were refunded: no replay-year was spent.
    assert record["budget_used"]["test_replays"] == 0
    verdict = {"status": "environment_blocked", "reasons": [reason]}
    assert result["verdict"] == verdict
    assert read_status(experiment / "hitl" / "status.json")["verdict"] == verdict
    assert f"experiment environment_blocked: {reason}" in capsys.readouterr().err
    assert all(item[0] != "heldout" for item in synthetic_provider.requests)
    row = summarize_experiment(experiment)
    assert row["ending"] == {"state": "environment_blocked", "reason": reason}
    assert row["attention"] == {"reason": reason}
    assert row["research_outcome"] == "environment_blocked"


def test_a_readme_edit_stops_a_new_session_and_not_one_already_seeded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The experiment's pinned image was built from the template README its session
    was seeded with, and a maintainer then edits the template in the checkout.
    The session's next attempt resumes on the copy it was seeded with and
    starts. A session seeded after the edit is refused before its Agent reads
    the new text, naming the README. The strategy containers never read the
    README (``test_sandbox_contract_fingerprint``)."""

    import shutil

    from autotrade.environment import contract_fingerprint
    from autotrade.environment.contract_fingerprint import (
        AGENT_CONTRACT_PATHS,
        ImageContract,
        SandboxImageContractMismatch,
        host_contract_fingerprint,
    )
    from autotrade.environment.executor import IMPORTED_RUNTIME_FINGERPRINT
    from autotrade.environment.identity import AgentRefStore
    from autotrade.pipelines import research_session

    class SandboxStarting(Exception):
        """The session passed its image check and reached its container."""

    class StoppedSandbox:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def start(self) -> None:
            raise SandboxStarting

        def stop(self) -> None:
            pass

    monkeypatch.setattr(research_session, "DockerSandbox", StoppedSandbox)
    # The pinned image, built from this checkout's template before the edit.
    built = ImageContract(IMPORTED_RUNTIME_FINGERPRINT, host_contract_fingerprint())
    monkeypatch.setattr(contract_fingerprint, "read_image_contract", lambda image, **_kwargs: built)
    monkeypatch.setenv("VLLM_API_KEY", "local-test-key")
    template = Path(contract_fingerprint.__file__).resolve().parents[3] / "configs" / "agent_output_template"

    def experiment(root: Path):
        repo, experiment = make_experiment(
            root, developer_mode="llm", strategy_path="configs/agent_output_template/main.py"
        )
        shutil.copytree(template, repo / "configs" / "agent_output_template")
        options = load_worker_options(experiment, repo_root=repo)
        build = worker.build_experiment_pipeline(
            options,
            ledger=ExperimentLedger(options.rolling.ledger_path),
            store=FilesystemArtifactStore(options.experiment_dir / "artifacts" / "strategy"),
            ref_store=AgentRefStore(options.experiment_dir),
            llm=ScriptedLLM([]),
        )
        return repo, build.pipeline

    repo, running = experiment(tmp_path / "running")
    with pytest.raises(SandboxStarting):
        running.run_research_session()
    readme = repo / AGENT_CONTRACT_PATHS[0]
    readme.write_text(readme.read_text(encoding="utf-8") + "\nEach decision gets 360 s.\n", encoding="utf-8")
    # The second attempt continues the first one's session runtime (a fresh
    # one over it would refuse), on the README that attempt was seeded with.
    with pytest.raises(SandboxStarting):
        running.run_research_session()

    later, fresh = experiment(tmp_path / "later")
    shutil.copyfile(readme, later / AGENT_CONTRACT_PATHS[0])
    with pytest.raises(SandboxImageContractMismatch, match=AGENT_CONTRACT_PATHS[0]):
        fresh.run_research_session()
