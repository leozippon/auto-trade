"""The research arm's orchestration: one research session, one freeze, one forward replay.

In-memory providers and a result writer stand in for the PIT backend; the real
artifact store, ledger, run markers and verdict statistics run unchanged.
``test_research_arm_worker.py`` drives the same stages through the real worker
and the real PIT replay.
"""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from autotrade.environment.artifacts import FilesystemArtifactStore
from autotrade.environment.executor import StrategyRaised
from autotrade.environment.replay.engine import BacktestError
from autotrade.environment.replay.style import STYLE_ARTIFACT_NAME
from autotrade.environment.strategy import CN_TZ
from autotrade.pipelines.calendar import FULL_SPAN, ResearchGeometry
from autotrade.pipelines.config import (
    ArtifactRevision,
    EvaluationRequest,
    EvaluationResult,
    ResearchSessionRequest,
    ResearchSessionResult,
    RollingExperimentConfig,
    SnapshotBundle,
    StepResult,
    acceptance_for,
)
from autotrade.pipelines.experiment import (
    RollingExperimentPipeline,
    _session_budgets,
    null_control_seed,
    research_step_record,
)
from autotrade.pipelines.ledger import (
    INTERRUPTED_RUN_ERROR,
    UNKNOWN_MARKER_LINK_KEY,
    UNREADABLE_RUN_MARKER_ERROR,
    ExperimentLedger,
    FrozenArtifactMutated,
    RunMarkers,
    experiment_verdict,
    forward_record,
    frozen_record,
    paper_candidate,
    research_over,
)

GEOMETRY = ResearchGeometry(
    research_start="20220701",
    research_end="20240630",
    forward_end="20250630",
    heldout_end="20250930",
)
# The release ends inside Held-out, so the Held-out slot is clipped.
RELEASE_END = "20250912"
DAYS = [
    stamp.strftime("%Y%m%d")
    for stamp in np.arange(
        np.datetime64("2022-06-01"), np.datetime64("2025-09-13"), dtype="datetime64[D]"
    ).astype(datetime)
    if stamp.weekday() < 5
]
MAIN = "def generate_orders(context):\n    return []\n"


class Snapshots:
    """Bundles named after what was asked for, with every request logged."""

    def __init__(self) -> None:
        self.prepared: list[tuple[str, str, str, datetime]] = []
        self.decisions: list[datetime] = []

    def prepare(self, *, phase, start, end, decision_time):
        self.prepared.append((phase, start, end, decision_time))
        return SnapshotBundle(
            snapshot_id=f"decision_{decision_time:%Y%m%d}",
            decision_ref=f"decision/{decision_time:%Y%m%d}",
            replay_ref=f"replay/{phase}/{start}_{end}",
        )

    def prepare_decision(self, *, decision_time):
        self.decisions.append(decision_time)
        return SnapshotBundle(
            snapshot_id=f"decision_{decision_time:%Y%m%d}",
            decision_ref=f"decision/{decision_time:%Y%m%d}",
            replay_ref="",
        )


class Evaluator:
    """Writes a result record and its style sidecar for the requested span.

    The daily strategy return is ``alpha`` plus CSI 300 and size exposure plus
    noise, so the sidecar carries a measurable neutralised excess and IR.
    ``alpha`` is read from the revision's ``main.py`` (``# alpha=<value>``).
    """

    def __init__(self, root: Path, *, raise_with: BaseException | None = None) -> None:
        self.root = root
        self.requests: list[EvaluationRequest] = []
        self.raise_with = raise_with

    def evaluate(self, request: EvaluationRequest) -> EvaluationResult:
        self.requests.append(request)
        if self.raise_with is not None:
            raise self.raise_with
        source = (Path(request.revision.output_path) / "main.py").read_text(encoding="utf-8")
        alpha = float(source.split("# alpha=")[1].split()[0]) if "# alpha=" in source else 0.0
        return _write_result(
            self.root / f"{request.mode}_{len(self.requests):03d}",
            [day for day in DAYS if request.start <= day <= min(request.end, RELEASE_END)],
            alpha=alpha,
            seed=len(self.requests),
        )


def _write_result(directory: Path, days: list[str], *, alpha: float, seed: int) -> EvaluationResult:
    rng = np.random.default_rng(seed)
    benchmark = rng.normal(0.0003, 0.01, len(days))
    size = rng.normal(0.0, 0.004, len(days))
    strategy = alpha + 0.9 * benchmark + 0.2 * size + rng.normal(0.0, 0.004, len(days))
    equity = 1_000_000 * np.cumprod(1 + strategy)
    curve = [
        {"trade_date": day, "equity": float(value), "initial_equity": 1_000_000.0, "cash": float(value) * 0.1}
        for day, value in zip(days, equity)
    ]
    executions = []
    for index in range(0, len(days) - 5, 10):
        executions.append(
            {"status": "filled", "action": "buy", "price": 10.0, "quantity": 1000,
             "matched_at": f"{days[index][:4]}-{days[index][4:6]}-{days[index][6:]}T09:30:00+08:00"}
        )
        executions.append(
            {"status": "filled", "action": "sell", "price": 10.5, "quantity": 1000, "realized_pnl": 500.0,
             "matched_at": f"{days[index + 5][:4]}-{days[index + 5][4:6]}-{days[index + 5][6:]}T09:30:00+08:00"}
        )
    directory.mkdir(parents=True)
    summary = {
        "total_return": float(equity[-1] / 1_000_000 - 1),
        "max_drawdown": 0.1,
        "sharpe": 1.0,
        "order_count": len(executions),
        "sub_windows": [],
    }
    (directory / "result.json").write_text(
        json.dumps(
            {
                "equity_curve": curve,
                "executions": executions,
                "inference_dates": [f"{day[:4]}-{day[4:6]}-{day[6:]}T08:30:00+08:00" for day in days],
                "stats": summary,
            }
        ),
        encoding="utf-8",
    )
    (directory / "style_analysis.json").write_text(
        json.dumps(
            {
                "strategy_daily": [[day, float(value)] for day, value in zip(days, strategy)],
                "benchmark_daily": [[day, float(value)] for day, value in zip(days, benchmark)],
                "size_factor_daily": [[day, float(value)] for day, value in zip(days, size)],
            }
        ),
        encoding="utf-8",
    )
    return EvaluationResult(summary, str(directory / "result.json"))


class Developer:
    """Scripted attempts: ``plan[k]`` lists the alphas attempt k validates and
    how it ends. Every candidate is a revision validated on the request's span."""

    def __init__(self, store: FilesystemArtifactStore, evaluator: Evaluator, plan):
        self.store = store
        self.evaluator = evaluator
        self.plan = plan
        self.requests: list[ResearchSessionRequest] = []

    def __call__(self, request: ResearchSessionRequest) -> ResearchSessionResult:
        self.requests.append(request)
        alphas, outcome, extra = self.plan[len(self.requests)]
        steps = []
        for number, alpha in enumerate(alphas):
            source = self.store.root.parent / "candidates" / f"{request.run_id}_{number}"
            source.mkdir(parents=True)
            (source / "main.py").write_text(f"{MAIN}# alpha={alpha}\n", encoding="utf-8")
            revision = self.store.create_revision(source)
            typed = ArtifactRevision(str(revision.revision_id), Path(revision.output_path))
            validation = self.evaluator.evaluate(
                request.validation.request(typed, schedule=CONFIG_SCHEDULE, broker_profile=CONFIG_PROFILE)
            )
            steps.append(StepResult(f"{request.session_key}_step_{number}", typed.revision_id, validation, span=request.validation.label))
        # Like the real developer: the Validations earlier attempts recorded
        # are this session's Steps too.
        steps = [*request.steps_before, *steps]
        node_id = extra.get("node_id")
        if outcome == "freeze" and node_id is None:
            node_id = steps[extra.get("nominee", 0)].step_id
        return ResearchSessionResult(
            f"conversation_{request.run_id}",
            tuple(steps),
            outcome,
            node_id=node_id,
            reason=extra.get("reason", ""),
            finish_reason=extra.get("finish_reason", ""),
            attempt=request.resume.attempt if request.resume is not None else 1,
        )


_DEFAULT = RollingExperimentConfig(experiment_id="arm", experiments_root=Path("unused"))
CONFIG_SCHEDULE = _DEFAULT.schedule
CONFIG_PROFILE = _DEFAULT.broker_profile
# A mandated arm end to end. The operator named a tracking_error_cap at
# creation, which is the only thing that turns the mandate on; a blank beta
# band is filled, drawdowns stay the rules' own 0.45 / 0.30.
CONFIG_ACCEPTANCE = acceptance_for({"tracking_error_cap": 0.08})


def _pipeline(tmp_path: Path, plan, *, evaluator_error: BaseException | None = None):
    config = RollingExperimentConfig(
        experiment_id="arm",
        experiments_root=tmp_path / "experiments",
        geometry=GEOMETRY,
        acceptance=CONFIG_ACCEPTANCE,
    )
    store = FilesystemArtifactStore(config.experiment_dir / "artifacts" / "strategy")
    evaluator = Evaluator(config.experiment_dir / "artifacts" / "results", raise_with=evaluator_error)
    snapshots = Snapshots()
    developer = Developer(store, evaluator, plan)
    ledger = ExperimentLedger(config.ledger_path)
    pipeline = RollingExperimentPipeline(
        config,
        snapshots=snapshots,
        artifacts=store,
        evaluator=evaluator,
        developer=developer,
        trading_days=DAYS,
        ledger=ledger,
    )
    return pipeline, snapshots, evaluator, developer, ledger


def _freezing(tmp_path: Path):
    """A session that validates two full-span candidates and nominates the first."""

    return _pipeline(tmp_path, {1: ([0.0012, 0.0002], "freeze", {"nominee": 0})})


def test_the_research_session_reads_only_the_research_period(tmp_path: Path):
    """The Agent's view is the decision view at research end, and every
    Validation replays the research years as one span; nothing the research
    session is given is anchored or stamped after research end."""

    pipeline, snapshots, evaluator, developer, _ledger = _freezing(tmp_path)
    pipeline.run_research_session()

    assert {decision for *_rest, decision in snapshots.prepared} == {
        slot.anchor for slot in GEOMETRY.research_years
    }
    assert all(end <= GEOMETRY.research_end for _phase, _start, end, _d in snapshots.prepared)
    assert snapshots.decisions == [GEOMETRY.research_decision_time]
    [request] = developer.requests
    assert request.session_key == "research"
    assert request.decision_time == GEOMETRY.research_decision_time
    assert request.snapshot.decision_ref == "decision/20240630"
    assert (request.validation.label, request.validation.mode) == ("full", "valid")
    assert (request.validation.start, request.validation.end) == ("20220701", "20240630")
    assert request.validation.snapshot.replay_ref == "replay/valid/20220701_20230630"
    assert request.validation.continuation == ("replay/valid/20230701_20240630",)
    replayed = {(item.start, item.end, item.continuation) for item in evaluator.requests}
    assert replayed == {("20220701", "20240630", ("replay/valid/20230701_20240630",))}


def test_a_freeze_passes_only_the_gate_and_records_the_frozen_block(tmp_path: Path):
    pipeline, _snapshots, _evaluator, _developer, ledger = _freezing(tmp_path)
    record = pipeline.run_research_session()
    gate = record["freeze_gate"]
    assert gate["passed"] is True
    assert gate["full_span_validations"] == 2
    assert gate["deflated_sharpe"]["trials"] == 2 == record["trials_to_date"]
    # The gate is judged under the arm's own rules -- here a tracking mandate,
    # because the arm was created with a cap -- over the research years of its
    # geometry, and the record states the account and the rules.
    rules = CONFIG_ACCEPTANCE
    assert rules.tracking_error_cap == 0.08
    assert record["initial_cash"] == CONFIG_PROFILE.initial_cash
    assert record["acceptance_rules"] == rules.to_record()
    assert gate["thresholds"]["research_years"] == len(GEOMETRY.research_years) == 2
    assert gate["thresholds"]["min_positive_years"] == 2
    assert {
        key: gate["thresholds"][key]
        for key in (
            "active_max_drawdown",
            "tracking_error_cap",
            "beta_min",
            "beta_max",
            "min_full_span_validations",
        )
    } == {
        key: rules.to_record()[key]
        for key in (
            "active_max_drawdown",
            "tracking_error_cap",
            "beta_min",
            "beta_max",
            "min_full_span_validations",
        )
    }
    assert gate["thresholds"]["min_information_ratio"] == rules.min_active_ir
    assert gate["mandate"]["market_beta"] == pytest.approx(0.9, abs=0.05)
    assert record["arm_end"] is None
    frozen = record["frozen"]
    assert frozen["artifact_id"].startswith("strategy_research_")
    assert frozen["source_step_id"] == "research_step_0"
    assert frozen["deflated_sharpe"]["deflated_sharpe_probability"] >= 0.5
    assert frozen["information_ratio"] == pytest.approx(gate["information_ratio"])
    assert frozen["forward_mde"] > 0
    assert frozen["fit_plan"] == {"fit": False, "refit_period": None}
    assert Path(frozen["output_path"], "main.py").is_file()
    assert (record["session_key"], record["fold_id"]) == ("research", "research")
    for retired in ("session_id", "session_index", "sessions_total", "next_start_node_id", "prior"):
        assert retired not in record
    assert frozen_record(ledger.read())["session_key"] == "research"
    assert research_over(ledger.read())
    with pytest.raises(RuntimeError, match="research is over"):
        pipeline.run_research_session()


def test_a_nomination_the_gate_refuses_ends_the_arm_without_a_deliverable(tmp_path: Path):
    # One full-span validation: the gate cannot measure the dispersion, and
    # no other session follows to add one.
    pipeline, *_rest, ledger = _pipeline(tmp_path, {1: ([0.0012], "freeze", {})})
    record = pipeline.run_research_session()
    assert record["frozen"] is None
    assert record["freeze_gate"]["passed"] is False
    assert "freeze_too_few_full_span_validations" in record["freeze_gate"]["reasons"]
    assert record["arm_end"]["status"] == "no_deliverable"
    assert record["arm_end"]["reason"].startswith("freeze refused by the gate (")
    assert "freeze_too_few_full_span_validations" in record["arm_end"]["reason"]
    assert research_over(ledger.read())
    assert experiment_verdict(ledger.read())["status"] == "no_deliverable"
    with pytest.raises(RuntimeError, match="research is over"):
        pipeline.run_research_session()


def test_a_nominee_below_the_deflated_sharpe_threshold_is_not_frozen(tmp_path: Path):
    pipeline, *_rest, ledger = _pipeline(
        tmp_path, {1: ([0.0012, -0.0004], "freeze", {"nominee": 1})}
    )
    record = pipeline.run_research_session()
    assert record["freeze_gate"]["reasons"] == [
        "freeze_information_ratio_below_threshold",
        "freeze_deflated_sharpe_below_threshold",
        "freeze_too_few_positive_years",
        "freeze_active_drawdown_exceeded",
    ]
    assert record["frozen"] is None
    assert record["arm_end"]["status"] == "no_deliverable"
    assert "freeze_deflated_sharpe_below_threshold" in record["arm_end"]["reason"]
    verdict = experiment_verdict(ledger.read())
    assert verdict["status"] == "no_deliverable"
    assert paper_candidate(ledger.read()) is None


def test_the_ledger_refuses_a_second_freeze(tmp_path: Path):
    pipeline, *_rest, ledger = _freezing(tmp_path)
    frozen = pipeline.run_research_session()
    with pytest.raises(ValueError, match="second freeze is refused"):
        ledger.append({**frozen, "run_id": "run_again"})


def test_no_edge_ends_the_arm_without_a_deliverable(tmp_path: Path):
    pipeline, *_rest, ledger = _pipeline(
        tmp_path, {1: ([0.0], "no_edge", {"reason": "the pack's termination rule fired"})}
    )
    record = pipeline.run_research_session()
    assert record["arm_end"] == {
        "status": "no_deliverable",
        "reason": "no_edge: the pack's termination rule fired",
    }
    assert research_over(ledger.read())
    assert experiment_verdict(ledger.read()) == {
        "status": "no_deliverable",
        "reasons": ["no_edge: the pack's termination rule fired"],
    }
    with pytest.raises(RuntimeError, match="needs a frozen artifact"):
        pipeline.run_forward()


# The host tree a dead session container is named after, as the worker sees it.
HOST_WORK_ROOT = f"{Path(__file__).resolve().parents[2]}/.runtime/sandboxes/arm/research"


def _interrupted_attempt(
    pipeline, ledger, *, summary: str | None, replay_years: int, node: bool, capped: bool = False
) -> str:
    """Leave what an interrupted attempt leaves: an attempt_failed row, a
    trace with budget checkpoints (and a compaction), a published tree node
    with its host sidecar and its revision in the store.

    ``capped`` leaves instead what an attempt that filled its trace leaves:
    the terminal marker, and nothing of what it spent afterwards."""

    from autotrade.environment.runtime import agent_trace_path
    from autotrade.environment.step_tree import StepTree
    from autotrade.pipelines.session_resume import record_step_sidecar

    def crash(_request):
        # An environment failure names host paths; the ledger keeps them, the
        # Agent's resume note must not. The path has to be one the host really
        # uses -- the sandbox work root under the repository -- because that is
        # what the redaction recognises; a pytest temp directory is not.
        raise RuntimeError(f"session container died: {HOST_WORK_ROOT}")

    keep = pipeline.developer
    pipeline.developer = crash
    with pytest.raises(RuntimeError):
        pipeline.run_research_session()
    pipeline.developer = keep
    run_id = str(ledger.read()[-1]["run_id"])
    trace = agent_trace_path(pipeline.config.experiment_dir / "artifacts", run_id)
    trace.parent.mkdir(parents=True, exist_ok=True)
    events = [
        {"event_type": "session_start", "ts": "2026-09-15T01:00:00+00:00", "run_id": "run_ref_first"},
        {
            "event_type": "llm_call",
            "ts": "2026-09-15T01:10:00+00:00",
            "run_id": "run_ref_first",
            "budget_used": {"inference_seconds": 600.0, "llm_calls": 7, "main_calls": 5, "subagent_calls": 2, "compact_calls": 0, "replay_years": replay_years, "null_controls": 1},
        },
    ]
    if summary is not None:
        events.append({"event_type": "context_compaction", "ts": "2026-09-15T01:20:00+00:00", "run_id": "run_ref_first", "status": "ok", "trigger": "agent", "summary": summary})
    if capped:
        events.append({"event_type": "trace_limit_reached", "ts": "2026-09-15T01:25:00+00:00", "run_id": "run_ref_first", "max_bytes": 33554432})
    else:
        events.append({"event_type": "session_error", "ts": "2026-09-15T01:30:00+00:00", "run_id": "run_ref_first", "budget_used": {"inference_seconds": 900.0, "llm_calls": 9, "main_calls": 7, "subagent_calls": 2, "compact_calls": 0, "replay_years": replay_years, "null_controls": 1}})
    trace.write_text("".join(json.dumps(event) + "\n" for event in events) + "{torn", encoding="utf-8")
    if not node:
        return run_id
    source = pipeline.config.experiment_dir.parent / "candidates" / "earlier"
    source.mkdir(parents=True)
    (source / "main.py").write_text(f"{MAIN}# alpha=0.0011\n", encoding="utf-8")
    revision = pipeline.artifacts.create_revision(source)
    typed = ArtifactRevision(str(revision.revision_id), Path(revision.output_path))
    validation = pipeline.evaluator.evaluate(
        pipeline.developer.requests[0].validation.request(typed, schedule=CONFIG_SCHEDULE, broker_profile=CONFIG_PROFILE)
        if pipeline.developer.requests
        else _request_span(pipeline).request(typed, schedule=CONFIG_SCHEDULE, broker_profile=CONFIG_PROFILE)
    )
    node_id = StepTree(pipeline.config.experiment_dir / "steps").record_step(
        typed.output_path,
        epoch_id="research",
        session_ref="session_ref_first",
        run_id="run_ref_first",
        result_name="valid_001",
        revision_id="strategy_ref_first",
        metrics={},
        metadata={"span": "full"},
    )
    record_step_sidecar(
        pipeline.config.experiment_dir,
        StepResult(node_id, typed.revision_id, validation, span="full"),
    )
    return run_id


def _request_span(pipeline):
    _decision, years = pipeline.research_inputs()
    from autotrade.pipelines.config import research_span

    return research_span(years, FULL_SPAN)


def test_a_resumed_attempt_continues_from_the_trace_and_the_recorded_node(tmp_path: Path):
    """The next attempt of an interrupted session starts where it stopped: the
    budget the trace last recorded, the last compaction summary, and the
    Validation the interrupted attempt recorded (nominable, its revision kept)."""

    pipeline, _snapshots, _evaluator, developer, ledger = _pipeline(
        tmp_path, {1: ([0.0012], "freeze", {"nominee": 1})}
    )
    _interrupted_attempt(pipeline, ledger, summary="## 目标\n继续动量腿", replay_years=2, node=True)
    revisions = pipeline.artifacts.root / "revisions"
    assert len(list(revisions.iterdir())) == 1, "a failed attempt keeps its revisions"

    record = pipeline.run_research_session()

    request = developer.requests[-1]
    assert request.resume is not None
    assert (request.resume.attempt, request.resume.interrupted_at) == (2, "2026-09-15T01:30:00+00:00")
    assert request.resume.error == "RuntimeError: session container died: [host_path]"
    assert HOST_WORK_ROOT in str(ledger.read()[0]["error"])
    assert request.resume.compaction_summary == "## 目标\n继续动量腿"
    assert request.resume.transcripts == ("run_ref_first.txt",)
    assert request.budget_used.to_record() == {
        "inference_seconds": 900.0, "llm_calls": 9, "main_calls": 7, "subagent_calls": 2,
        "compact_calls": 0, "replay_years": 2, "null_controls": 1,
    }
    [earlier] = request.steps_before
    assert earlier.span == "full" and earlier.validation.summary["order_count"] > 0
    # Two full-span validations: the earlier node and this attempt's; the gate
    # passes and the new node is frozen.
    assert record["steps"][0]["step_id"] == earlier.step_id
    assert record["attempts"] == 2 and record["trials_to_date"] == 2
    assert record["freeze_gate"]["passed"] is True and record["frozen"] is not None
    assert [row["record_type"] for row in ledger.read()] == ["attempt_failed", "research_session"]
    assert len(list(revisions.iterdir())) == 2, "recording the session keeps every validated revision"


def test_a_resumed_attempt_without_a_summary_starts_from_the_note_alone(tmp_path: Path):
    pipeline, _snapshots, _evaluator, developer, ledger = _pipeline(
        tmp_path, {1: ([0.0], "no_edge", {"reason": "nothing survived the second look"})}
    )
    _interrupted_attempt(pipeline, ledger, summary=None, replay_years=0, node=False)
    record = pipeline.run_research_session()
    request = developer.requests[-1]
    assert request.resume is not None and request.resume.compaction_summary is None
    assert request.steps_before == () and request.budget_used.llm_calls == 9
    assert record["attempts"] == 2 and record["arm_end"]["status"] == "no_deliverable"


def test_a_resume_refuses_a_trace_that_stopped_at_its_size_cap(tmp_path: Path):
    """The trace stops at the cap while the attempt keeps spending, so its last
    budget block understates the arm's use; a resume that seeded from it would
    hand the next attempt budget the arm no longer has."""

    pipeline, _snapshots, _evaluator, developer, ledger = _pipeline(
        tmp_path, {1: ([0.0012], "freeze", {"nominee": 1})}
    )
    _interrupted_attempt(pipeline, ledger, summary=None, replay_years=2, node=False, capped=True)
    with pytest.raises(RuntimeError, match="reached its size cap"):
        pipeline.run_research_session()
    assert developer.requests == []
    assert [row["record_type"] for row in ledger.read()] == ["attempt_failed"]


def test_a_complete_node_without_its_sidecar_fails_the_attempt(tmp_path: Path):
    from autotrade.environment.step_tree import StepTree

    pipeline, *_rest, ledger = _freezing(tmp_path)
    output = tmp_path / "orphan"
    output.mkdir()
    (output / "main.py").write_text(MAIN, encoding="utf-8")
    StepTree(pipeline.config.experiment_dir / "steps").record_step(
        output, epoch_id="research", session_ref="session_ref_first", run_id="run_ref_first",
        result_name="valid_001", revision_id="strategy_ref_first", metrics={},
    )
    with pytest.raises(RuntimeError, match="without its host sidecar"):
        pipeline.run_research_session()
    assert [row["record_type"] for row in ledger.read()] == []


def test_an_exhausted_budget_ends_the_arm_without_a_deliverable(tmp_path: Path):
    pipeline, *_rest, ledger = _pipeline(
        tmp_path, {1: ([0.0], "deadline", {"finish_reason": "llm_call_budget_exhausted"})}
    )
    record = pipeline.run_research_session()
    assert (record["outcome"], record["frozen"]) == ("deadline", None)
    assert record["arm_end"] == {
        "status": "no_deliverable",
        "reason": "research budget exhausted without a freeze (llm_call_budget_exhausted)",
    }
    assert research_over(ledger.read())
    assert experiment_verdict(ledger.read())["status"] == "no_deliverable"


def test_a_failed_attempt_is_recorded_and_the_session_runs_again(tmp_path: Path):
    pipeline, _snapshots, _evaluator, developer, ledger = _pipeline(
        tmp_path,
        {1: ([0.0012, 0.0002], "freeze", {"nominee": 0}), 2: ([0.0012, 0.0002], "freeze", {"nominee": 0})},
    )
    keep_skills = pipeline._publish_or_keep_skills

    def fail_before_the_record(*_args, **_kwargs):
        raise OSError("skills store unavailable")

    pipeline._publish_or_keep_skills = fail_before_the_record
    with pytest.raises(OSError):
        pipeline.run_research_session()
    assert [row["record_type"] for row in ledger.read()] == ["attempt_failed"]
    assert not research_over(ledger.read())

    pipeline._publish_or_keep_skills = keep_skills
    record = pipeline.run_research_session()
    assert record["frozen"] is not None
    assert len(developer.requests) == 2
    assert [row["record_type"] for row in ledger.read()] == [
        "attempt_failed",
        "research_session",
    ]


def test_the_forward_replay_is_one_span_from_forward_start_to_the_release(tmp_path: Path):
    pipeline, snapshots, evaluator, _developer, ledger = _freezing(tmp_path)
    frozen = pipeline.run_research_session()["frozen"]
    snapshots.prepared.clear()
    evaluator.requests.clear()

    record = pipeline.run_forward()

    assert snapshots.prepared == [
        ("heldout", "20240701", "20250630", GEOMETRY.forward.anchor),
        ("heldout", "20250701", RELEASE_END, GEOMETRY.heldout(DAYS).anchor),
    ]
    [request] = evaluator.requests
    assert (request.mode, request.start, request.end) == ("heldout", "20240701", RELEASE_END)
    assert request.continuation == (f"replay/heldout/20250701_{RELEASE_END}",)
    assert request.revision.revision_id == frozen["artifact_id"]
    assert record["replay"] == {
        "start": "20240701",
        "forward_end": "20250630",
        "heldout_start": "20250701",
        "replay_end": RELEASE_END,
        "requested_end": "20250930",
        "truncation_reason": f"release_ends_{RELEASE_END}",
    }
    assert record["status"] == "ok"
    forward, heldout = record["slices"]["forward"], record["slices"]["heldout"]
    assert (forward["start"], forward["end"]) == ("20240701", "20250630")
    assert (heldout["start"], heldout["end"]) == ("20250701", RELEASE_END)
    assert forward["activity"]["round_trips"] > 0
    # The Held-out tolerance is scaled by the forward slice's tracking error.
    assert heldout["tolerance"] < 0
    # The verdict is held to the arm's own rules and its record states them.
    rules = CONFIG_ACCEPTANCE.to_record()
    assert record["initial_cash"] == CONFIG_PROFILE.initial_cash
    assert record["acceptance_rules"] == rules
    graduation_keys = (
        "max_drawdown",
        "active_max_drawdown",
        "tracking_error_cap",
        "beta_min",
        "beta_max",
        "cost_stress_multiplier",
        "forward_confidence",
        "recency_months",
        "min_mean_gross",
        "heldout_tolerance_z",
    )
    assert {key: record["verdict"]["thresholds"][key] for key in graduation_keys} == {
        key: rules[key] for key in graduation_keys
    }
    assert set(forward["mandate"]) == {"tracking_error", "market_beta"}
    assert record["refits_executed"] == {"forward": 0, "heldout": 0}
    assert record["verdict"]["status"] in {"graduated", "discarded"}
    assert experiment_verdict(ledger.read())["status"] == record["verdict"]["status"]
    assert forward_record(ledger.read())["run_id"] == record["run_id"]
    with pytest.raises(RuntimeError, match="already recorded"):
        pipeline.run_forward()


def test_a_strategy_error_names_the_slice_it_raised_in_and_discards(tmp_path: Path):
    pipeline, _snapshots, evaluator, _developer, ledger = _freezing(tmp_path)
    pipeline.run_research_session()
    failure = BacktestError(
        "generate_orders failed at 2025-07-02T08:30:00+08:00: boom",
        inference_at=datetime(2025, 7, 2, 8, 30, tzinfo=CN_TZ),
    )
    failure.__cause__ = StrategyRaised("boom")
    evaluator.raise_with = failure

    record = pipeline.run_forward()

    assert record["status"] == "strategy_error"
    assert record["slices"] is None
    assert record["verdict"] == {
        "status": "discarded",
        "reasons": ["heldout_strategy_error"],
        "thresholds": {},
    }
    assert experiment_verdict(ledger.read())["status"] == "discarded"
    assert paper_candidate(ledger.read()) is None


def test_an_environment_failure_fails_the_attempt_and_leaves_no_verdict(tmp_path: Path):
    pipeline, _snapshots, evaluator, _developer, ledger = _freezing(tmp_path)
    pipeline.run_research_session()
    evaluator.raise_with = TimeoutError("strategy inference exceeded 360s")

    with pytest.raises(TimeoutError):
        pipeline.run_forward()

    [failed] = [row for row in ledger.read() if row["record_type"] == "attempt_failed"]
    assert (failed["phase"], failed["session_key"]) == ("forward", "forward")
    assert forward_record(ledger.read()) is None
    assert experiment_verdict(ledger.read()) is None
    assert sorted(RunMarkers(pipeline.config.experiment_dir).root.glob("*.json")) == []
    # The retry replays from forward start and records the verdict.
    evaluator.raise_with = None
    assert pipeline.run_forward()["status"] == "ok"


def test_an_unmeasurable_slice_fails_the_attempt(tmp_path: Path):
    pipeline, _snapshots, evaluator, _developer, ledger = _freezing(tmp_path)
    pipeline.run_research_session()

    class Unmeasurable(Evaluator):
        def evaluate(self, request):
            result = super().evaluate(request)
            sidecar = Path(result.result_ref).parent / "style_analysis.json"
            analysis = json.loads(sidecar.read_text(encoding="utf-8"))
            analysis["size_factor_daily"] = []
            sidecar.write_text(json.dumps(analysis), encoding="utf-8")
            return result

    pipeline.evaluator = Unmeasurable(evaluator.root / "unmeasurable")
    with pytest.raises(ValueError, match="not measurable"):
        pipeline.run_forward()
    assert [row["record_type"] for row in ledger.read()][-1] == "attempt_failed"
    assert forward_record(ledger.read()) is None


def test_frozen_trees_changed_during_the_replay_fail_closed(tmp_path: Path):
    pipeline, _snapshots, evaluator, _developer, ledger = _freezing(tmp_path)
    frozen = pipeline.run_research_session()["frozen"]
    main_py = Path(frozen["output_path"]) / "main.py"

    class Mutating(Evaluator):
        def evaluate(self, request):
            main_py.chmod(0o644)
            main_py.write_text(MAIN + "# edited\n", encoding="utf-8")
            return super().evaluate(request)

    pipeline.evaluator = Mutating(evaluator.root / "mutating")
    with pytest.raises(FrozenArtifactMutated):
        pipeline.run_forward()
    rows = ledger.read()
    assert rows[-1]["record_type"] == "forward"
    assert rows[-1]["state_changed_during_test"] is True
    assert not any(row["record_type"] == "attempt_failed" for row in rows)
    # Restored to the frozen bytes, and every later stage refuses.
    assert "# edited" not in main_py.read_text(encoding="utf-8")
    assert forward_record(rows) is None
    with pytest.raises(FrozenArtifactMutated):
        pipeline.run_forward()


def test_a_failing_session_records_attempt_failed_and_clears_its_marker(tmp_path: Path):
    pipeline, *_rest, ledger = _freezing(tmp_path)

    def terminated(_request):
        raise SystemExit(143)

    pipeline.developer = terminated
    with pytest.raises(SystemExit):
        pipeline.run_research_session()
    [failed] = [row for row in ledger.read() if row["record_type"] == "attempt_failed"]
    assert (failed["phase"], failed["session_key"], failed["fold_id"]) == ("research", "research", "research")
    assert failed["error"] == "SystemExit: 143"
    markers = RunMarkers(pipeline.config.experiment_dir)
    assert sorted(markers.root.glob("*.json")) == []
    assert markers.recover(ledger) == []


def test_a_session_keeps_its_marker_when_its_failure_cannot_be_recorded(tmp_path: Path):
    pipeline, *_rest, ledger = _freezing(tmp_path)
    original_append = ledger.append

    def refusing_append(record):
        if record.get("record_type") == "attempt_failed":
            raise OSError("ledger is read-only")
        return original_append(record)

    def crashing(_request):
        raise RuntimeError("session crashed")

    pipeline.developer = crashing
    ledger.append = refusing_append  # type: ignore[method-assign]
    try:
        with pytest.raises(OSError, match="ledger is read-only"):
            pipeline.run_research_session()
    finally:
        del ledger.append
    markers = RunMarkers(pipeline.config.experiment_dir)
    appended = markers.recover(ledger)
    assert [(row["error"], row["phase"]) for row in appended] == [
        (INTERRUPTED_RUN_ERROR, "research")
    ]
    assert markers.recover(ledger) == []


def test_an_unreadable_run_marker_becomes_one_attempt_failed_and_is_cleared(tmp_path: Path):
    """A torn marker is evidence, not a brick: it becomes exactly one
    ``attempt_failed`` with the link keys it cannot supply named unknown."""

    pipeline, *_rest, ledger = _freezing(tmp_path)
    markers = RunMarkers(pipeline.config.experiment_dir)
    markers.root.mkdir(parents=True, exist_ok=True)
    (markers.root / "run_empty.json").write_text("", encoding="utf-8")

    [row] = markers.recover(ledger)
    assert (row["run_id"], row["epoch_id"], row["fold_id"]) == (
        "run_empty",
        UNKNOWN_MARKER_LINK_KEY,
        UNKNOWN_MARKER_LINK_KEY,
    )
    assert row["error"].startswith(UNREADABLE_RUN_MARKER_ERROR)
    assert markers.recover(ledger) == []


def test_session_budgets_honor_nondefault_deadline_grace(tmp_path: Path):
    config = replace(
        RollingExperimentConfig(experiment_id="arm", experiments_root=tmp_path),
        max_research_minutes=60,
        deadline_grace_minutes=5,
    )
    budgets = _session_budgets(config, None)
    assert budgets["deadline_seconds"] == 65 * 60
    assert budgets["deadline_grace_seconds"] == 5 * 60


def test_the_null_control_seed_is_stable_per_key_and_role():
    assert null_control_seed("research", "frozen") == null_control_seed("research", "frozen")
    assert null_control_seed("research", "frozen") != null_control_seed("other", "frozen")
    assert null_control_seed("strategy_x", "forward") != null_control_seed("strategy_x", "frozen")


def test_a_step_row_refuses_a_missing_style_sidecar(tmp_path: Path):
    """A missing sidecar is a broken replay, not an unmeasurable span: dropping
    the row would silently narrow the freeze gate's IR dispersion."""

    result = EvaluationResult({}, str(tmp_path / "result.json"))
    step = StepResult("research_step_1", "rev_1", result, span=FULL_SPAN)
    with pytest.raises(FileNotFoundError):
        research_step_record(step)
    (tmp_path / STYLE_ARTIFACT_NAME).write_text("{}", encoding="utf-8")
    assert research_step_record(step)["neutralized"] is None
