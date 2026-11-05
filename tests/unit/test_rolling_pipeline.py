"""The research arm's orchestration: one research session, one freeze, one forward replay.

In-memory providers and a result writer stand in for the PIT backend; the real
artifact store, ledger, run markers and verdict statistics run unchanged.
``test_research_arm_worker.py`` drives the same stages through the real worker
and the real PIT replay.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from autotrade.environment.artifacts import FilesystemArtifactStore
from autotrade.environment.broker import BrokerProfile
from autotrade.environment.executor import StrategyMemoryExceeded, StrategyRaised
from autotrade.environment.replay.engine import BacktestError
from autotrade.environment.replay.style import STYLE_ARTIFACT_NAME
from autotrade.environment.strategy import CN_TZ
from autotrade.pipelines.calendar import FULL_SPAN, ResearchGeometry
from autotrade.pipelines.config import (
    EvaluationRequest,
    EvaluationResult,
    ResearchSessionRequest,
    ResearchSessionResult,
    RollingExperimentConfig,
    SnapshotBundle,
    StepResult,
    acceptance_for,
    research_span,
)
from autotrade.pipelines.experiment import (
    RollingExperimentPipeline,
    _session_budgets,
    null_control_seed,
    research_step_record,
)
from autotrade.pipelines.ledger import (
    INTERRUPTED_RUN_ERROR,
    TEST_STEP_RECORD_TYPE,
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
from tests.unit.acceptance_fixture import REQUIRED_RULES, RULES

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
    noise, so the sidecar carries a measurable active excess and IR.
    ``alpha`` is read from the revision's ``main.py`` (``# alpha=<value>``).
    """

    def __init__(self, root: Path, *, raise_with: BaseException | None = None) -> None:
        self.root = root
        self.requests: list[EvaluationRequest] = []
        self.raise_with = raise_with
        # The one replay mode ``raise_with`` is raised in, when set.
        self.raise_in: str | None = None
        # Every replay this evaluator ran, which names its result: a test may
        # clear ``requests`` between stages.
        self.count = 0

    def evaluate(self, request: EvaluationRequest) -> EvaluationResult:
        self.requests.append(request)
        self.count += 1
        if self.raise_with is not None and self.raise_in in (None, request.mode):
            raise self.raise_with
        source = (Path(request.revision.output_path) / "main.py").read_text(encoding="utf-8")
        marked = source.split("# alpha=")[1].split()[0] if "# alpha=" in source else "0.0"
        result = _write_result(
            self.root / f"{request.mode}_{self.count:03d}_{uuid.uuid4().hex[:6]}",
            [day for day in DAYS if request.start <= day <= min(request.end, RELEASE_END)],
            alpha=0.0 if marked == "flat" else float(marked),
            seed=self.count,
        )
        if marked == "flat":
            # A book that earns exactly what its zero-skill panel earns, as
            # one that stays in cash does against an empty panel.
            sidecar = Path(result.result_ref).parent / "style_analysis.json"
            analysis = json.loads(sidecar.read_text(encoding="utf-8"))
            analysis["strategy_daily"] = analysis["panel_daily"]
            sidecar.write_text(json.dumps(analysis), encoding="utf-8")
        return result


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
    total_return = float(equity[-1] / 1_000_000 - 1)
    summary = {
        "total_return": total_return,
        "max_drawdown": 0.1,
        "sharpe": 1.0,
        "order_count": len(executions),
        "sub_windows": [],
        "benchmark": {"excess_return": total_return - float(np.prod(1 + benchmark) - 1)},
        # What the freeze gate's cost stress reads: the slippage, charged
        # again on the replay's turnover.
        "turnover": 1.0,
        "cost_sensitivity": {"slippage_bps": 5.0, "cost_per_bp_per_side": 1e-4},
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
                # A value leg the book does not load on, from its own draws.
                "value_factor_daily": [
                    [day, float(value)]
                    for day, value in zip(
                        days, np.random.default_rng(seed + 10_000).normal(0.0, 0.003, len(days))
                    )
                ],
                # The zero-skill panel: random names on the book's skeleton,
                # riding the market as the book does.
                "panel_daily": [[day, float(0.9 * value)] for day, value in zip(days, benchmark)],
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
            typed, _fingerprint = self.store.create_revision(source)
            validation = self.evaluator.evaluate(
                request.full_span.request(typed, schedule=CONFIG_SCHEDULE, broker_profile=CONFIG_PROFILE)
            )
            steps.append(
                StepResult(
                    f"{request.session_key}_step_{number}",
                    typed.revision_id,
                    validation,
                    span=request.full_span.label,
                    control=number in extra.get("controls", ()),
                )
            )
        # Like the real developer: the Validations earlier attempts recorded
        # are this session's Steps too.
        steps = [*request.steps_before, *steps]
        node_id = extra.get("node_id")
        if outcome == "freeze" and node_id is None:
            node_id = steps[extra.get("nominee", 0)].step_id
        # The plan's ``freeze`` in the session's own terms: a session whose
        # last test took the nominee.
        outcome = {"freeze": "completed"}.get(outcome, outcome)
        return ResearchSessionResult(
            f"conversation_{request.run_id}",
            tuple(steps),
            outcome,
            node_id=node_id,
            reason=extra.get("reason", ""),
            finish_reason=extra.get("finish_reason", ""),
            attempt=request.resume.attempt if request.resume is not None else 1,
        )


_DEFAULT = RollingExperimentConfig(experiment_id="arm", experiments_root=Path("unused"), acceptance=RULES)
CONFIG_SCHEDULE = _DEFAULT.schedule
CONFIG_PROFILE = _DEFAULT.broker_profile
# A mandated arm end to end. The operator named a tracking_error_cap at
# creation, which is the only thing that turns the mandate on; a blank beta
# band is filled, drawdowns stay the rules' own 0.45 / 0.30.
CONFIG_ACCEPTANCE = acceptance_for({**REQUIRED_RULES, "tracking_error_cap": 0.08})


def _pipeline(
    tmp_path: Path,
    plan,
    *,
    evaluator_error: BaseException | None = None,
    broker_profile: BrokerProfile = CONFIG_PROFILE,
):
    config = RollingExperimentConfig(
        experiment_id="arm",
        experiments_root=tmp_path / "experiments",
        geometry=GEOMETRY,
        acceptance=CONFIG_ACCEPTANCE,
        broker_profile=broker_profile,
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


def test_the_research_session_opens_at_the_validation_start(tmp_path: Path):
    """The arm researches on its first year and validates through its second:
    the Agent's view is the decision view at the end of the research year,
    ``full`` is the research year alone, and the request carries every year
    of the geometry so a test can replay the validation year."""

    pipeline, snapshots, evaluator, developer, _ledger = _freezing(tmp_path)
    pipeline.run_research_session()

    research, validation = GEOMETRY.split_research(CONFIG_ACCEPTANCE.research_years)
    assert [slot.label for slot in research] == ["Y1"] and [slot.label for slot in validation] == ["Y2"]
    validation_start = validation[0].start
    assert snapshots.decisions == [validation[0].anchor]
    [request] = developer.requests
    assert request.session_key == "research"
    assert request.decision_time == validation[0].anchor < datetime(2023, 7, 1, tzinfo=CN_TZ)
    assert request.snapshot.decision_ref == "decision/20230630"
    assert [year.label for year in request.years] == ["Y1", "Y2"]
    assert request.research_years == 1
    assert (request.full_span.label, request.full_span.mode) == ("full", "valid")
    assert (request.full_span.start, request.full_span.end) == ("20220701", "20230630")
    # ``full`` and the research year end before the validation start; the
    # validation year parses but is the frontier's to allow.
    for label in ("full", "Y1"):
        assert research_span(request.years, label, research_years=1).end < validation_start
    assert research_span(request.years, "Y2", research_years=1).start == validation_start
    assert all(item.end < validation_start for item in evaluator.requests)
    replayed = {(item.start, item.end, item.continuation) for item in evaluator.requests}
    assert replayed == {("20220701", "20230630", ())}


def test_a_freeze_registers_and_records_the_frozen_block(tmp_path: Path):
    pipeline, _snapshots, _evaluator, _developer, ledger = _freezing(tmp_path)
    record = pipeline.run_research_session()
    gate = record["freeze_gate"]
    assert gate["passed"] is True and gate["reasons"] == []
    # Registration reads no score: no IR, no trial count, no bar.
    assert set(gate) == {"passed", "reasons", "mandate", "seed_replicates", "thresholds"}
    assert "trials_to_date" not in record and "terminal_offline_trials" not in record
    assert [row["control"] for row in record["steps"]] == [False, False]
    # It is judged under the arm's own rules -- here a tracking mandate,
    # because the arm was created with a cap -- and the record states the
    # account and the rules.
    rules = CONFIG_ACCEPTANCE
    assert rules.tracking_error_cap == 0.08
    assert record["initial_cash"] == CONFIG_PROFILE.initial_cash
    assert record["acceptance_rules"] == rules.to_record()
    assert gate["thresholds"] == {"max_drawdown": rules.max_drawdown, **rules.mandate}
    assert gate["mandate"]["market_beta"] == pytest.approx(0.9, abs=0.05)
    assert record["arm_end"] is None
    frozen = record["frozen"]
    assert frozen["artifact_id"].startswith("strategy_research_")
    assert frozen["source_step_id"] == "research_step_0"
    nominee = record["steps"][0]
    assert frozen["information_ratio"] == pytest.approx(nominee["graded"]["information_ratio"])
    # The statistic the sealed gate grades, read on the visible year only.
    assert frozen["residual_reading"]["years"] == pytest.approx(
        nominee["graded"]["days"] / 244 - 1 / 244, abs=0.02
    )
    assert "deflated_sharpe" not in frozen and "full_span_validations" not in frozen
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


def test_a_completed_session_whose_candidate_does_not_register_fails_the_attempt(tmp_path: Path):
    """``test`` refused every candidate that does not register, so a completed
    session names one that does; a candidate that no longer registers is a
    broken invariant, failed loudly and recorded as a failed attempt rather
    than frozen or quietly ended."""

    for reason, plan in (
        ("freeze_nominee_is_control", {1: ([0.0012, 0.0002], "freeze", {"nominee": 1, "controls": (1,)})}),
        # An all-zero active series has a plain reading but no residual one.
        ("freeze_unmeasurable", {1: (["flat"], "freeze", {})}),
    ):
        pipeline, *_rest, ledger = _pipeline(tmp_path / reason, plan)
        with pytest.raises(RuntimeError, match=f"no longer registers: {reason}"):
            pipeline.run_research_session()
        assert [row["record_type"] for row in ledger.read()] == ["attempt_failed"]
        assert not research_over(ledger.read())


def test_a_freeze_is_terminal_whatever_the_nominee_read(tmp_path: Path):
    """A lone full-span nominee that lost money every year registers and is
    frozen: the session never learns a score, so nothing about its readings
    can keep it from freezing."""

    pipeline, *_rest, ledger = _pipeline(tmp_path, {1: ([-0.0008], "freeze", {})})
    record = pipeline.run_research_session()
    assert record["steps"][0]["graded"]["plain_excess"] < 0
    assert record["freeze_gate"]["passed"] is True
    assert record["frozen"] is not None and record["arm_end"] is None
    assert research_over(ledger.read())
    assert experiment_verdict(ledger.read()) is None


def test_the_ledger_refuses_a_second_freeze(tmp_path: Path):
    pipeline, *_rest, ledger = _freezing(tmp_path)
    frozen = pipeline.run_research_session()
    with pytest.raises(ValueError, match="second freeze is refused"):
        ledger.append({**frozen, "run_id": "run_again"})


def test_a_budget_that_ends_before_any_test_freezes_the_last_registrable_candidate(tmp_path: Path):
    """The host completes what the session did not: the last full-span,
    non-control Step that registers is frozen, and the rolling record is
    filled from one replay of it over the whole validation period, each step
    read off that replay's window and marked as the host's completion."""

    for name, plan, frozen_step in (
        ("last", {1: ([0.0012, 0.0002], "deadline", {"finish_reason": "llm_call_budget_exhausted"})}, 1),
        ("control_skipped", {1: ([0.0012, 0.0002], "deadline", {"controls": (1,)})}, 0),
    ):
        pipeline, _snapshots, evaluator, _developer, ledger = _pipeline(tmp_path / name, plan)
        record = pipeline.run_research_session()
        assert (record["outcome"], record["arm_end"]) == ("deadline", None)
        assert record["frozen"]["source_step_id"] == f"research_step_{frozen_step}"
        assert record["freeze_gate"]["passed"] is True
        steps = pipeline.validation_steps()
        rows = [row for row in ledger.read() if row["record_type"] == TEST_STEP_RECORD_TYPE]
        assert [row["step"] for row in rows] == list(range(1, steps.count + 1))
        assert [row["frontier"] for row in rows] == [steps.frontier_after(k) for k in range(1, steps.count + 1)]
        assert {row["source"] for row in rows} == {"host_completion"}
        assert {row["candidate"]["step_id"] for row in rows} == {f"research_step_{frozen_step}"}
        assert (record["frontier"], record["steps_tested"]) == (steps.end, steps.count)
        # One replay filled every entry, over the validation span as one book.
        completion = evaluator.requests[-1]
        assert (completion.start, completion.end) == (steps.start, steps.end)
        assert len({row["replay"]["result_ref"] for row in rows}) == 1
        assert research_over(ledger.read())
        assert experiment_verdict(ledger.read()) is None


HOST_WORK_ROOT = f"{Path(__file__).resolve().parents[2]}/.runtime/sandboxes/arm/research"


def _interrupted_attempt(
    pipeline, ledger, *, summary: str | None, test_replays: int, node: bool, capped: bool = False
) -> str:
    """Leave what an interrupted attempt leaves: an attempt_failed row, a
    trace with budget checkpoints (and a compaction), a published tree node
    with its host sidecar and its revision in the store.

    ``capped`` leaves instead what an attempt that filled its trace leaves:
    the terminal marker, and nothing of what it spent afterwards."""

    from autotrade.environment.step_tree import StepTree
    from autotrade.environment.trace import agent_trace_path
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
            "budget_used": {"inference_seconds": 600.0, "llm_calls": 7, "main_calls": 5, "subagent_calls": 2, "compact_calls": 0, "test_replays": test_replays, "null_controls": 1},
        },
    ]
    if summary is not None:
        events.append({"event_type": "context_compaction", "ts": "2026-09-15T01:20:00+00:00", "run_id": "run_ref_first", "status": "ok", "trigger": "agent", "summary": summary})
    if capped:
        events.append({"event_type": "trace_limit_reached", "ts": "2026-09-15T01:25:00+00:00", "run_id": "run_ref_first", "max_bytes": 33554432})
    else:
        events.append({"event_type": "session_error", "ts": "2026-09-15T01:30:00+00:00", "run_id": "run_ref_first", "budget_used": {"inference_seconds": 900.0, "llm_calls": 9, "main_calls": 7, "subagent_calls": 2, "compact_calls": 0, "test_replays": test_replays, "null_controls": 1}})
    trace.write_text("".join(json.dumps(event) + "\n" for event in events) + "{torn", encoding="utf-8")
    if not node:
        return run_id
    source = pipeline.config.experiment_dir.parent / "candidates" / "earlier"
    source.mkdir(parents=True)
    (source / "main.py").write_text(f"{MAIN}# alpha=0.0011\n", encoding="utf-8")
    typed, _fingerprint = pipeline.artifacts.create_revision(source)
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
    return research_span(years, FULL_SPAN)


def test_a_resumed_attempt_continues_from_the_trace_and_the_recorded_node(tmp_path: Path):
    """The next attempt of an interrupted session starts where it stopped: the
    budget the trace last recorded, the last compaction summary, and the
    Validation the interrupted attempt recorded (nominable, its revision kept)."""

    pipeline, _snapshots, _evaluator, developer, ledger = _pipeline(
        tmp_path, {1: ([0.0012], "freeze", {"nominee": 1})}
    )
    _interrupted_attempt(pipeline, ledger, summary="## 目标\n继续动量腿", test_replays=2, node=True)
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
        "compact_calls": 0, "test_replays": 2, "null_controls": 1, "api_spend_usd": 0.0,
    }
    [earlier] = request.steps_before
    assert earlier.span == "full" and earlier.validation.summary["order_count"] > 0
    # Two full-span validations: the earlier node and this attempt's; the
    # nominee registers and is frozen.
    assert record["steps"][0]["step_id"] == earlier.step_id
    assert record["attempts"] == 2 and len(record["steps"]) == 2
    assert record["freeze_gate"]["passed"] is True and record["frozen"] is not None
    assert [row["record_type"] for row in ledger.read()] == ["attempt_failed", "research_session"]
    assert len(list(revisions.iterdir())) == 2, "recording the session keeps every validated revision"


def test_a_resumed_attempt_without_a_summary_starts_from_the_note_alone(tmp_path: Path):
    pipeline, _snapshots, _evaluator, developer, ledger = _pipeline(
        tmp_path, {1: ([], "deadline", {"finish_reason": "llm_call_budget_exhausted"})}
    )
    _interrupted_attempt(pipeline, ledger, summary=None, test_replays=0, node=False)
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
    _interrupted_attempt(pipeline, ledger, summary=None, test_replays=2, node=False, capped=True)
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


def test_a_moved_arm_resumes_from_the_recorded_node_where_it_now_lives(tmp_path: Path):
    """The records an interrupted attempt left name the host paths they were
    written at; once the repository moves, the next attempt reads the
    recorded Validation under the arm's current directory and records it
    there. A sidecar naming a place outside the arm, or a result no longer on
    disk, is refused."""

    from autotrade.pipelines.session_resume import STEP_SIDECAR_DIR, load_recorded_steps

    plan = {1: ([0.0012], "freeze", {"nominee": 1})}
    old, _snapshots, old_evaluator, _developer, old_ledger = _pipeline(tmp_path / "old", plan)
    _interrupted_attempt(old, old_ledger, summary=None, test_replays=2, node=True)
    (tmp_path / "old").rename(tmp_path / "new")
    pipeline, _snapshots, evaluator, developer, _ledger = _pipeline(tmp_path / "new", plan)
    # The fixture names results by count: number on after the moved ones.
    evaluator.requests[:] = old_evaluator.requests
    experiment = pipeline.config.experiment_dir
    [sidecar] = (experiment / STEP_SIDECAR_DIR).glob("*.json")
    recorded = json.loads(sidecar.read_text(encoding="utf-8"))
    assert recorded["result_ref"].startswith(str(tmp_path / "old"))

    record = pipeline.run_research_session()

    [earlier] = developer.requests[-1].steps_before
    assert Path(earlier.validation.result_ref).is_relative_to(experiment / "artifacts" / "results")
    assert record["steps"][0]["validation_result_ref"] == earlier.validation.result_ref
    assert record["freeze_gate"]["passed"] is True and record["frozen"] is not None
    for reference, refusal in (
        (str(tmp_path / "new/experiments/other_arm/artifacts/results/valid_001/result.json"), "outside the experiment"),
        (str(Path(recorded["result_ref"]).with_name("absent.json")), "not on disk"),
    ):
        sidecar.write_text(json.dumps({**recorded, "result_ref": reference}), encoding="utf-8")
        with pytest.raises(FileNotFoundError, match=refusal):
            load_recorded_steps(experiment)


def test_an_exhausted_budget_without_a_registrable_candidate_ends_the_arm_without_a_deliverable(
    tmp_path: Path,
):
    pipeline, *_rest, ledger = _pipeline(
        tmp_path, {1: ([], "deadline", {"finish_reason": "llm_call_budget_exhausted"})}
    )
    record = pipeline.run_research_session()
    assert (record["outcome"], record["frozen"]) == ("deadline", None)
    assert record["arm_end"] == {
        "status": "no_deliverable",
        "reason": (
            "research budget exhausted before any test, with no registrable full-span "
            "candidate (llm_call_budget_exhausted)"
        ),
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

    # The research years' slots are prepared again for the sealed reading,
    # then the forward and Held-out slots.
    assert snapshots.prepared[-2:] == [
        ("heldout", "20240701", "20250630", GEOMETRY.forward.anchor),
        ("heldout", "20250701", RELEASE_END, GEOMETRY.heldout(DAYS).anchor),
    ]
    # First the sealed year, as one book from the view at its start, then the
    # one continuous forward and Held-out replay.
    sealed_request, request = evaluator.requests
    assert (sealed_request.mode, sealed_request.start, sealed_request.end) == ("valid", "20230701", "20240630")
    assert sealed_request.revision.revision_id == frozen["artifact_id"]
    sealed = record["sealed"]
    assert (sealed["label"], sealed["start"], sealed["end"]) == ("validation", "20230701", "20240630")
    assert sealed["result_ref"] == str(Path(sealed["result_ref"]))
    reading = sealed["slice"]
    assert reading["series"] == "active" and reading["years"] == pytest.approx(reading["days"] / 244)
    assert reading["bar"] == pytest.approx(1.6449 / (reading["days"] / 244) ** 0.5, abs=1e-3)
    assert reading["reasons"] == (
        [] if reading["residual_information_ratio"] >= reading["bar"] else ["sealed_residual_ir_below_bar"]
    )
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
    # Both slices are read and reported, their residual reading beside them.
    for block in (forward, heldout):
        assert "reasons" not in block and block["residual_reading"]["series"] == "active"
    # The verdict is the sealed gate's alone, under the arm's own rules, and
    # the record states them.
    rules = CONFIG_ACCEPTANCE.to_record()
    assert record["initial_cash"] == CONFIG_PROFILE.initial_cash
    assert record["acceptance_rules"] == rules
    assert record["verdict"]["thresholds"] == reading["thresholds"]
    assert record["verdict"]["reasons"] == reading["reasons"]
    assert forward["parameters"]["cost_stress_multiplier"] == rules["cost_stress_multiplier"]
    assert set(forward["mandate"]) == {"tracking_error", "market_beta"}
    assert record["refits_executed"] == {"forward": 0, "heldout": 0}
    assert record["verdict"]["status"] == ("discarded" if reading["reasons"] else "graduated")
    assert experiment_verdict(ledger.read())["status"] == record["verdict"]["status"]
    assert forward_record(ledger.read())["run_id"] == record["run_id"]
    with pytest.raises(RuntimeError, match="already recorded"):
        pipeline.run_forward()


def test_the_forward_replay_trades_under_the_arms_own_cost_model(tmp_path: Path):
    """The verdict replays the frozen book on the Broker profile its arm was
    created with, the dividend tax included, not on today's defaults."""

    taxed = replace(CONFIG_PROFILE, dividend_tax=True)
    pipeline, _snapshots, evaluator, _developer, _ledger = _pipeline(
        tmp_path,
        {1: ([0.0012, 0.0002], "freeze", {"nominee": 0})},
        broker_profile=taxed,
    )
    pipeline.run_research_session()
    evaluator.requests.clear()
    pipeline.run_forward()
    # The sealed reading and the forward replay alike.
    assert len(evaluator.requests) == 2
    assert all(request.broker_profile == taxed for request in evaluator.requests)


def test_a_moved_arm_replays_the_artifact_its_frozen_block_names(tmp_path: Path):
    """The frozen block names the artifact's output where it was frozen; on an
    arm moved since, the forward stage holds the store to that path read under
    the arm's current directory. A block naming a place outside the arm, or
    one not on disk, fails the attempt."""

    _freezing(tmp_path / "old")[0].run_research_session()
    (tmp_path / "old").rename(tmp_path / "new")
    pipeline, *_rest, ledger = _freezing(tmp_path / "new")
    records = ledger.read()
    frozen = frozen_record(records)["frozen"]
    assert frozen["output_path"].startswith(str(tmp_path / "old"))
    for reference, refusal in (
        (frozen["output_path"].replace("/experiments/arm/", "/experiments/other_arm/"), "outside the experiment"),
        (frozen["output_path"].replace(frozen["artifact_id"], "strategy_research_absent"), "not on disk"),
    ):
        edited = json.loads(json.dumps(records))
        frozen_record(edited)["frozen"]["output_path"] = reference
        ledger.rewrite(edited)
        with pytest.raises(FileNotFoundError, match=refusal):
            pipeline.run_forward()
    ledger.rewrite(records)

    record = pipeline.run_forward()

    assert record["status"] == "ok" and record["artifact_id"] == frozen["artifact_id"]
    assert Path(record["result_ref"]).is_relative_to(pipeline.config.experiment_dir)


def test_the_forward_replay_writes_nothing_a_session_can_read(tmp_path: Path):
    """A session reads the experiment only through its Step tree, where every
    research Validation also leaves its daily graded series. The forward and
    Held-out replay writes its record to the host's results root and the
    ledger, never a Step node and never a daily series."""

    pipeline, _snapshots, _evaluator, _developer, _ledger = _freezing(tmp_path)
    pipeline.run_research_session()
    experiment = pipeline.config.experiment_dir
    steps = experiment / "steps"
    steps.mkdir()
    (steps / "tree.json").write_text('{"current_node_id": null, "nodes": []}', encoding="utf-8")

    def state() -> dict[Path, int]:
        return {path: path.stat().st_mtime_ns for path in experiment.rglob("*")}

    before = state()
    record = pipeline.run_forward()
    after = state()

    written = {path for path, stamp in after.items() if before.get(path) != stamp}
    assert Path(record["result_ref"]).parent in written
    assert all(
        path.is_relative_to(experiment / "artifacts" / "results")
        or path.is_relative_to(experiment / "ledgers")
        or path.is_relative_to(experiment / ".host")
        for path in written
    ), sorted(written)
    assert not any(path.is_relative_to(steps) for path in written)
    assert not any("active_daily" in path.name for path in after)


# A kill at the published memory cap is the strategy's own failure exactly as
# an exception its code raised is: a verdict, not a rerun of the attempt.
@pytest.mark.parametrize(
    "cause",
    [StrategyRaised("boom"), StrategyMemoryExceeded("killed at the 32g memory cap")],
    ids=["raised", "memory_cap"],
)
def test_a_strategy_error_names_the_slice_it_raised_in_and_discards(
    tmp_path: Path, cause: StrategyRaised
):
    pipeline, _snapshots, evaluator, _developer, ledger = _freezing(tmp_path)
    pipeline.run_research_session()
    failure = BacktestError(
        f"generate_orders failed at 2025-07-02T08:30:00+08:00: {cause}",
        inference_at=datetime(2025, 7, 2, 8, 30, tzinfo=CN_TZ),
    )
    failure.__cause__ = cause
    evaluator.raise_with = failure

    record = pipeline.run_forward()

    assert (record["status"], record["strategy_error"]) == ("strategy_error", "heldout")
    assert record["slices"] is None
    # The book raised in its sealed replay too: no reading there, a failed
    # sealed gate, and the forward replay still ran and raised.
    assert record["sealed"]["result_ref"] is None
    assert record["sealed"]["slice"]["strategy_error"] is True
    assert record["verdict"] == {
        "status": "discarded",
        "reasons": ["sealed_strategy_error", "heldout_strategy_error"],
        "thresholds": {},
    }
    assert experiment_verdict(ledger.read())["status"] == "discarded"
    assert paper_candidate(ledger.read()) is None

    # A book that passed its sealed gate and raised only in Held-out is
    # discarded too, and no Paper book is opened for it.
    pipeline, _snapshots, evaluator, _developer, ledger = _freezing(tmp_path / "late")
    pipeline.run_research_session()
    evaluator.raise_with, evaluator.raise_in = failure, "heldout"
    record = pipeline.run_forward()
    assert (record["status"], record["strategy_error"]) == ("strategy_error", "heldout")
    assert record["sealed"]["slice"]["reasons"] == []
    assert record["verdict"]["status"] == "discarded"
    assert record["verdict"]["reasons"] == ["heldout_strategy_error"]
    assert experiment_verdict(ledger.read())["status"] == "discarded"
    assert paper_candidate(ledger.read()) is None


def test_an_environment_failure_fails_the_attempt_and_leaves_no_verdict(tmp_path: Path):
    """A host failure of the sealed replay is a forward-stage failure: the
    research session stays finished and is never run again, and the retry
    reads the sealed years afresh."""

    pipeline, _snapshots, evaluator, developer, ledger = _freezing(tmp_path)
    research = pipeline.run_research_session()
    evaluator.requests.clear()
    evaluator.raise_with = TimeoutError("strategy inference exceeded 360s")

    with pytest.raises(TimeoutError):
        pipeline.run_forward()

    # The first replay the stage asked for was the sealed year's.
    [sealed_request] = evaluator.requests
    assert (sealed_request.start, sealed_request.end) == ("20230701", "20240630")
    [failed] = [row for row in ledger.read() if row["record_type"] == "attempt_failed"]
    assert (failed["phase"], failed["session_key"]) == ("forward", "forward")
    assert forward_record(ledger.read()) is None
    assert experiment_verdict(ledger.read()) is None
    assert sorted(RunMarkers(pipeline.config.experiment_dir).root.glob("*.json")) == []
    # The research session is over and untouched: its record stands and no
    # session runs again.
    assert research_over(ledger.read())
    recorded = frozen_record(ledger.read())
    assert (recorded["run_id"], recorded["frozen"]) == (research["run_id"], research["frozen"])
    with pytest.raises(RuntimeError, match="research is over"):
        pipeline.run_research_session()
    assert len(developer.requests) == 1
    # The retry reads the sealed year again and records the verdict.
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
            # A day the panel does not cover would be graded against nothing.
            analysis["panel_daily"] = analysis["panel_daily"][:-1]
            sidecar.write_text(json.dumps(analysis), encoding="utf-8")
            return result

    pipeline.evaluator = Unmeasurable(evaluator.root / "unmeasurable")
    with pytest.raises(ValueError, match="zero-skill panel has no return"):
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
        RollingExperimentConfig(experiment_id="arm", experiments_root=tmp_path, acceptance=RULES),
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
    assert research_step_record(step)["graded"] is None
