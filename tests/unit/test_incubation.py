"""Incubation: the entry and forward screens, the ``incubation`` record, and
the operator's command (``scripts/experiments/incubate.py``)."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from autotrade.environment.broker import BrokerProfile
from autotrade.environment.broker_core import board_of
from autotrade.environment.executor import StrategyRaised
from autotrade.environment.replay.engine import BacktestError
from autotrade.environment.strategy import CN_TZ
from autotrade.paper.book import INCUBATING_BOOK_CAP
from autotrade.pipelines import incubation
from autotrade.pipelines.experiment import _keep_frozen_artifact_ids, incubation_entry
from autotrade.pipelines.hitl_state import proc_start_ticks
from autotrade.pipelines.ledger import (
    INCUBATION_FIELDS,
    ExperimentLedger,
    FrozenArtifactMutated,
    RunMarkers,
    experiment_verdict,
    forward_record,
    frozen_record,
    incubation_candidate,
    paper_candidate,
    require_incubable,
)
from autotrade.pipelines.verdict import CONDITIONS, NOT_A_SESSION_STEP, UNMEASURABLE
from tests.unit.paper_book_fixture import write_book_record
from tests.unit.test_null_control import _board_draws
from tests.unit.test_rolling_pipeline import CONFIG_PROFILE, _pipeline
from tests.unit.webui_research_arm import build_arm

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_every_freeze_gate_reason_is_classified_exactly_once():
    # Every condition the graduation verdict does not judge is the gate's,
    # whatever stage a new one is filed under.
    gate = {condition.reason for condition in CONDITIONS if condition.stage not in ("replay", "forward", "heldout")}
    assert not incubation.DEFERRABLE & incubation.BLOCKING
    assert incubation.DEFERRABLE | incubation.BLOCKING == gate | {NOT_A_SESSION_STEP, UNMEASURABLE}
    assert incubation.GATE_REASONS == gate | {NOT_A_SESSION_STEP, UNMEASURABLE}


def _gate(*reasons: str, mean: float | None = None, floor: float = 0.75) -> dict[str, object]:
    return {
        "reasons": list(reasons),
        "thresholds": {"min_information_ratio": floor},
        "seed_replicates": {"mean_information_ratio": mean},
    }


@pytest.mark.parametrize("reason", sorted(incubation.BLOCKING))
def test_each_blocking_reason_is_refused(reason: str):
    screen = incubation.entry_screen(_gate(incubation.DEFLATED_SHARPE, reason))
    assert screen == {"passed": False, "deferred": [incubation.DEFLATED_SHARPE], "blocking": [reason]}


def test_a_deferrable_only_node_and_a_node_without_reasons_pass():
    assert incubation.entry_screen(_gate(incubation.DEFLATED_SHARPE))["passed"] is True
    assert incubation.entry_screen({"passed": True, "reasons": []}) == {
        "passed": True,
        "deferred": [],
        "blocking": [],
    }


def test_the_seed_mean_defers_only_at_or_above_min_active_ir():
    seed_mean = incubation.SEED_MEAN
    assert incubation.entry_screen(_gate(seed_mean, mean=0.75))["passed"] is True
    assert incubation.entry_screen(_gate(seed_mean, mean=0.7499))["blocking"] == [seed_mean]
    assert incubation.entry_screen(_gate(seed_mean, mean=None))["blocking"] == [seed_mean]


def _reading(*reasons: str, neutralized: float = 0.05, plain: float = 0.04, seed_mean: float | None = None):
    forward: dict[str, object] = {"neutralized_excess": neutralized, "plain_excess": plain}
    if seed_mean is not None:
        forward["seed_mean"] = {"plain_excess": seed_mean}
    return {"verdict": {"reasons": list(reasons)}, "slices": {"forward": forward}}


@pytest.mark.parametrize("reason", incubation.FORWARD_BLOCKING)
def test_each_forward_screen_reason_blocks(reason: str):
    reading = _reading(reason, "forward_lower_bound_not_positive")
    if reason.endswith("strategy_error"):
        reading = {"verdict": {"reasons": [reason]}, "slices": None}
    assert incubation.forward_screen(reading) == {"passed": False, "reasons": [f"incubation_{reason}"]}


def test_the_forward_screen_needs_positive_neutralized_and_plain_excess():
    # A graduation reason outside the screen does not block.
    assert incubation.forward_screen(_reading("forward_lower_bound_not_positive"))["passed"] is True
    assert incubation.forward_screen(_reading(neutralized=0.0))["reasons"] == [
        incubation.NEUTRALIZED_EXCESS_NOT_POSITIVE
    ]
    assert incubation.forward_screen(_reading(plain=-0.01))["reasons"] == [
        incubation.PLAIN_EXCESS_NOT_POSITIVE
    ]
    # With seed replicates the seed mean's plain excess decides, not the book's.
    assert incubation.forward_screen(_reading(plain=-0.01, seed_mean=0.02))["passed"] is True
    assert incubation.forward_screen(_reading(plain=0.02, seed_mean=-0.01))["reasons"] == [
        incubation.PLAIN_EXCESS_NOT_POSITIVE
    ]


def _session(kind: str, verdict: str | None = None) -> list[dict[str, object]]:
    keys = {"experiment_id": "arm", "epoch_id": "research", "fold_id": "research", "run_id": "run_1"}
    if kind == "ended":
        return [{**keys, "record_type": "research_session", "arm_end": {"status": "no_deliverable", "reason": "x"}}]
    records = [{**keys, "record_type": "research_session", "frozen": {"artifact_id": "a", "output_path": "p"}}]
    if verdict is not None:
        records.append({**keys, "epoch_id": "forward", "fold_id": "forward", "record_type": "forward",
                        "verdict": {"status": verdict, "reasons": []}})
    return records


def test_only_an_arm_whose_research_ended_without_a_graduation_is_incubable():
    require_incubable(_session("ended"))
    require_incubable(_session("frozen", "discarded"))
    voided = [*_session("frozen", "graduated"), {"record_type": "verdict_void", "voided_by": "o", "reason": "r",
                                                 "evidence_ref": "e"}]
    for records in ([], _session("frozen"), _session("frozen", "graduated"), voided):
        with pytest.raises(ValueError, match="can be incubated"):
            require_incubable(records)


def _ended_arm(tmp_path: Path, broker_profile: BrokerProfile = CONFIG_PROFILE):
    """An arm whose nominee (Step 1) the gate refused, so research ended
    without a deliverable; Step 0 is the better node."""

    pipeline, _snapshots, _evaluator, _developer, ledger = _pipeline(
        tmp_path, {1: ([0.0012, -0.0004], "freeze", {"nominee": 1})}, broker_profile=broker_profile
    )
    pipeline.run_research_session()
    assert experiment_verdict(ledger.read())["status"] == "no_deliverable"
    return pipeline, ledger


def _tree(root: Path) -> dict[str, tuple[int, int]]:
    return {str(path): (path.stat().st_mtime_ns, path.stat().st_size) for path in root.rglob("*")}


def test_a_passing_incubation_appends_one_record_and_leaves_the_verdict(tmp_path: Path):
    pipeline, ledger = _ended_arm(tmp_path)
    before = ledger.read()
    verdict, candidate = experiment_verdict(before), paper_candidate(before)
    tree = _tree(pipeline.config.experiment_dir)
    entry = incubation_entry(before, "research_step_0", config=pipeline.config)
    # Reading the entry screen writes nothing (what --dry-run prints).
    assert _tree(pipeline.config.experiment_dir) == tree
    assert entry["screen"]["passed"] is True

    record = pipeline.incubate("research_step_0", incubated_by="operator", reason="the better node")
    after = ledger.read()
    assert after[: len(before)] == before and len(after) == len(before) + 1
    assert {key: after[-1][key] for key in record} == json.loads(json.dumps(record))
    assert (record["record_type"], record["epoch_id"], record["fold_id"]) == ("incubation", "forward", "incubation")
    assert record["run_id"].startswith("incubation_")
    assert all(record[key] for key in INCUBATION_FIELDS)
    assert record["entry"]["deferred"] == entry["screen"]["deferred"]
    assert record["entry"]["reasons"] == entry["entry"]["reasons"]
    assert (record["source_step_id"], record["frozen"]["source_step_id"]) == ("research_step_0",) * 2
    reading = record["forward_reading"]
    assert reading["status"] == "ok" and reading["artifact_id"] == record["frozen"]["artifact_id"]
    assert record["screen"] == incubation.forward_screen(reading)
    assert experiment_verdict(after) == verdict and paper_candidate(after) == candidate

    assert record["screen"] == {"passed": True, "reasons": []}
    opened = incubation_candidate(after)
    assert opened == {
        "artifact_id": record["frozen"]["artifact_id"],
        "output_path": record["frozen"]["output_path"],
        "models_path": record["frozen"]["models_path"],
        "seed_replicates": [],
        "permitted_boards": record["permitted_boards"]["boards"],
        "forward": reading,
    }
    assert opened["forward"]["result_ref"] and opened["forward"]["replay"]["heldout_start"]

    # Pruning keeps the artifact the incubation froze.
    pipeline.artifacts.prune_superseded_frozen(keep_frozen_ids=_keep_frozen_artifact_ids(after))
    assert Path(record["frozen"]["output_path"], "main.py").is_file()

    with pytest.raises(ValueError, match="already incubated"):
        pipeline.incubate("research_step_0", incubated_by="operator", reason="again")
    with pytest.raises(ValueError, match="already incubated"):
        ledger.append({**record, "run_id": "incubation_again"})


def test_an_unstamped_arm_replays_on_the_boards_its_capital_qualifies_for(tmp_path: Path):
    """An arm stamped before boards existed buys on every board in research;
    its incubation replays on the boards its 100,000 CNY may buy on, Broker
    and zero-skill panel alike, records them as derived, and reads the entry
    screen exactly as the session recorded it."""

    pipeline, ledger = _ended_arm(tmp_path, BrokerProfile(initial_cash=100_000))
    assert pipeline.config.broker_profile.permitted_boards is None
    before = ledger.read()
    [session] = [row for row in before if row["record_type"] == "research_session"]
    nominee = incubation_entry(before, session["nominated_step_id"], config=pipeline.config)["entry"]
    assert {key: value for key, value in nominee.items() if key != "deferred"} == session["freeze_gate"]
    researched = len(pipeline.evaluator.requests)
    record = pipeline.incubate("research_step_0", incubated_by="operator", reason="the better node")
    assert record["entry"] == incubation_entry(before, "research_step_0", config=pipeline.config)["entry"]
    assert record["permitted_boards"] == {"boards": ["main", "gem"], "origin": "derived"}
    [request] = pipeline.evaluator.requests[researched:]
    assert request.broker_profile == BrokerProfile(initial_cash=100_000, permitted_boards=("main", "gem"))
    # The panel draws with the request's profile: never a STAR or Beijing name.
    drawn = {symbol for draw in _board_draws(request.broker_profile.permitted_boards, 200, 1) for symbol, *_ in draw}
    assert drawn and all(board_of(symbol) in ("main", "gem") for symbol in drawn)
    assert incubation_candidate(ledger.read())["permitted_boards"] == ["main", "gem"]


def test_a_stamped_arm_replays_on_its_own_boards(tmp_path: Path):
    stamped = BrokerProfile(initial_cash=100_000, permitted_boards=("main",))
    pipeline, _ledger = _ended_arm(tmp_path, stamped)
    researched = len(pipeline.evaluator.requests)
    record = pipeline.incubate("research_step_0", incubated_by="operator", reason="the better node")
    assert record["permitted_boards"] == {"boards": ["main"], "origin": "stamped"}
    assert [request.broker_profile for request in pipeline.evaluator.requests[researched:]] == [stamped]


def test_the_entry_reading_is_the_gate_the_session_recorded(tmp_path: Path):
    pipeline, ledger = _ended_arm(tmp_path)
    records = ledger.read()
    [session] = [record for record in records if record["record_type"] == "research_session"]
    entry = incubation_entry(records, session["nominated_step_id"], config=pipeline.config)["entry"]
    assert {key: value for key, value in entry.items() if key != "deferred"} == session["freeze_gate"]


def test_a_strategy_error_is_a_recorded_reading_the_screen_blocks(tmp_path: Path):
    pipeline, ledger = _ended_arm(tmp_path)
    failure = BacktestError(
        "generate_orders failed at 2025-07-02T08:30:00+08:00: boom",
        inference_at=datetime(2025, 7, 2, 8, 30, tzinfo=CN_TZ),
    )
    failure.__cause__ = StrategyRaised("boom")
    pipeline.evaluator.raise_with = failure
    record = pipeline.incubate("research_step_0", incubated_by="operator", reason="the better node")
    assert record["forward_reading"]["status"] == "strategy_error"
    assert record["screen"] == {"passed": False, "reasons": ["incubation_heldout_strategy_error"]}
    assert ledger.read("incubation") and incubation_candidate(ledger.read()) is None


def test_a_replay_that_measured_nothing_records_nothing(tmp_path: Path):
    pipeline, ledger = _ended_arm(tmp_path)
    before = ledger.read()
    pipeline.evaluator.raise_with = TimeoutError("strategy inference exceeded 360s")
    with pytest.raises(TimeoutError):
        pipeline.incubate("research_step_0", incubated_by="operator", reason="the better node")
    assert ledger.read() == before
    assert sorted(RunMarkers(pipeline.config.experiment_dir).root.glob("*.json")) == []


def test_frozen_trees_changed_during_the_replay_leave_a_later_incubation_unchanged(tmp_path: Path):
    pipeline, ledger = _ended_arm(tmp_path)
    before = ledger.read()
    evaluator = pipeline.evaluator

    class Mutating(type(evaluator)):
        def evaluate(self, request):
            main_py = Path(request.revision.output_path) / "main.py"
            main_py.chmod(0o644)
            main_py.write_text("# edited\n", encoding="utf-8")
            return super().evaluate(request)

    pipeline.evaluator = Mutating(evaluator.root / "mutating")
    with pytest.raises(FrozenArtifactMutated):
        pipeline.incubate("research_step_0", incubated_by="operator", reason="the better node")
    assert ledger.read() == before
    # The retry freezes the Step's unchanged revision afresh.
    pipeline.evaluator = evaluator
    record = pipeline.incubate("research_step_0", incubated_by="operator", reason="the better node")
    revision = pipeline.artifacts.revision(record["revision_id"])
    assert Path(record["frozen"]["output_path"], "main.py").read_bytes() == Path(
        revision.output_path, "main.py"
    ).read_bytes()


def test_a_discarded_arm_keeps_its_verdict_byte_for_byte(tmp_path: Path):
    pipeline, _snapshots, evaluator, _developer, ledger = _pipeline(
        tmp_path, {1: ([0.0012, 0.0002], "freeze", {"nominee": 0})}
    )
    pipeline.run_research_session()
    failure = BacktestError("generate_orders failed: boom", inference_at=datetime(2025, 7, 2, 8, 30, tzinfo=CN_TZ))
    failure.__cause__ = StrategyRaised("boom")
    evaluator.raise_with = failure
    pipeline.run_forward()
    before = ledger.read()
    verdict, candidate = json.dumps(experiment_verdict(before)), json.dumps(paper_candidate(before))
    assert experiment_verdict(before)["status"] == "discarded"
    evaluator.raise_with = None
    pipeline.incubate("research_step_0", incubated_by="operator", reason="the nominee, read once more")
    after = ledger.read()
    assert (json.dumps(experiment_verdict(after)), json.dumps(paper_candidate(after))) == (verdict, candidate)


def test_a_recorded_forward_screen_block_opens_no_book(tmp_path: Path):
    pipeline, ledger = _ended_arm(tmp_path)
    pipeline.incubate("research_step_0", incubated_by="operator", reason="the better node")
    [record] = ledger.read("incubation")
    blocked = {**record, "screen": {"passed": False, "reasons": [incubation.PLAIN_EXCESS_NOT_POSITIVE]}}
    assert incubation_candidate([*ledger.read()[:-1], blocked]) is None


def test_a_blocked_node_and_a_foreign_step_are_refused_without_a_record(tmp_path: Path):
    pipeline, ledger = _ended_arm(tmp_path)
    before = ledger.read()
    with pytest.raises(ValueError, match="entry screen blocks research_step_1"):
        pipeline.incubate("research_step_1", incubated_by="operator", reason="the nominee")
    for step, replicates in (("research_step_9", ()), ("research_step_0", ("other_step_0",))):
        with pytest.raises(ValueError, match=NOT_A_SESSION_STEP):
            incubation_entry(before, step, replicates, config=pipeline.config)
    assert ledger.read() == before


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [("control", True, "freeze_nominee_is_control"), ("span", "y1", "freeze_needs_full_span_validation")],
)
def test_controls_and_sub_span_nodes_are_blocked(tmp_path: Path, field: str, value: object, reason: str):
    pipeline, ledger = _ended_arm(tmp_path)
    records = ledger.read()
    session = next(record for record in records if record["record_type"] == "research_session")
    session["steps"] = [{**row, field: value} if row["step_id"] == "research_step_0" else row
                        for row in session["steps"]]
    screen = incubation_entry(records, "research_step_0", config=pipeline.config)["screen"]
    assert screen["passed"] is False and reason in screen["blocking"]


def test_the_command_refuses_an_arm_with_a_live_worker(tmp_path: Path):
    pipeline, ledger = _ended_arm(tmp_path)
    hitl = pipeline.config.experiment_dir / "hitl"
    hitl.mkdir(exist_ok=True)
    (hitl / "status.json").write_text(
        f'{{"schema_version": 1, "state": "running", "pid": {os.getpid()}, '
        f'"pid_start_ticks": {proc_start_ticks(os.getpid())}}}',
        encoding="utf-8",
    )
    before = ledger.read()
    completed = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "experiments" / "incubate.py"),
         "--experiment", "arm", "--step", "research_step_0", "--by", "operator", "--reason", "r",
         "--experiments-root", str(pipeline.config.experiments_root), "--dry-run"],
        capture_output=True, text=True, check=False,
    )
    assert completed.returncode == 2
    assert "live worker" in completed.stderr
    assert ledger.read() == before


def _incubate_script():
    spec = importlib.util.spec_from_file_location("incubate", REPO_ROOT / "scripts/experiments/incubate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_command_refuses_at_the_cap_before_anything_changes(tmp_path: Path, monkeypatch, capsys):
    pipeline, ledger = _ended_arm(tmp_path)
    (pipeline.config.experiment_dir / "hitl").mkdir(exist_ok=True)
    (pipeline.config.experiment_dir / "hitl" / "params.json").write_text('{"experiment_id": "arm"}', encoding="utf-8")
    paper = tmp_path / "paper"
    for index in range(INCUBATING_BOOK_CAP):
        write_book_record(paper / f"inc{index}", experiment_id=f"inc{index}", candidate_source="incubating")
    script = _incubate_script()
    # The arm's resolved options are the pipeline's own; nothing past the cap may read more of them.
    monkeypatch.setattr(script, "resolve_worker_options", lambda *_args, **_kwargs: SimpleNamespace(rolling=pipeline.config))
    monkeypatch.setattr(sys, "argv", [
        "incubate.py", "--experiment", "arm", "--step", "research_step_0", "--by", "operator", "--reason", "r",
        "--experiments-root", str(pipeline.config.experiments_root), "--paper-state-root", str(paper),
    ])
    before, tree = ledger.read(), _tree(tmp_path)
    assert script.main() == 2
    assert f"the cap is {INCUBATING_BOOK_CAP}" in capsys.readouterr().err
    assert ledger.read() == before and _tree(tmp_path) == tree


def _incubated_arm(experiments: Path, passed: bool = True) -> Path:
    """A discarded arm whose incubation record the command already appended:
    the forward replay of its frozen nominee, read as the incubation's."""

    arm = build_arm(experiments, "inc", "discarded")
    ledger = ExperimentLedger(arm / "ledgers/experiment_ledger.jsonl")
    records = ledger.read()
    frozen, forward = frozen_record(records)["frozen"], forward_record(records)
    ledger.append({
        "record_type": "incubation", "experiment_id": "inc", "epoch_id": "forward", "fold_id": "incubation",
        "run_id": "incubation_0", "incubated_by": "operator", "reason": "r",
        "source_step_id": frozen["source_step_id"], "revision_id": frozen["revision_id"],
        "entry": {"reasons": [], "deferred": []},
        "frozen": {key: frozen[key] for key in ("artifact_id", "output_path", "models_path", "source_step_id", "revision_id")},
        "permitted_boards": {"boards": ["main", "gem"], "origin": "derived"},
        "forward_reading": {key: forward[key] for key in ("artifact_id", "replay", "result_ref", "slices", "verdict")},
        "screen": {"passed": passed, "reasons": [] if passed else [incubation.PLAIN_EXCESS_NOT_POSITIVE]},
    })
    return arm


def _rerun(tmp_path: Path, step: str) -> dict[str, object]:
    completed = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "experiments" / "incubate.py"),
         "--experiment", "inc", "--step", step, "--by", "operator", "--reason", "r",
         "--experiments-root", str(tmp_path / "experiments"), "--paper-state-root", str(tmp_path / "paper")],
        capture_output=True, text=True, check=False,
    )
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout)


def test_a_rerun_opens_only_the_missing_book_and_appends_nothing(tmp_path: Path):
    arm = _incubated_arm(tmp_path / "experiments")
    ledger_bytes = (arm / "ledgers/experiment_ledger.jsonl").read_bytes()
    [record] = ExperimentLedger(arm / "ledgers/experiment_ledger.jsonl").read("incubation")

    opened = _rerun(tmp_path, record["source_step_id"])
    assert (opened["book"], opened["book_id"], opened["screen"]["passed"]) == ("opened", "inc", True)
    forward = record["forward_reading"]["slices"]["forward"]
    assert opened["forward"]["neutralized_excess_per_year"] == forward["neutralized_excess"]
    assert opened["forward"]["panel_return"] == forward["raw_readings"]["panel_return"]
    assert opened["forward"]["account_return"] == forward["raw_readings"]["strategy_return"]
    book = tmp_path / "paper" / "inc"
    pinned = json.loads((book / "book.json").read_text(encoding="utf-8"))
    assert (pinned["candidate_source"], pinned["artifact_id"]) == ("incubating", record["frozen"]["artifact_id"])
    assert pinned["profile"]["permitted_boards"] == ["main", "gem"]
    files = {path: path.read_bytes() for path in book.rglob("*") if path.is_file()}

    assert _rerun(tmp_path, record["source_step_id"])["book"] == "existed"
    assert {path: path.read_bytes() for path in book.rglob("*") if path.is_file()} == files
    assert (arm / "ledgers/experiment_ledger.jsonl").read_bytes() == ledger_bytes


def test_a_blocked_forward_screen_opens_no_book(tmp_path: Path):
    arm = _incubated_arm(tmp_path / "experiments", passed=False)
    [record] = ExperimentLedger(arm / "ledgers/experiment_ledger.jsonl").read("incubation")
    blocked = _rerun(tmp_path, record["source_step_id"])
    assert (blocked["book"], blocked["screen"]["passed"]) == ("blocked", False)
    assert not (tmp_path / "paper" / "inc").exists()
