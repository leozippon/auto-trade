"""The validation lived through in steps: ``test``, the frontier, the
per-interval ``validate`` allowance and what reaches the agent."""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import pytest

from autotrade.common.records import write_json_atomic
from autotrade.environment.executor import StrategyExecutionError, StrategyRaised
from autotrade.pipelines import rolling_record, session_tools
from autotrade.pipelines.calendar import anchor_on
from autotrade.pipelines.config import EvaluationResult, SnapshotBundle
from autotrade.pipelines.rolling_record import (
    SOURCE_TEST,
    ValidationSteps,
    candidate_changes,
    step_entry,
)
from autotrade.pipelines.session_tools import EnvironmentBlocked, ToolError
from tests.unit.test_batch_validate import _Session, _strategy

DATE = re.compile(r"(?<!\d)(20\d{6})(?!\d)")


class _WindowEvaluator:
    """A deterministic replay over the request's window.

    Every day's returns are a function of the candidate's bytes and the day
    alone, so the same candidate reads the same on any window that covers the
    day: what makes a record completed by one whole replay equal the record
    of successive tests. ``raise_for`` maps a marker in ``main.py`` to the
    error its replay raises.
    """

    def __init__(self, results_root: Path, *, raise_for: dict[str, Exception] | None = None) -> None:
        self.results_root = results_root
        self.raise_for = raise_for or {}
        self.requests: list[tuple[str, str, str | None]] = []
        self.sandbox = None

    def evaluate(self, request, max_days=None, start_day=None, end_day=None):
        source = (Path(request.revision.output_path) / "main.py").read_text(encoding="utf-8")
        self.requests.append((request.start, request.end, end_day))
        for marker, error in self.raise_for.items():
            if marker in source:
                raise error
        seed = int(hashlib.sha256(source.encode()).hexdigest()[:8], 16)
        end = min(request.end, end_day) if end_day else request.end
        days = [day.strftime("%Y%m%d") for day in pd.bdate_range(request.start, end)]
        rows = []
        for day in days:
            rng = np.random.default_rng([seed, int(day)])
            market = rng.normal(0.0003, 0.008)
            rows.append((day, market, rng.normal(0.0, 0.003), rng.normal(0.0, 0.003), market + rng.normal(0.0, 0.002), rng.normal(0.0, 0.004)))
        strategy = [market + 0.0015 + noise + 0.0 * panel for _d, market, _s, _v, panel, noise in rows]
        equity, curve, executions = 100_000.0, [], []
        for index, ((day, *_rest), value) in enumerate(zip(rows, strategy, strict=True)):
            equity *= 1.0 + value
            curve.append({"trade_date": day, "equity": equity, "cash": equity * 0.3, "initial_equity": 100_000.0})
            if index % 10 == 0:
                executions.append({"status": "filled", "action": "buy", "matched_at": f"{day[:4]}-{day[4:6]}-{day[6:]}T09:30:00+08:00", "price": 10.0, "quantity": 500, "commission": 5.0, "stamp_duty": 0.0})
        peak, drawdown = 100_000.0, 0.0
        for row in curve:
            peak = max(peak, row["equity"])
            drawdown = max(drawdown, (peak - row["equity"]) / peak)
        summary = {
            "total_return": equity / 100_000.0 - 1.0,
            "sharpe": 1.0,
            "max_drawdown": drawdown,
            "turnover": 0.5,
            "trade_count": len(executions),
            "replayed_trade_days": len(days),
            "sub_windows": [{"label": f"{days[0][:6]}-{days[-1][:6]}", "start": days[0], "end": days[-1], "return": summary_return} for summary_return in (equity / 100_000.0 - 1.0,)],
        }
        directory = self.results_root / f"valid_{uuid.uuid4().hex}"
        directory.mkdir(parents=True)
        write_json_atomic(directory / "result.json", {"equity_curve": curve, "executions": executions, "stats": summary, "start": days[0], "end": days[-1]})
        write_json_atomic(
            directory / "style_analysis.json",
            {
                "strategy_daily": [[day, float(value)] for (day, *_r), value in zip(rows, strategy, strict=True)],
                "panel_daily": [[day, float(panel)] for day, _m, _s, _v, panel, _n in rows],
                "benchmark_daily": [[day, float(market)] for day, market, *_r in rows],
                "size_factor_daily": [[day, float(size)] for day, _m, size, *_r in rows],
                "value_factor_daily": [[day, float(value)] for day, _m, _s, value, *_r in rows],
            },
        )
        return EvaluationResult(summary, str(directory / "result.json"))


class _Harness:
    """A session with a real tree, revisions and tools over the deterministic replay."""

    def __init__(self, root: Path, *, raise_for: dict[str, Exception] | None = None, **rules: object) -> None:
        self.session = _Session(root, acceptance_rules={"max_drawdown": 0.25, **rules})
        backtest = self.session.backtest
        self.evaluator = _WindowEvaluator(self.session.evaluator.results_root, raise_for=raise_for)
        backtest.evaluator = self.evaluator
        self.anchors: list[datetime] = []
        self.views: list[tuple[str, str]] = []

        def decision_view_at(anchor):
            self.anchors.append(anchor)
            return SnapshotBundle(f"view_{anchor:%Y%m%d}", f"decision_{anchor:%Y%m%d}", "")

        backtest.request = replace(backtest.request, decision_view_at=decision_view_at)
        # Through the module: pytest would otherwise collect the class by its name.
        self.test = session_tools.TestTool(backtest, advance_view=lambda bundle, frontier: self.views.append((bundle.snapshot_id, frontier)))

    @property
    def backtest(self):
        return self.session.backtest

    def node(self, name: str, marker: str, *, span: str = "full", control: bool = False) -> str:
        self.session.candidate(name, _strategy(marker))
        result = self.session.call(name, span=span, controls=(name,) if control else ())
        assert result.ok, result.error
        return str(result.value["candidates"][0]["node_id"])

    def steps(self) -> ValidationSteps:
        return self.backtest.steps_plan

    def records(self):
        return rolling_record.test_step_records(self.backtest.ledger.read())


def test_a_test_advances_the_frontier_one_step_and_records_it_once() -> None:
    with TemporaryDirectory() as tmp:
        harness = _Harness(Path(tmp))
        steps = harness.steps()
        assert (steps.count, harness.backtest.frontier) == (4, "20250630")
        node = harness.node("a", "1")
        first = harness.test.invoke({"candidate": node})
        assert first.ok and not first.finish
        value = first.value
        assert (value["step"], value["steps"], value["steps_left"], value["frontier"]) == (1, 4, 3, "20250930")
        assert harness.backtest.frontier == "20250930"
        # Durable before the view moved: the ledger holds the step, the view
        # mounted next is the one at the new frontier.
        [record] = harness.records()
        assert (record["step"], record["frontier"], record["source"]) == (1, "20250930", SOURCE_TEST)
        assert record["candidate"]["step_id"] == node and record["entry"]["start"] == "20250701"
        assert harness.anchors == [anchor_on("20250930")] and harness.views == [("view_20250930", "20250930")]
        # The replay ran from the validation start to the new frontier.
        assert harness.evaluator.requests[-1] == ("20250701", "20260630", "20250930")
        assert value["test_replays"] == 1
        second = harness.test.invoke({"candidate": node})
        assert second.value["frontier"] == "20251231" and [row["step"] for row in harness.records()] == [1, 2]
        assert harness.evaluator.requests[-1] == ("20250701", "20260630", "20251231")
        # The record to date lists both steps with their candidate; the
        # instrument reads them stitched.
        record_rows = second.value["record"]
        assert [(row["step"], row["candidate"]["node_id"]) for row in record_rows] == [(1, node), (2, node)]
        assert second.value["instrument"]["steps"] == 2 and "residual_information_ratio" in second.value["instrument"]
        assert [row["month"] for row in second.value["months"]] == ["202507", "202508", "202509", "202510", "202511", "202512"]
        frontiers = [row["frontier"] for row in harness.records()]
        assert frontiers == sorted(frontiers) and len(set(frontiers)) == len(frontiers)


def test_an_unregistrable_candidate_is_refused_without_advancing_or_counting() -> None:
    with TemporaryDirectory() as tmp:
        harness = _Harness(Path(tmp))
        control = harness.node("c", "7", control=True)
        partial = harness.node("p", "8", span="Y1")
        for node, reason in ((control, "freeze_nominee_is_control"), (partial, "freeze_needs_full_span_validation")):
            with pytest.raises(ToolError) as refused:
                harness.test.invoke({"candidate": node})
            assert refused.value.error_type == "test_registration_refused"
            assert reason in str(refused.value) and "frontier did not move" in str(refused.value)
        assert harness.records() == [] and harness.backtest.frontier == "20250630"
        assert harness.backtest.test_replays == 0 and harness.views == []
        # Nothing was replayed for a refused candidate.
        assert all(end_day is None for _s, _e, end_day in harness.evaluator.requests)


def test_validate_is_bounded_per_interval_and_freed_by_a_test() -> None:
    with TemporaryDirectory() as tmp:
        harness = _Harness(Path(tmp), validates_per_step=2)
        node = harness.node("a", "1")
        harness.node("b", "2")
        harness.session.candidate("c", _strategy("3"))
        with pytest.raises(ToolError) as refused:
            harness.session.call("c")
        assert refused.value.error_type == "validates_per_step_exhausted"
        assert "validates_per_step allows 2" in str(refused.value) and "0 remain" in str(refused.value)
        assert harness.backtest.validates_in_interval == 2
        harness.test.invoke({"candidate": node})
        assert harness.backtest.validates_in_interval == 0
        result = harness.session.call("c")
        assert result.ok and result.value["validates_in_step"] == 1 and result.value["validates_per_step"] == 2
        # The frontier has entered the validation year: a span naming it is
        # replayed up to the frontier, and says so.
        harness.session.candidate("d", _strategy("4"))
        cut = harness.session.call("d", span="Y5")
        assert cut.ok and cut.value["span"]["end"] == "20250930"
        assert harness.evaluator.requests[-1] == ("20250701", "20260630", "20250930")


def test_the_last_test_finishes_the_session_with_its_candidate() -> None:
    with TemporaryDirectory() as tmp:
        harness = _Harness(Path(tmp))
        a, b = harness.node("a", "1"), harness.node("b", "2")
        for step, node in enumerate((a, a, b, b), start=1):
            result = harness.test.invoke({"candidate": node})
            assert result.finish is (step == 4), step
        assert result.value["outcome"] == "completed" and result.value["candidate"]["node_id"] == b
        assert "frozen strategy" in result.value["pipeline_outcome"]
        records = harness.records()
        assert [row["candidate"]["step_id"] for row in records] == [a, a, b, b]
        assert candidate_changes(harness.backtest.ledger.read()) == 1
        # The last step restarts no container: the session ends.
        assert len(harness.views) == 3
        with pytest.raises(ToolError, match="validation period is complete"):
            harness.test.invoke({"candidate": b})


def test_nothing_returned_to_the_agent_dates_past_the_frontier() -> None:
    with TemporaryDirectory() as tmp:
        harness = _Harness(Path(tmp), test_step_months=1)
        node = harness.node("a", "1")
        result = harness.test.invoke({"candidate": node})
        frontier = result.value["frontier"]
        assert frontier == "20250731"
        dates = set(DATE.findall(json.dumps(result.value)))
        assert dates and max(dates) <= frontier, sorted(dates)[-3:]
        # The view mounted next is the frontier's, and the replay stopped there.
        assert harness.views == [(f"view_{frontier}", frontier)]
        assert all(day <= frontier for day in (row["end"] for row in result.value["months"]))


def test_a_test_whose_replay_fails_leaves_the_frontier_where_it_was() -> None:
    with TemporaryDirectory() as tmp:
        harness = _Harness(
            Path(tmp),
            raise_for={
                "_ = 5": StrategyRaised("KeyError: 'close'"),
                "_ = 6": StrategyExecutionError("Docker generate_orders failed: timeout"),
            },
        )
        good = harness.node("a", "1")
        harness.session.candidate("s", _strategy("5"))
        harness.session.candidate("e", _strategy("6"))
        # The replays raise only on the validation data: the Validations ran.
        harness.evaluator.raise_for = {}
        strategy, environment = harness.node("s", "5"), harness.node("e", "6")
        harness.evaluator.raise_for = {
            "_ = 5": StrategyRaised("KeyError: 'close'"),
            "_ = 6": StrategyExecutionError("Docker generate_orders failed: timeout"),
        }
        with pytest.raises(ToolError) as raised:
            harness.test.invoke({"candidate": strategy})
        assert raised.value.error_type == "test_replay_failed" and raised.value.details["cause"] == "strategy"
        assert "frontier did not move" in str(raised.value)
        with pytest.raises(ToolError) as failed:
            harness.test.invoke({"candidate": environment})
        assert failed.value.details["cause"] == "environment" and "resent unchanged" in str(failed.value)
        assert harness.records() == [] and harness.backtest.frontier == "20250630"
        assert harness.backtest.test_replays == 2 and harness.views == []
        # A second host failure in a row ends the attempt, not the research.
        with pytest.raises(EnvironmentBlocked, match="2 replays failed on the environment"):
            harness.test.invoke({"candidate": environment})
        # A good candidate still advances.
        assert harness.test.invoke({"candidate": good}).value["step"] == 1


def test_a_seed_that_lacks_the_frontiers_view_ends_the_attempt_by_name() -> None:
    with TemporaryDirectory() as tmp:
        harness = _Harness(Path(tmp))
        node = harness.node("a", "1")

        def missing(anchor):
            raise FileNotFoundError(f"the PIT view seed carries no decision view at {anchor.isoformat()}")

        harness.backtest.request = replace(harness.backtest.request, decision_view_at=missing)
        with pytest.raises(EnvironmentBlocked, match="no decision view at 2025-09-30T23:59:59"):
            harness.test.invoke({"candidate": node})
        assert harness.records() == [] and harness.evaluator.requests[-1][2] is None


def test_a_completed_record_equals_the_record_of_successive_tests() -> None:
    """With a deterministic replay, the entries one whole-period replay yields
    for the remaining steps are the entries successive tests would have
    recorded: what the host's completion of an early stop promises."""

    with TemporaryDirectory() as tmp:
        harness = _Harness(Path(tmp))
        node = harness.node("a", "1")
        steps = harness.steps()
        successive = []
        for _step in range(steps.count):
            successive.append(harness.test.invoke({"candidate": node}).value)
        entries = [dict(row["entry"]) for row in harness.records()]
        from autotrade.pipelines.experiment import replay_and_analysis

        revision = harness.backtest.artifact_store.revision(harness.backtest.steps[0].revision_id)
        whole = harness.evaluator.evaluate(harness.backtest.validation_request(revision, harness.backtest.request.validation))
        replay, analysis = replay_and_analysis(whole.result_ref)
        for step, entry in enumerate(entries, start=1):
            first, last = steps.window(step)
            assert step_entry(replay, analysis, start=first, end=last) == entry, step
