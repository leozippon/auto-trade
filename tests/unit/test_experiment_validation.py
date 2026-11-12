"""The validation lived through in steps: ``test``, the test horizon, the
per-interval ``validate`` allowance, the research-years bound on ``validate``
and what reaches the agent."""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import pytest

from autotrade.common.records import write_json_atomic
from autotrade.environment.executor import StrategyExecutionError, StrategyRaised
from autotrade.environment.sandbox import SandboxConfig, SandboxSpec
from autotrade.pipelines import session_tools, validation_record
from autotrade.pipelines.config import (
    DEFAULT_BATCH_VALIDATE_MAX_CONCURRENCY,
    EvaluationResult,
)
from autotrade.pipelines.experiment import replay_and_analysis
from autotrade.pipelines.session_tools import EnvironmentBlocked, ToolError
from autotrade.pipelines.validation_record import (
    SOURCE_TEST,
    ValidationSteps,
    candidate_changes,
    step_entry,
)
from tests.unit.test_batch_validate import _Session, _strategy

DATE = re.compile(r"(?<!\d)(20\d{6})(?!\d)")


def _seeded(seed: int) -> str:
    return f"SEED_BASE = {seed}\n\n\n{_strategy('1')}"


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
        # Each replay's source, card and (inference, fit) clocks as it ran.
        self.legs: list[tuple[str, tuple[int, ...] | None, tuple[float, float] | None]] = []
        self.sandbox = None

    def evaluate(self, request, max_days=None, start_day=None, end_day=None):
        source = (Path(request.revision.output_path) / "main.py").read_text(encoding="utf-8")
        self.requests.append((request.start, request.end, end_day))
        clocks = None if self.sandbox is None else (self.sandbox.timeout_seconds, self.sandbox.fit_timeout_seconds)
        self.legs.append((source, request.gpu_devices, clocks))
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
        strategy = [market + 0.0015 + noise + 0.0 * baselines for _d, market, _s, _v, baselines, noise in rows]
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
                "baseline_daily": [[day, float(baselines)] for day, _m, _s, _v, baselines, _n in rows],
                "benchmark_daily": [[day, float(market)] for day, market, *_r in rows],
                "size_factor_daily": [[day, float(size)] for day, _m, size, *_r in rows],
                "value_factor_daily": [[day, float(value)] for day, _m, _s, value, *_r in rows],
            },
        )
        return EvaluationResult(summary, str(directory / "result.json"))


class _Harness:
    """A session with a real tree, revisions and tools over the deterministic replay."""

    def __init__(
        self,
        root: Path,
        *,
        raise_for: dict[str, Exception] | None = None,
        max_concurrency: int = DEFAULT_BATCH_VALIDATE_MAX_CONCURRENCY,
        **rules: object,
    ) -> None:
        self.session = _Session(root, acceptance_rules={"max_drawdown": 0.25, **rules}, max_concurrency=max_concurrency)
        backtest = self.session.backtest
        self.evaluator = _WindowEvaluator(self.session.evaluator.results_root, raise_for=raise_for)
        backtest.evaluator = self.evaluator
        # Through the module: pytest would otherwise collect the class by its name.
        self.test = session_tools.TestTool(backtest, max_concurrency=max_concurrency)

    @property
    def backtest(self):
        return self.session.backtest

    def node(self, name: str, marker: str, *, span: str = "full", control: bool = False) -> str:
        return self.validated(name, _strategy(marker), span=span, control=control)

    def seeded(self, name: str, seed: int) -> str:
        """A full-span node of one strategy body on the training seed ``seed``:
        nodes of different seeds are each other's seed replicates."""

        return self.validated(name, _seeded(seed))

    def validated(self, name: str, source: str, *, span: str = "full", control: bool = False) -> str:
        self.session.candidate(name, source)
        result = self.session.call(name, span=span, controls=(name,) if control else ())
        assert result.ok, result.error
        return str(result.value["candidates"][0]["node_id"])

    def steps(self) -> ValidationSteps:
        return self.backtest.steps_plan

    def records(self):
        return validation_record.test_step_records(self.backtest.ledger.read())


def test_a_test_advances_the_test_horizon_one_step_and_records_it_once() -> None:
    with TemporaryDirectory() as tmp:
        harness = _Harness(Path(tmp))
        steps = harness.steps()
        assert (steps.count, harness.backtest.test_horizon) == (4, "20250630")
        node = harness.node("a", "1")
        first = harness.test.invoke({"candidate": node})
        assert first.ok and not first.finish
        value = first.value
        assert (value["step"], value["steps"], value["steps_left"], value["test_horizon"]) == (1, 4, 3, "20250930")
        assert harness.backtest.test_horizon == "20250930"
        # Durable before the result returned: the ledger holds the step.
        [record] = harness.records()
        assert (record["step"], record["test_horizon"], record["source"]) == (1, "20250930", SOURCE_TEST)
        assert record["candidate"]["step_id"] == node and record["entry"]["start"] == "20250701"
        # No seed replicate named: the record and every row keep the candidate's shape alone.
        assert "replicates" not in record
        assert all("seed_replicates" not in row for row in (*value["candidate_by_step"], *value["record"]))
        # The result is a projection: each block once, nothing the call or the
        # facts already state.
        assert list(value) == [
            "status", "step", "steps", "steps_left", "test_horizon", "candidate", "replay",
            "whole_replay", "candidate_by_step", "record", "significance",
        ]
        assert value["candidate"] == {"handle": "valid_001", "node_id": node}
        assert list(value["whole_replay"]) == ["active", "book"]
        assert "threshold" not in value["whole_replay"]["active"]
        assert value["candidate_by_step"] == [
            {"step": 1, "active_return": pytest.approx(value["record"][0]["active_return"], abs=1e-6),
             "return": pytest.approx(value["record"][0]["return"], abs=1e-6)}
        ]
        assert list(value["record"][0]) == [
            "step", "start", "end", "days", "candidate", "return", "active_return", "index_return",
            "max_drawdown", "turnover", "fees",
        ]
        assert list(value["significance"]) == [
            "residual_information_ratio", "threshold", "p_value", "residual_excess", "residual_tracking_error",
            "market_beta", "size_beta", "value_beta", "days", "years",
        ]
        # The replay ran from the validation start to the new test horizon.
        assert harness.evaluator.requests[-1] == ("20250701", "20260630", "20250930")
        assert harness.backtest.test_replays == 1
        second = harness.test.invoke({"candidate": node})
        assert second.value["test_horizon"] == "20251231" and [row["step"] for row in harness.records()] == [1, 2]
        assert harness.evaluator.requests[-1] == ("20250701", "20260630", "20251231")
        # The record to date lists both steps with their candidate; the
        # significance test reads them stitched.
        record_rows = second.value["record"]
        assert [(row["step"], row["candidate"], row["start"], row["end"]) for row in record_rows] == [
            (1, "valid_001", "20250701", "20250930"),
            (2, "valid_001", "20251001", "20251231"),
        ]
        assert second.value["significance"]["days"] == sum(row["days"] for row in record_rows)
        # This replay's view has one row per quarterly step tested.
        assert [row["step"] for row in second.value["candidate_by_step"]] == [1, 2]
        assert "months" not in second.value
        horizons = [row["test_horizon"] for row in harness.records()]
        assert horizons == sorted(horizons) and len(set(horizons)) == len(horizons)


def test_an_unregistrable_candidate_is_refused_without_advancing_or_counting() -> None:
    with TemporaryDirectory() as tmp:
        harness = _Harness(Path(tmp))
        control = harness.node("c", "7", control=True)
        partial = harness.node("p", "8", span="Y1")
        for node, reason in ((control, "registration_nominee_is_control"), (partial, "registration_needs_full_span_validation")):
            with pytest.raises(ToolError) as refused:
                harness.test.invoke({"candidate": node})
            assert refused.value.error_type == "test_registration_refused"
            assert reason in str(refused.value) and "test horizon did not move" in str(refused.value)
        assert harness.records() == [] and harness.backtest.test_horizon == "20250630"
        assert harness.backtest.test_replays == 0
        # Nothing was replayed for a refused candidate.
        assert all(end_day is None for _s, _e, end_day in harness.evaluator.requests)


def test_validate_is_bounded_per_interval_freed_by_a_test_and_kept_to_the_research_years() -> None:
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
        # The view does not move with the test horizon: a span reaching into the
        # validation year is refused by name after a test as before it, and
        # the refusal neither replays nor counts.
        harness.session.candidate("d", _strategy("4"))
        replays = len(harness.evaluator.requests)
        for span in ("Y5", "Y4..Y5"):
            with pytest.raises(ToolError) as refused:
                harness.session.call("d", span=span)
            assert refused.value.error_type == "schema_error"
            assert "past the end of the research years 20250630" in str(refused.value)
            assert "span revealed replays the steps tested so far" in str(refused.value)
        assert len(harness.evaluator.requests) == replays and harness.backtest.validates_in_interval == 1


def test_validate_revealed_replays_exactly_the_tested_steps_and_nothing_before_a_test() -> None:
    """Span revealed is the validation span as a test replays it, through the
    steps tested so far: refused before the first test, then one fresh book
    from the validation start stopped at the end of the last tested step --
    the request the last test made -- read step by step as that test read
    its candidate. No span names a step past it, and it counts against the
    interval's allowance like any validate call."""

    with TemporaryDirectory() as tmp:
        harness = _Harness(Path(tmp))
        node = harness.node("a", "1")
        harness.session.candidate("b", _strategy("2"))
        replays = len(harness.evaluator.requests)
        with pytest.raises(ToolError) as refused:
            harness.session.call("b", span="revealed")
        assert refused.value.error_type == "schema_error" and "no step is tested yet" in str(refused.value)
        assert len(harness.evaluator.requests) == replays and harness.backtest.validates_in_interval == 1

        tested = harness.test.invoke({"candidate": node}).value
        # The tested candidate's own bytes over the revealed span: the test's
        # request, and the test's reading of it.
        same = harness.session.call("a", span="revealed").value
        assert harness.evaluator.requests[-2:] == [("20250701", "20260630", "20250930")] * 2
        assert same["span"] == {"label": "revealed:1", "start": "20250701", "end": "20250930", "steps": 1, "days": 66}
        [row] = same["candidates"]
        assert row["by_step"] == tested["candidate_by_step"] and "by_year" not in row and "concentration" not in row
        assert (row["active"], row["book"]) == (tested["whole_replay"]["active"], tested["whole_replay"]["book"])
        assert same["validates_in_step"] == 1
        dates = set(DATE.findall(json.dumps(same)))
        assert dates and max(dates) <= "20250930", sorted(dates)[-3:]
        # Nothing names a later step: there is no step sub-range, and a year
        # span into the validation year is refused.
        for span in ("revealed:2", "Y5"):
            with pytest.raises(ToolError) as refused:
                harness.session.call("b", span=span)
            assert refused.value.error_type == "schema_error", span
        assert harness.backtest.validates_in_interval == 1 and len(harness.evaluator.requests) == replays + 2

        harness.test.invoke({"candidate": node})
        other = harness.session.call("b", span="revealed").value
        assert harness.evaluator.requests[-1] == ("20250701", "20260630", "20251231")
        assert (other["span"]["label"], other["span"]["end"], other["span"]["steps"]) == ("revealed:2", "20251231", 2)
        assert [step["step"] for step in other["candidates"][0]["by_step"]] == [1, 2]
        # Counted as any call is, live and as a resumed attempt reads it.
        assert other["validates_in_step"] == harness.backtest.validates_in_interval == 1
        since = harness.records()[-1]["recorded_at"]
        assert session_tools._validates_since(harness.backtest.tree, since) == 1


def test_a_revealed_validation_changes_no_record_and_never_registers() -> None:
    """A revealed-span node is a reading: the validation record, the tested
    candidate and the test horizon stay as they were, test takes it neither
    as a candidate nor as a seed replicate, and the null control, which
    ranks a node over research years, refuses it before charging."""

    with TemporaryDirectory() as tmp:
        harness = _Harness(Path(tmp))
        nominee = harness.seeded("a", 1000)
        harness.test.invoke({"candidate": nominee})
        ledger = harness.backtest.ledger.path
        before = (ledger.read_bytes(), json.dumps(harness.backtest.tests), harness.backtest.test_horizon)
        harness.session.candidate("b", _seeded(2000))
        revealed = harness.session.call("b", span="revealed").value["candidates"][0]["node_id"]
        assert (ledger.read_bytes(), json.dumps(harness.backtest.tests), harness.backtest.test_horizon) == before
        for arguments, reason in (
            ({"candidate": revealed}, "registration_needs_full_span_validation"),
            ({"candidate": nominee, "seed_replicates": [revealed]}, "registration_seed_replicate_invalid"),
        ):
            with pytest.raises(ToolError) as refused:
                harness.test.invoke(arguments)
            assert refused.value.error_type == "test_registration_refused" and reason in str(refused.value)
        null_control = session_tools.NullControlTool(harness.backtest, max_calls=1)
        with pytest.raises(ToolError, match="replayed span revealed:1"):
            null_control.invoke({"node_id": revealed})
        assert null_control.used == 0
        assert (ledger.read_bytes(), json.dumps(harness.backtest.tests), harness.backtest.test_horizon) == before


def test_a_validate_call_counts_only_when_a_candidate_completes_its_replay() -> None:
    """The allowance rations readings: a call none of whose candidates
    completed its replay is not counted, whether their own code raised or the
    host failed them, and a call with one completed candidate counts once
    however many failed beside it. A resumed attempt reads the same count off
    the step tree."""

    with TemporaryDirectory() as tmp:
        harness = _Harness(
            Path(tmp),
            validates_per_step=2,
            raise_for={
                "_ = 5": StrategyRaised("KeyError: 'close'"),
                "_ = 6": StrategyExecutionError("Docker generate_orders failed: timeout"),
            },
        )
        harness.session.candidate("s", _strategy("5"))
        harness.session.candidate("e", _strategy("6"))
        for name in ("s", "e"):
            with pytest.raises(ToolError) as failed:
                harness.session.call(name)
            assert failed.value.error_type == "validation_failed"
            assert "the call was not counted" in str(failed.value)
        assert harness.backtest.validates_in_interval == 0
        # Both dead ends are in the tree; neither batch counts on resume either.
        assert len(harness.backtest.tree.nodes()) == 2
        assert session_tools._validates_since(harness.backtest.tree, None) == 0
        harness.session.candidate("a", _strategy("1"))
        mixed = harness.session.call("a", "s", "e")
        assert mixed.ok and mixed.value["failed"] == 2 and mixed.value["validates_in_step"] == 1
        harness.node("b", "2")
        assert harness.backtest.validates_in_interval == 2
        assert session_tools._validates_since(harness.backtest.tree, None) == 2
        harness.session.candidate("c", _strategy("3"))
        with pytest.raises(ToolError) as refused:
            harness.session.call("c")
        assert refused.value.error_type == "validates_per_step_exhausted"
        assert "next test" not in str(refused.value)


def test_the_last_test_finishes_the_session_with_its_candidate() -> None:
    with TemporaryDirectory() as tmp:
        harness = _Harness(Path(tmp))
        a, b = harness.node("a", "1"), harness.node("b", "2")
        for step, node in enumerate((a, a, b, b), start=1):
            result = harness.test.invoke({"candidate": node})
            assert result.finish is (step == 4), step
        assert result.value["outcome"] == "completed" and result.value["candidate"]["node_id"] == b
        assert "registered strategy" in result.value["pipeline_outcome"]
        records = harness.records()
        assert [row["candidate"]["step_id"] for row in records] == [a, a, b, b]
        assert candidate_changes(harness.backtest.ledger.read()) == 1
        with pytest.raises(ToolError, match="validation period is complete"):
            harness.test.invoke({"candidate": b})


def test_nothing_returned_to_the_agent_dates_past_the_test_horizon() -> None:
    with TemporaryDirectory() as tmp:
        harness = _Harness(Path(tmp), test_step_months=1)
        node = harness.node("a", "1")
        result = harness.test.invoke({"candidate": node})
        test_horizon = result.value["test_horizon"]
        assert test_horizon == "20250731"
        dates = set(DATE.findall(json.dumps(result.value)))
        assert dates and max(dates) <= test_horizon, sorted(dates)[-3:]
        # The replay stopped at the test horizon, and its table has one monthly row.
        assert harness.evaluator.requests[-1][2] == test_horizon
        assert [(row["step"], row["start"], row["end"]) for row in result.value["record"]] == [
            (1, "20250701", "20250731")
        ]
        assert [row["step"] for row in result.value["candidate_by_step"]] == [1]


def test_a_test_whose_replay_fails_leaves_the_test_horizon_where_it_was() -> None:
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
        assert "test horizon did not move" in str(raised.value)
        with pytest.raises(ToolError) as failed:
            harness.test.invoke({"candidate": environment})
        assert failed.value.details["cause"] == "environment" and "resent unchanged" in str(failed.value)
        assert harness.records() == [] and harness.backtest.test_horizon == "20250630"
        assert harness.backtest.test_replays == 2
        # A second host failure in a row ends the attempt, not the research.
        with pytest.raises(EnvironmentBlocked, match="2 replays failed on the environment"):
            harness.test.invoke({"candidate": environment})
        # A good candidate still advances.
        assert harness.test.invoke({"candidate": good}).value["step"] == 1


def test_seed_replicates_are_replayed_beside_the_candidate_and_read_step_by_step() -> None:
    """Every test that names seed replicates replays each beside the candidate
    from the validation start: each candidate_by_step row reads every
    replicate over its step, the record keeps their recorded entries beside
    the candidate's, and what the candidate reads -- its rows, its whole
    replay, the record's significance -- is what it reads alone."""

    with TemporaryDirectory() as tmp, TemporaryDirectory() as alone_tmp:
        harness = _Harness(Path(tmp))
        nominee, second, third = (harness.seeded(name, seed) for name, seed in (("a", 1000), ("b", 2000), ("c", 3000)))
        handles = harness.backtest.named([second, third])
        first = harness.test.invoke({"candidate": nominee, "seed_replicates": handles}).value
        # Three replays from the validation start to the new horizon, one per leg.
        assert harness.evaluator.requests[-3:] == [("20250701", "20260630", "20250930")] * 3
        [record] = harness.records()
        assert record["candidate"]["seed_replicates"] == [second, third]
        steps = {item.step_id: item for item in harness.backtest.steps}
        assert [(item["step_id"], item["revision_id"]) for item in record["replicates"]] == [
            (second, steps[second].revision_id),
            (third, steps[third].revision_id),
        ]
        assert all((item["entry"]["start"], item["entry"]["end"]) == ("20250701", "20250930") for item in record["replicates"])
        # Each replicate's row is its own replay's reading over the step.
        revision = harness.backtest.artifact_store.revision(steps[second].revision_id)
        own = harness.evaluator.evaluate(
            harness.backtest.validation_request(revision, harness.backtest.request.validation), end_day="20250930"
        )
        expected = step_entry(*replay_and_analysis(own.result_ref), start="20250701", end="20250930")
        # The host registers the last test's replicates by their full ids.
        assert first["candidate"]["seed_replicates"] == [second, third]
        [row] = first["candidate_by_step"]
        assert [cell["handle"] for cell in row["seed_replicates"]] == handles
        cell = row["seed_replicates"][0]
        assert list(cell) == ["handle", "active_return", "return"]
        for key in ("active_return", "return"):
            assert cell[key] == pytest.approx(expected[key], abs=1e-6), key
        assert cell["active_return"] != pytest.approx(row["active_return"], abs=1e-6)
        # The record carries the replicates' recorded entries, by handle.
        assert first["record"][0]["seed_replicates"] == [
            {
                "handle": handle,
                "active_return": pytest.approx(item["entry"]["active_return"], abs=1e-6),
                "return": pytest.approx(item["entry"]["return"], abs=1e-6),
            }
            for handle, item in zip(handles, record["replicates"], strict=True)
        ]
        # The next test, naming them by full id, reads them over every step so far.
        later = harness.test.invoke({"candidate": nominee, "seed_replicates": [second, third]}).value
        assert [row["step"] for row in later["candidate_by_step"]] == [1, 2]
        assert later["candidate_by_step"][0]["seed_replicates"] == row["seed_replicates"]
        assert [cell["handle"] for cell in later["candidate_by_step"][1]["seed_replicates"]] == handles
        assert [len(row["seed_replicates"]) for row in later["record"]] == [2, 2]

        # The same candidate tested alone reads the same as it did beside them.
        solo = _Harness(Path(alone_tmp))
        solo.seeded("a", 1000)
        alone = [solo.test.invoke({"candidate": solo.backtest.steps[0].step_id}).value for _ in range(2)][-1]
        assert [
            {k: v for k, v in row.items() if k != "seed_replicates"} for row in later["candidate_by_step"]
        ] == alone["candidate_by_step"]
        assert later["significance"] == alone["significance"] and later["whole_replay"] == alone["whole_replay"]


def test_a_replicate_that_raises_is_a_reading_and_one_the_host_fails_fails_the_step() -> None:
    """A replicate whose own code raises is recorded with its error and the
    step stands, as forward reads one; a replicate the host failed measured
    nothing, so the whole step fails as a host failure of the candidate does:
    nothing is recorded, the horizon stays, and the call may be resent."""

    with TemporaryDirectory() as tmp:
        harness = _Harness(Path(tmp))
        nominee, raised, hosted = (harness.seeded(name, seed) for name, seed in (("a", 1000), ("b", 2000), ("c", 3000)))
        host_failure = StrategyExecutionError("Docker generate_orders failed: timeout")
        harness.evaluator.raise_for = {"SEED_BASE = 3000": host_failure}
        handle = harness.backtest.named([hosted])[0]
        with pytest.raises(ToolError) as failed:
            harness.test.invoke({"candidate": nominee, "seed_replicates": [raised, hosted]})
        assert failed.value.error_type == "test_replay_failed"
        assert failed.value.details["cause"] == "environment" and failed.value.details["seed_replicate"] == handle
        assert f"seed replicate {handle}'s replay" in str(failed.value) and "resent unchanged" in str(failed.value)
        assert harness.records() == [] and harness.backtest.test_horizon == "20250630"
        # It counts toward ending the attempt like the candidate's own host failure.
        with pytest.raises(EnvironmentBlocked, match="2 replays failed on the environment"):
            harness.test.invoke({"candidate": nominee, "seed_replicates": [hosted]})
        assert harness.records() == []

        harness.evaluator.raise_for = {"SEED_BASE = 2000": StrategyRaised("KeyError: 'close'")}
        result = harness.test.invoke({"candidate": nominee, "seed_replicates": [raised, hosted]})
        assert result.ok and result.value["step"] == 1 and harness.backtest.test_horizon == "20250930"
        [record] = harness.records()
        error_row, completed = record["replicates"]
        assert (error_row["step_id"], error_row["result_ref"]) == (raised, None) and "entry" not in error_row
        assert "KeyError: 'close'" in error_row["strategy_error"]
        assert completed["step_id"] == hosted and completed["entry"]["end"] == "20250930"
        name = harness.backtest.named([raised])[0]
        [row] = result.value["candidate_by_step"]
        assert row["seed_replicates"][0] == {"handle": name, "strategy_error": error_row["strategy_error"]}
        assert result.value["record"][0]["seed_replicates"][0] == row["seed_replicates"][0]
        assert harness.backtest.test_replays == 3


def test_a_test_replays_its_legs_under_the_card_and_clock_rules_of_validate() -> None:
    """The candidate and its replicates replay as validate's candidates do, at
    batch_validate_max_concurrency at once: side by side, each leg on one of
    the experiment's cards under both strategy clocks widened to the width;
    a lone replay, of a test or of a validate, attaches every card under the
    base clocks. So a candidate's environment depends only on the width,
    never on which tool replays it."""

    with TemporaryDirectory() as tmp:
        harness = _Harness(Path(tmp), max_concurrency=2)
        nominee, second, third = (harness.seeded(name, seed) for name, seed in (("a", 1000), ("b", 2000), ("c", 3000)))
        cards = SandboxConfig(spec=SandboxSpec(gpu=(1, 4), gpu_count=2))
        base = (cards.timeout_seconds, cards.fit_timeout_seconds)
        widened = (2 * cards.timeout_seconds, 2 * cards.fit_timeout_seconds)
        harness.evaluator.sandbox = cards
        legs = harness.evaluator.legs
        legs.clear()
        harness.test.invoke({"candidate": nominee, "seed_replicates": [second, third]})
        assert sorted(source for source, _card, _clocks in legs) == sorted(_seeded(seed) for seed in (1000, 2000, 3000))
        assert {(card, clocks) for _source, card, clocks in legs} <= {((1,), widened), ((4,), widened)}
        assert harness.evaluator.sandbox is cards
        legs.clear()
        harness.test.invoke({"candidate": nominee})
        assert legs == [(_seeded(1000), None, base)]
        # validate, at the same widths, under the same rules.
        for name, marker in (("x", "21"), ("y", "22"), ("z", "23")):
            harness.session.candidate(name, _strategy(marker))
        legs.clear()
        assert harness.session.call("x", "y", span="Y2").ok
        assert {(card, clocks) for _source, card, clocks in legs} == {((1,), widened), ((4,), widened)}
        legs.clear()
        assert harness.session.call("z", span="Y2").ok
        assert [(card, clocks) for _source, card, clocks in legs] == [(None, base)]


def test_a_seed_that_lacks_the_prebuilt_asof_state_ends_the_attempt_by_name() -> None:
    with TemporaryDirectory() as tmp:
        harness = _Harness(Path(tmp))
        node = harness.node("a", "1")

        def missing(request):
            raise RuntimeError(f"the prebuilt view cache carries no prebuilt as-of state for replay slot {request.start}")

        harness.evaluator.require_prebuilt_asof_state = missing
        with pytest.raises(EnvironmentBlocked, match=r"test step 1: .*no prebuilt as-of state for replay slot 20250701"):
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
