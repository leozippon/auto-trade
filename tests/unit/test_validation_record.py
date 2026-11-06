"""The validation record: step geometry, entries, the significance test and the ledger."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from autotrade.pipelines import validation_record
from autotrade.pipelines.calendar import Slot, step_windows
from autotrade.pipelines.ledger import TEST_STEP_RECORD_TYPE, ExperimentLedger
from autotrade.pipelines.validation_record import (
    SOURCE_HOST_COMPLETION,
    SOURCE_TEST,
    ValidationSteps,
    candidate_changes,
    horizon_after,
    record_significance,
    step_entry,
    step_table,
)

VALIDATION = (
    Slot("Y6", "20200801", "20210731", "20210731"),
    Slot("Y7", "20210801", "20220731", "20220731"),
)
HEAD = {
    "experiment_id": "experiment",
    "epoch_id": "research",
    "fold_id": "research",
    "run_id": "run_1",
    "session_key": "research",
}


def test_step_windows_tile_whole_months_and_refuse_a_ragged_period() -> None:
    quarters = step_windows("20200801", "20250731", 3)
    assert len(quarters) == 20
    assert quarters[0] == ("20200801", "20201031") and quarters[-1] == ("20250501", "20250731")
    assert len(step_windows("20200801", "20250731", 1)) == 60
    # Month ends and leap days need no special case.
    assert step_windows("20231201", "20240229", 1) == (("20231201", "20231231"), ("20240101", "20240131"), ("20240201", "20240229"))
    for start, end, months in (("20200801", "20250731", 7), ("20200802", "20250731", 3), ("20200801", "20250730", 3)):
        with pytest.raises(ValueError):
            step_windows(start, end, months)
    with pytest.raises(ValueError, match="positive number of months"):
        step_windows("20200801", "20250731", 0)


def test_the_validation_steps_know_their_test_horizon() -> None:
    steps = ValidationSteps.of(VALIDATION, 3)
    assert (steps.start, steps.end, steps.count) == ("20200801", "20220731", 8)
    # Before any test the test horizon is the day before the validation start.
    assert steps.horizon_after(0) == "20200731"
    assert steps.horizon_after(1) == "20201031" and steps.horizon_after(8) == "20220731"
    assert steps.window(1) == ("20200801", "20201031") and steps.window(8) == ("20220501", "20220731")
    # The test horizon never moves backwards: every step ends after the one before.
    horizons = [steps.horizon_after(k) for k in range(steps.count + 1)]
    assert horizons == sorted(horizons) and len(set(horizons)) == len(horizons)
    for wrong in (0, 9):
        with pytest.raises(ValueError):
            steps.window(wrong)
    with pytest.raises(ValueError):
        steps.horizon_after(9)


def _days(start: str, count: int) -> list[str]:
    import pandas as pd

    return [day.strftime("%Y%m%d") for day in pd.bdate_range(start, periods=count)]


def _replay(days: list[str], rng: np.random.Generator, *, edge: float = 0.002):
    """A replay record and its style sidecar over ``days``: a book with a
    daily edge over its random baseline, a benchmark and two factor legs."""

    market = rng.normal(0.0003, 0.008, len(days))
    size = rng.normal(0.0, 0.003, len(days))
    value = rng.normal(0.0, 0.003, len(days))
    baselines = market + rng.normal(0.0, 0.002, len(days))
    strategy = baselines + edge + rng.normal(0.0, 0.004, len(days))
    equity = 100_000.0
    curve = []
    executions = []
    for index, (day, value_) in enumerate(zip(days, strategy, strict=True)):
        equity *= 1.0 + value_
        curve.append({"trade_date": day, "equity": equity, "cash": equity * 0.2, "initial_equity": 100_000.0})
        if index % 5 == 0:
            executions.append(
                {
                    "status": "filled",
                    "action": "buy",
                    "matched_at": f"{day[:4]}-{day[4:6]}-{day[6:]}T09:30:00+08:00",
                    "price": 10.0,
                    "quantity": 1000,
                    "commission": 5.0,
                    "stamp_duty": 0.0,
                }
            )
    replay = {"equity_curve": curve, "executions": executions}
    analysis = {
        "strategy_daily": [[day, float(v)] for day, v in zip(days, strategy, strict=True)],
        "panel_daily": [[day, float(v)] for day, v in zip(days, baselines, strict=True)],
        "benchmark_daily": [[day, float(v)] for day, v in zip(days, market, strict=True)],
        "size_factor_daily": [[day, float(v)] for day, v in zip(days, size, strict=True)],
        "value_factor_daily": [[day, float(v)] for day, v in zip(days, value, strict=True)],
    }
    return replay, analysis


def test_an_entry_reads_one_window_of_a_replay_and_nothing_outside_it() -> None:
    days = _days("20200803", 70)
    replay, analysis = _replay(days, np.random.default_rng(1))
    entry = step_entry(replay, analysis, start="20200801", end="20201031")
    inside = [day for day in days if day <= "20201031"]
    assert entry["days"] == len(inside) and [row[0] for row in entry["daily"]] == inside
    assert entry["fees"] == pytest.approx(5.0 * math.ceil(len(inside) / 5))
    assert entry["active_return"] == pytest.approx(entry["return"] - entry["baseline_return"])
    assert entry["index_return"] is not None and entry["turnover"] > 0
    assert 0 <= entry["max_drawdown"] < 1
    with pytest.raises(ValueError, match="holds no day"):
        step_entry(replay, analysis, start="20210801", end="20211031")
    # A replay without a random baseline over the window is no entry.
    partial = {**analysis, "panel_daily": analysis["panel_daily"][1:]}
    with pytest.raises(ValueError, match="random baseline"):
        step_entry(replay, partial, start="20200801", end="20201031")
    # The table a test returns reads its replay step by step: one row per
    # step up to the one tested, each the step's entry without daily rows.
    monthly = ValidationSteps("20200801", "20201031", step_windows("20200801", "20201031", 1))
    rows = step_table(replay, analysis, monthly, through=3)
    assert [(row["step"], row["start"], row["end"]) for row in rows] == [
        (1, "20200801", "20200831"),
        (2, "20200901", "20200930"),
        (3, "20201001", "20201031"),
    ]
    assert all("daily" not in row for row in rows)
    assert sum(row["days"] for row in rows) == entry["days"]
    quarterly = ValidationSteps("20200801", "20210731", step_windows("20200801", "20210731", 3))
    [quarter] = step_table(replay, analysis, quarterly, through=1)
    assert {key: value for key, value in quarter.items() if key != "step"} == {
        key: value for key, value in entry.items() if key != "daily"
    }


def test_the_significance_test_reads_the_stitched_entries_with_the_threshold_for_their_length() -> None:
    days = _days("20200803", 250)
    replay, analysis = _replay(days, np.random.default_rng(2), edge=0.0005)
    steps = ValidationSteps("20200801", "20210731", step_windows("20200801", "20210731", 3))
    entries = [step_entry(replay, analysis, start=first, end=last) for first, last in steps.windows[:3]]
    stitched = record_significance(entries, alpha=0.05)
    assert stitched["steps"] == 3 and stitched["days"] == sum(entry["days"] for entry in entries)
    assert stitched["series"] == "active"
    assert stitched["years"] == pytest.approx(stitched["days"] / 244)
    assert stitched["threshold"] == pytest.approx(1.6449 / math.sqrt(stitched["years"]), abs=1e-3)
    assert 0 < stitched["p_value"] < 1 and stitched["unmeasured"] is None
    # The stitched reading equals the one whole-span reading over the same days.
    whole = step_entry(replay, analysis, start=steps.windows[0][0], end=steps.windows[2][1])
    alone = record_significance([whole], alpha=0.05)
    assert alone["residual_information_ratio"] == pytest.approx(stitched["residual_information_ratio"])
    assert alone["return"] == pytest.approx(stitched["return"])
    # Too few days to regress is said, not failed.
    short = record_significance([{**entries[0], "daily": entries[0]["daily"][:3]}], alpha=0.05)
    assert "unmeasured" in short and "residual_information_ratio" not in short
    assert record_significance([], alpha=0.05)["unmeasured"] == "the record has no entry yet"


def test_the_ledger_records_each_step_once_in_order(tmp_path: Path) -> None:
    days = _days("20200803", 200)
    replay, analysis = _replay(days, np.random.default_rng(3))
    steps = ValidationSteps.of(VALIDATION, 3)
    ledger = ExperimentLedger(tmp_path / "ledger.jsonl")
    assert validation_record.test_step_records([]) == [] and horizon_after([], steps) == "20200731"

    def record(step: int, fingerprint: str, source: str = SOURCE_TEST) -> dict[str, object]:
        window = steps.window(step)
        return validation_record.test_step_record(
            head=HEAD,
            steps=steps,
            step=step,
            candidate={"step_id": f"node_{fingerprint}", "revision_id": "rev", "fingerprint": fingerprint, "seed_replicates": [], "deterministic": False},
            replay={"start": steps.start, "end": steps.horizon_after(step), "result_ref": "result.json"},
            entry=step_entry(replay, analysis, start=window[0], end=window[1]),
            registration={"passed": True, "reasons": []},
            source=source,
        )

    first = record(1, "aaa")
    assert first["record_type"] == TEST_STEP_RECORD_TYPE and first["test_horizon"] == "20201031"
    assert (first["window"], first["steps"], first["source"]) == ({"start": "20200801", "end": "20201031"}, 8, SOURCE_TEST)
    ledger.append(first)
    with pytest.raises(ValueError, match="already recorded test step 1"):
        ledger.append(record(1, "bbb"))
    # An entry that is not the step's own window is refused.
    with pytest.raises(ValueError, match="not step 2"):
        validation_record.test_step_record(
            head=HEAD, steps=steps, step=2, candidate=first["candidate"], replay=first["replay"],
            entry=first["entry"], registration={}, source=SOURCE_TEST,
        )
    with pytest.raises(ValueError, match="unknown test_step source"):
        record(2, "aaa", source="agent")
    ledger.append(record(2, "aaa"))
    ledger.append(record(3, "ccc", source=SOURCE_HOST_COMPLETION))
    rows = validation_record.test_step_records(ledger.read())
    assert [row["step"] for row in rows] == [1, 2, 3]
    assert horizon_after(ledger.read(), steps) == "20210430"
    assert candidate_changes(ledger.read()) == 1
    # A ledger whose steps skip one is corrupt for every reader.
    with pytest.raises(ValueError, match="contiguous"):
        validation_record.test_step_records([first, record(3, "ccc")])
