"""The validation record: the test horizon, its test steps and their significance test.

An experiment's validation years (the research geometry's years after the first
``research_years``) are tiled into steps of ``test_step_months`` (``ValidationSteps``).
Each ``test`` the session makes replays its candidate as one fresh book from
the validation start to the end of the next step, and the candidate's result
over that newest step is the step's entry in the experiment's validation record: one
``test_step`` ledger record per step (:func:`test_step_record`), appended as
the test completes and never rewritten, so a resumed attempt continues from
the test horizon the ledger states (:func:`horizon_after`). A step the host
completes after the session (budget out, or the agent never tested) is a
record of the same shape with ``source`` ``host_completion``.

The record's significance test (:func:`record_significance`) reads the entries' daily
rows stitched in step order with the residual statistic every other reading
uses (``pipelines/verdict.py``): the residual information ratio of the active
series on the market, size and value legs, its betas, the threshold for the record's
length and the one-sided null p-value, beside the stitched return, drawdown
and turnover. The per-step table a test returns (:func:`step_table`) reads
its own replay step by step.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from itertools import pairwise

from autotrade.environment.replay.stats import (
    compounded_drawdown,
    compounded_return,
    window_activity,
    window_fees,
)
from autotrade.environment.replay.style import FACTOR_SERIES, _series_pairs

from .calendar import Slot, step_windows
from .config import ReplaySpan
from .ledger import TEST_STEP_RECORD_TYPE
from .verdict import null_p_value, residual_reading

# The columns of one entry's daily rows, in order: the date, the book's return,
# the random baseline's, and the three factor legs the significance test regresses on.
DAILY_COLUMNS = ("trade_date", "strategy_return", "baseline_return", *(name for name, _series in FACTOR_SERIES))
_DAILY_SERIES = ("strategy_daily", "baseline_daily", *(series for _name, series in FACTOR_SERIES))
# Where a test_step record came from: the session's own test, or the host
# filling the record after the session.
SOURCE_TEST = "test"
SOURCE_HOST_COMPLETION = "host_completion"


@dataclass(frozen=True)
class ValidationSteps:
    """The validation years as consecutive test steps.

    ``start``/``end`` are the validation years' calendar bounds and
    ``windows`` the steps' ``(first day, last day)`` in order. The test horizon is
    the last day a session may read: the day before ``start`` after no test,
    the end of step k after k tests.
    """

    start: str
    end: str
    windows: tuple[tuple[str, str], ...]

    @classmethod
    def of(cls, validation: Sequence[Slot] | Sequence[ReplaySpan], step_months: int) -> ValidationSteps:
        start, end = validation[0].start, validation[-1].end
        return cls(start, end, step_windows(start, end, step_months))

    @property
    def count(self) -> int:
        return len(self.windows)

    def window(self, step: int) -> tuple[str, str]:
        """The calendar bounds of the 1-based ``step``."""

        if not 1 <= step <= self.count:
            raise ValueError(f"step must be 1..{self.count}, got {step!r}")
        return self.windows[step - 1]

    def horizon_after(self, steps_tested: int) -> str:
        """The last readable day once ``steps_tested`` steps are revealed."""

        if not 0 <= steps_tested <= self.count:
            raise ValueError(f"steps_tested must be 0..{self.count}, got {steps_tested!r}")
        if steps_tested == 0:
            return (date.fromisoformat(self.start) - timedelta(days=1)).strftime("%Y%m%d")
        return self.windows[steps_tested - 1][1]


def _in_window(series: object, start: str, end: str) -> list[tuple[str, float]]:
    return [(day, value) for day, value in _series_pairs(series) if start <= day <= end]


def step_entry(
    replay: Mapping[str, object], analysis: Mapping[str, object], *, start: str, end: str
) -> dict[str, object]:
    """What one window of a completed replay reads as a record entry.

    ``replay`` is the result record and ``analysis`` the style sidecar beside
    it. ``return`` and ``max_drawdown`` compound the book's daily returns over
    the window, ``baseline_return`` and ``index_return`` the random baseline's and the
    benchmark's, ``active_return`` is book minus random baseline, ``turnover`` and
    ``fees`` are the window's own fills (``stats.window_activity``,
    ``stats.window_fees``), and ``daily`` carries the window's rows
    (:data:`DAILY_COLUMNS`) for the record's significance test. A window the replay
    does not cover raises ``ValueError``: an entry is never read off nothing.
    """

    strategy = _in_window(analysis.get("strategy_daily"), start, end)
    if not strategy:
        raise ValueError(f"the replay holds no day in {start}..{end}")
    baseline = dict(_in_window(analysis.get("baseline_daily"), start, end))
    if set(baseline) != {day for day, _value in strategy}:
        raise ValueError(f"the replay carries no random baseline over every day of {start}..{end}")
    legs = {series: dict(_in_window(analysis.get(series), start, end)) for _name, series in FACTOR_SERIES}
    benchmark = [legs["benchmark_daily"][day] for day, _value in strategy if day in legs["benchmark_daily"]]
    activity = window_activity(replay["equity_curve"], replay["executions"], start=start, end=end)  # type: ignore[arg-type]
    book = compounded_return(value for _day, value in strategy)
    baseline_return = compounded_return(baseline[day] for day, _value in strategy)
    return {
        "start": start,
        "end": end,
        "days": len(strategy),
        "return": book,
        "max_drawdown": compounded_drawdown(value for _day, value in strategy),
        "turnover": activity["turnover"],
        "round_trips": activity["round_trips"],
        "fees": window_fees(replay["executions"], start=start, end=end),  # type: ignore[arg-type]
        "baseline_return": baseline_return,
        "active_return": book - baseline_return,
        "index_return": compounded_return(benchmark) if len(benchmark) == len(strategy) else None,
        "daily": [
            [day, value, baseline[day], *(legs[series].get(day) for _name, series in FACTOR_SERIES)]
            for day, value in strategy
        ],
    }


def step_table(
    replay: Mapping[str, object], analysis: Mapping[str, object], steps: ValidationSteps, *, through: int
) -> list[dict[str, object]]:
    """One row per test step 1..``through`` of a replay from the validation
    start, each the step's :func:`step_entry` without its daily rows, so the
    rows follow the step length: quarterly for quarterly steps, monthly for
    monthly ones."""

    rows = []
    for step in range(1, through + 1):
        first, last = steps.window(step)
        entry = step_entry(replay, analysis, start=first, end=last)
        del entry["daily"]
        rows.append({"step": step, **entry})
    return rows


def record_significance(entries: Sequence[Mapping[str, object]], *, alpha: float) -> dict[str, object]:
    """The validation record read as one series.

    The entries' daily rows, stitched in order, give the book's, the random baseline's
    and the legs' daily series; ``verdict.residual_reading`` grades the
    active series at ``alpha`` (residual IR, betas, days, years, threshold), and the
    one-sided null p-value (``verdict.null_p_value``) is stated beside it.
    ``return`` and ``max_drawdown`` compound the stitched book, ``turnover``
    sums the entries' own. A record the statistic cannot read yet (too few
    days, or legs that explain it exactly) says why under ``unmeasured`` and
    carries the compounded figures alone; one it read carries ``unmeasured``
    as None, so every reader sees one shape.
    """

    daily = [row for entry in entries for row in entry.get("daily") or ()]
    series: dict[str, list[list[object]]] = {name: [] for name in _DAILY_SERIES}
    for row in daily:
        day = str(row[0])
        for name, value in zip(_DAILY_SERIES, row[1:], strict=True):
            if value is not None:
                series[name].append([day, value])
    returns = [float(row[1]) for row in daily]
    figures: dict[str, object] = {
        "steps": len(entries),
        "days": len(daily),
        "return": compounded_return(returns),
        "max_drawdown": compounded_drawdown(returns),
        "turnover": sum(float(entry.get("turnover") or 0.0) for entry in entries),
    }
    if not daily:
        return {**figures, "unmeasured": "the record has no entry yet"}
    try:
        reading = residual_reading(series, alpha=alpha)
    except ValueError as exc:
        return {**figures, "unmeasured": str(exc)}
    return {
        **figures,
        **reading,
        "p_value": null_p_value(float(reading["residual_information_ratio"]), float(reading["years"])),  # type: ignore[arg-type]
        "unmeasured": None,
    }


def test_step_record(
    *,
    head: Mapping[str, object],
    steps: ValidationSteps,
    step: int,
    candidate: Mapping[str, object],
    replay: Mapping[str, object],
    entry: Mapping[str, object],
    registration: Mapping[str, object],
    source: str,
) -> dict[str, object]:
    """One ``test_step`` ledger record.

    ``head`` carries the link keys and the session key; ``candidate`` names
    the Step tested (``step_id``, ``revision_id``, ``fingerprint``,
    ``seed_replicates``, ``deterministic``); ``replay`` the whole replay
    (``start``, ``end``, ``result_ref``) the entry was read from; ``entry`` is
    :func:`step_entry` over the step's window, daily rows included;
    ``registration`` what the registration read; ``source`` is
    :data:`SOURCE_TEST` or :data:`SOURCE_HOST_COMPLETION`.
    """

    if source not in (SOURCE_TEST, SOURCE_HOST_COMPLETION):
        raise ValueError(f"unknown test_step source {source!r}")
    first, last = steps.window(step)
    if (entry.get("start"), entry.get("end")) != (first, last):
        raise ValueError(f"the entry covers {entry.get('start')}..{entry.get('end')}, not step {step} {first}..{last}")
    return {
        "record_type": TEST_STEP_RECORD_TYPE,
        **{key: head[key] for key in ("experiment_id", "epoch_id", "fold_id", "run_id", "session_key")},
        "step": step,
        "steps": steps.count,
        "window": {"start": first, "end": last},
        "test_horizon": steps.horizon_after(step),
        "candidate": dict(candidate),
        "replay": dict(replay),
        "entry": dict(entry),
        "registration": dict(registration),
        "source": source,
    }


# The state a registered strategy's readout gives its experiment: a registered strategy
# with its validation record and its forward reading (``ledger.VerdictStatus``).
REGISTERED_WITH_RECORD = "registered_with_record"


def readout(
    records: Sequence[Mapping[str, object]],
    *,
    alpha: float,
    forward: Mapping[str, object] | None = None,
    strategy_error: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """The experiment's readout, what its forward record states where a verdict once
    stood: nothing here passes or fails.

    ``record`` is the validation record read as one series
    (:func:`record_significance`: residual IR, one-sided p-value, threshold, years)
    with its span and how often the candidate changed
    (:func:`candidate_changes`). ``forward`` is the registered strategy's reading
    over the Forward year, the slice ``verdict.forward_slice`` read: its
    lower bound and residual reading; a book that raised in its own code
    during the replay has ``strategy_error`` there instead (the slice it
    raised in and the error), which is a reading too. The block has one shape
    either way, with None where a reading does not exist.
    """

    tests = test_step_records(records)
    block: dict[str, object] = {
        "state": REGISTERED_WITH_RECORD,
        "record": {
            **record_significance([row["entry"] for row in tests], alpha=alpha),  # type: ignore[misc]
            "start": str(tests[0]["window"]["start"]) if tests else None,  # type: ignore[index]
            "end": str(tests[-1]["test_horizon"]) if tests else None,
            "candidate_changes": candidate_changes(records),
        },
    }
    if forward is not None:
        block["forward"] = {
            "lower_bound": forward.get("lower_bound"),
            "residual_reading": dict(forward["residual_reading"]),  # type: ignore[arg-type]
            "strategy_error": None,
        }
    else:
        if strategy_error is None:
            raise ValueError("a readout without a forward slice names the strategy error that left it unread")
        block["forward"] = {"lower_bound": None, "residual_reading": None, "strategy_error": dict(strategy_error)}
    return block


def test_step_records(records: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    """The experiment's ``test_step`` records in step order; a ledger whose steps are
    not exactly 1..k raises, since every reader counts on a contiguous record."""

    rows = sorted(
        (dict(record) for record in records if record.get("record_type") == TEST_STEP_RECORD_TYPE),
        key=lambda record: int(record["step"]),
    )
    if [int(row["step"]) for row in rows] != list(range(1, len(rows) + 1)):
        raise ValueError(
            "the ledger's test steps are not the contiguous 1..k: "
            + ", ".join(str(row["step"]) for row in rows)
        )
    return rows


def horizon_after(records: Sequence[Mapping[str, object]], steps: ValidationSteps) -> str:
    """The last day the session may read, by the ledger's test steps."""

    return steps.horizon_after(len(test_step_records(records)))


def candidate_changes(records: Sequence[Mapping[str, object]]) -> int:
    """How many times the candidate changed between consecutive steps of the
    record, by the fingerprint of the bytes each step tested."""

    fingerprints = [str(row["candidate"]["fingerprint"]) for row in test_step_records(records)]
    return sum(1 for before, after in pairwise(fingerprints) if before != after)


__all__ = [
    "DAILY_COLUMNS",
    "REGISTERED_WITH_RECORD",
    "SOURCE_HOST_COMPLETION",
    "SOURCE_TEST",
    "ValidationSteps",
    "candidate_changes",
    "horizon_after",
    "readout",
    "record_significance",
    "step_entry",
    "step_table",
    "test_step_record",
    "test_step_records",
]
