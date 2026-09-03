"""Development calendar (docs/pipeline-design.md chapter 1).

The development window is one contiguous range of cadence periods. By default
every period is one regular Fold whose validation region is that period and
which has no test region; the Folds run in chronological order with a Meta
session between them, and the last frozen strategy goes straight to the
automatic Held-out replay. ``validation_periods`` widens that validation
region into the trailing window of that many consecutive periods ending at the
Fold's own period, so consecutive Folds step forward by one period and only the
last period of each window is new; that newest period is the Fold's step
region. A window written as one explicit ``start..end`` range is a single
period and therefore a single Fold. With ``test_stage`` the window is cut into
rolling Folds instead: the first period is validation only and every later
period is a Fold named after it, with the preceding period as its validation
region. Either way the months before a validation region are its input window.

Each region's decision-input snapshot is anchored at 23:59:59 of the last
trading day BEFORE the region begins: the agent's frozen research baseline then
holds everything published through that prior day's close but nothing from the
region's first day, whose intraday/pre-open data rolls in only later as each
fixed-cycle inference instant crosses the row's available_at. The inference
time-of-day is the user's choice at experiment creation and is a separate
schedule concern defined in environment-design.md, not this schedule.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, time
from pathlib import Path

import pandas as pd

from autotrade.environment.data.contracts import CN_TZ

QUARTER_PATTERN = re.compile(r"^(\d{4})Q([1-4])$")
# Research-snapshot anchor: end of the prior trading day (close of business),
# not an intraday moment. The decision-input view is frozen as of this time.
RESEARCH_ANCHOR_TIME = time(23, 59, 59)
PERIOD_UNITS = ("week", "month", "quarter", "year")
# Every validation/test/held-out region needs at least two trading days to be
# backtestable at all: a single-day region yields a one-point equity curve with
# no daily return series behind it. Guarded here at schedule build time so a
# fold can never reach the (expensive) sandbox + LLM session doomed.
MIN_REGION_TRADE_DAYS = 2


@dataclass(frozen=True)
class FoldSpec:
    """One development Fold.

    The test region is optional: a regular development Fold (no Test stage)
    has none, and its ledger record carries ``test_period=None`` rather than a
    placeholder. Held-out is the verdict for such Folds.
    """

    fold_id: str
    input_window_start: str
    input_window_end: str
    validation_start: str
    validation_end: str
    valid_decision_time: datetime
    test_start: str | None = None
    test_end: str | None = None
    test_decision_time: datetime | None = None
    # The walk-forward step inside a trailing validation window: the bounds of
    # the Fold's own (newest) period, which is the only part of the window the
    # inherited parent has not been developed on. None when the validation
    # window is a single period and the step is the whole window.
    step_start: str | None = None
    step_end: str | None = None

    def __post_init__(self) -> None:
        test_fields = (self.test_start, self.test_end, self.test_decision_time)
        if any(value is None for value in test_fields) and any(
            value is not None for value in test_fields
        ):
            raise ValueError("a fold test region needs start, end and decision time together")
        if (self.step_start is None) != (self.step_end is None):
            raise ValueError("a fold step region needs start and end together")

    @property
    def has_test(self) -> bool:
        return self.test_start is not None

    @property
    def has_step(self) -> bool:
        """True when the validation window is wider than this Fold's own step."""

        return self.step_start is not None

    def to_record(self) -> dict[str, object]:
        return {
            "fold_id": self.fold_id,
            "input_window": f"{self.input_window_start}..{self.input_window_end}",
            "validation_period": f"{self.validation_start}..{self.validation_end}",
            "test_period": (
                f"{self.test_start}..{self.test_end}" if self.has_test else None
            ),
            "valid_decision_time": self.valid_decision_time.isoformat(),
            "test_decision_time": (
                self.test_decision_time.isoformat()
                if self.test_decision_time is not None
                else None
            ),
        }


def parse_quarter(label: str) -> tuple[int, int]:
    match = QUARTER_PATTERN.match(label.strip())
    if not match:
        raise ValueError(f"invalid quarter label: {label!r} (expected e.g. 2022Q1)")
    return int(match.group(1)), int(match.group(2))


def quarter_bounds(label: str) -> tuple[str, str]:
    year, quarter = parse_quarter(label)
    start = pd.Timestamp(year=year, month=3 * (quarter - 1) + 1, day=1)
    end = start + pd.DateOffset(months=3) - pd.Timedelta(days=1)
    return start.strftime("%Y%m%d"), end.strftime("%Y%m%d")


def period_range(first: str, last: str, *, period: str = "quarter") -> list[str]:
    period = normalize_period(period)
    if _is_explicit_range(first) or _is_explicit_range(last):
        # An explicit label already names one whole region. Cadence arithmetic
        # cannot walk from one such region to another, and re-deriving a cadence
        # label from its start would silently widen it: a 20260101..20260630
        # held-out would come back as the whole of 2026.
        if str(first).strip() != str(last).strip():
            raise ValueError(
                f"an explicit date-range period cannot be enumerated: {first}..{last}"
            )
        period_bounds(first, period=period)
        return [str(first).strip()]
    first_start, _ = period_bounds(first, period=period)
    last_start, _ = period_bounds(last, period=period)
    labels: list[str] = []
    current = pd.Timestamp(first_start)
    last_ts = pd.Timestamp(last_start)
    while current <= last_ts:
        labels.append(_period_label(current, period))
        current = _advance_period(current, period, 1)
        if len(labels) > 5000:
            raise ValueError(f"period range too large or inverted: {first}..{last} ({period})")
    if not labels:
        raise ValueError(f"period range is inverted: {first}..{last} ({period})")
    return labels


def period_bounds(label: str, *, period: str = "quarter") -> tuple[str, str]:
    """Inclusive ``YYYYMMDD`` bounds of one period label.

    A cadence label (``2022Q1``, ``202201``, ``2022``) spans its whole period.
    An explicit ``YYYYMMDD..YYYYMMDD`` label names its region directly at any
    cadence, which is how a held-out window that is not a whole cadence period
    is expressed.
    """
    period = normalize_period(period)
    if _is_explicit_range(label):
        start, end = [yyyymmdd(part) for part in str(label).split("..", maxsplit=1)]
        if end < start:
            raise ValueError(f"period range end precedes start: {label!r}")
        return start, end
    if period == "quarter":
        return quarter_bounds(str(label))
    start = _period_start(str(label), period)
    end = _advance_period(start, period, 1) - pd.Timedelta(days=1)
    return start.strftime("%Y%m%d"), end.strftime("%Y%m%d")


def first_trading_day(start: str, end: str, trading_days: list[str]) -> str:
    for day in trading_days:
        if start <= day <= end:
            return day
    raise ValueError(f"no trading day inside {start}..{end}")


def build_fold_schedule(
    development_first_period: str,
    development_last_period: str,
    trading_days: list[str],
    *,
    window_months: int,
    period: str = "quarter",
    min_region_trade_days: int = MIN_REGION_TRADE_DAYS,
    test_stage: bool = False,
    validation_periods: int = 1,
) -> list[FoldSpec]:
    """Folds of the development window ``first..last`` (inclusive labels).

    ``test_stage=False``: one regular Fold per period of the window, in
    chronological order, each named after its period and validated on it, with
    no test region. An explicit ``start..end`` window is one period, so it
    yields the single Fold ``fold_<start>..<end>``. With
    ``validation_periods=N`` (quarterly cadence only) a Fold is still named
    after one period but is validated on the N consecutive periods ending at
    it, so the first N-1 periods of the window only ever serve as history,
    consecutive Folds differ by their last period alone, and that last period
    is carried as the Fold's step region.
    ``test_stage=True``: rolling Folds inside the window, one per period after
    the first; each is named after its test period and validated on the period
    before it, so the whole window is used and nothing hidden precedes it.
    """
    period = normalize_period(period)
    if (
        isinstance(validation_periods, bool)
        or not isinstance(validation_periods, int)
        or validation_periods < 1
    ):
        raise ValueError(
            f"validation_periods must be a positive integer, got {validation_periods!r}"
        )
    if validation_periods > 1 and period != "quarter":
        # Quarterly steps are the only cadence this design is defined for; a
        # trailing window at another cadence would silently reshape the
        # research calendar instead of failing.
        raise ValueError(
            "a multi-period validation window is only supported at quarterly cadence: "
            f"fold_period={period!r} with validation_periods={validation_periods}"
        )
    if validation_periods > 1 and test_stage:
        raise ValueError(
            "a rolling Test stage does not support a multi-period validation window: "
            f"test_stage=True with validation_periods={validation_periods}"
        )
    labels = period_range(development_first_period, development_last_period, period=period)
    if not test_stage:
        if len(labels) < validation_periods:
            raise ValueError(
                f"validation_periods={validation_periods} needs at least that many development "
                f"periods, got {len(labels)}: {labels}"
            )
        folds = []
        for index in range(validation_periods - 1, len(labels)):
            label = labels[index]
            step_start, step_end = period_bounds(label, period=period)
            validation_start = (
                step_start
                if validation_periods == 1
                else period_bounds(labels[index - validation_periods + 1], period=period)[0]
            )
            validation_end = step_end
            _require_min_trade_days(
                f"fold_{label} validation", validation_start, validation_end, trading_days, min_region_trade_days
            )
            folds.append(
                _fold_spec(
                    f"fold_{label}",
                    validation_start,
                    validation_end,
                    trading_days,
                    window_months=window_months,
                    # A single-period window has no separate step: the whole
                    # validation region is the step, and a second copy of the
                    # same bounds would be a second source for it.
                    step_start=None if validation_periods == 1 else step_start,
                    step_end=None if validation_periods == 1 else step_end,
                )
            )
        return folds
    if len(labels) < 2:
        raise ValueError(
            f"test_stage needs at least two development periods, got {labels}: the first "
            "period is validation only and every later period is a test period"
        )
    folds: list[FoldSpec] = []
    for validation_label, test_label in zip(labels, labels[1:]):
        validation_start, validation_end = period_bounds(validation_label, period=period)
        test_start, test_end = period_bounds(test_label, period=period)
        _require_min_trade_days(
            f"fold_{test_label} validation", validation_start, validation_end, trading_days, min_region_trade_days
        )
        _require_min_trade_days(f"fold_{test_label} test", test_start, test_end, trading_days, min_region_trade_days)
        folds.append(
            _fold_spec(
                f"fold_{test_label}",
                validation_start,
                validation_end,
                trading_days,
                window_months=window_months,
                test_start=test_start,
                test_end=test_end,
            )
        )
    return folds


def _fold_spec(
    fold_id: str,
    validation_start: str,
    validation_end: str,
    trading_days: list[str],
    *,
    window_months: int,
    test_start: str | None = None,
    test_end: str | None = None,
    step_start: str | None = None,
    step_end: str | None = None,
) -> FoldSpec:
    window_start = pd.Timestamp(validation_start) - pd.DateOffset(months=window_months)
    window_end = pd.Timestamp(validation_start) - pd.Timedelta(days=1)
    return FoldSpec(
        fold_id=fold_id,
        input_window_start=window_start.strftime("%Y%m%d"),
        input_window_end=window_end.strftime("%Y%m%d"),
        validation_start=validation_start,
        validation_end=validation_end,
        valid_decision_time=_decision_time(validation_start, validation_end, trading_days),
        test_start=test_start,
        test_end=test_end,
        test_decision_time=(
            _decision_time(test_start, test_end, trading_days)
            if test_start is not None and test_end is not None
            else None
        ),
        step_start=step_start,
        step_end=step_end,
    )


def heldout_periods(
    first_period: str,
    last_period: str,
    trading_days: list[str],
    *,
    period: str = "quarter",
    min_region_trade_days: int = MIN_REGION_TRADE_DAYS,
) -> list[dict[str, object]]:
    """Held-out replay periods.

    Cadence labels enumerate; a single explicit ``YYYYMMDD..YYYYMMDD`` label
    replays exactly that region, which is how a held-out window shorter than one
    cadence period is configured.
    """
    periods = []
    period = normalize_period(period)
    for label in period_range(first_period, last_period, period=period):
        start, end = period_bounds(label, period=period)
        _require_min_trade_days(f"held-out {label}", start, end, trading_days, min_region_trade_days)
        periods.append(
            {
                "label": label,
                "start": start,
                "end": end,
                "decision_time": _decision_time(start, end, trading_days),
            }
        )
    return periods


def assert_no_overlap(development_last_period: str, heldout_first_period: str, *, period: str = "quarter") -> None:
    """Held-out must be configured upfront and not overlap development."""
    dev_end = period_bounds(development_last_period, period=period)[1]
    heldout_start = period_bounds(heldout_first_period, period=period)[0]
    if heldout_start <= dev_end:
        raise ValueError(
            f"held-out starts {heldout_start} but development runs through {dev_end}; periods must not overlap"
        )


def load_sse_trading_days(raw_dir: str | Path) -> list[str]:
    calendar_dir = Path(raw_dir) / "trade_cal" / "exchange=SSE"
    if not calendar_dir.exists():
        raise FileNotFoundError(f"missing SSE trade calendar: {calendar_dir}")
    frames = [pd.read_parquet(path, columns=["cal_date", "is_open"]) for path in sorted(calendar_dir.glob("year=*.parquet"))]
    if not frames:
        raise FileNotFoundError(f"no trade calendar partitions under {calendar_dir}")
    calendar = pd.concat(frames, ignore_index=True)
    open_days = calendar[calendar["is_open"].astype(str) == "1"]["cal_date"].astype(str)
    return sorted(set(open_days))


def _require_min_trade_days(
    region: str, start: str, end: str, trading_days: list[str], minimum: int = MIN_REGION_TRADE_DAYS
) -> None:
    count = sum(1 for day in trading_days if start <= day <= end)
    if count < minimum:
        raise ValueError(
            f"{region} region {start}..{end} has {count} trading day(s); replay needs at least "
            f"{minimum} (a shorter region has no daily return series)"
        )


def _decision_time(start: str, end: str, trading_days: list[str]) -> datetime:
    """Research-snapshot anchor: 23:59:59 of the trading day before the period.

    Freezing the decision-input snapshot at the close of the prior trading day keeps
    the agent's research baseline strictly pre-period; the period's own data becomes
    visible only as each fixed-cycle inference instant crosses each row's available_at.
    """
    first_day = first_trading_day(start, end, trading_days)
    anchor_day = _prior_trading_day(first_day, trading_days)
    return datetime.strptime(anchor_day, "%Y%m%d").replace(
        hour=RESEARCH_ANCHOR_TIME.hour,
        minute=RESEARCH_ANCHOR_TIME.minute,
        second=RESEARCH_ANCHOR_TIME.second,
        tzinfo=CN_TZ,
    )


def _prior_trading_day(day: str, trading_days: list[str]) -> str:
    earlier = [d for d in trading_days if d < day]
    if not earlier:
        raise ValueError(f"no trading day before {day}; cannot anchor the research snapshot")
    return max(earlier)


def _is_explicit_range(label: object) -> bool:
    return ".." in str(label)


def normalize_period(period: str) -> str:
    value = str(period or "quarter").lower().strip()
    aliases = {"weekly": "week", "monthly": "month", "quarterly": "quarter", "yearly": "year", "annual": "year"}
    value = aliases.get(value, value)
    if value not in PERIOD_UNITS:
        raise ValueError(f"unsupported fold period: {period!r}; expected one of {PERIOD_UNITS}")
    return value


def _period_start(label: str, period: str) -> pd.Timestamp:
    text = str(label).strip()
    if period == "year" and re.fullmatch(r"\d{4}", text):
        return pd.Timestamp(year=int(text), month=1, day=1)
    if period == "month" and re.fullmatch(r"\d{6}", text):
        return pd.Timestamp(year=int(text[:4]), month=int(text[4:6]), day=1)
    return pd.Timestamp(yyyymmdd(text))


def _advance_period(value: pd.Timestamp, period: str, step: int) -> pd.Timestamp:
    if period == "week":
        return value + pd.Timedelta(weeks=step)
    if period == "month":
        return value + pd.DateOffset(months=step)
    if period == "quarter":
        return value + pd.DateOffset(months=3 * step)
    if period == "year":
        return value + pd.DateOffset(years=step)
    raise ValueError(f"unsupported fold period: {period}")


def _period_label(start: pd.Timestamp, period: str) -> str:
    if period == "month":
        return start.strftime("%Y%m")
    if period == "quarter":
        quarter = (start.month - 1) // 3 + 1
        return f"{start.year}Q{quarter}"
    if period == "year":
        return start.strftime("%Y")
    return start.strftime("%Y%m%d")


def yyyymmdd(value: str) -> str:
    text = str(value).strip()
    parsed = pd.Timestamp(text)
    return parsed.strftime("%Y%m%d")
