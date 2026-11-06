"""The research calendar: SSE trading days and the research geometry.

An experiment researches on whole years that begin on the first of a month. The
first ``research_years`` of them are the research years a session sees; the
rest are the validation years, revealed to the session one step at a time
through its tests (``step_windows``). The frozen artifact is then replayed
once, continuously, over the twelve months after research and the Held-out
period after those, as one book:

    research    research_start .. research_end    one slot per year, Y1..Yn
    forward     research_end+1 .. forward_end     slot F
    Held-out    forward_end+1  .. heldout_end     slot H, clipped to the release

Every slot is anchored at 23:59:59 +08:00 of the calendar day before it starts,
and it carries the rows whose ``available_at`` falls after that anchor and no
later than its own last day. Consecutive slots therefore partition the rows
exactly, so a chain of them reads what one long slot would. An anchor on the
previous trading day would not: a year ending on a weekend (2024-06-30) would
leave that weekend's rows in both neighbours. The forward slot's anchor is
research end, which is also the one decision view a research session reads, so
nothing stamped after research end reaches the Agent.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, fields
from datetime import date, datetime, time, timedelta
from pathlib import Path

import pandas as pd

from autotrade.common.time import CN_TZ
from autotrade.environment.replay.style import MIN_REGRESSION_DAYS

# Snapshot anchor time of day: close of business, not an intraday moment.
RESEARCH_ANCHOR_TIME = time(23, 59, 59)
# The span label of a replay over the research years, Y1 through the last of
# them; and of the validation years as one span, which the tests replay up to
# their test horizon.
FULL_SPAN = "full"
VALIDATION_SPAN = "validation"


def yyyymmdd(value: object) -> str:
    """One calendar date as the ``YYYYMMDD`` key every layer stores it under."""

    return pd.Timestamp(str(value).strip()).strftime("%Y%m%d")


def replay_window(
    trade_days: Sequence[str],
    *,
    start: str | None = None,
    max_days: int | None = None,
    end: str | None = None,
) -> list[str]:
    """The trading days a truncated replay actually covers.

    One definition for both evaluation backends. ``start`` moves the window's
    first day to the first trading day at or after that date and ``max_days``
    caps how many days follow: an unofficial rehearsal runs such a short
    window of a span it must not be able to pass off as a full Validation.
    ``end`` cuts the window at the last trading day on or before that date:
    a test replays its span from the first day to the test horizon.
    None can reach outside the span's own days, and all are applied after the
    slot identity check, never before it.
    """

    days = sorted(str(day) for day in trade_days)
    if start is not None:
        anchor = yyyymmdd(start)
        days = [day for day in days if day >= anchor]
    if end is not None:
        last = yyyymmdd(end)
        days = [day for day in days if day <= last]
    if max_days is not None:
        days = days[:max_days]
    return days


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


def anchor_before(start: str) -> datetime:
    """23:59:59 +08:00 of the calendar day before ``start``."""

    day = date.fromisoformat(start) - timedelta(days=1)
    return datetime.combine(day, RESEARCH_ANCHOR_TIME, tzinfo=CN_TZ)


def months_before(end: str, months: int) -> str:
    """The first day of the ``months``-long window that ends on ``end``."""

    stamp = pd.Timestamp(end) - pd.DateOffset(months=months) + pd.Timedelta(days=1)
    return stamp.strftime("%Y%m%d")


def step_windows(start: str, end: str, months: int) -> tuple[tuple[str, str], ...]:
    """Consecutive ``months``-long calendar windows that tile ``start..end``.

    ``start`` is the first day of a month and ``end`` the last day of one
    (``YYYYMMDD``); each window is ``(first day, last day)``. ``ValueError``
    unless the period is a whole number of such windows, so a step never
    ends inside a month and the last step ends exactly on ``end``.
    """

    if isinstance(months, bool) or not isinstance(months, int) or months < 1:
        raise ValueError(f"step length must be a positive number of months, got {months!r}")
    first = date.fromisoformat(start)
    after = date.fromisoformat(end) + timedelta(days=1)
    if first.day != 1 or after.day != 1 or after <= first:
        raise ValueError(
            f"steps tile a period from the first of a month to the last of one, got {start}..{end}"
        )
    span = (after.year - first.year) * 12 + after.month - first.month
    if span % months:
        raise ValueError(f"{start}..{end} is {span} months, not a whole number of {months}-month steps")
    windows = []
    for index in range(span // months):
        opened = _plus_months(first, index * months)
        closed = _plus_months(first, (index + 1) * months) - timedelta(days=1)
        windows.append((opened.strftime("%Y%m%d"), closed.strftime("%Y%m%d")))
    return tuple(windows)


# The research store holds no daily bars before DAILY_HISTORY_START, and none
# of the index and Shenwan macro series backfilled for the eight-year experiments
# before MACRO_HISTORY_START. A requested input window is clipped to the daily
# floor; ``history_floors`` names both. Other macro tables are not listed:
# each starts at its own download floor. These mirror the lake's download
# floors (docs/DATA_DOCUMENTATION.md) by hand. A view manifest's
# ``coverage_start`` cannot stand in: the research-end view's window rarely
# reaches the floor, and a year's replay view is built only when replayed.
DAILY_HISTORY_START = "20100101"
MACRO_HISTORY_START = "20140701"
BACKFILLED_MACRO_DATASETS = ("index_daily", "index_dailybasic", "index_weight", "sw_daily")


def _next_day(day: str) -> str:
    return (date.fromisoformat(day) + timedelta(days=1)).strftime("%Y%m%d")


def _plus_years(month_start: date, years: int) -> date:
    """The same month-start ``years`` later. ``month_start`` is day 1."""

    return date(month_start.year + years, month_start.month, 1)


def _plus_months(month_start: date, months: int) -> date:
    """The month-start ``months`` later. ``month_start`` is day 1."""

    index = month_start.year * 12 + month_start.month - 1 + months
    return date(index // 12, index % 12 + 1, 1)


@dataclass(frozen=True)
class Slot:
    """One replay slot: ``start..end``, read on top of the view at its anchor."""

    label: str
    start: str
    end: str
    # The configured end. It differs from ``end`` only for a Held-out slot the
    # release cut short.
    requested_end: str

    @property
    def anchor(self) -> datetime:
        return anchor_before(self.start)

    @property
    def truncation_reason(self) -> str | None:
        return f"release_ends_{self.end}" if self.end != self.requested_end else None

    def to_record(self) -> dict[str, object]:
        return {
            "label": self.label,
            "start": self.start,
            "end": self.end,
            "requested_end": self.requested_end,
            "truncation_reason": self.truncation_reason,
            "anchor": self.anchor.isoformat(),
        }


@dataclass(frozen=True)
class ResearchGeometry:
    """Research, forward and Held-out dates of one experiment, as ``YYYYMMDD``.

    Refused unless research runs from the first of a month to the day before
    that same month at least one year later (whole years, so every block is
    comparable), the forward period is exactly the twelve months after research,
    and Held-out ends after the forward period. The stages are contiguous by
    construction and cannot overlap. The day after research end is that
    anniversary month-start, so February and leap days need no special case.
    """

    research_start: str
    research_end: str
    forward_end: str
    heldout_end: str

    def __post_init__(self) -> None:
        for name in GEOMETRY_PARAMETERS:
            value = getattr(self, name)
            if not isinstance(value, str) or not re.fullmatch(r"\d{8}", value):
                raise ValueError(f"{name} must be a YYYYMMDD string, got {value!r}")
            try:
                date.fromisoformat(value)
            except ValueError:
                raise ValueError(f"{name} is not a calendar date: {value!r}") from None
        start = date.fromisoformat(self.research_start)
        anniversary = date.fromisoformat(self.research_end) + timedelta(days=1)
        if (
            start.day != 1
            or anniversary.day != 1
            or anniversary.month != start.month
            or anniversary.year <= start.year
        ):
            raise ValueError(
                "research must run from the first of a month to the day before that "
                "same month at least one year later, so its blocks are whole years, "
                f"got {self.research_start}..{self.research_end}"
            )
        forward_end = (_plus_years(anniversary, 1) - timedelta(days=1)).strftime("%Y%m%d")
        if self.forward_end != forward_end:
            raise ValueError(
                "the forward period is the twelve months after research: "
                f"forward_end must be {forward_end} for research_end "
                f"{self.research_end}, got {self.forward_end}"
            )
        if self.heldout_end <= self.forward_end:
            raise ValueError(
                f"Held-out follows the forward period: heldout_end {self.heldout_end} "
                f"must be after forward_end {self.forward_end}"
            )

    @property
    def forward_start(self) -> str:
        return _next_day(self.research_end)

    @property
    def heldout_start(self) -> str:
        return _next_day(self.forward_end)

    @property
    def research_decision_time(self) -> datetime:
        """The decision view a research session reads: research end, 23:59:59."""

        return anchor_before(self.forward_start)

    @property
    def research_years(self) -> tuple[Slot, ...]:
        start = date.fromisoformat(self.research_start)
        # forward_start is the anniversary month-start: research end plus one day.
        years = date.fromisoformat(self.forward_start).year - start.year
        slots = []
        for index in range(1, years + 1):
            opened = _plus_years(start, index - 1)
            closed = _plus_years(start, index) - timedelta(days=1)
            begin, finish = opened.strftime("%Y%m%d"), closed.strftime("%Y%m%d")
            slots.append(Slot(f"Y{index}", begin, finish, finish))
        return tuple(slots)

    def split_research(self, research_years: int) -> tuple[tuple[Slot, ...], tuple[Slot, ...]]:
        """The first ``research_years`` years, which a session researches on,
        and the validation years after them, which it lives through one step
        at a time. ``ValueError`` unless at least one year is on each side."""

        years = self.research_years
        if (
            isinstance(research_years, bool)
            or not isinstance(research_years, int)
            or not 1 <= research_years < len(years)
        ):
            raise ValueError(
                f"research_years must take at least one of the {len(years)} research "
                f"years and leave at least one validation year, got {research_years!r}"
            )
        return years[:research_years], years[research_years:]

    @property
    def forward(self) -> Slot:
        return Slot("F", self.forward_start, self.forward_end, self.forward_end)

    def heldout(self, trading_days: Sequence[str]) -> Slot:
        """The Held-out slot, its end clipped to the release's last trading day.

        ``trading_days`` are the pinned release's daily dates. The clip is
        stated on the slot (``requested_end``, ``truncation_reason``) rather
        than hidden in a shorter replay under the configured label. Held-out
        is graded by the attribution regression, so a release that leaves fewer
        Held-out trading days than it needs (``MIN_REGRESSION_DAYS``) is refused
        here, before the forward replay runs, not when the verdict reads it.
        """

        if not trading_days:
            raise ValueError("the Held-out slot needs the release's trading days")
        end = min(self.heldout_end, max(trading_days))
        count = sum(1 for day in trading_days if self.heldout_start <= day <= end)
        if count < MIN_REGRESSION_DAYS:
            raise ValueError(
                f"the release ends {max(trading_days)}, leaving {count} trading day(s) of "
                f"Held-out {self.heldout_start}..{self.heldout_end}; grading Held-out "
                f"needs at least {MIN_REGRESSION_DAYS}"
            )
        return Slot("H", self.heldout_start, end, self.heldout_end)

    def to_record(self) -> dict[str, str]:
        return {name: getattr(self, name) for name in GEOMETRY_PARAMETERS}


# The geometry's parameter names, which are also its experiment parameter keys.
GEOMETRY_PARAMETERS: tuple[str, ...] = tuple(field.name for field in fields(ResearchGeometry))


__all__ = [
    "FULL_SPAN",
    "GEOMETRY_PARAMETERS",
    "RESEARCH_ANCHOR_TIME",
    "VALIDATION_SPAN",
    "ResearchGeometry",
    "Slot",
    "anchor_before",
    "load_sse_trading_days",
    "replay_window",
    "step_windows",
    "yyyymmdd",
]
