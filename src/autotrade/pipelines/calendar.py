"""The research calendar: SSE trading days and the research geometry.

An arm researches on whole July-June years, freezes at most once, and the frozen
artifact is then replayed once, continuously, over the twelve months after
research and the Held-out period after those:

    research  research_start .. research_end    one slot per year, Y1..Yn
    forward   research_end+1 .. forward_end     slot F
    Held-out  forward_end+1  .. heldout_end     slot H, clipped to the release

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

from autotrade.environment.data.contracts import CN_TZ

# Snapshot anchor time of day: close of business, not an intraday moment.
RESEARCH_ANCHOR_TIME = time(23, 59, 59)
# A replay region needs at least two trading days: one day is a one-point
# equity curve with no daily return series behind it.
MIN_REGION_TRADE_DAYS = 2
# The span label of a replay over the whole research period, Y1 through the
# last research year.
FULL_SPAN = "full"


def yyyymmdd(value: object) -> str:
    """One calendar date as the ``YYYYMMDD`` key every layer stores it under."""

    return pd.Timestamp(str(value).strip()).strftime("%Y%m%d")


def replay_window(
    trade_days: Sequence[str],
    *,
    start: str | None = None,
    max_days: int | None = None,
) -> list[str]:
    """The trading days a truncated replay actually covers.

    One definition for both evaluation backends. An unofficial rehearsal runs a
    short window of a span it must not be able to pass off as a full
    Validation: ``start`` moves its first day to the first trading day at or
    after that date and ``max_days`` caps how many days follow. Neither can
    reach outside the span's own days, and both are applied after the slot
    identity check, never before it.
    """

    days = sorted(str(day) for day in trade_days)
    if start is not None:
        anchor = yyyymmdd(start)
        days = [day for day in days if day >= anchor]
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


def _next_day(day: str) -> str:
    return (date.fromisoformat(day) + timedelta(days=1)).strftime("%Y%m%d")


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
    """Research, forward and Held-out dates of one arm, as ``YYYYMMDD``.

    Refused unless research runs from a July 1 to a June 30 (whole July-June
    years, so every block is comparable), the forward period is exactly the
    twelve months after research, and Held-out ends after the forward period.
    The stages are contiguous by construction and cannot overlap.
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
        if (
            self.research_start[4:] != "0701"
            or self.research_end[4:] != "0630"
            or self.research_end < self.research_start
        ):
            raise ValueError(
                "research must run from a July 1 to a later June 30 so its blocks are "
                f"whole July-June years, got {self.research_start}..{self.research_end}"
            )
        forward_end = f"{int(self.research_end[:4]) + 1}0630"
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
        first, last = int(self.research_start[:4]), int(self.research_end[:4])
        return tuple(
            Slot(f"Y{index}", f"{year}0701", f"{year + 1}0630", f"{year + 1}0630")
            for index, year in enumerate(range(first, last), start=1)
        )

    @property
    def forward(self) -> Slot:
        return Slot("F", self.forward_start, self.forward_end, self.forward_end)

    def heldout(self, trading_days: Sequence[str]) -> Slot:
        """The Held-out slot, its end clipped to the release's last trading day.

        ``trading_days`` are the pinned release's daily dates. The clip is
        stated on the slot (``requested_end``, ``truncation_reason``) rather
        than hidden in a shorter replay under the configured label. A release
        that does not reach two trading days into Held-out cannot replay it --
        and has not finished the forward period either -- so it is refused.
        """

        if not trading_days:
            raise ValueError("the Held-out slot needs the release's trading days")
        end = min(self.heldout_end, max(trading_days))
        count = sum(1 for day in trading_days if self.heldout_start <= day <= end)
        if count < MIN_REGION_TRADE_DAYS:
            raise ValueError(
                f"the release ends {max(trading_days)}, leaving {count} trading day(s) of "
                f"Held-out {self.heldout_start}..{self.heldout_end}; a replay needs at "
                f"least {MIN_REGION_TRADE_DAYS}"
            )
        return Slot("H", self.heldout_start, end, self.heldout_end)

    def to_record(self) -> dict[str, str]:
        return {name: getattr(self, name) for name in GEOMETRY_PARAMETERS}


# The geometry's parameter names, which are also its experiment parameter keys.
GEOMETRY_PARAMETERS: tuple[str, ...] = tuple(field.name for field in fields(ResearchGeometry))


__all__ = [
    "FULL_SPAN",
    "GEOMETRY_PARAMETERS",
    "MIN_REGION_TRADE_DAYS",
    "RESEARCH_ANCHOR_TIME",
    "ResearchGeometry",
    "Slot",
    "anchor_before",
    "load_sse_trading_days",
    "replay_window",
    "yyyymmdd",
]
