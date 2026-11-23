"""Delistings and delisting consolidation periods, the exchange facts the
Broker's delisting rules (``BrokerProfile.delisting_rules``) act on.

Read host-side from the raw lake and never shown to a strategy:

- ``stock_basic`` (``list_status=D``) dates each delisted code's delisting;
- ``namechange`` gives the name in force on a day, read as a decision view
  reads it (``snapshot.SnapshotBuilder._names_as_of``): of the rows started and
  announced by that day, the one that started last.

A position still held on its delisting date is settled at that session's open:
at zero for a forced delisting, at its last close for a clean one (a merger,
privatisation or transfer of listing, whose holders receive shares or cash).
The delisting is forced when the name in force on the delisting date is ST,
*ST or 退, or when the last close is below the one-CNY face value (the
face-value delisting, whose names need not carry ST). A code with no name on
record by its delisting date and a last close of at least one CNY cannot be
told either way, and settling it is refused.

A buy is refused while the name in force is the code's delisting consolidation
period (退市整理期) name and the code is not yet delisted. Each fact is acted on
only from the day it is public: the delisting on its date, the consolidation
period from the later of its start and its announcement.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

import pandas as pd

# ``namechange.change_reason`` of a delisting consolidation period's name.
CONSOLIDATION_REASON = "退市整理期"
# A forced delisting's name marks: ST and *ST (risk warnings) and 退 (the
# consolidation period's name).
DISTRESS_MARKS = ("ST", "退")
# A name whose last close is below the face value is delisted by force
# (交易类强制退市), whatever its name.
FACE_VALUE_CNY = 1.0

T = TypeVar("T")


@dataclass(frozen=True)
class Delisting:
    """One code's delisting: its date (``YYYYMMDD``) and whether the name in
    force on it carries a distress mark (None: no name on record by then)."""

    symbol: str
    date: str
    distressed: bool | None

    def settlement_price(self, last_close: float) -> float:
        """What one held share is settled at, given the last close it was marked at."""

        if self.distressed or last_close < FACE_VALUE_CNY:
            return 0.0
        if self.distressed is None:
            raise ValueError(
                f"{self.symbol} was delisted on {self.date} with no name on record by then and a last "
                f"close of {last_close}: a forced delisting cannot be told from a clean one, so the "
                "position cannot be settled"
            )
        return last_close


class DelistingTable:
    """Every delisting of the lake, and the names that tell a consolidation period."""

    def __init__(
        self,
        delistings: Mapping[str, Delisting],
        names: Mapping[str, Sequence[tuple[str, bool]]],
    ) -> None:
        """``names`` maps each code that ever had a consolidation-period name
        to its names in start order, each as (the day it is public from,
        whether it is the consolidation period's name)."""

        self._delistings = dict(delistings)
        self._names = {symbol: tuple(rows) for symbol, rows in names.items()}
        self._consolidating: dict[str, frozenset[str]] = {}

    @classmethod
    def from_raw(cls, raw_dir: str | Path) -> DelistingTable:
        root = Path(raw_dir)
        return cls.from_frames(
            pd.read_parquet(root / "stock_basic" / "list_status=D.parquet", columns=["ts_code", "delist_date"]),
            pd.read_parquet(
                root / "namechange" / "namechange.parquet",
                columns=["ts_code", "name", "start_date", "ann_date", "change_reason"],
            ),
        )

    @classmethod
    def from_frames(cls, delisted: pd.DataFrame, namechange: pd.DataFrame) -> DelistingTable:
        """``delisted`` is ``stock_basic``'s ``list_status=D`` rows
        (``ts_code``, ``delist_date``), ``namechange`` the name history."""

        dates: dict[str, str] = {}
        for symbol, value in zip(delisted["ts_code"].astype(str), delisted["delist_date"], strict=True):
            day = _day(value)
            if day is None:
                raise ValueError(f"stock_basic lists {symbol} as delisted without a valid delist_date: {value!r}")
            if dates.setdefault(symbol, day) != day:
                raise ValueError(f"stock_basic lists {symbol} as delisted on {dates[symbol]} and on {day}")
        consolidation = set(namechange.loc[namechange["change_reason"] == CONSOLIDATION_REASON, "ts_code"].astype(str))
        rows = namechange[namechange["ts_code"].astype(str).isin(set(dates) | consolidation)]
        # Each code's names in start order (ties in file order), each as the
        # day it is public from, the name, and whether it is a consolidation
        # period's name.
        history: dict[str, list[tuple[str, str, str, bool]]] = {}
        for symbol, name, start, announced, reason in zip(
            rows["ts_code"].astype(str),
            rows["name"],
            rows["start_date"],
            rows["ann_date"],
            rows["change_reason"],
            strict=True,
        ):
            first = _day(start)
            if first is None:
                raise ValueError(f"namechange has a name of {symbol} without a valid start_date: {start!r}")
            history.setdefault(symbol, []).append(
                (first, max(first, _day(announced) or first), str(name), reason == CONSOLIDATION_REASON)
            )
        names: dict[str, tuple[tuple[str, tuple[str, bool]], ...]] = {}
        for symbol, entries in history.items():
            entries.sort(key=lambda entry: entry[0])
            names[symbol] = tuple((public, (name, flag)) for _start, public, name, flag in entries)
        delistings = {}
        for symbol, day in dates.items():
            name = _in_force(names.get(symbol, ()), day)
            delistings[symbol] = Delisting(
                symbol, day, None if name is None else any(mark in name[0] for mark in DISTRESS_MARKS)
            )
        return cls(
            delistings,
            {symbol: tuple((public, flag) for public, (_name, flag) in names[symbol]) for symbol in consolidation},
        )

    def delisting(self, symbol: str, day: str) -> Delisting | None:
        """``symbol``'s delisting when it is dated on or before ``day``."""

        delisting = self._delistings.get(symbol)
        return delisting if delisting is not None and delisting.date <= day else None

    def consolidating(self, day: str) -> frozenset[str]:
        """The codes in their delisting consolidation period on ``day``."""

        codes = self._consolidating.get(day)
        if codes is None:
            codes = frozenset(
                symbol
                for symbol, names in self._names.items()
                if _in_force(names, day) and self.delisting(symbol, day) is None
            )
            self._consolidating[day] = codes
        return codes


def _in_force(names: Sequence[tuple[str, T]], day: str) -> T | None:
    """Of ``names`` (in start order, each with the day it is public from),
    the last one public by ``day``: the name in force, as a decision view
    reads it. None before the first."""

    current = None
    for public, name in names:
        if public <= day:
            current = name
    return current


def _day(value: object) -> str | None:
    """``value`` as ``YYYYMMDD``, or None when it states no date."""

    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    text = str(value).strip()
    return text if len(text) == 8 and text.isdigit() else None


__all__ = [
    "CONSOLIDATION_REASON",
    "DISTRESS_MARKS",
    "FACE_VALUE_CNY",
    "Delisting",
    "DelistingTable",
]
