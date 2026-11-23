"""Delistings and delisting consolidation periods, the exchange facts the
Broker's delisting rules (``BrokerProfile.delisting_rules``) act on.

Read host-side from the raw lake and never shown to a strategy:

- ``stock_basic`` (``list_status=D``) dates each delisted code's delisting;
- ``namechange`` gives the name in force on a day, read as a decision view
  reads it (``snapshot.SnapshotBuilder._names_as_of``): of the rows started and
  announced by that day, the one that started last.

A position still held on its delisting date is settled at that session's open:
at zero for a forced delisting, at its last close for any other (a merger,
privatisation, transfer of listing or voluntary delisting, whose holders
receive shares or cash or are offered cash). The delisting is forced when the
code went through a consolidation period, or when the last close is below the
one-CNY face value (the face-value delisting, whose names need not carry ST).
It is not when the name in force on the delisting date carries no ST. The
rest -- an ST name (or none) in force, no consolidation period, a last close
of at least one CNY -- is a voluntary delisting or a merger as often as a
forced one, so each is settled by what its announcements say (``ANNOUNCED``);
one not listed there is refused when a held position has to be settled.

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
# The marks of a name under a risk warning (ST, *ST) or in a consolidation
# period (退): a delisting whose name in force has none is not forced.
DISTRESS_MARKS = ("ST", "退")
# A name whose last close is below the face value is delisted by force
# (交易类强制退市), whatever its name.
FACE_VALUE_CNY = 1.0
# The delistings the rules cannot classify (an ST name in force, no
# consolidation period, a last close of at least the face value), each by what
# its announcements say: True forced (settled at zero), False not (at the last
# close).
ANNOUNCED: Mapping[str, bool] = {
    "000016.SZ": False,  # *ST康佳A, 2026-10-23: voluntary delisting by shareholder resolution, cash option
    "000023.SZ": True,  # *ST深天, 2024-09-02: forced, market value below 300 million CNY for 20 days
    "000627.SZ": False,  # *ST天茂, 2025-09-30: voluntary delisting by shareholder resolution, cash option
    "000780.SZ": False,  # ST平能, 2022-01-24: absorbed by 龙源电力 in a share swap, dissenters' cash option
    "600213.SH": False,  # *ST亚星, 2024-10-17: voluntary delisting by shareholder resolution, cash option
    "600485.SH": True,  # *ST信威, 2021-06-01: forced, negative net assets and a disclaimer of audit opinion
    "600677.SH": False,  # *ST航通, 2021-03-18: voluntary delisting by shareholder resolution, cash option
    "600680.SH": False,  # *ST上普, 2019-05-23: voluntary delisting by shareholder resolution, cash option
    "603963.SH": True,  # *ST大药, 2025-03-21: forced, market value below 500 million CNY for 20 days
}

T = TypeVar("T")


@dataclass(frozen=True)
class Delisting:
    """One code's delisting: its date (``YYYYMMDD``) and whether it was forced
    (None: neither the rules nor ``ANNOUNCED`` say)."""

    symbol: str
    date: str
    forced: bool | None

    def settlement_price(self, last_close: float) -> float:
        """What one held share is settled at, given the last close it was marked at."""

        if self.forced or last_close < FACE_VALUE_CNY:
            return 0.0
        if self.forced is None:
            raise ValueError(
                f"{self.symbol} was delisted on {self.date} with an ST name (or none) in force, no "
                f"consolidation period and a last close of {last_close}: the rules cannot tell a forced "
                "delisting from a voluntary one or a merger, so the position cannot be settled; add "
                f"{self.symbol} to delisting.ANNOUNCED from its delisting announcement"
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
            entries = names.get(symbol, ())
            name = _in_force(entries, day)
            if any(flag and public <= day for public, (_name, flag) in entries):
                forced: bool | None = True
            elif name is not None and not any(mark in name[0] for mark in DISTRESS_MARKS):
                forced = False
            else:
                forced = ANNOUNCED.get(symbol)
            delistings[symbol] = Delisting(symbol, day, forced)
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
    "ANNOUNCED",
    "CONSOLIDATION_REASON",
    "DISTRESS_MARKS",
    "FACE_VALUE_CNY",
    "Delisting",
    "DelistingTable",
]
