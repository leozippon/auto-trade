"""Where a dataset's partitions live: one layout for the downloader that
writes a partition, its skip check, and the audit that expects it.

A partition is ``<raw>/<dataset>/[<key>=<value>/...]<time>=<unit>.parquet``.
Its directories are the request parameters that split the dataset besides
time (a board tag, a market, a code, a news source, a venue), outermost
first and in the order the downloader walks them; its file is its time unit
(a trade date, a day, a month, a year, a report period) or, for a table
pulled per code, its code. Whole-history files have fixed
names (``full.parquet``, ``<loop>=<value>.parquet``, one range file).

Which time units a run covers stays with the caller: a pull clamps them to
its own window, the audit to the window it checks (:func:`time_units`).
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .common import (
    cli_values,
    date_range_days,
    load_sse_open_dates,
    month_windows,
    safe_partition_value,
    selected_board_hot_is_new,
    selected_news_sources,
    year_windows,
    yyyymmdd_to_month,
    yyyymmdd_to_quarter,
)
from .specs import BOARD_TRADING_SPECS, DatasetSpec, FetchStrategy


@dataclass(frozen=True)
class Partition:
    """One partition: its path, its time unit (a ``YYYYMMDD`` trade date or
    day, a ``(start, end, month|year)`` window, or None for a whole-history
    file) and the other request parameters that name it."""

    path: Path
    unit: Any = None
    dims: dict[str, Any] = field(default_factory=dict)


_TIME_KEYS = {
    FetchStrategy.DAILY_FAMILY: "trade_date",
    FetchStrategy.TRADE_DATE: "trade_date",
    FetchStrategy.TRADE_DATE_BY_TAG: "trade_date",
    FetchStrategy.TRADE_DATE_BY_LIMIT_TYPE: "trade_date",
    FetchStrategy.TRADE_DATE_BY_MARKET: "trade_date",
    FetchStrategy.TRADE_DATE_BY_MARKET_HOT_TYPE: "trade_date",
    FetchStrategy.DAY: "date",
    FetchStrategy.NEWS_SRC_DAY: "date",
    FetchStrategy.RANGE_MONTH: "month",
    FetchStrategy.TIME_RANGE_MONTH: "month",
    FetchStrategy.MONTH_LOOP: "month",
    FetchStrategy.ECO_CAL_MONTH: "month",
    FetchStrategy.DATE_YEAR: "year",
    FetchStrategy.DATE_YEAR_BY_TS_CODE: "year",
    FetchStrategy.PERIOD: "period",
    FetchStrategy.ANN_MONTH: "ann_month",
    FetchStrategy.TS_CODE: "ts_code",
}


def directory_name(value: Any) -> str:
    """A dimension's directory value: percent-encoded, ``all`` for no filter."""
    return safe_partition_value(value) if value else "all"

def by_ts_code_universe(dataset: str, args: argparse.Namespace) -> list[str]:
    """The codes a by-ts_code macro dataset is pulled and audited over."""
    if dataset == "index_global":
        return cli_values(args, "index_code")
    if dataset in ("index_daily", "index_dailybasic"):
        return cli_values(args, "cn_index_code")
    if dataset == "fx_daily":
        return cli_values(args, "fx_code")
    raise RuntimeError(f"no ts_code universe defined for by-ts_code macro dataset {dataset!r}")


def dimensions(spec: DatasetSpec, args: argparse.Namespace | None) -> list[dict[str, Any]]:
    """The values of the dataset's non-time dimensions, outermost first."""
    strategy = spec.strategy
    if strategy == FetchStrategy.TRADE_DATE_BY_TAG:
        return [{"tag": tag} for tag in cli_values(args, "kpl_tag")]
    if strategy == FetchStrategy.TRADE_DATE_BY_LIMIT_TYPE:
        return [{"limit_type": limit_type} for limit_type in cli_values(args, "ths_limit_type")]
    if strategy == FetchStrategy.TRADE_DATE_BY_MARKET:
        return [
            {"market": market, "is_new": is_new}
            for market in cli_values(args, "ths_hot_market")
            for is_new in selected_board_hot_is_new(args)
        ]
    if strategy == FetchStrategy.TRADE_DATE_BY_MARKET_HOT_TYPE:
        return [
            {"market": market, "hot_type": hot_type, "is_new": is_new}
            for market in cli_values(args, "dc_hot_market")
            for hot_type in cli_values(args, "dc_hot_type")
            for is_new in selected_board_hot_is_new(args)
        ]
    if strategy == FetchStrategy.DATE_YEAR_BY_TS_CODE:
        return [{"ts_code": ts_code} for ts_code in by_ts_code_universe(spec.api_name, args)]
    if strategy == FetchStrategy.ECO_CAL_MONTH:
        return [
            {"country": country, "currency": currency, "event": event}
            for country in cli_values(args, "eco_country")
            for currency in cli_values(args, "eco_currency")
            for event in cli_values(args, "eco_event")
        ]
    if strategy == FetchStrategy.TIME_RANGE_MONTH:
        return [{"src": source} for source in getattr(args, "major_news_src", None) or [""]]
    if strategy == FetchStrategy.NEWS_SRC_DAY:
        return [{"src": source} for source in selected_news_sources(getattr(args, "news_src", None))]
    if strategy == FetchStrategy.TRADE_DATE and spec.loop_values:
        return [{spec.loop_param: value} for value in spec.loop_values]
    return [{}]


def partitions(
    raw_dir: Path, spec: DatasetSpec, units: list[Any] | tuple[Any, ...] = (), args: argparse.Namespace | None = None
) -> list[Partition]:
    """The partitions of ``spec`` over ``units``, in the downloader's order:
    every time unit of the first dimension value, then of the next.

    A whole-history file ignores ``units``; a range dataset has its own
    :func:`range_partition`."""
    directory = Path(raw_dir) / spec.api_name
    if spec.strategy == FetchStrategy.STATIC_FULL:
        if spec.loop_values:
            return [
                Partition(directory / f"{spec.loop_param}={safe_partition_value(value)}.parquet", dims={spec.loop_param: value})
                for value in spec.loop_values
            ]
        # The board registry's one whole-table file predates the full.parquet name.
        name = f"{spec.api_name}.parquet" if spec.api_name in BOARD_TRADING_SPECS else "full.parquet"
        return [Partition(directory / name)]
    if spec.strategy not in _TIME_KEYS:
        raise ValueError(f"{spec.api_name}: strategy {spec.strategy} has no partition layout here")
    time_key = _TIME_KEYS[spec.strategy]
    out: list[Partition] = []
    for dims in dimensions(spec, args):
        base = directory.joinpath(*(f"{key}={directory_name(value)}" for key, value in dims.items()))
        # A venue loop honours each venue's own listing date.
        first = spec.loop_start_date(dims[spec.loop_param]) if spec.loop_param in dims and spec.strategy == FetchStrategy.TRADE_DATE else ""
        for unit in units:
            label = unit[2] if isinstance(unit, tuple) else unit
            if label >= first:
                out.append(Partition(base / f"{time_key}={label}.parquet", unit, dims))
    return out

def retained_range_start(dataset_dir: Path, start_date: str) -> str:
    """The first date a range dataset's canonical file must keep covering.

    A quarter_once/month_once pull rewrites ONE canonical
    ``range=<start>_latest`` file, so a rewrite from a later start than the
    file on disk holds would drop the history below it -- a backfill under the
    research-history floor included. The retained start is therefore never
    later than the start of the canonical file already there.
    """

    starts = [start_date]
    for path in dataset_dir.glob("range=*_latest.parquet"):
        token = path.name[len("range="):-len("_latest.parquet")]
        quarter = re.fullmatch(r"(\d{4})Q([1-4])", token)
        if quarter:
            starts.append(f"{quarter.group(1)}{(int(quarter.group(2)) - 1) * 3 + 1:02d}01")
        elif re.fullmatch(r"\d{6}", token):
            starts.append(f"{token}01")
    return min(starts)


def range_partition(raw_dir: Path, spec: DatasetSpec, start_date: str) -> Partition:
    """The one ``range=<start>_latest`` file of a quarter_once/month_once
    dataset pulled from ``start_date``. Its unit is ``(start date, start
    quarter or month)``, the start never later than the stored file's
    (:func:`retained_range_start`) nor earlier than the spec's."""
    directory = Path(raw_dir) / spec.api_name
    start_date = retained_range_start(directory, start_date)
    if spec.strategy == FetchStrategy.QUARTER_ONCE:
        start = yyyymmdd_to_quarter(max(start_date, spec.start_date))
    elif spec.strategy == FetchStrategy.MONTH_ONCE:
        start = yyyymmdd_to_month(max(start_date, spec.start_date))
    else:
        raise ValueError(f"{spec.api_name}: strategy {spec.strategy} has no range file")
    return Partition(directory / f"range={start}_latest.parquet", (start_date, start))


def refuse_moved_range_start(path: Path) -> None:
    """Refuse a pull that would write the range file ``path`` beside a stored
    one.

    The file's name carries its start, so a pull whose start differs from the
    file already on disk would land as a new partition: outside the versioned
    path, with every row first seen today (later views would hide the whole
    series) and the stored file's history and first-seen table left behind. A
    later start never gets here: :func:`range_partition` keeps the stored one.
    """
    stored = sorted(item.name for item in path.parent.glob("range=*_latest.parquet") if item != path)
    if stored:
        held = stored[0][len("range="):-len("_latest.parquet")]
        start = path.name[len("range="):-len("_latest.parquet")]
        raise RuntimeError(
            f"{path.parent.name}: refusing to move the start of its range file from {held} "
            f"(stored: {', '.join(stored)}) to {start}; the file would be replaced under a new "
            "name outside the versioned write path"
        )


def time_units(raw_dir: Path, spec: DatasetSpec, start_date: str, end_date: str) -> list[Any]:
    """Every time unit of ``spec`` from ``start_date`` to ``end_date``: SSE
    open dates, calendar days, month windows or year windows."""
    time_key = _TIME_KEYS.get(spec.strategy)
    if time_key == "trade_date":
        return list(load_sse_open_dates(raw_dir, start_date, end_date))
    if time_key == "date":
        return date_range_days(start_date, end_date)
    if time_key == "month":
        return month_windows(start_date, end_date)
    if time_key == "year":
        return year_windows(start_date, end_date)
    return []
