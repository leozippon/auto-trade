"""Derived coarse intraday bars over the 1-minute bar lake.

One row per ``(ts_code, trade_date, bar)``, aggregated from that trade date's
own minute bars and nothing else, built and stamped exactly like
:mod:`intraday_flow` (same source, same build, same close contract), so a
strategy can feed intraday bars to a model without mounting the minute domain.
One dataset per bar length N in minutes (``contracts.INTRADAY_BARS_DATASETS``).

Grid. A session is two 120-minute halves and N divides 120, so every stock-day
has 240/N bars. ``trade_time`` is the bar END: the k-th morning bar ends at
09:30 + kN, the k-th afternoon bar at 13:00 + kN (N = 30: 10:00, 10:30, 11:00,
11:30, 13:30, 14:00, 14:30, 15:00). A bar holds the minute bars labelled in
(end - N, end], so none spans the lunch break. The minute labelled 09:30 -- the
opening call auction, for Shenzhen names also the first seconds of continuous
trading -- belongs to the first bar, and the 15:00 minute, which carries the
closing call auction, to the last. Minutes off the session grid (09:30,
09:31-11:30, 13:01-15:00) are ignored.

Values. ``open`` is the first source minute's open, ``close`` the last one's
close, ``high`` and ``low`` the extremes, ``vol`` (shares) and ``amount`` (CNY)
the sums; prices are unadjusted, as in the minute bars and ``daily``. ``n_min``
counts the bar's source minutes: N, and N + 1 for the first bar of the day. A
no-trade minute is in the source with the previous close and ``vol = 0``, so a
paused stretch gives flat bars with zero volume. A grid bar with no source
minute at all is still written -- null prices, zero ``vol`` and ``amount``,
``n_min = 0`` -- and a bar missing some of its minutes aggregates those
present: every stock-day keeps the full grid and no value is invented.

Stock-days and order. As in :mod:`intraday_flow`, ``.BJ`` names are excluded
and a stock-day with no traded volume is not written. Rows are ordered by
``ts_code``, ``trade_date``, ``trade_time``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from autotrade.environment.data.contracts import (
    INTRADAY_BARS_CONTRACTS,
    INTRADAY_BARS_DATASETS,
)
from autotrade.environment.data.intraday_flow import MINUTE_COLUMNS

# The flow columns plus the rest of the bar.
INTRADAY_BARS_MINUTE_COLUMNS = (*MINUTE_COLUMNS, "open", "high", "low")

_PRICE_COLUMNS = ("open", "high", "low", "close")
INTRADAY_BARS_COLUMNS = (
    "ts_code",
    "trade_date",
    "trade_time",
    *_PRICE_COLUMNS,
    "vol",
    "amount",
    "n_min",
    "available_at",
    "available_at_rule",
)

_HALF_SESSION_MINUTES = 120
# Minute of day each half's first bar starts after: 09:30 and 13:00.
_HALF_STARTS = (9 * 60 + 30, 13 * 60)


def _label(minute: int) -> str:
    return f"{minute // 60:02d}:{minute % 60:02d}"


def _grid(bar_minutes: int) -> tuple[dict[str, int], list[str]]:
    """The bar each minute label (``HH:MM``) falls in, and each bar's end label."""
    per_half, remainder = divmod(_HALF_SESSION_MINUTES, bar_minutes)
    if remainder:
        raise ValueError(f"a bar of {bar_minutes} minutes does not divide the {_HALF_SESSION_MINUTES}-minute half session")
    bar_of = {_label(_HALF_STARTS[0]): 0}  # the opening auction minute
    ends: list[str] = []
    for half, start in enumerate(_HALF_STARTS):
        for offset in range(1, _HALF_SESSION_MINUTES + 1):
            bar_of[_label(start + offset)] = half * per_half + (offset - 1) // bar_minutes
        ends += [_label(start + (bar + 1) * bar_minutes) for bar in range(per_half)]
    return bar_of, ends


def aggregate_intraday_bars(minutes: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """Aggregate raw minute bars to the PIT-stamped bars of ``dataset``.

    ``minutes`` must carry :data:`INTRADAY_BARS_MINUTE_COLUMNS` and may span
    several trade dates; every bar is formed inside one ``(ts_code,
    trade_date)``, so the result never depends on call shape.
    """
    bar_of, ends = _grid(INTRADAY_BARS_DATASETS[dataset])
    per_day = len(ends)
    missing = [column for column in INTRADAY_BARS_MINUTE_COLUMNS if column not in minutes.columns]
    if missing:
        raise ValueError(f"minute frame is missing columns {missing}")
    frame = minutes[~minutes["ts_code"].str.endswith(".BJ")]
    frame = frame.sort_values(["ts_code", "trade_date", "trade_time"], kind="stable")
    # Read the clock once per distinct timestamp rather than once per row.
    stamp, stamps = pd.factorize(frame["trade_time"])
    bar = np.array([bar_of.get(label, -1) for label in pd.Index(stamps).str.slice(11, 16)], dtype="int64")[stamp]
    on_grid = bar >= 0
    if not on_grid.any():
        return _empty()
    frame, bar = frame[on_grid], bar[on_grid]
    code = frame["ts_code"].to_numpy()
    date = frame["trade_date"].to_numpy()
    prices = {column: frame[column].to_numpy(dtype="float64") for column in _PRICE_COLUMNS}
    volume = frame["vol"].to_numpy(dtype="float64")
    amount = frame["amount"].to_numpy(dtype="float64")
    if not all(np.isfinite(values).all() and (values > 0).all() for values in prices.values()):
        raise ValueError("minute frame carries a non-positive or missing session price")
    if not (np.isfinite(volume).all() and np.isfinite(amount).all() and (volume >= 0).all() and (amount >= 0).all()):
        raise ValueError("minute frame carries a negative or missing volume or amount")

    first = np.concatenate(([True], (code[1:] != code[:-1]) | (date[1:] != date[:-1])))
    day = np.cumsum(first) - 1
    n_days = int(day[-1]) + 1
    # A stock-day with no traded volume (whole-day halt) has no `daily` bar
    # either. Drop it rather than publish a grid of repeated previous closes.
    keep = np.repeat(np.bincount(day, weights=volume, minlength=n_days) > 0, per_day)
    if not keep.any():
        return _empty()

    # Rows are time-ordered inside a stock-day, so each bar is one run of rows.
    cell = day * per_day + bar
    starts = np.flatnonzero(np.concatenate(([True], cell[1:] != cell[:-1])))
    lasts = np.concatenate((starts[1:], [len(cell)])) - 1
    present = cell[starts]

    def on_full_grid(values: np.ndarray, absent: float) -> np.ndarray:
        column = np.full(n_days * per_day, absent, dtype="float64")
        column[present] = values
        return column

    session = pd.Series(frame["trade_time"].to_numpy()[first]).str.slice(0, 10).to_numpy()
    out = pd.DataFrame(
        {
            "ts_code": np.repeat(code[first], per_day),
            "trade_date": np.repeat(date[first], per_day),
            "trade_time": np.repeat(session, per_day) + " " + np.tile(np.array(ends, dtype=object), n_days) + ":00",
            "open": on_full_grid(prices["open"][starts], np.nan),
            "high": on_full_grid(np.maximum.reduceat(prices["high"], starts), np.nan),
            "low": on_full_grid(np.minimum.reduceat(prices["low"], starts), np.nan),
            "close": on_full_grid(prices["close"][lasts], np.nan),
            "vol": on_full_grid(np.add.reduceat(volume, starts), 0.0),
            "amount": on_full_grid(np.add.reduceat(amount, starts), 0.0),
            "n_min": on_full_grid(lasts - starts + 1, 0.0).astype("int32"),
        }
    )[keep].reset_index(drop=True)
    return INTRADAY_BARS_CONTRACTS[dataset].stamp(out)[list(INTRADAY_BARS_COLUMNS)]


def _empty() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ts_code": pd.Series(dtype="object"),
            "trade_date": pd.Series(dtype="object"),
            "trade_time": pd.Series(dtype="object"),
            **{column: pd.Series(dtype="float64") for column in (*_PRICE_COLUMNS, "vol", "amount")},
            "n_min": pd.Series(dtype="int32"),
            "available_at": pd.Series(dtype="object"),
            "available_at_rule": pd.Series(dtype="object"),
        }
    )
