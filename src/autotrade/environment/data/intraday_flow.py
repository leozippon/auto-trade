"""Derived daily order-flow summary over the 1-minute bar lake.

One row per ``(ts_code, trade_date)``, computed from that trade date's own
minute bars and nothing else, so a strategy can use signed order flow without
mounting the minute domain (a minute-carrying view seed costs ~235 GB).

The signed-imbalance construction is the minute-close tick rule:

    r_m    = close_m - close_{m-1}, only between contiguous minutes of the same
             (ts_code, trade_date). The 09:30 bar is the opening auction and
             has no predecessor, and 11:30 -> 13:01 is the lunch break, so a
             full 241-bar session yields at most 239 signed minutes.
    ofi_1d = sum_m sign(r_m) * vol_m / sum_m vol_m

``ofi_amt_1d`` is the same quantity weighted by turnover instead of volume. A
no-trade minute repeats the previous close with ``vol = 0`` and therefore
contributes zero to both sums: paused minutes are neutral by construction, not
by a filter.

Two diagnostics travel with every row instead of being applied as filters here,
because the thresholds are research choices and the data layer must not freeze
them: ``nret`` (how many signed minutes the day actually produced) and
``zero_share`` / ``sealed_limit`` (how much of the session had no trade at all,
the sealed-limit-board signature). The reference study keeps a stock-day when
``nret >= 200`` and ``zero_share < 0.5``.

Rolling windows are deliberately NOT precomputed. Every window the research
uses (21 and 5 trading days, the up-day/down-day conditional legs) is a mean
over these daily values, cheap to form from this table, and a window baked into
the lake would both pin one study's parameters and make a partition
non-derivable from its own trade date.

``.BJ`` names are excluded: they carry after-hours 15:01-15:30 bars, which
break both the lunch-only contiguity rule and the 241-bar denominator.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from autotrade.environment.data.contracts import INTRADAY_FLOW_CONTRACT

INTRADAY_FLOW_DATASET = "intraday_flow"
MINUTE_DATASET = "stk_mins_1min_by_date"
MINUTE_COLUMNS = ("ts_code", "trade_date", "trade_time", "close", "vol", "amount")

# One non-BJ A-share session is exactly 241 one-minute bars (09:30 auction,
# 09:31-11:30, 13:01-15:00); verified on every sampled partition 2022-2026.
BARS_PER_DAY = 241
# A stock-day whose no-trade share reaches this is a sealed limit board.
SEALED_LIMIT_ZERO_SHARE = 0.5
_EPS = 1e-12

INTRADAY_FLOW_COLUMNS = (
    "ts_code",
    "trade_date",
    "ofi_1d",
    "ofi_amt_1d",
    "nret",
    "zero_share",
    "sealed_limit",
    "available_at",
    "available_at_rule",
)


def aggregate_intraday_flow(minutes: pd.DataFrame) -> pd.DataFrame:
    """Reduce raw minute bars to one PIT-stamped order-flow row per stock-day.

    ``minutes`` must carry :data:`MINUTE_COLUMNS` and may span several trade
    dates; every quantity is grouped inside one ``(ts_code, trade_date)`` so
    the result never depends on which dates share the call.
    """
    missing = [column for column in MINUTE_COLUMNS if column not in minutes.columns]
    if missing:
        raise ValueError(f"minute frame is missing columns {missing}")
    frame = minutes[~minutes["ts_code"].str.endswith(".BJ")]
    if frame.empty:
        return _empty()
    frame = frame.sort_values(["ts_code", "trade_date", "trade_time"], kind="stable")
    code = frame["ts_code"].to_numpy()
    date = frame["trade_date"].to_numpy()
    hour = frame["trade_time"].str.slice(11, 13).to_numpy()
    price = frame["close"].to_numpy(dtype="float64")
    volume = frame["vol"].to_numpy(dtype="float64")
    amount = frame["amount"].to_numpy(dtype="float64")
    previous = np.concatenate(([np.nan], price[:-1]))
    same = np.concatenate(([False], (code[1:] == code[:-1]) & (date[1:] == date[:-1])))
    lunch = np.concatenate(([False], (hour[:-1] == "11") & (hour[1:] == "13")))
    contiguous = same & ~lunch & (price > 0) & (previous > 0)
    sign = np.zeros(len(price), dtype="float64")
    marked = np.flatnonzero(contiguous)
    # sign(log(p/q)) == sign(p - q) for positive p, q.
    sign[marked] = np.sign(price[marked] - previous[marked])
    sums = (
        pd.DataFrame(
            {
                "ts_code": code,
                "trade_date": date,
                "signed_vol": sign * volume,
                "total_vol": volume,
                "signed_amt": sign * amount,
                "total_amt": amount,
                "nret": contiguous.astype("float64"),
                "nzero": (volume <= 0).astype("float64"),
            }
        )
        .groupby(["ts_code", "trade_date"], sort=False)
        .sum()
        .reset_index()
    )
    # A stock-day with no traded volume at all (whole-day halt) has no defined
    # imbalance; it also has no `daily` bar. Drop it rather than invent a zero.
    out = sums[sums["total_vol"] > 0].copy()
    if out.empty:
        return _empty()
    out["ofi_1d"] = out["signed_vol"] / out["total_vol"]
    out["ofi_amt_1d"] = out["signed_amt"] / out["total_amt"].clip(lower=_EPS)
    out["zero_share"] = out["nzero"] / BARS_PER_DAY
    out["sealed_limit"] = out["zero_share"] >= SEALED_LIMIT_ZERO_SHARE
    out["nret"] = out["nret"].astype("int32")
    out = out.sort_values(["ts_code", "trade_date"], kind="stable").reset_index(drop=True)
    return stamp_available_at(out)[list(INTRADAY_FLOW_COLUMNS)]


def stamp_available_at(frame: pd.DataFrame) -> pd.DataFrame:
    """Write the dataset's PIT stamp from :data:`INTRADAY_FLOW_CONTRACT`.

    The events domain reads a raw ``available_at`` column, so the contract is
    applied once here at build time and never restated by a reader.
    """
    contract = INTRADAY_FLOW_CONTRACT
    clock = contract.available_time
    dates = pd.to_datetime(frame[contract.partition_key].astype(str), format="%Y%m%d")
    stamped = dates + pd.Timedelta(
        days=contract.lag_days, hours=clock.hour, minutes=clock.minute, seconds=clock.second
    )
    out = frame.copy()
    out["available_at"] = stamped.dt.strftime("%Y-%m-%d %H:%M:%S+08:00")
    out["available_at_rule"] = contract.rule
    return out


def _empty() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ts_code": pd.Series(dtype="object"),
            "trade_date": pd.Series(dtype="object"),
            "ofi_1d": pd.Series(dtype="float64"),
            "ofi_amt_1d": pd.Series(dtype="float64"),
            "nret": pd.Series(dtype="int32"),
            "zero_share": pd.Series(dtype="float64"),
            "sealed_limit": pd.Series(dtype="bool"),
            "available_at": pd.Series(dtype="object"),
            "available_at_rule": pd.Series(dtype="object"),
        }
    )
