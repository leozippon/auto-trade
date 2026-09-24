"""The pool a decision ranks: the constituents this account can hold today, with every score input.

One read per decision, no cache: the newest visible section of `knobs.INDEX`
(`lib/index.py`), then one window of daily rows for those members plus
whatever the account still holds, then the as-of universe for names and
industries.

Visibility. `daily` has no `available_at` column -- the as-of view holds exactly
the rows visible at the decision, so the newest row an 08:30 decision reads is
the previous trading day's (T-1). `daily` carries the vendor's `daily_basic`
columns (stamped 18:00 of their own session): `pe_ttm` / `pe` are multiples
(null for a loss), `dv_ttm` / `dv_ratio` and `turnover_rate` are DECIMALS
(0.05 = 5 %), `total_mv` is CNY. `universe` is the replay slot's own table; its
`l1_name` is the SW level-1 industry (SW2014 before 2021-12-13, SW2021 after;
empty falls in 未分类).

The pool is the section's members minus what this account cannot or should not
trade: STAR codes (688/689), `.BJ` codes, names carrying ST or 退, and names
without a T-1 bar (suspended). Whether a member can be scored is the score's
rule (`lib/score.py`); affordability is not a pool rule, it only stops a buy.

The inputs, all as of the T-1 row or ending on it:
    has_basic  the T-1 row carries daily_basic (total_mv > 0)
    pe         `knobs.EP_COLUMN` of the T-1 row
    dv         the first positive value among `knobs.DV_COLUMNS` of the T-1 row
    turn_ratio mean turnover_rate over the name's valid bars among the last
               TURN_SHORT market sessions / the same over the last TURN_LONG;
               a bar is valid with turnover_rate > 0, a full-day suspension
               has no bar; NaN below ceil(VALID_SHARE x sessions) valid bars
    range      mean (high - low) / pre_close over the name's own last
               RANGE_DAYS bars (the range20 diagnostic); NaN below RANGE_DAYS
The market-session axis is every date on which a name read has a bar.
"""

import math
from datetime import timedelta

import numpy as np
import pandas as pd

from lib import index, knobs

VALID_SHARE = 0.75
# Calendar days read: TURN_LONG sessions plus weekends and the longest holidays.
LOOKBACK_DAYS = int(max(knobs.TURN_LONG, knobs.RANGE_DAYS) * 7 / 5) + 60
DAILY_COLUMNS = ["ts_code", "trade_date", "high", "low", "pre_close", "close", "turnover_rate", "total_mv",
                 knobs.EP_COLUMN, *knobs.DV_COLUMNS]
UNIVERSE_COLUMNS = ["ts_code", "name", "l1_name"]


def read(context, held):
    """(pool frame, last close of every name read, industry of every name read, section date, section weights).

    The pool frame has one row per tradable constituent: ts_code, close (T-1,
    raw), industry and the inputs above. Prices cover the held names too, so a
    holding that left the section or stopped trading is still valued at its
    own last close.
    """

    members, section = index.newest_section(context)
    codes = sorted(set(members["ts_code"]) | set(held))
    start = (context.inference_at - timedelta(days=LOOKBACK_DAYS)).strftime("%Y%m%d")
    daily = pd.read_parquet(
        context.asof_dir + "/daily",
        columns=DAILY_COLUMNS,
        filters=[("trade_date", ">=", start), ("ts_code", "in", codes)],
    )
    missing = [name for name in DAILY_COLUMNS if name not in daily.columns]
    if missing:
        raise RuntimeError(f"daily is missing columns {missing}")
    if daily.empty:
        raise RuntimeError(f"the daily window holds no bar of the {section} {knobs.INDEX} section")
    daily = daily.assign(trade_date=daily["trade_date"].astype(str)).sort_values(["ts_code", "trade_date"])
    sessions = np.array(sorted(daily["trade_date"].unique()))
    latest = str(sessions[-1])
    last = daily[daily["close"] > 0].groupby("ts_code", sort=False).tail(1).set_index("ts_code")
    prices = last["close"].astype(float)

    turn = daily[daily["turnover_rate"] > 0]
    ratio = _window_mean(turn, sessions, knobs.TURN_SHORT) / _window_mean(turn, sessions, knobs.TURN_LONG)
    with np.errstate(divide="ignore", invalid="ignore"):
        amp = (daily["high"] - daily["low"]) / daily["pre_close"].where(daily["pre_close"] > 0)
    bars = daily.assign(amp=amp)[np.isfinite(amp)]
    tail = bars.groupby("ts_code", sort=False).tail(knobs.RANGE_DAYS)
    stats = tail.groupby("ts_code")["amp"].agg(["mean", "size"])
    rng = stats["mean"].where(stats["size"] >= knobs.RANGE_DAYS)

    universe = pd.read_parquet(context.asof_dir + "/universe", columns=UNIVERSE_COLUMNS)
    info = universe.drop_duplicates("ts_code").set_index("ts_code").reindex(pd.Index(codes))
    industry = info["l1_name"].fillna("未分类").astype(str)
    names = info["name"].fillna("").astype(str)

    frame = members[["ts_code"]].copy()
    code = frame["ts_code"].astype(str)
    row = last.reindex(code)
    dv = np.full(len(code), np.nan)
    for column in knobs.DV_COLUMNS:
        value = row[column].to_numpy(dtype=np.float64)
        dv = np.where(np.isnan(dv) & (value > 0), value, dv)
    frame["close"] = prices.reindex(code).to_numpy()
    frame["industry"] = industry.reindex(code).to_numpy()
    frame["has_basic"] = (row["total_mv"] > 0).to_numpy()
    frame["pe"] = row[knobs.EP_COLUMN].to_numpy(dtype=np.float64)
    frame["dv"] = dv
    frame["turn_ratio"] = ratio.reindex(code).to_numpy()
    frame["range"] = rng.reindex(code).to_numpy()
    on_t1 = row["trade_date"].to_numpy() == latest
    keep = (
        on_t1
        & ~code.str.startswith(("688", "689")).to_numpy()
        & ~code.str.endswith(".BJ").to_numpy()
        & ~names.reindex(code).str.contains("ST|退").to_numpy()
    )
    weights = members.set_index("ts_code")["weight"]
    return frame[keep].reset_index(drop=True), prices, industry, section, weights


def _window_mean(turn, sessions, count):
    """Per name: mean turnover_rate over its valid bars among the last `count` sessions (NaN below the minimum)."""

    window = turn[turn["trade_date"] >= str(sessions[-count:][0])]
    stats = window.groupby("ts_code")["turnover_rate"].agg(["mean", "size"])
    return stats["mean"].where(stats["size"] >= math.ceil(VALID_SHARE * count))
