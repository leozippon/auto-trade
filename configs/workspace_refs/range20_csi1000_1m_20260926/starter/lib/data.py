"""The pool a decision ranks: the constituents this account can hold today.

One read per decision, no cache: the newest visible section of `knobs.INDEX`
(`lib/index.py`), then the daily window of those members plus whatever the
account still holds, then the as-of universe for names and industries.

Visibility. `daily` has no `available_at` column -- the as-of view holds exactly
the rows visible at the decision, so the newest bar an 08:30 decision reads is
the previous trading day's (T-1). `universe` is the replay slot's own table;
its `l1_name` is the SW level-1 industry the host's `top_industry_weight` is
computed from (SW2014 before 2021-12-13, SW2021 after; empty falls in 未分类).

The pool is the section's members minus what this account cannot or should not
trade: STAR codes (688/689: a 200-share minimum lot and a 20 % band), `.BJ`
codes, names carrying ST or 退, names without a T-1 bar (suspended), and names
with fewer than `knobs.RANGE_DAYS` usable bars of their own in the window (the
score is undefined, not imputed). Affordability is not a pool rule: it only
stops a buy (`lib/trade.py`).

The range of a bar is (high - low) / pre_close. `pre_close` is the exchange's
previous close adjusted for that day's corporate action, so a split or a
dividend never shows up as a range. A bar whose range is not finite (no
pre_close) is dropped, not counted as a quiet day.
"""

from datetime import timedelta

import numpy as np
import pandas as pd

from lib import index, knobs

DAILY_LOOKBACK_DAYS = 200   # calendar days: 60 own bars for r2 plus holidays and short suspensions
DAILY_COLUMNS = ["ts_code", "trade_date", "high", "low", "pre_close", "close"]
UNIVERSE_COLUMNS = ["ts_code", "name", "l1_name"]


def read(context, held):
    """(pool frame, last close of every name read, industry of every name read, section date, section weights).

    The pool frame has one row per tradable, scored constituent: ts_code,
    range (mean over its own last RANGE_DAYS bars), close (T-1, raw) and
    industry. Prices cover the held names too, so a holding that left the
    section or stopped trading is still valued at its own last close.
    """

    members, section = index.newest_section(context)
    codes = sorted(set(members["ts_code"]) | set(held))
    start = (context.inference_at - timedelta(days=DAILY_LOOKBACK_DAYS)).strftime("%Y%m%d")
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
    latest = str(daily["trade_date"].max())
    last = daily[daily["close"] > 0].groupby("ts_code", sort=False).tail(1).set_index("ts_code")
    prices = last["close"].astype(float)

    with np.errstate(divide="ignore", invalid="ignore"):
        amp = (daily["high"] - daily["low"]) / daily["pre_close"].where(daily["pre_close"] > 0)
    bars = daily.assign(amp=amp)[np.isfinite(amp)]
    tail = bars.groupby("ts_code", sort=False).tail(knobs.RANGE_DAYS)
    stats = tail.groupby("ts_code")["amp"].agg(["mean", "size"])

    universe = pd.read_parquet(context.asof_dir + "/universe", columns=UNIVERSE_COLUMNS)
    info = universe.drop_duplicates("ts_code").set_index("ts_code").reindex(pd.Index(codes))
    industry = info["l1_name"].fillna("未分类").astype(str)
    names = info["name"].fillna("").astype(str)

    frame = members[["ts_code"]].copy()
    code = frame["ts_code"].astype(str)
    frame["range"] = stats["mean"].reindex(code).to_numpy()
    frame["bars"] = stats["size"].reindex(code).fillna(0).to_numpy()
    frame["close"] = prices.reindex(code).to_numpy()
    frame["industry"] = industry.reindex(code).to_numpy()
    on_t1 = last["trade_date"].reindex(code).to_numpy() == latest
    keep = (
        on_t1
        & (frame["bars"].to_numpy() >= knobs.RANGE_DAYS)
        & np.isfinite(frame["range"].to_numpy())
        & ~code.str.startswith(("688", "689")).to_numpy()
        & ~code.str.endswith(".BJ").to_numpy()
        & ~names.reindex(code).str.contains("ST|退").to_numpy()
    )
    weights = members.set_index("ts_code")["weight"]
    return frame[keep].reset_index(drop=True), prices, industry, section, weights
