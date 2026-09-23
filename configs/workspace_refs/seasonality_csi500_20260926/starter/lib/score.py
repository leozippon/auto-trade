"""s1: same-calendar-month return seasonality (Heston and Sadka), point in time.

At a decision in calendar month m, a pool name's score is the mean, over the
annual lags k = `knobs.FIRST_LAG` .. `knobs.LAGS`, of its return in month
m - 12k minus the equal-weight mean of that month's return over the pool names
that have one. A month's return is the adjusted close (`close * adj_factor`) of
the name's last bar in the month over that of its last bar in the month before,
minus one; it is missing when either month has no bar. A name with fewer than
`knobs.MIN_LAGS` lags has no score -- the book neither buys nor keeps it.

Only months strictly before m are read, so nothing near the decision enters;
the read is bounded to the oldest lag month's predecessor onwards and to the
pool's names. Each replay year's input window is 60 months, so four lags
(49 months back) always fit; a missing lag means the name was not listed, or
did not trade for a whole month, not that the window was short.

`shuffle` is the control: the same values permuted among the scored names with
a generator seeded by the decision day, so a cold worker and a warm one emit the
same orders.
"""

import numpy as np
import pandas as pd

from lib import knobs

NAME = "s1"
COLUMNS = ["ts_code", "trade_date", "close", "adj_factor"]


def lag_months(decision_at):
    month = pd.Period(pd.Timestamp(decision_at).strftime("%Y-%m"), freq="M")
    return [month - 12 * k for k in range(knobs.FIRST_LAG, knobs.LAGS + 1)]


def score(context, codes):
    """(Series over the sorted `codes`: mean same-month excess or NaN, order metadata)."""

    codes = sorted(codes)
    lags = lag_months(context.inference_at)
    start = (min(lags) - 1).start_time.strftime("%Y%m%d")
    end = (max(lags) + 1).start_time.strftime("%Y%m%d")
    frame = pd.read_parquet(context.asof_dir + "/daily", columns=COLUMNS,
                            filters=[("trade_date", ">=", start), ("trade_date", "<", end), ("ts_code", "in", codes)])
    missing = [name for name in COLUMNS if name not in frame.columns]
    if missing:
        raise RuntimeError(f"daily is missing columns {missing}")
    frame = frame[(frame["close"] > 0) & frame["adj_factor"].notna()]
    if frame.empty:
        raise RuntimeError("the lag window holds no bar for the pool")
    frame = frame.assign(adj=frame["close"] * frame["adj_factor"], month=frame["trade_date"].astype(str).str[:6])
    last = frame.sort_values("trade_date").groupby(["ts_code", "month"])["adj"].last().unstack("month")
    columns = {}
    for lag in lags:
        now, before = lag.strftime("%Y%m"), (lag - 1).strftime("%Y%m")
        if now in last.columns and before in last.columns:
            columns[now] = last[now] / last[before] - 1.0
        else:
            columns[now] = pd.Series(np.nan, index=last.index)
    block = pd.DataFrame(columns).reindex(codes)
    excess = block - block.mean(axis=0)
    count = excess.notna().sum(axis=1)
    values = excess.mean(axis=1).where(count >= knobs.MIN_LAGS)
    meta = {"lag_months": [lag.strftime("%Y%m") for lag in lags],
            "all_lags_share": round(float((count == len(lags)).mean()), 4),
            "scorable_share": round(float(values.notna().mean()), 4)}
    return values, meta


def shuffle(values, decision_at):
    """The scored values permuted among the scored names, seeded by the decision day."""

    scored = values.dropna().sort_index()
    day = int(pd.Timestamp(decision_at).strftime("%Y%m%d"))
    order = np.random.default_rng(day).permutation(len(scored))
    return pd.Series(scored.to_numpy()[order], index=scored.index)


def describe(values, book):
    """Order metadata about the target book under this score."""

    held = values.reindex(book).dropna()
    return {"book_score": [round(float(held.min()), 5), round(float(held.max()), 5)] if len(held) else None}
