"""The label: a name's return over the book's own holding, against names of its size.

For a unit fresh at review t (`lib/clock.py`) the book would buy at the 15:00
close of t and sell about `knobs.HOLD` trading days later, so the label is the
close-to-close return on adjusted prices from t to t + HOLD, minus the
equal-weight mean of the same return over the pool in the same float-cap
quintile, the quintile known at the T-1 close of t. The pool is every
main-board and ChiNext name with a bar on t; a suspended name carries its
last close forward, at most 120 days. A name whose close on t is locked at
the upper limit has no label, since a buy at that close does not fill (the
designer's rule; a ranker would otherwise learn to chase the limit-up runs of
new listings, which the Broker refuses); it stays in its quintile's pool. This
is the designer's event-study abnormal return (logs/notes/round_20261012/
scripts/panel.py) on the book's clock. A label is used only once t + HOLD is
a visible bar, so every price it reads was visible at the fit.
"""

import numpy as np
import pandas as pd

from lib import knobs, titles

QUINTILES = 5
FFILL_DAYS = 120
# A close at or above this share of the upper limit is locked there.
LOCKED = 0.9995
COLUMNS = ["ts_code", "trade_date", "close", "up_limit", "adj_factor", "circ_mv", "is_suspended"]


def panel(context, days):
    """Wide arrays over the visible trading days `days[:-1]` (the last entry is today):
    codes, adjusted close (carried over suspensions), bar present, locked at the upper
    limit, float cap."""

    bars = days[:-1]
    rows = pd.read_parquet(context.asof_dir + "/daily", columns=COLUMNS, filters=[("trade_date", ">=", bars[0])])
    rows = rows.assign(trade_date=rows["trade_date"].astype(str), ts_code=rows["ts_code"].astype(str))
    rows = rows[rows["ts_code"].str.match(titles.STOCK) & rows["trade_date"].isin(set(bars))]
    rows = rows.assign(
        adj=rows["close"] * rows["adj_factor"],
        bar=(rows["close"] > 0) & ~rows["is_suspended"].eq(True),
        locked=rows["close"] >= rows["up_limit"] * LOCKED,
    )
    codes = np.array(sorted(rows["ts_code"].unique()), dtype=object)

    def wide(column):
        frame = rows.pivot(index="trade_date", columns="ts_code", values=column)
        return frame.reindex(index=bars, columns=codes)

    return {
        "codes": codes,
        "adj": wide("adj").ffill(limit=FFILL_DAYS).to_numpy(dtype=float),
        "bar": wide("bar").eq(True).to_numpy(),
        "locked": wide("locked").eq(True).to_numpy(),
        "mv": wide("circ_mv").ffill(limit=FFILL_DAYS).to_numpy(dtype=float),
    }


def size_rank(mv):
    """Float-cap percentile rank (0..1] across the names that have one; NaN elsewhere."""

    return pd.Series(mv).rank(pct=True).to_numpy()


def abnormal(data, t):
    """Size-matched HOLD-day return of every code bought at the close of bar t (NaN where
    the code had no bar on t or closed locked at the upper limit); t + HOLD must be a
    visible bar."""

    adj, bar = data["adj"], data["bar"]
    forward = adj[t + knobs.HOLD] / adj[t] - 1.0
    forward[~bar[t] | ~np.isfinite(forward)] = np.nan
    rank = size_rank(data["mv"][t - 1])
    quintile = np.where(np.isfinite(rank), np.minimum((rank * QUINTILES).astype(int), QUINTILES - 1), -1)
    out = np.full(len(forward), np.nan)
    for q in range(QUINTILES):
        pool = (quintile == q) & np.isfinite(forward)
        if pool.any():
            out[pool] = forward[pool] - forward[pool].mean()
    out[data["locked"][t]] = np.nan
    return out
