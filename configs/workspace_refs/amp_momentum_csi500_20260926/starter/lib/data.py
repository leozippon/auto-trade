"""The one panel builder: the pool, its daily bars and the book's prices.

`panel(context, held)` reads the newest visible section (lib/index.py) and one
window of daily bars for its members plus every name the account holds, and
returns (names x sessions) matrices. The session axis is every date on which
one of those names has a bar, which is the market calendar; the newest column
is the newest visible session, T-1 at an 08:30 decision.

Visibility. `daily` has no `available_at` column -- the as-of view holds exactly
the rows visible at the decision. `pct_chg` is a DECIMAL in the snapshot
(0.05 = 5 %), measured against the vendor's `pre_close`, which is already
adjusted for the day's corporate action, so no adjustment factor is needed:
amplitude and return are both same-day ratios. The book is sized at the RAW
T-1 close.
"""

from datetime import timedelta

import numpy as np
import pandas as pd

from lib import index, knobs

DAILY_COLUMNS = ["ts_code", "trade_date", "high", "low", "close", "pre_close", "pct_chg"]
UNIVERSE_COLUMNS = ["ts_code", "name", "l1_name"]
# Calendar days read: WINDOW sessions plus weekends and the longest holidays.
LOOKBACK_DAYS = int(knobs.WINDOW * 7 / 5) + 60


def panel(context, held):
    members, section = index.latest(context)
    codes = sorted(set(members["ts_code"]) | set(held))
    start = (context.inference_at - timedelta(days=LOOKBACK_DAYS)).strftime("%Y%m%d")
    frame = pd.read_parquet(
        context.asof_dir + "/daily",
        columns=DAILY_COLUMNS,
        filters=[("trade_date", ">=", start), ("ts_code", "in", codes)],
    )
    missing = [name for name in DAILY_COLUMNS if name not in frame.columns]
    if missing:
        raise RuntimeError(f"daily is missing columns {missing}")
    frame = frame[frame["close"] > 0]
    frame = frame.assign(trade_date=frame["trade_date"].astype(str))
    dates = sorted(frame["trade_date"].unique())
    if len(dates) < knobs.WINDOW:
        raise RuntimeError(f"only {len(dates)} sessions in the daily window; the score needs {knobs.WINDOW}")
    names = sorted(frame["ts_code"].unique())
    si = np.searchsorted(np.array(names), frame["ts_code"].to_numpy())
    ti = np.searchsorted(np.array(dates), frame["trade_date"].to_numpy())

    def dense(column):
        out = np.full((len(names), len(dates)), np.nan)
        out[si, ti] = frame[column].to_numpy(dtype=np.float64)
        return out

    universe = pd.read_parquet(context.asof_dir + "/universe", columns=UNIVERSE_COLUMNS)
    info = universe.drop_duplicates("ts_code").set_index("ts_code").reindex(pd.Index(names))
    weight = members.set_index("ts_code")["weight"]
    return {
        "dates": dates,
        "symbols": np.array(names, dtype=object),
        "H": dense("high"), "L": dense("low"), "PC": dense("pre_close"), "R": dense("pct_chg"),
        "close": dense("close"),
        "member": np.isin(np.array(names, dtype=object), members["ts_code"].to_numpy()),
        "index_weight": weight.reindex(pd.Index(names)).fillna(0.0).to_numpy(),
        "section": section,
        "industry": info["l1_name"].fillna("未分类").astype(str).to_numpy(),
        "name": info["name"].fillna("").astype(str).to_numpy(),
    }


def tradable(panel):
    """(names,) bool: in the newest section, with a T-1 bar, not STAR / BSE / ST."""

    codes = pd.Series([str(code) for code in panel["symbols"]])
    names = pd.Series(panel["name"])
    return (
        panel["member"]
        & np.isfinite(panel["close"][:, -1])
        & ~codes.str.startswith(("688", "689")).to_numpy()
        & ~codes.str.endswith(".BJ").to_numpy()
        & ~names.str.contains("ST|退").to_numpy()
    )
