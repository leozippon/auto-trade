"""m1: how close each CSI 500 member is to the CSI 300 entry rule, point in time.

The CSI 300 methodology (reviews in late May / November on the past year's data,
effective the trading day after the second Friday of June / December) selects,
from non-ST A shares listed long enough, the names in the top half by mean daily
traded amount -- incumbents in the top 60 % -- and ranks them by mean daily total
market value; the top 300 are the index, with a buffer of 240 for newcomers and
360 for incumbents. This module reproduces the ranking on the visible past year
at every decision, so "the next names the rule would admit" is a number a
decision can compute without any announcement. `refs/families.md` registers what is
and is not reproduced (the loss-making rule, the top-30 listing exception and the
10 % cap are not).

The score of a pool name is minus its entry rank; a name outside the sample
space or the liquidity screen has no score -- it cannot be admitted, so the book
neither buys nor keeps it. The window reads every A share's `amount` and
`total_mv` over the past `knobs.WINDOW_DAYS` calendar days (rows with a positive
amount; units do not matter to a rank), ending at T-1 because the as-of view
holds nothing later.

`shuffle` is the control: the same values permuted among the scored names with
a generator seeded by the decision day, so a cold worker and a warm one emit the
same orders.
"""

from datetime import timedelta

import numpy as np
import pandas as pd

from lib import index, knobs

NAME = "m1"
COLUMNS = ["ts_code", "trade_date", "amount", "total_mv"]
UNIVERSE_COLUMNS = ["ts_code", "name", "list_date"]
GROWTH_BOARDS = ("688", "689", "300", "301")


def score(context, codes):
    """(Series over the sorted `codes`: -entry rank or NaN, order metadata)."""

    start = (context.inference_at - timedelta(days=knobs.WINDOW_DAYS)).strftime("%Y%m%d")
    frame = pd.read_parquet(context.asof_dir + "/daily", columns=COLUMNS, filters=[("trade_date", ">=", start)])
    missing = [name for name in COLUMNS if name not in frame.columns]
    if missing:
        raise RuntimeError(f"daily is missing columns {missing}")
    frame = frame[frame["amount"] > 0]
    if frame.empty:
        raise RuntimeError("the entry-rule window holds no traded bar")
    last = pd.Timestamp(str(frame["trade_date"].astype(str).max()))
    stats = frame.groupby("ts_code").agg(avg_amount=("amount", "mean"), avg_mv=("total_mv", "mean"))
    universe = pd.read_parquet(context.asof_dir + "/universe", columns=UNIVERSE_COLUMNS)
    info = universe.drop_duplicates("ts_code").set_index("ts_code").reindex(stats.index)
    code = stats.index.to_series()
    listed = pd.to_datetime(info["list_date"].astype(str), format="%Y%m%d", errors="coerce")
    age = (last - listed).dt.days
    need = np.where(code.str.startswith(GROWTH_BOARDS), knobs.LISTED_DAYS_GROWTH, knobs.LISTED_DAYS)
    space = ~code.str.endswith(".BJ") & ~info["name"].fillna("").astype(str).str.contains("ST|退") & (age > need)
    stats = stats[space.to_numpy()]
    incumbents, _ = index.latest(context, knobs.TARGET)
    liquidity = stats["avg_amount"].rank(ascending=False, method="first") / len(stats)
    member = pd.Series(stats.index.isin(list(incumbents)), index=stats.index)
    eligible = (liquidity <= knobs.LIQUIDITY_CUT) | (member & (liquidity <= knobs.INCUMBENT_CUT))
    entry = stats.loc[eligible, "avg_mv"].rank(ascending=False, method="first")
    values = -entry.reindex(sorted(codes))
    meta = {"sample_space": int(len(stats)), "eligible": int(eligible.sum()),
            "window_last_bar": last.strftime("%Y%m%d")}
    return values, meta


def shuffle(values, decision_at):
    """The scored values permuted among the scored names, seeded by the decision day."""

    scored = values.dropna().sort_index()
    day = int(pd.Timestamp(decision_at).strftime("%Y%m%d"))
    order = np.random.default_rng(day).permutation(len(scored))
    return pd.Series(scored.to_numpy()[order], index=scored.index)


def describe(values, book):
    """Order metadata about the target book under this score."""

    ranks = (-values.reindex(book)).dropna()
    return {"book_entry_rank": [int(ranks.min()), int(ranks.max())] if len(ranks) else None}
