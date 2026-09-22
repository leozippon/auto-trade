"""One read of the events domain, PIT-correct, shared by every block builder.

The events domain is a column union of many datasets in one parts directory,
exactly like `macro`. Two consequences the block builders must not re-derive:

    filter by `dataset` FIRST, then talk about fields and units. Two datasets
    in this domain can carry the same column name with different meanings and
    different scales, so a read that does not pin `dataset` in BOTH the
    push-down filter and the column projection is reading a mixture.

    judge visibility on the row-level `available_at` only, never on
    `trade_date`. Every dataset in this domain stamps its own rule -- 16:00 of
    the trade date, 20:00 of the trade date, 08:30 of the next day -- and the
    as-of view already holds only rows visible at the decision, so this filter
    is a belt-and-braces check that also keeps an offline census honest.

Pushing `ts_code` down matters more here than anywhere else: the events
tables are whole-market, and the constituent union is an order of magnitude
smaller. A read without it pulls millions of rows into a 16 GiB
container for nothing.

A dataset that is selected but has no visible row in the window is a hard
failure, not an empty frame. A block silently built from nothing would score
every name identically and the replay would report success.
"""

from datetime import timedelta

import pandas as pd


def read(context, dataset, columns, calendar_days, codes=None):
    """Visible rows of one events dataset over the window, with `ts_code` pushed down."""

    start = (context.inference_at - timedelta(days=calendar_days)).strftime("%Y%m%d")
    wanted = ["dataset", "available_at", "ts_code", "trade_date", *columns]
    filters = [("dataset", "=", dataset), ("trade_date", ">=", start)]
    if codes is not None:
        filters.append(("ts_code", "in", list(codes)))
    frame = pd.read_parquet(context.asof_dir + "/events", columns=wanted, filters=filters)
    missing = [name for name in wanted if name not in frame.columns]
    if missing:
        raise RuntimeError(f"events dataset {dataset} is missing columns {missing}")
    stamp = pd.to_datetime(frame["available_at"], utc=True)
    frame = frame[stamp <= pd.Timestamp(context.inference_at).tz_convert("UTC")]
    if frame.empty:
        raise RuntimeError(
            f"no visible {dataset} row in the events domain over the last {calendar_days} days; "
            "the dataset is not selected into this experiment's snapshot, or the window is wrong"
        )
    frame = frame.assign(trade_date=frame["trade_date"].astype(str))
    return frame.reset_index(drop=True)


def dense(frame, column, dates, codes, aggregate="last"):
    """(len(codes), len(dates)) matrix of one column, NaN where the dataset has no row.

    `aggregate` picks what happens when a (name, date) pair carries more than
    one row -- `top_inst` gives up to five seats a side, `kpl_list` up to five
    tags -- so the caller states the reduction instead of relying on whichever
    row the reader happened to return last.
    """

    import numpy as np

    position = {code: index for index, code in enumerate(codes)}
    slot = {date: index for index, date in enumerate(dates)}
    rows = frame[frame["ts_code"].isin(position) & frame["trade_date"].isin(slot)]
    out = np.full((len(codes), len(dates)), np.nan)
    if rows.empty:
        return out
    values = pd.to_numeric(rows[column], errors="coerce")
    grouped = pd.DataFrame({"s": rows["ts_code"].map(position).to_numpy(),
                            "t": rows["trade_date"].map(slot).to_numpy(),
                            "v": values.to_numpy()}).dropna(subset=["v"])
    if grouped.empty:
        return out
    reduced = getattr(grouped.groupby(["s", "t"], sort=False)["v"], aggregate)()
    index = reduced.index.to_frame(index=False)
    out[index["s"].to_numpy(dtype=int), index["t"].to_numpy(dtype=int)] = reduced.to_numpy()
    return out
