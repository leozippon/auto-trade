"""Point-in-time reads of the events domain (five-year arms only): a correctness aid, not a score.

    read(context, dataset, columns, since=None)
        one dataset's rows in this decision's as-of view, projected to the key
        columns plus `columns`, from `since` (YYYYMMDD, by visible time) on,
        with `stamp` (UTC); a row stamped after the decision fails the read

The domain is ONE wide table of seventeen datasets (about 160 columns, each
dataset filling its own): filter on `dataset` and project columns while
reading, as `read` does, or a late view's read is killed at the strategy
container's memory limit. Every row carries its own `available_at` (a string
with offset, e.g. "2021-03-05 19:00:00+08:00"); the replay releases rows into
the as-of view by that time, so a read never sees the future, but a per-day
table is stamped after the close of its day (`moneyflow`, `cyq_perf`,
`bak_daily` 19:00, `block_trade` 21:00, `top_list` / `top_inst` 20:00,
`limit_list_d` 16:00, `kpl_list` 08:30 the next morning, `margin` /
`margin_detail` 09:00 the next trading day) and a disclosure table at its
announcement day (`stk_holdertrade`, `stk_holdernumber`, `top10_*`,
`share_float_complete`). Read at a review, not at every decision: the as-of
directory gains one part per trading day.

Columns that must never be used, and other traps (refs/README.md has the rest):
`cyq_perf.his_high` and `his_low` are back-filled with prices after the row's
day (future data); `top10_holders` and `top10_floatholders` are each
incomplete for some periods and only their union is whole, and their
`end_date` carries non-quarter dates; `share_float_complete` holds duplicate
and re-dated announcements of one release, and `float_ratio` is a PERCENT of
total shares; `margin` is exchange-level (no `ts_code`); `block_trade` and
`top_list` legitimately repeat a natural key (aggregate before use);
`hm_detail` starts 2022-08.
"""

from datetime import timedelta

import pandas as pd

KEYS = ["dataset", "ts_code", "available_at"]


def read(context, dataset, columns, since=None):
    """One events dataset as this decision sees it, with `stamp` (UTC), sorted by name and stamp."""

    filters = [("dataset", "==", dataset)]
    if since is not None:
        # One calendar day of slack: a stamp written in another offset can carry the previous date.
        lower = (pd.Timestamp(since) - timedelta(days=1)).strftime("%Y-%m-%d")
        filters.append(("available_at", ">=", lower))
    rows = pd.read_parquet(
        context.asof_dir + "/events",
        columns=list(dict.fromkeys([*KEYS, *columns])),
        filters=filters,
    )
    stamp = pd.to_datetime(rows["available_at"], utc=True)
    if (stamp > pd.Timestamp(context.inference_at)).any():
        raise RuntimeError(f"the as-of events view holds a {dataset} row stamped after this decision")
    rows = rows.assign(stamp=stamp, ts_code=rows["ts_code"].astype("string"))
    if since is not None:
        local = rows["stamp"].dt.tz_convert("Asia/Shanghai").dt.strftime("%Y%m%d")
        rows = rows[local >= str(since)]
    return rows.sort_values(["ts_code", "stamp"], kind="stable").reset_index(drop=True)
