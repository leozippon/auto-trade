"""Northbound (Stock Connect) holdings from the two top-10 holder tables, point in time.

What is read. `top10_holders` (top ten of all shareholders) and
`top10_floatholders` (top ten of float holders) in the EVENTS domain, one row
per (`ts_code`, `ann_date`, `end_date`, `holder_name`). Neither table alone is
complete: the vendor returned only part of several report periods of each, so a
read of one table silently turns half the firms of those quarters into "no
northbound holder". Only their UNION is used, and a snapshot that selects only
one of them is refused here rather than read as half a signal.

Which row is northbound. Stock Connect A shares are registered to 香港中央结算有限公司
(HKSCC) itself. 香港中央结算(代理人)有限公司 / HKSCC NOMINEES hold H shares of dual-listed
firms and are NOT northbound. So a northbound row is one whose name (whitespace
removed) starts with `NORTHBOUND_PREFIX` and contains none of `NOT_NORTHBOUND`;
the variants this admits (...有限公司(A股), (沪股通), (陆股通), 香港中央结算公司) are the
same holder written differently.

Point in time. These rows carry no trade date -- their `trade_date` column is
empty in the events domain -- so the window is keyed on `ann_date` and
visibility on the row-level `available_at` (the announcement day's 23:59:59;
the host's nightly disclosure node releases it the evening after, so the
earliest decision that reads it is two calendar days after the announcement,
see `pit-field-map.md`). Only quarter-end `end_date`s count: the float table also
carries prospectus and other off-period dates. Per (`ts_code`, `end_date`) and
table the latest visible announcement wins.

The two numbers, per name, at the newest quarter-end E with a visible report:

    lvl   `hold_float_ratio` of the northbound row at E, in PERCENT of float.
          The float table's value is preferred, the total table's is the
          fallback (they agree on the same report).
    chg   lvl(E) / mean(lvl at the up to `knobs.CHG_QUARTERS` earlier
          quarter-ends that carry a northbound row) - 1: the quarterly,
          share-count analog of an abnormal-holding ratio (holdings over their
          own trailing average).

A report without a northbound row means HKSCC sat below the tenth holder: the
level is censored, not zero. Both numbers are then missing, and the score gives
a missing component the neutral rank (`score`). A name with no visible report at
all is missing the same way. Values are forward-filled until the next report,
with no age cap; `age_days` is reported so a stale value is visible.
"""

from datetime import timedelta

import numpy as np
import pandas as pd

from lib import knobs

DATASETS = ["top10_holders", "top10_floatholders"]
PREFERRED = "top10_floatholders"
COLUMNS = ["dataset", "available_at", "ts_code", "ann_date", "end_date", "holder_name",
           "hold_float_ratio"]
QUARTER_ENDS = ("0331", "0630", "0930", "1231")
NORTHBOUND_PREFIX = "香港中央结算"
NOT_NORTHBOUND = ("代理", "H股", "NOMINEE")


def is_northbound(names):
    """(rows,) bool: the holder is HKSCC itself, the Stock Connect nominee."""

    clean = names.fillna("").astype(str).str.replace(r"\s+", "", regex=True)
    upper = clean.str.upper()
    out = clean.str.startswith(NORTHBOUND_PREFIX)
    for word in NOT_NORTHBOUND:
        out &= ~upper.str.contains(word.upper(), regex=False)
    return out.to_numpy()


def quarter_back(end_date, count):
    """The quarter-end `count` quarters before a quarter-end YYYYMMDD string."""

    slot = int(end_date[:4]) * 4 + QUARTER_ENDS.index(end_date[4:]) - count
    return f"{slot // 4}{QUARTER_ENDS[slot % 4]}"


def read(context, codes):
    """Visible quarter-end rows of both tables over the window, `ts_code` pushed down."""

    start = (context.inference_at - timedelta(days=knobs.HOLDINGS_LOOKBACK_DAYS)).strftime("%Y%m%d")
    frame = pd.read_parquet(
        context.asof_dir + "/events",
        columns=COLUMNS,
        filters=[("dataset", "in", DATASETS), ("ann_date", ">=", start),
                 ("ts_code", "in", list(codes))],
    )
    missing = [name for name in COLUMNS if name not in frame.columns]
    if missing:
        raise RuntimeError(f"the holder tables are missing columns {missing}")
    stamp = pd.to_datetime(frame["available_at"], utc=True)
    frame = frame[stamp <= pd.Timestamp(context.inference_at).tz_convert("UTC")]
    absent = [name for name in DATASETS if not (frame["dataset"] == name).any()]
    if absent:
        raise RuntimeError(
            f"no visible {absent} row over the last {knobs.HOLDINGS_LOOKBACK_DAYS} days; this arm "
            "reads the UNION of both top-10 tables and both must be selected into the snapshot")
    frame = frame.assign(ann_date=frame["ann_date"].astype(str), end_date=frame["end_date"].astype(str))
    return frame[frame["end_date"].str[4:].isin(QUARTER_ENDS)].reset_index(drop=True)


def table(rows):
    """[ts_code, end_date, lvl] for every (name, quarter-end) with a visible report; lvl NaN = censored."""

    newest = rows.groupby(["dataset", "ts_code", "end_date"])["ann_date"].transform("max")
    rows = rows[rows["ann_date"] == newest]
    reports = rows[["ts_code", "end_date"]].drop_duplicates()
    north = rows[is_northbound(rows["holder_name"])]
    if north.empty:
        raise RuntimeError("visible holder reports carry no northbound (HKSCC) row at all; "
                           "the name rule or the read is wrong, not the market")
    north = north.assign(value=pd.to_numeric(north["hold_float_ratio"], errors="coerce"))
    per_table = north.groupby(["ts_code", "end_date", "dataset"])["value"].max().unstack("dataset")
    for name in DATASETS:
        if name not in per_table.columns:
            per_table[name] = np.nan
    other = [name for name in DATASETS if name != PREFERRED][0]
    lvl = per_table[PREFERRED].fillna(per_table[other]).rename("lvl").reset_index()
    return reports.merge(lvl, on=["ts_code", "end_date"], how="left")


def signals(context, codes):
    """Indexed by ts_code: lvl, chg, end_date E and its age in days, for names with a visible report."""

    tab = table(read(context, codes))
    lookup = tab.set_index(["ts_code", "end_date"])["lvl"]
    newest = tab.groupby("ts_code")["end_date"].max()
    out = pd.DataFrame({"end_date": newest})
    out["lvl"] = lookup.reindex(list(zip(out.index, out["end_date"]))).to_numpy()
    earlier = np.column_stack([
        lookup.reindex([(code, quarter_back(end, k)) for code, end in zip(out.index, out["end_date"])]).to_numpy()
        for k in range(1, knobs.CHG_QUARTERS + 1)
    ]) if len(out) else np.zeros((0, knobs.CHG_QUARTERS))
    present = np.isfinite(earlier)
    count = present.sum(axis=1)
    base = np.where(count > 0, np.where(present, earlier, 0.0).sum(axis=1) / np.maximum(count, 1), np.nan)
    level = out["lvl"].to_numpy(dtype=np.float64)
    usable = np.isfinite(level) & np.isfinite(base) & (base > 0)
    chg = np.where(usable, level / np.where(usable, base, 1.0) - 1.0, np.nan)
    out["chg"] = chg
    decision = pd.Timestamp(context.inference_at).tz_localize(None).normalize()
    out["age_days"] = (decision - pd.to_datetime(out["end_date"], format="%Y%m%d")).dt.days
    return out


def pct_rank(values):
    """Cross-sectional percentile rank in (0, 1); ties average, non-finite stays NaN."""

    values = np.asarray(values, dtype=np.float64)
    out = np.full(values.shape, np.nan)
    finite = np.isfinite(values)
    count = int(finite.sum())
    if count:
        out[finite] = (pd.Series(values[finite]).rank(method="average").to_numpy() - 0.5) / count
    return out


def neutral_rank(values, keep):
    """Percentile rank over the `keep` names, a missing value at the neutral 0.5; NaN off `keep`."""

    out = np.full(len(values), np.nan)
    ranked = pct_rank(np.where(keep, values, np.nan))
    out[keep] = np.where(np.isfinite(ranked[keep]), ranked[keep], 0.5)
    return out


def components(context, panel, keep):
    """(lvl, chg, age_days) aligned to panel["symbols"], NaN where the name has no value."""

    codes = [str(code) for code in panel["symbols"]]
    sig = signals(context, codes).reindex(pd.Index(codes))
    return (sig["lvl"].to_numpy(dtype=np.float64), sig["chg"].to_numpy(dtype=np.float64),
            sig["age_days"].to_numpy(dtype=np.float64))


def report(keep, lvl, chg, age):
    """Coverage readings every buy order carries: shares of the tradable pool with each value."""

    count = int(keep.sum())
    if count == 0:
        return {"nb_pool": 0}
    finite_age = age[keep][np.isfinite(age[keep])]
    return {
        "nb_pool": count,
        "nb_lvl_coverage": round(float(np.isfinite(lvl[keep]).mean()), 4),
        "nb_chg_coverage": round(float(np.isfinite(chg[keep]).mean()), 4),
        "nb_report_age_median_days": float(np.median(finite_age)) if finite_age.size else None,
    }
