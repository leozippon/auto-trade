"""Fundamentals for the fundamentals lane: point-in-time features, the two fusion forms' inputs, and c_perm.

Features. One dataset, `fina_indicator_vip` (vendor indicators stamped at the
announcement, near-complete from the first research year on), one column per
family, chosen by coverage before any return was read:

    q_roe               profitability, single quarter (percent)
    grossprofit_margin  margin, year to date (percent)
    debt_to_assets      leverage (percent)
    netprofit_yoy       earnings growth, year to date on last year's (percent)
    q_sales_yoy         sales growth, single quarter on last year's (percent)
    age                 calendar days since the report in use became visible

Each name carries its newest report period as a decision sees it -- the newest
visible version of that period, and a restated older period never replaces a
newer one -- and each feature is ranked over the tradable names of the date
into (0, 1]. Missing stays missing: a field the report lacks is NaN, never
taken from an older report, and a name without a report is NaN throughout.

Visibility. Row t of the panel is what the 08:30 decision of the next trading
day sees -- the day the label's position opens -- by `fund.event_days`: a
stamp of day D is first usable on the first trading day after D, except that
a Monday does not see the weekend just before it. The newest row is the
decision itself.

Forms (`knobs.FUND`):

    late   `lib/tree.py` fits a LightGBM on these ranks (NaN as unknown) to
           the same label over the fit dates on which at least MIN_COVERAGE of
           the names have a report; at a review its prediction's rank is
           blended into the bag's: rank((1 - WEIGHT) x bag + WEIGHT x fund).
    early  every head's last linear layer reads its hidden units plus, per
           feature, the centred rank (0 when missing) and a missing flag.

c_perm (`knobs.FUND_PERM`): on every date, in training and at a review alike,
the feature rows are permuted among the names that date uses, seeded by the
date. The model, its capacity and the features' distribution stay; only the
link between a stock and its own fundamentals is cut.
"""

import numpy as np
import pandas as pd

from lib import fund, knobs, panel as P

DATASET = "fina_indicator_vip"
COLUMNS = ("q_roe", "grossprofit_margin", "debt_to_assets", "netprofit_yoy", "q_sales_yoy")
FEATURES = (*COLUMNS, "age")
WIDTH = 2 * len(FEATURES)    # early fusion: centred rank and missing flag per feature
WEIGHT = 0.25                # late fusion: the fundamentals score's share of the blend
MIN_COVERAGE = 0.5           # late fusion: share of a date's names that must have a report


def asof_panel(rows, codes, days):
    """{feature: (len(days) - 1, len(codes)) float64}: row t is what the decision on days[t + 1] sees.

    `rows` are `fund.read` rows of one dataset; `codes` the panel's sorted
    names; `days` its sorted trading days followed by the decision day.
    """

    days = np.asarray(days, dtype=str)
    codes = np.asarray(codes, dtype=str)
    first = fund.event_days(rows["available_at"], days.tolist())
    seen = np.array([day is not None for day in first], dtype=bool)
    rows = rows.loc[seen].assign(first=[day for day in first if day is not None])
    rows = rows[rows["ts_code"].isin(codes)]
    if rows.empty:
        raise RuntimeError(f"no {DATASET} report of any panel name is visible")
    rows = rows.assign(
        row=np.maximum(np.searchsorted(days, rows["first"].to_numpy(dtype=str)) - 1, 0),
        period=rows["end_date"].astype(int),
    ).sort_values(["ts_code", "row", "stamp"], kind="stable")
    rows = rows[rows["period"] == rows.groupby("ts_code")["period"].cummax()]
    rows = rows.sort_values(["row", "stamp"], kind="stable").reset_index(drop=True)
    # The latest kept report at or before each row: reports are numbered in
    # visibility order, so a running maximum is the newest one seen so far.
    pick = np.full((len(days) - 1, len(codes)), -1, dtype=np.int64)
    where = (rows["row"].to_numpy(), np.searchsorted(codes, rows["ts_code"].to_numpy(dtype=str)))
    np.maximum.at(pick, where, np.arange(len(rows)))
    pick = np.maximum.accumulate(pick, axis=0)
    have = pick >= 0
    at = np.maximum(pick, 0)
    out = {name: np.where(have, rows[name].to_numpy(dtype=np.float64)[at], np.nan) for name in COLUMNS}
    stamped = pd.to_datetime(rows["available_at"].astype(str).str[:10]).to_numpy()
    decided = pd.to_datetime(pd.Series(days[1:].tolist()), format="%Y%m%d").to_numpy()
    out["age"] = np.where(have, (decided[:, None] - stamped[at]) / np.timedelta64(1, "D"), np.nan)
    return out


def features(context, data):
    """(dates, names, len(FEATURES)) float32 ranks in (0, 1] over each date's tradable names; NaN = missing."""

    rows = fund.read(context, DATASET, list(COLUMNS))
    today = pd.Timestamp(context.inference_at).tz_convert(fund.TIMEZONE).strftime("%Y%m%d")
    raw = asof_panel(rows, data["codes"], [*data["dates"], today])
    tradable = P.tradable(data, slice(None))
    out = np.full((*tradable.shape, len(FEATURES)), np.nan, dtype=np.float32)
    for k, name in enumerate(FEATURES):
        out[:, :, k] = pd.DataFrame(np.where(tradable, raw[name], np.nan)).rank(axis=1, pct=True).to_numpy()
    return out


def take(fx, data, t, idx):
    """(len(idx), len(FEATURES)) feature rows of date index t for the names idx; permuted under c_perm."""

    rows = fx[t, idx]
    if knobs.FUND_PERM:
        rows = rows[np.random.default_rng(int(data["dates"][t])).permutation(len(idx))]
    return rows


def early_inputs(rows):
    """(n, WIDTH) float32 head inputs: the centred rank (0 when missing), then the missing flags."""

    missing = np.isnan(rows)
    return np.concatenate([np.where(missing, 0.0, rows - 0.5), missing], axis=1).astype(np.float32)


def blend(bag, other, weight=WEIGHT):
    """Rank in [0, 1] of (1 - weight) x bag + weight x other where both are finite; NaN elsewhere."""

    both = np.isfinite(bag) & np.isfinite(other)
    out = np.full(len(bag), np.nan)
    mixed = (1.0 - weight) * bag[both] + weight * other[both]
    out[both] = np.argsort(np.argsort(mixed, kind="stable"), kind="stable") / max(1, int(both.sum()) - 1)
    return out
