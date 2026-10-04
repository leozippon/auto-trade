"""Industry readings of the cycle lane, point in time, computed at a review.

Members. A name's industry is its Shenwan L1 label in the replay year's
`universe` (one vintage per research year: the SW2014 families through the
2021 vintage, SW2021 from the 2022-06-30 vintage); a name without a label is
never scored. Industry price series (`sw_daily`) are not read: before
2021-12 they are the SW2021 family back-calculated with hindsight.

Firm readings, by the rules of `lib/fund.py` (the newest visible version of
each period; a firm's newest period is its latest `end_date`):

    sales_accel   `q_sales_yoy` (single-quarter revenue growth on the same
                  quarter a year earlier, percent, a vendor field) of the
                  newest period minus that of the quarter before it
    roe_change    `q_dt_roe` (single-quarter return on equity excluding
                  non-recurring items, percent) of the newest period minus
                  that of the same quarter a year earlier
    capex_change  capex intensity -- year-to-date cash paid for fixed,
                  intangible and other long-term assets over total assets --
                  at the newest period minus at the same period a year earlier

A reading is NaN when a period it needs is not visible, and every reading of
a firm whose newest period ended more than STALE_DAYS before the decision is
NaN. Rows stamped after the decision cannot be in the as-of view; the count
of such rows is reported anyway (`fund_after_decision`, expected 0).

Industry readings: the median over the industry's scored names with a
finite reading, NaN when fewer than MIN_COVERAGE of its names have one.

    G    mean of the cross-industry percentile ranks of median sales_accel and
         median roe_change (an industry needs both)
    GI   G minus half the percentile rank of median capex_change, ranked again
    px   the control c_px: the mean over the industry's names of the trailing
         MOMENTUM_DAYS return on adjusted closes (a name needs both ends)

`ranking` returns the industries in order, best first.
"""

import numpy as np
import pandas as pd

from lib import data, fund

STALE_DAYS = 200
MIN_COVERAGE = 0.5
MOMENTUM_DAYS = 120
UNLABELLED = "未分类"
PREVIOUS_QUARTER = {"0331": (-1, "1231"), "0630": (0, "0331"), "0930": (0, "0630"), "1231": (0, "0930")}


def labels(context, codes):
    """Series code -> Shenwan L1 label of the replay year's universe; NaN where there is none."""

    universe = pd.read_parquet(context.asof_dir + "/universe", columns=["ts_code", "l1_name"])
    label = universe.drop_duplicates("ts_code").set_index("ts_code")["l1_name"].reindex(list(codes))
    text = label.fillna("").astype(str)
    return label.where((text != "") & (text != UNLABELLED))


def _shift(ends, years, mmdd=None):
    """End dates moved by `years` (keeping MMDD, or replacing it with `mmdd`)."""

    year = ends.str[:4].astype(int) + years
    return year.astype(str) + (ends.str[4:] if mmdd is None else mmdd)


def firm_readings(context, codes):
    """(frame indexed by code: sales_accel, roe_change, capex_change; audit dict)."""

    codes = pd.Index(list(codes))
    indicator = fund.per_period(fund.read(context, "fina_indicator_vip", ["q_sales_yoy", "q_dt_roe"]))
    cash = fund.per_period(fund.read(context, "cashflow_vip", ["c_pay_acq_const_fiolta"]))
    assets = fund.per_period(fund.read(context, "balancesheet_vip", ["total_assets"]))
    decision = pd.Timestamp(context.inference_at)
    audit = {
        "fund_newest_stamp": str(max(frame["stamp"].max() for frame in (indicator, cash, assets))),
        "fund_after_decision": int(sum(int((frame["stamp"] > decision).sum()) for frame in (indicator, cash, assets))),
    }
    indicator = indicator[indicator["ts_code"].isin(codes)
                          & indicator["end_date"].astype(str).str[4:].isin(PREVIOUS_QUARTER)]
    newest = fund.newest_period(indicator).set_index("ts_code")
    ends = newest["end_date"].astype(str)
    fresh = (decision.tz_localize(None).normalize() - pd.to_datetime(ends, format="%Y%m%d")).dt.days <= STALE_DAYS
    quarter = ends.str[4:].map(PREVIOUS_QUARTER)
    previous = (ends.str[:4].astype(int) + quarter.str[0]).astype(str) + quarter.str[1]
    year_ago = _shift(ends, -1)

    def at(frame, column, keys):
        table = frame.set_index(["ts_code", "end_date"])[column]
        index = pd.MultiIndex.from_arrays([keys.index.to_numpy(dtype=object), keys.to_numpy(dtype=object)])
        return pd.Series(table.reindex(index).to_numpy(dtype="float64"), index=keys.index)

    intensity_now = at(cash, "c_pay_acq_const_fiolta", ends) / at(assets, "total_assets", ends)
    intensity_then = at(cash, "c_pay_acq_const_fiolta", year_ago) / at(assets, "total_assets", year_ago)
    out = pd.DataFrame({
        "sales_accel": at(indicator, "q_sales_yoy", ends) - at(indicator, "q_sales_yoy", previous),
        "roe_change": at(indicator, "q_dt_roe", ends) - at(indicator, "q_dt_roe", year_ago),
        "capex_change": intensity_now - intensity_then,
    })
    out = out.mul(np.where(fresh, 1.0, np.nan), axis=0).replace([np.inf, -np.inf], np.nan)
    return out.reindex(codes), audit


def _median(values, label):
    """Per-industry median of `values` over the labelled names, NaN below MIN_COVERAGE."""

    frame = pd.DataFrame({"value": values, "industry": label}).dropna(subset=["industry"])
    grouped = frame.groupby("industry")["value"]
    coverage = grouped.apply(lambda column: column.notna().mean())
    return grouped.median().where(coverage >= MIN_COVERAGE)


def momentum(context, codes, label):
    """Per-industry mean trailing MOMENTUM_DAYS return on adjusted closes."""

    panel = data.panel(context, ["close", "adj_factor"], MOMENTUM_DAYS + 1)
    adjusted = (panel["close"] * panel["adj_factor"]).reindex(columns=list(codes))
    trailing = adjusted.iloc[-1] / adjusted.iloc[0] - 1.0
    frame = pd.DataFrame({"value": trailing, "industry": label}).dropna(subset=["industry"])
    grouped = frame.groupby("industry")["value"]
    coverage = grouped.apply(lambda column: column.notna().mean())
    return grouped.mean().where(coverage >= MIN_COVERAGE)


def ranking(context, codes, signal, label, firm):
    """Industries best first under `signal`; industries without a reading are left out."""

    if signal == "px":
        value = momentum(context, codes, label)
    else:
        sales = _median(firm["sales_accel"], label)
        roe = _median(firm["roe_change"], label)
        value = (sales.rank(pct=True) + roe.rank(pct=True)) / 2.0
        if signal == "GI":
            capex = _median(firm["capex_change"], label)
            value = value.rank(pct=True) - 0.5 * capex.rank(pct=True)
        elif signal != "G":
            raise ValueError(f"signal must be 'G', 'GI' or 'px', got {signal!r}")
    value = value.dropna()
    return list(value.sort_values(ascending=False, kind="stable").index)
