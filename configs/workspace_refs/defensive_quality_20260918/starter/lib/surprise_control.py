"""The running earnings-surprise arm's composite, rebuilt as this arm's control.

Rules copied from `configs/workspace_refs/earnings_surprise_20260918/starter/lib/surprise.py`
so that `c_es` scores exactly what that arm trades:

    sue_c  annual actual (first of income_vip annual / express_vip) minus the FY
           consensus -- mean over brokers of each broker's latest report_rc.np
           (万元 x 1e4) with available_at in the CONSENSUS_DAYS before 00:00 of
           the announcement day, >= MIN_BROKERS -- over max(|consensus|, floor)
    sue_q  the quarter's increment of n_income_attr_p minus the same quarter a
           year earlier, over max(circ_mv, floor)
    score  mean of the live legs' freshness weight x cross-sectional median/MAD z
           (clipped +-3); a leg is live for MAX_AGE trading days

Reads the `fundamentals` and `events` as-of domains with column projection and
date windows; a round whose events selection lacks `report_rc` gets the
quarterly leg only, and the fold record must say so.
"""

from datetime import timedelta

import numpy as np
import pandas as pd

EPS = 1e-12
CONSENSUS_DAYS = 120
MIN_BROKERS = 2
CONSENSUS_FLOOR = 1.0e6
CIRC_MV_FLOOR = 1.0e8
FRESHNESS = ((10, 1.0), (20, 0.5), (40, 0.25), (120, 0.10))
MAX_AGE = FRESHNESS[-1][0]
EVENT_LOOKBACK_DAYS = 700
CONSENSUS_LOOKBACK_DAYS = 150


def _visible(frame, context):
    stamp = pd.to_datetime(frame["available_at"], utc=True)
    return frame[stamp <= pd.Timestamp(context.inference_at).tz_convert("UTC")]


def read_statements(context):
    start = (context.inference_at - timedelta(days=EVENT_LOOKBACK_DAYS)).strftime("%Y%m%d")
    frame = pd.read_parquet(
        context.asof_dir + "/fundamentals",
        columns=["dataset", "available_at", "ts_code", "end_date", "ann_date",
                 "report_type", "n_income_attr_p", "n_income"],
        filters=[("dataset", "in", ["income_vip", "express_vip"]), ("ann_date", ">=", start)],
    )
    frame = _visible(frame, context).copy()
    frame["stamp"] = pd.to_datetime(frame["available_at"], utc=True)
    frame["end_date"] = frame["end_date"].astype(str)
    income = frame[(frame["dataset"] == "income_vip") & (frame["report_type"] == "1")
                   & frame["n_income_attr_p"].notna()]
    income = income.sort_values("stamp").drop_duplicates(["ts_code", "end_date"], keep="first")
    express = frame[(frame["dataset"] == "express_vip") & frame["n_income"].notna()
                    & frame["end_date"].str.endswith("1231")]
    express = express.sort_values("stamp").drop_duplicates(["ts_code", "end_date"], keep="first")
    return income, express


def read_consensus_reports(context):
    start = (context.inference_at - timedelta(days=CONSENSUS_LOOKBACK_DAYS)).strftime("%Y%m%d")
    frame = pd.read_parquet(
        context.asof_dir + "/events",
        columns=["dataset", "available_at", "ts_code", "quarter", "np", "org_name", "report_date"],
        filters=[("dataset", "=", "report_rc"), ("report_date", ">=", start)],
    )
    frame = frame[frame["dataset"] == "report_rc"]
    frame = _visible(frame, context).copy()
    frame = frame[frame["np"].notna() & frame["quarter"].astype(str).str.fullmatch(r"\d{4}Q4")]
    frame["fy"] = frame["quarter"].astype(str).str[:4]
    frame["np_cny"] = frame["np"].astype("float64") * 1.0e4
    frame["stamp"] = pd.to_datetime(frame["available_at"], utc=True)
    return frame.sort_values("stamp")


def annual_events(income, express, reports):
    inc = income[income["end_date"].str.endswith("1231")][["ts_code", "end_date", "stamp", "n_income_attr_p"]]
    inc = inc.rename(columns={"n_income_attr_p": "actual"})
    exp = express[["ts_code", "end_date", "stamp", "n_income"]].rename(columns={"n_income": "actual"})
    events = pd.concat([inc, exp], ignore_index=True).sort_values("stamp")
    events = events.drop_duplicates(["ts_code", "end_date"], keep="first").copy()
    if events.empty or reports.empty:
        return events.assign(sue_c=np.nan).iloc[0:0]
    events["fy"] = events["end_date"].str[:4]
    day0 = events["stamp"].dt.tz_convert("Asia/Shanghai").dt.normalize().dt.tz_convert("UTC")
    events["win_end"] = day0
    events["win_start"] = day0 - pd.Timedelta(days=CONSENSUS_DAYS)
    merged = events.merge(reports[["ts_code", "fy", "org_name", "np_cny", "stamp"]].rename(columns={"stamp": "rstamp"}),
                          on=["ts_code", "fy"], how="inner")
    merged = merged[(merged["rstamp"] >= merged["win_start"]) & (merged["rstamp"] < merged["win_end"])]
    merged = merged.sort_values("rstamp").drop_duplicates(["ts_code", "end_date", "org_name"], keep="last")
    cons = merged.groupby(["ts_code", "end_date"]).agg(consensus=("np_cny", "mean"), n_brokers=("np_cny", "size"))
    events = events.merge(cons, on=["ts_code", "end_date"], how="inner")
    events = events[events["n_brokers"] >= MIN_BROKERS].copy()
    events["sue_c"] = (events["actual"] - events["consensus"]) / events["consensus"].abs().clip(lower=CONSENSUS_FLOOR)
    return events[["ts_code", "end_date", "stamp", "sue_c"]]


def quarterly_events(income):
    frame = income[["ts_code", "end_date", "stamp", "n_income_attr_p"]].copy()
    frame["fy"] = frame["end_date"].str[:4]
    frame["qn"] = frame["end_date"].str[4:6].map({"03": 1, "06": 2, "09": 3, "12": 4})
    frame = frame.dropna(subset=["qn"]).sort_values(["ts_code", "end_date"])
    prev_cum = frame.groupby(["ts_code", "fy"], sort=False)["n_income_attr_p"].shift(1)
    frame["qinc"] = np.where(frame["qn"] == 1, frame["n_income_attr_p"], frame["n_income_attr_p"] - prev_cum)
    frame["d_earn"] = frame["qinc"] - frame.groupby("ts_code", sort=False)["qinc"].shift(4)
    return frame[["ts_code", "end_date", "stamp", "d_earn"]]


def latest_with_age(events, trading_days, latest_day):
    if events.empty:
        return events.assign(age=np.nan).iloc[0:0]
    days = np.array(trading_days)
    ann_day = events["stamp"].dt.tz_convert("Asia/Shanghai").dt.strftime("%Y%m%d").to_numpy()
    idx = np.searchsorted(days, ann_day, side="left")
    latest_idx = int(np.searchsorted(days, latest_day, side="left"))
    out = events.copy()
    out["age"] = latest_idx - idx
    out = out[out["age"] >= 0]
    return out.sort_values(["ts_code", "age"]).drop_duplicates("ts_code", keep="first")


def freshness(age):
    weight = np.zeros(len(age), dtype="float64")
    remaining = np.ones(len(age), dtype=bool)
    for max_age, value in FRESHNESS:
        hit = remaining & (age.to_numpy() <= max_age)
        weight[hit] = value
        remaining &= ~hit
    return weight


def standardize(values):
    x = values.astype("float64").replace([np.inf, -np.inf], np.nan)
    median = x.median()
    mad = (x - median).abs().median() * 1.4826
    if not np.isfinite(mad) or mad <= EPS:
        return pd.Series(0.0, index=values.index)
    return ((x - median) / mad).clip(-3.0, 3.0)


def composite(context, trading_days, latest_day, section):
    """The composite keyed by ts_code over the names of `section` (ts_code, circ_mv); NaN when no leg is live."""
    income, express = read_statements(context)
    reports = read_consensus_reports(context)
    annual = latest_with_age(annual_events(income, express, reports), trading_days, latest_day)
    quarterly = latest_with_age(quarterly_events(income), trading_days, latest_day)
    frame = section[["ts_code", "circ_mv"]].copy()
    frame = frame.merge(annual[["ts_code", "age", "sue_c"]].rename(columns={"age": "age_c"}), on="ts_code", how="left")
    frame = frame.merge(quarterly[["ts_code", "age", "d_earn"]].rename(columns={"age": "age_q"}), on="ts_code", how="left")
    frame["sue_q"] = frame["d_earn"] / frame["circ_mv"].clip(lower=CIRC_MV_FLOOR)
    legs = []
    for column, age_column in (("sue_c", "age_c"), ("sue_q", "age_q")):
        live = frame[age_column].notna() & (frame[age_column] <= MAX_AGE) & frame[column].notna()
        weight = np.where(live, freshness(frame[age_column].fillna(MAX_AGE + 1)), 0.0)
        z = standardize(frame[column].where(live))
        legs.append(pd.Series(np.where(live, weight * z.fillna(0.0), np.nan), index=frame.index))
    score = pd.concat(legs, axis=1).mean(axis=1, skipna=True)
    return pd.Series(score.to_numpy(), index=frame["ts_code"].to_numpy())
