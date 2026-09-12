"""PIT-safe construction of the earnings-surprise face and its two controls.

One builder serves every decision, so a cold and a warm worker score the same
cross-section. Everything is read from the rolling as-of view with column
projection and date filters; nothing is fitted, nothing is cached across
calls, and the heaviest step is one bounded groupby.

Events (first announced version per (ts_code, end_date), visible at 18:00 of
the announcement day, so a decision at 08:30 sees announcements up to T-1):

    annual actual   fundamentals.income_vip, report_type "1", end_date *1231,
                    n_income_attr_p (CNY)
    express         fundamentals.express_vip, end_date *1231, n_income (CNY);
                    whichever of the two comes first for a firm-year is the
                    annual event (the other is ignored)
    quarterly       fundamentals.income_vip, every end_date: the quarter's
                    increment of n_income_attr_p minus the same quarter a year
                    earlier, divided by circ_mv (CNY) on the signal date

Consensus for the annual event: mean over brokers of each broker's latest
report_rc.np (万元, x 1e4) for the fiscal year (`quarter` == "YYYYQ4"; it is a
fiscal-year label, not a calendar quarter) with available_at inside the
CONSENSUS_DAYS before 00:00 of the announcement day -- strictly before it, so
a same-evening broker update cannot enter -- and at least MIN_BROKERS brokers.

    sue_c = (actual - consensus) / max(|consensus|, CONSENSUS_FLOOR)

Signal age is counted in visible trading days since the first decision day
that could see the event; the freshness weight is a declared step function.
Units come from the snapshot contract: daily prices CNY/share, `amount` and
`circ_mv` CNY, `turnover_rate` and `pct_chg` decimals; fundamentals CNY;
report_rc `np` 万元.
"""

from datetime import timedelta

import numpy as np
import pandas as pd

EPS = 1e-12

# candidate -> what the score is. `main.CANDIDATE` is the only line that
# differs between the pre-registered legs of one batch.
CANDIDATES = {
    "s1": "surprise",       # main candidate: consensus surprise + quarterly SUE
    "c_growth": "growth",   # gate-1 control: the same report's yoy growth only
    "c_timing": "timing",   # gate-2 control: announcement freshness only
}

CONSENSUS_DAYS = 120          # calendar days of broker reports before the announcement
MIN_BROKERS = 2
CONSENSUS_FLOOR = 1.0e6       # CNY, denominator floor for sue_c
CIRC_MV_FLOOR = 1.0e8         # CNY, denominator floor for sue_q
QUARTERS_FOR_SUE = 5          # current quarter plus the same quarter a year earlier
FRESHNESS = ((10, 1.0), (20, 0.5), (40, 0.25), (120, 0.10))  # (max age, weight)
MAX_AGE = FRESHNESS[-1][0]
EVENT_LOOKBACK_DAYS = 700     # calendar days of fundamentals read per decision (the lag-4 quarter of a 120-day-old statement)
CONSENSUS_LOOKBACK_DAYS = 150 # calendar days of report_rc read per decision (>= CONSENSUS_DAYS)
DAILY_LOOKBACK_DAYS = 200     # calendar days of daily rows read per decision: longer than MAX_AGE trading days, so an event announced before the window is dead
PERIOD_RECENCY_DAYS = 400     # a statement whose period ended earlier than this before its announcement is a restatement of an old period, not an event

NEUTRAL_COLUMNS = ["log_circ_mv", "mom_20", "mom_60", "r_5", "turn_20", "vol_20", "max_20", "ep", "yoy_np"]
# A control is never neutralized on its own face: residualizing the growth
# skeleton on yoy_np would leave noise, not a growth book. The candidate and
# the timing placebo always use the full set.
CONTROL_EXCLUDES = {"growth": ("yoy_np",)}

PRICE_CAP = 30.0              # T-1 close, CNY: one 100-share lot <= half a position
ADV_FLOOR = 3.0e7             # CNY
MIN_LISTED_DAYS = 120


def kind_of(candidate):
    return CANDIDATES[candidate]


# --------------------------------------------------------------------------- reads


def _visible(frame, context):
    stamp = pd.to_datetime(frame["available_at"], utc=True)
    return frame[stamp <= pd.Timestamp(context.inference_at).tz_convert("UTC")]


def read_daily(context):
    start = (context.inference_at - timedelta(days=DAILY_LOOKBACK_DAYS)).strftime("%Y%m%d")
    frame = pd.read_parquet(
        context.asof_dir + "/daily",
        columns=["ts_code", "trade_date", "close", "pct_chg", "amount", "adj_factor",
                 "turnover_rate", "circ_mv", "pe_ttm", "is_suspended"],
        filters=[("trade_date", ">=", start)],
    )
    frame = frame[(frame["close"] > 0) & frame["adj_factor"].notna()]
    frame["trade_date"] = frame["trade_date"].astype(str)
    frame = frame.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    anchor = frame.groupby("ts_code", sort=False)["adj_factor"].transform("last")
    frame["q_close"] = frame["close"] * frame["adj_factor"] / anchor
    return frame


def read_universe(context):
    return pd.read_parquet(
        context.asof_dir + "/universe",
        columns=["ts_code", "name", "list_date", "l1_code"],
    )


def read_statements(context):
    """First announced version of every income statement visible at the decision."""
    start = (context.inference_at - timedelta(days=EVENT_LOOKBACK_DAYS)).strftime("%Y%m%d")
    frame = pd.read_parquet(
        context.asof_dir + "/fundamentals",
        columns=["dataset", "available_at", "ts_code", "end_date", "ann_date",
                 "report_type", "n_income_attr_p", "n_income"],
        filters=[("dataset", "in", ["income_vip", "express_vip"]), ("ann_date", ">=", start)],
    )
    frame = _visible(frame, context).copy()
    frame["stamp"] = pd.to_datetime(frame["available_at"], utc=True)
    # A restatement of an old period whose first version predates the read window
    # looks like a first version; only a period that ended recently can be an event.
    ann_day = frame["stamp"].dt.tz_convert("Asia/Shanghai").dt.tz_localize(None)
    period_end = pd.to_datetime(frame["end_date"].astype(str), format="%Y%m%d", errors="coerce")
    frame = frame[period_end >= ann_day - pd.Timedelta(days=PERIOD_RECENCY_DAYS)]
    income = frame[(frame["dataset"] == "income_vip") & (frame["report_type"] == "1")
                   & frame["n_income_attr_p"].notna()]
    income = income.sort_values("stamp").drop_duplicates(["ts_code", "end_date"], keep="first")
    express = frame[(frame["dataset"] == "express_vip") & frame["n_income"].notna()
                    & frame["end_date"].astype(str).str.endswith("1231")]
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


# ------------------------------------------------------------------ event values


def annual_events(income, express, reports):
    """One row per firm-year: the first annual announcement and its consensus surprise."""
    inc = income[income["end_date"].astype(str).str.endswith("1231")][["ts_code", "end_date", "stamp", "n_income_attr_p"]]
    inc = inc.rename(columns={"n_income_attr_p": "actual"})
    exp = express[["ts_code", "end_date", "stamp", "n_income"]].rename(columns={"n_income": "actual"})
    events = pd.concat([inc, exp], ignore_index=True).sort_values("stamp")
    events = events.drop_duplicates(["ts_code", "end_date"], keep="first").copy()
    if events.empty or reports.empty:
        return events.assign(sue_c=np.nan).iloc[0:0]
    events["fy"] = events["end_date"].astype(str).str[:4]
    # consensus window: [announcement day 00:00 - CONSENSUS_DAYS, announcement day 00:00)
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
    """One row per statement: the quarter's earnings change vs a year earlier, and yoy growth."""
    frame = income[["ts_code", "end_date", "stamp", "n_income_attr_p"]].copy()
    frame["end_date"] = frame["end_date"].astype(str)
    frame["fy"] = frame["end_date"].str[:4]
    frame["qn"] = frame["end_date"].str[4:6].map({"03": 1, "06": 2, "09": 3, "12": 4})
    frame = frame.dropna(subset=["qn"]).sort_values(["ts_code", "end_date"])
    grouped = frame.groupby(["ts_code", "fy"], sort=False)["n_income_attr_p"]
    prev_cum = grouped.shift(1)
    frame["qinc"] = np.where(frame["qn"] == 1, frame["n_income_attr_p"], frame["n_income_attr_p"] - prev_cum)
    by_stock = frame.groupby("ts_code", sort=False)
    frame["qinc_lag4"] = by_stock["qinc"].shift(4)
    frame["cum_lag4"] = by_stock["n_income_attr_p"].shift(4)
    frame["d_earn"] = frame["qinc"] - frame["qinc_lag4"]
    frame["yoy_np"] = (frame["n_income_attr_p"] - frame["cum_lag4"]) / frame["cum_lag4"].abs().clip(lower=CONSENSUS_FLOOR)
    return frame[["ts_code", "end_date", "stamp", "d_earn", "yoy_np"]]


def latest_with_age(events, trading_days, latest_day):
    """Each firm's most recent event (by stamp), with its age in visible trading days.

    `trading_days` is the sorted visible trading calendar; an event stamped
    after 08:30 of day d is first usable at the next trading day's decision,
    so its signal date is the first visible trading day whose close follows
    the stamp, and age = index(latest_day) - index(signal date). The latest
    event is chosen by stamp, never by age: every event announced before the
    window's first day would tie at the same age, and a tie-break by row order
    picks an arbitrary -- in practice the oldest -- statement. Such an event is
    older than the window, which DAILY_LOOKBACK_DAYS keeps longer than MAX_AGE
    trading days, so its age is reported as the window length and it is dead.
    """
    if events.empty:
        return events.assign(age=np.nan).iloc[0:0]
    latest = events.sort_values("stamp").drop_duplicates("ts_code", keep="last").copy()
    days = np.array(trading_days)
    # signal date = first trading day on or after the announcement day
    ann_day = latest["stamp"].dt.tz_convert("Asia/Shanghai").dt.strftime("%Y%m%d").to_numpy()
    idx = np.searchsorted(days, ann_day, side="left")
    latest_idx = int(np.searchsorted(days, latest_day, side="left"))
    latest["age"] = np.where(ann_day < days[0], len(days), latest_idx - idx)
    return latest[latest["age"] >= 0]


def freshness(age):
    weight = np.zeros(len(age), dtype="float64")
    remaining = np.ones(len(age), dtype=bool)
    for max_age, value in FRESHNESS:
        hit = remaining & (age.to_numpy() <= max_age)
        weight[hit] = value
        remaining &= ~hit
    return weight


# ------------------------------------------------------------------- daily context


def daily_context(daily):
    """Neutralization inputs and screening columns at the latest visible date.

    These are never candidate features: the raw surprise ranks are negative at
    a 20-day horizon before neutralization, so dropping any of them changes
    the mechanism, not just the noise.
    """
    grouped = daily.groupby("ts_code", sort=False)
    ret = grouped["q_close"].pct_change()
    close = daily["q_close"]
    frame = daily[["ts_code", "trade_date", "close", "circ_mv", "is_suspended"]].copy()
    frame["log_circ_mv"] = np.log(daily["circ_mv"].clip(lower=1.0))
    frame["mom_20"] = close / grouped["q_close"].shift(20) - 1.0
    frame["mom_60"] = close / grouped["q_close"].shift(60) - 1.0
    frame["r_5"] = close / grouped["q_close"].shift(5) - 1.0
    frame["turn_20"] = grouped["turnover_rate"].transform(lambda s: s.rolling(20, min_periods=10).mean())
    frame["vol_20"] = ret.groupby(daily["ts_code"], sort=False).transform(lambda s: s.rolling(20, min_periods=10).std())
    frame["max_20"] = ret.groupby(daily["ts_code"], sort=False).transform(lambda s: s.rolling(20, min_periods=10).max())
    frame["adv_20"] = grouped["amount"].transform(lambda s: s.rolling(20, min_periods=10).mean())
    pe = daily["pe_ttm"].where(daily["pe_ttm"] > 0)
    frame["ep"] = (1.0 / pe).fillna(0.0)
    return frame


def eligible(section, universe, latest_day):
    """Declared pool: no BJ/STAR/ST, listed >= 120 days, ADV floor, price cap, not halted."""
    codes = section["ts_code"]
    keep = (~codes.str.endswith(".BJ")) & (~codes.str.startswith(("688", "689")))
    keep &= (section["close"] <= PRICE_CAP) & (section["adv_20"] >= ADV_FLOOR)
    keep &= ~section["is_suspended"].fillna(False).astype(bool)
    info = universe.set_index("ts_code").reindex(codes)
    name = info["name"].astype(str)
    keep &= ~(name.str.contains("ST") | name.str.contains("退")).to_numpy()
    listed = pd.to_datetime(info["list_date"], format="%Y%m%d", errors="coerce")
    keep &= (listed <= pd.Timestamp(latest_day) - pd.Timedelta(days=MIN_LISTED_DAYS)).to_numpy()
    return section[keep.to_numpy()]


# --------------------------------------------------------------------- scoring


def standardize(values):
    """Cross-sectional median/MAD z-score clipped to +-3; all-zero when degenerate."""
    x = values.astype("float64").replace([np.inf, -np.inf], np.nan)
    median = x.median()
    mad = (x - median).abs().median() * 1.4826
    if not np.isfinite(mad) or mad <= EPS:
        return pd.Series(0.0, index=values.index)
    return ((x - median) / mad).clip(-3.0, 3.0)


def build_cross_section(context, candidate):
    """Scored, neutralization-ready cross-section at the latest visible date.

    Returns (frame, latest_day): one row per eligible name carrying `raw`
    (the candidate's score before neutralization), `age` and the
    NEUTRAL_COLUMNS, plus `close` for lot sizing. Names without any event
    inside MAX_AGE are excluded -- off-season the pool is last season's tail.
    """
    daily = read_daily(context)
    if daily.empty:
        return None, None
    trading_days = sorted(daily["trade_date"].unique())
    latest_day = trading_days[-1]
    universe = read_universe(context)
    income, express = read_statements(context)
    quarterly = latest_with_age(quarterly_events(income), trading_days, latest_day)
    context_frame = daily_context(daily)
    section = context_frame[context_frame["trade_date"] == latest_day].copy()
    section = eligible(section, universe, latest_day)
    section = section.merge(quarterly[["ts_code", "age", "d_earn", "yoy_np"]].rename(columns={"age": "age_q"}),
                            on="ts_code", how="left")
    section["yoy_np"] = section["yoy_np"].fillna(0.0)
    kind = kind_of(candidate)
    if kind == "surprise":
        reports = read_consensus_reports(context)
        annual = latest_with_age(annual_events(income, express, reports), trading_days, latest_day)
        section = section.merge(annual[["ts_code", "age", "sue_c"]].rename(columns={"age": "age_c"}), on="ts_code", how="left")
        section["sue_q"] = section["d_earn"] / section["circ_mv"].clip(lower=CIRC_MV_FLOOR)
        legs = []
        for column, age_column in (("sue_c", "age_c"), ("sue_q", "age_q")):
            live = section[age_column].notna() & (section[age_column] <= MAX_AGE) & section[column].notna()
            weight = np.where(live, freshness(section[age_column].fillna(MAX_AGE + 1)), 0.0)
            z = standardize(section[column].where(live))
            legs.append(pd.Series(np.where(live, weight * z.fillna(0.0), np.nan), index=section.index))
        raw = pd.concat(legs, axis=1).mean(axis=1, skipna=True)
        section["age"] = section[["age_c", "age_q"]].min(axis=1)
    elif kind == "growth":
        live = section["age_q"].notna() & (section["age_q"] <= MAX_AGE)
        raw = pd.Series(np.where(live, freshness(section["age_q"].fillna(MAX_AGE + 1)) * standardize(section["yoy_np"].where(live)).fillna(0.0), np.nan), index=section.index)
        section["age"] = section["age_q"]
    else:  # timing: newest announcement first, nothing else
        live = section["age_q"].notna() & (section["age_q"] <= MAX_AGE)
        raw = pd.Series(np.where(live, -section["age_q"].astype("float64"), np.nan), index=section.index)
        section["age"] = section["age_q"]
    section["raw"] = raw
    section = section[section["raw"].notna()].copy()
    industry = universe.set_index("ts_code")["l1_code"].reindex(section["ts_code"]).fillna("NA").to_numpy()
    section["l1_code"] = industry
    return section.reset_index(drop=True), latest_day


def neutralize(section, candidate):
    """Residualize the raw score on NEUTRAL_COLUMNS ranks and SW-L1 dummies.

    A control skips its own face (CONTROL_EXCLUDES); the candidate keeps the
    full set.
    """
    if len(section) < 30:
        return section["raw"].to_numpy(dtype="float64")
    excluded = CONTROL_EXCLUDES.get(kind_of(candidate), ())
    design = [np.ones((len(section), 1))]
    for column in NEUTRAL_COLUMNS:
        if column in excluded:
            continue
        ranked = section[column].astype("float64").rank(pct=True).fillna(0.5).to_numpy()
        design.append(ranked.reshape(-1, 1))
    dummies = pd.get_dummies(section["l1_code"].astype(str), drop_first=True, dtype="float64")
    if dummies.shape[1] > 0:
        design.append(dummies.to_numpy())
    matrix = np.hstack(design)
    target = np.nan_to_num(section["raw"].to_numpy(dtype="float64"), nan=0.0)
    solution, _, _, _ = np.linalg.lstsq(matrix, target, rcond=None)
    return target - matrix @ solution
