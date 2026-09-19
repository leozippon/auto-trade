"""PIT-safe cross-section of the index constituents, scored by a four-leg rank composite.

One builder serves every decision, so a cold and a warm worker score the same
cross-section. Everything is read from the rolling as-of view with column
projection and date filters; nothing is fitted and nothing is cached across calls.

The pool is the as-of constituent list (lib/index.py) minus what this account
cannot trade: STAR names (200-share minimum lot, so one lot does not fit a seat
at either account size), names carrying ST/退, names with no T-1 bar, and names
halted through T-1. Nothing else is screened out -- the index's own membership
rules already do the liquidity work, and the thinnest constituent measured on
the research-year anchors still turns over CNY 53-83 million a day, well above
the floor the whole-market packs have to impose. Affordability is NOT screened
here: it depends on the seat size and therefore on the leg, so `lib/trade.py`
applies it where the seats are known.

Legs (each a cross-sectional percentile rank over the pool, high = long). They
are a RUNNABLE BASELINE, not this arm's thesis -- which signal families carry
alpha among index constituents is exactly what has not been measured:

    quality    cash-flow quality of the FIRST announced version of the latest
               statement (income_vip, cashflow_vip, balancesheet_vip joined on
               (ts_code, end_date), report_type "1", CNY):
               accr = (n_income_attr_p - n_cashflow_act) / total_assets,
               ocf_ta = n_cashflow_act / total_assets * 4 / quarter number;
               leg = mean(rank(-accr), rank(ocf_ta)); a statement older than
               STATE_MAX_AGE visible trading days carries no signal
    lowvol     -std of the residual of the stock's daily return on the CSI300
               return over the last VOL_WINDOW visible days (>= MIN_VOL_DAYS)
    reversal   -(adjusted close / adjusted close REVERSAL_DAYS bars earlier - 1)
    value      mean(rank(1 / pe_ttm), rank(1 / pb)), non-positive multiples ranked last

Score = mean of the leg ranks named in LEGS, then the residual of an OLS on
[1, rank(log circ_mv)] (size-neutral). Inside the index the cap range is still
two orders of magnitude, so the size leg is not a formality.

The section also carries `industry`, the decision day's SW L1 name from
`universe` (empty rows fall in "未分类"), and `weight`, the constituent's percent
of the index. Nothing here uses either -- they are there so a candidate can read
its own industry concentration and its index coverage without rebuilding the
reader, and `industry` is the same column the result's
`stats.benchmark.top_industry_weight` is computed from.

Stamps: statements f_ann_date_or_ann_date 18:00, the index bar and the
constituent section 17:30, so an 08:30 decision sees T-1 and the previous
month-end section; `daily` and `universe` have no available_at column and the
as-of view itself holds only visible rows. Units from the snapshot contract:
daily prices CNY/share, `amount` and `circ_mv` CNY, `pct_chg` a decimal;
fundamentals CNY; `macro.index_daily.pct_chg` and `macro.index_weight.weight`
are percent numbers.
"""

from datetime import timedelta

import numpy as np
import pandas as pd

from lib import index

LEGS = ("quality", "lowvol", "reversal", "value")

VOL_WINDOW = 60
MIN_VOL_DAYS = 40
REVERSAL_DAYS = 20
STATE_MAX_AGE = 130           # visible trading days a statement stays live (about two quarters)
DAILY_LOOKBACK_DAYS = 220     # calendar days: longer than STATE_MAX_AGE trading days, so older statements are dead
STATEMENT_LOOKBACK_DAYS = 400 # calendar days of statements read per decision
PERIOD_RECENCY_DAYS = 400     # a statement for a period that ended earlier than this is a restatement, not news
INDEX_CODE = index.INDEX_CODE
ASSET_FLOOR = 1.0             # CNY
MIN_POOL = 60                 # scorable constituents below which the decision is refused


def _visible(frame, context):
    stamp = pd.to_datetime(frame["available_at"], utc=True)
    return frame[stamp <= pd.Timestamp(context.inference_at).tz_convert("UTC")]


def read_daily(context):
    start = (context.inference_at - timedelta(days=DAILY_LOOKBACK_DAYS)).strftime("%Y%m%d")
    frame = pd.read_parquet(
        context.asof_dir + "/daily",
        columns=["ts_code", "trade_date", "close", "adj_factor", "circ_mv",
                 "pe_ttm", "pb", "is_suspended"],
        filters=[("trade_date", ">=", start)],
    )
    frame = frame[(frame["close"] > 0) & frame["adj_factor"].notna()]
    frame = frame.assign(trade_date=frame["trade_date"].astype(str))
    frame = frame.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    anchor = frame.groupby("ts_code", sort=False)["adj_factor"].transform("last")
    frame["q_close"] = frame["close"] * frame["adj_factor"] / anchor
    return frame


def read_index_returns(context):
    """CSI300 daily returns as decimals keyed by trade_date."""
    start = (context.inference_at - timedelta(days=DAILY_LOOKBACK_DAYS)).strftime("%Y%m%d")
    frame = pd.read_parquet(
        context.asof_dir + "/macro",
        columns=["dataset", "available_at", "ts_code", "trade_date", "pct_chg"],
        filters=[("dataset", "=", "index_daily"), ("ts_code", "=", INDEX_CODE), ("trade_date", ">=", start)],
    )
    frame = _visible(frame, context).dropna(subset=["pct_chg"])
    frame = frame.assign(trade_date=frame["trade_date"].astype(str))
    frame = frame.sort_values("trade_date").drop_duplicates("trade_date", keep="last")
    return pd.Series(frame["pct_chg"].astype("float64").to_numpy() / 100.0, index=frame["trade_date"].to_numpy())


def read_statements(context):
    """First announced version of every income / cash-flow / balance-sheet statement visible at the decision."""
    start = (context.inference_at - timedelta(days=STATEMENT_LOOKBACK_DAYS)).strftime("%Y%m%d")
    frame = pd.read_parquet(
        context.asof_dir + "/fundamentals",
        columns=["dataset", "available_at", "ts_code", "end_date", "ann_date", "report_type",
                 "n_income_attr_p", "n_cashflow_act", "total_assets"],
        filters=[("dataset", "in", ["income_vip", "cashflow_vip", "balancesheet_vip"]), ("ann_date", ">=", start)],
    )
    frame = _visible(frame, context)
    frame = frame[frame["report_type"] == "1"].copy()
    frame["stamp"] = pd.to_datetime(frame["available_at"], utc=True)
    frame["end_date"] = frame["end_date"].astype(str)
    ann_day = frame["stamp"].dt.tz_convert("Asia/Shanghai").dt.tz_localize(None)
    period_end = pd.to_datetime(frame["end_date"], format="%Y%m%d", errors="coerce")
    frame = frame[period_end >= ann_day - pd.Timedelta(days=PERIOD_RECENCY_DAYS)]
    parts = []
    for dataset, column in (("income_vip", "n_income_attr_p"), ("cashflow_vip", "n_cashflow_act"),
                            ("balancesheet_vip", "total_assets")):
        part = frame[(frame["dataset"] == dataset) & frame[column].notna()]
        part = part.sort_values("stamp").drop_duplicates(["ts_code", "end_date"], keep="first")
        parts.append(part[["ts_code", "end_date", "stamp", column]].rename(columns={"stamp": "stamp_" + dataset}))
    joined = parts[0].merge(parts[1], on=["ts_code", "end_date"]).merge(parts[2], on=["ts_code", "end_date"])
    # visible only when all three parts are: the statement's time is the last of the three stamps
    joined["stamp"] = joined[["stamp_income_vip", "stamp_cashflow_vip", "stamp_balancesheet_vip"]].max(axis=1)
    return joined[["ts_code", "end_date", "stamp", "n_income_attr_p", "n_cashflow_act", "total_assets"]]


def latest_quality(statements, trading_days, latest_day):
    """Each name's newest statement (by stamp) with accr, ocf_ta and its age in visible trading days.

    The newest statement is chosen by stamp, never by age: every statement announced
    before the daily window ties at the window length, and such a statement is dead.
    """
    frame = statements.copy()
    frame["qn"] = frame["end_date"].str[4:6].map({"03": 1, "06": 2, "09": 3, "12": 4})
    frame = frame.dropna(subset=["qn"])
    assets = frame["total_assets"].clip(lower=ASSET_FLOOR)
    frame["accr"] = (frame["n_income_attr_p"] - frame["n_cashflow_act"]) / assets
    frame["ocf_ta"] = frame["n_cashflow_act"] / assets * (4.0 / frame["qn"])
    latest = frame.sort_values("stamp").drop_duplicates("ts_code", keep="last").copy()
    days = np.array(trading_days)
    ann_day = latest["stamp"].dt.tz_convert("Asia/Shanghai").dt.strftime("%Y%m%d").to_numpy()
    idx = np.searchsorted(days, ann_day, side="left")
    latest_idx = int(np.searchsorted(days, latest_day, side="left"))
    latest["age"] = np.where(ann_day < days[0], len(days), latest_idx - idx)
    latest = latest[(latest["age"] >= 0) & (latest["age"] <= STATE_MAX_AGE)]
    return latest[["ts_code", "accr", "ocf_ta"]]


def residual_volatility(daily, trading_days, market):
    """VOL_WINDOW-day residual volatility on the CSI300 per name at the newest date."""
    window = trading_days[-VOL_WINDOW:]
    m = market.reindex(window).to_numpy(dtype="float64")
    if np.isfinite(m).sum() < MIN_VOL_DAYS:
        raise RuntimeError(f"{INDEX_CODE} has {int(np.isfinite(m).sum())} visible days in the volatility window")
    rows = daily[daily["trade_date"].isin(window)]
    returns = rows.pivot(index="trade_date", columns="ts_code", values="ret").reindex(window)
    r = returns.to_numpy(dtype="float64")
    both = np.isfinite(r) & np.isfinite(m)[:, None]
    n = both.sum(axis=0)
    safe_n = np.maximum(n, 2)
    r0 = np.where(both, r, 0.0)
    m0 = np.where(both, m[:, None], 0.0)
    mean_r = r0.sum(axis=0) / np.maximum(n, 1)
    mean_m = m0.sum(axis=0) / np.maximum(n, 1)
    dr = np.where(both, r - mean_r, 0.0)
    dm = np.where(both, m[:, None] - mean_m, 0.0)
    cov = (dr * dm).sum(axis=0) / (safe_n - 1)
    var_m = (dm ** 2).sum(axis=0) / (safe_n - 1)
    var_r = (dr ** 2).sum(axis=0) / (safe_n - 1)
    beta = np.where(var_m > 0, cov / np.where(var_m > 0, var_m, 1.0), 0.0)
    vol = np.sqrt(np.clip(var_r - beta * cov, 0.0, None))
    return pd.Series(np.where(n >= MIN_VOL_DAYS, vol, np.nan), index=returns.columns)


def pct_rank(values):
    return values.astype("float64").replace([np.inf, -np.inf], np.nan).rank(pct=True)


def build(context):
    """(scored constituent frame, T-1 close of every visible name, the section's date).

    The frame carries ts_code, industry, weight (percent of the index), close and
    score, one row per tradable constituent. It is None when fewer than MIN_POOL
    constituents are scorable -- a state the caller must refuse, not paper over.
    """
    members, section = index.constituents(context)
    daily = read_daily(context)
    if daily.empty:
        raise RuntimeError("the daily window of the as-of view is empty")
    trading_days = sorted(daily["trade_date"].unique())
    latest_day = trading_days[-1]
    grouped = daily.groupby("ts_code", sort=False)
    daily["ret"] = grouped["q_close"].pct_change(fill_method=None)
    daily["r_rev"] = daily["q_close"] / grouped["q_close"].shift(REVERSAL_DAYS) - 1.0
    cross = daily[daily["trade_date"] == latest_day].copy()
    closes = cross.set_index("ts_code")["close"]
    cross["ivol"] = cross["ts_code"].map(residual_volatility(daily, trading_days, read_index_returns(context)))
    section_frame = members.merge(cross, on="ts_code")
    section_frame = section_frame.merge(latest_quality(read_statements(context), trading_days, latest_day),
                                        on="ts_code")

    universe = pd.read_parquet(context.asof_dir + "/universe", columns=["ts_code", "name", "l1_name"])
    info = universe.drop_duplicates("ts_code").set_index("ts_code").reindex(section_frame["ts_code"])
    names = info["name"].fillna("").astype(str)
    section_frame["industry"] = info["l1_name"].fillna("未分类").astype(str).to_numpy()
    keep = (
        ~section_frame["ts_code"].str.startswith(("688", "689")).to_numpy()
        & ~names.str.contains("ST|退").to_numpy()
        & ~section_frame["is_suspended"].fillna(False).astype(bool).to_numpy()
        & section_frame[["ivol", "r_rev", "accr", "ocf_ta", "circ_mv"]].notna().all(axis=1).to_numpy()
    )
    section_frame = section_frame[keep].reset_index(drop=True)
    if len(section_frame) < MIN_POOL:
        return None, closes, section

    pe = section_frame["pe_ttm"].where(section_frame["pe_ttm"] > 0)
    pb = section_frame["pb"].where(section_frame["pb"] > 0)
    legs = {
        "quality": (1.0 - pct_rank(section_frame["accr"]) + pct_rank(section_frame["ocf_ta"])) / 2.0,
        "lowvol": 1.0 - pct_rank(section_frame["ivol"]),
        "reversal": 1.0 - pct_rank(section_frame["r_rev"]),
        "value": (pct_rank((1.0 / pe).fillna(0.0)) + pct_rank((1.0 / pb).fillna(0.0))) / 2.0,
    }
    raw = sum(pct_rank(legs[name]) for name in LEGS).to_numpy() / len(LEGS)
    size = pct_rank(np.log(section_frame["circ_mv"].clip(lower=1.0))).to_numpy()
    design = np.column_stack([np.ones(len(section_frame)), size])
    coef, _, _, _ = np.linalg.lstsq(design, raw, rcond=None)
    section_frame["score"] = raw - design @ coef
    return section_frame[["ts_code", "industry", "weight", "close", "score"]], closes, section
