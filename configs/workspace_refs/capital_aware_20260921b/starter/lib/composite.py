"""The PIT-safe daily cross-section, and the four scores the book shapes use.

One builder serves every decision, so a cold worker and a warm one score the
same cross-section. Everything is read from the rolling as-of view with column
projection and date filters; nothing is fitted and nothing is cached across
calls.

`market` / `tradable` are shared. Affordability is applied in `lib/trade.py`
where the seat cash is known. Scores:

    residual_momentum  sum of daily residuals on the CSI 300 over MOM_WINDOW
                       visible days (default 90, the middle of 60-120). Not
                       a 20-day reversal.
    overlay_weights    benchmark weight times exp(OVERLAY_TILT * z(score));
                       a name with no score keeps its index weight (z = 0)
    indneutral_value   industry weights matched to the index; EP/BP ranks
                       pick names inside each industry
    eyield             mean of rank(1/pe_ttm) and rank(annualised OCF / circ_mv).
                       No PB.

Units: daily prices CNY/share, `amount` and `circ_mv` CNY, `pct_chg` a decimal;
fundamentals CNY; `macro.index_daily.pct_chg` and `index_weight.weight` are
percent numbers.
"""

from datetime import timedelta

import numpy as np
import pandas as pd

# Absolute import of the sibling package; relative imports are refused.
from lib import index

MOM_WINDOW = 90
MIN_MOM_DAYS = 60
OVERLAY_TILT = 0.5
ADV_WINDOW = 20
STATE_MAX_AGE = 130
DAILY_LOOKBACK_DAYS = 280
STATEMENT_LOOKBACK_DAYS = 400
PERIOD_RECENCY_DAYS = 400
INDEX_CODE = index.INDEX_CODE
ASSET_FLOOR = 1.0


def _visible(frame, context):
    stamp = pd.to_datetime(frame["available_at"], utc=True)
    return frame[stamp <= pd.Timestamp(context.inference_at).tz_convert("UTC")]


def read_daily(context):
    start = (context.inference_at - timedelta(days=DAILY_LOOKBACK_DAYS)).strftime("%Y%m%d")
    frame = pd.read_parquet(
        context.asof_dir + "/daily",
        columns=["ts_code", "trade_date", "close", "adj_factor", "amount", "circ_mv",
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


def read_cashflow(context):
    """First announced cash-flow statement visible at the decision."""
    start = (context.inference_at - timedelta(days=STATEMENT_LOOKBACK_DAYS)).strftime("%Y%m%d")
    frame = pd.read_parquet(
        context.asof_dir + "/fundamentals",
        columns=["dataset", "available_at", "ts_code", "end_date", "ann_date", "report_type",
                 "n_cashflow_act"],
        filters=[("dataset", "=", "cashflow_vip"), ("ann_date", ">=", start)],
    )
    frame = _visible(frame, context)
    frame = frame[frame["report_type"] == "1"].dropna(subset=["n_cashflow_act"]).copy()
    frame["stamp"] = pd.to_datetime(frame["available_at"], utc=True)
    frame["end_date"] = frame["end_date"].astype(str)
    ann_day = frame["stamp"].dt.tz_convert("Asia/Shanghai").dt.tz_localize(None)
    period_end = pd.to_datetime(frame["end_date"], format="%Y%m%d", errors="coerce")
    frame = frame[period_end >= ann_day - pd.Timedelta(days=PERIOD_RECENCY_DAYS)]
    frame = frame.sort_values("stamp").drop_duplicates(["ts_code", "end_date"], keep="first")
    return frame[["ts_code", "end_date", "stamp", "n_cashflow_act"]]


def latest_cashflow(statements, trading_days, latest_day):
    """Each name's newest cash-flow statement with quarter number and age."""
    frame = statements.copy()
    frame["qn"] = frame["end_date"].str[4:6].map({"03": 1, "06": 2, "09": 3, "12": 4})
    frame = frame.dropna(subset=["qn"])
    latest = frame.sort_values("stamp").drop_duplicates("ts_code", keep="last").copy()
    days = np.array(trading_days)
    ann_day = latest["stamp"].dt.tz_convert("Asia/Shanghai").dt.strftime("%Y%m%d").to_numpy()
    idx = np.searchsorted(days, ann_day, side="left")
    latest_idx = int(np.searchsorted(days, latest_day, side="left"))
    latest["age"] = np.where(ann_day < days[0], len(days), latest_idx - idx)
    latest = latest[(latest["age"] >= 0) & (latest["age"] <= STATE_MAX_AGE)]
    return latest[["ts_code", "n_cashflow_act", "qn"]]


def residual_momentum(daily, trading_days, market_returns):
    """MOM_WINDOW-day residual return on the CSI300 per name at the newest date."""
    window = trading_days[-MOM_WINDOW:]
    m = market_returns.reindex(window).to_numpy(dtype="float64")
    if np.isfinite(m).sum() < MIN_MOM_DAYS:
        raise RuntimeError(f"{INDEX_CODE} has {int(np.isfinite(m).sum())} visible days in the momentum window")
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
    beta = np.where(var_m > 0, cov / np.where(var_m > 0, var_m, 1.0), 0.0)
    resid = np.where(both, r - beta * m[:, None], 0.0)
    return pd.Series(np.where(n >= MIN_MOM_DAYS, resid.sum(axis=0), np.nan), index=returns.columns)


def pct_rank(values):
    return values.astype("float64").replace([np.inf, -np.inf], np.nan).rank(pct=True)


def _mean_ranks(*legs):
    stacked = np.column_stack([leg.to_numpy(dtype="float64") for leg in legs])
    n = np.isfinite(stacked).sum(axis=1)
    total = np.where(np.isfinite(stacked), stacked, 0.0).sum(axis=1)
    return np.where(n > 0, total / n, np.nan)


def market(context):
    """(T-1 cross-section of every priced name, its closes, the daily window, its trading days)."""
    daily = read_daily(context)
    if daily.empty:
        raise RuntimeError("the daily window of the as-of view is empty")
    trading_days = sorted(daily["trade_date"].unique())
    latest_day = trading_days[-1]
    grouped = daily.groupby("ts_code", sort=False)
    daily["ret"] = grouped["q_close"].pct_change(fill_method=None)
    cross = daily[daily["trade_date"] == latest_day].copy()
    closes = cross.set_index("ts_code")["close"]
    recent = daily[daily["trade_date"].isin(trading_days[-ADV_WINDOW:])]
    cross["adv"] = cross["ts_code"].map(recent.groupby("ts_code", sort=False)["amount"].mean())
    universe = pd.read_parquet(context.asof_dir + "/universe", columns=["ts_code", "name", "l1_name"])
    info = universe.drop_duplicates("ts_code").set_index("ts_code").reindex(cross["ts_code"])
    cross["name"] = info["name"].fillna("").astype(str).to_numpy()
    cross["industry"] = info["l1_name"].fillna("未分类").astype(str).to_numpy()
    return cross.reset_index(drop=True), closes, daily, trading_days


def tradable(cross):
    """The names this account can trade at all, before any seat or signal enters."""
    keep = (
        ~cross["ts_code"].str.startswith(("688", "689")).to_numpy()
        & ~cross["ts_code"].str.endswith(".BJ").to_numpy()
        & ~cross["name"].str.contains("ST|退").to_numpy()
        & ~cross["is_suspended"].fillna(False).astype(bool).to_numpy()
    )
    return cross[keep].reset_index(drop=True)


def overlay_weights(context, members, daily, trading_days):
    """Constituents with overlay target weight `w` and the raw index percent."""
    frame = members.copy()
    frame["index_weight_pct"] = frame["weight"].astype("float64")
    frame["score"] = frame["ts_code"].map(residual_momentum(daily, trading_days, read_index_returns(context)))
    s = frame["score"].astype("float64")
    mu = float(s.mean(skipna=True)) if s.notna().any() else 0.0
    sd = float(s.std(skipna=True, ddof=0)) if s.notna().sum() > 1 else 0.0
    z = ((s - mu) / sd) if sd > 0 else pd.Series(0.0, index=frame.index)
    frame["w"] = frame["index_weight_pct"] / 100.0 * np.exp(OVERLAY_TILT * z.fillna(0.0).to_numpy())
    return frame.sort_values(["w", "ts_code"], ascending=[False, True]).reset_index(drop=True)


def allocate_seats(weights, n_seats):
    """Largest-remainder seats so every industry with weight gets at least one when possible."""
    w = weights[weights > 0]
    if w.empty:
        raise RuntimeError("no industry weights on the visible constituent section")
    if len(w) >= n_seats:
        return pd.Series(1, index=w.nlargest(n_seats).index)
    rest = n_seats - len(w)
    quota = w / w.sum() * rest
    floors = np.floor(quota.to_numpy()).astype(int)
    leftover = rest - int(floors.sum())
    frac = quota.to_numpy() - floors
    order = np.argsort(-frac, kind="stable")
    floors[order[:leftover]] += 1
    return pd.Series(1 + floors, index=w.index)


def indneutral_value(members, seats):
    """Industry weights matched to the index; EP/BP pick names inside each industry."""
    frame = members.copy()
    frame["index_weight_pct"] = frame["weight"].astype("float64")
    frame["_ind"] = frame["industry"].fillna("未分类").astype(str)
    frame["ep"] = np.where(frame["pe_ttm"] > 0, 1.0 / frame["pe_ttm"], np.nan)
    frame["bp"] = np.where(frame["pb"] > 0, 1.0 / frame["pb"], np.nan)
    frame["score"] = _mean_ranks(
        frame.groupby("_ind", sort=False)["ep"].rank(pct=True),
        frame.groupby("_ind", sort=False)["bp"].rank(pct=True),
    )
    industry_w = frame.groupby("_ind", sort=False)["index_weight_pct"].sum()
    alloc = allocate_seats(industry_w, seats)
    picked = []
    for industry, count in alloc.items():
        chunk = frame[frame["_ind"] == industry]
        scored = chunk.dropna(subset=["score"]).sort_values(["score", "ts_code"], ascending=[False, True])
        if scored.empty:
            scored = chunk.sort_values(["index_weight_pct", "ts_code"], ascending=[False, True])
        if scored.empty:
            raise RuntimeError(f"industry {industry!r} has no tradable constituents")
        take = scored.head(int(count)).copy()
        take["w"] = (float(industry_w[industry]) / 100.0) / len(take)
        picked.append(take)
    out = pd.concat(picked, ignore_index=True)
    out["w"] = out["w"] / out["w"].sum()
    return out.sort_values(["w", "ts_code"], ascending=[False, True]).reset_index(drop=True)


def momentum_score(context, pool, daily, trading_days):
    """`pool` with a residual-momentum `score`; names that cannot be scored are dropped."""
    frame = pool.copy()
    frame["score"] = frame["ts_code"].map(residual_momentum(daily, trading_days, read_index_returns(context)))
    frame = frame[frame["score"].notna()].reset_index(drop=True)
    return frame


def eyield_score(context, pool, trading_days):
    """`pool` with an earnings-yield / cash-flow-yield `score`. No PB."""
    latest_day = trading_days[-1]
    frame = pool.merge(latest_cashflow(read_cashflow(context), trading_days, latest_day), on="ts_code", how="left")
    ep = np.where(frame["pe_ttm"] > 0, 1.0 / frame["pe_ttm"], np.nan)
    assets = frame["circ_mv"].clip(lower=ASSET_FLOOR)
    cfy = np.where(
        frame["n_cashflow_act"].notna() & frame["qn"].notna(),
        frame["n_cashflow_act"].to_numpy(dtype="float64") / assets.to_numpy(dtype="float64") * (4.0 / frame["qn"].to_numpy(dtype="float64")),
        np.nan,
    )
    frame["score"] = _mean_ranks(pct_rank(pd.Series(ep, index=frame.index)), pct_rank(pd.Series(cfy, index=frame.index)))
    frame = frame[np.isfinite(frame["score"].to_numpy(dtype="float64"))].reset_index(drop=True)
    return frame
