"""The PIT-safe daily cross-section and the six book scores.

One builder serves every decision, so a cold worker and a warm one score the
same cross-section. Everything is read from the rolling as-of view with column
projection and date filters; nothing is fitted and nothing is cached across
calls. A requested column that is not on the file raises instead of substituting
another field.

`market` / `tradable` are shared. Affordability is applied in `lib/trade.py`
where the seat cash is known.

Overlay books keep every tradable constituent. A name with no score keeps its
benchmark weight (z = 0). The tilt is
`w ∝ bench * (1 + k * clip(z(score), ±2.5))`, then renormalised, k = 0.5.
There is no industry or name box on an overlay book.

    quality_overlay   0.50 z(OP) + 0.25 z(EQ) + 0.25 z(−AG). Financials
                      (银行 / 非银金融 / 金融服务) use only OP. No EP/BP.
    high52_overlay    adj_close / max(adj_high, 252 days excluding today),
                      minus the SW L1 median. No momentum overlay.
    rmax_overlay      −max(pct_chg) over the last 20 visible days. A limit-up
                      day mildly inflates that day's contribution to the max.
                      Not a 20-day cumulative return.

Pool / allocated books drop names that cannot be scored.

    pacc_index        (NI − OCF) / max(|NI|, 1), low better. Seats follow
                      index L1 weights; −pacc fills each industry's quota.
    net_issuance      s_adj = total_share / adj_factor (shares already);
                      score = −(log s_adj_t − log s_adj_{t−252}).
    lottery_reverse   mean of the 5 largest daily pct_chg in the last 25
                      visible days after dropping the most recent 5;
                      score = −rank within SW L1.

Units: daily prices CNY/share; `pct_chg` a decimal; `total_share` shares
(snapshot-normalised from 10k shares); fundamentals CNY; `fina_indicator_vip`
ratios and `assets_yoy` are percent numbers; `macro.index_weight.weight` is a
percent number.
"""

from datetime import timedelta

import numpy as np
import pandas as pd

OVERLAY_K = 0.5
OVERLAY_CLIP = 2.5
HIGH52_WINDOW = 252
HIGH52_MIN_DAYS = 200
RMAX_WINDOW = 20
RMAX_MIN_DAYS = 15
LIMIT_UP_BUMP = 1.10
LOTTERY_SPAN = 25
LOTTERY_DROP = 5
LOTTERY_TAIL = 5
ISSUE_WINDOW = 252
STATE_MAX_AGE = 130
DAILY_LOOKBACK_DAYS = 430
STATEMENT_LOOKBACK_DAYS = 800
PERIOD_RECENCY_DAYS = 400
ASSET_FLOOR = 1.0
FINANCIALS = frozenset({"银行", "非银金融", "金融服务"})

DAILY_COLUMNS = (
    "ts_code", "trade_date", "close", "high", "adj_factor",
    "pct_chg", "up_limit", "total_share", "is_suspended",
)
FUND_COLUMNS = (
    "dataset", "available_at", "ts_code", "end_date", "ann_date", "report_type",
    "operate_profit", "n_income_attr_p", "n_cashflow_act", "total_assets",
    "assets_yoy", "roa",
)
UNIVERSE_COLUMNS = ("ts_code", "name", "l1_name")
STATEMENT_DATASETS = ("income_vip", "cashflow_vip", "balancesheet_vip")


def _read_parquet(path, columns, filters=None):
    frame = pd.read_parquet(path, columns=list(columns), filters=filters)
    missing = [name for name in columns if name not in frame.columns]
    if missing:
        raise RuntimeError(f"{path} is missing columns {missing}")
    return frame


def _visible(frame, context):
    stamp = pd.to_datetime(frame["available_at"], utc=True)
    return frame[stamp <= pd.Timestamp(context.inference_at).tz_convert("UTC")]


def require_industry(frame, what):
    industry = frame["industry"].astype(str)
    if industry.empty or industry.eq("未分类").all():
        raise RuntimeError(f"no visible SW L1 industry on the {what}; refusing a silent whole-market fallback")


def _is_financial(industry):
    name = str(industry)
    return name in FINANCIALS or ("金融" in name and name != "未分类")


def _zscore(values):
    series = values.astype("float64").replace([np.inf, -np.inf], np.nan)
    if series.notna().sum() <= 1:
        return pd.Series(np.where(series.notna(), 0.0, np.nan), index=series.index)
    mu = float(series.mean(skipna=True))
    sd = float(series.std(skipna=True, ddof=0))
    if not np.isfinite(sd) or sd <= 0:
        return pd.Series(np.where(series.notna(), 0.0, np.nan), index=series.index)
    return (series - mu) / sd


def _at_limit_up(close, up_limit, pct_chg):
    close = pd.to_numeric(close, errors="coerce")
    up_limit = pd.to_numeric(up_limit, errors="coerce")
    pct_chg = pd.to_numeric(pct_chg, errors="coerce")
    return (
        up_limit.notna()
        & (up_limit > 0)
        & close.notna()
        & (close >= up_limit * 0.995)
        & (pct_chg > 0)
    )


def _prev_year(end_date):
    text = str(end_date)
    if len(text) != 8 or not text.isdigit():
        return ""
    return f"{int(text[:4]) - 1}{text[4:]}"


def read_daily(context):
    start = (context.inference_at - timedelta(days=DAILY_LOOKBACK_DAYS)).strftime("%Y%m%d")
    frame = _read_parquet(
        context.asof_dir + "/daily",
        DAILY_COLUMNS,
        filters=[("trade_date", ">=", start)],
    )
    frame = frame[(frame["close"] > 0) & frame["adj_factor"].notna()]
    frame = frame.assign(trade_date=frame["trade_date"].astype(str))
    frame = frame.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    anchor = frame.groupby("ts_code", sort=False)["adj_factor"].transform("last")
    scale = frame["adj_factor"] / anchor
    frame["q_close"] = frame["close"] * scale
    frame["q_high"] = pd.to_numeric(frame["high"], errors="coerce") * scale
    return frame


def read_fundamentals(context):
    """First-announced statements plus fina_indicator fallbacks, PIT by available_at."""
    start = (context.inference_at - timedelta(days=STATEMENT_LOOKBACK_DAYS)).strftime("%Y%m%d")
    frame = _read_parquet(
        context.asof_dir + "/fundamentals",
        FUND_COLUMNS,
        filters=[("dataset", "in", list(STATEMENT_DATASETS) + ["fina_indicator_vip"]),
                 ("ann_date", ">=", start)],
    )
    frame = _visible(frame, context).copy()
    frame["stamp"] = pd.to_datetime(frame["available_at"], utc=True)
    frame["end_date"] = frame["end_date"].astype(str)
    ann_day = frame["stamp"].dt.tz_convert("Asia/Shanghai").dt.tz_localize(None)
    period_end = pd.to_datetime(frame["end_date"], format="%Y%m%d", errors="coerce")
    frame = frame[period_end.notna() & (period_end >= ann_day - pd.Timedelta(days=PERIOD_RECENCY_DAYS))]
    return frame


def _first_version(frame, dataset, columns):
    part = frame[frame["dataset"] == dataset].copy()
    if dataset in STATEMENT_DATASETS:
        part = part[part["report_type"].astype(str) == "1"]
    keep = [name for name in columns if name in part.columns]
    if dataset == "income_vip":
        part = part[part["operate_profit"].notna() | part["n_income_attr_p"].notna()]
    elif dataset == "cashflow_vip":
        part = part[part["n_cashflow_act"].notna()]
    elif dataset == "balancesheet_vip":
        part = part[part["total_assets"].notna()]
    elif dataset == "fina_indicator_vip":
        part = part[part["assets_yoy"].notna() | part["roa"].notna()]
    if part.empty:
        return pd.DataFrame(columns=["ts_code", "end_date"] + keep)
    part = part.sort_values("stamp").drop_duplicates(["ts_code", "end_date"], keep="first")
    return part[["ts_code", "end_date", "stamp"] + keep].rename(columns={"stamp": "stamp_" + dataset})


def latest_quality(statements, trading_days, latest_day):
    """Newest visible (ts_code) quality pillars: op, eq, ag. Age-capped."""
    income = _first_version(statements, "income_vip", ["operate_profit", "n_income_attr_p"])
    cash = _first_version(statements, "cashflow_vip", ["n_cashflow_act"])
    balance = _first_version(statements, "balancesheet_vip", ["total_assets"])
    indica = _first_version(statements, "fina_indicator_vip", ["assets_yoy", "roa"])
    joined = income.merge(cash, on=["ts_code", "end_date"], how="outer")
    joined = joined.merge(balance, on=["ts_code", "end_date"], how="outer")
    joined = joined.merge(indica, on=["ts_code", "end_date"], how="outer")
    if joined.empty:
        return pd.DataFrame(columns=["ts_code", "op", "eq", "ag", "n_income_attr_p", "n_cashflow_act"])
    stamp_cols = [name for name in joined.columns if name.startswith("stamp_")]
    joined["stamp"] = joined[stamp_cols].max(axis=1)
    joined["qn"] = joined["end_date"].str[4:6].map({"03": 1, "06": 2, "09": 3, "12": 4})
    assets = pd.to_numeric(joined["total_assets"], errors="coerce").clip(lower=ASSET_FLOOR)
    op_raw = pd.to_numeric(joined["operate_profit"], errors="coerce")
    ni = pd.to_numeric(joined["n_income_attr_p"], errors="coerce")
    ocf = pd.to_numeric(joined["n_cashflow_act"], errors="coerce")
    roa = pd.to_numeric(joined["roa"], errors="coerce")
    qn = joined["qn"].astype("float64")
    ann = np.where(qn > 0, 4.0 / qn, np.nan)
    joined["op"] = np.where(op_raw.notna() & assets.notna() & np.isfinite(ann), op_raw / assets * ann, roa / 100.0)
    eq_flow = np.where(ocf.notna() & op_raw.notna() & assets.notna() & np.isfinite(ann),
                       (ocf - op_raw) / assets * ann, np.nan)
    acc = np.where(ni.notna() & ocf.notna() & assets.notna(), -(ni - ocf) / assets, np.nan)
    joined["eq"] = np.where(np.isfinite(eq_flow), eq_flow, acc)
    latest = joined.dropna(subset=["stamp"]).sort_values("stamp").drop_duplicates("ts_code", keep="last").copy()
    hist = balance[["ts_code", "end_date", "total_assets"]].rename(
        columns={"end_date": "prev_end", "total_assets": "assets_prev"}
    )
    latest["prev_end"] = latest["end_date"].map(_prev_year)
    latest = latest.merge(hist, on=["ts_code", "prev_end"], how="left")
    here = pd.to_numeric(latest["total_assets"], errors="coerce")
    prev = pd.to_numeric(latest["assets_prev"], errors="coerce")
    ag_stmt = np.where((here > 0) & (prev > 0), here / prev - 1.0, np.nan)
    ag_yoy = pd.to_numeric(latest["assets_yoy"], errors="coerce") / 100.0
    latest["ag"] = np.where(np.isfinite(ag_stmt), ag_stmt, ag_yoy)
    days = np.array(trading_days)
    ann_day = latest["stamp"].dt.tz_convert("Asia/Shanghai").dt.strftime("%Y%m%d").to_numpy()
    idx = np.searchsorted(days, ann_day, side="left")
    latest_idx = int(np.searchsorted(days, latest_day, side="left"))
    latest["age"] = np.where(ann_day < days[0], len(days), latest_idx - idx)
    latest = latest[(latest["age"] >= 0) & (latest["age"] <= STATE_MAX_AGE)]
    return latest[["ts_code", "op", "eq", "ag", "n_income_attr_p", "n_cashflow_act"]]


def market(context):
    """(T-1 cross-section, closes, the daily window, its trading days)."""
    daily = read_daily(context)
    if daily.empty:
        raise RuntimeError("the daily window of the as-of view is empty")
    trading_days = sorted(daily["trade_date"].unique())
    latest_day = trading_days[-1]
    cross = daily[daily["trade_date"] == latest_day].copy()
    closes = cross.set_index("ts_code")["close"]
    universe = _read_parquet(context.asof_dir + "/universe", UNIVERSE_COLUMNS)
    info = universe.drop_duplicates("ts_code").set_index("ts_code").reindex(cross["ts_code"])
    cross["name"] = info["name"].fillna("").astype(str).to_numpy()
    cross["industry"] = info["l1_name"].fillna("未分类").astype(str).to_numpy()
    cross["at_limit_up"] = _at_limit_up(cross["close"], cross["up_limit"], cross["pct_chg"]).to_numpy()
    return cross.reset_index(drop=True), closes, daily, trading_days


def tradable(cross):
    """Names this account can trade at all, before any seat or signal enters."""
    keep = (
        ~cross["ts_code"].str.startswith(("688", "689")).to_numpy()
        & ~cross["ts_code"].str.endswith(".BJ").to_numpy()
        & ~cross["name"].str.contains("ST|退").to_numpy()
        & ~cross["is_suspended"].fillna(False).astype(bool).to_numpy()
    )
    return cross[keep].reset_index(drop=True)


def overlay_weights(members):
    """Benchmark weight times (1 + k clip z), missing score → z = 0, then normalise."""
    frame = members.copy()
    frame["index_weight_pct"] = frame["weight"].astype("float64")
    z = _zscore(frame["score"]).fillna(0.0)
    tilt = 1.0 + OVERLAY_K * z.clip(-OVERLAY_CLIP, OVERLAY_CLIP)
    raw = frame["index_weight_pct"] / 100.0 * tilt.to_numpy()
    total = float(np.nansum(raw))
    if not np.isfinite(total) or total <= 0:
        raise RuntimeError("overlay weights summed to 0")
    frame["w"] = raw / total
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


def _members(pool, members):
    frame = pool.merge(members, on="ts_code")
    frame = frame[frame["weight"] > 0].copy()
    if frame.empty:
        raise RuntimeError("no tradable CSI 300 constituents visible")
    return frame


def quality_overlay(context, pool, members, trading_days):
    frame = _members(pool, members)
    quality = latest_quality(read_fundamentals(context), trading_days, trading_days[-1])
    frame = frame.merge(quality, on="ts_code", how="left")
    financial = frame["industry"].map(_is_financial).to_numpy()
    z_op = _zscore(frame["op"])
    non_fin = ~financial
    z_eq = pd.Series(np.nan, index=frame.index)
    z_nag = pd.Series(np.nan, index=frame.index)
    if non_fin.any():
        z_eq.loc[non_fin] = _zscore(frame.loc[non_fin, "eq"]).to_numpy()
        z_nag.loc[non_fin] = _zscore(-frame.loc[non_fin, "ag"]).to_numpy()
    combo = 0.50 * z_op.fillna(0.0) + 0.25 * z_eq.fillna(0.0) + 0.25 * z_nag.fillna(0.0)
    has = z_op.notna() | z_eq.notna() | z_nag.notna()
    frame["score"] = np.where(financial, z_op, np.where(has, combo, np.nan))
    return overlay_weights(frame)


def high52_overlay(pool, members, daily, trading_days):
    frame = _members(pool, members)
    if "high" not in daily.columns:
        raise RuntimeError("daily.high is required for high52_overlay")
    prior = trading_days[-HIGH52_WINDOW - 1:-1]
    if len(prior) < HIGH52_MIN_DAYS:
        raise RuntimeError(f"only {len(prior)} prior trading days for a 252-day high")
    rows = daily[daily["trade_date"].isin(prior)]
    peak = rows.groupby("ts_code", sort=False)["q_high"].max()
    counts = rows.groupby("ts_code", sort=False)["q_high"].count()
    peak = peak.where(counts >= HIGH52_MIN_DAYS)
    latest = daily[daily["trade_date"] == trading_days[-1]].drop_duplicates("ts_code").set_index("ts_code")
    near = latest["q_close"] / peak.replace(0.0, np.nan)
    frame["near"] = frame["ts_code"].map(near)
    med = frame.groupby("industry", sort=False)["near"].transform("median")
    frame["score"] = frame["near"] - med
    return overlay_weights(frame)


def rmax_overlay(pool, members, daily, trading_days):
    frame = _members(pool, members)
    window = trading_days[-RMAX_WINDOW:]
    rows = daily[daily["trade_date"].isin(window)].copy()
    rows["r"] = pd.to_numeric(rows["pct_chg"], errors="coerce")
    if "up_limit" in rows.columns:
        bump = _at_limit_up(rows["close"], rows["up_limit"], rows["r"])
        rows["r"] = np.where(bump, rows["r"] * LIMIT_UP_BUMP, rows["r"])
    panel = rows.pivot_table(index="trade_date", columns="ts_code", values="r", aggfunc="last").reindex(window)
    values = panel.to_numpy(dtype="float64")
    n = np.isfinite(values).sum(axis=0)
    filled = np.where(np.isfinite(values), values, -np.inf)
    peak = np.where(n >= RMAX_MIN_DAYS, filled.max(axis=0), np.nan)
    score = pd.Series(np.where(n >= RMAX_MIN_DAYS, -peak, np.nan), index=panel.columns)
    frame["score"] = frame["ts_code"].map(score)
    return overlay_weights(frame)


def pacc_index(context, pool, members, trading_days, seats):
    """Constituents only. Industry seats follow index weights; −pacc fills the quota."""
    frame = _members(pool, members)
    require_industry(frame, "CSI 300 section")
    quality = latest_quality(read_fundamentals(context), trading_days, trading_days[-1])
    frame = frame.merge(quality, on="ts_code", how="left")
    ni = pd.to_numeric(frame["n_income_attr_p"], errors="coerce")
    ocf = pd.to_numeric(frame["n_cashflow_act"], errors="coerce")
    pacc = (ni - ocf) / np.maximum(np.abs(ni), 1.0)
    frame["pacc"] = pacc
    frame["score"] = -pacc
    frame["_ind"] = frame["industry"].astype(str)
    scored = frame[np.isfinite(frame["score"].to_numpy(dtype="float64"))].copy()
    if scored.empty:
        raise RuntimeError("no percent-accrual scores on the visible constituent section")
    industry_w = frame.groupby("_ind", sort=False)["weight"].sum()
    alloc = allocate_seats(industry_w, seats)
    picked = []
    used = set()
    for industry, count in alloc.items():
        chunk = scored[scored["_ind"] == industry].sort_values(["score", "ts_code"], ascending=[False, True])
        take = chunk.head(int(count))
        if take.empty:
            continue
        picked.append(take)
        used.update(take["ts_code"].tolist())
    if not picked:
        raise RuntimeError("index-weight seat allocation produced no scored constituents")
    out = pd.concat(picked, ignore_index=True)
    if len(out) < seats:
        rest = scored[~scored["ts_code"].isin(used)].sort_values(["score", "ts_code"], ascending=[False, True])
        need = seats - len(out)
        extra = rest.head(need)
        if len(out) + len(extra) < seats:
            raise RuntimeError(f"only {len(out) + len(extra)} scored constituents for a {seats}-seat pacc book")
        out = pd.concat([out, extra], ignore_index=True)
    out = out.drop_duplicates("ts_code").head(seats).copy()
    out["index_weight_pct"] = out["weight"].astype("float64")
    out["w"] = 1.0 / len(out)
    return out.sort_values(["score", "ts_code"], ascending=[False, True]).reset_index(drop=True)


def net_issuance_score(pool, daily, trading_days):
    if len(trading_days) <= ISSUE_WINDOW:
        raise RuntimeError(f"only {len(trading_days)} trading days for a 252-day issuance window")
    latest_day = trading_days[-1]
    base_day = trading_days[-1 - ISSUE_WINDOW]
    now = daily[daily["trade_date"] == latest_day].drop_duplicates("ts_code").set_index("ts_code")
    then = daily[daily["trade_date"] == base_day].drop_duplicates("ts_code").set_index("ts_code")
    share_now = pd.to_numeric(now["total_share"], errors="coerce") / pd.to_numeric(now["adj_factor"], errors="coerce")
    share_then = pd.to_numeric(then["total_share"], errors="coerce") / pd.to_numeric(then["adj_factor"], errors="coerce")
    iss = np.log(share_now.where(share_now > 0)) - np.log(share_then.where(share_then > 0))
    frame = pool.copy()
    frame["score"] = frame["ts_code"].map(-iss)
    frame = frame[np.isfinite(frame["score"].to_numpy(dtype="float64"))].reset_index(drop=True)
    return frame


def lottery_reverse_score(pool, daily, trading_days):
    require_industry(pool, "tradable pool")
    if len(trading_days) < LOTTERY_SPAN:
        raise RuntimeError(f"only {len(trading_days)} trading days for the 25-day lottery window")
    window = trading_days[-LOTTERY_SPAN:-LOTTERY_DROP]
    rows = daily[daily["trade_date"].isin(window)].copy()
    rows["r"] = pd.to_numeric(rows["pct_chg"], errors="coerce")
    panel = rows.pivot_table(index="trade_date", columns="ts_code", values="r", aggfunc="last").reindex(window)
    values = panel.to_numpy(dtype="float64")
    finite = np.where(np.isfinite(values), values, -np.inf)
    order = np.sort(finite, axis=0)
    tail = order[-LOTTERY_TAIL:, :]
    n = np.isfinite(values).sum(axis=0)
    mean_tail = np.where(n >= LOTTERY_TAIL, np.mean(np.where(np.isfinite(tail), tail, np.nan), axis=0), np.nan)
    lottery = pd.Series(mean_tail, index=panel.columns)
    frame = pool.copy()
    frame["lottery"] = frame["ts_code"].map(lottery)
    frame = frame[np.isfinite(frame["lottery"].to_numpy(dtype="float64"))].copy()
    if frame.empty:
        raise RuntimeError("no lottery scores on the visible pool")
    require_industry(frame, "lottery-scored pool")
    frame["score"] = -frame.groupby("industry", sort=False)["lottery"].rank(pct=True)
    return frame.reset_index(drop=True)
