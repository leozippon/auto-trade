"""The PIT-safe daily cross-section and the four book scores.

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

    gp_overlay     z of annualised (total_revenue − oper_cost) / total_assets.
                   Missing oper_cost uses operate_profit / total_assets.
                   No EP/BP.
    cma_overlay    z of −asset year-over-year. Prefer last-year same-quarter
                   total_assets; else fina_indicator_vip.assets_yoy / 100.
                   No profit or accrual overlay.

Pool / allocated books drop names that cannot be scored.

    noa_index      NOA = (TA − money_cap) − (TA − st_borr − lt_borr
                   − minority_int − total_hldr_eqy_exc_min_int);
                   score = −NOA / lagged TA. Missing NOA columns fail.
                   Seats follow index L1 weights.
    cashdiv_pool   visible dividend.cash_div summed over the last 365 days
                   of already-ex or already-paid records, divided by close
                   (a trailing per-share yield, not cash_div / circ_mv).
                   Not an earnings or OCF yield.

Units: daily prices CNY/share; `circ_mv` CNY; fundamentals CNY;
`fina_indicator_vip` ratios and `assets_yoy` are percent numbers;
`dividend.cash_div` is CNY/share; `macro.index_weight.weight` is a percent
number.
"""

from datetime import timedelta

import numpy as np
import pandas as pd

OVERLAY_K = 0.5
OVERLAY_CLIP = 2.5
STATE_MAX_AGE = 130
DAILY_LOOKBACK_DAYS = 220
STATEMENT_LOOKBACK_DAYS = 800
PERIOD_RECENCY_DAYS = 400
DIV_WINDOW_DAYS = 365
ASSET_FLOOR = 1.0

DAILY_COLUMNS = ("ts_code", "trade_date", "close", "circ_mv", "is_suspended")
FUND_COLUMNS = (
    "dataset", "available_at", "ts_code", "end_date", "ann_date", "report_type",
    "total_revenue", "oper_cost", "operate_profit", "total_assets", "assets_yoy",
    "money_cap", "st_borr", "lt_borr", "minority_int", "total_hldr_eqy_exc_min_int",
)
NOA_COLUMNS = (
    "total_assets", "money_cap", "st_borr", "lt_borr",
    "minority_int", "total_hldr_eqy_exc_min_int",
)
DIVIDEND_COLUMNS = (
    "dataset", "available_at", "ts_code", "end_date", "cash_div", "ex_date", "pay_date",
)
UNIVERSE_COLUMNS = ("ts_code", "name", "l1_name")
STATEMENT_DATASETS = ("income_vip", "balancesheet_vip")


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


def _zscore(values):
    series = values.astype("float64").replace([np.inf, -np.inf], np.nan)
    if series.notna().sum() <= 1:
        return pd.Series(np.where(series.notna(), 0.0, np.nan), index=series.index)
    mu = float(series.mean(skipna=True))
    sd = float(series.std(skipna=True, ddof=0))
    if not np.isfinite(sd) or sd <= 0:
        return pd.Series(np.where(series.notna(), 0.0, np.nan), index=series.index)
    return (series - mu) / sd


def _prev_year(end_date):
    text = str(end_date)
    if len(text) != 8 or not text.isdigit():
        return ""
    return f"{int(text[:4]) - 1}{text[4:]}"


def _age_cap(latest, trading_days, latest_day):
    days = np.array(trading_days)
    ann_day = latest["stamp"].dt.tz_convert("Asia/Shanghai").dt.strftime("%Y%m%d").to_numpy()
    idx = np.searchsorted(days, ann_day, side="left")
    latest_idx = int(np.searchsorted(days, latest_day, side="left"))
    latest = latest.copy()
    latest["age"] = np.where(ann_day < days[0], len(days), latest_idx - idx)
    return latest[(latest["age"] >= 0) & (latest["age"] <= STATE_MAX_AGE)]


def read_daily(context):
    start = (context.inference_at - timedelta(days=DAILY_LOOKBACK_DAYS)).strftime("%Y%m%d")
    frame = _read_parquet(
        context.asof_dir + "/daily",
        DAILY_COLUMNS,
        filters=[("trade_date", ">=", start)],
    )
    frame = frame[frame["close"] > 0]
    frame = frame.assign(trade_date=frame["trade_date"].astype(str))
    return frame.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)


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
    missing = [name for name in columns if name not in part.columns]
    if missing:
        raise RuntimeError(f"{dataset} is missing columns {missing}")
    if dataset in STATEMENT_DATASETS:
        part = part[part["report_type"].astype(str) == "1"]
    if dataset == "income_vip":
        part = part[part["total_revenue"].notna() | part["operate_profit"].notna()]
    elif dataset == "balancesheet_vip":
        part = part[part["total_assets"].notna()]
    elif dataset == "fina_indicator_vip":
        part = part[part["assets_yoy"].notna()]
    if part.empty:
        return pd.DataFrame(columns=["ts_code", "end_date"] + list(columns))
    part = part.sort_values("stamp").drop_duplicates(["ts_code", "end_date"], keep="first")
    return part[["ts_code", "end_date", "stamp"] + list(columns)].rename(
        columns={"stamp": "stamp_" + dataset}
    )


def _latest_join(parts, trading_days, latest_day):
    joined = None
    for part in parts:
        joined = part if joined is None else joined.merge(part, on=["ts_code", "end_date"], how="outer")
    if joined is None or joined.empty:
        return pd.DataFrame(columns=["ts_code", "end_date", "stamp"])
    stamp_cols = [name for name in joined.columns if name.startswith("stamp_")]
    joined["stamp"] = joined[stamp_cols].max(axis=1)
    latest = joined.dropna(subset=["stamp"]).sort_values("stamp").drop_duplicates("ts_code", keep="last")
    return _age_cap(latest, trading_days, latest_day)


def _annualiser(end_date):
    qn = end_date.astype(str).str[4:6].map({"03": 1, "06": 2, "09": 3, "12": 4}).astype("float64")
    return np.where(qn > 0, 4.0 / qn, np.nan)


def latest_gp(statements, trading_days, latest_day):
    """Newest visible gross-profit / assets, annualised. Age-capped."""
    income = _first_version(statements, "income_vip", ["total_revenue", "oper_cost", "operate_profit"])
    balance = _first_version(statements, "balancesheet_vip", ["total_assets"])
    latest = _latest_join([income, balance], trading_days, latest_day)
    if latest.empty:
        return pd.DataFrame(columns=["ts_code", "gp"])
    assets = pd.to_numeric(latest["total_assets"], errors="coerce").clip(lower=ASSET_FLOOR)
    revenue = pd.to_numeric(latest["total_revenue"], errors="coerce")
    cost = pd.to_numeric(latest["oper_cost"], errors="coerce")
    profit = pd.to_numeric(latest["operate_profit"], errors="coerce")
    ann = _annualiser(latest["end_date"])
    gp_rev = np.where(
        revenue.notna() & cost.notna() & assets.notna() & np.isfinite(ann),
        (revenue - cost) / assets * ann,
        np.nan,
    )
    gp_op = np.where(
        profit.notna() & assets.notna() & np.isfinite(ann),
        profit / assets * ann,
        np.nan,
    )
    latest["gp"] = np.where(np.isfinite(gp_rev), gp_rev, gp_op)
    return latest[["ts_code", "gp"]]


def latest_cma(statements, trading_days, latest_day):
    """Newest visible −asset growth. Same-quarter TA, else assets_yoy / 100."""
    balance = _first_version(statements, "balancesheet_vip", ["total_assets"])
    indica = _first_version(statements, "fina_indicator_vip", ["assets_yoy"])
    latest = _latest_join([balance, indica], trading_days, latest_day)
    if latest.empty:
        return pd.DataFrame(columns=["ts_code", "ag"])
    hist = balance[["ts_code", "end_date", "total_assets"]].rename(
        columns={"end_date": "prev_end", "total_assets": "assets_prev"}
    )
    latest = latest.copy()
    latest["prev_end"] = latest["end_date"].map(_prev_year)
    latest = latest.merge(hist, on=["ts_code", "prev_end"], how="left")
    here = pd.to_numeric(latest["total_assets"], errors="coerce")
    prev = pd.to_numeric(latest["assets_prev"], errors="coerce")
    ag_stmt = np.where((here > 0) & (prev > 0), here / prev - 1.0, np.nan)
    ag_yoy = pd.to_numeric(latest["assets_yoy"], errors="coerce") / 100.0
    latest["ag"] = np.where(np.isfinite(ag_stmt), ag_stmt, ag_yoy)
    return latest[["ts_code", "ag"]]


def latest_noa(statements, trading_days, latest_day):
    """Newest visible −NOA / lagged TA. Missing NOA columns raise."""
    missing = [name for name in NOA_COLUMNS if name not in statements.columns]
    if missing:
        raise RuntimeError(f"balancesheet_vip is missing columns {missing}")
    balance = _first_version(statements, "balancesheet_vip", list(NOA_COLUMNS))
    latest = _latest_join([balance], trading_days, latest_day)
    if latest.empty:
        return pd.DataFrame(columns=["ts_code", "score"])
    hist = balance[["ts_code", "end_date", "total_assets"]].rename(
        columns={"end_date": "prev_end", "total_assets": "assets_prev"}
    )
    latest = latest.copy()
    latest["prev_end"] = latest["end_date"].map(_prev_year)
    latest = latest.merge(hist, on=["ts_code", "prev_end"], how="left")
    ta = pd.to_numeric(latest["total_assets"], errors="coerce")
    cash = pd.to_numeric(latest["money_cap"], errors="coerce")
    st = pd.to_numeric(latest["st_borr"], errors="coerce")
    lt = pd.to_numeric(latest["lt_borr"], errors="coerce")
    minority = pd.to_numeric(latest["minority_int"], errors="coerce")
    equity = pd.to_numeric(latest["total_hldr_eqy_exc_min_int"], errors="coerce")
    lagged = pd.to_numeric(latest["assets_prev"], errors="coerce")
    operating_assets = ta - cash
    operating_liab = ta - st - lt - minority - equity
    noa = operating_assets - operating_liab
    latest["score"] = np.where(
        ta.notna() & cash.notna() & st.notna() & lt.notna()
        & minority.notna() & equity.notna() & (lagged > 0),
        -noa / lagged,
        np.nan,
    )
    return latest[["ts_code", "score"]]


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
    require_industry(frame, "CSI 300 section")
    return frame


def gp_overlay(context, pool, members, trading_days):
    frame = _members(pool, members)
    gross = latest_gp(read_fundamentals(context), trading_days, trading_days[-1])
    frame = frame.merge(gross, on="ts_code", how="left")
    frame["score"] = pd.to_numeric(frame["gp"], errors="coerce")
    return overlay_weights(frame)


def cma_overlay(context, pool, members, trading_days):
    frame = _members(pool, members)
    growth = latest_cma(read_fundamentals(context), trading_days, trading_days[-1])
    frame = frame.merge(growth, on="ts_code", how="left")
    frame["score"] = -pd.to_numeric(frame["ag"], errors="coerce")
    return overlay_weights(frame)


def noa_index(context, pool, members, trading_days, seats):
    """Constituents only. Industry seats follow index weights; −NOA fills the quota."""
    frame = _members(pool, members)
    scored = latest_noa(read_fundamentals(context), trading_days, trading_days[-1])
    frame = frame.merge(scored, on="ts_code", how="left")
    frame["_ind"] = frame["industry"].astype(str)
    picked_src = frame[np.isfinite(frame["score"].to_numpy(dtype="float64"))].copy()
    if picked_src.empty:
        raise RuntimeError("no NOA scores on the visible constituent section")
    industry_w = frame.groupby("_ind", sort=False)["weight"].sum()
    alloc = allocate_seats(industry_w, seats)
    picked = []
    used = set()
    for industry, count in alloc.items():
        chunk = picked_src[picked_src["_ind"] == industry].sort_values(
            ["score", "ts_code"], ascending=[False, True]
        )
        take = chunk.head(int(count))
        if take.empty:
            continue
        picked.append(take)
        used.update(take["ts_code"].tolist())
    if not picked:
        raise RuntimeError("index-weight seat allocation produced no scored constituents")
    out = pd.concat(picked, ignore_index=True)
    if len(out) < seats:
        rest = picked_src[~picked_src["ts_code"].isin(used)].sort_values(
            ["score", "ts_code"], ascending=[False, True]
        )
        extra = rest.head(seats - len(out))
        if len(out) + len(extra) < seats:
            raise RuntimeError(f"only {len(out) + len(extra)} scored constituents for a {seats}-seat noa book")
        out = pd.concat([out, extra], ignore_index=True)
    out = out.drop_duplicates("ts_code").head(seats).copy()
    out["index_weight_pct"] = out["weight"].astype("float64")
    out["w"] = 1.0 / len(out)
    return out.sort_values(["score", "ts_code"], ascending=[False, True]).reset_index(drop=True)


def cashdiv_score(context, pool):
    """Trailing 365-day visible cash_div / close. Already-ex or already-paid only."""
    frame = _read_parquet(
        context.asof_dir + "/fundamentals",
        DIVIDEND_COLUMNS,
        filters=[("dataset", "=", "dividend")],
    )
    frame = _visible(frame, context).copy()
    if frame.empty:
        raise RuntimeError("no visible dividend rows")
    decision = pd.Timestamp(context.inference_at).tz_convert("Asia/Shanghai").tz_localize(None).normalize()
    decision_key = decision.strftime("%Y%m%d")
    ex = frame["ex_date"].astype("string").str.strip().fillna("")
    pay = frame["pay_date"].astype("string").str.strip().fillna("")
    ex_ok = ex.str.match(r"^\d{8}$").fillna(False) & (ex <= decision_key)
    pay_ok = pay.str.match(r"^\d{8}$").fillna(False) & (pay <= decision_key)
    keep = (ex_ok | pay_ok).to_numpy()
    frame = frame.loc[keep].copy()
    event = np.where(ex_ok.to_numpy()[keep], ex.to_numpy()[keep], pay.to_numpy()[keep])
    event_ts = pd.to_datetime(event, format="%Y%m%d", errors="coerce")
    frame = frame.loc[(event_ts >= decision - pd.Timedelta(days=DIV_WINDOW_DAYS)) & event_ts.notna()].copy()
    frame["cash_div"] = pd.to_numeric(frame["cash_div"], errors="coerce")
    frame = frame[frame["cash_div"].notna()]
    if frame.empty:
        raise RuntimeError("no already-ex or already-paid cash dividends in the last 365 days")
    frame["stamp"] = pd.to_datetime(frame["available_at"], utc=True)
    frame["end_date"] = frame["end_date"].astype(str)
    frame = frame.sort_values("stamp").drop_duplicates(["ts_code", "end_date"], keep="last")
    trail = frame.groupby("ts_code", sort=False)["cash_div"].sum()
    out = pool.copy()
    px = pd.to_numeric(out["close"], errors="coerce")
    out["score"] = out["ts_code"].map(trail) / px
    out = out[np.isfinite(out["score"].to_numpy(dtype="float64")) & (px > 0)].reset_index(drop=True)
    if out.empty:
        raise RuntimeError("no cash-dividend scores on the affordable pool")
    return out
