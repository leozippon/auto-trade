"""Return correlation with the industry's amount changes.

Current CSI 1000 members only. The score is the correlation of the stock's
daily return with the daily percent change in its Shenwan level-1 amount.
A higher correlation ranks higher. It is not the industry's price return.
"""

from datetime import timedelta

import pandas as pd

NAME = "m1"
INDEX = "000852.SH"
LOOKBACK_DAYS = 100
MIN_DAYS = 30
WEIGHT_COLUMNS = ["dataset", "available_at", "index_code", "con_code", "trade_date", "weight"]
DAILY_COLUMNS = ["ts_code", "trade_date", "pct_chg"]
SW_COLUMNS = ["dataset", "ts_code", "trade_date", "available_at", "amount"]
UNI_COLUMNS = ["ts_code", "l1_code"]


def _norm(code):
    text = "" if code is None or (isinstance(code, float) and pd.isna(code)) else str(code).strip()
    if not text or text == "nan":
        return ""
    if text.endswith(".SI"):
        return text
    if text.isdigit():
        return text + ".SI"
    return text


def _corr(left, right):
    paired = pd.DataFrame({"y": left, "x": right}).dropna()
    if len(paired) < MIN_DAYS:
        return None
    y = paired["y"] - paired["y"].mean()
    x = paired["x"] - paired["x"].mean()
    denom = float((x * x).sum() * (y * y).sum()) ** 0.5
    if denom <= 0:
        return None
    return float((x * y).sum()) / denom


def score(context, codes):
    start = (context.inference_at - timedelta(days=LOOKBACK_DAYS)).strftime("%Y%m%d")
    frame = pd.read_parquet(
        context.asof_dir + "/macro",
        columns=WEIGHT_COLUMNS,
        filters=[("dataset", "=", "index_weight"), ("index_code", "=", INDEX), ("trade_date", ">=", start)],
    )
    stamp = pd.to_datetime(frame["available_at"], utc=True)
    frame = frame[(stamp <= pd.Timestamp(context.inference_at).tz_convert("UTC")) & frame["con_code"].notna()]
    frame = frame.assign(con_code=frame["con_code"].astype(str), trade_date=frame["trade_date"].astype(str))
    frame = frame[pd.to_numeric(frame["weight"], errors="coerce") > 0]
    if frame.empty:
        raise RuntimeError("no visible CSI 1000 weights")
    latest = str(frame["trade_date"].max())
    members = set(frame.loc[frame["trade_date"] == latest, "con_code"])
    industry = pd.read_parquet(
        context.asof_dir + "/macro",
        columns=SW_COLUMNS,
        filters=[("dataset", "=", "sw_daily"), ("trade_date", ">=", start)],
    )
    seen = pd.to_datetime(industry["available_at"], utc=True)
    industry = industry[seen <= pd.Timestamp(context.inference_at).tz_convert("UTC")]
    industry = industry.assign(ts_code=industry["ts_code"].map(_norm), trade_date=industry["trade_date"].astype(str))
    industry = industry[industry["ts_code"].str.fullmatch(r"801\d\d0\.SI")]
    industry["amount"] = pd.to_numeric(industry["amount"], errors="coerce")
    changes = {}
    for code, group in industry.groupby("ts_code"):
        ordered = group.sort_values("trade_date").drop_duplicates("trade_date")
        amount = ordered["amount"].where(ordered["amount"] > 0)
        changes[code] = (amount / amount.shift(1) - 1.0).set_axis(ordered["trade_date"])
    if not changes:
        raise RuntimeError("no visible Shenwan amount")
    universe = pd.read_parquet(context.asof_dir + "/universe", columns=UNI_COLUMNS)
    info = universe.drop_duplicates("ts_code").assign(ts_code=lambda raw: raw["ts_code"].astype(str)).set_index("ts_code")
    home = info["l1_code"].map(_norm)
    bars = pd.read_parquet(context.asof_dir + "/daily", columns=DAILY_COLUMNS, filters=[("trade_date", ">=", start)])
    bars = bars.assign(
        trade_date=bars["trade_date"].astype(str),
        ts_code=bars["ts_code"].astype(str),
        ret=pd.to_numeric(bars["pct_chg"], errors="coerce"),
    )
    bars = bars[bars["ts_code"].isin(members)]
    values = {}
    for code, group in bars.groupby("ts_code"):
        series = changes.get(home.get(code, ""), None)
        if series is None:
            continue
        corr = _corr(group["ret"], group["trade_date"].map(series))
        if corr is None:
            continue
        values[code] = corr
    if len(values) < 12:
        raise RuntimeError("fewer than 12 CSI 1000 names have an industry-amount correlation")
    return pd.Series(values, dtype="float64").reindex(list(codes)), {"scored": len(values), "section": latest}


def shuffle(values, decision_at):
    import numpy as np

    scored = values.dropna().sort_index()
    day = int(pd.Timestamp(decision_at).strftime("%Y%m%d"))
    order = np.random.default_rng(day).permutation(len(scored))
    return pd.Series(scored.to_numpy()[order], index=scored.index)


def describe(values, book):
    picked = values.reindex(book).dropna()
    return {"book_score_span": [float(picked.min()), float(picked.max())] if len(picked) else None}
