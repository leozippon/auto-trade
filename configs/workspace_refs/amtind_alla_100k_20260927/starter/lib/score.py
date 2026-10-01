"""Whether a name's amount moves with its industry's amount.

The score is that correlation over 20 sessions. Names that trade when the
industry trades score higher. This is not a return correlation.
"""

from datetime import timedelta

import pandas as pd

NAME = "m1"
LOOKBACK_DAYS = 50
SESSIONS = 20
MIN_PAIRS = 15
UNI_COLUMNS = ["ts_code", "l1_code"]
DAILY_COLUMNS = ["ts_code", "trade_date", "amount"]
SW_COLUMNS = ["dataset", "ts_code", "trade_date", "available_at", "amount"]


def _norm(code):
    text = "" if code is None or (isinstance(code, float) and pd.isna(code)) else str(code).strip()
    if not text or text == "nan":
        return ""
    if text.endswith(".SI"):
        return text
    if text.isdigit():
        return text + ".SI"
    return text


def _corr(frame):
    pair = frame.dropna()
    if len(pair) < MIN_PAIRS:
        return float("nan")
    return float(pair["stock"].corr(pair["industry"]))


def score(context, codes):
    start = (context.inference_at - timedelta(days=LOOKBACK_DAYS)).strftime("%Y%m%d")
    bars = pd.read_parquet(context.asof_dir + "/daily", columns=DAILY_COLUMNS, filters=[("trade_date", ">=", start)])
    bars = bars.assign(trade_date=bars["trade_date"].astype(str), ts_code=bars["ts_code"].astype(str))
    dates = sorted(bars["trade_date"].unique())
    if len(dates) < MIN_PAIRS:
        raise RuntimeError("not enough sessions to correlate amount with the industry")
    use = set(dates[-SESSIONS:])
    window = bars[bars["trade_date"].isin(use)].copy()
    window["amount"] = pd.to_numeric(window["amount"], errors="coerce")
    macro = pd.read_parquet(
        context.asof_dir + "/macro",
        columns=SW_COLUMNS,
        filters=[("dataset", "=", "sw_daily"), ("trade_date", ">=", start)],
    )
    stamp = pd.to_datetime(macro["available_at"], utc=True)
    macro = macro[stamp <= pd.Timestamp(context.inference_at).tz_convert("UTC")]
    macro = macro.assign(ts_code=macro["ts_code"].map(_norm), trade_date=macro["trade_date"].astype(str))
    macro = macro[macro["ts_code"].str.fullmatch(r"801\d\d0\.SI") & macro["trade_date"].isin(use)]
    macro = macro.assign(amount=pd.to_numeric(macro["amount"], errors="coerce"))
    industry_amount = macro.dropna(subset=["amount"]).drop_duplicates(["ts_code", "trade_date"]).set_index(["ts_code", "trade_date"])["amount"]
    universe = pd.read_parquet(context.asof_dir + "/universe", columns=UNI_COLUMNS)
    info = universe.drop_duplicates("ts_code").set_index("ts_code")
    industry = info["l1_code"].map(_norm)
    window["industry"] = window["ts_code"].map(industry)
    window["ind_amount"] = [industry_amount.get((ind, day), float("nan")) for ind, day in zip(window["industry"], window["trade_date"])]
    aligned = window.rename(columns={"amount": "stock"})[["ts_code", "stock", "ind_amount"]].rename(columns={"ind_amount": "industry"})
    corr = aligned.groupby("ts_code")[["stock", "industry"]].apply(_corr, include_groups=False)
    values = corr.reindex(list(codes))
    return values, {"scored": int(values.notna().sum())}


def shuffle(values, decision_at):
    import numpy as np

    scored = values.dropna().sort_index()
    day = int(pd.Timestamp(decision_at).strftime("%Y%m%d"))
    order = np.random.default_rng(day).permutation(len(scored))
    return pd.Series(scored.to_numpy()[order], index=scored.index)


def describe(values, book):
    picked = values.reindex(book).dropna()
    return {"book_score_span": [float(picked.min()), float(picked.max())] if len(picked) else None}
