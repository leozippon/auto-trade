"""Change in Shenwan level-1 price-to-book over 21 sessions.

The score is minus that change. Industries whose pb fell score higher.
It is not the level of pe, and not a stock's own book yield. Names in one
industry share the score.
"""

from datetime import timedelta

import pandas as pd

NAME = "m1"
LOOKBACK_DAYS = 50
SESSIONS = 21
UNI_COLUMNS = ["ts_code", "l1_code"]
SW_COLUMNS = ["dataset", "ts_code", "trade_date", "available_at", "pb"]


def _norm(code):
    text = "" if code is None or (isinstance(code, float) and pd.isna(code)) else str(code).strip()
    if not text or text == "nan":
        return ""
    if text.endswith(".SI"):
        return text
    if text.isdigit():
        return text + ".SI"
    return text


def score(context, codes):
    start = (context.inference_at - timedelta(days=LOOKBACK_DAYS)).strftime("%Y%m%d")
    frame = pd.read_parquet(
        context.asof_dir + "/macro",
        columns=SW_COLUMNS,
        filters=[("dataset", "=", "sw_daily"), ("trade_date", ">=", start)],
    )
    stamp = pd.to_datetime(frame["available_at"], utc=True)
    frame = frame[stamp <= pd.Timestamp(context.inference_at).tz_convert("UTC")]
    frame = frame.assign(ts_code=frame["ts_code"].map(_norm), trade_date=frame["trade_date"].astype(str))
    frame = frame[frame["ts_code"].str.fullmatch(r"801\d\d0\.SI")]
    frame = frame.assign(pb=pd.to_numeric(frame["pb"], errors="coerce"))
    frame = frame[frame["pb"] > 0]
    if frame.empty:
        raise RuntimeError("no visible positive Shenwan pb")
    change = {}
    for code, part in frame.groupby("ts_code"):
        part = part.sort_values("trade_date")
        if len(part) < SESSIONS:
            continue
        prev = float(part["pb"].iloc[-SESSIONS])
        last = float(part["pb"].iloc[-1])
        if prev <= 0:
            continue
        change[code] = last / prev - 1.0
    if not change:
        raise RuntimeError("no industry has 21 pb sessions")
    universe = pd.read_parquet(context.asof_dir + "/universe", columns=UNI_COLUMNS)
    info = universe.drop_duplicates("ts_code").set_index("ts_code")
    industry = info["l1_code"].map(_norm)
    values = {code: -change[industry.get(code, "")] if industry.get(code, "") in change else float("nan") for code in codes}
    return pd.Series(values, dtype="float64"), {"industries": len(change)}


def shuffle(values, decision_at):
    import numpy as np

    scored = values.dropna().sort_index()
    day = int(pd.Timestamp(decision_at).strftime("%Y%m%d"))
    order = np.random.default_rng(day).permutation(len(scored))
    return pd.Series(scored.to_numpy()[order], index=scored.index)


def describe(values, book):
    picked = values.reindex(book).dropna()
    return {"book_score_span": [float(picked.min()), float(picked.max())] if len(picked) else None}
