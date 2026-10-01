"""Cheap Shenwan level-1 indexes, from sw_daily.pe.

The score is minus the newest visible PE of the industry index. It is the
industry index's own valuation, not a stock earnings yield. At most two names
per industry are held by the book.
"""

from datetime import timedelta

import pandas as pd

LOOKBACK_DAYS = 40
UNI_COLUMNS = ["ts_code", "l1_code"]
SW_COLUMNS = ["dataset", "ts_code", "trade_date", "available_at", "pe"]


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
    frame = frame.assign(pe=pd.to_numeric(frame["pe"], errors="coerce"))
    frame = frame[frame["pe"] > 0]
    if frame.empty:
        raise RuntimeError("no visible positive Shenwan PE")
    frame = frame.sort_values("trade_date").drop_duplicates("ts_code", keep="last")
    pe = dict(zip(frame["ts_code"], frame["pe"]))
    universe = pd.read_parquet(context.asof_dir + "/universe", columns=UNI_COLUMNS)
    info = universe.drop_duplicates("ts_code").set_index("ts_code")
    industry = info["l1_code"].map(_norm)
    values = {}
    for code in codes:
        raw = pe.get(industry.get(code, ""), None)
        values[code] = float("nan") if raw is None else -float(raw)
    return pd.Series(values, dtype="float64"), {"industries": len(pe)}


def shuffle(values, decision_at):
    import numpy as np
    scored = values.dropna().sort_index()
    day = int(pd.Timestamp(decision_at).strftime("%Y%m%d"))
    order = np.random.default_rng(day).permutation(len(scored))
    return pd.Series(scored.to_numpy()[order], index=scored.index)


def describe(values, book):
    picked = values.reindex(book).dropna()
    return {"book_score_span": [float(picked.min()), float(picked.max())] if len(picked) else None}
