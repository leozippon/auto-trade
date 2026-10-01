"""Industry-index score. Every name in an industry shares one number.

The number is the Shenwan level-1 index close-to-close return over the last
21 visible sessions. Names with no industry code are left unscored.
"""

from datetime import timedelta

import pandas as pd

from lib import knobs

SW_COLUMNS = ["dataset", "ts_code", "trade_date", "available_at", "close"]
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


def _industry_return(context):
    start = (context.inference_at - timedelta(days=knobs.WINDOW_DAYS)).strftime("%Y%m%d")
    frame = pd.read_parquet(
        context.asof_dir + "/macro",
        columns=SW_COLUMNS,
        filters=[("dataset", "=", "sw_daily"), ("trade_date", ">=", start)],
    )
    missing = [name for name in SW_COLUMNS if name not in frame.columns]
    if missing:
        raise RuntimeError(f"sw_daily is missing columns {missing}")
    stamp = pd.to_datetime(frame["available_at"], utc=True)
    frame = frame[stamp <= pd.Timestamp(context.inference_at).tz_convert("UTC")]
    frame = frame.assign(ts_code=frame["ts_code"].map(_norm), trade_date=frame["trade_date"].astype(str))
    frame = frame[frame["ts_code"].str.fullmatch(r"801\d\d0\.SI")]
    if frame.empty:
        raise RuntimeError("no visible Shenwan level-1 rows")
    close = pd.to_numeric(frame["close"], errors="coerce")
    frame = frame.assign(close=close).dropna(subset=["close"])
    frame = frame.sort_values("trade_date").drop_duplicates(["ts_code", "trade_date"], keep="last")
    out = {}
    for code, group in frame.groupby("ts_code"):
        closes = group["close"].to_numpy()
        if len(closes) < 21 or closes[-21] <= 0:
            continue
        out[code] = float(closes[-1] / closes[-21] - 1.0)
    if not out:
        raise RuntimeError("no Shenwan industry has 21 closes")
    return out


def score(context, codes):
    industry_return = _industry_return(context)
    universe = pd.read_parquet(context.asof_dir + "/universe", columns=UNI_COLUMNS)
    info = universe.drop_duplicates("ts_code").set_index("ts_code")
    industry = info["l1_code"].map(_norm) if "l1_code" in info.columns else pd.Series(dtype=str)
    signed = {}
    for code in codes:
        raw = industry_return.get(industry.get(code, ""), None)
        if raw is None:
            signed[code] = float("nan")
        elif knobs.SIGN == 0:
            signed[code] = -abs(raw)
        else:
            signed[code] = knobs.SIGN * raw
    signed = pd.Series(signed, dtype="float64")
    meta = {"industries": len(industry_return)}
    return signed, meta


def shuffle(values, decision_at):
    scored = values.dropna().sort_index()
    day = int(pd.Timestamp(decision_at).strftime("%Y%m%d"))
    import numpy as np
    order = np.random.default_rng(day).permutation(len(scored))
    return pd.Series(scored.to_numpy()[order], index=scored.index)


def describe(values, book):
    ranks = (-values.reindex(book)).dropna()
    return {"book_score_span": [int(ranks.min()), int(ranks.max())] if len(ranks) else None}
