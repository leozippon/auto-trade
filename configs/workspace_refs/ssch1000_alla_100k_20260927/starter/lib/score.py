"""Share of industry implied sales inside CSI 1000.

Current members of industries with at least four names. Implied sales are
circulating market value divided by ps_ttm. The score is that figure divided
by the industry total. The sales yield itself is not the score.
"""

from datetime import timedelta

import pandas as pd

NAME = "m1"
INDEX = "000852.SH"
LOOKBACK_DAYS = 80
WEIGHT_COLUMNS = ["dataset", "available_at", "index_code", "con_code", "trade_date", "weight"]
DAILY_COLUMNS = ["ts_code", "trade_date", "ps_ttm", "circ_mv"]
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
    current = frame[frame["trade_date"] == latest].drop_duplicates("con_code").set_index("con_code")
    bars = pd.read_parquet(context.asof_dir + "/daily", columns=DAILY_COLUMNS, filters=[("trade_date", ">=", start)])
    bars = bars.assign(trade_date=bars["trade_date"].astype(str), ts_code=bars["ts_code"].astype(str))
    last = bars.sort_values("trade_date").drop_duplicates("ts_code", keep="last").set_index("ts_code")
    ps = pd.to_numeric(last["ps_ttm"], errors="coerce")
    size = pd.to_numeric(last["circ_mv"], errors="coerce")
    universe = pd.read_parquet(context.asof_dir + "/universe", columns=UNI_COLUMNS)
    info = universe.drop_duplicates("ts_code").assign(ts_code=lambda raw: raw["ts_code"].astype(str)).set_index("ts_code")
    rows = []
    for code in current.index:
        multiple = ps.get(code)
        mv = size.get(code)
        if multiple is None or mv is None or not (multiple == multiple) or not (mv == mv):
            continue
        if multiple <= 0 or mv <= 0:
            continue
        industry = _norm(info["l1_code"].get(code) if code in info.index else None)
        if not industry:
            continue
        rows.append((code, industry, float(mv) / float(multiple)))
    if len(rows) < 12:
        raise RuntimeError("fewer than 12 current CSI 1000 members are scored")
    table = pd.DataFrame(rows, columns=["code", "l1", "sales"]).set_index("code")
    count = table.groupby("l1")["sales"].transform("size")
    table = table[count >= 4]
    total = table.groupby("l1")["sales"].transform("sum")
    values = (table["sales"] / total).reindex(list(codes))
    if int(values.notna().sum()) < 12:
        raise RuntimeError("fewer than 12 current CSI 1000 members are scored")
    return values, {"scored": int(values.notna().sum()), "section": latest}


def shuffle(values, decision_at):
    import numpy as np

    scored = values.dropna().sort_index()
    day = int(pd.Timestamp(decision_at).strftime("%Y%m%d"))
    order = np.random.default_rng(day).permutation(len(scored))
    return pd.Series(scored.to_numpy()[order], index=scored.index)


def describe(values, book):
    picked = values.reindex(book).dropna()
    return {"book_score_span": [float(picked.min()), float(picked.max())] if len(picked) else None}
