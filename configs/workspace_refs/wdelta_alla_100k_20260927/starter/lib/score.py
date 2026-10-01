"""Change in published index weight between the two newest visible sections.

`index_weight` keeps the index in `index_code` and the stock in `con_code`.
A name absent from a section has weight 0. The score is the later weight minus
the earlier one. Optional cash-dividend filter keeps only names with a visible
cash distribution whose ex-date falls in the trailing year.
"""

from datetime import timedelta

import pandas as pd

from lib import knobs

WEIGHT_COLUMNS = ["dataset", "available_at", "index_code", "con_code", "trade_date", "weight"]
ACTION_COLUMNS = ["ts_code", "ex_date", "cash_per_share"]


def _weights(context):
    start = (context.inference_at - timedelta(days=140)).strftime("%Y%m%d")
    frame = pd.read_parquet(
        context.asof_dir + "/macro",
        columns=WEIGHT_COLUMNS,
        filters=[("dataset", "=", "index_weight"), ("index_code", "=", knobs.TARGET),
                 ("trade_date", ">=", start)],
    )
    stamp = pd.to_datetime(frame["available_at"], utc=True)
    frame = frame[(stamp <= pd.Timestamp(context.inference_at).tz_convert("UTC"))
                  & frame["con_code"].notna() & (frame["weight"] > 0)]
    if frame.empty:
        raise RuntimeError(f"no visible {knobs.TARGET} weight section")
    frame = frame.assign(trade_date=frame["trade_date"].astype(str), con_code=frame["con_code"].astype(str))
    days = sorted(frame["trade_date"].unique())
    if len(days) < 2:
        raise RuntimeError("need two index-weight sections to measure a change")
    def book(day):
        rows = frame[frame["trade_date"] == day]
        return {code: float(weight) for code, weight in zip(rows["con_code"], rows["weight"])}
    return book(days[-1]), book(days[-2]), days[-1]


def _payers(context):
    if not knobs.REQUIRE_CASH:
        return None
    start = (context.inference_at - timedelta(days=365)).strftime("%Y%m%d")
    end = context.inference_at.strftime("%Y%m%d")
    frame = pd.read_parquet(context.asof_dir + "/corporate_actions", columns=ACTION_COLUMNS)
    frame = frame.assign(ex_date=frame["ex_date"].astype(str), ts_code=frame["ts_code"].astype(str))
    frame = frame[(frame["ex_date"] >= start) & (frame["ex_date"] < end) & (frame["cash_per_share"] > 0)]
    return set(frame["ts_code"])


def score(context, codes):
    now, prev, section = _weights(context)
    payers = _payers(context)
    values = {}
    for code in codes:
        if code not in now and code not in prev:
            values[code] = float("nan")
            continue
        delta = now.get(code, 0.0) - prev.get(code, 0.0)
        if payers is not None and code not in payers:
            values[code] = float("nan")
            continue
        values[code] = delta
    return pd.Series(values, dtype="float64"), {"weight_section": section, "names_now": len(now)}


def shuffle(values, decision_at):
    import numpy as np
    scored = values.dropna().sort_index()
    day = int(pd.Timestamp(decision_at).strftime("%Y%m%d"))
    order = np.random.default_rng(day).permutation(len(scored))
    return pd.Series(scored.to_numpy()[order], index=scored.index)


def describe(values, book):
    picked = values.reindex(book).dropna()
    return {"book_score_span": [float(picked.min()), float(picked.max())] if len(picked) else None}
