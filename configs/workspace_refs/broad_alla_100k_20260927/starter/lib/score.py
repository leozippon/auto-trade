"""Broader Shenwan industries rank higher.

The score is how many of today's tradable names share the stock's level-1
industry. Names in one industry share the score.
"""

import pandas as pd

NAME = "m1"
UNI_COLUMNS = ["ts_code", "l1_name"]


def score(context, codes):
    universe = pd.read_parquet(context.asof_dir + "/universe", columns=UNI_COLUMNS)
    info = universe.drop_duplicates("ts_code").assign(ts_code=lambda raw: raw["ts_code"].astype(str)).set_index("ts_code")
    home = info["l1_name"].fillna("").astype(str)
    counted = [code for code in codes if home.get(code, "").strip()]
    if len(counted) < 12:
        raise RuntimeError("fewer than 12 names have a Shenwan industry")
    industry = home.reindex(counted)
    size = industry.value_counts()
    values = {code: float(size[industry[code]]) for code in counted}
    return pd.Series(values, dtype="float64").reindex(list(codes)), {"scored": len(values), "industries": int(size.shape[0])}


def shuffle(values, decision_at):
    import numpy as np

    scored = values.dropna().sort_index()
    day = int(pd.Timestamp(decision_at).strftime("%Y%m%d"))
    order = np.random.default_rng(day).permutation(len(scored))
    return pd.Series(scored.to_numpy()[order], index=scored.index)


def describe(values, book):
    picked = values.reindex(book).dropna()
    return {"book_score_span": [float(picked.min()), float(picked.max())] if len(picked) else None}
