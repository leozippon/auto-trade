"""Listing age on the decision day.

The score is how many days the name has been listed. Older names score
higher. This is not a return and not a new-listing decay stack.
"""

import pandas as pd

NAME = "m1"
UNI_COLUMNS = ["ts_code", "list_date"]


def score(context, codes):
    universe = pd.read_parquet(context.asof_dir + "/universe", columns=UNI_COLUMNS)
    info = universe.drop_duplicates("ts_code").set_index("ts_code")
    listed = pd.to_datetime(info["list_date"].astype(str), format="%Y%m%d", errors="coerce")
    decision = pd.Timestamp(context.inference_at)
    if decision.tzinfo is not None:
        decision = decision.tz_convert("Asia/Shanghai").tz_localize(None)
    age = (decision - listed).dt.days
    values = age.where(age > 0).reindex(list(codes))
    if int(values.notna().sum()) < 12:
        raise RuntimeError("listing age scored fewer than 12 names")
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
