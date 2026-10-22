"""The decision's score: what the morning's fit wrote under `context.state_dir`, never computed here.

What `trade.run` relies on:

    score(context, codes) -> (values, meta)
        values: float Series indexed by ts_code over `codes` (the day's
        tradable names); higher is bought first, NaN is not scored.
        meta: a small JSON-safe dict copied into the buy orders' metadata.
    shuffle(values, decision_at) -> the control c_shuf (lib/controls.held_shuffle).
    describe(values, book) -> extra metadata about the chosen book.
    NAME -> the leg label written on the orders.

The fit of the same morning (lib/learn.py) scores every name with a valid
pattern and writes scores_<day>.npz; a decision reads only that file, so it
needs no model, no card and no module state, and a Paper worker that re-sends
earlier decisions reads each one's own day.
"""

import numpy as np
import pandas as pd

from lib import controls, knobs

NAME = "retrieval"


def score(context, codes):
    day = context.inference_at.strftime("%Y%m%d")
    try:
        record = np.load(context.state_dir + f"/scores_{day}.npz")
    except FileNotFoundError as exc:
        raise RuntimeError(f"no scores for {day} under context.state_dir: this morning's fit has not run") from exc
    values = pd.Series(record["values"].astype("float64"), index=record["codes"].astype(str))
    meta = {"scored": int(values.notna().sum())}
    for item in record["meta"].astype(str):
        name, _, value = item.partition("=")
        meta[name] = value
    meta.update({"text": bool(knobs.TEXT), "memory": knobs.MEMORY, "neighbours": knobs.NEIGHBOURS})
    return values.reindex(list(codes)), meta


def shuffle(values, decision_at):
    return controls.held_shuffle(values, values.index)


def describe(values, book):
    picked = values.reindex(book).dropna()
    return {"book_score_span": [round(float(picked.min()), 5), round(float(picked.max()), 5)] if len(picked) else None}
