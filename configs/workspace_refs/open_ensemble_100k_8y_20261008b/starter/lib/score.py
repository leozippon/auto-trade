"""Placeholder score p0_placeholder: a fixed arbitrary order. It is not a hypothesis and not a core.

This is the module the session replaces. What `trade.run` relies on:

    score(context, codes) -> (values, meta)
        values: float Series indexed by ts_code over `codes` (the day's
        tradable names); higher is bought first, NaN is not scored.
        meta: a small JSON-safe dict copied into the buy orders' metadata.
    shuffle(values, decision_at) -> the control c_shuf: the same values on
        stand-in names fixed across reviews (lib/controls.held_shuffle over
        the day's tradable names), so a book with a keep band holds each
        stand-in as long as the candidate holds the name it stands for.
    describe(values, book) -> extra metadata about the chosen book.
    NAME -> the leg label written on the orders.

p0_placeholder orders names by a hash of their code, so it carries no
information about returns and exists only to exercise the read path. It is a
FIXED BASKET, not a neutral or random core: the same few dozen names head the
order every day for years, so a book on it holds one basket's luck. Round
20261014 read it once as a factor score and once as a "neutral core"; neither
is what it is. A core under an exclusion layer is a declared construction with
its own control (`lib/controls.py`), and no nominated leg is named p0. It
scores only names with a bar on the first day of the last HISTORY_DAYS trading
days, the history a windowed feature of that length needs.
"""

import pandas as pd

from lib import controls, data

NAME = "p0_placeholder"
HISTORY_DAYS = 60


def score(context, codes):
    closes = data.panel(context, ["close"], HISTORY_DAYS)["close"]
    seasoned = closes.iloc[0].reindex(list(codes)).notna()
    picked = pd.Series(seasoned[seasoned].index, dtype="object")
    order = pd.util.hash_pandas_object(picked, index=False).to_numpy() % 1_000_003
    values = pd.Series(order.astype("float64"), index=picked.to_numpy())
    return values.reindex(list(codes)), {"scored": int(len(values)), "window_start": str(closes.index[0])}


def shuffle(values, decision_at):
    return controls.held_shuffle(values, values.index)


def describe(values, book):
    picked = values.reindex(book).dropna()
    return {"book_score_span": [float(picked.min()), float(picked.max())] if len(picked) else None}
