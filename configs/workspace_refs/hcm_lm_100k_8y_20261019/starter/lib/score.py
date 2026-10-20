"""The baseline score: the fitted head read from `context.state_dir`, never trained here.

What `trade.run` relies on:

    score(context, codes) -> (values, meta)
        values: float Series indexed by ts_code over `codes` (the day's
        tradable names); higher is bought first, NaN is not scored.
        meta: a small JSON-safe dict copied into the buy orders' metadata.
    shuffle(values, decision_at) -> the control c_shuf (lib/controls.held_shuffle).
    describe(values, book) -> extra metadata about the chosen book.
    NAME -> the leg label written on the orders.

A decision reads the newest bar's features, the titles first visible within
the last TEXT_DAYS decisions (today's included) and the head as the morning's
fit left it; a name is scored on a day it is `valid` (lib/features.py).
"""

import pandas as pd
import torch

from lib import controls, features, knobs, learn

NAME = "lm"


def score(context, codes):
    dev = learn.seeded()
    state = learn.load(context, dev)
    if state is None:
        raise RuntimeError("no fitted state under context.state_dir: fit(context) has not run in this replay")
    # A lead-in of another HISTORY_DAYS: read at exactly HISTORY_DAYS + 1 days,
    # a name without a bar on the 60-day lag day lost the forward-filled close
    # a fit's longer panel gives it, and that feature's cross-section moved.
    panel = features.panel(context, 2 * features.HISTORY_DAYS + 1)
    calendar = panel.dates + [learn.today(context)]
    names = len(panel.codes)
    table, ids, _ = learn.texts(context, panel.codes, calendar, calendar[-knobs.TEXT_DAYS], dev)
    e, has = learn.inputs(table, ids, [len(calendar) - 1], names, state["dim"], dev)
    net = state["model"].eval()
    with torch.no_grad():
        s = net(e, has, torch.as_tensor(panel.x[-1], device=dev))
    scored = panel.valid[-1]
    values = pd.Series(s.cpu().numpy().astype("float64"), index=panel.codes).where(scored)
    meta = {
        "scored": int(scored.sum()),
        "bar": panel.dates[-1],
        "update": knobs.UPDATE,
        "text": bool(knobs.TEXT),
        "lm": knobs.LM if knobs.TEXT else None,
        "with_text": int((has > 0).sum()),
        "titles": int(table.shape[0]) if table is not None else 0,
        "steps": int(state["steps"]),
        "fit_quarter": state["quarter"],
        "fit_last_day": state["last_day"],
    }
    return values.reindex(list(codes)), meta


def shuffle(values, decision_at):
    return controls.held_shuffle(values, values.index)


def describe(values, book):
    picked = values.reindex(book).dropna()
    return {"book_score_span": [float(picked.min()), float(picked.max())] if len(picked) else None}
