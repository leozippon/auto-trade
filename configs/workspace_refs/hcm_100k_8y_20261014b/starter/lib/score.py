"""The HCM score: the fitted model read from `context.state_dir`, never trained here.

What `trade.run` relies on:

    score(context, codes) -> (values, meta)
        values: float Series indexed by ts_code over `codes` (the day's
        tradable names); higher is bought first, NaN is not scored.
        meta: a small JSON-safe dict copied into the buy orders' metadata.
    shuffle(values, decision_at) -> the control c_shuf (lib/controls.held_shuffle).
    describe(values, book) -> extra metadata about the chosen book.
    NAME -> the leg label written on the orders.

In carried mode the decision reads each name's state as the morning's fit
left it -- the newest bar's state, which the fit rolled before this call --
and refuses to score from a state that is not the newest bar's. In window
mode it recomputes the state from zero over the last SEQ_DAYS days. Either
way the embedding, the retrieved memory signal and the head give the score;
a name is scored on a day it is `valid` (lib/features.py).
"""

import pandas as pd
import torch

from lib import controls, features, hcm, knobs

NAME = "hcm"


def score(context, codes):
    dev = hcm.seeded()
    ckpt = hcm.load(context, dev)
    if ckpt is None:
        raise RuntimeError("no fitted state under context.state_dir: fit(context) has not run in this replay")
    latest = features.latest_day(context)
    net = ckpt["model"].eval()
    with torch.no_grad():
        if knobs.STATE == "carried":
            if ckpt["ring_dates"][-1] != latest:
                raise RuntimeError(f"the carried state stands at {ckpt['ring_dates'][-1]}, the newest bar is {latest}")
            names, h, ok = ckpt["codes"], ckpt["ring"][-1], ckpt["seen"] == int(latest)
        else:
            panel = features.panel(context, features.HISTORY_DAYS + knobs.SEQ_DAYS)
            x = torch.as_tensor(panel.x[-knobs.SEQ_DAYS:], device=dev)
            names, ok = panel.codes, torch.as_tensor(panel.valid[-1], device=dev)
            h = net.run(x, torch.zeros(len(names), knobs.HIDDEN, device=dev))
        e = net.embed(h)
        m = torch.zeros(len(names), device=dev)
        if ckpt["store"] is not None:
            m = ckpt["store"].signal(e, torch.full((len(names),), int(latest), dtype=torch.int64, device=dev), knobs.KNN)
        s = net.score(e, m)
    scored = ok.cpu().numpy()
    values = pd.Series(s.cpu().numpy().astype("float64"), index=names).where(scored)
    meta = {
        "scored": int(scored.sum()),
        "state_day": latest,
        "update": knobs.UPDATE,
        "state": knobs.STATE,
        "memory": bool(knobs.MEMORY),
        "steps": int(ckpt["steps"]),
        "store_filled": int(ckpt["store"].filled) if ckpt["store"] is not None else 0,
        "memory_signal_mean": round(float(m[ok].mean()), 6) if scored.any() else 0.0,
    }
    return values.reindex(list(codes)), meta


def shuffle(values, decision_at):
    return controls.held_shuffle(values, values.index)


def describe(values, book):
    picked = values.reindex(book).dropna()
    return {"book_score_span": [float(picked.min()), float(picked.max())] if len(picked) else None}
