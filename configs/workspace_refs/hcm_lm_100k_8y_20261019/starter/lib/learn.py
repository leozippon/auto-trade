"""`fit(context)`: the head under `context.state_dir`, refitted or frozen by knobs.UPDATE, and the daily hook.

The first call of a replay pretrains the head from its initialisation on the
newest PRETRAIN_DAYS trading days and saves it. Under "cold_quarter" the
first call of every later quarter does the same, from the initialisation
again (nothing is carried across, which is what the control is). Every other
daily call -- under "frozen", every call after the first -- hands the loaded
state to `update.daily`, the hook an online rule fills in, which saves what
it changes. Nothing lives in module variables but the encoder's cache: every
call loads the state directory and writes it back, so the fit worker and the
inference worker agree by construction and a cold worker reproduces a warm
one. `load`, `save`, `seeded`, `today` and `texts` are what the decision
(lib/score.py) and the hook share.

The checkpoint, head.pt, holds the head's weights and width, the encoder's
name (a decision refuses a checkpoint from another encoder), the quarter and
the newest labelled day of the fit that wrote it, a step counter, and
`extra`: a mapping of tensors and plain values the hook owns.

A pretraining sample is one name on one bar: the bar's features, the text
the decision that read the bar could see (the next trading day's, today's for
the newest bar), and the bar's label; the newest HORIZON + 1 bars have no
label yet and are not trained on.
"""

import numpy as np
import pandas as pd
import torch
from torch.nn import functional

from lib import device, encoder, features, knobs, model, text, update

MODEL_FILE = "/head.pt"
PANEL_READ = knobs.PRETRAIN_DAYS + features.HISTORY_DAYS + knobs.HORIZON + 1
UPDATES = ("cold_quarter", "frozen")


def fit(context):
    if knobs.UPDATE not in UPDATES:
        raise ValueError(f"knobs.UPDATE must be one of {UPDATES}, got {knobs.UPDATE!r}")
    dev = seeded()
    quarter = f"{context.inference_at.year}Q{(context.inference_at.month - 1) // 3 + 1}"
    state = load(context, dev)
    if state is None or (knobs.UPDATE == "cold_quarter" and state["quarter"] != quarter):
        state = pretrain(context, dev)
        state["quarter"] = quarter
        save(context, state)
        return
    update.daily(context, state)


def seeded():
    """Seed torch from the package's one seed line and return the device."""

    torch.manual_seed(knobs.SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return device.device()


def today(context):
    """The decision day as YYYYMMDD (`inference_at` is Asia/Shanghai)."""

    return context.inference_at.strftime("%Y%m%d")


def load(context, dev):
    """The saved state as live objects, or None before the first fit."""

    try:
        record = torch.load(context.state_dir + MODEL_FILE, map_location=dev, weights_only=True)
    except FileNotFoundError:
        return None
    if str(record["lm"]) != _lm():
        raise RuntimeError(f"the checkpoint was fitted with encoder {record['lm']!r}, the knobs name {_lm()!r}")
    net = model.Head(int(record["dim"])).to(dev)
    net.load_state_dict(record["model"])
    return {
        "model": net, "dim": int(record["dim"]), "quarter": str(record["quarter"]),
        "last_day": str(record["last_day"]), "steps": int(record["steps"]), "extra": dict(record["extra"]),
    }


def save(context, state):
    record = {
        "model": state["model"].state_dict(), "dim": state["dim"], "lm": _lm(), "quarter": state["quarter"],
        "last_day": state["last_day"], "steps": state["steps"], "extra": state["extra"],
    }
    torch.save(record, context.state_dir + MODEL_FILE)


def pretrain(context, dev):
    panel = features.panel(context, PANEL_READ)
    # The decision that reads bar i is the next trading day's; the newest
    # bar's is today's.
    decisions = panel.dates[1:] + [today(context)]
    days = [i for i in range(features.HISTORY_DAYS, len(panel.dates)) if np.isfinite(panel.label[i]).any()]
    if not days:
        raise RuntimeError("the pretraining window holds no labelled day")
    table, ids, width = texts(context, panel.codes, decisions, decisions[days[0]], dev)
    x = torch.as_tensor(panel.x, device=dev)
    valid = torch.as_tensor(panel.valid, device=dev)
    label = torch.as_tensor(panel.label, device=dev)
    names = len(panel.codes)
    net = model.Head(width).to(dev)
    optim = torch.optim.AdamW(net.parameters(), lr=knobs.LR)
    rng = np.random.default_rng([knobs.SEED, int(today(context))])
    net.train()
    for _ in range(knobs.EPOCHS):
        order = rng.permutation(days)
        for start in range(0, len(order), knobs.DAYS_PER_BATCH):
            idx = [int(i) for i in order[start:start + knobs.DAYS_PER_BATCH]]
            y = label[idx].reshape(-1)
            ok = valid[idx].reshape(-1) & torch.isfinite(y)
            if not bool(ok.any()):
                continue
            e, has = inputs(table, ids, idx, names, width, dev)
            s = net(e, has, x[idx].reshape(-1, features.FEATURES))
            optim.zero_grad()
            functional.mse_loss(s[ok], y[ok]).backward()
            optim.step()
    return {"model": net.eval(), "dim": width, "quarter": "", "last_day": panel.dates[days[-1]], "steps": 0, "extra": {}}


def texts(context, codes, decisions, since, dev):
    """The title table (titles, dim) in float16 on `dev`, the id array
    (len(decisions), len(codes), MAX_TITLES) as a tensor, and the embedding
    width, for the titles first seen on or after `since`; (None, None, 1)
    with the text channel off."""

    if not knobs.TEXT:
        return None, None, 1
    rows = text.read(context, decisions, since)
    title_id, uniques = pd.factorize(rows["title"])
    vectors = encoder.embed(context, list(uniques))
    encoder.retain(uniques)
    ids = text.table(rows.assign(title_id=title_id), codes, decisions)
    table = torch.as_tensor(vectors, device=dev, dtype=torch.float16)
    return table, torch.as_tensor(ids, device=dev), int(vectors.shape[1])


def inputs(table, ids, idx, names, width, dev):
    """(text embedding, has-text) of the names on the days `idx`, flattened
    day-major as the features are."""

    if table is None:
        return torch.zeros(len(idx) * names, width, device=dev), torch.zeros(len(idx) * names, device=dev)
    return model.gather(table, ids[idx].reshape(-1, knobs.MAX_TITLES))


def _lm():
    return knobs.LM if knobs.TEXT else ""
