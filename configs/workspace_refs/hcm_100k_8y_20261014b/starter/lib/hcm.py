"""`fit(context)`: the model's three memories, kept under `context.state_dir`.

The first call of a replay pretrains from scratch on the newest PRETRAIN_DAYS
trading days (sequences from a zero state, no memory signal), then, with
MEMORY on, fills the store from that window and trains one more pass with the
retrieved signal, and, in carried mode, rolls every name's state over the
window's last days so the ring holds a state per day. Every later daily call
(REFIT_PERIOD "day"):

  1. reads the newest RECENT_DAYS trading days;
  2. carried mode: rolls every name's state forward through each new day,
     names without a bar keeping theirs;
  3. for each day whose label matured since the last call -- normally the one
     day whose HORIZON-day window closed with the newest bar -- takes one
     gradient step on that day's names (the last SEQ_DAYS days from the state
     stored SEQ_DAYS days earlier, or from zero in window mode) plus a replay
     minibatch of stored episodes through the body and head, and steps each
     parameter group only when its period has elapsed, its gradient having
     accumulated over the days in between (knobs.UPDATE);
  4. offers the matured episodes to the store;
  5. saves the weights, the optimiser moments, the gradients still pending
     in the groups not yet stepped, the state ring, the counters and the
     store.

"frozen" skips step 3; "cold_quarter" skips it too and instead pretrains
afresh at the first call of each quarter. Nothing lives in module variables:
every call loads the state directory and writes it back, so the fit worker and
the inference worker agree by construction, and a cold worker reproduces a
warm one. `load` and `seeded` are what the decision (lib/score.py) shares.
"""

import numpy as np
import pandas as pd
import torch
from torch.nn import functional

from lib import device, features, knobs, model

MODEL_FILE = "/hcm_model.pt"
STORE_FILE = "/hcm_store.pt"
# Days of carried state kept: a daily step's sequence starts SEQ_DAYS before
# the matured day, which lies HORIZON + 1 days before the newest bar.
RING = knobs.SEQ_DAYS + knobs.HORIZON + 2
# The daily read: the matured day's sequence and the beta window behind it.
RECENT_DAYS = features.BETA_DAYS + RING
PRETRAIN_READ = knobs.PRETRAIN_DAYS + features.HISTORY_DAYS + knobs.HORIZON + 1
DAYS_PER_BATCH = 8
UPDATES = ("graded", "uniform", "reversed", "frozen", "cold_quarter")
LEARNING = ("graded", "uniform", "reversed")


def fit(context):
    if knobs.UPDATE not in UPDATES:
        raise ValueError(f"knobs.UPDATE must be one of {UPDATES}, got {knobs.UPDATE!r}")
    if knobs.STATE not in ("carried", "window"):
        raise ValueError(f"knobs.STATE must be 'carried' or 'window', got {knobs.STATE!r}")
    dev = seeded()
    today = int(context.inference_at.strftime("%Y%m%d"))
    rng = np.random.default_rng([knobs.SEED, today])
    quarter = f"{context.inference_at.year}Q{(context.inference_at.month - 1) // 3 + 1}"
    ckpt = load(context, dev)
    if ckpt is None or (knobs.UPDATE == "cold_quarter" and ckpt["quarter"] != quarter):
        ckpt = _pretrain(context, dev, rng)
        ckpt["quarter"] = quarter
        _save(context, ckpt)
        return
    if knobs.STATE == "window" and knobs.UPDATE not in LEARNING and not knobs.MEMORY:
        return
    _continue(context, dev, rng, ckpt)
    _save(context, ckpt)


def seeded():
    """Seed torch from the package's one seed line and return the device."""

    torch.manual_seed(knobs.SEED)
    torch.backends.cudnn.benchmark = False
    return device.device()


def load(context, dev):
    """The saved state as live objects, or None before the first fit."""

    try:
        record = torch.load(context.state_dir + MODEL_FILE, map_location=dev, weights_only=True)
    except FileNotFoundError:
        return None
    net = model.HCM(features.FEATURES, knobs.HIDDEN).to(dev)
    net.load_state_dict(record["model"])
    parameters = dict(net.named_parameters())
    for name, grad in record["grads"].items():
        parameters[name].grad = grad
    optim = None
    if record["optim"] is not None:
        optim = _optimisers(net)
        for name, state in record["optim"].items():
            optim[name].load_state_dict(state)
    store = None
    if knobs.MEMORY:
        episodes = torch.load(context.state_dir + STORE_FILE, map_location=dev, weights_only=True)
        store = model.Store.load(episodes, knobs.HIDDEN, dev)
    ring = record["ring"].to(dev) if record["ring"] is not None else None
    seen = record["seen"].to(dev) if record["seen"] is not None else None
    return {
        "model": net, "optim": optim, "store": store, "codes": list(record["codes"]),
        "ring": ring, "ring_dates": list(record["ring_dates"] or []), "seen": seen,
        "last_day": str(record["last_day"]), "steps": int(record["steps"]), "quarter": str(record["quarter"]),
    }


def _save(context, ckpt):
    record = {
        "model": ckpt["model"].state_dict(),
        # What each group has accumulated since its last step (`_step`): the
        # next fit rebuilds the model from this record and adds to it.
        "grads": {name: p.grad for name, p in ckpt["model"].named_parameters() if p.grad is not None},
        "optim": {name: opt.state_dict() for name, opt in ckpt["optim"].items()} if ckpt["optim"] else None,
        "codes": ckpt["codes"],
        "ring": ckpt["ring"].cpu() if ckpt["ring"] is not None else None,
        "ring_dates": ckpt["ring_dates"],
        "seen": ckpt["seen"].cpu() if ckpt["seen"] is not None else None,
        "last_day": ckpt["last_day"],
        "steps": ckpt["steps"],
        "quarter": ckpt["quarter"],
    }
    torch.save(record, context.state_dir + MODEL_FILE)
    if ckpt["store"] is not None:
        torch.save(ckpt["store"].state(), context.state_dir + STORE_FILE)


def _pretrain(context, dev, rng):
    panel = features.panel(context, PRETRAIN_READ)
    x, valid, label = _tensors(panel, panel.codes, dev)
    day_ids = torch.as_tensor([int(day) for day in panel.dates], device=dev)
    due_ids = torch.as_tensor([int(day) if day else 0 for day in panel.due], device=dev)
    days = [i for i in range(knobs.SEQ_DAYS - 1, len(panel.dates)) if np.isfinite(panel.label[i]).any()]
    if not days:
        raise RuntimeError("the pretraining window holds no labelled day")
    net = model.HCM(features.FEATURES, knobs.HIDDEN).to(dev)
    optim = torch.optim.AdamW(net.parameters(), lr=knobs.PRETRAIN_LR)
    names = len(panel.codes)

    def batch(idx, store):
        seq = torch.cat([x[i - knobs.SEQ_DAYS + 1:i + 1] for i in idx], dim=1)
        qday = day_ids[idx].repeat_interleave(names)
        h, e, s = _forward(net, store, seq, torch.zeros(seq.shape[1], knobs.HIDDEN, device=dev), qday)
        y = label[idx].reshape(-1)
        return h, e, s, y, valid[idx].reshape(-1) & torch.isfinite(y), qday, due_ids[idx].repeat_interleave(names)

    def fill():
        store = model.Store(knobs.STORE_SIZE, knobs.HIDDEN, dev)
        net.eval()
        with torch.no_grad():
            for idx in _batches(days):
                h, e, _, y, ok, qday, due = batch(idx, None)
                store.add(e[ok], h[ok], y[ok], qday[ok], due[ok], rng)
        return store

    store = None
    for memory in [False] * knobs.PRETRAIN_EPOCHS + [True] * int(knobs.MEMORY):
        if memory:
            store = fill()
        net.train()
        for idx in _batches(rng.permutation(days)):
            _, _, s, y, ok, _, _ = batch(idx, store)
            if not bool(ok.any()):
                continue
            optim.zero_grad()
            functional.mse_loss(s[ok], y[ok]).backward()
            optim.step()
    # The daily schedule starts accumulating from nothing.
    optim.zero_grad(set_to_none=True)
    if knobs.MEMORY:
        store = fill()
    ring = ring_dates = seen = None
    if knobs.STATE == "carried":
        net.eval()
        seen = torch.zeros(names, dtype=torch.int64, device=dev)
        h = torch.zeros(names, knobs.HIDDEN, device=dev)
        states = []
        with torch.no_grad():
            for i in range(max(0, len(panel.dates) - 2 * RING), len(panel.dates)):
                h = _roll(net, h, x[i], valid[i])
                seen[valid[i]] = int(panel.dates[i])
                states.append(h)
        ring, ring_dates = torch.stack(states[-RING:]), panel.dates[-RING:]
    return {
        "model": net, "optim": _optimisers(net) if knobs.UPDATE in LEARNING else None, "store": store,
        "codes": panel.codes, "ring": ring, "ring_dates": ring_dates, "seen": seen,
        "last_day": panel.dates[days[-1]], "steps": 0,
    }


def _continue(context, dev, rng, ckpt):
    panel = features.panel(context, RECENT_DAYS)
    known = set(ckpt["codes"])
    codes = ckpt["codes"] + [code for code in panel.codes if code not in known]
    x, valid, label = _tensors(panel, codes, dev)
    net, store, dates = ckpt["model"], ckpt["store"], panel.dates
    if knobs.STATE == "carried":
        grown = len(codes) - len(ckpt["codes"])
        ring = torch.cat([ckpt["ring"], torch.zeros(ckpt["ring"].shape[0], grown, knobs.HIDDEN, device=dev)], dim=1)
        seen = torch.cat([ckpt["seen"], torch.zeros(grown, dtype=torch.int64, device=dev)])
        ring_dates = ckpt["ring_dates"]
        if ring_dates[-1] not in dates:
            raise RuntimeError(f"the carried state stands at {ring_dates[-1]}, outside the {RECENT_DAYS}-day read")
        net.eval()
        with torch.no_grad():
            for i in range(dates.index(ring_dates[-1]) + 1, len(dates)):
                ring = torch.cat([ring, _roll(net, ring[-1], x[i], valid[i]).unsqueeze(0)])[-RING:]
                ring_dates = (ring_dates + [dates[i]])[-RING:]
                seen[valid[i]] = int(dates[i])
        ckpt.update(codes=codes, ring=ring, ring_dates=ring_dates, seen=seen)
    else:
        ckpt["codes"] = codes
    for i in [i for i, day in enumerate(dates) if day > ckpt["last_day"] and np.isfinite(panel.label[i]).any()]:
        if i - knobs.SEQ_DAYS < 0:
            raise RuntimeError(f"matured day {dates[i]} has no {knobs.SEQ_DAYS}-day sequence in the read")
        seq = x[i - knobs.SEQ_DAYS + 1:i + 1]
        if knobs.STATE == "carried":
            if dates[i - knobs.SEQ_DAYS] not in ckpt["ring_dates"]:
                raise RuntimeError(f"no carried state for {dates[i - knobs.SEQ_DAYS]}: the ring holds {RING} days")
            h0 = ckpt["ring"][ckpt["ring_dates"].index(dates[i - knobs.SEQ_DAYS])]
        else:
            h0 = torch.zeros(len(codes), knobs.HIDDEN, device=dev)
        qday = torch.full((len(codes),), int(dates[i]), dtype=torch.int64, device=dev)
        y = label[i]
        ok = valid[i] & torch.isfinite(y)
        if knobs.UPDATE in LEARNING and bool(ok.any()):
            net.train()
            _, _, s = _forward(net, store, seq, h0, qday)
            loss = functional.mse_loss(s[ok], y[ok])
            if store is not None and store.filled:
                picked = torch.as_tensor(rng.integers(0, store.filled, knobs.REPLAY_BATCH), device=dev)
                e = net.embed(store.hin[picked])
                replayed = net.score(e, store.signal(e, store.day[picked], knobs.KNN))
                loss = loss + functional.mse_loss(replayed, store.label[picked])
            loss.backward()
            ckpt["steps"] += 1
            _step(ckpt["optim"], net, ckpt["steps"])
        if store is not None:
            net.eval()
            with torch.no_grad():
                h, e, _ = _forward(net, store, seq, h0, qday)
            due = torch.full((len(codes),), int(panel.due[i]), dtype=torch.int64, device=dev)
            store.add(e[ok], h[ok], y[ok], qday[ok], due[ok], rng)
        ckpt["last_day"] = dates[i]


def _forward(net, store, seq, h0, qday):
    """(state, embedding, score) of one sequence batch from the states `h0`;
    the memory signal comes from `store` as known on each row's `qday`."""

    h = net.run(seq, h0)
    e = net.embed(h)
    m = store.signal(e, qday, knobs.KNN) if store is not None else torch.zeros(e.shape[0], device=e.device)
    return h, e, net.score(e, m)


def _roll(net, h, x, valid):
    """Every name's state one day on; a name without a bar keeps its state."""

    return torch.where(valid.unsqueeze(1), net.step(x, h), h)


def _schedule():
    pairs = {
        "head": (knobs.HEAD_LR, knobs.HEAD_PERIOD),
        "body": (knobs.BODY_LR, knobs.BODY_PERIOD),
        "input": (knobs.INPUT_LR, knobs.INPUT_PERIOD),
    }
    if knobs.UPDATE == "uniform":
        pairs = {name: pairs["head"] for name in pairs}
    elif knobs.UPDATE == "reversed":
        pairs["head"], pairs["input"] = pairs["input"], pairs["head"]
    return pairs


def _optimisers(net):
    groups = model.groups(net)
    return {name: torch.optim.AdamW(groups[name], lr=lr) for name, (lr, _) in _schedule().items()}


def _step(optim, net, steps):
    """Step the groups whose period divides the step count; the others keep
    accumulating their gradient until theirs does."""

    groups = model.groups(net)
    for name, (_, period) in _schedule().items():
        if steps % period == 0:
            optim[name].step()
            for parameter in groups[name]:
                parameter.grad = None


def _batches(order):
    for start in range(0, len(order), DAYS_PER_BATCH):
        yield [int(i) for i in order[start:start + DAYS_PER_BATCH]]


def _tensors(panel, codes, dev):
    """The panel's arrays over `codes` (absent names unknown), as tensors."""

    at = pd.Index(panel.codes).get_indexer(codes)
    present = at >= 0
    x = np.zeros((len(panel.dates), len(codes), features.FEATURES), dtype="float32")
    valid = np.zeros((len(panel.dates), len(codes)), dtype=bool)
    label = np.full((len(panel.dates), len(codes)), np.nan, dtype="float32")
    x[:, present] = panel.x[:, at[present]]
    valid[:, present] = panel.valid[:, at[present]]
    label[:, present] = panel.label[:, at[present]]
    return tuple(torch.as_tensor(array, device=dev) for array in (x, valid, label))
