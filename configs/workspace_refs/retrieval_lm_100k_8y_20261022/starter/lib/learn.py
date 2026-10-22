"""`fit(context)`: file the day's keys and matured outcomes, then score every name for the day's decision.

The whole day's work happens here, in the one place the state directory is
writable; the decision (lib/score.py) only reads the scores this writes.

The first fit of a replay fills the memory: it keys every decision day of the
newest MEMORY_DAYS trading days before today from the PIT history (pattern at
the bar that decision read, mean title vector over the TEXT_DAYS decisions up
to it), files the outcomes of those whose label has matured and leaves the
rest pending. Every later fit keys today, files the outcomes that have just
matured (one day's, normally) -- unless knobs.MEMORY is "frozen" -- and keeps
the newest day pending. Then it scores today's names, each one whose pattern
is valid, by knobs.SCORER:

    "retrieval"     the kernel-weighted outcomes of its K nearest filed cases
                    (lib/memory.py)
    "cold_quarter"  C1: the head fitted on the filed cases at the first fit of
                    the quarter (lib/model.py), saved as head.pt

and writes scores_<today>.npz. A key is the pattern (lib/features.patterns)
beside the text block: the name's mean title vector (lib/encoder.py over
lib/text.read), scaled so that its squared length is TEXT_WEIGHT times the
pattern's expected one; a name without titles in the window has a zero text
block, and with knobs.TEXT off there is none.

Per-day files, rather than one overwritten file, let a Paper worker that
re-sends every earlier decision to rebuild its module state find each day's
own scores; nothing in a decision depends on module state.
"""

import numpy as np
import pandas as pd
import torch

from lib import device, encoder, features, knobs, memory, model, text

SCORERS = ("retrieval", "cold_quarter")
MEMORIES = ("continuous", "frozen")
NEIGHBOURS = ("similar", "placebo")
HEAD_FILE = "/head.pt"
# Bars behind today a panel reads: the first fit keys MEMORY_DAYS decisions,
# each needing its label's beta window and the pattern's history; a later fit
# needs the oldest pending label's window (HORIZON + 1 bars back) and its beta.
FIRST_PANEL = knobs.MEMORY_DAYS + features.LABEL_LEAD + 2
DAILY_PANEL = knobs.HORIZON + 1 + features.LABEL_LEAD + 20


def fit(context):
    for name, value, allowed in (
        ("SCORER", knobs.SCORER, SCORERS), ("MEMORY", knobs.MEMORY, MEMORIES), ("NEIGHBOURS", knobs.NEIGHBOURS, NEIGHBOURS),
    ):
        if value not in allowed:
            raise ValueError(f"knobs.{name} must be one of {allowed}, got {value!r}")
    dev = seeded()
    day = today(context)
    filed_pending = memory.index(context)
    first = filed_pending is None
    p = features.panel(context, FIRST_PANEL if first else DAILY_PANEL)
    calendar = p.dates + [day]
    if first:
        start = len(calendar) - 1 - knobs.MEMORY_DAYS
        filed, pending = [], []
    else:
        start = len(calendar) - 1
        filed, pending = filed_pending
    grow = first or knobs.MEMORY == "continuous"
    vectors, positions = titles(context, p, calendar, start)
    for at in range(start, len(calendar)):
        codes, key = keys(p, at, vectors, positions)
        if grow:
            memory.write_keys(context, calendar[at], codes, key)
            pending.append(calendar[at])
    if grow:
        filed, pending = mature(context, p, filed, pending)
        memory.save_index(context, filed, pending)
    values, meta = score(context, filed, key, dev)
    np.savez(
        context.state_dir + f"/scores_{day}.npz",
        codes=np.array(codes, dtype="S12"),
        values=np.asarray(values, dtype="float32"),
        meta=np.array([f"{k}={v}" for k, v in meta.items()], dtype="S64"),
    )
    if dev.type == "cuda":
        # Hand the allocator's cached blocks back, so that what the card shows
        # between fits is the memory and the encoder, not the day's transients.
        torch.cuda.empty_cache()


def seeded():
    """Seed torch from the package's one seed line and return the device."""

    torch.manual_seed(knobs.SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return device.device()


def today(context):
    """The decision day as YYYYMMDD (`inference_at` is Asia/Shanghai)."""

    return context.inference_at.strftime("%Y%m%d")


def titles(context, p, calendar, start):
    """(per-row vectors, (code position, day position) per row) of every title
    a key from decision `start` on can see; (None, None) with the text off."""

    if not knobs.TEXT:
        return None, None
    since = calendar[max(start - knobs.TEXT_DAYS + 1, 1)]
    rows = text.read(context, calendar, since)
    unique, inverse = np.unique(rows["title"].to_numpy(dtype=object).astype(str), return_inverse=True)
    table = encoder.embed(context, list(unique))
    encoder.retain(unique)
    code = pd.Index(p.codes).get_indexer(rows["ts_code"].astype(str))
    at = pd.Index(calendar).get_indexer(rows["day"].astype(str))
    return table[inverse], (code, at)


def keys(p, at, vectors, positions):
    """(codes, (cases, KEY) float32 keys) of the names whose pattern is valid at
    the decision numbered `at` in the calendar, which read bar at - 1."""

    bar = at - 1
    pattern = features.patterns(p, bar)
    blocks = [pattern]
    if knobs.TEXT:
        mean, count = text.means(positions[0], positions[1], vectors, len(p.codes), at)
        norm = np.linalg.norm(mean, axis=1, keepdims=True)
        unit = np.where(norm > 0, mean / np.maximum(norm, 1e-12), 0.0)
        blocks.append(unit * np.sqrt(knobs.TEXT_WEIGHT * features.PATTERN))
    valid = p.valid[bar]
    return [p.codes[i] for i in np.flatnonzero(valid)], np.concatenate(blocks, axis=1)[valid].astype("float32")


def mature(context, p, filed, pending):
    """File the outcome of every pending day whose label the panel now closes."""

    still = []
    for day in pending:
        if day not in p.dates:
            if day != pending[-1]:
                raise RuntimeError(f"pending day {day} is older than this fit's panel")
            still.append(day)
            continue
        bar = p.dates.index(day) - 1
        if p.due[bar] is None:
            still.append(day)
            continue
        codes, _ = memory.read_keys(context, day)
        column = pd.Index(p.codes).get_indexer(codes.astype(str))
        memory.write_outcome(context, day, np.where(column >= 0, p.label[bar][np.maximum(column, 0)], np.nan))
        filed.append(day)
    return filed, still


def score(context, filed, query, dev):
    """(today's scores, a short metadata mapping) by knobs.SCORER."""

    mem = memory.load(context, filed, dev)
    meta = {"scorer": knobs.SCORER, "memory_days": len(mem.days), "memory_cases": mem.n}
    if knobs.SCORER == "retrieval":
        idx, d2 = memory.retrieve(mem, query, knobs.K)
        meta["median_d2"] = round(float(np.median(d2)), 3)
        meta["neighbour_days"] = int(np.median([len(np.unique(mem.day[row])) for row in idx]))
        return memory.predict(mem, idx, d2), meta
    quarter = f"{context.inference_at.year}Q{(context.inference_at.month - 1) // 3 + 1}"
    try:
        record = torch.load(context.state_dir + HEAD_FILE, map_location=dev, weights_only=True)
    except FileNotFoundError:
        record = None
    if record is None or str(record["quarter"]) != quarter:
        net = model.fit(mem, today(context), dev)
        torch.save({"model": net.state_dict(), "width": int(query.shape[1]), "quarter": quarter}, context.state_dir + HEAD_FILE)
    else:
        net = model.Head(int(record["width"])).to(dev)
        net.load_state_dict(record["model"])
    with torch.no_grad():
        values = net.eval()(torch.as_tensor(query, device=dev)).cpu().numpy()
    meta["fit_quarter"] = quarter
    return values, meta
