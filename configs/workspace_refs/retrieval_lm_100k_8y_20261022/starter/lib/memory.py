"""The case memory: every matured (name, decision day) case of this replay, kept in `context.state_dir`.

A case is one name at one decision day D: its key (the pattern and text the
decision of D could see, lib/learn.keys) and its outcome (the residual label
of the bar D read, lib/features.py). The key is filed by D's own fit; the
outcome only by the first fit whose panel closes the label's forward window,
HORIZON + 1 bars later, and a case enters the memory only then. Retrieval
therefore never sees an outcome a decision could not have known, and every
case was keyed with exactly the data its own decision had.

Files, flat under the state directory (a strategy cannot make a directory or
list one, so the index names every file):

    keys_<D>.npz      codes (bytes) and key (cases, KEY) float16: the keys of
                      every name with a valid pattern at decision D
    outcome_<D>.npz   y (cases,) float32 aligned with keys_<D>, NaN where the
                      label is undefined
    memory.npz        filed: the decision days whose outcomes are filed, in
                      order; pending: the days keyed but not yet matured

    index(context)               (filed, pending) lists, or None before the
                                 replay's first fit
    save_index(context, filed, pending)
    write_keys(context, day, codes, key)
    write_outcome(context, day, y)
    read_keys(context, day)      (codes, key) of one keyed day
    load(context, filed, dev)    the filed cases as one Memory, kept in this
                                 process and extended by the days it lacks
    retrieve(memory, query, k)   (indices, squared distances) of each query
                                 key's k nearest cases
    predict(memory, idx, d2)     the kernel-weighted outcome of each query
    swapped(codes, y, day)       the placebo's outcomes: y permuted within the
                                 day by a fixed hash of each case's code

The process cache is only a speed-up: it holds exactly the float16 keys the
files hold, so a cold worker -- each Paper morning starts one -- reads the
same memory from the files and retrieves the same neighbours. The state
directory is discarded with the replay; nothing here outlives it.

Neighbours are found on the card in float16 chunks of CHUNK cases and their
distances recomputed exactly in float32; on the CPU the same code runs in
float32.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from lib import knobs

INDEX_FILE = "/memory.npz"
CHUNK = 65536
_CACHE = {}


@dataclass
class Memory:
    days: list  # the filed decision days, in order
    key: torch.Tensor  # (capacity, KEY) on the device; rows [:n] are cases
    norm: torch.Tensor  # (capacity,) squared key lengths, same dtype
    y: np.ndarray  # (n,) float32 outcomes
    outcome: np.ndarray  # (n,) float32 what a retrieval averages: `y`, or under
    # knobs.NEIGHBOURS == "placebo" each case's outcome swapped within its day
    day: np.ndarray  # (n,) int32 position of each case's day in `days`
    code: np.ndarray  # (n,) bytes ts_code
    n: int


def index(context):
    try:
        record = np.load(context.state_dir + INDEX_FILE)
    except FileNotFoundError:
        return None
    return [str(day) for day in record["filed"].astype(str)], [str(day) for day in record["pending"].astype(str)]


def save_index(context, filed, pending):
    np.savez(
        context.state_dir + INDEX_FILE,
        filed=np.array(list(filed), dtype="S8"),
        pending=np.array(list(pending), dtype="S8"),
    )


def write_keys(context, day, codes, key):
    np.savez(
        context.state_dir + f"/keys_{day}.npz",
        codes=np.array(list(codes), dtype="S12"),
        key=np.asarray(key, dtype="float16"),
    )


def write_outcome(context, day, y):
    np.savez(context.state_dir + f"/outcome_{day}.npz", y=np.asarray(y, dtype="float32"))


def read_keys(context, day):
    record = np.load(context.state_dir + f"/keys_{day}.npz")
    return record["codes"], record["key"]


def load(context, filed, dev):
    """The memory of the filed days on `dev`, extended in place when this
    process already holds an earlier prefix of them."""

    cached = _CACHE.get("memory")
    if cached is None or cached.key.device != dev or cached.days != list(filed)[: len(cached.days)]:
        cached = _empty(dev)
    days, codes, keys, ys = [], [], [], []
    for day in list(filed)[len(cached.days):]:
        code, key = read_keys(context, day)
        y = np.load(context.state_dir + f"/outcome_{day}.npz")["y"]
        keep = np.isfinite(y)
        days.append(day)
        codes.append(code[keep])
        keys.append(key[keep])
        ys.append(y[keep])
    if days:
        _append(cached, days, codes, keys, ys)
    _CACHE["memory"] = cached
    return cached


def retrieve(memory, query, k):
    """(m, k) int64 indices into the memory and (m, k) float32 squared
    distances of each query row's k nearest cases, nearest first."""

    if memory.n < k:
        raise RuntimeError(f"the memory holds {memory.n} cases, fewer than k = {k}")
    dev = memory.key.device
    q = torch.as_tensor(np.asarray(query), device=dev).to(memory.key.dtype)
    best_d = best_i = None
    for start in range(0, memory.n, CHUNK):
        stop = min(start + CHUNK, memory.n)
        # The query's own length is the same for every case and left out.
        d = torch.addmm(memory.norm[start:stop].unsqueeze(0), q, memory.key[start:stop].T, beta=1, alpha=-2)
        values, positions = torch.topk(d, min(k, stop - start), dim=1, largest=False, sorted=True)
        positions = positions + start
        if best_d is None:
            best_d, best_i = values, positions
        else:
            merged_d = torch.cat([best_d, values], dim=1)
            merged_i = torch.cat([best_i, positions], dim=1)
            order = torch.sort(merged_d, dim=1, stable=True).indices[:, :k]
            best_d, best_i = torch.gather(merged_d, 1, order), torch.gather(merged_i, 1, order)
    exact = ((memory.key[best_i].float() - q.float().unsqueeze(1)) ** 2).sum(dim=2)
    order = torch.sort(exact, dim=1, stable=True).indices
    best_i = torch.gather(best_i, 1, order)
    exact = torch.gather(exact, 1, order)
    return best_i.cpu().numpy(), exact.cpu().numpy()


def predict(memory, idx, d2):
    """(m,) the kernel-weighted outcome of each query's neighbours: weights
    exp(-d2 / (BANDWIDTH * median d2)), the retrieved cases' own outcomes or,
    under knobs.NEIGHBOURS == "placebo", their swapped ones."""

    scale = knobs.BANDWIDTH * np.maximum(np.median(d2, axis=1, keepdims=True), 1e-6)
    logits = -(d2 - d2[:, :1]) / scale
    weights = np.exp(logits)
    weights /= weights.sum(axis=1, keepdims=True)
    return (weights * memory.outcome[idx]).sum(axis=1)


def swapped(codes, y, day):
    """`y` with each case's outcome replaced by the next case's in a fixed
    hash order of (code, day): a within-day permutation with no fixed point
    that does not depend on the order the cases were filed in."""

    labels = np.array([f"{code.decode() if isinstance(code, bytes) else code}|{day}|placebo" for code in codes], dtype=object)
    order = np.argsort(pd.util.hash_array(labels), kind="stable")
    out = np.empty_like(y)
    out[order] = y[np.roll(order, -1)]
    return out


def _empty(dev):
    dtype = torch.float16 if dev.type == "cuda" else torch.float32
    width = 2 * knobs.WINDOW + (knobs.TEXT_DIM if knobs.TEXT else 0)
    return Memory(
        days=[], key=torch.zeros((0, width), dtype=dtype, device=dev), norm=torch.zeros(0, dtype=dtype, device=dev),
        y=np.zeros(0, "float32"), outcome=np.zeros(0, "float32"), day=np.zeros(0, "int32"),
        code=np.zeros(0, "S12"), n=0,
    )


def _append(memory, days, codes, keys, ys):
    """Add the cases of `days` (lists aligned day by day) in one copy."""

    rows = sum(len(y) for y in ys)
    if memory.n + rows > memory.key.shape[0]:
        capacity = max(2 * memory.key.shape[0], memory.n + rows, CHUNK)
        key_grown = torch.zeros((capacity, memory.key.shape[1]), dtype=memory.key.dtype, device=memory.key.device)
        norm_grown = torch.zeros(capacity, dtype=memory.norm.dtype, device=memory.norm.device)
        key_grown[: memory.n] = memory.key[: memory.n]
        norm_grown[: memory.n] = memory.norm[: memory.n]
        memory.key, memory.norm = key_grown, norm_grown
    block = np.concatenate(keys).astype("float16")
    for start in range(0, rows, CHUNK):
        part = torch.as_tensor(block[start:start + CHUNK]).to(device=memory.key.device, dtype=memory.key.dtype)
        memory.key[memory.n + start : memory.n + start + len(part)] = part
        memory.norm[memory.n + start : memory.n + start + len(part)] = (part.float() ** 2).sum(dim=1).to(memory.norm.dtype)
    y = np.concatenate(ys).astype("float32")
    if knobs.NEIGHBOURS == "placebo":
        outcome = np.concatenate([swapped(code, part.astype("float32"), day) for day, code, part in zip(days, codes, ys)])
    else:
        outcome = y
    first = len(memory.days)
    memory.day = np.concatenate([memory.day, np.repeat(np.arange(first, first + len(days), dtype="int32"), [len(part) for part in ys])])
    memory.y = np.concatenate([memory.y, y])
    memory.outcome = np.concatenate([memory.outcome, outcome]) if knobs.NEIGHBOURS == "placebo" else memory.y
    memory.code = np.concatenate([memory.code, *[np.asarray(code, dtype="S12") for code in codes]])
    memory.days.extend(days)
    memory.n += rows
