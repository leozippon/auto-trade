"""GRU sequence ranker with a persisted, warm-started refit. CUDA only.

Model: GRU(SEQ_FEATURES -> HIDDEN) over the last `panel.SEQ_LEN` days ->
LayerNorm -> Linear -> one score. Inputs and their normalisation come from
`panel.window`, so the control (`lib/tree.py`) sees the same numbers flattened.

Training dates of one refit are the signal dates of the trailing `TRAIN_YEARS`
whose N-day residual label is realised by T-1; validation is the last
`VALID_DAYS` of them and training stops `EMBARGO_DAYS` before the validation
start, so no training row's label window overlaps a validation row's. One date
is one batch; loss = -Pearson(score, label rank) over that date's constituents.

Persistence, the part no arm in this repository has done before. `fit` writes
one checkpoint per seed under `context.state_dir`, and the NEXT refit loads it
and continues training instead of starting from a fresh initialisation:

    refit 0 of a replay    the state directory is empty -> cold,
                           COLD_EPOCHS at COLD_LR
    refits 1, 2, 3         warm -> WARM_EPOCHS at WARM_LR from the checkpoint
    every state.COLD_EVERY-th refit   cold again, so a chain cannot run forever

A warm refit starts its early-stopping bookkeeping from the loaded weights' own
validation IC on the NEW window, so a refit that cannot improve on the new data
keeps the previous weights rather than shipping something worse. That is the
whole point: the model carries what it learned in the first research year into
the last one instead of rediscovering it from three years of data every quarter.

`state_dir` is empty at the start of every replay, so the first fit of a replay
is always cold and a replay stays reproducible from PIT data alone; `models_dir`
is read-only and carries static assets only, so nothing fitted ever goes there.

There is no CPU path. This arm is created with `gpu_count = 1`; on the
container's CPU one refit of this family does not fit in the fit budget at all,
so `fit` and a review raise rather than silently computing something else.
"""

import numpy as np
import pandas as pd
import torch
from torch import nn

from lib import panel as P

SEEDS = (1000, 1001, 1002)
HIDDEN = 64
TRAIN_YEARS = 3
FIT_CALENDAR_DAYS = int(365.25 * TRAIN_YEARS) + 150   # the window plus SEQ_LEN bars of warm-up
VALID_DAYS = 60
EMBARGO_DAYS = P.HOLD
COLD_EPOCHS = 40
COLD_LR = 1e-3
WARM_EPOCHS = 10
WARM_LR = 3e-4
PATIENCE = 5
MIN_NAMES = 30


class GRURanker(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(P.SEQ_FEATURES, HIDDEN, batch_first=True)
        self.norm = nn.LayerNorm(HIDDEN)
        self.head = nn.Linear(HIDDEN, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.head(self.norm(out[:, -1])).squeeze(-1)


def cuda_device():
    if not torch.cuda.is_available():
        raise RuntimeError("seq_gru_persist needs a CUDA device: create the arm with gpu_count >= 1")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return torch.device("cuda")


def state_path(context, seed):
    return context.state_dir + "/gru_seed" + str(seed) + ".pt"


def _checkpoint(context, seed, device):
    try:
        return torch.load(state_path(context, seed), map_location=device)
    except FileNotFoundError:
        return None


def scorable(data, t):
    """Name indexes tradable at date index t with a full sequence behind them."""

    return np.nonzero(P.tradable(data, t))[0]


def _pearson(a, b):
    a = a - a.mean()
    b = b - b.mean()
    return (a * b).mean() / (a.std() * b.std() + 1e-8)


def fit_dates(context, data):
    """(training, validation) date indexes of one refit; shared by the GRU and its control."""

    dates = data["dates"]
    last = len(dates) - 1
    first = (pd.Timestamp(context.inference_at.date()) - pd.DateOffset(years=TRAIN_YEARS)).strftime("%Y%m%d")
    candidates = [t for t in range(P.SEQ_LEN - 1, last - P.HOLD) if dates[t] >= first]
    if len(candidates) < VALID_DAYS + EMBARGO_DAYS + 100:
        raise RuntimeError(f"fit has only {len(candidates)} labelled dates in its window")
    valid = candidates[-VALID_DAYS:]
    return [t for t in candidates if t <= valid[0] - EMBARGO_DAYS], valid


def batches(data, dates, device):
    """{date index: (float16 inputs on the card, label ranks)} for the requested dates."""

    out = {}
    for t in dates:
        idx = scorable(data, t)
        idx = idx[np.isfinite(data["yrank"][t, idx])]
        if len(idx) < MIN_NAMES:
            continue
        inputs = torch.from_numpy(P.window(data, t, idx)).to(device).half()
        target = torch.from_numpy(data["yrank"][t, idx].astype(np.float32)).to(device)
        out[int(t)] = (inputs, target)
    return out


def _validate(model, cache, valid):
    model.eval()
    with torch.no_grad():
        scores = [_pearson(model(cache[t][0].float()), cache[t][1]) for t in valid if t in cache]
    if not scores:
        raise RuntimeError("the validation segment holds no scorable date")
    return float(torch.stack(scores).mean())


def fit(context, cold):
    """Train every seed on the trailing window; cold from scratch, warm from the checkpoint."""

    device = cuda_device()
    data = P.build(context, FIT_CALENDAR_DAYS, labels=True)
    train, valid = fit_dates(context, data)
    cache = batches(data, train + valid, device)
    order = [t for t in train if t in cache]
    if not order:
        raise RuntimeError("the training segment holds no scorable date")
    epochs = COLD_EPOCHS if cold else WARM_EPOCHS
    rate = COLD_LR if cold else WARM_LR
    reported = []
    for seed in SEEDS:
        torch.manual_seed(seed)
        shuffler = np.random.default_rng(seed)
        model = GRURanker().to(device)
        checkpoint = None if cold else _checkpoint(context, seed, device)
        if checkpoint is not None:
            model.load_state_dict(checkpoint)
        optimizer = torch.optim.Adam(model.parameters(), lr=rate)
        if checkpoint is None:
            best, best_state = -np.inf, None
        else:
            best = _validate(model, cache, valid)
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        stale = 0
        for _ in range(epochs):
            model.train()
            for t in shuffler.permutation(order):
                inputs, target = cache[int(t)]
                loss = -_pearson(model(inputs.float()), target)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            score_now = _validate(model, cache, valid)
            if score_now > best + 1e-5:
                best, stale = score_now, 0
                best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            else:
                stale += 1
                if stale >= PATIENCE:
                    break
        if best_state is None:
            raise RuntimeError(f"seed {seed} never produced a finite validation IC")
        model.load_state_dict(best_state)
        torch.save(model.state_dict(), state_path(context, seed))
        reported.append(best)
    del cache, model, optimizer
    torch.cuda.empty_cache()
    return float(np.mean(reported))


def score(context, data):
    """Mean per-seed cross-sectional rank in [0, 1] on the newest row; NaN off the universe."""

    device = cuda_device()
    last = len(data["dates"]) - 1
    idx = scorable(data, last)
    out = np.full(len(data["codes"]), np.nan)
    if len(idx) < MIN_NAMES:
        return out
    inputs = torch.from_numpy(P.window(data, last, idx)).to(device)
    ranks = []
    with torch.no_grad():
        for seed in SEEDS:
            checkpoint = _checkpoint(context, seed, device)
            if checkpoint is None:
                raise RuntimeError(f"no fitted checkpoint for seed {seed} under the state directory")
            model = GRURanker().to(device)
            model.load_state_dict(checkpoint)
            model.eval()
            prediction = model(inputs)
            ranks.append(torch.argsort(torch.argsort(prediction)).double() / max(1, len(idx) - 1))
    out[idx] = torch.stack(ranks).mean(0).cpu().numpy()
    del inputs
    torch.cuda.empty_cache()
    return out
