"""The four heads, the bag and the one training loop every member goes through.

Heads. Each maps a (names, SEQ_LEN, 12) window from `panel.window` to one score
per name. They are the heads the single-head arms of rounds 20260925/26 ran on
this recipe, with the input width cut from 14 to the 12 live channels:

    mlp    last step and window mean concatenated -> Linear(24, 64) -> ReLU -> Linear(64, 1)
    lstm   LSTM(12 -> 64) -> LayerNorm on the last output -> Linear(64, 1)
    tcn    Conv1d(12, 64, k5) -> ReLU -> Conv1d(64, 64, k5, dilation 2) -> ReLU
           -> last step -> Linear(64, 1)
    attn   Linear(12, 64) -> 4-head self-attention over the 60 steps (no positional
           encoding, as in the single-head arm) -> last step -> LayerNorm -> Linear(64, 1)

Early fusion (`knobs.FUND == "early"`) widens only each head's last linear
layer: it reads the 64 hidden units followed by `fusion.WIDTH` fundamentals
inputs of the same name and date. Every other layer is built, initialised and
trained exactly as without it, so with the switch off the bag is the 100k
bag's to the bit.

The bag is `b4s2`: every head, two seeds each. Member k of head h trains with
seed `knobs.SEED_BASE` + 100 x (position of h in `HEADS`) + k, so the
legs on another seed base differ in one constant.

Training, one loop for every member (the recipe's): training dates are the
signal dates of the trailing `TRAIN_YEARS` whose label is realised by T-1;
validation is the last `VALID_DAYS` of them and training stops `EMBARGO_DAYS`
(the label horizon) before the validation start. One date is one batch (the
whole tradable cross-section); loss = -Pearson(score, label rank); Adam. A
cold refit trains `COLD_EPOCHS` at `COLD_LR` from a fresh initialisation; a
warm refit continues the member's checkpoint for `WARM_EPOCHS` at `WARM_LR`
and starts its early stopping from the loaded weights' validation IC on the
new window, so it keeps the old weights when it cannot beat them. Patience
`PATIENCE`; the best epoch is restored and saved under `context.state_dir`.
Which refits are cold is `lib/state.py`'s counter.

The training inputs of one refit are built once (half precision on the device)
and reused by every member; a member's model and optimiser are freed before
the next one starts, so the device peak is the cache plus the largest head.

Device: CUDA only. The arm is created with one GPU; a container without one is
a different computation, so `device()` stops instead of falling back.
"""

import numpy as np
import pandas as pd
import torch
from torch import nn

from lib import fusion, knobs, label, panel as P

HIDDEN = 64
TRAIN_YEARS = 3
FIT_CALENDAR_DAYS = int(365.25 * TRAIN_YEARS) + 150   # the window plus SEQ_LEN bars of warm-up
VALID_DAYS = 60
EMBARGO_DAYS = label.HOLD
COLD_EPOCHS = 40
COLD_LR = 1e-3
WARM_EPOCHS = 10
WARM_LR = 3e-4
PATIENCE = 5
MIN_NAMES = 30
PAD_NAMES = 256              # training batches are padded to a multiple of this many rows
SEEDS_PER_HEAD = 2


def _join(hidden, extra):
    return hidden if extra is None else torch.cat([hidden, extra], dim=1)


class MLPHead(nn.Module):
    """Last step and mean of the sequence, no recurrence and no convolution."""

    def __init__(self, extra=0):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(P.SEQ_FEATURES * 2, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN + extra, 1))

    def forward(self, x, extra=None):
        hidden = self.net[1](self.net[0](torch.cat([x[:, -1], x.mean(dim=1)], dim=1)))
        return self.net[2](_join(hidden, extra)).squeeze(-1)


class LSTMHead(nn.Module):
    """One LSTM layer over the sequence."""

    def __init__(self, extra=0):
        super().__init__()
        self.lstm = nn.LSTM(P.SEQ_FEATURES, HIDDEN, batch_first=True)
        self.norm = nn.LayerNorm(HIDDEN)
        self.head = nn.Linear(HIDDEN + extra, 1)

    def forward(self, x, extra=None):
        out, _ = self.lstm(x)
        return self.head(_join(self.norm(out[:, -1]), extra)).squeeze(-1)


class TCNHead(nn.Module):
    """Two temporal convolutions, the second dilated, read at the last step."""

    def __init__(self, extra=0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(P.SEQ_FEATURES, HIDDEN, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(HIDDEN, HIDDEN, kernel_size=5, dilation=2, padding=4),
            nn.ReLU(),
        )
        self.head = nn.Linear(HIDDEN + extra, 1)

    def forward(self, x, extra=None):
        return self.head(_join(self.net(x.transpose(1, 2))[:, :, -1], extra)).squeeze(-1)


class AttnHead(nn.Module):
    """Self-attention along each name's own sequence, read at the last step."""

    def __init__(self, extra=0):
        super().__init__()
        self.proj = nn.Linear(P.SEQ_FEATURES, HIDDEN)
        self.attn = nn.MultiheadAttention(HIDDEN, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(HIDDEN)
        self.head = nn.Linear(HIDDEN + extra, 1)

    def forward(self, x, extra=None):
        h = self.proj(x)
        out, _ = self.attn(h, h, h, need_weights=False)
        return self.head(_join(self.norm(out[:, -1]), extra)).squeeze(-1)


HEADS = {"mlp": MLPHead, "lstm": LSTMHead, "tcn": TCNHead, "attn": AttnHead}


def members():
    """[(head, seed)] of the bag."""

    return [(head, knobs.SEED_BASE + 100 * k + j) for k, head in enumerate(HEADS) for j in range(SEEDS_PER_HEAD)]


def _net(head, early, dev):
    return HEADS[head](fusion.WIDTH if early else 0).to(dev)


def device():
    if not torch.cuda.is_available():
        raise RuntimeError("no CUDA device in the strategy container; this package does not run on the CPU")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return torch.device("cuda")


def _release(dev):
    torch.cuda.empty_cache()


def _path(context, head, seed):
    return context.state_dir + "/" + head + "_seed" + str(seed) + ".pt"


def _checkpoint(context, head, seed, dev):
    try:
        return torch.load(_path(context, head, seed), map_location=dev)
    except FileNotFoundError:
        return None


def scorable(data, t):
    """Name indexes tradable at date index t."""

    return np.nonzero(P.tradable(data, t))[0]


def _pearson(a, b):
    a = a - a.mean()
    b = b - b.mean()
    return (a * b).mean() / (a.std() * b.std() + 1e-8)


def fit_dates(context, data):
    """(training, validation) date indexes of one refit; shared by every member and every tree."""

    dates = data["dates"]
    last = len(dates) - 1
    first = (pd.Timestamp(context.inference_at.date()) - pd.DateOffset(years=TRAIN_YEARS)).strftime("%Y%m%d")
    candidates = [t for t in range(P.SEQ_LEN - 1, last - label.HOLD) if dates[t] >= first]
    if len(candidates) < VALID_DAYS + EMBARGO_DAYS + 100:
        raise RuntimeError(f"fit has only {len(candidates)} labelled dates in its window")
    valid = candidates[-VALID_DAYS:]
    return [t for t in candidates if t <= valid[0] - EMBARGO_DAYS], valid


def _extra(fx, data, t, idx, dev):
    """Early fusion's inputs for the names idx of date index t on the device, or None without it."""

    if fx is None:
        return None
    return torch.from_numpy(fusion.early_inputs(fusion.take(fx, data, t, idx))).to(dev)


def batches(data, dates, dev, fx):
    """{date index: (half-precision inputs, half-precision fundamentals or None, label ranks)}.

    Each date's inputs are zero-padded to a multiple of `PAD_NAMES` rows and
    the predictions cut back to the date's own names: rows never interact in
    any head, so the scores are unchanged, while the device sees a handful of
    batch shapes instead of one per date -- with one shape per date the LSTM's
    cuDNN buffers fragment the allocator's cache to several times the memory
    actually in use.
    """

    out = {}
    for t in dates:
        idx = scorable(data, t)
        idx = idx[np.isfinite(data["yrank"][t, idx])]
        if len(idx) < MIN_NAMES:
            continue
        rows = -(-len(idx) // PAD_NAMES) * PAD_NAMES
        inputs = torch.zeros((rows, P.SEQ_LEN, P.SEQ_FEATURES), dtype=torch.float16, device=dev)
        inputs[: len(idx)] = torch.from_numpy(P.window(data, t, idx)).to(dev).half()
        extra = _extra(fx, data, t, idx, dev)
        if extra is not None:
            padded = torch.zeros((rows, fusion.WIDTH), dtype=torch.float16, device=dev)
            padded[: len(idx)] = extra.half()
            extra = padded
        target = torch.from_numpy(data["yrank"][t, idx].astype(np.float32)).to(dev)
        out[int(t)] = (inputs, extra, target)
    return out


def _predict(net, inputs, extra, target):
    return net(inputs.float(), None if extra is None else extra.float())[: len(target)]


def _validate(net, cache, valid):
    net.eval()
    with torch.no_grad():
        scores = [_pearson(_predict(net, *cache[t]), cache[t][2]) for t in valid if t in cache]
    if not scores:
        raise RuntimeError("the validation segment holds no scorable date")
    return float(torch.stack(scores).mean())


def _snapshot(net):
    return {key: value.detach().clone() for key, value in net.state_dict().items()}


def _train_member(context, head, seed, cache, order, valid, cold, early, dev):
    """Train one (head, seed) member, save its best weights, return its best validation IC."""

    torch.manual_seed(seed)
    shuffler = np.random.default_rng(seed)
    net = _net(head, early, dev)
    best, best_state = -np.inf, None
    if not cold:
        checkpoint = _checkpoint(context, head, seed, dev)
        if checkpoint is None:
            raise RuntimeError(f"warm refit without a checkpoint for {head} seed {seed}")
        net.load_state_dict(checkpoint)
        best, best_state = _validate(net, cache, valid), _snapshot(net)
    optimizer = torch.optim.Adam(net.parameters(), lr=COLD_LR if cold else WARM_LR)
    stale = 0
    for _ in range(COLD_EPOCHS if cold else WARM_EPOCHS):
        net.train()
        for t in shuffler.permutation(order):
            inputs, extra, target = cache[int(t)]
            loss = -_pearson(_predict(net, inputs, extra, target), target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        now = _validate(net, cache, valid)
        if now > best + 1e-5:
            best, best_state, stale = now, _snapshot(net), 0
        else:
            stale += 1
            if stale >= PATIENCE:
                break
    if best_state is None:
        raise RuntimeError(f"{head} seed {seed} never produced a finite validation IC")
    net.load_state_dict(best_state)
    torch.save(net.state_dict(), _path(context, head, seed))
    del net, optimizer, best_state
    _release(dev)
    return best


def fit(context, data, cold, fx):
    """Train every member of the bag on one shared cache; mean best validation IC.

    `fx` is `fusion.features` for early fusion and None otherwise.
    """

    dev = device()
    train, valid = fit_dates(context, data)
    cache = batches(data, train + valid, dev, fx)
    order = [t for t in train if t in cache]
    if not order:
        raise RuntimeError("the training segment holds no scorable date")
    readings = [_train_member(context, head, seed, cache, order, valid, cold, fx is not None, dev)
                for head, seed in members()]
    del cache
    _release(dev)
    return float(np.mean(readings))


def score(context, data, fx):
    """Mean per-member cross-sectional rank in [0, 1] on the newest row; NaN off the tradable set."""

    dev = device()
    last = len(data["dates"]) - 1
    idx = scorable(data, last)
    out = np.full(len(data["codes"]), np.nan)
    if len(idx) < MIN_NAMES:
        return out
    inputs = torch.from_numpy(P.window(data, last, idx)).to(dev)
    extra = _extra(fx, data, last, idx, dev)
    ranks = []
    with torch.no_grad():
        for head, seed in members():
            checkpoint = _checkpoint(context, head, seed, dev)
            if checkpoint is None:
                raise RuntimeError(f"no fitted checkpoint for {head} seed {seed} under the state directory")
            net = _net(head, fx is not None, dev)
            net.load_state_dict(checkpoint)
            net.eval()
            prediction = net(inputs, extra)
            ranks.append(torch.argsort(torch.argsort(prediction)).double() / max(1, len(idx) - 1))
    out[idx] = torch.stack(ranks).mean(0).cpu().numpy()
    del inputs
    _release(dev)
    return out
