"""Market-gated cross-sectional attention over one date's constituents. CUDA only.

Architecture. A name is one token; its input is the last WINDOW_DAYS rows of
the 158 z-scored operators of `lib/panel.py`, so the temporal and the
cross-sectional halves read the same feature panel the control reads. A linear
projection plus a GRU reduce each name's window to one HIDDEN-dimensional
state. The benchmark's own state on that date (five fixed-scale numbers from
`market_state`) becomes a sigmoid gate that scales every name's state --
MASTER's market guidance, reduced to its one load-bearing part. One
multi-head attention layer then runs ACROSS the names of that date, followed
by a residual feed-forward mix, and a linear head reads one score off each
name.

The gate and the attention are separable on purpose: MARKET_GATE off leaves
plain cross-sectional attention, and the `c_temporal` candidate leaves the
temporal encoder alone. An independent re-implementation in the literature
could not tell MASTER's gating apart from no gating, so this arm measures the
two parts rather than assuming the published pair.

Training. One date is one batch of names; loss = -Pearson(score, label rank)
on that date's constituents. Adam, early stop on the mean validation IC with
PATIENCE, best epoch restored, one model per seed in SEEDS, and a review
averages the per-seat ranks of the seeds.

Warm start. A refit reloads the previous weights from `state_dir` and
continues from them at WARM_LEARNING_RATE for WARM_EPOCHS instead of
re-initialising; `state_dir` is empty at the start of every replay, so the
first fit of a replay is always cold and only the later ones are warm. The
refit counter and each seed's validation IC are saved beside the weights and
ride out on the order metadata, so a replay shows how many refits ran and
whether the warm start helped.

There is no CPU path. This arm is created with a GPU; without a CUDA device a
fit and a review raise rather than quietly computing something else.
"""

import numpy as np
import torch
from torch import nn

from lib import knobs, panel

MARKET_FEATURES = 5
# Fixed scales, not fitted statistics: a normalisation estimated over the
# panel would differ between a training date and the decision row.
MARKET_SCALE = np.array([0.02, 0.04, 0.08, 0.25, 0.05])
MARKET_CLIP = 3.0
STATE_PREFIX = "/xs_seed"
META_FILE = "/fit_meta.npy"
EPS = 1e-8


def cuda_device():
    if not torch.cuda.is_available():
        raise RuntimeError("this arm trains on CUDA: create the experiment with gpu_count >= 1")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return torch.device("cuda")


def _finite(values):
    series = np.asarray(values, dtype=np.float64)
    if not np.isfinite(series).all():
        raise RuntimeError("the benchmark close series has gaps over the panel's trading days")
    return series


def market_state(bench_close):
    """(T, 5) benchmark state: 1/5/20-day return, 20-day volatility, gap to its mean."""

    close = _finite(bench_close)
    days = len(close)
    out = np.zeros((days, MARKET_FEATURES))
    for column, lag in enumerate((1, 5, 20)):
        out[lag:, column] = close[lag:] / close[:days - lag] - 1.0
    daily = np.zeros(days)
    daily[1:] = close[1:] / close[:-1] - 1.0
    for t in range(20, days):
        window = daily[t - 19: t + 1]
        out[t, 3] = float(np.std(window)) * np.sqrt(244.0)
        out[t, 4] = close[t] / float(np.mean(close[t - 19: t + 1])) - 1.0
    out[:, 3] = out[:, 3] - MARKET_SCALE[3]
    return np.clip(out / MARKET_SCALE, -MARKET_CLIP, MARKET_CLIP).astype(np.float32)


class MarketAttentionRanker(nn.Module):
    def __init__(self, features, use_attention):
        super().__init__()
        self.project = nn.Linear(features, knobs.HIDDEN)
        self.temporal = nn.GRU(knobs.HIDDEN, knobs.HIDDEN, batch_first=True)
        self.norm = nn.LayerNorm(knobs.HIDDEN)
        self.gate = nn.Sequential(
            nn.Linear(MARKET_FEATURES, knobs.HIDDEN), nn.GELU(),
            nn.Linear(knobs.HIDDEN, knobs.HIDDEN))
        self.attention = nn.MultiheadAttention(
            knobs.HIDDEN, knobs.HEADS, dropout=knobs.DROPOUT, batch_first=True
        ) if use_attention else None
        self.mix = nn.Sequential(
            nn.LayerNorm(knobs.HIDDEN), nn.Linear(knobs.HIDDEN, knobs.HIDDEN), nn.GELU(),
            nn.Linear(knobs.HIDDEN, knobs.HIDDEN))
        self.drop = nn.Dropout(knobs.DROPOUT)
        self.head = nn.Linear(knobs.HIDDEN, 1)

    def forward(self, x, market):
        state, _ = self.temporal(nn.functional.gelu(self.project(x)))
        h = self.drop(self.norm(state[:, -1]))
        if knobs.MARKET_GATE:
            h = h * torch.sigmoid(self.gate(market))
        if self.attention is not None:
            attended, _ = self.attention(
                h.unsqueeze(0), h.unsqueeze(0), h.unsqueeze(0), need_weights=False)
            h = h + self.drop(attended.squeeze(0))
        h = h + self.drop(self.mix(h))
        return self.head(h).squeeze(-1)


def _pearson(a, b):
    a = a - a.mean()
    b = b - b.mean()
    return (a * b).mean() / (a.std() * b.std() + EPS)


def _state_path(context, seed):
    return context.state_dir + STATE_PREFIX + str(seed) + ".pt"


def _previous(path, device):
    """The last refit's weights, or None on the first fit of a replay."""

    try:
        return torch.load(path, map_location=device)
    except FileNotFoundError:
        return None


def _refits(context):
    try:
        return int(np.load(context.state_dir + META_FILE)[0, 0])
    except FileNotFoundError:
        return 0


def _window(features, t, rows):
    """(names, WINDOW_DAYS, 158) input of one date."""

    return features[t - knobs.WINDOW_DAYS + 1: t + 1, rows].permute(1, 0, 2)


def _batches(data, dates, device):
    """{date index: (row indexes, market state, label)} for every epoch."""

    market = torch.from_numpy(market_state(data["bench_close"])).to(device)
    ranks = torch.from_numpy(data["yrank"]).to(device)
    out = {}
    for t in dates:
        rows = np.nonzero(data["member"][t] & np.isfinite(data["yrank"][t]))[0]
        index = torch.from_numpy(rows).to(device)
        out[int(t)] = (index, market[t], ranks[t, index].float())
    return out


def _epoch_ic(model, features, batches, dates):
    model.eval()
    with torch.no_grad():
        scores = []
        for t in dates:
            rows, market, label = batches[int(t)]
            scores.append(_pearson(model(_window(features, int(t), rows), market), label))
        return float(torch.stack(scores).mean())


def fit(context, use_attention):
    device = cuda_device()
    data, train, valid = panel.fit_window(context)
    features = torch.from_numpy(data["feat"]).to(device)
    batches = _batches(data, train + valid, device)
    refits = _refits(context) + 1
    meta = []
    for seed in knobs.SEEDS:
        torch.manual_seed(seed)
        shuffle = np.random.default_rng(seed + refits)
        model = MarketAttentionRanker(features.shape[2], use_attention).to(device)
        warm = _previous(_state_path(context, seed), device) if knobs.WARM_START else None
        if warm is not None:
            model.load_state_dict(warm)
        rate = knobs.WARM_LEARNING_RATE if warm is not None else knobs.LEARNING_RATE
        epochs = knobs.WARM_EPOCHS if warm is not None else knobs.MAX_EPOCHS
        optimizer = torch.optim.Adam(model.parameters(), lr=rate)
        best, best_state, stale, ran = -np.inf, None, 0, 0
        if warm is not None:
            # Score the weights this refit inherited on the NEW window first, so a
            # warm refit can only replace them with something that validates better.
            # Without this the best-of-N epochs is kept even when every one of them
            # is worse than what the previous refit already had.
            best = _epoch_ic(model, features, batches, valid)
            best_state = {key: value.detach().clone()
                          for key, value in model.state_dict().items()}
        for _ in range(epochs):
            model.train()
            for t in shuffle.permutation(np.array(train)):
                rows, market, label = batches[int(t)]
                loss = -_pearson(model(_window(features, int(t), rows), market), label)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            ran += 1
            score = _epoch_ic(model, features, batches, valid)
            if score > best + 1e-5:
                best, stale = score, 0
                best_state = {key: value.detach().clone()
                              for key, value in model.state_dict().items()}
            else:
                stale += 1
                if stale >= knobs.PATIENCE:
                    break
        if best_state is None:
            raise RuntimeError(f"seed {seed} never produced a finite validation IC")
        model.load_state_dict(best_state)
        torch.save(model.state_dict(), _state_path(context, seed))
        meta.append([refits, float(seed), float(ran), best, 1.0 if warm is not None else 0.0])
    np.save(context.state_dir + META_FILE, np.asarray(meta, dtype=np.float64))
    del batches, features, model, optimizer
    torch.cuda.empty_cache()


def score(context, data, use_attention):
    """Mean per-seed rank in [0, 1] of every scorable name on the newest row."""

    device = cuda_device()
    last = len(data["dates"]) - 1
    rows = panel.scorable(data, last)
    if len(rows) == 0:
        raise RuntimeError("no visible constituent with a bar on the newest row")
    index = torch.from_numpy(rows).to(device)
    features = torch.from_numpy(
        np.ascontiguousarray(data["feat"][last - knobs.WINDOW_DAYS + 1: last + 1])).to(device)
    inputs = features[:, index].permute(1, 0, 2)
    market = torch.from_numpy(market_state(data["bench_close"])[last]).to(device)
    ranks = []
    with torch.no_grad():
        for seed in knobs.SEEDS:
            model = MarketAttentionRanker(features.shape[2], use_attention).to(device)
            model.load_state_dict(torch.load(_state_path(context, seed), map_location=device))
            model.eval()
            prediction = model(inputs, market)
            ranks.append(torch.argsort(torch.argsort(prediction)).double()
                         / max(1, len(rows) - 1))
    out = np.full(len(data["codes"]), np.nan)
    out[rows] = torch.stack(ranks).mean(0).cpu().numpy()
    del features, inputs
    torch.cuda.empty_cache()
    return out


def report(context):
    """[refits, seeds, mean validation IC, warm] for the order metadata."""

    try:
        meta = np.load(context.state_dir + META_FILE)
    except FileNotFoundError:
        return {}
    return {"refits": int(meta[0, 0]),
            "epochs": int(np.max(meta[:, 2])),
            "valid_ic": round(float(np.mean(meta[:, 3])), 5),
            "warm_start": bool(meta[0, 4] > 0.0)}
