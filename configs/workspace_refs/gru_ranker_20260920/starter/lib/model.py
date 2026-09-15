"""GRU sequence ranker: trained in fit on the trailing three years, scored on a review day. CUDA only.

Model: GRU(8 -> 64) over the last SEQ_LEN days -> LayerNorm -> Linear -> one score.
Inputs of one date's cross-section: log(O, H, L, C, VWAP / C_t), log1p(V / mean V
of the window), log1p(turnover %), has-bar flag; each (lag, feature) is z-scored
across that date's names and clipped at +-3.

Training (one refit): signal dates t of the trailing TRAIN_YEARS whose 10-day label
is realized by T-1 (t + HOLD + 1 <= T-1); validation = the last VALID_DAYS of them,
training = the dates at least EMBARGO_DAYS before validation. One date is one
batch; loss = -Pearson(score, label rank). Adam at LEARNING_RATE, at most
MAX_EPOCHS, early stop on the mean validation IC with PATIENCE, best epoch
restored. One model per seed in SEEDS; a review scores with all of them and
averages their per-date ranks.

There is no CPU path: without a CUDA device `fit` and a review raise, so a replay
that did not get a GPU fails instead of producing a different computation.
"""

import numpy as np
import pandas as pd
import torch
from torch import nn

from lib import panel as P

SEEDS = (1000, 1001, 1002)
HIDDEN = 64
TRAIN_YEARS = 3
FIT_CALENDAR_DAYS = int(365.25 * TRAIN_YEARS) + 150   # the window plus SEQ_LEN bars of warm-up before its first date
VALID_DAYS = 60
EMBARGO_DAYS = 10
LEARNING_RATE = 1e-3
MAX_EPOCHS = 40
PATIENCE = 5


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
        raise RuntimeError("gru_ranker needs a CUDA device: create the arm with gpu_count >= 1")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return torch.device("cuda")


def state_path(context, seed):
    return context.state_dir + "/gru_seed" + str(seed) + ".pt"


def window(seq, t, idx):
    """(names, SEQ_LEN, 8) normalized inputs for date index t."""

    x = seq[t - P.SEQ_LEN + 1: t + 1, idx, :].permute(1, 0, 2)
    prices = torch.log(torch.clamp(x[:, :, 0:5] / x[:, -1:, 3:4], min=1e-4))
    volume = x[:, :, 5]
    log_volume = torch.log1p(volume / (volume.mean(1, keepdim=True) + 1e-9))
    log_turnover = torch.log1p(torch.clamp(x[:, :, 6], min=0.0))
    features = torch.cat([prices, log_volume.unsqueeze(2), log_turnover.unsqueeze(2), x[:, :, 7:8]], dim=2)
    features = torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    mean, std = features.mean(0, keepdim=True), features.std(0, keepdim=True)
    return torch.clamp((features - mean) / (std + 1e-6), -3.0, 3.0)


def scorable(panel_arrays, t):
    """Name indexes with a bar on day t and a full sequence behind it."""

    return np.nonzero(panel_arrays["has_bar"][t] & (panel_arrays["nbars"][t] >= P.SEQ_LEN))[0]


def _pearson(a, b):
    a = a - a.mean()
    b = b - b.mean()
    return (a * b).mean() / (a.std() * b.std() + 1e-8)


def fit_dates(context, data):
    """(training, validation) date indexes of one refit; shared by every learned candidate."""

    dates = data["dates"]
    last = len(dates) - 1
    first_date = (pd.Timestamp(context.inference_at.date()) - pd.DateOffset(years=TRAIN_YEARS)).strftime("%Y%m%d")
    candidates = [t for t in range(P.SEQ_LEN - 1, last - P.HOLD) if dates[t] >= first_date]
    if len(candidates) < VALID_DAYS + EMBARGO_DAYS + 100:
        raise RuntimeError(f"fit has only {len(candidates)} labelled dates in its window")
    valid = candidates[-VALID_DAYS:]
    return [t for t in candidates if t <= valid[0] - EMBARGO_DAYS], valid


def fit(context):
    device = cuda_device()
    data = P.build(context, FIT_CALENDAR_DAYS, labels=True)
    train, valid = fit_dates(context, data)
    seq = torch.from_numpy(data["seq"]).to(device)
    yrank = torch.from_numpy(data["yrank"]).to(device)
    batches = {}
    for t in train + valid:
        idx = scorable(data, t)
        idx = idx[np.isfinite(data["yrank"][t, idx])]
        index = torch.from_numpy(idx).to(device)
        # float16 on the card halves the cache (about 3 GiB for 60-day sequences); cast back per step
        batches[t] = (window(seq, t, index).half(), yrank[t, index])
    for seed in SEEDS:
        torch.manual_seed(seed)
        order = np.random.default_rng(seed)
        model = GRURanker().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        best, best_state, stale = -np.inf, None, 0
        for _ in range(MAX_EPOCHS):
            model.train()
            for t in order.permutation(train):
                x, y = batches[int(t)]
                loss = -_pearson(model(x.float()), y)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            model.eval()
            with torch.no_grad():
                score = float(torch.stack([_pearson(model(batches[t][0].float()), batches[t][1]) for t in valid]).mean())
            if score > best + 1e-5:
                best, stale = score, 0
                best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            else:
                stale += 1
                if stale >= PATIENCE:
                    break
        if best_state is None:
            raise RuntimeError(f"gru_ranker seed {seed} never produced a finite validation IC")
        model.load_state_dict(best_state)
        torch.save(model.state_dict(), state_path(context, seed))
    del batches, seq, yrank, model, optimizer
    torch.cuda.empty_cache()


def score(context, data):
    """Mean per-seed rank in [0, 1] of every scorable name on the newest row (NaN elsewhere)."""

    device = cuda_device()
    last = len(data["dates"]) - 1
    idx = scorable(data, last)
    seq = torch.from_numpy(data["seq"][last - P.SEQ_LEN + 1:]).to(device)
    x = window(seq, P.SEQ_LEN - 1, torch.from_numpy(idx).to(device))
    ranks = []
    with torch.no_grad():
        for seed in SEEDS:
            model = GRURanker().to(device)
            model.load_state_dict(torch.load(state_path(context, seed), map_location=device))
            model.eval()
            prediction = model(x)
            ranks.append(torch.argsort(torch.argsort(prediction)).double() / max(1, len(idx) - 1))
    out = np.full(len(data["codes"]), np.nan)
    out[idx] = torch.stack(ranks).mean(0).cpu().numpy()
    del seq, x
    torch.cuda.empty_cache()
    return out
