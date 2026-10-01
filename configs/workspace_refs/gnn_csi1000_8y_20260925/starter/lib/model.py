"""Relational graph ranker over one date's constituent cross-section. CUDA only.

Architecture. A node is a constituent on that date; its feature vector is the
158 z-scored operators of `lib/panel.py`. An encoder lifts it to HIDDEN
dimensions; GNN_LAYERS relational layers then mix each node with its
neighbours on two graphs at once -- SW L1 industry co-membership (static) and
a CORR_WINDOW-day return-correlation kNN (dynamic, rebuilt for every date) --
each with its own value projection and, under GRAPH_MODE "gat", its own
attention over that neighbourhood; "mean" replaces the attention with the
degree-normalised mean, which is the GCN reading of the same layer. Both
graphs carry self-loops, every layer is residual and LayerNormed, and a linear
head reads one score off each node.

Training. One date is one batch and one graph; loss = -Pearson(score, label
rank) on that date's constituents. Adam, early stop on the mean validation IC
with PATIENCE, best epoch restored, one model per seed in SEEDS, and a review
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

RELATIONS = 2
STATE_PREFIX = "/gnn_seed"
META_FILE = "/fit_meta.npy"
ATTENTION_FLOOR = -1.0e9
EPS = 1e-8


def cuda_device():
    if not torch.cuda.is_available():
        raise RuntimeError("this arm trains on CUDA: create the experiment with gpu_count >= 1")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return torch.device("cuda")


class RelationLayer(nn.Module):
    """One residual message-passing step over RELATIONS masked neighbourhoods."""

    def __init__(self, dim, mode):
        super().__init__()
        self.mode = mode
        self.own = nn.Linear(dim, dim)
        self.value = nn.ModuleList(nn.Linear(dim, dim) for _ in range(RELATIONS))
        self.left = nn.ModuleList(nn.Linear(dim, 1) for _ in range(RELATIONS))
        self.right = nn.ModuleList(nn.Linear(dim, 1) for _ in range(RELATIONS))
        self.norm = nn.LayerNorm(dim)

    def forward(self, h, adjacency):
        message = self.own(h)
        for relation, adjacent in enumerate(adjacency):
            value = self.value[relation](h)
            if self.mode == "gat":
                logit = nn.functional.leaky_relu(
                    self.left[relation](value) + self.right[relation](value).transpose(0, 1), 0.2)
                weight = torch.softmax(logit.masked_fill(~adjacent, ATTENTION_FLOOR), dim=1)
            else:
                weight = adjacent.to(value.dtype)
                weight = weight / weight.sum(1, keepdim=True).clamp(min=1.0)
            message = message + weight @ value
        return self.norm(h + nn.functional.gelu(message))


class RelationRanker(nn.Module):
    def __init__(self, features, use_graph):
        super().__init__()
        self.encode = nn.Sequential(
            nn.LayerNorm(features), nn.Linear(features, knobs.HIDDEN), nn.GELU())
        layers = [RelationLayer(knobs.HIDDEN, knobs.GRAPH_MODE)
                  for _ in range(knobs.GNN_LAYERS)] if use_graph else []
        self.layers = nn.ModuleList(layers)
        self.drop = nn.Dropout(knobs.DROPOUT)
        self.head = nn.Linear(knobs.HIDDEN, 1)

    def forward(self, x, adjacency):
        h = self.drop(self.encode(x))
        for layer in self.layers:
            h = self.drop(layer(h, adjacency))
        return self.head(h).squeeze(-1)


def graphs(returns, industry, rows, device):
    """(industry co-membership, correlation kNN) boolean adjacencies with self-loops."""

    count = len(rows)
    eye = torch.eye(count, dtype=torch.bool, device=device)
    label = industry[rows]
    same = (label.unsqueeze(1) == label.unsqueeze(0)) | eye
    window = torch.nan_to_num(returns[:, rows], nan=0.0)
    centred = window - window.mean(0, keepdim=True)
    centred = centred / (centred.std(0, keepdim=True) + EPS)
    correlation = centred.transpose(0, 1) @ centred / max(1, window.shape[0] - 1)
    correlation.fill_diagonal_(-2.0)
    neighbours = torch.zeros((count, count), dtype=torch.bool, device=device)
    k = int(min(knobs.KNN_K, count - 1))
    if k > 0:
        neighbours.scatter_(1, correlation.topk(k, dim=1).indices, True)
    return same, neighbours | neighbours.transpose(0, 1) | eye


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


def _batches(data, dates, device, use_graph):
    """{date index: (row indexes, adjacency, label)} held on the card for every epoch."""

    returns = torch.from_numpy(np.ascontiguousarray(data["returns"], dtype=np.float32)).to(device)
    industry = torch.from_numpy(data["industry"]).to(device)
    ranks = torch.from_numpy(data["yrank"]).to(device)
    out = {}
    for t in dates:
        rows = np.nonzero(data["member"][t] & np.isfinite(data["yrank"][t]))[0]
        index = torch.from_numpy(rows).to(device)
        window = returns[t - knobs.CORR_WINDOW + 1: t + 1]
        adjacency = graphs(window, industry, index, device) if use_graph else ()
        out[int(t)] = (index, adjacency, ranks[t, index].float())
    return out


def _epoch_ic(model, features, batches, dates):
    model.eval()
    with torch.no_grad():
        scores = []
        for t in dates:
            rows, adjacency, label = batches[int(t)]
            scores.append(_pearson(model(features[t, rows], adjacency), label))
        return float(torch.stack(scores).mean())


def fit(context, use_graph):
    device = cuda_device()
    data, train, valid = panel.fit_window(context)
    features = torch.from_numpy(data["feat"]).to(device)
    batches = _batches(data, train + valid, device, use_graph)
    refits = _refits(context) + 1
    meta = []
    for seed in knobs.SEEDS:
        torch.manual_seed(seed)
        shuffle = np.random.default_rng(seed + refits)
        model = RelationRanker(features.shape[2], use_graph).to(device)
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
                rows, adjacency, label = batches[int(t)]
                loss = -_pearson(model(features[t, rows], adjacency), label)
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


def score(context, data, use_graph):
    """Mean per-seed rank in [0, 1] of every scorable name on the newest row."""

    device = cuda_device()
    last = len(data["dates"]) - 1
    rows = panel.scorable(data, last)
    if len(rows) == 0:
        raise RuntimeError("no visible constituent with a bar on the newest row")
    index = torch.from_numpy(rows).to(device)
    features = torch.from_numpy(data["feat"][last, rows]).to(device)
    adjacency = ()
    if use_graph:
        returns = torch.from_numpy(
            np.ascontiguousarray(data["returns"][last - knobs.CORR_WINDOW + 1: last + 1],
                                 dtype=np.float32)).to(device)
        industry = torch.from_numpy(data["industry"]).to(device)
        adjacency = graphs(returns, industry, index, device)
    ranks = []
    with torch.no_grad():
        for seed in knobs.SEEDS:
            model = RelationRanker(features.shape[1], use_graph).to(device)
            model.load_state_dict(torch.load(_state_path(context, seed), map_location=device))
            model.eval()
            prediction = model(features, adjacency)
            ranks.append(torch.argsort(torch.argsort(prediction)).double()
                         / max(1, len(rows) - 1))
    out = np.full(len(data["codes"]), np.nan)
    out[rows] = torch.stack(ranks).mean(0).cpu().numpy()
    del features
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
