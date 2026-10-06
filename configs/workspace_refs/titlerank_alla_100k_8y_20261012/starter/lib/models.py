"""The two rankers behind one interface, and the validation metric they share.

Both map a unit -- the list of its fresh texts (`lib/ranker.py`) -- to a
score. Each is fitted on the training segment of one refit, chooses its one
search setting on the validation segment by the mean per-review rank IC, and
saves what scoring needs under `context.state_dir`; the score threshold of the
book is the (1 - TOP) quantile of its validation scores (`lib/ranker.py`).

ridge  The unit's texts joined by "|" and hashed into FEATURES character 1- to
       3-grams (presence, each row l2-normalised). The hash is fixed, so the
       vocabulary needs no fit and the twin has the same size at every refit.
       A ridge regression on the label rank, with alpha chosen from ALPHAS.
       Nothing in it is random.
nn     A character-level convolutional net in plain torch (`CharCNN`). Each
       text, cut to LENGTH characters, is embedded (EMBED), passed through one
       convolution per width in WIDTHS (CHANNELS each, ReLU) and max-pooled over
       its positions; the unit is the maximum over its texts, so a unit reads
       like its strongest document; a two-layer head (HIDDEN, dropout DROPOUT)
       gives the score. The vocabulary is every character seen at least
       MIN_CHARS times in that refit's training texts; the rest share one id.
       Adam (LR) on squared error to the label rank, BATCH units a step in an
       order drawn from `knobs.SEED` and the fit day, at most MAX_EPOCHS passes,
       stopped after PATIENCE passes without a better validation IC, the best
       pass restored. Weights start from `torch.manual_seed(knobs.SEED)`.

Device. The ridge runs on the CPU. The net is CUDA only: the arm owns one
card, and a net trained on a CPU would be another computation, so `device()`
stops instead of falling back. cuDNN is held to deterministic kernels, so a
replay re-trains the same weights.
"""

import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import Ridge
from torch import nn

from lib import knobs

FEATURES = 2 ** 18
NGRAMS = (1, 3)
ALPHAS = (1.0, 10.0, 100.0)
RIDGE_FILE = "/ridge.npz"

LENGTH = 48
MIN_CHARS = 20
EMBED = 64
CHANNELS = 128
WIDTHS = (2, 3, 4)
HIDDEN = 64
DROPOUT = 0.2
BATCH = 512
MAX_EPOCHS = 10
PATIENCE = 2
LR = 1e-3
SCORE_BATCH = 1024
NET_FILE = "/net.pt"

_HASHER = HashingVectorizer(
    analyzer="char", ngram_range=NGRAMS, n_features=FEATURES, alternate_sign=False,
    binary=True, norm="l2", lowercase=False, dtype=np.float32,
)


def mean_ic(pred, y, groups):
    """Mean over groups (reviews) of the correlation between the rank of `pred` and `y`;
    -inf when it is undefined (a constant prediction)."""

    frame = pd.DataFrame({"pred": pred, "y": y, "group": groups})
    frame["pred"] = frame.groupby("group")["pred"].rank()
    ic = frame.groupby("group")[["pred", "y"]].apply(lambda rows: rows["pred"].corr(rows["y"])).mean()
    return float(ic) if np.isfinite(ic) else -np.inf


def fit(train, valid, day):
    """(saved parameters, validation scores) of the leg's model on one refit's segments.

    `train` and `valid` hold `inputs` (a list of text lists), `y` (label ranks) and
    `groups` (the review of each unit); `day` (YYYYMMDD as int) seeds the batch order.
    """

    return _fit_ridge(train, valid) if knobs.MODEL == "ridge" else _fit_net(train, valid, day)


def save(context, params, threshold):
    if knobs.MODEL == "ridge":
        np.savez(context.state_dir + RIDGE_FILE, coef=params["coef"],
                 intercept=np.array([params["intercept"]]), threshold=np.array([threshold]))
    else:
        torch.save({"state": params["state"], "vocab": torch.from_numpy(params["vocab"]),
                    "threshold": torch.tensor([threshold], dtype=torch.float64)}, context.state_dir + NET_FILE)


def predict(context, inputs):
    """(scores, threshold) of units under the model the last refit saved."""

    if knobs.MODEL == "ridge":
        saved = np.load(context.state_dir + RIDGE_FILE)
        return _ridge_scores(_hash(inputs), saved["coef"], float(saved["intercept"][0])), float(saved["threshold"][0])
    dev = device()
    saved = torch.load(context.state_dir + NET_FILE, map_location=dev)
    vocab = saved["vocab"].cpu().numpy()
    net = CharCNN(len(vocab)).to(dev)
    net.load_state_dict(saved["state"])
    return _net_scores(net, encode(inputs, vocab), dev), float(saved["threshold"][0])


# --- ridge -------------------------------------------------------------------

def _hash(inputs):
    return _HASHER.transform(["|".join(texts) for texts in inputs])


def _ridge_scores(features, coef, intercept):
    return np.asarray(features @ coef, dtype=np.float64).ravel() + intercept


def _fit_ridge(train, valid):
    x_train, x_valid = _hash(train["inputs"]), _hash(valid["inputs"])
    best = None
    for alpha in ALPHAS:
        model = Ridge(alpha=alpha, solver="sparse_cg", tol=1e-4).fit(x_train, train["y"])
        coef, intercept = model.coef_.astype(np.float32).ravel(), float(model.intercept_)
        scores = _ridge_scores(x_valid, coef, intercept)
        ic = mean_ic(scores, valid["y"], valid["groups"])
        if best is None or ic > best[0]:
            best = (ic, {"coef": coef, "intercept": intercept, "alpha": alpha}, scores)
    return best[1], best[2]


# --- the net -------------------------------------------------------------------

class CharCNN(nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.embed = nn.Embedding(vocab + 2, EMBED, padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv1d(EMBED, CHANNELS, width) for width in WIDTHS])
        self.head = nn.Sequential(
            nn.Dropout(DROPOUT), nn.Linear(CHANNELS * len(WIDTHS), HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, 1),
        )

    def forward(self, tokens):
        """tokens: (units, texts, LENGTH) ids, 0 for padding; an all-zero text is padding."""

        units, texts, length = tokens.shape
        flat = tokens.reshape(units * texts, length)
        x = self.embed(flat).transpose(1, 2)
        present = (flat > 0).to(x.dtype)
        pooled = []
        for width, conv in zip(WIDTHS, self.convs):
            # A window counts when it starts on a character; ReLU features are >= 0, so
            # zeroing the rest (and every padding text) leaves each maximum unchanged.
            pooled.append((torch.relu(conv(x)) * present[:, None, : length - width + 1]).amax(dim=2))
        features = torch.cat(pooled, dim=1).reshape(units, texts, -1).amax(dim=1)
        return self.head(features).squeeze(1)


def device():
    if not torch.cuda.is_available():
        raise RuntimeError("no CUDA device in the strategy container; the nn ranker does not run on the CPU")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return torch.device("cuda")


def vocabulary(inputs):
    """Sorted code points seen at least MIN_CHARS times in the texts."""

    text = "".join(item for texts in inputs for item in texts)
    counts = np.bincount(np.frombuffer(text.encode("utf-32-le"), dtype=np.uint32))
    return np.nonzero(counts >= MIN_CHARS)[0].astype(np.int64)


def encode(inputs, vocab):
    """(ids of every text as rows of LENGTH plus one empty row last, each unit's first row, its text count)."""

    count = np.array([len(texts) for texts in inputs], dtype=np.int64)
    start = np.concatenate([[0], np.cumsum(count)[:-1]]).astype(np.int64)
    flat = pd.Series([item for texts in inputs for item in texts], dtype=object)
    padded = flat.str.slice(0, LENGTH).str.pad(LENGTH, side="right", fillchar="\0")
    points = np.frombuffer("".join(padded).encode("utf-32-le"), dtype=np.uint32).astype(np.int64)
    points = points.reshape(len(flat), LENGTH)
    slot = np.searchsorted(vocab, points)
    known = (slot < len(vocab)) & (vocab[np.minimum(slot, len(vocab) - 1)] == points)
    ids = np.where(points == 0, 0, np.where(known, slot + 1, len(vocab) + 1))
    return np.vstack([ids, np.zeros((1, LENGTH), dtype=np.int64)]).astype(np.int32), start, count


def _tokens(ids, start, count, units, dev):
    """(len(units), widest unit's texts, LENGTH) ids on the device; missing texts are the empty row."""

    width = int(count[units].max())
    column = np.arange(width)[None, :]
    rows = np.where(column < count[units][:, None], start[units][:, None] + column, ids.shape[0] - 1)
    return ids[torch.from_numpy(rows).to(dev)].long()


def _net_scores(net, encoded, dev):
    ids, start, count = encoded
    if len(count) == 0:
        return np.zeros(0)
    ids = torch.from_numpy(ids).to(dev)
    net.eval()
    out = []
    with torch.no_grad():
        for begin in range(0, len(count), SCORE_BATCH):
            units = np.arange(begin, min(begin + SCORE_BATCH, len(count)))
            out.append(net(_tokens(ids, start, count, units, dev)).double().cpu().numpy())
    return np.concatenate(out)


def _fit_net(train, valid, day):
    dev = device()
    vocab = vocabulary(train["inputs"])
    ids, start, count = encode(train["inputs"], vocab)
    valid_encoded = encode(valid["inputs"], vocab)
    torch.manual_seed(knobs.SEED)
    order = np.random.default_rng([knobs.SEED, day])
    net = CharCNN(len(vocab)).to(dev)
    optimizer = torch.optim.Adam(net.parameters(), lr=LR)
    ids_dev = torch.from_numpy(ids).to(dev)
    y = torch.from_numpy(np.asarray(train["y"], dtype=np.float32)).to(dev)
    best, best_state, best_epochs, best_scores, stale = -np.inf, None, 0, None, 0
    for epoch in range(1, MAX_EPOCHS + 1):
        net.train()
        shuffled = order.permutation(len(count))
        for begin in range(0, len(shuffled), BATCH):
            units = shuffled[begin: begin + BATCH]
            pred = net(_tokens(ids_dev, start, count, units, dev))
            loss = torch.mean((pred - y[torch.from_numpy(units).to(dev)]) ** 2)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        scores = _net_scores(net, valid_encoded, dev)
        ic = mean_ic(scores, valid["y"], valid["groups"])
        if ic > best + 1e-5:
            best, best_epochs, best_scores, stale = ic, epoch, scores, 0
            best_state = {key: value.detach().clone() for key, value in net.state_dict().items()}
        else:
            stale += 1
            if stale >= PATIENCE:
                break
    if best_state is None:
        raise RuntimeError("the net never produced a finite validation IC")
    del net, optimizer, ids_dev, y
    torch.cuda.empty_cache()
    return {"state": best_state, "vocab": vocab, "epochs": best_epochs}, best_scores
