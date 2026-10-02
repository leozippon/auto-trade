"""Pool, label, sign fit and the composite score of the three legs: l1, c_shuf, c_fix.

The composite. On each decision day every one of the 44 census features is
replaced by its percentile rank among the pool's names (NaN stays NaN); the
score is sum_f w_f x sign_f x (rank_f - 0.5), a missing rank counting 0. With
`knobs.WEIGHTING = "feature"` every w_f is 1/44 (computed as the plain sum
divided by 44); with `"family"` each of the 19 census families weighs 1/19,
split evenly over its features. Nothing else is learned: no weight, no
feature choice.

The sign fit, in `fit`. sign_f is the sign of feature f's mean per-date rank
correlation with the label over the fit's dates. The dates are those the
20261001 packs gave their LightGBM fit, kept so that `l1` is that pack's
`c_lin` unchanged: the signal dates of the trailing `knobs.SIGN_YEARS` whose
label is realised by the newest row, every STRIDE-th counted back from the
newest, split into the last VALID_SESSIONS sessions and the dates ending
`knobs.HOLD` sessions before them (the dates in that gap are not used).

Label. y = (O[t+1+H] / O[t+1] - 1) - beta x (B[t+1+H] / B[t+1] - 1), on
adjusted opens, B = `knobs.INDEX`'s opens, beta = the trailing 120-session OLS
beta of daily returns on the index (at least 60), shrunk 0.7 toward 1 and
clipped to [0.3, 2.0]; a row without a beta or a benchmark leg is dropped.
The target is the residual's cross-sectional rank in the pool, minus 0.5.

Legs (`main.CANDIDATE`):
    l1      the composite with its signs refitted at every quarterly fit
    c_shuf  l1's scores permuted among the pool on each decision day, with a
            generator seeded by that day: the book without the ranking
    c_fix   the composite with the signs of the FIRST fit of the replay kept
            for the whole replay (later fits change nothing): what is left
            when the trailing sign fit is switched off

Every fitted thing lives under `context.state_dir`.
"""

import numpy as np
import pandas as pd

from lib import census_features as cf, knobs

MIN_BARS = 60
STRIDE = 5
VALID_SESSIONS = 60
MIN_TRAIN_DATES = 60
MIN_NAMES = 200
BETA_SESSIONS, BETA_MINIMUM, BETA_SHRINK, BETA_CLIP = 120, 60, 0.7, (0.3, 2.0)
SIGNS_FILE = "/signs.npy"
META_FILE = "/fit_meta.npy"
LEGS = ("l1", "c_shuf", "c_fix")


def fit_days():
    """Calendar days one fit reads: the sign-fit years plus a review's own window."""

    return int(365.25 * knobs.SIGN_YEARS) + cf.DECISION_DAYS + 30


def pool(panel):
    """(sessions x names) bool: CSI 1000 members of the section in force this account may hold on each row."""

    codes = pd.Series(panel["codes"])
    names = pd.Series(panel["name"])
    static = ~codes.str.startswith(("688", "689")).to_numpy() & ~names.str.contains("ST|退").to_numpy()
    bars = np.cumsum(panel["has_bar"], axis=0)
    mask = panel["has_bar"] & (panel["d"]["close"] > 0) & (bars >= MIN_BARS) & static[None, :]
    return mask & (cf.member_tier(panel) == 1)


def label(panel, mask):
    """(sessions x names) float32 benchmark-residual forward rank in [-0.5, 0.5]; NaN off the pool."""

    bench = panel["index"].get(knobs.INDEX)
    if bench is None or not np.isfinite(bench["open"]).any():
        raise RuntimeError(f"no visible {knobs.INDEX} index_daily rows: filter index_daily on ts_code")
    hold = knobs.HOLD
    rows = len(panel["dates"])
    market = cf.index_returns(panel, knobs.INDEX)
    raw = cf.rolling_slope(cf.daily_returns(panel), np.repeat(market[:, None], len(panel["codes"]), axis=1),
                           BETA_SESSIONS, BETA_MINIMUM)
    beta = np.clip(BETA_SHRINK * raw + (1.0 - BETA_SHRINK), *BETA_CLIP)
    opens = panel["d"]["open"] * panel["d"]["adj_factor"]
    stock = np.full_like(opens, np.nan)
    leg = np.full(rows, np.nan)
    if rows > hold + 1:
        stock[: rows - hold - 1] = opens[hold + 1:] / opens[1: rows - hold] - 1.0
        leg[: rows - hold - 1] = bench["open"][hold + 1:] / bench["open"][1: rows - hold] - 1.0
    residual = np.where(mask, stock - beta * leg[:, None], np.nan)
    ranks = pd.DataFrame(residual).rank(axis=1, pct=True).to_numpy()
    return (ranks - 0.5).astype(np.float32)


def ranked(features, mask, rows):
    """(len(rows), names, features) float32: each feature's percentile rank in the pool on its date."""

    out = np.full((len(rows), mask.shape[1], features.shape[0]), np.nan, dtype=np.float32)
    for k in range(features.shape[0]):
        block = np.where(mask[rows], features[k][rows], np.nan)
        out[:, :, k] = pd.DataFrame(block).rank(axis=1, pct=True).to_numpy(dtype=np.float32)
    return out


def fit_rows(context, panel, target):
    """(training dates, validation dates) as panel row indexes; the sign fit uses both."""

    first = (pd.Timestamp(context.inference_at.date()) - pd.DateOffset(years=knobs.SIGN_YEARS)).strftime("%Y%m%d")
    counts = np.isfinite(target).sum(axis=1)
    labelled = [t for t in range(len(panel["dates"])) if panel["dates"][t] >= first and counts[t] >= MIN_NAMES]
    if not labelled:
        raise RuntimeError("no labelled signal date in the sign-fit window")
    valid_from = labelled[-1] - VALID_SESSIONS + 1
    valid = [t for t in labelled if t >= valid_from][::-1][::STRIDE][::-1]
    train = [t for t in labelled if t < valid_from - knobs.HOLD][::-1][::STRIDE][::-1]
    if len(train) < MIN_TRAIN_DATES:
        raise RuntimeError(f"only {len(train)} sign-fit dates (need {MIN_TRAIN_DATES}); "
                           f"labelled dates start {panel['dates'][labelled[0]]}")
    return np.array(train), np.array(valid)


def _stack(x, target, rows, mask):
    keep = mask[rows] & np.isfinite(target[rows])
    return x[keep], target[rows][keep]


def _dates_of(rows, mask, target):
    keep = mask[rows] & np.isfinite(target[rows])
    return np.repeat(np.asarray(rows), keep.sum(axis=1))


def fit(context, candidate):
    """Write the signs under `context.state_dir`; `c_fix` keeps the first fit's signs for the whole replay."""

    if candidate not in LEGS:
        raise ValueError(f"unknown candidate {candidate!r}")
    if candidate == "c_fix" and _exists(context.state_dir + SIGNS_FILE):
        return
    panel = cf.read_panel(context, fit_days())
    mask = pool(panel)
    target = label(panel, mask)
    features = cf.compute(panel)
    train, valid = fit_rows(context, panel, target)
    x_train, y_train = _stack(ranked(features, mask, train), target, train, mask)
    x_valid, y_valid = _stack(ranked(features, mask, valid), target, valid, mask)
    del features
    np.save(context.state_dir + SIGNS_FILE, _signs(np.vstack([x_train, x_valid]),
                                                   np.concatenate([y_train, y_valid]),
                                                   np.concatenate([_dates_of(train, mask, target),
                                                                   _dates_of(valid, mask, target)])))
    dates = panel["dates"]
    np.save(context.state_dir + META_FILE, np.array([
        len(train), len(valid), len(y_train), int(dates[train[0]]), int(dates[valid[-1]]), 0,
        knobs.HOLD, len(cf.names())], dtype=np.int64))


def _exists(path):
    try:
        np.load(path)
    except FileNotFoundError:
        return False
    return True


def _signs(x, y, dates):
    """Sign of each feature's mean per-date rank correlation with the label; 0 for a feature never seen."""

    sums = np.zeros(x.shape[1])
    seen = np.zeros(x.shape[1])
    for day in np.unique(dates):
        rows = dates == day
        for j in range(x.shape[1]):
            xs, ys = x[rows, j], y[rows]
            both = np.isfinite(xs) & np.isfinite(ys)
            if both.sum() < MIN_NAMES // 4:
                continue
            a, b = xs[both] - xs[both].mean(), ys[both] - ys[both].mean()
            scale = np.sqrt((a * a).sum() * (b * b).sum())
            if scale > 0:
                sums[j] += (a * b).sum() / scale
                seen[j] += 1
    return np.sign(np.where(seen > 0, sums / np.maximum(seen, 1), 0.0))


def family_weights():
    """(features,) weights: each census family 1 / (number of families), split evenly over its features."""

    families = np.array(cf.families())
    labels, counts = np.unique(families, return_counts=True)
    size = dict(zip(labels, counts))
    return np.array([1.0 / (len(labels) * size[family]) for family in families])


def score(context, panel, candidate):
    """(names,) scores on the panel's newest row; NaN off the pool."""

    if candidate not in LEGS:
        raise ValueError(f"unknown candidate {candidate!r}")
    mask = pool(panel)
    last = len(panel["dates"]) - 1
    idx = np.flatnonzero(mask[last])
    out = np.full(len(panel["codes"]), np.nan)
    if not len(idx):
        return out
    x = ranked(cf.compute(panel), mask, np.array([last]))[0][idx]
    meta = _load(context.state_dir + META_FILE)
    if int(meta[7]) != x.shape[1] or int(meta[6]) != knobs.HOLD:
        raise RuntimeError("the fitted signs were built for another feature set or horizon; refit")
    signs = _load(context.state_dir + SIGNS_FILE)
    terms = signs[None, :] * (x - 0.5)
    if knobs.WEIGHTING == "feature":
        values = np.nansum(terms, axis=1) / x.shape[1]
    elif knobs.WEIGHTING == "family":
        values = np.nansum(terms * family_weights()[None, :], axis=1)
    else:
        raise ValueError(f"unknown WEIGHTING {knobs.WEIGHTING!r}")
    if candidate == "c_shuf":
        day = int(pd.Timestamp(context.inference_at).strftime("%Y%m%d"))
        values = values[np.random.default_rng(day).permutation(len(values))]
    out[idx] = values
    return out


def _load(path):
    try:
        return np.load(path)
    except (FileNotFoundError, OSError) as error:
        raise RuntimeError(f"no fitted state at {path}: fit did not run for this candidate") from error
