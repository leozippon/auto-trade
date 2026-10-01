"""Pool, label, design matrix, and the three ways a score is made: g1, c_shuf, c_lin.

Training set of one fit. The panel covers the trailing `knobs.TRAIN_YEARS`
of signal dates plus `census_features.DECISION_DAYS` of warm-up, so every
feature on a training row is the value a review on that date would have
computed. A signal date t is labelled once its label is realised by the newest
row (t + 1 + HOLD <= T-1). The last VALID_SESSIONS labelled sessions validate
LightGBM's early stopping; training dates end HOLD sessions before them, so no
training label overlaps a validation label. Both sets keep every STRIDE-th
date counted back from their newest one (a 20-session label overlaps itself
for four strides anyway).

Label. y = (O[t+1+H] / O[t+1] - 1) - beta x (B[t+1+H] / B[t+1] - 1), on
adjusted opens, B = `knobs.INDEX`'s opens, beta = the trailing 120-session OLS
beta of daily returns on the index (at least 60), shrunk 0.7 toward 1 and
clipped to [0.3, 2.0]. No beta, no label: a row whose beta or benchmark leg
cannot be computed is dropped, never filled. The target is the residual's
cross-sectional rank inside the pool on that date, minus 0.5.

Inputs. Every feature is replaced by its percentile rank among the pool's
names on its date (NaN stays NaN), the same transform on training rows and on
the newest row a review scores.

Candidates (`main.CANDIDATE`):
    g1      LightGBM regression on the ranked features
    c_shuf  g1's scores permuted among the pool on each decision day, with a
            generator seeded by that day: the book without the ranking
    c_lin   no model: the equal-weight mean of sign_f x (rank_f - 0.5) over
            every feature (missing = 0); sign_f is the sign of feature f's mean
            per-date rank correlation with the label over this fit's training
            and validation dates -- fitted in `fit`, nothing else learned

Every fitted thing lives under `context.state_dir` and is rebuilt each
quarter from scratch; nothing carries over between fits.
"""

import lightgbm as lgb
import numpy as np
import pandas as pd

from lib import census_features as cf, knobs

MIN_BARS = 60
STRIDE = 5
VALID_SESSIONS = 60
MIN_TRAIN_DATES = 60
MIN_NAMES = 200
BETA_SESSIONS, BETA_MINIMUM, BETA_SHRINK, BETA_CLIP = 120, 60, 0.7, (0.3, 2.0)
PARAMS = {"objective": "regression", "num_leaves": 31, "learning_rate": 0.05, "feature_fraction": 0.8,
          "bagging_fraction": 0.8, "bagging_freq": 1, "min_data_in_leaf": 200, "lambda_l2": 10.0,
          "num_threads": 8, "deterministic": True, "force_row_wise": True, "verbose": -1}
ROUNDS = 600
EARLY_STOP = 50
MODEL_FILE = "/model.txt"
SIGNS_FILE = "/signs.npy"
META_FILE = "/fit_meta.npy"


def fit_days():
    """Calendar days one fit reads: the training years plus a review's own window."""

    return int(365.25 * knobs.TRAIN_YEARS) + cf.DECISION_DAYS + 30


def used_features():
    """Indexes of the features the model reads, after `knobs.DROP_FAMILIES`."""

    known = set(cf.families())
    unknown = sorted(set(knobs.DROP_FAMILIES) - known)
    if unknown:
        raise ValueError(f"DROP_FAMILIES names unknown families {unknown}; known: {sorted(known)}")
    used = [k for k, family in enumerate(cf.families()) if family not in knobs.DROP_FAMILIES]
    if not used:
        raise ValueError("DROP_FAMILIES removes every feature")
    return np.array(used)


def pool(panel):
    """(sessions x names) bool: names this account may hold on each row."""

    codes = pd.Series(panel["codes"])
    names = pd.Series(panel["name"])
    static = ~codes.str.startswith(("688", "689")).to_numpy() & ~names.str.contains("ST|退").to_numpy()
    bars = np.cumsum(panel["has_bar"], axis=0)
    mask = panel["has_bar"] & (panel["d"]["close"] > 0) & (bars >= MIN_BARS) & static[None, :]
    if knobs.POOL == "csi1000":
        return mask & (cf.member_tier(panel) == 1)
    if knobs.POOL == "alla":
        return mask
    raise ValueError(f"unknown POOL {knobs.POOL!r}")


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


def ranked(features, mask, rows, used):
    """(len(rows), names, len(used)) float32: each feature's percentile rank in the pool on its date."""

    out = np.full((len(rows), mask.shape[1], len(used)), np.nan, dtype=np.float32)
    for j, k in enumerate(used):
        block = np.where(mask[rows], features[k][rows], np.nan)
        out[:, :, j] = pd.DataFrame(block).rank(axis=1, pct=True).to_numpy(dtype=np.float32)
    return out


def fit_rows(context, panel, target):
    """(training dates, validation dates) as panel row indexes."""

    first = (pd.Timestamp(context.inference_at.date()) - pd.DateOffset(years=knobs.TRAIN_YEARS)).strftime("%Y%m%d")
    counts = np.isfinite(target).sum(axis=1)
    labelled = [t for t in range(len(panel["dates"])) if panel["dates"][t] >= first and counts[t] >= MIN_NAMES]
    if not labelled:
        raise RuntimeError("no labelled signal date in the training window")
    valid_from = labelled[-1] - VALID_SESSIONS + 1
    valid = [t for t in labelled if t >= valid_from][::-1][::STRIDE][::-1]
    train = [t for t in labelled if t < valid_from - knobs.HOLD][::-1][::STRIDE][::-1]
    if len(train) < MIN_TRAIN_DATES:
        raise RuntimeError(f"only {len(train)} training dates (need {MIN_TRAIN_DATES}); "
                           f"labelled dates start {panel['dates'][labelled[0]]}")
    return np.array(train), np.array(valid)


def _stack(x, target, rows, mask):
    keep = mask[rows] & np.isfinite(target[rows])
    return x[keep], target[rows][keep]


def fit(context, candidate):
    """Rebuild the candidate's fitted state under `context.state_dir` from the trailing window."""

    panel = cf.read_panel(context, fit_days())
    mask = pool(panel)
    target = label(panel, mask)
    features = cf.compute(panel)
    used = used_features()
    train, valid = fit_rows(context, panel, target)
    x_train, y_train = _stack(ranked(features, mask, train, used), target, train, mask)
    x_valid, y_valid = _stack(ranked(features, mask, valid, used), target, valid, mask)
    del features
    rounds = 0
    if candidate == "c_lin":
        np.save(context.state_dir + SIGNS_FILE, _signs(np.vstack([x_train, x_valid]),
                                                       np.concatenate([y_train, y_valid]),
                                                       np.concatenate([_dates_of(train, mask, target),
                                                                       _dates_of(valid, mask, target)])))
    elif candidate in ("g1", "c_shuf"):
        train_set = lgb.Dataset(x_train, y_train, free_raw_data=True)
        valid_set = lgb.Dataset(x_valid, y_valid, reference=train_set, free_raw_data=True)
        booster = lgb.train({**PARAMS, "seed": knobs.SEED}, train_set, num_boost_round=ROUNDS,
                            valid_sets=[valid_set], valid_names=["valid"],
                            callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False), lgb.log_evaluation(0)])
        booster.save_model(context.state_dir + MODEL_FILE)
        rounds = booster.best_iteration
    else:
        raise ValueError(f"unknown candidate {candidate!r}")
    dates = panel["dates"]
    np.save(context.state_dir + META_FILE, np.array([
        len(train), len(valid), len(y_train), int(dates[train[0]]), int(dates[valid[-1]]), rounds,
        knobs.HOLD, len(used)], dtype=np.int64))


def _dates_of(rows, mask, target):
    keep = mask[rows] & np.isfinite(target[rows])
    return np.repeat(np.asarray(rows), keep.sum(axis=1))


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


def score(context, panel, candidate):
    """(names,) scores on the panel's newest row; NaN off the pool."""

    mask = pool(panel)
    last = len(panel["dates"]) - 1
    used = used_features()
    idx = np.flatnonzero(mask[last])
    out = np.full(len(panel["codes"]), np.nan)
    if not len(idx):
        return out
    x = ranked(cf.compute(panel), mask, np.array([last]), used)[0][idx]
    meta = _load(np.load, context.state_dir + META_FILE)
    if int(meta[7]) != len(used) or int(meta[6]) != knobs.HOLD:
        raise RuntimeError("the fitted state was built for another feature set or horizon; refit")
    if candidate == "c_lin":
        signs = _load(np.load, context.state_dir + SIGNS_FILE)
        values = np.nansum(signs[None, :] * (x - 0.5), axis=1) / x.shape[1]
    elif candidate in ("g1", "c_shuf"):
        values = _load(lambda path: lgb.Booster(model_file=path), context.state_dir + MODEL_FILE).predict(x)
        if candidate == "c_shuf":
            day = int(pd.Timestamp(context.inference_at).strftime("%Y%m%d"))
            values = values[np.random.default_rng(day).permutation(len(values))]
    else:
        raise ValueError(f"unknown candidate {candidate!r}")
    out[idx] = values
    return out


def _load(reader, path):
    try:
        return reader(path)
    except (FileNotFoundError, OSError, lgb.basic.LightGBMError) as error:
        raise RuntimeError(f"no fitted state at {path}: fit did not run for this candidate") from error
