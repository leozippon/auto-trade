"""Reported control `c_lgbm`: the graduate's Alpha158 + LightGBM ranker on the same panels and schedule.

Faithful to the graduated artifact where it matters: its Alpha158 code
(`lib/alpha158.py`) and per-date robust z-score, its 10-day open-to-open rank
label, objective and fixed parameters, 600 rounds with early stopping 50 on the
last VALID_DAYS training dates after an EMBARGO_DAYS gap, training on every name
with a full sequence. Same simplifications as the direction study that measured
it: the trailing TRAIN_YEARS window instead of the full history, one
(num_leaves, learning_rate) point instead of its four-point grid, and the 158
Alpha columns without its fundamentals, money-flow and listing-age columns.

Alpha158 needs NaN on days without a bar, so the panel's forward-filled prices
and zero volume are masked back to NaN before the operators run. Features are
computed in BLOCK-date blocks with CONTEXT dates of warm-up, so memory stays
bounded on a three-year panel.
"""

import numpy as np
import lightgbm as lgb

from lib import alpha158, model, panel as P

PARAMS = {"objective": "regression", "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1,
          "min_data_in_leaf": 200, "lambda_l2": 10, "num_leaves": 31, "learning_rate": 0.05,
          "num_threads": 8, "seed": 7, "verbose": -1}
ROUNDS = 600
EARLY_STOP = 50
BLOCK = 120
CONTEXT = 60


def model_path(context):
    return context.state_dir + "/lgbm_alpha158.txt"


def _wide(data, lo, hi):
    """(S, hi - lo) float64 inputs for alpha158.compute_158, NaN on days without a bar."""

    bars = data["has_bar"][lo:hi].T
    seq = data["seq"][lo:hi]
    names = ("O", "H", "L", "C", "VWAP", "V")
    return {name: np.where(bars, seq[:, :, k].T.astype(np.float64), np.nan) for k, name in enumerate(names)}


def features(data, rows):
    """{date index: (S, 158) float32 z-scored features} for the requested date indexes."""

    rows = sorted(rows)
    out = {}
    T = len(data["dates"])
    for start in range(rows[0], rows[-1] + 1, BLOCK):
        end = min(T, start + BLOCK)
        wanted = [t for t in rows if start <= t < end]
        if not wanted:
            continue
        lo = max(0, start - CONTEXT)
        raw = alpha158.compute_158(_wide(data, lo, end))
        block = np.stack([raw[name][:, [t - lo for t in wanted]] for name in alpha158.ALPHA158], axis=2)
        block = np.transpose(block, (1, 0, 2)).astype(np.float32)
        block[~data["has_bar"][wanted]] = np.nan
        z = alpha158.robust_zscore(block)
        for i, t in enumerate(wanted):
            out[t] = z[i]
    return out


def fit(context):
    data = P.build(context, model.FIT_CALENDAR_DAYS, labels=True)
    train, valid = model.fit_dates(context, data)
    table = features(data, train + valid)

    def rows(dates):
        X, y = [], []
        for t in dates:
            idx = model.scorable(data, t)
            idx = idx[np.isfinite(data["yrank"][t, idx])]
            X.append(table[t][idx])
            y.append(data["yrank"][t, idx])
        return np.vstack(X), np.concatenate(y)

    Xtr, ytr = rows(train)
    Xva, yva = rows(valid)
    del table
    dtrain = lgb.Dataset(Xtr, ytr, free_raw_data=True)
    dvalid = lgb.Dataset(Xva, yva, reference=dtrain, free_raw_data=True)
    booster = lgb.train(PARAMS, dtrain, num_boost_round=ROUNDS, valid_sets=[dvalid], valid_names=["valid"],
                        callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False), lgb.log_evaluation(0)])
    booster.save_model(model_path(context), num_iteration=booster.best_iteration or ROUNDS)


def score(context, data):
    """Percentile rank in [0, 1] of every scorable name on the newest row (NaN elsewhere)."""

    last = len(data["dates"]) - 1
    idx = model.scorable(data, last)
    X = features(data, [last])[last][idx]
    booster = lgb.Booster(model_file=model_path(context))
    prediction = booster.predict(X)
    out = np.full(len(data["codes"]), np.nan)
    order = np.argsort(np.argsort(prediction, kind="stable"), kind="stable")
    out[idx] = order / max(1, len(idx) - 1)
    return out
