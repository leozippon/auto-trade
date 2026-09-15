"""LightGBM cross-sectional ranker: the graduate's grid, trained in fit, scored on a review day.

Mechanism (the graduate's scorer): pre-registered 4-point grid num_leaves {31, 63} x
learning_rate {0.05, 0.1}; objective=regression on the per-bar label rank - 0.5,
feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, min_data_in_leaf=200,
lambda_l2=10, num_threads=8; validation = the last VALID_DAYS training bars after an
EMBARGO_DAYS gap (lib/common.py); 600 rounds with early stopping 50; among the points
within 5% of the minimum validation L2 the lexicographically smallest
(num_leaves, learning_rate) wins. Each refit trains one new booster and replaces the
previous one.

State files under context.state_dir, rewritten by every fit:
  model.txt           the selected booster at its best iteration (save_model)
  model_meta.npy      float64[3] = [leaf_idx, lr_idx, best_iter]
  feature_names.npy   the column names the booster was trained on

There is no fallback scorer: a fit with too few samples raises, and a review day with
no usable state raises, so a replay fails instead of trading a different strategy.
"""

import numpy as np
import lightgbm as lgb
from scipy.stats import rankdata

from lib import common, data

LEAVES = (31, 63)
LRS = (0.05, 0.1)
NUM_ROUNDS = 600
EARLY_STOP = 50
FIXED_PARAMS = {
    "objective": "regression",
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "min_data_in_leaf": 200,
    "lambda_l2": 10,
    "verbose": -1,
    # Pin the thread count: lgbm's default (auto) can oversubscribe the container quota.
    "num_threads": 8,
}


def _pct_rank(pred):
    """Cross-sectional percentile rank in [0, 1]; only finite values participate, ties
    get the average rank, mapped to (r - 0.5) / n_finite; non-finite input stays NaN."""
    pred = np.asarray(pred, dtype=np.float64)
    out = np.full(pred.shape, np.nan, dtype=np.float64)
    m = np.isfinite(pred)
    n = int(m.sum())
    if n == 0:
        return out
    out[m] = (rankdata(pred[m], method="average") - 0.5) / n
    return out


def _train_grid(Xtr, ytr, Xva, yva, names, seed=None, num_boost_round=NUM_ROUNDS,
                early_stop=EARLY_STOP):
    """4-point grid + embargoed early stopping + 5% plateau / lexicographic selection.
    Returns (best, l2s): best is {"li","ri","l2","bi","booster"} and non-best boosters
    are released. Raises on failure."""
    results = []
    for li, nl in enumerate(LEAVES):
        for ri, lr in enumerate(LRS):
            params = dict(FIXED_PARAMS, num_leaves=nl, learning_rate=lr)
            if seed is not None:
                params["seed"] = int(seed)
            dtr = lgb.Dataset(Xtr, ytr, feature_name=names, free_raw_data=True)
            dva = lgb.Dataset(Xva, yva, reference=dtr, free_raw_data=True)
            booster = lgb.train(
                params, dtr, num_boost_round=num_boost_round,
                valid_sets=[dva], valid_names=["val"],
                callbacks=[lgb.early_stopping(early_stop, verbose=False), lgb.log_evaluation(0)],
            )
            bi = booster.best_iteration or num_boost_round
            pred = booster.predict(Xva, num_iteration=bi)
            l2 = float(np.mean((pred - yva) ** 2))
            results.append({"li": li, "ri": ri, "l2": l2, "bi": int(bi), "booster": booster})
            del dtr, dva
    l2s = np.array([r["l2"] for r in results])
    cands = [i for i in range(len(results)) if l2s[i] <= 1.05 * l2s.min()]
    sel = min(cands, key=lambda i: (results[i]["li"], results[i]["ri"]))
    best = results[sel]
    for r in results:
        if r is not best:
            r["booster"] = None
    return best, [float(x) for x in l2s]


def fit(context):
    """Train one booster on the trailing window and replace the state files."""
    samples = data.build_fit_samples(context)
    X, y, date_idx = samples["X"], samples["y"], samples["date_idx"]
    tr, va, n_train, n_train_dates = common.split_fit_dates(date_idx)
    if n_train < common.MIN_SAMPLES or n_train_dates < common.MIN_SAMPLES or not va.any():
        raise RuntimeError(
            f"fit has {n_train} training rows on {n_train_dates} dates and "
            f"{int(va.sum())} validation rows; need >= {common.MIN_SAMPLES} of each"
        )
    Xtr, ytr = X[tr], y[tr]
    Xva, yva = X[va], y[va]
    del X, samples
    names = [str(n) for n in data.FEATURE_NAMES]
    best, _l2s = _train_grid(Xtr, ytr, Xva, yva, names)
    best["booster"].save_model(context.state_dir + "/model.txt", num_iteration=best["bi"])
    np.save(context.state_dir + "/model_meta.npy",
            np.asarray((best["li"], best["ri"], best["bi"]), dtype=np.float64))
    np.save(context.state_dir + "/feature_names.npy", np.array(names))


def score(context, X_z):
    """Percentile ranks of the booster's predictions for one cross-section (S, F).

    Loaded per call, so a refit that replaced the state files between two reviews is
    picked up without a worker restart. Returns (scores (S,) float64, order metadata)."""
    names = [str(x) for x in np.load(context.state_dir + "/feature_names.npy")]
    if names != [str(n) for n in data.FEATURE_NAMES]:
        raise RuntimeError("the fitted booster's feature names differ from data.FEATURE_NAMES")
    meta = np.load(context.state_dir + "/model_meta.npy")
    booster = lgb.Booster(model_file=context.state_dir + "/model.txt")
    pred = booster.predict(np.ascontiguousarray(X_z, dtype=np.float32))
    extra = {"num_leaves": int(LEAVES[int(meta[0])]),
             "learning_rate": float(LRS[int(meta[1])]),
             "best_iter": int(meta[2])}
    return _pct_rank(pred), extra
