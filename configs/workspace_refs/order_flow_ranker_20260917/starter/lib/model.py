"""Embargoed walk-forward training of the constrained LightGBM leg.

`fit` runs once per quarter (`REFIT_PERIOD`) inside its own container and sees
only rows visible at that decision instant. Training rows keep a fully realized
label (entry at the next visible close, exit LABEL_HORIZON trading days later,
both inside the visible daily history). The last VALIDATION_BARS labelled
trading days are the validation slice and another LABEL_HORIZON trading days of
embargo separate it from training, so no training label window overlaps it.

`fit` never raises: a failure writes `ok=0`, and the decision path degrades to
this arm's equal-weight composite on the same face and stamps `degraded` on
every order, because a silent degrade is indistinguishable from a working model
in the aggregates.

That path is an anomaly, not a known limitation. Both faces are daily events
tables (`intraday_flow` and `moneyflow`) with a row for every visible trading
day, so every quarterly refit of every leg trains on the full input window and
the legs stay symmetric across all four sub-windows. A fold whose orders carry
`_degraded` should be investigated, not explained away -- see
`exploration-plan.md`.
"""

import lightgbm as lgb
import numpy as np
import pandas as pd

from lib import flow

LABEL_HORIZON = 20          # trading days; equal to the rebalance cadence by contract
FEATURE_STRIDE = 5          # training-panel grid, in trading days
VALIDATION_BARS = 60
EMBARGO_BARS = LABEL_HORIZON
FIT_LOOKBACK_TRADE_DAYS = 260
FIT_LOOKBACK_DAYS = 900     # calendar days of daily history read for training
MIN_TRAIN_ROWS = 8000
MIN_DATES = 20
GRID = ((15, 0.05), (15, 0.1), (31, 0.05), (31, 0.1))
NUM_THREADS = 5             # 3 concurrent batch_validate containers x 5 <= 16 CPU quota
MAX_ROUNDS = 400
EARLY_STOP = 40


def fit(context, candidate):
    if flow.scorer_of(candidate) != "lgb":
        return None
    names = flow.feature_names(candidate)
    try:
        panel = build_panel(context, candidate, names)
        booster, chosen, score = _train(panel, names)
        np.save(context.state_dir + "/model.npy", np.array([booster.model_to_string()]))
        _save_meta(context, names, chosen, score, ok=1.0)
    except Exception:  # never raise out of fit; the decision path degrades explicitly
        _save_meta(context, names, (0, 0.0, 0), float("nan"), ok=0.0)
    return None


def _save_meta(context, names, chosen, score, ok):
    np.savez(
        context.state_dir + "/meta.npz",
        names=np.array(names),
        leaves=np.array([float(chosen[0])]),
        learning_rate=np.array([float(chosen[1])]),
        rounds=np.array([float(chosen[2])]),
        valid_ic=np.array([float(score)]),
        ok=np.array([float(ok)]),
    )


def label_frame(daily, horizon):
    """Realized close-to-close forward return under the frozen adjustment anchor.

    Entry is the next visible close, which is where the buy leg executes at
    15:00 of the decision day. The exit leg is timed at 09:30 so its proceeds
    are credited before the same day's buys are sized, leaving the realized exit
    half a session ahead of the label -- a declared approximation, identical
    across all four legs, so it cannot move the comparison.
    """
    grouped = daily.groupby("ts_code", sort=False)["q_close"]
    label = grouped.shift(-(1 + horizon)) / grouped.shift(-1) - 1.0
    return daily[["ts_code", "trade_date"]].assign(label=label.to_numpy())


def panel_dates(daily, face_dates, horizon, stride):
    """Feature dates that carry both a realized label and a value on this face."""
    dates = sorted(daily["trade_date"].unique())
    usable = [d for d in dates[: max(0, len(dates) - (1 + horizon))] if d in face_dates]
    return usable[::-1][::stride][::-1]


def build_panel(context, candidate, names):
    daily = flow.read_daily(context, FIT_LOOKBACK_DAYS)
    universe = flow.read_universe(context)
    dates = sorted(daily["trade_date"].unique())[-FIT_LOOKBACK_TRADE_DAYS:]
    block, context_frame = flow.build_face(context, candidate, daily, dates)
    if block.empty:
        raise ValueError("empty feature face")
    face_dates = set(block.index.get_level_values("trade_date"))
    labels = label_frame(daily, LABEL_HORIZON).set_index(["ts_code", "trade_date"])["label"]
    rows = []
    for feature_date in panel_dates(daily, face_dates, LABEL_HORIZON, FEATURE_STRIDE):
        pool = flow.eligible(context, daily, universe, context_frame, feature_date)
        if len(pool) < 200:
            continue
        section = flow.cross_section(block, context_frame, names, feature_date, pool)
        if section.empty:
            continue
        target = labels.reindex(
            pd.MultiIndex.from_product([section.index, [feature_date]],
                                       names=["ts_code", "trade_date"]))
        section = section.assign(label=target.to_numpy()).dropna(subset=["label"])
        if len(section) < 50:
            continue
        excess = section["label"] - section["label"].mean()
        section = section.assign(target=excess.rank(pct=True) - 0.5, trade_date=feature_date)
        rows.append(section[names + ["target", "trade_date"]])
    if not rows:
        raise ValueError("no labelled feature date on this face")
    return pd.concat(rows, ignore_index=True)


def split(panel):
    """Train / embargo / validation on the feature-date grid."""
    dates = sorted(panel["trade_date"].unique())
    if len(dates) < MIN_DATES:
        raise ValueError("not enough feature dates for an embargoed split")
    valid_n = max(4, VALIDATION_BARS // FEATURE_STRIDE)
    gap_n = max(1, EMBARGO_BARS // FEATURE_STRIDE)
    valid_dates = dates[-valid_n:]
    train_dates = dates[: max(0, len(dates) - valid_n - gap_n)]
    if not train_dates:
        raise ValueError("embargo consumed the whole training window")
    train = panel[panel["trade_date"].isin(train_dates)]
    valid = panel[panel["trade_date"].isin(valid_dates)]
    if len(train) < MIN_TRAIN_ROWS:
        raise ValueError("training panel below the declared minimum")
    return train, valid


def _params(leaves, learning_rate):
    return {
        "objective": "regression",
        "num_leaves": int(leaves),
        "learning_rate": float(learning_rate),
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "min_data_in_leaf": 200,
        "lambda_l2": 10.0,
        "num_threads": NUM_THREADS,
        "verbose": -1,
        "seed": 20260917,
    }


def rank_ic(frame, prediction):
    """Mean per-date Spearman between the score and the rank target."""
    scored = frame[["trade_date", "target"]].assign(prediction=prediction)
    values = []
    for _, group in scored.groupby("trade_date", sort=False):
        if len(group) < 30:
            continue
        left = group["prediction"].rank().to_numpy()
        right = group["target"].rank().to_numpy()
        if left.std() <= 0 or right.std() <= 0:
            continue
        values.append(float(np.corrcoef(left, right)[0, 1]))
    return float(np.mean(values)) if values else float("nan")


def _train(panel, names):
    train, valid = split(panel)
    train_set = lgb.Dataset(train[names], label=train["target"])
    valid_set = lgb.Dataset(valid[names], label=valid["target"], reference=train_set)
    best = None
    for leaves, learning_rate in GRID:
        booster = lgb.train(
            _params(leaves, learning_rate), train_set, num_boost_round=MAX_ROUNDS,
            valid_sets=[valid_set],
            callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)])
        rounds = int(booster.best_iteration or MAX_ROUNDS)
        score = rank_ic(valid, booster.predict(valid[names], num_iteration=rounds))
        if not np.isfinite(score):
            continue
        if best is None or score > best[0]:
            best = (score, leaves, learning_rate, rounds)
    if best is None:
        raise ValueError("no grid point produced a finite validation rank IC")
    score, leaves, learning_rate, rounds = best
    # Refit the selected point on train + embargo + validation: every label there is
    # already realized, so this adds recent data without seeing the future.
    full = lgb.Dataset(panel[names], label=panel["target"])
    booster = lgb.train(_params(leaves, learning_rate), full, num_boost_round=rounds)
    return booster, (leaves, learning_rate, rounds), score


def load(context, candidate):
    """Booster and the persisted feature order, or (None, names) when fit degraded."""
    names = flow.feature_names(candidate)
    meta = np.load(context.state_dir + "/meta.npz", allow_pickle=False)
    stored = [str(item) for item in meta["names"]]
    if stored != names:
        raise RuntimeError("persisted feature order does not match the declared block")
    if float(meta["ok"][0]) != 1.0:
        return None, names
    payload = np.load(context.state_dir + "/model.npy", allow_pickle=False)
    text = str(payload[0])
    if len(text) < 100:
        raise RuntimeError("persisted booster is empty or truncated")
    booster = lgb.Booster(model_str=text)
    if list(booster.feature_name()) != names:
        raise RuntimeError("persisted booster feature order does not match the block")
    return booster, names
