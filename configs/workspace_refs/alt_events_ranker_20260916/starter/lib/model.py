"""Embargoed walk-forward training of the cross-sectional ranker.

`fit` runs once per quarter (`REFIT_PERIOD`) inside its own container and sees
only rows visible at that decision instant. Training rows keep a fully realized
label (`T + 1 + h <= D - 1`); the last 60 labelled trading days are the
validation slice and another `h` trading days of embargo separate it from the
training slice, so no training row's label window overlaps validation.

`fit` never raises: a failure writes `ok=0` and the decision path degrades to
the declared equal-weight fallback and says so in the order metadata (the
`fit-state-pitfalls` lesson — a silent degrade is indistinguishable from a
working model in the aggregate metrics).
"""

import lightgbm as lgb
import numpy as np
import pandas as pd

from lib import features

LABEL_HORIZON = 20          # trading days; equal to the rebalance cadence by contract
FEATURE_STRIDE = 5          # training-panel grid, in trading days
VALIDATION_BARS = 60        # trading days held out at the end of the labelled history
EMBARGO_BARS = LABEL_HORIZON
FIT_LOOKBACK_DAYS = 1000    # calendar days of daily history read for training
MIN_TRAIN_ROWS = 20000
MIN_DATES = 40
PRICE_CAP = 30.0
ADV_FLOOR = 3.0e7
GRID = ((31, 0.05), (31, 0.1), (63, 0.05), (63, 0.1))
NUM_THREADS = 5             # 3 concurrent batch_validate containers x 5 <= 16 CPU quota
MAX_ROUNDS = 600
EARLY_STOP = 50


def fit(context, daily_only):
    names = features.feature_names(daily_only)
    try:
        panel = build_panel(context, daily_only, names)
        booster, chosen, score = _train(panel, names)
        text = booster.model_to_string()
        np.save(context.state_dir + "/model.npy", np.array([text]))
        _save_meta(context, names, chosen, score, ok=1.0)
    except Exception:  # never raise out of fit; the decision path degrades explicitly
        _save_meta(context, names, (0, 0.0, 0), float("nan"), ok=0.0)


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

    The close convention matches the entry leg: buys execute at 15:00 of the
    decision day, so the entry price the label assumes is the price the Broker
    actually pays. Exits are timed at 09:30 (so the proceeds are credited before
    the same day's buys are sized), which leaves the exit half a session ahead of
    the label — a declared approximation, identical across the main candidate and
    both controls, so it cannot move the comparison.
    """
    grouped = daily.groupby("ts_code", sort=False)["q_close"]
    entry = grouped.shift(-1)
    exit_price = grouped.shift(-(1 + horizon))
    label = exit_price / entry - 1.0
    return daily[["ts_code", "trade_date"]].assign(label=label.to_numpy())


def panel_dates(daily, horizon, stride):
    """Feature dates whose label is fully realized inside the visible history."""
    dates = sorted(daily["trade_date"].unique())
    usable = dates[: max(0, len(dates) - (1 + horizon))]
    return usable[::-1][::stride][::-1]


def build_panel(context, daily_only, names):
    daily = features.read_daily(context, FIT_LOOKBACK_DAYS)
    universe = features.read_universe(context)
    tables = features.read_all_events(context) if not daily_only else features.empty_tables()
    price, block_ev, unlocks, adv, amount_lookup = features.prepare(context, daily, tables)
    labels = label_frame(daily, LABEL_HORIZON).set_index(["ts_code", "trade_date"])["label"]
    rows = []
    for feature_date in panel_dates(daily, LABEL_HORIZON, FEATURE_STRIDE):
        pool = features.eligible(
            context, daily, universe, feature_date, PRICE_CAP, ADV_FLOOR, adv)
        if len(pool) < 200:
            continue
        section = features.cross_section(
            context, feature_date, daily, price, tables, block_ev, unlocks,
            features.adv_sum_at(adv, feature_date), amount_lookup, daily_only)
        section = section.reindex(pool)
        section = features.standardize(section, names)
        target = labels.reindex(
            pd.MultiIndex.from_product([pool, [feature_date]], names=["ts_code", "trade_date"]))
        section = section.assign(label=target.to_numpy()).dropna(subset=["label"])
        if len(section) < 50:
            continue
        excess = section["label"] - section["label"].mean()
        section = section.assign(
            target=excess.rank(pct=True) - 0.5, trade_date=feature_date)
        rows.append(section[names + ["target", "trade_date"]])
    if not rows:
        raise ValueError("empty training panel")
    return pd.concat(rows, ignore_index=False)


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
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "min_data_in_leaf": 200,
        "lambda_l2": 10.0,
        "num_threads": NUM_THREADS,
        "verbose": -1,
        "seed": 20260916,
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


def load(context):
    """Booster and the persisted feature order, or (None, names) when fit failed."""
    meta = np.load(context.state_dir + "/meta.npz", allow_pickle=False)
    names = [str(item) for item in meta["names"]]
    if float(meta["ok"][0]) != 1.0:
        return None, names
    payload = np.load(context.state_dir + "/model.npy", allow_pickle=False)
    text = str(payload[0])
    if len(text) < 100:
        raise RuntimeError("persisted booster is empty or truncated")
    booster = lgb.Booster(model_str=text)
    if list(booster.feature_name()) != names:
        raise RuntimeError("persisted feature order does not match the declared block")
    return booster, names
