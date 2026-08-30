"""Formal daily strategy entrypoint -- minimal working baseline.

The Environment calls ``fit(context)`` at the first decision of a replay and
again at the first decision of every new ``REFIT_PERIOD``, then
``generate_orders(context)`` at every scheduled decision. Both receive the
same point-in-time context: ``fit`` may write files under
``context.state_dir``; ``generate_orders`` can only read them.

This baseline fits a two-feature ridge ranker (5-day and 20-day adjusted
returns against the following 5-day return) on the visible daily history and
stores its coefficients as one NumPy array. While flat, it buys an
equal-budget basket of the highest-scoring symbols at the next same-day daily
price timestamp; an after-close invocation emits no order because the strategy
does not receive a future trading calendar. It is deliberately small: replace
the features and add an explicit exit lifecycle. A larger strategy splits into
sibling modules under ``output/`` (see README.md); this one-file baseline is
the smallest such package.
"""

from __future__ import annotations

import math
from datetime import timedelta

import numpy as np
import pandas as pd

REFIT_PERIOD = "quarter"
FIT_LOOKBACK_DAYS = 400  # calendar days of daily history read by one fit
FEATURE_LOOKBACK_DAYS = 45  # calendar days read per decision; covers 20 trading days
MIN_FIT_SAMPLES = 200
RIDGE_LAMBDA = 1.0
TOP_N = 10
CASH_FRACTION = 0.95


def _same_day_execution(inference_at):
    market_open = inference_at.replace(hour=9, minute=30, second=0, microsecond=0)
    if inference_at <= market_open:
        return market_open
    market_close = inference_at.replace(hour=15, minute=0, second=0, microsecond=0)
    if inference_at <= market_close:
        return market_close
    return None


def _daily_since(context, lookback_days):
    """Visible daily rows from the PIT view, bounded to a calendar lookback."""

    start = (context.inference_at - timedelta(days=lookback_days)).strftime("%Y%m%d")
    frame = pd.read_parquet(
        context.asof_dir + "/daily",
        columns=["ts_code", "trade_date", "close", "adj_factor"],
        filters=[("trade_date", ">=", start)],
    ).dropna()
    frame = frame[frame["close"] > 0].sort_values(["ts_code", "trade_date"])
    frame["adj_close"] = frame["close"] * frame["adj_factor"]
    return frame.reset_index(drop=True)


def _features(frame):
    """Per row: cross-sectionally standardized 5-day and 20-day adjusted returns."""

    grouped = frame.groupby("ts_code")["adj_close"]
    out = frame[["ts_code", "trade_date"]].copy()
    out["ret_5"] = grouped.pct_change(5).to_numpy()
    out["ret_20"] = grouped.pct_change(20).to_numpy()
    out = out.dropna()
    for column in ("ret_5", "ret_20"):
        by_day = out.groupby("trade_date")[column]
        spread = by_day.transform("std").replace(0.0, np.nan)
        out[column] = ((out[column] - by_day.transform("mean")) / spread).fillna(0.0)
    return out


def fit(context):
    """Fit the ridge ranker on realized labels and persist it under state_dir."""

    frame = _daily_since(context, FIT_LOOKBACK_DAYS)
    features = _features(frame)
    # Label: the 5-day forward adjusted return, kept only where it has fully
    # realized before this decision (every row read is already visible).
    forward = frame.groupby("ts_code")["adj_close"].shift(-5) / frame["adj_close"] - 1.0
    sample = features.join(forward.rename("target"), how="inner").dropna()
    coef = np.zeros(3)
    if len(sample) >= MIN_FIT_SAMPLES:
        design = np.column_stack(
            [np.ones(len(sample)), sample["ret_5"].to_numpy(), sample["ret_20"].to_numpy()]
        )
        target = sample["target"].to_numpy()
        penalty = RIDGE_LAMBDA * np.eye(design.shape[1])
        coef = np.linalg.solve(design.T @ design + penalty, design.T @ target)
    np.save(context.state_dir + "/ridge_coef.npy", coef)


def generate_orders(context):
    if context.account.positions:
        return []

    coef = np.load(context.state_dir + "/ridge_coef.npy")
    frame = _daily_since(context, FEATURE_LOOKBACK_DAYS)
    if frame.empty:
        return []
    latest_date = frame["trade_date"].max()
    features = _features(frame)
    features = features[features["trade_date"] == latest_date].copy()
    if features.empty:
        return []
    features["score"] = (
        coef[0] + coef[1] * features["ret_5"] + coef[2] * features["ret_20"]
    )
    ranked = features.sort_values(["score", "ts_code"], ascending=[False, True])
    prices = (
        frame[frame["trade_date"] == latest_date].set_index("ts_code")["close"].to_dict()
    )
    symbols = [
        symbol
        for symbol in ranked["ts_code"].tolist()
        if math.isfinite(float(prices.get(symbol, 0.0))) and float(prices.get(symbol, 0.0)) > 0
    ][:TOP_N]
    if not symbols:
        return []

    execution_at = _same_day_execution(context.inference_at)
    if execution_at is None:
        return []
    execution = execution_at.isoformat()
    remaining = float(context.account.cash) * CASH_FRACTION
    orders = []
    for index, symbol in enumerate(symbols):
        price = float(prices[symbol])
        target_budget = remaining / (len(symbols) - index)
        quantity = int(target_budget / price // 100 * 100)
        if quantity <= 0:
            continue
        orders.append(
            {
                "symbol": symbol,
                "action": "buy",
                "quantity": quantity,
                "execute_at": execution,
                "reason": "ridge_rank_equal_budget_baseline",
            }
        )
        remaining -= quantity * price
    return orders
