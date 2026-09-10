"""Neutralized top-15 equal-weight basket on a 20-trading-day cadence.

Cadence: the label horizon is 20 trading days, so the book turns over on the
first decision day of each calendar month (a calendar month averages 20.3
A-share trading days). The rule needs no strategy state -- it reads only whether
the last visible daily bar still belongs to the current month -- so a cold
worker and a warm one produce identical orders. On a non-rebalance day the
function returns before building any face, so nineteen decisions in twenty cost
one bounded `daily` read.

Sells are timed at 09:30 and buys at 15:00 of the same day: the Broker
processes pending orders in timestamp order, so the proceeds are credited
before the buys are sized. `context.account` is the pre-call snapshot and never
changes mid-call, so the buy budget is decremented locally.

At most half the basket rotates per rebalance, which is what makes the declared
cost budget (~600% annual turnover, 1.8-2.2%/yr on a CNY 100,000 account)
match the one written into `README.md`.
"""

import numpy as np
import pandas as pd

from lib import flow
from lib import model

TOP_N = 15
MAX_REPLACE = 7             # at most half the basket rotates per rebalance
CASH_BUFFER = 0.97
SELL_HAIRCUT = 0.98
DECISION_LOOKBACK_DAYS = 200
FACE_WARMUP_TRADE_DAYS = 40  # >= flow.WINDOW plus room for filtered-out stock-days


def run(context, candidate):
    decision = pd.Timestamp(context.inference_at)
    sell_at = decision.replace(hour=9, minute=30, second=0, microsecond=0)
    buy_at = decision.replace(hour=15, minute=0, second=0, microsecond=0)
    if decision > buy_at:
        return []
    if decision > sell_at:
        sell_at = buy_at
    daily = flow.read_daily(context, DECISION_LOOKBACK_DAYS)
    if daily.empty:
        return []
    latest = str(daily["trade_date"].max())
    positions = {str(k): int(v) for k, v in dict(context.account.positions).items() if int(v) > 0}
    if not _is_rebalance(decision, latest, positions):
        return []
    ranked, degraded = _ranked_candidates(context, candidate, daily, latest)
    if ranked is None:
        return []
    prices = daily[daily["trade_date"] == latest].set_index("ts_code")["close"]
    target = list(ranked.index[:TOP_N])
    orders = _sell_orders(positions, target, ranked, sell_at, degraded)
    proceeds = sum(
        float(prices.get(order["symbol"], 0.0)) * order["quantity"] for order in orders)
    keep = [code for code in positions if code not in {order["symbol"] for order in orders}]
    budget = (float(context.account.cash) + proceeds * SELL_HAIRCUT) * CASH_BUFFER
    orders.extend(_buy_orders(candidate, target, keep, prices, budget, buy_at, degraded))
    return orders


def _is_rebalance(decision, latest, positions):
    """First decision day of a calendar month, or the first decision of the replay."""
    if not positions:
        return True
    return pd.Timestamp(latest).month != decision.month


def _ranked_candidates(context, candidate, daily, latest):
    universe = flow.read_universe(context)
    dates = sorted(daily["trade_date"].unique())[-FACE_WARMUP_TRADE_DAYS:]
    block, context_frame = flow.build_face(context, candidate, daily, dates)
    if block.empty:
        return None, False
    names = flow.feature_names(candidate)
    pool = flow.eligible(context, daily, universe, context_frame, latest)
    if len(pool) < 50:
        return None, False
    section = flow.cross_section(block, context_frame, names, latest, pool)
    if section.empty:
        return None, False
    booster = None
    degraded = False
    if flow.scorer_of(candidate) == "lgb":
        booster, names = model.load(context, candidate)
        degraded = booster is None
    if booster is not None:
        raw = booster.predict(section[names])
    else:
        raw = flow.equal_weight_score(section, candidate)
    industry = universe.set_index("ts_code")["l1_code"].reindex(section.index)
    residual = flow.neutralize(np.asarray(raw, dtype="float64"), section, industry)
    scored = pd.DataFrame({"score": residual, "code": section.index}, index=section.index)
    # Deterministic tie-break: a flat degraded score must not rank alphabetically by accident.
    ranked = scored.sort_values(["score", "code"], ascending=[False, True])["score"]
    return ranked, degraded


def _sell_orders(positions, target, ranked, execute_at, degraded):
    """Exit anything out of the pool or unscored, then the worst holdings, capped."""
    forced = [code for code in positions if code not in ranked.index]
    held = [code for code in positions if code in ranked.index and code not in target]
    held.sort(key=lambda code: (float(ranked[code]), code))
    drop = forced + held[: max(0, MAX_REPLACE - len(forced))]
    reason = "rebalance_exit_degraded" if degraded else "rebalance_exit"
    return [
        {
            "symbol": code,
            "action": "sell",
            "quantity": int(positions[code]),
            "execute_at": execute_at.isoformat(),
            "reason": reason,
        }
        for code in sorted(drop)
    ]


def _buy_orders(candidate, target, keep, prices, budget, execute_at, degraded):
    """Equal-cash top-up to TOP_N names, whole 100-share lots, locally decremented budget."""
    buys = [code for code in target if code not in keep]
    buys = buys[: max(0, TOP_N - len(keep))]
    orders = []
    remaining = budget
    for index, code in enumerate(buys):
        price = float(prices.get(code, float("nan")))
        if not np.isfinite(price) or price <= 0:
            continue
        share = remaining / max(1, len(buys) - index)
        quantity = int(share / price // 100 * 100)
        if quantity <= 0:
            continue
        orders.append(
            {
                "symbol": code,
                "action": "buy",
                "quantity": quantity,
                "execute_at": execute_at.isoformat(),
                "reason": (candidate + "_degraded") if degraded else candidate,
            }
        )
        remaining -= quantity * price
    return orders
