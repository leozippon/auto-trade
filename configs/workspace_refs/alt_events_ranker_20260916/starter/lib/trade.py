"""Neutralized top-N equal-weight basket, monthly cadence, lot- and T+1-aware.

Cadence: the label horizon is 20 trading days, so the book turns over on the
first decision day of each calendar month (a calendar month averages 20.3
A-share trading days). The rule needs no strategy state — it reads only whether
the last visible bar still belongs to the current month — so a cold worker and
a warm one produce identical orders.

Sells are timed at 09:30 and buys at 15:00 of the same day: the Broker
processes pending orders in timestamp order, so the proceeds are credited
before the buys are sized. `context.account` is the pre-call snapshot and never
changes mid-call, so the buy budget is decremented locally.
"""

import numpy as np
import pandas as pd

from lib import features
from lib import model

TOP_N = 15
MAX_REPLACE = 7             # at most half the basket rotates per rebalance
CASH_BUFFER = 0.97
SELL_HAIRCUT = 0.98         # conservative estimate of proceeds credited before the buys
DECISION_LOOKBACK_DAYS = 300


def run(context, daily_only, use_learner):
    decision = pd.Timestamp(context.inference_at)
    sell_at = decision.replace(hour=9, minute=30, second=0, microsecond=0)
    buy_at = decision.replace(hour=15, minute=0, second=0, microsecond=0)
    if decision > buy_at:
        return []
    if decision > sell_at:
        sell_at = buy_at
    daily = features.read_daily(context, DECISION_LOOKBACK_DAYS)
    if daily.empty:
        return []
    latest = str(daily["trade_date"].max())
    positions = {str(k): int(v) for k, v in dict(context.account.positions).items() if int(v) > 0}
    if not _is_rebalance(decision, latest, positions):
        return []
    ranked, degraded = _ranked_candidates(context, daily, latest, daily_only, use_learner)
    if ranked is None:
        return []
    prices = daily[daily["trade_date"] == latest].set_index("ts_code")["close"]
    target = list(ranked.index[:TOP_N])
    orders = _sell_orders(positions, target, ranked, sell_at, degraded)
    proceeds = sum(
        float(prices.get(order["symbol"], 0.0)) * order["quantity"] for order in orders)
    keep = [code for code in positions if code not in
            {order["symbol"] for order in orders}]
    budget = (float(context.account.cash) + proceeds * SELL_HAIRCUT) * CASH_BUFFER
    orders.extend(_buy_orders(target, keep, prices, budget, buy_at, degraded))
    return orders


def _is_rebalance(decision, latest, positions):
    """First decision day of a calendar month, or the first decision of the replay."""
    if not positions:
        return True
    return pd.Timestamp(latest).month != decision.month


def _ranked_candidates(context, daily, latest, daily_only, use_learner):
    universe = features.read_universe(context)
    tables = features.read_all_events(context) if not daily_only else features.empty_tables()
    price, block_ev, unlocks, adv, amount_lookup = features.prepare(context, daily, tables)
    pool = features.eligible(
        context, daily, universe, latest, model.PRICE_CAP, model.ADV_FLOOR, adv)
    if len(pool) < 50:
        return None, False
    section = features.cross_section(
        context, latest, daily, price, tables, block_ev, unlocks,
        features.adv_sum_at(adv, latest), amount_lookup, daily_only)
    if section.empty:
        return None, False
    section = section.reindex(pool)
    booster, names = (None, features.feature_names(daily_only))
    degraded = False
    if use_learner:
        booster, names = model.load(context)
        degraded = booster is None
    standardized = features.standardize(section, names)
    if use_learner and booster is not None:
        raw = booster.predict(standardized[names])
    elif use_learner:
        raw = features.equal_weight_score(standardized)
    else:
        raw = features.equal_weight_score(standardized)
    industry = universe.set_index("ts_code")["l1_code"].reindex(standardized.index)
    scored = standardized.assign(l1_code=industry)
    residual = features.neutralize(np.asarray(raw, dtype="float64"), scored)
    ranked = pd.Series(residual, index=standardized.index).sort_values(ascending=False)
    # Deterministic tie-break: the block-trade and chip columns carry large tied blocks.
    ranked = ranked.to_frame("score").assign(code=ranked.index)
    ranked = ranked.sort_values(["score", "code"], ascending=[False, True])["score"]
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


def _buy_orders(target, keep, prices, budget, execute_at, degraded):
    """Equal-cash top-up to TOP_N names, whole 100-share lots, locally decremented budget."""
    buys = [code for code in target if code not in keep]
    slots = max(0, TOP_N - len(keep))
    buys = buys[:slots]
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
                "reason": "events_rank_degraded" if degraded else "events_rank",
            }
        )
        remaining -= quantity * price
    return orders
