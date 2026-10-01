"""Top-15 equal-cash book reviewed once a week, with a keep band and a swap cap.

Review: the first decision of each ISO week, and any decision while the account is
flat. The rule compares the decision day's ISO week with that of the newest visible
trading day, so it is stateless: a cold worker and a warm one produce the same orders.
A non-review decision reads only a few days of trade dates.

Pool on the newest visible bar (T-1): a bar that day, >= MIN_BARS bars in the decision
window, not STAR (688/689) and not BJ, name without ST or 退, listed >= MIN_LISTED_DAYS,
PRICE_FLOOR < close <= PRICE_CAP, 20-day mean amount >= ADV_FLOOR, and a finite score.

Costs: at CNY 100k a 15-name position is about CNY 6,300, so the CNY 5 minimum
commission alone is ~8 bp a side and a round trip ~25-35 bp. The book therefore sells a
holding only when it left the pool, or ranks outside TOP_N * KEEP_BAND and a fresh top
name can replace it, and makes at most MAX_SWAPS replacements a review (forced exits
come first and are always sold).

Sells are timed 09:30 and buys 15:00 of the same day, so proceeds are credited before
buys are sized; `context.account` never changes mid-call, so the buy budget is
decremented locally and sized at the T-1 close in 100-share lots.
"""

from datetime import timedelta

import numpy as np
import pandas as pd

from lib import data, score_lgbm

TOP_N = 15
KEEP_BAND = 2.0              # a holding ranked inside TOP_N * KEEP_BAND is kept
MAX_SWAPS = 2                # replacements per weekly review, forced exits included
MIN_BARS = 60
MIN_LISTED_DAYS = 120
PRICE_FLOOR = 1.0            # CNY
PRICE_CAP = 30.0             # T-1 close, CNY: one 100-share lot stays under half a position
ADV_FLOOR = 3.0e7            # 20-day mean amount, CNY
ADV_DAYS = 20
CASH_BUFFER = 0.97
SELL_HAIRCUT = 0.98
REASON = "a158_lgbm"


def run(context):
    decision = pd.Timestamp(context.inference_at)
    sell_at = decision.replace(hour=9, minute=30, second=0, microsecond=0)
    buy_at = decision.replace(hour=15, minute=0, second=0, microsecond=0)
    if decision > buy_at:
        return []
    if decision > sell_at:
        sell_at = buy_at
    positions = {str(k): int(v) for k, v in dict(context.account.positions).items() if int(v) > 0}
    if positions and not _new_week(context, decision):
        return []
    W = data.decision_wide(context)
    if W["symbols"].shape[0] == 0:
        return []
    scores, extra = score_lgbm.score(context, data.decision_features(context, W))
    close = W["raw_close"][:, -1]
    pool = pool_mask(context, W) & np.isfinite(scores)
    ranked = pd.DataFrame({"ts_code": W["symbols"][pool].astype(str), "score": scores[pool]})
    ranked = ranked.sort_values(["score", "ts_code"], ascending=[False, True]).reset_index(drop=True)
    rank = {code: index for index, code in enumerate(ranked["ts_code"])}
    prices = pd.Series(close, index=W["symbols"].astype(str))
    target = list(ranked["ts_code"][:TOP_N])
    sells = _sell_orders(positions, rank, target, sell_at)
    proceeds = sum(_price(prices, order["symbol"]) * order["quantity"] for order in sells)
    keep = [code for code in positions if code not in {order["symbol"] for order in sells}]
    budget = (float(context.account.cash) + proceeds * SELL_HAIRCUT) * CASH_BUFFER
    return sells + _buy_orders(target, keep, prices, budget, buy_at, extra)


def pool_mask(context, W):
    """(S,) bool: the names a review may hold or buy, on the newest visible bar."""
    symbols = pd.Series(W["symbols"].astype(str))
    universe = pd.read_parquet(context.asof_dir + "/universe", columns=["ts_code", "name", "list_date"])
    universe = universe.drop_duplicates("ts_code").set_index("ts_code")
    names = universe["name"].reindex(symbols).fillna("").astype(str)
    listed = pd.to_datetime(universe["list_date"].reindex(symbols), format="%Y%m%d", errors="coerce")
    listed_days = (pd.Timestamp(context.inference_at.date()) - listed).dt.days.to_numpy()
    amount = W["raw_amount"][:, -ADV_DAYS:]
    present = np.isfinite(amount).sum(axis=1)
    adv = np.where(present >= ADV_DAYS // 2, np.nansum(amount, axis=1) / np.maximum(present, 1), 0.0)
    close = W["raw_close"][:, -1]
    with np.errstate(invalid="ignore"):
        return (
            np.isfinite(close)
            & (W["n_bars"] >= MIN_BARS)
            & ~symbols.str.startswith(("688", "689")).to_numpy()
            & ~names.str.contains("ST|退").to_numpy()
            & (np.nan_to_num(listed_days, nan=-1.0) >= MIN_LISTED_DAYS)
            & (close > PRICE_FLOOR) & (close <= PRICE_CAP)
            & (adv >= ADV_FLOOR)
        )


def _new_week(context, decision):
    start = (context.inference_at - timedelta(days=10)).strftime("%Y%m%d")
    days = pd.read_parquet(context.asof_dir + "/daily", columns=["trade_date"], filters=[("trade_date", ">=", start)])
    if days.empty:
        return True
    latest = pd.Timestamp(str(days["trade_date"].max()))
    return latest.isocalendar()[:2] != decision.isocalendar()[:2]


def _price(prices, code):
    price = float(prices.get(code, float("nan")))
    return price if np.isfinite(price) and price > 0 else 0.0


def _sell_orders(positions, rank, target, execute_at):
    forced = [code for code in positions if code not in rank]
    outside = [code for code in positions if code in rank and rank[code] >= TOP_N * KEEP_BAND]
    outside.sort(key=lambda code: (-rank[code], code))
    fresh = [code for code in target if code not in positions]
    replaceable = max(0, min(MAX_SWAPS - len(forced), len(fresh)))
    drop = forced + outside[:replaceable]
    return [
        {"symbol": code, "action": "sell", "quantity": int(positions[code]),
         "execute_at": execute_at.isoformat(), "reason": REASON + "_exit"}
        for code in sorted(drop)
    ]


def _buy_orders(target, keep, prices, budget, execute_at, extra):
    buys = [code for code in target if code not in keep][: max(0, TOP_N - len(keep))]
    orders = []
    remaining = budget
    for index, code in enumerate(buys):
        price = _price(prices, code)
        if price <= 0:
            continue
        quantity = int(remaining / max(1, len(buys) - index) / price // 100 * 100)
        if quantity <= 0:
            continue
        orders.append({"symbol": code, "action": "buy", "quantity": quantity,
                       "execute_at": execute_at.isoformat(), "reason": REASON + "_entry", **extra})
        remaining -= quantity * price
    return orders
