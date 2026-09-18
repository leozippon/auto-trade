"""Two sleeves held side by side in one account at fixed capital weights.

What this arm registers is the DENOMINATOR of the information ratio. Sleeve
`composite` (lib/composite.py) is fit-free, reviewed on the first decision of each
calendar month, and ranks a four-leg rank composite on a float-cap-floored pool.
Sleeve `pv` (lib/data.py + lib/score_lgbm.py) is a learned cross-sectional ranker
reviewed on the first decision of each ISO week. Their residuals correlate about
0.56, so holding them together at fixed weights is expected to cut tracking error
to sqrt((1 + rho) / 2) of the average leg's while alpha stays the weighted average.
`lib/common.py` says which sleeves a leg runs, at what weight and with how many
seats; the four legs differ in nothing else.

Ownership, and why it is defined by the keep band rather than by memory. A strategy
sees only cash and positions, never which sleeve bought a name, and nothing may be
cached across calls (a cold worker and a warm worker must produce the same orders).
So a holding BELONGS to a sleeve when it sits inside that sleeve's keep band --
ranked inside `seats * KEEP_BAND` of that sleeve's own ranking -- and a sleeve keeps
at most its `seats` best-ranked owned names. A holding may be owned by both sleeves
at once (an overlap) and is then one position at the summed weight; a holding owned
by neither is loose. This is exactly the keep-band semantics of the two source
packs, read off the current cross-section instead of off a remembered roster.

Selling. A holding that has left EVERY pool this leg ranks is a forced exit and is
always sold. A loose holding -- still in a pool, outside every keep band -- is sold
at the next review of either sleeve, at most `_swap_cap` names per reviewing sleeve
(forced exits count against that budget). A holding a sleeve OWNS is never sold by
the other sleeve, so a weekly review cannot unseat the monthly sleeve's book.

Selling a loose name on the other sleeve's review is what keeps the book bounded:
sells happen before fills, no sleeve fills past its own seats, and a loose name the
swap cap could not clear this review still occupies one of the book's seats, so the
book never holds more than the sum of the seats. It does not move the capital split,
because a buy is sized as a share of the BOOK and not as a share of the cash: a
sleeve with no free seat buys nothing, so the cash a monthly name's exit raises
simply waits for the monthly review that refills that seat.

Buying. Only a reviewing sleeve fills seats. A sleeve fills its free seats from its
own ranking, skipping names the book already holds -- those belong to the other
sleeve at that sleeve's weight, and this package never tops up an existing position,
so the entry size of every position is exact. An overlap therefore materialises when
both sleeves pick the same name in the same review, and `overlap_count` is reported
on every buy order. Target weight of a name = sum over the sleeves that seat it of
`sleeve weight / sleeve seats`, and its cash is that weight times the book's value,
in 100-share lots rounded to the NEAREST lot and clipped to the cash on hand -- see
`_buy_orders` for why rounding down is not an option at these position sizes.

Costs, which are this arm's real bar. The blend holds up to 30 positions, so at CNY
100k a position is about CNY 3,200 and the CNY 5 minimum commission alone is ~15.5 bp
a side, against ~7.9 bp for a 15-name book; a round trip runs ~36 bp plus stamp duty.
Halving the seat counts is a registered variant axis for exactly this reason.

Sells are timed 09:30 and buys 15:00 of the same day, so proceeds are credited before
buys are sized; `context.account` never changes mid-call, so the buy budget is
decremented locally and sized at the T-1 close.
"""

from datetime import timedelta

import numpy as np
import pandas as pd

from lib import common, composite, data, score_lgbm

KEEP_BAND = 2.0              # a holding ranked inside seats * KEEP_BAND is kept by that sleeve
SEATS_REF = common.SEATS     # the seat count the swap caps below are quoted at
MAX_SWAPS = {                # discretionary sells per review of that sleeve, at SEATS_REF seats
    common.SLEEVE_A: 10,     # monthly review
    common.SLEEVE_B: 2,      # weekly review
}
MIN_BARS = 60
MIN_LISTED_DAYS = 120
PRICE_FLOOR = 1.0            # CNY
PRICE_CAP = 30.0             # T-1 close, CNY: one 100-share lot costs at most CNY 3,000 --
                             # more than a sleeve-A seat at W_A = 0.30, hence the nearest-lot rounding
ADV_FLOOR = 3.0e7            # 20-day mean amount, CNY
ADV_DAYS = 20
CASH_BUFFER = 0.97
SELL_HAIRCUT = 0.98


def run(context, candidate):
    decision = pd.Timestamp(context.inference_at)
    sell_at = decision.replace(hour=9, minute=30, second=0, microsecond=0)
    buy_at = decision.replace(hour=15, minute=0, second=0, microsecond=0)
    if decision > buy_at:
        return []
    if decision > sell_at:
        sell_at = buy_at
    positions = {str(k): int(v) for k, v in dict(context.account.positions).items() if int(v) > 0}
    sleeves = common.LEGS[candidate]
    due = set(sleeves) if not positions else _due_sleeves(context, decision, sleeves)
    if not due:
        return []

    ranked, prices, extra = {}, pd.Series(dtype="float64"), {}
    for sleeve in sleeves:
        order, sleeve_prices, sleeve_extra = _rank(context, sleeve)
        ranked[sleeve] = order
        prices = prices.combine_first(sleeve_prices)
        extra.update(sleeve_extra)
    rank = {sleeve: {code: index for index, code in enumerate(order)}
            for sleeve, order in ranked.items()}

    owned = _owned(positions, rank, sleeves)
    seated = {code for codes in owned.values() for code in codes}
    forced = [code for code in positions if not any(code in rank[s] for s in sleeves)]
    loose = [code for code in positions if code not in seated and code not in forced]
    loose.sort(key=lambda code: (-_best_rank(code, rank, sleeves), code))
    budget_swaps = sum(_swap_cap(sleeve, sleeves[sleeve][1]) for sleeve in due)
    available = max(0, sum(len(order) for order in ranked.values()) - len(positions))
    drop = forced + loose[: max(0, min(budget_swaps - len(forced), available))]
    sells = [
        {"symbol": code, "action": "sell", "quantity": int(positions[code]),
         "execute_at": sell_at.isoformat(), "reason": candidate + "_exit"}
        for code in sorted(drop)
    ]

    held = set(positions) - set(drop)
    rosters = _rosters(ranked, owned, sleeves, due, held)
    weights, sleeve_of = _targets(rosters, sleeves)
    overlap = sum(1 for names in sleeve_of.values() if len(names) > 1)
    proceeds = sum(_price(prices, order["symbol"]) * order["quantity"] for order in sells)
    budget = (float(context.account.cash) + proceeds * SELL_HAIRCUT) * CASH_BUFFER
    book_value = budget + sum(_price(prices, code) * positions[code] for code in held)
    extra = {**extra, "overlap_count": overlap}
    buys = _buy_orders(weights, sleeve_of, held, prices, budget, book_value, buy_at, candidate, extra)
    return sells + buys


def _rank(context, sleeve):
    """(ranked ts_code list best first, T-1 close of every visible name, buy metadata)."""
    if sleeve == common.SLEEVE_A:
        section, closes = composite.build(context)
        if section is None:
            raise RuntimeError("fewer than 30 scorable names in the composite pool")
        order = section.sort_values(["score", "ts_code"], ascending=[False, True])["ts_code"]
        return list(order.astype(str)), closes.astype("float64"), {}
    W = data.decision_wide(context)
    if W["symbols"].shape[0] == 0:
        raise RuntimeError("the daily window of the as-of view holds no names")
    scores, meta = score_lgbm.score(context, data.decision_features(context, W))
    pool = _pool_mask(context, W) & np.isfinite(scores)
    frame = pd.DataFrame({"ts_code": W["symbols"][pool].astype(str), "score": scores[pool]})
    frame = frame.sort_values(["score", "ts_code"], ascending=[False, True])
    prices = pd.Series(W["raw_close"][:, -1], index=W["symbols"].astype(str))
    return list(frame["ts_code"]), prices.astype("float64"), meta


def _pool_mask(context, W):
    """(S,) bool: the names sleeve `pv` may hold or buy, on the newest visible bar."""
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


def _due_sleeves(context, decision, sleeves):
    """The sleeves reviewing today: monthly for `composite`, weekly for `pv`.

    Both rules compare the newest visible trading day with the decision day, so they
    are stateless. One bounded read of trade dates serves both.
    """
    start = (context.inference_at - timedelta(days=10)).strftime("%Y%m%d")
    days = pd.read_parquet(context.asof_dir + "/daily", columns=["trade_date"],
                           filters=[("trade_date", ">=", start)])
    if days.empty:
        return set(sleeves)
    latest = pd.Timestamp(str(days["trade_date"].max()))
    due = set()
    if common.SLEEVE_A in sleeves and (latest.year, latest.month) != (decision.year, decision.month):
        due.add(common.SLEEVE_A)
    if common.SLEEVE_B in sleeves and latest.isocalendar()[:2] != decision.isocalendar()[:2]:
        due.add(common.SLEEVE_B)
    return due


def _swap_cap(sleeve, seats):
    """Discretionary sells one review of `sleeve` may make, scaled with its seat count."""
    return max(1, round(MAX_SWAPS[sleeve] * seats / SEATS_REF))


def _owned(positions, rank, sleeves):
    """sleeve -> the holdings it owns: inside its keep band, its `seats` best kept."""
    owned = {}
    for sleeve, (_weight, seats) in sleeves.items():
        inside = [code for code in positions
                  if code in rank[sleeve] and rank[sleeve][code] < seats * KEEP_BAND]
        inside.sort(key=lambda code: (rank[sleeve][code], code))
        owned[sleeve] = inside[:seats]
    return owned


def _best_rank(code, rank, sleeves):
    """The best rank a holding reaches in any sleeve; a name outside every pool is last."""
    return min(rank[sleeve].get(code, len(rank[sleeve])) for sleeve in sleeves)



def _rosters(ranked, owned, sleeves, due, held):
    """sleeve -> the names it seats after this review.

    A reviewing sleeve fills its free seats from its own ranking, skipping names the
    book already holds: those are the other sleeve's position at the other sleeve's
    weight, and this package never tops up an existing position. A name both reviewing
    sleeves pick is one position, seated by both, and costs one seat of the book. A
    sleeve that is not reviewing seats exactly what it still owns.

    The book has `sum(seats)` seats and no more. A loose holding the swap cap could not
    clear this review still occupies one, so a reviewing sleeve fills only what the book
    has room for and the rest of its seats wait for the next review. The reviewing
    sleeves are served in name order, which gives the slower sleeve the scarce seat --
    it is the one that cannot come back next week.
    """
    capacity = sum(seats for _weight, seats in sleeves.values()) - len(held)
    rosters = {sleeve: [code for code in owned[sleeve] if code in held] for sleeve in sleeves}
    added = set()
    for sleeve in sorted(due):
        _weight, seats = sleeves[sleeve]
        roster = rosters[sleeve]
        for code in ranked[sleeve]:
            if len(roster) >= seats:
                break
            if code in roster or code in held:
                continue
            if code not in added:
                if capacity <= 0:
                    break
                capacity -= 1
                added.add(code)
            roster.append(code)
    return rosters


def _targets(rosters, sleeves):
    """(ts_code -> target weight of the book, ts_code -> the sleeves that seat it)."""
    weights, sleeve_of = {}, {}
    for sleeve, roster in rosters.items():
        weight, seats = sleeves[sleeve]
        for code in roster:
            weights[code] = weights.get(code, 0.0) + weight / seats
            sleeve_of.setdefault(code, []).append(sleeve)
    return weights, sleeve_of


def _price(prices, code):
    """The T-1 close of one name, or 0 when it has none: a holding halted all of T-1
    has a row in the window but no price on its last bar, and a NaN would poison the
    whole buy budget rather than just its own sale."""
    price = float(prices.get(code, float("nan")))
    return price if np.isfinite(price) and price > 0 else 0.0


def _buy_orders(weights, sleeve_of, held, prices, budget, book_value, execute_at, candidate, extra):
    """Whole-lot buys of the names this review adds, each sized as a share of the book.

    A name's cash is `book_value * target_weight`, capped by what is left, rather than a
    share of the day's budget: the two sleeves seat their names at different weights, an
    overlap seats one name at the sum of the two, and a sleeve that opens no seat must
    not spend the cash the other sleeve's exit raised.

    The lot is rounded to the NEAREST 100 shares and then clipped to the cash on hand,
    not rounded down. At W_A = 0.30 a sleeve-A seat is about CNY 2,000 while one lot of a
    name priced between the pool's floor and cap costs CNY 100 to 3,000, so rounding down
    would leave the light sleeve about a quarter short of its weight on every review and
    the realised capital split would not be the registered one (measured: 0.157 / 0.700
    on the first smoke review, against 0.30 / 0.70 targeted). Rounding to the nearest lot
    makes the error two-sided instead; it does not make it small, so every buy reports its
    `realized_weight` beside its `target_weight` and the realised split is a reading this
    arm owes on every round, not an assumption.
    """
    buys = sorted((code for code in weights if code not in held),
                  key=lambda code: (-weights[code], code))
    orders = []
    remaining = budget
    for code in buys:
        weight = weights[code]
        price = _price(prices, code)
        if price <= 0:
            continue
        quantity = int(min(remaining, book_value * weight) / price / 100 + 0.5) * 100
        while quantity > 0 and quantity * price > remaining:
            quantity -= 100
        if quantity <= 0:
            continue
        orders.append({
            "symbol": code, "action": "buy", "quantity": quantity,
            "execute_at": execute_at.isoformat(), "reason": candidate + "_entry",
            "sleeve": "+".join(sleeve_of[code]),
            "target_weight": round(weight, 6),
            "realized_weight": round(quantity * price / book_value, 6) if book_value > 0 else 0.0,
            **extra,
        })
        remaining -= quantity * price
    return orders
