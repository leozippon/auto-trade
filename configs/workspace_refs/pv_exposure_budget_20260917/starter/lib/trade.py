"""Top-15 equal-cash book reviewed once a week, filled inside an exposure budget.

The score is borrowed and fixed (lib/score_lgbm.py). What this arm registers is the
step after it: the 15 seats are filled inside SW L1 industry x float-cap tercile
buckets -- at most INDUSTRY_CAP names of one industry, at least one name of each
tercile -- instead of straight off the whole-pool ranking.

Review: the first decision of each ISO week, and any decision while the account is
flat. The rule compares the decision day's ISO week with that of the newest visible
trading day, so it is stateless: a cold worker and a warm one produce the same orders.
A non-review decision reads only a few days of trade dates.

Pool on the newest visible bar (T-1): a bar that day, >= MIN_BARS bars in the decision
window, not STAR (688/689) and not BJ, name without ST or 退, listed >= MIN_LISTED_DAYS,
PRICE_FLOOR < close <= PRICE_CAP, 20-day mean amount >= ADV_FLOOR, a finite float cap
(the buckets rank on it) and a finite score.

The cap is counted against the book the review will actually hold, the retained
holdings included, never against a freshly ranked top 15. That is what makes it hold:
the seats the keep band carries over came from a book that already respected the cap,
and the free seats are filled under a counter seeded with them, so no review can push
an industry past INDUSTRY_CAP. Capping a freshly ranked top 15 instead would leak
straight through the keep band, which retains holdings ranked behind that 15 and never
counts their industries.

With the buckets off, `_book` returns the retained holdings plus the best-scored names
that are not retained, filled to TOP_N -- which is the unbucketed book name for name.
`c_uncapped` is therefore the borrowed score's own whole-pool weekly top 15, and the
buckets are the only free variable of this arm.

Costs: at CNY 100k a 15-name position is about CNY 6,300, so the CNY 5 minimum
commission alone is ~8 bp a side and a round trip ~25-35 bp. The book sells a holding
only when it left the pool, or ranks outside TOP_N * KEEP_BAND, and makes at most
MAX_SWAPS replacements a review (forced exits come first and are always sold). The
discretionary sells are gated on what the pool can replace rather than on how many
names of the fresh top 15 are new, because under buckets the replacements do not come
off that 15; the two gates pick the same holdings, since a book of TOP_N names always
has at least as many fresh top-15 names as holdings outside the band.

Sells are timed 09:30 and buys 15:00 of the same day, so proceeds are credited before
buys are sized; `context.account` never changes mid-call, so the buy budget is
decremented locally and sized at the T-1 close in 100-share lots.
"""

from datetime import timedelta

import numpy as np
import pandas as pd

from lib import common, data, score_lgbm

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

INDUSTRY_CAP = 2             # at most this many names of one SW L1 industry in the book
CAP_TERCILES = 3             # float-cap terciles of the pool; the book holds >= 1 of each

# An arbitrary but permanently fixed constant: `c_rand` is only a control while its
# draw never moves, neither between legs of a batch nor between rounds.
RANDOM_SEED = 2718281828
CODE_SPACE = 1_000_000       # A-share numeric codes; one draw per code makes the random score a property of the name, not of the day's pool


def run(context, candidate):
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
    section = data.decision_section(context, W["symbols"])
    pool = _pool_mask(W, section)
    scores, extra = _scores(context, W, pool, candidate)
    ranked = _shape(section, scores, pool & np.isfinite(scores))
    rank = {code: index for index, code in enumerate(ranked["ts_code"])}
    # Prices cover every name with a bar, not just the pool: a holding that left the
    # pool is a forced sell this review still has to value, and pricing it at zero
    # would shrink the buy budget by exactly the cash that sale raises.
    prices = pd.Series(W["raw_close"][:, -1], index=W["symbols"].astype(str))
    sells = _sell_orders(positions, rank, len(ranked), sell_at, candidate)
    proceeds = sum(_price(prices, order["symbol"]) * order["quantity"] for order in sells)
    keep = [code for code in positions if code not in {order["symbol"] for order in sells}]
    target, cover, top_industry = _book(ranked, keep, candidate)
    budget = (float(context.account.cash) + proceeds * SELL_HAIRCUT) * CASH_BUFFER
    extra = {**extra, "tercile_cover": cover, "top_industry_count": top_industry}
    return sells + _buy_orders(target, keep, prices, budget, buy_at, candidate, extra)


def _price(prices, code):
    """The T-1 close of one name, or 0 when it has none: a holding halted all of T-1
    has a row in the window but no price on its last bar, and a NaN would poison the
    whole buy budget rather than just its own sale."""
    price = float(prices.get(code, float("nan")))
    return price if np.isfinite(price) and price > 0 else 0.0


def _scores(context, W, pool, candidate):
    """(S,) score in `symbols` order, and the buy-order metadata of the leg's source."""
    source = common.LEGS[candidate][0]
    if source == "lgbm":
        return score_lgbm.score(context, data.decision_features(context, W))
    if source == "random":
        return _random_score(pd.Series(W["symbols"].astype(str))).to_numpy(), {}
    return _pool_spread(pool), {}


def _random_score(codes):
    """A fixed random score, one permanent draw per A-share numeric code.

    Drawing once per code rather than once per decision makes the random book a stable
    15 names that only turns over when a name leaves the pool -- the bucketing control
    has to hold the buckets fixed and the ranking uninformative, not add turnover the
    candidate does not have. The same seed and the same table give the same value in a
    cold worker, a warm worker and any later round.
    """
    table = np.random.default_rng(RANDOM_SEED).random(CODE_SPACE)
    numeric = pd.to_numeric(codes.str.slice(0, 6), errors="coerce").fillna(0).astype("int64")
    return pd.Series(table[numeric.to_numpy() % CODE_SPACE], index=codes.index)


def _pool_spread(pool):
    """A score-free ranking that spreads TOP_N names evenly over the eligible pool.

    The pool cannot be held on this account -- a couple of thousand names at one lot
    each is an order of magnitude more cash than the book has -- so the equal-weight-pool
    control is a stratified sample of it at book size: the names sitting at TOP_N evenly
    spaced positions of the pool ordered by `ts_code`, which spreads the sample across
    boards and code ranges and carries no information about the score. Descending order
    puts those TOP_N names first and their neighbours behind them, so the keep band and
    the swap cap work exactly as for a scored leg.
    """
    scores = np.full(pool.shape, np.nan, dtype=np.float64)
    size = int(pool.sum())
    if size == 0:
        return scores
    positions = np.arange(size)
    anchors = np.round(np.arange(TOP_N) * size / TOP_N).astype(int)
    scores[pool] = -np.abs(positions[:, None] - anchors[None, :]).min(axis=1).astype(np.float64)
    return scores


def _pool_mask(W, section):
    """(S,) bool: the names a review may hold or buy, on the newest visible bar."""
    amount = W["raw_amount"][:, -ADV_DAYS:]
    present = np.isfinite(amount).sum(axis=1)
    adv = np.where(present >= ADV_DAYS // 2, np.nansum(amount, axis=1) / np.maximum(present, 1), 0.0)
    close = W["raw_close"][:, -1]
    with np.errstate(invalid="ignore"):
        return (
            np.isfinite(close)
            & (W["n_bars"] >= MIN_BARS)
            & ~section["ts_code"].str.startswith(("688", "689")).to_numpy()
            & ~section["name"].str.contains("ST|退").to_numpy()
            & (np.nan_to_num(section["listed_days"].to_numpy(dtype=np.float64), nan=-1.0) >= MIN_LISTED_DAYS)
            & (close > PRICE_FLOOR) & (close <= PRICE_CAP)
            & (adv >= ADV_FLOOR)
            & np.isfinite(section["circ_mv"].to_numpy(dtype=np.float64))
        )


def _shape(section, scores, pool):
    """The pooled cross-section, tercile-tagged and then ranked by score.

    `cap_tercile` is attached on the pool and before the ranking, because the terciles
    are a property of that pool and not of the day's score.
    """
    frame = section.loc[pool].copy()
    frame["score"] = scores[pool]
    frame = frame.sort_values("ts_code").reset_index(drop=True)
    frame["cap_tercile"] = _terciles(frame["circ_mv"].to_numpy(dtype=np.float64))
    return frame.sort_values(["score", "ts_code"], ascending=[False, True]).reset_index(drop=True)


def _terciles(circ_mv):
    """Float-cap terciles of the pool: 0 is the smallest third.

    Split by position in the sorted pool rather than by a value quantile, so duplicate
    float caps cannot collapse an edge and the three groups always exist. The pool is
    ordered by `ts_code` here, which makes the tie-break stable across workers.
    """
    size = len(circ_mv)
    tercile = np.zeros(size, dtype="int64")
    tercile[np.argsort(circ_mv, kind="stable")] = np.minimum(
        np.arange(size) * CAP_TERCILES // max(size, 1), CAP_TERCILES - 1
    )
    return tercile


def _book(ranked, keep, candidate):
    """The TOP_N names this review holds, the terciles they cover and the largest
    industry count in them.

    The retained holdings take their seats first and seed the industry counter; the free
    seats go to the best-scored names their industry still has room for. The cover and
    the industry count are reported rather than enforced: a tercile with no name the cap
    can take is a property of that day's cross-section, and every buy order carries both
    readings, so a shortfall shows up in the run instead of being absorbed in silence.
    """
    bucketed = common.LEGS[candidate][1]
    codes = list(ranked["ts_code"])
    industry = dict(zip(codes, ranked["industry"]))
    tercile = dict(zip(codes, ranked["cap_tercile"]))
    retained = set(keep)
    book = [code for code in codes if code in retained]
    counts = {}
    for code in book:
        counts[industry[code]] = counts.get(industry[code], 0) + 1
    added = []
    for code in codes:
        if len(book) + len(added) >= TOP_N:
            break
        if code in retained:
            continue
        if bucketed and counts.get(industry[code], 0) >= INDUSTRY_CAP:
            continue
        counts[industry[code]] = counts.get(industry[code], 0) + 1
        added.append(code)
    if bucketed:
        added = _cover_terciles(book, added, codes, industry, tercile, counts)
    book = [code for code in codes if code in retained or code in set(added)]
    held = [industry[code] for code in book]
    top = max((held.count(name) for name in set(held)), default=0)
    return book, len({tercile[code] for code in book}), top


def _cover_terciles(retained, added, codes, industry, tercile, counts):
    """Give every float-cap tercile a seat, by trading free seats only.

    For a tercile the book misses, the best-scored name of that tercile whose industry
    still has room takes the seat of the worst-scored *newly added* name whose own
    tercile keeps a second representative. Retained holdings are never displaced: the
    keep band has already decided them, and unseating one would mean a sell this review
    has closed.
    """
    for wanted in range(CAP_TERCILES):
        book = retained + added
        held = [tercile[code] for code in book]
        if wanted in held:
            continue
        seated = set(book)
        promote = next(
            (code for code in codes
             if code not in seated and tercile[code] == wanted
             and counts.get(industry[code], 0) < INDUSTRY_CAP),
            None,
        )
        if promote is None:
            continue
        demote = next((code for code in reversed(added) if held.count(tercile[code]) > 1), None)
        if demote is None:
            continue
        counts[industry[demote]] -= 1
        counts[industry[promote]] = counts.get(industry[promote], 0) + 1
        added[added.index(demote)] = promote
    return added


def _new_week(context, decision):
    start = (context.inference_at - timedelta(days=10)).strftime("%Y%m%d")
    days = pd.read_parquet(context.asof_dir + "/daily", columns=["trade_date"], filters=[("trade_date", ">=", start)])
    if days.empty:
        return True
    latest = pd.Timestamp(str(days["trade_date"].max()))
    return latest.isocalendar()[:2] != decision.isocalendar()[:2]


def _sell_orders(positions, rank, pool_size, execute_at, candidate):
    """Exit names that left the pool, then the worst-ranked holdings outside the keep
    band -- but only as many as the pool can replace."""
    forced = [code for code in positions if code not in rank]
    held = [code for code in positions if code in rank and rank[code] >= TOP_N * KEEP_BAND]
    held.sort(key=lambda code: (-rank[code], code))
    available = max(0, pool_size - len(positions))
    replaceable = max(0, min(MAX_SWAPS - len(forced), available))
    drop = forced + held[:replaceable]
    return [
        {
            "symbol": code,
            "action": "sell",
            "quantity": int(positions[code]),
            "execute_at": execute_at.isoformat(),
            "reason": candidate + "_exit",
        }
        for code in sorted(drop)
    ]


def _buy_orders(target, keep, prices, budget, execute_at, candidate, extra):
    """Equal-cash top-up to TOP_N names, whole 100-share lots, locally decremented budget."""
    buys = [code for code in target if code not in keep][: max(0, TOP_N - len(keep))]
    orders = []
    remaining = budget
    for index, code in enumerate(buys):
        price = _price(prices, code)
        if price <= 0:
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
                "reason": candidate + "_entry",
                **extra,
            }
        )
        remaining -= quantity * price
    return orders
