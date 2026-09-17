"""The sibling pack's bucketed top-15 book, behind three exclusion screens.

Construction, in this order at every review:

1. pool floor -- keep the larger CAP_FLOOR fraction of the eligible pool by
   float market cap;
2. accrual screen -- drop the ACCR_SCREEN worst fraction of the survivors by
   accruals (`accr` highest first: earnings the cash flow does not back);
3. **exclusion screens** -- drop the names `lib.screens` flags for this leg:
   a non-standard audit opinion, an imminent unlock of restricted shares, a
   revenue base concentrated in one product segment;
4. rank the survivors by the leg's neutralized score;
5. fill the book to TOP_N off that ranking, taking a name only while its SW L1
   industry holds fewer than INDUSTRY_CAP of the book, then making sure each
   float-cap tercile of the screened pool is represented. Equal cash, whole
   100-share lots.

Everything except step 3 is `lowvol_bucketed_20260917`, byte for byte, which is
why `c_noscreen` -- this book with no screen at all -- IS that pack's main
candidate `s_lb`, and why the two arms cross-check each other. Step 3 is the
whole of what this arm registers.

The screens sit between the accrual screen and the ranking because what they
change is POOL MEMBERSHIP. A name the screens removed is out of the book, out of
the keep band and out of the float-cap terciles, exactly as a name that fell
below the pool floor is; nothing downstream knows a screen was the reason. That
also makes the terciles a property of the pool the book really chooses from.

The buckets are counted against the book the review will actually hold, the
retained holdings included, never against a freshly ranked top 15. That is what
makes the cap hold: the seats the keep band carries over came from a book that
already respected the cap, and the free seats are filled under a counter seeded
with them, so no review can push an industry past INDUSTRY_CAP. Capping a
freshly ranked top 15 instead would leak straight through the keep band, which
retains holdings ranked behind that 15 and never counts their industries.

`SCREENS` says which screens each pre-registered leg runs and `SHAPES` which of
steps 1, 2 and 5 it runs; the score itself comes from `lib.face`.

Cadence: the book is reviewed on the first decision day of each calendar
quarter and at the replay's first decision. The rule is stateless -- it
compares the latest visible trading day's quarter with the decision day's -- so
a cold worker and a warm one produce identical orders. The screens therefore
read their three panels four times a year, not every day.

Sells are timed at 09:30 and buys at 15:00 of the same day: the Broker
processes pending orders in timestamp order, so the proceeds are credited
before the buys are sized. `context.account` is the pre-call snapshot and never
changes mid-call, so the buy budget is decremented locally.

Keep band: a holding still ranked inside TOP_N * KEEP_BAND of the screened pool
is not sold. A holding is sold when it left that pool (halted, price cap, ADV
floor, stale statement, no volatility window, no longer in the larger half by
float cap, fallen into the screened accrual tail, or newly flagged by a screen)
or fell outside the band. At most MAX_REPLACE discretionary swaps per review.
"""

import numpy as np
import pandas as pd

from lib import face, screens

TOP_N = face.BOOK_SIZE
MAX_REPLACE = 8              # at most this many discretionary swaps per review
KEEP_BAND = 2.0              # a holding ranked inside TOP_N * KEEP_BAND is not sold
CASH_BUFFER = 0.97
SELL_HAIRCUT = 0.98

CAP_FLOOR = 0.5              # keep this fraction of the pool, largest float cap first
ACCR_SCREEN = 0.1            # drop this fraction of the survivors, highest accruals first
INDUSTRY_CAP = 2             # at most this many names of one SW L1 industry in the book
CAP_TERCILES = 3             # float-cap terciles of the screened pool; the book holds >= 1 of each

# leg -> which exclusion screens run. This is the arm.
SCREENS = {
    "s_scr": ("audit", "unlock", "segment"),
    "c_noscreen": (),
    "c_placebo": ("placebo",),
    "c_pool": (),
}

# leg -> (pool floor, accrual screen, buckets)
SHAPES = {
    "s_scr": (True, True, True),
    "c_noscreen": (True, True, True),
    "c_placebo": (True, True, True),
    "c_pool": (False, False, False),
}


def run(context, candidate):
    decision = pd.Timestamp(context.inference_at)
    sell_at = decision.replace(hour=9, minute=30, second=0, microsecond=0)
    buy_at = decision.replace(hour=15, minute=0, second=0, microsecond=0)
    if decision > buy_at:
        return []
    if decision > sell_at:
        sell_at = buy_at
    positions = {str(k): int(v) for k, v in dict(context.account.positions).items() if int(v) > 0}
    section, latest_day, vol_basis = face.build_cross_section(context, candidate)
    if section is None:
        return []
    if not _is_review(decision, latest_day, positions):
        return []
    section["score"] = face.neutralize(section, candidate)
    ranked, screen_hits = _shape(context, section, candidate)
    rank = {code: index for index, code in enumerate(ranked["ts_code"])}
    prices = ranked.set_index("ts_code")["close"]
    sells = _sell_orders(positions, rank, len(ranked), sell_at, candidate)
    proceeds = sum(float(prices.get(order["symbol"], 0.0)) * order["quantity"] for order in sells)
    keep = [code for code in positions if code not in {order["symbol"] for order in sells}]
    target, cover = _book(ranked, keep, candidate)
    budget = (float(context.account.cash) + proceeds * SELL_HAIRCUT) * CASH_BUFFER
    return sells + _buy_orders(target, keep, prices, budget, buy_at, candidate, vol_basis, cover, screen_hits)


def _shape(context, section, candidate):
    """The pool floor, the accrual screen and the exclusion screens, then the ranking.

    Returns the ranked screened pool and the per-review screen reading that every
    buy order carries: the size of the pool the screens acted on, and how many
    names each of them removed from it. The nomination gate reads it, because a
    screen that removes a large slice of the pool is no longer a tail screen.

    `cap_tercile` is attached here, on the screened pool and before the ranking,
    because the terciles are a property of that pool and not of the day's score.
    """
    floored, screened, _bucketed = SHAPES[candidate]
    if floored:
        keep = int(len(section) * CAP_FLOOR)
        section = section.nlargest(max(keep, TOP_N), "circ_mv")
    if screened:
        keep = len(section) - int(len(section) * ACCR_SCREEN)
        section = section.nsmallest(max(keep, TOP_N), "accr")
    hits = {}
    reading = f"pool={len(section)}"   # the pool the screens act on, so the gate's fractions read off this line
    if SCREENS[candidate]:
        drop, hits = screens.excluded(context, section, SCREENS[candidate])
        section = section[~section["ts_code"].isin(drop)]
        reading = " ".join([reading] + [f"{name}={count}" for name, count in hits.items()])
    section = section.sort_values("ts_code").reset_index(drop=True)
    section["cap_tercile"] = _terciles(section["circ_mv"].to_numpy(dtype="float64"))
    ranked = section.sort_values(["score", "ts_code"], ascending=[False, True]).reset_index(drop=True)
    return ranked, reading


def _terciles(circ_mv):
    """Float-cap terciles of the screened pool: 0 is the smallest third.

    Split by position in the sorted pool rather than by a value quantile, so
    duplicate float caps cannot collapse an edge and the three groups always
    exist. The pool is ordered by `ts_code` here, which makes the tie-break
    stable across workers.
    """
    size = len(circ_mv)
    tercile = np.zeros(size, dtype="int64")
    tercile[np.argsort(circ_mv, kind="stable")] = np.minimum(
        np.arange(size) * CAP_TERCILES // max(size, 1), CAP_TERCILES - 1
    )
    return tercile


def _book(ranked, keep, candidate):
    """The TOP_N names this review holds, and how many terciles they cover.

    The retained holdings take their seats first and seed the industry counter;
    the free seats go to the best-scored names their industry still has room
    for. The cover is reported rather than enforced: a tercile with no name the
    cap can take is a property of that day's cross-section, and every buy order
    carries the reading, so a shortfall shows up in the run instead of being
    absorbed in silence.
    """
    bucketed = SHAPES[candidate][2]
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
    return book, len({tercile[code] for code in book})


def _cover_terciles(retained, added, codes, industry, tercile, counts):
    """Give every float-cap tercile a seat, by trading free seats only.

    For a tercile the book misses, the best-scored name of that tercile whose
    industry still has room takes the seat of the worst-scored *newly added*
    name whose own tercile keeps a second representative. Retained holdings are
    never displaced: the keep band has already decided them, and unseating one
    would mean a sell this review has closed.
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


def _is_review(decision, latest_day, positions):
    """First decision of a calendar quarter, or the first decision of the replay."""
    if not positions:
        return True
    latest = pd.Timestamp(latest_day)
    return (latest.year, (latest.month - 1) // 3) != (decision.year, (decision.month - 1) // 3)


def _sell_orders(positions, rank, pool_size, execute_at, candidate):
    """Exit names that left the screened pool, then the worst-ranked holdings
    outside the keep band -- but only as many as the pool can replace."""
    forced = [code for code in positions if code not in rank]
    held = [code for code in positions if code in rank and rank[code] >= TOP_N * KEEP_BAND]
    held.sort(key=lambda code: (-rank[code], code))
    available = max(0, pool_size - len(positions))
    replaceable = max(0, min(MAX_REPLACE - len(forced), available))
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


def _buy_orders(target, keep, prices, budget, execute_at, candidate, vol_basis, cover, screen_hits):
    """Equal-cash top-up to TOP_N names, whole 100-share lots, locally decremented budget."""
    buys = [code for code in target if code not in keep][: max(0, TOP_N - len(keep))]
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
                "reason": candidate + "_entry",
                "vol_basis": vol_basis,
                "tercile_cover": cover,
                "screen_hits": screen_hits,
            }
        )
        remaining -= quantity * price
    return orders
