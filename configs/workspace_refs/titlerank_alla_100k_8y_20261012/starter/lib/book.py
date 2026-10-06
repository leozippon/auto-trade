"""The event book: the events of a weekly review fill empty seats, best score first, held HOLD trading days.

This is the incdraft lane's book (its 100k execution, no liquidity floor)
with one change: its events come from the records `lib/events.py` wrote at
each review, not from a rule evaluated here, and fresh events fill the seats
in score order. With `knobs.MODEL = "rule"` the records are the incdraft
drafts scored oldest first, and the book trades as the incdraft book does at
the same seats and hold.

Review. The first decision of each ISO week (`lib/clock.py`); no other
decision trades. The book keeps no state of its own: at every review it
re-reads the event records of the reviews in its window, so a cold worker
and a warm one emit the same orders.

Events. Today's events are the units fresh at today's review whose score
cleared the threshold of the model in force (`lib/ranker.py`). They fill the
empty seats highest score first, ties by code. An event that cannot be bought
at its review is skipped for good: no waiting list.

Exits. A holding is sold at the first review at least knobs.HOLD trading days
after the latest review that recorded an event for it -- the review that
bought it, or a later one that recorded it again, which renews the hold. A
holding with no event in the window is past its hold and is sold. Exits are
the only sells. An exit without a fresh T-1 bar (a suspension) is still sent,
but its seat and proceeds are not counted free that day, so a refused exit
never makes room for an extra name.

Entry filters, on the T-1 bar: a fresh, unsuspended bar with a positive close;
main board or ChiNext; no ST / 退 in the replay year's `universe` name; one
100-share lot at the T-1 close fits a seat; at most max(1, floor(SEATS x
INDUSTRY_SHARE)) names per Shenwan L1 industry counting the kept ones.

Sizing. Seat cash is book value / SEATS, the book value counting cash (plus
the day's sell proceeds at a haircut) times CAPITAL and the kept holdings at
the T-1 close (0 without a usable close). A buy is the seat cash in 100-share
lots rounded to the nearest lot and cut to the cash left, since
`context.account` never changes mid-call. Sells go at 09:30 and buys at
15:00. A limit-up close or a suspension is the Broker's to reject; it is
reported, not retried.
"""

from datetime import timedelta

import numpy as np
import pandas as pd

from lib import clock, events, knobs, titles

LOT = 100
CAPITAL = 0.97
SELL_HAIRCUT = 0.98
INDUSTRY_SHARE = 0.2
# Trading days of slack behind a holding's buying review, so the event of any
# holding not yet past its hold is inside the window that is read.
MARGIN_DAYS = 15
# Trading days read back for a holding's last close, which values it while suspended.
PRICE_DAYS = 20
DAILY_COLUMNS = ["ts_code", "trade_date", "close", "is_suspended"]
UNIVERSE_COLUMNS = ["ts_code", "name", "l1_name"]


def run(context):
    leg = knobs.leg()
    decision = clock.decision(context)
    buy_at = decision.replace(hour=15, minute=0, second=0, microsecond=0)
    sell_at = decision.replace(hour=9, minute=30, second=0, microsecond=0)
    if decision > buy_at or not clock.is_review(context):
        return []
    if decision > sell_at:
        sell_at = buy_at

    span = knobs.HOLD + MARGIN_DAYS
    first_day = (decision - timedelta(days=int(1.6 * span) + 14)).strftime("%Y%m%d")
    days = clock.calendar(context, first_day)
    today = len(days) - 1
    if clock.reviews(days)[-1] != today:
        raise RuntimeError(f"{days[-1]} is not the first trading day of its ISO week")
    recorded, scored = _records(context, days)

    positions = {str(code): int(quantity)
                 for code, quantity in dict(context.account.positions).items() if int(quantity) > 0}
    bought = recorded[recorded["entry"] < today].groupby("ts_code")["entry"].max()
    drops = sorted(code for code in positions if code not in bought.index or today - bought[code] >= knobs.HOLD)
    fresh = recorded[(recorded["entry"] == today) & ~recorded["ts_code"].isin(list(positions))]
    fresh = fresh.sort_values(["score", "ts_code"], ascending=[False, True], kind="stable").drop_duplicates("ts_code")

    frame = _pool(context, sorted(set(positions) | set(fresh["ts_code"])), days)
    prices = frame["close"]
    # An exit without a fresh T-1 bar (suspended) is still sent, but neither its
    # seat nor its proceeds are counted free today.
    freed = [code for code in drops if frame.at[code, "sellable"]]
    keep = [code for code in positions if code not in freed]
    proceeds = sum(_price(prices, code) * positions[code] for code in freed)
    budget = (float(context.account.cash) + proceeds * SELL_HAIRCUT) * CAPITAL
    book_value = budget + sum(_price(prices, code) * positions[code] for code in keep)
    seat_cash = book_value / knobs.SEATS
    cap = max(1, int(knobs.SEATS * INDUSTRY_SHARE))
    counts = frame["industry"].reindex(keep).value_counts().to_dict()
    skipped = {"untradable": 0, "unaffordable": 0, "industry": 0, "no_seat": 0}
    buys = []
    for code in fresh["ts_code"]:
        row = frame.loc[code]
        reason = (
            "untradable" if not row["tradable"]
            else "unaffordable" if row["close"] * LOT > seat_cash
            else "industry" if counts.get(row["industry"], 0) >= cap
            else "no_seat" if len(keep) + len(buys) >= knobs.SEATS
            else None
        )
        if reason:
            skipped[reason] += 1
            continue
        buys.append(code)
        counts[row["industry"]] = counts.get(row["industry"], 0) + 1

    meta = {
        "leg": leg,
        "seats": knobs.SEATS,
        "hold": knobs.HOLD,
        "book_size": len(keep) + len(buys),
        "exits": len(drops),
        "exits_unsellable": len(drops) - len(freed),
        "units": scored,
        "fresh": int(len(fresh)),
        **{f"skipped_{key}": value for key, value in skipped.items()},
        "seat_cash": round(seat_cash, 2),
        "last_bar": days[-2],
    }
    sells = [{"symbol": code, "action": "sell", "quantity": int(positions[code]),
              "execute_at": sell_at.isoformat(), "reason": leg + "_exit"} for code in drops]
    return sells + _buy_orders(buys, prices, budget, seat_cash, book_value, buy_at, leg, meta)


def _records(context, days):
    """(ts_code, score, entry) of every event recorded at a review in `days`, entry its
    position there; and the number of units scored today. Today's record must exist."""

    index = np.load(context.state_dir + events.INDEX)
    position = {int(day): i for i, day in enumerate(days)}
    today = int(days[-1])
    if today not in set(index[:, 0].tolist()):
        raise RuntimeError(f"fit recorded no events for the review of {today}")
    frames = []
    for day, scored, _ in index:
        if int(day) in position:
            rows = pd.read_parquet(context.state_dir + events.EVENTS.format(int(day)))
            frames.append(rows.assign(ts_code=rows["ts_code"].astype(str), entry=position[int(day)]))
    scored_today = int(index[index[:, 0] == today][-1, 1])
    return pd.concat(frames, ignore_index=True), scored_today


def _pool(context, codes, days):
    """Frame indexed by `codes`: last close in the PRICE_DAYS to T-1, sellable, tradable, industry."""

    start = days[max(0, len(days) - 1 - PRICE_DAYS)]
    bars = pd.read_parquet(
        context.asof_dir + "/daily",
        columns=DAILY_COLUMNS,
        filters=[("trade_date", ">=", start), ("ts_code", "in", codes or [""])],
    )
    bars = bars.assign(trade_date=bars["trade_date"].astype(str), ts_code=bars["ts_code"].astype(str))
    last = bars.sort_values("trade_date").drop_duplicates("ts_code", keep="last").set_index("ts_code")
    universe = pd.read_parquet(context.asof_dir + "/universe", columns=UNIVERSE_COLUMNS)
    info = universe.assign(ts_code=universe["ts_code"].astype(str)).drop_duplicates("ts_code").set_index("ts_code")
    frame = pd.DataFrame(index=pd.Index(codes, name="ts_code"))
    frame["close"] = pd.to_numeric(last["close"], errors="coerce").reindex(codes)
    frame["industry"] = info["l1_name"].reindex(codes).fillna("未分类").astype(str)
    name = info["name"].reindex(codes).fillna("").astype(str)
    code = frame.index.to_series()
    fresh_bar = last["trade_date"].reindex(codes).eq(days[-2]) & ~last["is_suspended"].reindex(codes).eq(True)
    frame["sellable"] = fresh_bar.fillna(False) & (frame["close"] > 0)
    frame["tradable"] = frame["sellable"] & code.str.match(titles.STOCK) & ~name.str.contains("ST|退")
    return frame


def _price(prices, code):
    price = float(prices.get(code, float("nan")))
    return price if np.isfinite(price) and price > 0 else 0.0


def _buy_orders(buys, prices, budget, seat_cash, book_value, execute_at, leg, meta):
    orders = []
    remaining = budget
    for code in buys:
        price = _price(prices, code)
        quantity = int(min(remaining, seat_cash) / price / LOT + 0.5) * LOT
        while quantity > 0 and quantity * price > remaining:
            quantity -= LOT
        if quantity <= 0:
            continue
        orders.append({
            "symbol": code, "action": "buy", "quantity": quantity,
            "execute_at": execute_at.isoformat(), "reason": leg + "_entry",
            "realized_weight": round(quantity * price / book_value, 6) if book_value > 0 else 0.0,
            **meta,
        })
        remaining -= quantity * price
    return orders
