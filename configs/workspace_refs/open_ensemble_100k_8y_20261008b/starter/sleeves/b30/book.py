"""The draft book: fresh events fill empty seats at a weekly review and are held HOLD trading days.

Review. The first decision of each ISO week -- the newest visible trading day
(T-1) lies in an earlier week -- and no other decision trades. Everything
below is recomputed from the point-in-time view at every review; the book
keeps no state, so a cold worker and a warm one emit the same orders.

Events (lib/titles.py). An event dated D is counted from its visible day, the
first trading day after D. It is fresh at the first review after its visible
day: at a review, the events whose visible day falls on or after the previous
review and before today. On research titles that is the offline rule's timing
(entry at the close of the second trading day after D at the earliest), and
never before the replay shows the title. Fresh events fill the empty seats
oldest visible day first, ties by code. An event that cannot be bought at its
review is skipped for good: no ranking, no waiting list.

Late-entry control (knobs.CANDIDATE == "late"): every event's visible day is
moved LATE_DAYS trading days later; the book is otherwise the same.

Exits. A holding is sold at the first review at least knobs.HOLD trading days
after the review that bought it. That review is recomputed as the first
review after the visible day of the holding's latest event; a holding with no
event in the window is past its hold and is sold. Exits are the only sells. An
exit without a fresh T-1 bar (a suspension) is still sent, but its seat and
proceeds are not counted free that day, so a refused exit never makes room
for an extra name.

Entry filters, on the T-1 bar: a fresh, unsuspended bar with a positive close;
not STAR (688 / 689) and not Beijing; no ST / 退 in the replay year's
`universe` name; one 100-share lot at the T-1 close fits a seat; at most
max(1, floor(SEATS x INDUSTRY_SHARE)) names per Shenwan L1 industry counting
the kept ones; and at 500k a median daily amount over the last AMOUNT_DAYS
visible trading days of at least knobs.MIN_AMOUNT.

Sizing. Seat cash is book value / SEATS, the book value counting cash (plus
the day's sell proceeds at a haircut) times CAPITAL and the kept holdings at
the T-1 close (0 without a usable close). A buy is the seat cash in 100-share
lots rounded to the nearest lot and cut to the cash left, since
`context.account` never changes mid-call. At 100k sells go at 09:30 and buys
at 15:00; at 500k both at 15:00, sells listed first. A limit-up close or a
suspension is the Broker's to reject; it is reported, not retried.
"""

from datetime import timedelta

import numpy as np
import pandas as pd

from sleeves.b30 import knobs, titles

LOT = 100
CAPITAL = 0.97
SELL_HAIRCUT = 0.98
LATE_DAYS = 120
INDUSTRY_SHARE = 0.2
AMOUNT_DAYS = 20
# Trading days of slack behind a holding's buying review, so the event of any
# holding not yet past its hold is inside the window that is read.
MARGIN_DAYS = 15
TIMEZONE = "Asia/Shanghai"
DAILY_COLUMNS = ["ts_code", "trade_date", "close", "amount", "is_suspended"]
UNIVERSE_COLUMNS = ["ts_code", "name", "l1_name"]


def run(context):
    leg = knobs.leg()
    decision = pd.Timestamp(context.inference_at).tz_convert(TIMEZONE)
    buy_at = decision.replace(hour=15, minute=0, second=0, microsecond=0)
    sell_at = buy_at if knobs.EXECUTION[knobs.ACCOUNT] == "close" else decision.replace(
        hour=9, minute=30, second=0, microsecond=0)
    if decision > buy_at or not _new_week(context, decision):
        return []
    if decision > sell_at:
        sell_at = buy_at

    late = LATE_DAYS if knobs.CANDIDATE == "late" else 0
    span = late + knobs.HOLD + MARGIN_DAYS
    first_day = (decision - timedelta(days=int(1.6 * span) + 14)).strftime("%Y%m%d")
    quiet_day = (decision - timedelta(days=int(1.6 * span) + 14 + titles.QUIET_DAYS + 1)).strftime("%Y%m%d")
    days = _calendar(context, first_day, decision)
    today = len(days) - 1
    reviews = _reviews(days)
    if reviews[-1] != today:
        raise RuntimeError(f"{days[-1]} is not the first trading day of its ISO week")

    events = titles.events(titles.drafts(context, quiet_day), first_day)
    visible = np.searchsorted(days, events["day"].to_numpy(dtype=object), side="right") + late
    events = events.assign(visible=visible)[visible < today]
    events["entry"] = reviews[np.searchsorted(reviews, events["visible"].to_numpy(), side="right")]

    positions = {str(code): int(quantity)
                 for code, quantity in dict(context.account.positions).items() if int(quantity) > 0}
    bought = events[events["entry"] < today].groupby("ts_code")["entry"].max()
    drops = sorted(code for code in positions if code not in bought.index or today - bought[code] >= knobs.HOLD)
    fresh = events[(events["entry"] == today) & ~events["ts_code"].isin(list(positions))]
    fresh = fresh.sort_values(["visible", "ts_code"], kind="stable").drop_duplicates("ts_code")

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
    skipped = {"untradable": 0, "unaffordable": 0, "illiquid": 0, "industry": 0, "no_seat": 0}
    buys = []
    for code in fresh["ts_code"]:
        row = frame.loc[code]
        reason = (
            "untradable" if not row["tradable"]
            else "unaffordable" if row["close"] * LOT > seat_cash
            else "illiquid" if not row["amount"] >= knobs.MIN_AMOUNT[knobs.ACCOUNT]
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
        "account": knobs.ACCOUNT,
        "seats": knobs.SEATS,
        "hold": knobs.HOLD,
        "book_size": len(keep) + len(buys),
        "exits": len(drops),
        "exits_unsellable": len(drops) - len(freed),
        "fresh": int(len(fresh)),
        **{f"skipped_{key}": value for key, value in skipped.items()},
        "seat_cash": round(seat_cash, 2),
        "last_bar": days[-2],
    }
    sells = [{"symbol": code, "action": "sell", "quantity": int(positions[code]),
              "execute_at": sell_at.isoformat(), "reason": leg + "_exit"} for code in drops]
    return sells + _buy_orders(buys, prices, budget, seat_cash, book_value, buy_at, leg, meta)


def _new_week(context, decision):
    start = (decision - timedelta(days=12)).strftime("%Y%m%d")
    days = pd.read_parquet(context.asof_dir + "/daily", columns=["trade_date"], filters=[("trade_date", ">=", start)])
    if days.empty:
        return True
    latest = pd.Timestamp(str(days["trade_date"].max()))
    return tuple(latest.isocalendar())[:2] != tuple(decision.isocalendar())[:2]


def _calendar(context, start, decision):
    """Visible trading days from `start` (YYYYMMDD), plus the decision day, as a sorted array."""

    rows = pd.read_parquet(context.asof_dir + "/daily", columns=["trade_date"], filters=[("trade_date", ">=", start)])
    return np.array(sorted({*rows["trade_date"].astype(str), decision.strftime("%Y%m%d")}), dtype=object)


def _reviews(days):
    """Positions in `days` of the first trading day of each ISO week (the first day itself excluded)."""

    weeks = [tuple(pd.Timestamp(day).isocalendar())[:2] for day in days]
    return np.array([i for i in range(1, len(days)) if weeks[i] != weeks[i - 1]], dtype=int)


def _pool(context, codes, days):
    """Frame indexed by `codes`: T-1 close, tradable, median amount, industry."""

    start = days[max(0, len(days) - 1 - AMOUNT_DAYS)]
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
    frame["amount"] = bars.groupby("ts_code")["amount"].median().reindex(codes)
    frame["industry"] = info["l1_name"].reindex(codes).fillna("未分类").astype(str)
    name = info["name"].reindex(codes).fillna("").astype(str)
    code = frame.index.to_series()
    fresh_bar = last["trade_date"].reindex(codes).eq(days[-2]) & ~last["is_suspended"].reindex(codes).eq(True)
    frame["sellable"] = fresh_bar.fillna(False) & (frame["close"] > 0)
    frame["tradable"] = (
        frame["sellable"] & ~code.str.startswith(("688", "689")) & ~code.str.endswith(".BJ")
        & ~name.str.contains("ST|退")
    )
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
