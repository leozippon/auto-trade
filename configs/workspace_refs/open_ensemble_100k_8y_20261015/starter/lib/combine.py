"""One account, several sleeves: each sleeve's own strategy runs on its own sub-account.

    combine(context, sleeves)  the orders of every sleeve, each run on a view of
                               the context whose account is that sleeve's part
    Sleeve(name, run, holds, share)
        run(context) -> orders   the sleeve's own generate_orders
        holds(context, codes)    the codes among `codes` the sleeve's own rule
                                 says it holds now (never called for the
                                 last sleeve)
        share                    the part of the account's value it is sized on
    holds_b30, holds_esop60    `holds` of the two fixed sleeves (sleeves/)

The book keeps no state, so ownership is read off the rules at every call: a
holding belongs to the first sleeve in order whose `holds` claims it, and
every holding no earlier sleeve claims goes to the LAST sleeve, whose `holds`
is therefore never called. The last sleeve's own rule must sell what it holds
no reason for: b30 does (it sells, at its weekly review, every name without a
draft in its window), which is why main.py lists it last; esop60's past-hold
names then reach it and leave at the same weekly review esop60 would have
sold them at. A name two sleeves claim at once belongs to the first; the two
fixed event sets share about 5 % of their buys over the research period, so
it is rare, but the combined book differs from the sum of its sleeves there.

Cash. The account's value is its cash plus every holding at the T-1 close. A
sleeve gets cash up to its share of that value minus the value of its own
holdings, never below 0; when the sleeves together ask for more than the
account holds, every ask is scaled down alike. Cash a sleeve over its share
does not use stays idle: nothing is ever sold to rebalance, so the shares are
targets the sleeves drift around, reached through their own entries.

Orders. Each sleeve sizes and times its own orders on its view. A buy of a
name another sleeve holds, or one an earlier sleeve buys in the same call, is
dropped (its seat stays empty until that sleeve's next entry); every order is
labelled with its sleeve. Sells come first, in sleeve order, then buys.
"""

from collections import namedtuple
from dataclasses import dataclass
from datetime import timedelta
from typing import Callable

import numpy as np
import pandas as pd

from sleeves.b30 import book as b30_book
from sleeves.b30 import knobs as b30_knobs
from sleeves.b30 import titles as b30_titles
from sleeves.esop60 import score as esop60_score

Account = namedtuple("Account", ["cash", "positions"])
PRICE_LOOKBACK_DAYS = 20


@dataclass(frozen=True)
class Sleeve:
    name: str
    run: Callable
    holds: Callable
    share: float


class View:
    """The context as one sleeve sees it: everything the same but the account."""

    def __init__(self, context, account):
        self._context = context
        self.account = account

    def __getattr__(self, name):
        return getattr(self._context, name)


def combine(context, sleeves):
    if not sleeves:
        raise ValueError("a combined book needs at least one sleeve")
    shares = [float(sleeve.share) for sleeve in sleeves]
    if min(shares) <= 0 or sum(shares) > 1 + 1e-9:
        raise ValueError(f"sleeve shares must be positive and sum to at most 1, got {shares}")
    names = [sleeve.name for sleeve in sleeves]
    if len(set(names)) != len(names):
        raise ValueError(f"sleeve names must be distinct, got {names}")

    positions = {str(code): int(quantity)
                 for code, quantity in dict(context.account.positions).items() if int(quantity) > 0}
    owner = {}
    for sleeve in sleeves[:-1]:
        free = sorted(code for code in positions if code not in owner)
        if not free:
            break
        for code in sorted(set(sleeve.holds(context, free)) & set(free)):
            owner[code] = sleeve.name
    for code in positions:
        owner.setdefault(code, sleeves[-1].name)

    prices = _closes(context, sorted(positions))
    held = {name: {code: quantity for code, quantity in positions.items() if owner[code] == name} for name in names}
    worth = {name: sum(prices.get(code, 0.0) * quantity for code, quantity in held[name].items()) for name in names}
    cash = float(context.account.cash)
    total = cash + sum(worth.values())
    asks = {sleeve.name: max(0.0, sleeve.share * total - worth[sleeve.name]) for sleeve in sleeves}
    scale = min(1.0, cash / sum(asks.values())) if sum(asks.values()) > 0 else 0.0

    sells, buys, taken = [], [], set(positions)
    for sleeve in sleeves:
        view = View(context, Account(cash=asks[sleeve.name] * scale, positions=held[sleeve.name]))
        for order in sleeve.run(view) or []:
            order = {**order, "sleeve": sleeve.name}
            symbol = str(order["symbol"])
            if order["action"] == "sell":
                if symbol not in held[sleeve.name]:
                    raise RuntimeError(f"sleeve {sleeve.name} sells {symbol}, which it does not hold")
                sells.append(order)
            elif symbol not in taken:
                taken.add(symbol)
                buys.append(order)
    return sells + buys


def holds_b30(context, codes):
    """The names the b30 book holds now: those with a draft whose entry review
    (the first weekly review after the draft became visible) is before today,
    inside the window the book reads -- `bought` in sleeves/b30/book.py, on a
    review day and between reviews alike. A name past its hold stays claimed
    until the book's own review sells it."""

    decision = pd.Timestamp(context.inference_at).tz_convert(b30_book.TIMEZONE)
    late = b30_book.LATE_DAYS if b30_knobs.CANDIDATE == "late" else 0
    span = late + b30_knobs.HOLD + b30_book.MARGIN_DAYS
    first_day = (decision - timedelta(days=int(1.6 * span) + 14)).strftime("%Y%m%d")
    quiet_day = (decision - timedelta(days=int(1.6 * span) + 14 + b30_titles.QUIET_DAYS + 1)).strftime("%Y%m%d")
    days = b30_book._calendar(context, first_day, decision)
    today = len(days) - 1
    reviews = b30_book._reviews(days)
    if len(reviews) == 0:
        return set()
    events = b30_titles.events(b30_titles.drafts(context, quiet_day), first_day)
    visible = np.searchsorted(days, events["day"].to_numpy(dtype=object), side="right") + late
    after = np.searchsorted(reviews, visible, side="right")
    entry = np.where(after < len(reviews), reviews[np.minimum(after, len(reviews) - 1)], len(days))
    bought = set(events["ts_code"].to_numpy()[entry < today])
    return bought & {str(code) for code in codes}


def holds_esop60(context, codes):
    """The names the esop60 book holds now: those its own score still scores
    (an ESOP draft entered fewer than HOLD_DAYS trading days ago). A name past
    its window is unclaimed and goes to the last sleeve."""

    values, _meta = esop60_score.score(context, list(codes))
    return set(values.dropna().index)


def _closes(context, codes):
    """{code: T-1 close} for `codes`; a code without a positive close is absent (valued at 0)."""

    if not codes:
        return {}
    start = (context.inference_at - timedelta(days=PRICE_LOOKBACK_DAYS)).strftime("%Y%m%d")
    bars = pd.read_parquet(
        context.asof_dir + "/daily",
        columns=["ts_code", "trade_date", "close"],
        filters=[("trade_date", ">=", start), ("ts_code", "in", codes)],
    )
    bars = bars.assign(trade_date=bars["trade_date"].astype(str), ts_code=bars["ts_code"].astype(str))
    last = bars.sort_values("trade_date").drop_duplicates("ts_code", keep="last")
    close = pd.to_numeric(last["close"], errors="coerce")
    return {code: float(price) for code, price in zip(last["ts_code"], close) if np.isfinite(price) and price > 0}
