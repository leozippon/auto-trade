"""Real-fill reconciliation: a Paper book that follows what its owner traded.

A book is traded by hand from its order sheet. Once the owner switches it to
real fills (an ``enable`` record), every later session settles from the
owner's record of what each order actually did, so the account the strategy is
shown the next morning -- and with it the next order sheet -- is the real one.
A book that was never switched keeps settling from the simulator exactly as
before: none of this runs and none of its files exist.

The owner's records live in ``fills.jsonl``, append-only; the latest record for
a key wins. A key is one session's ``(time, side, symbol)``: the orders a
person places as one, so two strategy orders for the same name, side and
minute are recorded together. A record says the key filled as simulated,
filled at a stated price and share count (and, optionally, commission), or did
not trade; a key with no order behind it is a trade the sheet did not ask for.
A ``session`` record sets the outcome of every order of one session that has
no record of its own -- the one-click "as simulated" or "nothing traded".

Every real-fill session is settled twice from the same morning account: a copy
of it fills each due order as the simulator would (the simulated track,
``simulated_executions_<day>.jsonl``), and the account itself takes the
recorded outcomes. Both are marked at the close, so the equity row carries the
two values and their gap is what that day's manual execution cost. A session
whose orders have no recorded outcome is not settled at all: the run stops and
says so, rather than assume the orders filled. Orders of a session the book
decided only after that session's day (a back-fill) could not have been
traded, and read as not traded unless recorded otherwise.

A record that changes a session already settled is a correction. History is
not rewritten: the next settlement books the difference -- shares, cash, cost
basis and realized P&L at the recorded prices -- on the session it settles, in
``corrections.jsonl``. A correction that cannot be booked exactly (a share
change across an ex-date of that name, shares the book no longer holds, or a
changed sale quantity on a book that taxes dividends) stops the run with the
reason instead.
"""

from __future__ import annotations

import copy
import math
import re
import uuid
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import date, datetime, time
from pathlib import Path

from autotrade.environment.broker import DailyBroker, Execution, Lot, Position
from autotrade.environment.broker_core import validate_buy_lot
from autotrade.environment.strategy import CN_TZ, StrategyOrder

from .storage import append_jsonl_once, read_jsonl

FILLS_NAME = "fills.jsonl"
CORRECTIONS_NAME = "corrections.jsonl"
SIMULATED_EXECUTIONS_PREFIX = "simulated_executions_"
# The ``.paper_state.json`` key of a book in real-fill mode; absent otherwise.
STATE_KEY = "real_fills"
SIMULATED = "simulated"
FILLED = "filled"
NOT_TRADED = "not_traded"
OUTCOMES = (SIMULATED, FILLED, NOT_TRADED)
# Execution status of an order the owner recorded as not traded.
UNFILLED = "unfilled"
SYMBOL_PATTERN = re.compile(r"\d{6}\.(SH|SZ|BJ)")
CLOCK_PATTERN = re.compile(r"([01]\d|2[0-3]):[0-5]\d")
DATE_PATTERN = re.compile(r"\d{8}")
# Half a cent: two money figures closer than this are the same figure.
CENT = 0.005


@dataclass(frozen=True)
class Record:
    """One recorded outcome, as the latest line for its key says."""

    event_id: str
    outcome: str
    quantity: int = 0
    price: float | None = None
    commission: float | None = None
    recorded_at: str = ""


# The outcome of an order of a back-filled session: nobody held its sheet.
POST_HOC = Record("", NOT_TRADED)


def key_text(clock: str, action: str, symbol: str) -> str:
    return f"{clock} {action} {symbol}"


def parse_key(key: str) -> tuple[str, str, str]:
    clock, action, symbol = key.split(" ")
    return clock, action, symbol


def order_key(order: StrategyOrder) -> str:
    return key_text(order.execute_at.astimezone(CN_TZ).strftime("%H:%M"), order.action, order.symbol)


def order_session(order: StrategyOrder) -> str:
    return order.execute_at.astimezone(CN_TZ).strftime("%Y%m%d")


# ------------------------------------------------------------------ records


def normalize_record(raw: Mapping[str, object]) -> dict[str, object]:
    """One record's fields, checked; ``ValueError`` names what is wrong.

    The same check guards a record on its way into the file and on its way
    out, so a hand-edited line is refused exactly like a bad request.
    """

    kind = raw.get("kind")
    if kind == "enable":
        return {"kind": "enable"}
    day = raw.get("trade_date")
    if not isinstance(day, str) or not DATE_PATTERN.fullmatch(day):
        raise ValueError("trade_date must be YYYYMMDD")
    outcome = raw.get("outcome")
    if kind == "session":
        if outcome not in (SIMULATED, NOT_TRADED):
            raise ValueError("a session record is 'simulated' or 'not_traded'")
        return {"kind": "session", "trade_date": day, "outcome": outcome}
    if kind != "fill":
        raise ValueError(f"unknown record kind {kind!r}")
    clock, action, symbol = raw.get("time"), raw.get("action"), raw.get("symbol")
    if not isinstance(clock, str) or not CLOCK_PATTERN.fullmatch(clock):
        raise ValueError("time must be HH:MM")
    if action not in ("buy", "sell"):
        raise ValueError("action must be buy or sell")
    if not isinstance(symbol, str) or not SYMBOL_PATTERN.fullmatch(symbol):
        raise ValueError(f"symbol {symbol!r} is not a code like 000001.SZ")
    record: dict[str, object] = {
        "kind": "fill", "trade_date": day, "time": clock, "symbol": symbol, "action": action, "outcome": outcome,
    }
    if outcome in (SIMULATED, NOT_TRADED):
        return record
    if outcome != FILLED:
        raise ValueError(f"outcome must be one of {', '.join(OUTCOMES)}")
    quantity, price, commission = raw.get("quantity"), raw.get("price"), raw.get("commission")
    if isinstance(quantity, bool) or not isinstance(quantity, int) or quantity <= 0:
        raise ValueError("a fill's quantity must be a positive whole number of shares")
    if action == "buy":
        validate_buy_lot(quantity, symbol)  # the exchange accepts no other buy
    if not _positive(price):
        raise ValueError("a fill's price must be a positive number")
    if commission is not None and not (_finite(commission) and float(commission) >= 0):
        raise ValueError("commission must be a non-negative number")
    return {
        **record, "quantity": quantity, "price": float(price),
        "commission": None if commission is None else float(commission),
    }


@dataclass(frozen=True)
class Records:
    """Everything ``fills.jsonl`` says, latest line first."""

    enabled_at: str | None
    sessions: Mapping[str, Record]
    fills: Mapping[tuple[str, str], Record]
    # Lines the reader could not parse at all: a write the console never
    # confirmed (it answers only after the append is on disk), counted.
    skipped: int

    def keys(self, day: str) -> list[str]:
        return [key for session, key in self.fills if session == day]


def read_records(root: Path) -> Records:
    rows, skipped = read_jsonl(Path(root) / FILLS_NAME)
    enabled_at = None
    sessions: dict[str, Record] = {}
    fills: dict[tuple[str, str], Record] = {}
    for line, row in enumerate(rows, start=1):
        try:
            record = normalize_record(row)
        except ValueError as exc:
            raise ValueError(f"{FILLS_NAME} record {line} is invalid: {exc}") from exc
        event_id, stamp = str(row.get("event_id") or ""), str(row.get("recorded_at") or "")
        if record["kind"] == "enable":
            enabled_at = enabled_at or stamp
        elif record["kind"] == "session":
            sessions[str(record["trade_date"])] = Record(event_id, str(record["outcome"]), recorded_at=stamp)
        else:
            key = key_text(str(record["time"]), str(record["action"]), str(record["symbol"]))
            fills[(str(record["trade_date"]), key)] = Record(
                event_id, str(record["outcome"]), int(record.get("quantity") or 0),
                record.get("price"), record.get("commission"), stamp,
            )
    return Records(enabled_at, sessions, fills, skipped)


def append_records(root: Path, records: Iterable[Mapping[str, object]]) -> list[dict[str, object]]:
    """Validate every record, then append them in order; the stamped lines."""

    now = datetime.now(CN_TZ).isoformat(timespec="seconds")
    lines = [
        {"event_id": f"fill_{uuid.uuid4().hex}", "recorded_at": now, **normalize_record(record)}
        for record in records
    ]
    for line in lines:
        append_jsonl_once(Path(root) / FILLS_NAME, line)
    return lines


def enable(root: Path) -> bool:
    """Ask for real fills; the next run switches the book. False if asked before."""

    if read_records(root).enabled_at is not None:
        return False
    append_records(root, [{"kind": "enable"}])
    return True


def record(
    root: Path,
    state: Mapping[str, object],
    trade_date: str,
    *,
    outcome: str | None = None,
    fills: Sequence[Mapping[str, object]] = (),
    today: str,
) -> list[dict[str, object]]:
    """The owner's outcomes for one session, checked against the book and
    appended: a session-wide ``outcome``, per-key ``fills``, or both.

    The session must be one the book decided or has orders for, after the
    switch, and not in the future; "as simulated" needs an order behind it."""

    records = read_records(root)
    current = mode(state)
    if current is None and records.enabled_at is None:
        raise ValueError("this book settles simulated fills; switch it to real fills first")
    after = str(current["after"]) if current is not None else str(state.get("settled_through") or "")
    if not isinstance(trade_date, str) or not DATE_PATTERN.fullmatch(trade_date):
        raise ValueError("trade_date must be YYYYMMDD")
    if trade_date <= after:
        raise ValueError(f"{trade_date} settled before the book followed real fills")
    if trade_date > today:
        raise ValueError(f"{trade_date} has not traded yet")
    ordered = {
        order_key(order): order
        for order in pending_orders(state)
        if order_session(order) == trade_date
    }
    settled = current["sessions"].get(trade_date) if current is not None else None
    ordered_keys = set(ordered) | {key for key, entry in (settled or {}).items() if entry["ordered"]}
    decided = {str(row.get("trade_date")) for row in state.get("decisions") or () if isinstance(row, Mapping)}
    if not ordered and settled is None and trade_date not in decided:
        raise ValueError(f"the book has no session {trade_date}: record trades on a session it decided")
    lines: list[dict[str, object]] = []
    if outcome is not None:
        lines.append({"kind": "session", "trade_date": trade_date, "outcome": outcome})
    for item in fills:
        line = normalize_record({**item, "kind": "fill", "trade_date": trade_date})
        key = key_text(str(line["time"]), str(line["action"]), str(line["symbol"]))
        if line["outcome"] == SIMULATED and key not in ordered_keys:
            raise ValueError(f"{trade_date} {key}: no order to have filled as simulated")
        lines.append(line)
    if not lines:
        raise ValueError("nothing to record")
    return append_records(root, lines)


def pending_orders(state: Mapping[str, object]) -> list[StrategyOrder]:
    orders = []
    for item in state.get("pending_orders") or ():
        execute_at = datetime.fromisoformat(str(item["execute_at"]))
        orders.append(StrategyOrder.from_record(item, inference_at=execute_at))
    return orders


def compare(key: str, entry: Mapping[str, object]) -> dict[str, object]:
    """One settled key's simulated fill against the recorded one. Costs are
    signed so that positive is what manual execution cost: a buy dearer or a
    sale cheaper than simulated, higher fees."""

    _clock, action, _symbol = parse_key(key)
    simulated, real = entry["simulated"], entry["real"]
    sign = 1.0 if action == "buy" else -1.0
    priced = simulated["price"] is not None and real["price"] is not None
    gap = sign * (float(real["price"]) - float(simulated["price"])) if priced else None
    sim_q, real_q = int(simulated["quantity"]), int(real["quantity"])
    if not entry["ordered"]:
        status = "off_sheet"
    elif sim_q == real_q:
        status = "filled" if real_q else "none"
    elif real_q == 0:
        status = "missed"
    else:
        status = "partial" if real_q < sim_q else "extra"
    return {
        "status": status,
        "price_bp": gap / float(simulated["price"]) * 1e4 if priced else None,
        "price_cost": gap * real_q if priced else None,
        "fee_cost": (float(real["commission"]) + float(real["stamp_duty"]))
        - (float(simulated["commission"]) + float(simulated["stamp_duty"])),
        "missed_quantity": max(sim_q - real_q, 0),
        "missed_notional": max(sim_q - real_q, 0) * float(simulated["price"]) if simulated["price"] else 0.0,
    }


def resolve(records: Records, day: str, key: str, *, ordered: bool, post_hoc: bool) -> Record | None:
    """The outcome a key settles with, or None while nothing says.

    A key's own record wins; an ordered key without one takes the session's
    record, and a back-filled session's orders read as not traded."""

    record = records.fills.get((day, key))
    if record is not None or not ordered:
        return record
    session = records.sessions.get(day)
    if session is not None:
        return session
    return POST_HOC if post_hoc else None


def post_hoc_sessions(state: Mapping[str, object]) -> set[str]:
    """Sessions the book decided only after their own calendar day."""

    late = set()
    for row in state.get("decisions") or ():
        stamp = row.get("decided_at") if isinstance(row, Mapping) else None
        if isinstance(stamp, str) and stamp:
            day = str(row.get("trade_date"))
            if datetime.fromisoformat(stamp).astimezone(CN_TZ).strftime("%Y%m%d") > day:
                late.add(day)
    return late


def mode(state: Mapping[str, object]) -> dict[str, object] | None:
    value = state.get(STATE_KEY)
    return value if isinstance(value, dict) else None


def adopt(state: dict[str, object], records: Records) -> bool:
    """Switch the book to real fills on its first run after the owner asked.

    Every session after the last one settled so far settles from records,
    so the switch never reaches back into a session already settled."""

    if records.enabled_at is None or mode(state) is not None:
        return False
    state[STATE_KEY] = {
        "enabled_at": records.enabled_at,
        "after": str(state.get("settled_through") or ""),
        # Per settled real-fill session: per key, the ordered shares and the
        # simulated and recorded outcomes, which corrections are measured on.
        "sessions": {},
    }
    return True


def unresolved(
    state: Mapping[str, object], records: Records, orders: Iterable[StrategyOrder], *, before: str
) -> dict[str, int]:
    """Per real-fill session before ``before``: its orders without an outcome."""

    current = mode(state)
    if current is None:
        return {}
    after = str(current["after"])
    late = post_hoc_sessions(state)
    missing: dict[str, int] = {}
    for order in orders:
        day = order_session(order)
        if after < day < before and resolve(
            records, day, order_key(order), ordered=True, post_hoc=day in late
        ) is None:
            missing[day] = missing.get(day, 0) + 1
    return missing


# --------------------------------------------------------------- settlement


def _finite(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(float(value))


def _positive(value: object) -> bool:
    return _finite(value) and float(value) > 0


def record_fill(
    broker: DailyBroker, order: StrategyOrder, price: float, commission: float | None
) -> Execution:
    """Book one fill that happened at ``price`` onto ``broker``.

    It goes through the Broker's own ``execute``, so cash, cost basis, T+1,
    lots, stamp duty, dividend tax and realized P&L are booked exactly as a
    simulated fill books them. Only what the market already decided is taken
    out: no slippage on a known price, no price-limit or holding caps on a
    trade that took place, and a stated commission in place of the model's
    (a minimum of exactly that amount and no rate). The account's own rules
    still hold, so a fill the account could not have made is refused.
    """

    base = broker.profile
    fee = {} if commission is None else {
        "commission_bps": 0.0, "min_commission_cny": float(commission), "transfer_fee_bps": 0.0,
    }
    broker.profile = replace(base, slippage_bps=0.0, max_total_holdings=None, max_single_name_weight=None, **fee)
    try:
        bar = {"up_limit": price * 2.0, "down_limit": price / 2.0, "is_suspended": False}
        execution = broker.execute(order, bar, matched_at=order.execute_at, raw_price=price)
    finally:
        broker.profile = base
    if execution.status != "filled":
        raise ValueError(
            f"{order_session(order)} {order_key(order)}: the recorded fill of {order.quantity} shares at "
            f"{price:.2f} cannot be booked ({execution.reason}); correct the record"
        )
    return execution


def _outcome(executions: Sequence[Execution]) -> dict[str, object]:
    filled = [row for row in executions if row.status == "filled"]
    quantity = sum(row.quantity for row in filled)
    notional = sum(float(row.price) * row.quantity for row in filled)
    realized = [row.realized_pnl for row in filled if row.realized_pnl is not None]
    return {
        "quantity": quantity,
        "price": notional / quantity if quantity else None,
        "notional": notional,
        "commission": sum(row.commission for row in filled),
        "stamp_duty": sum(row.stamp_duty for row in filled),
        "dividend_tax": sum(row.dividend_tax for row in filled),
        "realized_pnl": sum(realized) if realized else None,
        "reason": None if quantity else next((row.reason for row in executions if row.reason), None),
    }


_NOTHING = _outcome(())


class RealFills:
    """One run's view of a real-fill book: its records and its mode state."""

    def __init__(self, root: Path, state: dict[str, object], records: Records) -> None:
        self.root = Path(root)
        self.state = state
        self.records = records

    @property
    def mode(self) -> dict[str, object]:
        current = mode(self.state)
        assert current is not None
        return current

    def covers(self, day: str) -> bool:
        return day > str(self.mode["after"])

    def settle(
        self,
        broker: DailyBroker,
        day: str,
        simulated: Sequence[tuple[StrategyOrder, Execution]],
    ) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
        """Book ``day`` onto the account: first the corrections of earlier
        sessions, then the day's recorded outcomes, in placement order. The
        journal rows of both; the session joins the ledger."""

        corrections = self._corrections(broker, day)
        rows, ledger = self._apply(broker, day, simulated)
        self.mode["sessions"][day] = ledger
        return rows, corrections

    # ---- the day's own fills

    def _apply(
        self, broker: DailyBroker, day: str, simulated: Sequence[tuple[StrategyOrder, Execution]]
    ) -> tuple[list[dict[str, object]], dict[str, dict[str, object]]]:
        by_key: dict[str, list[tuple[StrategyOrder, Execution]]] = {}
        for order, execution in simulated:
            by_key.setdefault(order_key(order), []).append((order, execution))
        # A trade off the sheet sells before it buys within its minute, the
        # order a person's cash allows.
        off_sheet = sorted(
            (key for key in self.records.keys(day) if key not in by_key),
            key=lambda key: (parse_key(key)[0], parse_key(key)[1] != "sell"),
        )
        late = post_hoc_sessions(self.state)
        real: dict[str, list[Execution]] = {key: [] for key in [*by_key, *off_sheet]}
        rows: list[dict[str, object]] = []
        settled: set[str] = set()
        clocks = sorted({parse_key(key)[0] for key in real})
        for clock in clocks:
            for order, execution in simulated:
                key = order_key(order)
                if parse_key(key)[0] != clock or key in settled:
                    continue
                # Recorded under the session the order names, as the sheet
                # and the console show it.
                session = order_session(order)
                record = resolve(self.records, session, key, ordered=True, post_hoc=session in late)
                if record is None:
                    raise ValueError(f"{session} {key}: the order has no recorded outcome")
                if record.outcome == SIMULATED:
                    if execution.status == "filled":
                        execution = record_fill(broker, order, float(execution.price), execution.commission)
                    real[key].append(execution)
                    rows.append({"kind": "execution", **execution.to_record()})
                    continue
                settled.add(key)
                first = by_key[key][0][0]
                ordered = sum(item.quantity for item, _ in by_key[key])
                rows.append(self._recorded(broker, real[key], first, ordered, record))
            for key in off_sheet:
                record = self.records.fills[(day, key)]
                if parse_key(key)[0] != clock or record.outcome != FILLED:
                    continue  # a trade recorded and then withdrawn
                _clock, action, symbol = parse_key(key)
                execute_at = datetime.combine(_date(day), time.fromisoformat(clock), tzinfo=CN_TZ)
                order = StrategyOrder(symbol, action, record.quantity, execute_at)
                rows.append({**self._recorded(broker, real[key], order, 0, record), "off_sheet": True})
            if broker.cash < -1e-6:
                raise ValueError(
                    f"{day} {clock}: the recorded fills leave the book's cash at {broker.cash:,.2f}; "
                    "correct the records"
                )
        ledger = {
            key: {
                "ordered": sum(order.quantity for order, _ in by_key.get(key, ())),
                "simulated": _outcome([execution for _, execution in by_key.get(key, ())]),
                "real": _outcome(real[key]),
            }
            for key in real
        }
        return rows, ledger

    @staticmethod
    def _recorded(
        broker: DailyBroker, real: list[Execution], order: StrategyOrder, ordered: int, record: Record
    ) -> dict[str, object]:
        if record.outcome == FILLED:
            fill = StrategyOrder(order.symbol, order.action, record.quantity, order.execute_at, order.metadata)
            execution = record_fill(broker, fill, float(record.price), record.commission)
        else:
            execution = Execution(
                symbol=order.symbol, action=order.action, quantity=ordered,
                execute_at=order.execute_at.isoformat(), matched_at=order.execute_at.isoformat(),
                status=UNFILLED, reason=NOT_TRADED, metadata=order.metadata,
            )
        real.append(execution)
        return {"kind": "execution", **execution.to_record(), "recorded": record.event_id or None}

    # ---- corrections of settled sessions

    def _corrections(self, broker: DailyBroker, day: str) -> list[dict[str, object]]:
        late = post_hoc_sessions(self.state)
        rows = []
        for session, ledger in sorted(self.mode["sessions"].items()):
            keys = [*ledger, *(key for key in self.records.keys(session) if key not in ledger)]
            for key in keys:
                entry = ledger.get(key) or {"ordered": 0, "simulated": _NOTHING, "real": _NOTHING}
                record = resolve(
                    self.records, session, key, ordered=bool(entry["ordered"]), post_hoc=session in late
                )
                if record is None:
                    continue
                applied = entry["real"]
                target = self._target(broker, session, key, record, entry)
                if _same(applied, target):
                    continue
                row = self._book(broker, day, session, key, applied, target)
                ledger[key] = {**entry, "real": target}
                rows.append({**row, "recorded": record.event_id or None})
        return rows

    def _target(
        self, broker: DailyBroker, session: str, key: str, record: Record, entry: Mapping[str, object]
    ) -> dict[str, object]:
        if record.outcome == SIMULATED:
            return dict(entry["simulated"])
        if record.outcome == NOT_TRADED:
            return dict(_NOTHING)
        _clock, action, _symbol = parse_key(key)
        notional = float(record.price) * record.quantity
        commission, stamp = broker.profile.costs.fees(notional, action=action, trade_date=session)
        applied = entry["real"]
        # The tax a sale paid depends on its share count only; a changed count
        # is refused where it is booked.
        tax = float(applied["dividend_tax"]) if record.quantity == applied["quantity"] else 0.0
        return {
            **_NOTHING, "quantity": record.quantity, "price": float(record.price), "notional": notional,
            "commission": commission if record.commission is None else float(record.commission),
            "stamp_duty": stamp, "dividend_tax": tax,
        }

    def _book(
        self,
        broker: DailyBroker,
        day: str,
        session: str,
        key: str,
        applied: Mapping[str, object],
        target: dict[str, object],
    ) -> dict[str, object]:
        """Move the account by ``target`` less ``applied``; the journal row.

        Both are outcomes of the same key on ``session``, at their own prices,
        so the difference is exact in shares, cash and cost basis; only the
        marks of the days in between are not restated."""

        _clock, action, symbol = parse_key(key)
        position = broker.positions.get(symbol)
        held = position.quantity if position is not None else 0
        basis = position.average_cost * held if position is not None else 0.0

        def proceeds(outcome: Mapping[str, object]) -> float:
            return float(outcome["notional"]) - float(outcome["commission"]) - float(outcome["stamp_duty"]) - float(
                outcome["dividend_tax"]
            )

        if action == "buy":
            shares = int(target["quantity"]) - int(applied["quantity"])
            cost = (float(target["notional"]) + float(target["commission"])) - (
                float(applied["notional"]) + float(applied["commission"])
            )
            cash, basis_change, realized = -cost, cost, 0.0
        else:
            if applied["quantity"]:
                # The cost basis the sale released, per share.
                unit = (proceeds(applied) - float(applied["realized_pnl"])) / int(applied["quantity"])
            elif position is not None:
                unit = position.average_cost
            else:
                raise ValueError(f"{session} {key}: the book holds no {symbol} to have sold")
            shares = int(applied["quantity"]) - int(target["quantity"])
            cash = proceeds(target) - proceeds(applied)
            basis_change = unit * shares
            target["realized_pnl"] = proceeds(target) - unit * int(target["quantity"]) if target["quantity"] else None
            realized = float(target["realized_pnl"] or 0.0) - float(applied["realized_pnl"] or 0.0)
        if shares:
            if self._ex_dated(broker, symbol, session):
                raise ValueError(
                    f"{session} {key}: {symbol} went ex-date after {session}, so a changed share count cannot "
                    "be booked exactly; record the difference as a trade on an open session"
                )
            if held + shares < 0 or (shares < 0 and position.available_quantity < -shares):
                raise ValueError(
                    f"{session} {key}: the correction takes {-shares} shares of {symbol} back, but the book "
                    f"now holds {held}; correct the later sale first"
                )
            if broker.profile.dividend_tax:
                if action == "sell":
                    # Which lots a sale took, and the tax each owed, is gone
                    # once it is booked.
                    raise ValueError(
                        f"{session} {key}: a settled sale's share count cannot be corrected on a book that "
                        "taxes dividends; record the difference as a trade on an open session"
                    )
                _move_lots(broker, symbol, session, shares)
        quantity = held + shares
        basis += basis_change
        if quantity == 0:
            # Nothing left to carry a cost: what is left of it is realized.
            realized -= basis
            broker.positions.pop(symbol, None)
        elif position is None:
            price = target["price"] if target["price"] is not None else applied["price"]
            broker.positions[symbol] = Position(symbol, quantity, quantity, basis / quantity, float(price))
        else:
            position.quantity = quantity
            position.available_quantity += shares
            position.average_cost = basis / quantity
        broker.cash += cash
        if broker.cash < -1e-6:
            raise ValueError(f"{session} {key}: the correction leaves the book's cash at {broker.cash:,.2f}")
        return {
            "kind": "correction", "trade_date": day, "corrects": session, "key": key,
            "symbol": symbol, "action": action, "quantity": shares, "cash": cash,
            "commission": float(target["commission"]) - float(applied["commission"]),
            "stamp_duty": float(target["stamp_duty"]) - float(applied["stamp_duty"]),
            "dividend_tax": float(target["dividend_tax"]) - float(applied["dividend_tax"]),
            "realized_pnl": realized,
        }

    def _ex_dated(self, broker: DailyBroker, symbol: str, session: str) -> bool:
        if any(action.symbol == symbol and action.trade_date > session for action in broker.corporate_actions):
            return True
        for path in self.root.glob("corporate_actions_[0-9]*.jsonl"):
            if path.stem.removeprefix("corporate_actions_") > session and any(
                row.get("symbol") == symbol for row in read_jsonl(path)[0]
            ):
                return True
        return False


def _same(left: Mapping[str, object], right: Mapping[str, object]) -> bool:
    return left["quantity"] == right["quantity"] and all(
        abs(float(left[field]) - float(right[field])) < CENT
        for field in ("notional", "commission", "stamp_duty", "dividend_tax")
    )


def _move_lots(broker: DailyBroker, symbol: str, session: str, shares: int) -> None:
    """Add shares bought on ``session`` to the lot ledger, or take them back.

    Called only where no ex-date followed ``session``, so those shares carry
    no dividend income; taken back, they must still be in a lot of that day."""

    lots = broker.lots.setdefault(symbol, [])
    if shares > 0:
        index = sum(1 for lot in lots if lot.acquired <= session)
        lots.insert(index, Lot(shares, session))
    else:
        remaining = -shares
        for lot in reversed([lot for lot in lots if lot.acquired == session]):
            taken = min(remaining, lot.quantity)
            lot.quantity -= taken
            remaining -= taken
        if remaining:
            raise ValueError(
                f"{session} {symbol}: the shares bought that day were already sold, so the buy cannot be reduced"
            )
        lots[:] = [lot for lot in lots if lot.quantity > 0]
    if not lots:
        del broker.lots[symbol]


def _date(day: str) -> date:
    return date(int(day[:4]), int(day[4:6]), int(day[6:]))


def simulate(broker: DailyBroker) -> DailyBroker:
    """The simulated track's account for one session: the real account as it
    opened the day, filled by the simulator."""

    return copy.deepcopy(broker)


__all__ = [
    "CORRECTIONS_NAME",
    "FILLED",
    "FILLS_NAME",
    "NOT_TRADED",
    "SIMULATED",
    "SIMULATED_EXECUTIONS_PREFIX",
    "STATE_KEY",
    "UNFILLED",
    "RealFills",
    "Record",
    "Records",
    "adopt",
    "append_records",
    "compare",
    "enable",
    "key_text",
    "mode",
    "normalize_record",
    "order_key",
    "order_session",
    "parse_key",
    "pending_orders",
    "post_hoc_sessions",
    "read_records",
    "record",
    "record_fill",
    "resolve",
    "simulate",
    "unresolved",
]
