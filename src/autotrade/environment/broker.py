"""Daily stock Broker consuming the shared JSON strategy-order contract."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime
from types import MappingProxyType

from autotrade.common.records import dataclass_from_record, dataclass_record
from autotrade.environment.broker_core import (
    BOARDS,
    STAMP_DUTY_CUTOVER,
    board_of,
    reduce_amount_reject,
    validate_buy_lot,
)
from autotrade.environment.distributions import Distribution
from autotrade.environment.dividend_tax import (
    BONUS_SHARE_PAR_CNY,
    OLD_RULE,
    dividend_tax_due,
    dividend_tax_rule,
)
from autotrade.environment.strategy import StrategyOrder

# A-share prices are quoted to 0.01 CNY, so two quotes closer than half a tick
# are one price.
QUOTE_HALF_TICK = 0.005
# A ``pre_close`` that differs from the last close by more than half a tick is
# therefore an exchange price reset (an ex-date), never rounding.
EX_DATE_PRICE_TOLERANCE = QUOTE_HALF_TICK
# The assets (CNY) an individual account needs before the exchange lets it buy
# on each board: none for the main board, 100,000 for ChiNext, 500,000 for the
# STAR Market and the Beijing exchange. Each also asks two years of trading
# experience, which an account is taken to have.
BOARD_MIN_ASSETS_CNY: dict[str, float] = {"main": 0.0, "gem": 100_000.0, "star": 500_000.0, "bj": 500_000.0}


def default_permitted_boards(initial_cash: float) -> tuple[str, ...]:
    """The boards an account opened with ``initial_cash`` may buy on, by the
    exchanges' asset thresholds: what creation stamps on a new experiment unless its
    request names the boards itself."""

    return tuple(board for board in BOARDS if initial_cash >= BOARD_MIN_ASSETS_CNY[board])


def stamped_permitted_boards(params: Mapping[str, object]) -> tuple[str, ...] | None:
    """The boards an experiment's stamped parameters let its account buy on.

    None, for an experiment stamped before the parameter existed: its account buys on
    every board, as it always did, and nothing re-derives a restriction for it.
    """

    value = params.get("permitted_boards")
    return None if value is None else _permitted_boards(value)


def _permitted_boards(value: object) -> tuple[str, ...]:
    """``value`` as a permitted-board set in the vocabulary's order, or a refusal."""

    if not isinstance(value, list | tuple | frozenset | set) or not all(isinstance(board, str) for board in value):
        raise ValueError("permitted_boards must be a list of board names")
    unknown = sorted(set(value) - set(BOARDS))
    if unknown:
        raise ValueError(f"unknown permitted_boards: {unknown}")
    if "main" not in value:
        raise ValueError("permitted_boards must include the main board")
    return tuple(board for board in BOARDS if board in value)


@dataclass(frozen=True)
class BrokerProfile:
    initial_cash: float = 1_000_000.0
    commission_bps: float = 1.0
    min_commission_cny: float = 5.0
    stamp_duty_sell_bps_before_cutover: float = 10.0
    stamp_duty_sell_bps_from_cutover: float = 5.0
    transfer_fee_bps: float = 0.1
    slippage_bps: float = 5.0
    max_total_holdings: int | None = None
    max_single_name_weight: float | None = None
    # Individual dividend tax under the record date's rule (dividend_tax.py),
    # withheld on old-rule distributions and supplemented at sales. Every experiment's
    # account pays it (``worker.resolve_worker_options``), and so does every
    # Paper book opened from one; it reads each ex-date's bonus shares, which
    # a bare Broker over data without an ex-date table does not have, so the
    # field's own default is off.
    dividend_tax: bool = False
    # The boards (``broker_core.BOARDS``) the account may buy on; a buy on any
    # other is rejected, a sale never is. None, no restriction, so that a
    # profile recorded before the field existed replays as it was recorded;
    # creation stamps every new experiment's boards (``default_permitted_boards``).
    permitted_boards: tuple[str, ...] | None = None
    profile_id: str = "gjzq_cash"
    source: str = "docs/ENVIRONMENT_DESIGN.md §3.4"

    def __post_init__(self) -> None:
        if isinstance(self.initial_cash, bool) or not math.isfinite(self.initial_cash) or self.initial_cash <= 0:
            raise ValueError("initial_cash must be a positive finite number")
        if not isinstance(self.dividend_tax, bool):
            raise ValueError("dividend_tax must be a boolean")
        if self.permitted_boards is not None:
            # Canonical, so a profile read back from JSON compares equal.
            object.__setattr__(self, "permitted_boards", _permitted_boards(self.permitted_boards))
        if self.max_total_holdings is not None and (
            isinstance(self.max_total_holdings, bool)
            or not isinstance(self.max_total_holdings, int)
            or self.max_total_holdings <= 0
        ):
            raise ValueError("max_total_holdings must be a positive integer")
        if self.max_single_name_weight is not None and (
            isinstance(self.max_single_name_weight, bool)
            or not math.isfinite(self.max_single_name_weight)
            or self.max_single_name_weight <= 0
        ):
            raise ValueError("max_single_name_weight must be a positive finite number")
        for name in (
            "commission_bps",
            "min_commission_cny",
            "stamp_duty_sell_bps_before_cutover",
            "stamp_duty_sell_bps_from_cutover",
            "transfer_fee_bps",
            "slippage_bps",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be a non-negative finite number")

    def to_record(self) -> dict[str, object]:
        """Every field by name: the one record of a profile, in a run
        manifest and a Paper book alike, which ``from_record`` inverts."""

        return dataclass_record(self)

    @classmethod
    def from_record(cls, record: object) -> BrokerProfile:
        return dataclass_from_record(cls, record)

    # ---- the account's cost math, priced from the fields above ----

    def fill_price(self, price: float, *, action: str) -> float:
        if not math.isfinite(price) or price <= 0:
            raise ValueError("market price must be a positive finite number")
        direction = 1.0 if action == "buy" else -1.0
        return float(price) * (1.0 + direction * self.slippage_bps / 10_000.0)

    def commission(self, notional: float) -> float:
        return max(notional * self.commission_bps / 10_000.0, self.min_commission_cny)

    def transfer_fee(self, notional: float) -> float:
        # 过户费 0.01‰ = 0.1 bps by default, both buy and sell side.
        return notional * self.transfer_fee_bps / 10_000.0

    def trade_fee(self, notional: float) -> float:
        return self.commission(notional) + self.transfer_fee(notional)

    def stamp_duty_on_sale(self, notional: float, trade_date: str) -> float:
        bps = (
            self.stamp_duty_sell_bps_from_cutover
            if str(trade_date) >= STAMP_DUTY_CUTOVER
            else self.stamp_duty_sell_bps_before_cutover
        )
        return notional * bps / 10_000.0

    def fees(self, notional: float, *, action: str, trade_date: str) -> tuple[float, float]:
        """Fee and stamp duty for one fill.

        The first element is the whole trade fee (佣金 plus 过户费), matching how
        the fill's cash movement is settled; ``stamp_duty`` is sell-side only and
        follows the 2023-08-28 cutover.
        """
        if not math.isfinite(notional) or notional <= 0:
            raise ValueError("notional must be a positive finite number")
        stamp_duty = self.stamp_duty_on_sale(notional, trade_date) if action == "sell" else 0.0
        return self.trade_fee(notional), stamp_duty


@dataclass
class Position:
    symbol: str
    quantity: int
    available_quantity: int
    average_cost: float
    last_price: float

    @property
    def market_value(self) -> float:
        return self.quantity * self.last_price


@dataclass
class Lot:
    """The shares of one buy still held, taken first-in first-out on a sale."""

    quantity: int
    # Trade date of the buy (YYYYMMDD); shares an ex-date adds to the lot keep it.
    acquired: str
    # Gross cash plus bonus shares at par, separated by the record-date rule.
    # The old-rule base has already paid 5%; no redundant withheld ledger.
    dividend_income_new: float = 0.0
    dividend_income_old: float = 0.0


@dataclass(frozen=True)
class CorporateAction:
    """One ex-date settlement of a held position, as the Broker applied it."""

    trade_date: str
    symbol: str
    last_close: float
    pre_close: float
    cash_per_share: float
    quantity_before: int
    quantity_after: int
    cash_credit: float
    dividend_tax_withheld: float = 0.0

    def to_record(self) -> dict[str, object]:
        return {
            "trade_date": self.trade_date,
            "symbol": self.symbol,
            "last_close": self.last_close,
            "pre_close": self.pre_close,
            "cash_per_share": self.cash_per_share,
            "quantity_before": self.quantity_before,
            "quantity_after": self.quantity_after,
            "cash_credit": self.cash_credit,
            "dividend_tax_withheld": self.dividend_tax_withheld,
        }


@dataclass(frozen=True)
class Execution:
    symbol: str
    action: str
    quantity: int
    execute_at: str
    matched_at: str
    status: str
    price: float | None = None
    commission: float = 0.0
    stamp_duty: float = 0.0
    # Dividend tax a sale paid on the dividends its shares received; 0 on buys
    # and whenever the profile does not charge the tax.
    dividend_tax: float = 0.0
    # Realized P&L of a position-reducing fill, net of the fees on both legs
    # and of its dividend tax, measured against the released cost basis.
    # ``None`` on buys and rejections: the Broker is the single source of
    # realized P&L, so the return statistics never re-derive a cost basis from
    # the fill stream.
    realized_pnl: float | None = None
    reason: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)
    # The order's session and limit price (``StrategyOrder``), recorded only
    # when it named them: the random baseline replays them.
    session: str | None = None
    limit_price: float | None = None

    def to_record(self) -> dict[str, object]:
        record = {
            "symbol": self.symbol,
            "action": self.action,
            "quantity": self.quantity,
            "execute_at": self.execute_at,
            "matched_at": self.matched_at,
            "status": self.status,
            "price": self.price,
            "commission": self.commission,
            "stamp_duty": self.stamp_duty,
            "dividend_tax": self.dividend_tax,
            "realized_pnl": self.realized_pnl,
            "reason": self.reason,
            "metadata": dict(self.metadata),
        }
        if self.session is not None:
            record["session"] = self.session
        if self.limit_price is not None:
            record["limit_price"] = self.limit_price
        return record


class DailyBroker:
    """Long-only A-share account matched against a trusted price observation."""

    def __init__(self, profile: BrokerProfile | None = None) -> None:
        self.profile = profile or BrokerProfile()
        self.cash = float(self.profile.initial_cash)
        self.initial_equity = float(self.profile.initial_cash)
        self.positions: dict[str, Position] = {}
        self.executions: list[Execution] = []
        # Ex-date settlements, in order: the only way a quantity or the cash
        # balance changes without a fill, so they are kept as auditable records.
        self.corporate_actions: list[CorporateAction] = []
        # Each held name's buys, oldest first, kept only when the profile
        # charges the dividend tax: the tax is the one thing that needs to
        # know when each sold share was bought and what it received.
        self.lots: dict[str, list[Lot]] = {}
        self._current_day: str | None = None

    def open_day(
        self,
        trade_date: str,
        bars: Mapping[str, Mapping[str, object]],
        cash_dividends: Mapping[str, float] | None = None,
        *,
        distributions: Mapping[str, Distribution] | None = None,
    ) -> None:
        """Enter ``trade_date``: release the T+1 locks, then settle ex-dates.

        ``bars`` is the day's market frame and ``cash_dividends`` maps a symbol
        to the cash dividend per share going ex on this day. The exchange's
        ex-rights reference price (the bar's ``pre_close``) is the truth for the
        share leg: a held name whose ``pre_close`` falls below its last close by
        more than the cash dividend (a bonus issue or split) is reset so that
        its value at ``pre_close`` equals its value at the last close, with the
        cash dividend credited and any fractional share paid out in cash. A
        cash dividend alone never changes the share count: the declared cash is
        credited in full even where the exchange's reset is smaller. Shares
        created on the ex-date stay locked until the next day, exactly like a
        same-day buy. The dividend tax follows what is delivered: the cash in
        every case, a bonus share's par value only where the reset creates the
        share.

        ``distributions`` is the authoritative ex-date source: gross cash,
        bonus shares apart from conversions, and record date. Taxed profiles
        require it; ``cash_dividends`` alone is only for untaxed synthetic use.
        """
        if self._current_day == trade_date:
            return
        if self.profile.dividend_tax and distributions is None:
            raise ValueError(f"{trade_date}: dividend tax requires record-dated distributions")
        if distributions is not None and cash_dividends is not None:
            raise ValueError("pass distributions or cash_dividends, not both")
        self._current_day = str(trade_date)
        events = distributions if distributions is not None else {
            symbol: Distribution(cash_per_share=cash) for symbol, cash in (cash_dividends or {}).items()
        }
        for symbol, position in self.positions.items():
            position.available_quantity = position.quantity
            bar = bars.get(symbol)
            # No bar today (full-day suspension or delisting): nothing marks or
            # resets the name, so there is no ex-date to settle either.
            if bar is not None:
                self._settle_ex_date(
                    position, bar, events.get(symbol, Distribution())
                )

    def _settle_ex_date(
        self,
        position: Position,
        bar: Mapping[str, object],
        distribution: Distribution,
    ) -> None:
        symbol, day = position.symbol, self._current_day
        pre_close = _price(bar.get("pre_close"))
        if pre_close is None:
            raise ValueError(
                f"{symbol} on {day}: the day's bar has no usable pre_close, so a held "
                "position cannot be carried across the day"
            )
        cash_per_share, bonus_per_share = distribution.cash_per_share, distribution.bonus_per_share
        if self.profile.dividend_tax and bonus_per_share is None:
            raise ValueError(f"{symbol} on {day}: ex-date table lacks bonus_per_share; rebuild the replay slot")
        for name, value in (("cash dividend", cash_per_share),
                            ("bonus shares", 0.0 if bonus_per_share is None else bonus_per_share)):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"{symbol} on {day}: invalid {name} per share {value!r}")
        cash_per_share = float(cash_per_share)
        last_close = position.last_price
        quantity_before = position.quantity
        try:
            ratio = ex_date_share_multiplier(last_close, pre_close, cash_per_share)
        except ValueError as exc:
            raise ValueError(f"{symbol} on {day}: {exc}") from None
        # Income is what the holder actually receives: the cash dividend, and
        # the bonus shares' par value only when the exchange's reset does grant
        # them. A ratio of None delivers no share, so a per-share bonus the
        # reset does not show (a flat quote, or a fall the cash explains) is not
        # income and carries no tax base, exactly as before the tax existed.
        income_per_share = cash_per_share + (
            (bonus_per_share or 0.0) * BONUS_SHARE_PAR_CNY if ratio is not None else 0.0
        )
        rule = None
        if self.profile.dividend_tax and income_per_share:
            rule = dividend_tax_rule(distribution.record_date or "")
        withheld = position.quantity * income_per_share * rule.withholding_rate if rule else 0.0
        if ratio is None:
            # No share change: no exchange price reset, or a reset that is a
            # cash dividend alone. A cash dividend the table records for today
            # is credited in full either way.
            if not income_per_share:
                return
            quantity_after = quantity_before
            credit = quantity_before * cash_per_share
            if _price_reset(last_close, pre_close):
                position.last_price = pre_close
        else:
            quantity_after = ex_date_shares(quantity_before, ratio)
            if quantity_after <= 0:
                raise ValueError(
                    f"{symbol} on {day}: ex-date reset leaves no whole share of {quantity_before}"
                )
            # Whole shares only: the account is worth at pre_close exactly
            # what it was worth at the last close once the dividend and
            # the fractional share are paid in cash.
            credit = quantity_before * last_close - quantity_after * pre_close
        if self.cash + credit < withheld - 1e-9:
            raise ValueError(f"{symbol} on {day}: insufficient cash for dividend tax withholding {withheld}")
        if ratio is not None:
            position.quantity = quantity_after
            position.available_quantity = min(position.available_quantity, quantity_after)
            position.last_price = pre_close
        # The cost basis carries over and the cash paid out is a return of
        # capital, so a later sale's realized P&L still closes the loop.
        position.average_cost = (
            position.average_cost * quantity_before - credit
        ) / quantity_after
        self.cash += credit - withheld
        if self.profile.dividend_tax:
            self._settle_lots(symbol, income_per_share, ratio, old_rule=rule == OLD_RULE)
        self.corporate_actions.append(
            CorporateAction(
                trade_date=str(day),
                symbol=symbol,
                last_close=last_close,
                pre_close=pre_close,
                cash_per_share=cash_per_share,
                quantity_before=quantity_before,
                quantity_after=quantity_after,
                cash_credit=credit,
                dividend_tax_withheld=withheld,
            )
        )

    def _settle_lots(
        self, symbol: str, income_per_share: float, ratio: float | None, *, old_rule: bool
    ) -> None:
        """Credit one ex-date's dividend income to each lot of ``symbol`` and
        carry the share leg into the lots.

        The lots are resized on their running totals, so they always add up to
        the whole shares the position became; a lot rounded to nothing (only a
        reverse split can do that) leaves with its income.
        """

        held = settled = 0
        kept: list[Lot] = []
        for lot in self.lots[symbol]:
            income = lot.quantity * income_per_share
            if old_rule:
                lot.dividend_income_old += income
            else:
                lot.dividend_income_new += income
            if ratio is not None:
                held += lot.quantity
                previous, settled = settled, ex_date_shares(held, ratio)
                lot.quantity = settled - previous
            if lot.quantity > 0:
                kept.append(lot)
        self.lots[symbol] = kept

    def _release_lots(self, symbol: str, quantity: int) -> float:
        """Take ``quantity`` shares off ``symbol``'s lots, first in first out,
        and return the dividend tax the sold shares owe."""

        lots = self.lots[symbol]
        tax = 0.0
        while quantity:
            lot = lots[0]
            taken = min(quantity, lot.quantity)
            old_income = lot.dividend_income_old * taken / lot.quantity
            new_income = lot.dividend_income_new * taken / lot.quantity
            tax += dividend_tax_due(
                lot.acquired, str(self._current_day), old_income=old_income, new_income=new_income
            )
            lot.quantity -= taken
            lot.dividend_income_old -= old_income
            lot.dividend_income_new -= new_income
            quantity -= taken
            if lot.quantity == 0:
                lots.pop(0)
        if not lots:
            del self.lots[symbol]
        return tax

    def lot_records(self) -> dict[str, list[dict[str, object]]]:
        """The lot ledger as plain records, for a Paper checkpoint."""

        return {symbol: [asdict(lot) for lot in lots] for symbol, lots in sorted(self.lots.items())}

    def restore_lots(self, records: Mapping[str, Sequence[Mapping[str, object]]]) -> None:
        """Reload a ``lot_records`` ledger after the positions it belongs to.

        A profile without the dividend tax keeps no ledger; with it, every held
        name's lots must add up to its quantity and no other name may have any.
        Legacy zero-income lots are unambiguous. Nonzero legacy income must be
        classified offline using its actual distributions, never today's date
        or even acquisition alone: ex-date settlement does not enforce record-
        date entitlement and can credit a pre-acquisition record date.
        """

        lots: dict[str, list[Lot]] = {}
        for symbol, rows in records.items():
            restored = []
            for row in rows:
                complete = "dividend_income_new" in row and "dividend_income_old" in row
                legacy_zero = (
                    "dividend_income_new" not in row and "dividend_income_old" not in row
                    and row.get("dividend_income") == 0.0
                )
                if not complete and not legacy_zero:
                    raise ValueError(
                        "unclassified dividend tax ledger: checkpoint needs explicit dividend_income_new "
                        "and dividend_income_old; verify historical income provenance before an offline conversion"
                    )
                quantity = row["quantity"]
                new = row["dividend_income_new"] if complete else 0.0
                old = row["dividend_income_old"] if complete else 0.0
                if isinstance(quantity, bool) or not isinstance(quantity, int):
                    raise ValueError("invalid quantity in lot ledger")
                if (isinstance(new, bool) or not isinstance(new, (int, float))
                        or isinstance(old, bool) or not isinstance(old, (int, float))):
                    raise ValueError("invalid dividend tax income in lot ledger")
                lot = Lot(quantity, str(row["acquired"]), float(new), float(old))
                if any(not math.isfinite(value) or value < 0 for value in (
                    lot.dividend_income_new, lot.dividend_income_old
                )):
                    raise ValueError("invalid dividend tax income in lot ledger")
                restored.append(lot)
            lots[str(symbol)] = restored
        expected = (
            {symbol: position.quantity for symbol, position in self.positions.items()}
            if self.profile.dividend_tax
            else {}
        )
        held = {symbol: sum(lot.quantity for lot in rows) for symbol, rows in lots.items()}
        if held != expected or any(lot.quantity <= 0 for rows in lots.values() for lot in rows):
            raise ValueError(f"the lot ledger {held} does not match the positions {expected}")
        self.lots = lots

    def account_snapshot(self) -> tuple[float, Mapping[str, int]]:
        return self.cash, MappingProxyType(
            {symbol: position.quantity for symbol, position in sorted(self.positions.items())}
        )

    def mark(self, bars: Mapping[str, Mapping[str, object]], *, price_field: str = "close") -> None:
        for symbol, position in self.positions.items():
            bar = bars.get(symbol)
            if bar is not None:
                price = _price(bar.get(price_field))
                if price is not None:
                    position.last_price = price

    def equity(self) -> float:
        return self.cash + sum(position.market_value for position in self.positions.values())

    def execute(
        self,
        order: StrategyOrder,
        bar: Mapping[str, object] | None,
        *,
        matched_at: datetime,
        raw_price: object,
    ) -> Execution:
        """Fill ``order`` whole against its trading day's ``bar``, or record why not.

        ``raw_price`` is the market price the order meets
        (``replay.engine.resolve_execution_price``): the price at its exact
        ``execute_at`` minute, or a session order's ``session_price``. An
        order that does not fill is recorded as rejected with its reason; an
        unfilled session order (``unfilled_*``) simply expires with its day.
        """
        reason = self._reject_reason(order, bar, raw_price=raw_price)
        if reason is not None:
            return self._record(order, matched_at, status="rejected", reason=reason)
        assert bar is not None
        base_price = _price(raw_price)
        assert base_price is not None
        fill_price, reason = self._fill_price(order, bar, base_price)
        if reason is not None:
            return self._record(order, matched_at, status="rejected", reason=reason)
        assert fill_price is not None
        notional = fill_price * order.quantity
        if self._current_day is None:
            raise RuntimeError("open_day must set the trading day before an order is executed")
        commission, stamp_duty = self.profile.fees(
            notional, action=order.action, trade_date=self._current_day
        )
        realized_pnl: float | None = None
        dividend_tax = 0.0
        if order.action == "buy":
            required_cash = notional + commission
            if required_cash > self.cash + 1e-9:
                return self._record(order, matched_at, status="rejected", reason="insufficient_cash")
            self.cash -= required_cash
            if self.profile.dividend_tax:
                self.lots.setdefault(order.symbol, []).append(Lot(order.quantity, self._current_day))
            existing = self.positions.get(order.symbol)
            if existing is None:
                self.positions[order.symbol] = Position(
                    symbol=order.symbol,
                    quantity=order.quantity,
                    available_quantity=0,
                    average_cost=required_cash / order.quantity,
                    last_price=fill_price,
                )
            else:
                total_cost = existing.average_cost * existing.quantity + required_cash
                existing.quantity += order.quantity
                existing.average_cost = total_cost / existing.quantity
                existing.last_price = fill_price
        else:
            position = self.positions[order.symbol]
            # Deducted at the sale, as the broker deducts what the depository
            # computes on the sold shares; shares still held owe nothing yet.
            if self.profile.dividend_tax:
                dividend_tax = self._release_lots(order.symbol, order.quantity)
            proceeds = notional - commission - stamp_duty - dividend_tax
            basis_released = position.average_cost * order.quantity
            realized_pnl = proceeds - basis_released
            self.cash += proceeds
            position.quantity -= order.quantity
            position.available_quantity -= order.quantity
            position.last_price = fill_price
            if position.quantity == 0:
                del self.positions[order.symbol]
        return self._record(
            order,
            matched_at,
            status="filled",
            price=fill_price,
            commission=commission,
            stamp_duty=stamp_duty,
            dividend_tax=dividend_tax,
            realized_pnl=realized_pnl,
        )

    def _fill_price(
        self, order: StrategyOrder, bar: Mapping[str, object], base_price: float
    ) -> tuple[float | None, str | None]:
        """The price ``order`` fills at from ``base_price``, or why it does not fill.

        A buy is held against the day's upper price limit and a sale against
        its lower one, the adverse limit. An order without a session, or an
        open or close order, fills at ``base_price`` plus slippage unless that
        reaches the limit. An intraday order does not fill on a day locked at
        the limit (high and low both on it). Otherwise, without a limit price
        it fills at the day's average price plus slippage, held at the limit
        the day traded inside of; with one it fills at ``base_price`` without
        slippage once the day's low (buy) or high (sale) reaches its limit.
        """

        buy = order.action == "buy"
        limit = _price(bar.get("up_limit" if buy else "down_limit"))
        if limit is None:
            return None, "missing_daily_price_limit"
        if order.session != "intraday":
            price = self.profile.fill_price(base_price, action=order.action)
            if _reaches_limit(price, limit, buy=buy):
                return None, "daily_price_limit" if order.session is None else "unfilled_price_limit"
            return price, None
        high, low = _price(bar.get("high")), _price(bar.get("low"))
        if high is None or low is None:
            return None, "missing_execution_price"
        if abs(high - limit) < QUOTE_HALF_TICK and abs(low - limit) < QUOTE_HALF_TICK:
            return None, "unfilled_locked"
        if order.limit_price is None:
            price = self.profile.fill_price(base_price, action=order.action)
            return (min(price, limit) if buy else max(price, limit)), None
        reached = (
            low <= order.limit_price + QUOTE_HALF_TICK
            if buy
            else high >= order.limit_price - QUOTE_HALF_TICK
        )
        return (base_price, None) if reached else (None, "unfilled_limit")

    def _reject_reason(
        self,
        order: StrategyOrder,
        bar: Mapping[str, object] | None,
        *,
        raw_price: object,
    ) -> str | None:
        # The account's permission, settled before the market is looked at.
        if (
            order.action == "buy"
            and self.profile.permitted_boards is not None
            and board_of(order.symbol) not in self.profile.permitted_boards
        ):
            return "board_not_permitted"
        if bar is None:
            return "missing_execution_price"
        if bool(bar.get("is_suspended", False)):
            return "suspended"
        if _price(raw_price) is None:
            return "missing_execution_price"
        if order.action == "buy":
            try:
                validate_buy_lot(order.quantity, order.symbol)
            except ValueError:
                return "invalid_buy_lot"
            if (
                self.profile.max_total_holdings is not None
                and order.symbol not in self.positions
                and len(self.positions) >= self.profile.max_total_holdings
            ):
                return "max_holdings_reached"
            return self._single_name_cap_reject(order.symbol, order.quantity, _price(raw_price))
        position = self.positions.get(order.symbol)
        if position is None or position.available_quantity < order.quantity:
            return "insufficient_available_position"
        return reduce_amount_reject(order.quantity, position.available_quantity, order.symbol)

    def _single_name_cap_reject(self, symbol: str, shares: int, raw_price: float | None) -> str | None:
        """Reject an opening order that would breach the single-name cap."""
        if self.profile.max_single_name_weight is None:
            return None
        if raw_price is None or raw_price <= 0:
            return "single_name_weight_cap"
        cap_notional = self.profile.max_single_name_weight * self.initial_equity
        position = self.positions.get(symbol)
        held_notional = position.quantity * raw_price if position is not None else 0.0
        if held_notional + shares * raw_price > cap_notional + 1e-6:
            return "single_name_weight_cap"
        return None

    def _record(
        self,
        order: StrategyOrder,
        matched_at: datetime,
        *,
        status: str,
        price: float | None = None,
        commission: float = 0.0,
        stamp_duty: float = 0.0,
        dividend_tax: float = 0.0,
        realized_pnl: float | None = None,
        reason: str | None = None,
    ) -> Execution:
        execution = Execution(
            symbol=order.symbol,
            action=order.action,
            quantity=order.quantity,
            execute_at=order.execute_at.isoformat(),
            matched_at=matched_at.isoformat(),
            status=status,
            price=price,
            commission=commission,
            stamp_duty=stamp_duty,
            dividend_tax=dividend_tax,
            realized_pnl=realized_pnl,
            reason=reason,
            metadata=order.metadata,
            session=order.session,
            limit_price=order.limit_price,
        )
        self.executions.append(execution)
        return execution


def session_price(order: StrategyOrder, bar: Mapping[str, object]) -> float | None:
    """The price a session order meets on its trading day's bar, before
    slippage and the Broker's gates; None when the bar lacks a field it reads.

    ``open`` and ``close`` meet the day's auction prices. ``intraday`` meets
    the day's volume-weighted average price, ``amount / vol`` (CNY over shares
    in the normalized bar). With a ``limit_price`` it is a day limit order,
    which, if it trades, trades at the better of the open and its limit: a buy
    at the lower, a sale at the higher, so an open that gaps through the limit
    fills at the open.
    """

    if order.session in ("open", "close"):
        return _price(bar.get(order.session))
    if order.limit_price is None:
        amount, volume = _price(bar.get("amount")), _price(bar.get("vol"))
        return None if amount is None or volume is None else amount / volume
    opening = _price(bar.get("open"))
    if opening is None:
        return None
    return min(opening, order.limit_price) if order.action == "buy" else max(opening, order.limit_price)


def _price(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    price = float(value)
    return price if math.isfinite(price) and price > 0 else None


def _price_reset(last_close: float, pre_close: float) -> bool:
    # Quotes are two-decimal, so the gap is compared at a precision well
    # below a tick: binary noise (10.005 - 10.0 > 0.005) must not count.
    return round(abs(pre_close - last_close), 6) > EX_DATE_PRICE_TOLERANCE


def ex_date_share_multiplier(
    last_close: float, pre_close: float, cash_per_share: float
) -> float | None:
    """The share multiplier one day's opening reset applies to a holding, or
    None when it changes no share count.

    ``pre_close = (last_close - cash) / (1 + r)``: the multiplier is recovered
    from the exchange's own reset, never from a ratio table. No reset changes
    no share count, and neither does a cash dividend without a bonus (a reset
    no larger than the cash, to half a tick): the exchange quotes the reset on
    the dividend rounded to a tick, or on a smaller virtual per-share amount
    when treasury shares take none, yet every holder is paid the declared cash.
    A cash dividend never shrinks a holding -- flooring a price-derived 0.9997
    would shave a share off most of them, and leave a one-share holding with
    none. The one rule the Broker settles a holding by and the zero-skill
    random baseline sizes its exits by.
    """

    if not _price_reset(last_close, pre_close):
        return None
    ratio = (last_close - cash_per_share) / pre_close
    if not math.isfinite(ratio) or ratio <= 0.0:
        raise ValueError(
            f"ex-date reset from {last_close} to {pre_close} with cash "
            f"{cash_per_share} implies an invalid share multiplier {ratio!r}"
        )
    if (
        cash_per_share > 0.0
        and round(last_close - cash_per_share - pre_close, 6) <= EX_DATE_PRICE_TOLERANCE
    ):
        return None
    return ratio


def ex_date_shares(quantity: int, ratio: float) -> int:
    """The whole shares a holding of ``quantity`` becomes under ``ratio``."""

    return math.floor(round(quantity * ratio, 6))


# A-share limits are quoted to 0.01 CNY, but the gate compares a modeled fill
# price, base * (1 + slippage_bps/10_000), which binary arithmetic can place a
# few ulps on either side of the exact boundary: 10.0 * 1.0010 evaluates to
# 10.009999999999998 against a 10.01 limit and would slip past a plain
# `>= limit`. The tolerance absorbs that arithmetic noise only: it is far below
# a tick, so a price a tick inside the limit still fills, and far above the
# worst-case double error at A-share price magnitudes (~1e-12 at 1e4 CNY).
_PRICE_LIMIT_TOLERANCE = 1e-9


def _reaches_limit(price: float, limit: float, *, buy: bool) -> bool:
    return price >= limit - _PRICE_LIMIT_TOLERANCE if buy else price <= limit + _PRICE_LIMIT_TOLERANCE


__all__ = [
    "EX_DATE_PRICE_TOLERANCE",
    "QUOTE_HALF_TICK",
    "BrokerProfile",
    "CorporateAction",
    "DailyBroker",
    "Execution",
    "Lot",
    "Position",
    "ex_date_share_multiplier",
    "ex_date_shares",
    "session_price",
]
