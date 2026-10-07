"""Daily stock Broker consuming the shared JSON strategy-order contract."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime
from functools import cached_property
from types import MappingProxyType

from autotrade.environment.broker_core import (
    BOARDS,
    BONUS_SHARE_PAR_CNY,
    STAMP_DUTY_CUTOVER,
    CostModel,
    board_of,
    dividend_tax_rate,
    reduce_amount_reject,
    validate_buy_lot,
)
from autotrade.environment.strategy import StrategyOrder

# A-share prices are quoted to 0.01 CNY, so a ``pre_close`` that differs from
# the last close by more than half a tick is an exchange price reset (an
# ex-date), never rounding.
EX_DATE_PRICE_TOLERANCE = 0.005
# The assets (CNY) an individual account needs before the exchange lets it buy
# on each board: none for the main board, 100,000 for ChiNext, 500,000 for the
# STAR Market and the Beijing exchange. Each also asks two years of trading
# experience, which an account is taken to have.
BOARD_MIN_ASSETS_CNY: dict[str, float] = {"main": 0.0, "gem": 100_000.0, "star": 500_000.0, "bj": 500_000.0}


def default_permitted_boards(initial_cash: float) -> tuple[str, ...]:
    """The boards an account opened with ``initial_cash`` may buy on, by the
    exchanges' asset thresholds: what creation stamps on a new arm unless its
    request names the boards itself."""

    return tuple(board for board in BOARDS if initial_cash >= BOARD_MIN_ASSETS_CNY[board])


def stamped_permitted_boards(params: Mapping[str, object]) -> tuple[str, ...] | None:
    """The boards an arm's stamped parameters let its account buy on.

    None, for an arm stamped before the parameter existed: its account buys on
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
    # The individual dividend tax, charged at each sale on the dividends the
    # sold shares received (``broker_core.dividend_tax_rate``). Every arm's
    # account pays it (``worker.resolve_worker_options``), and so does every
    # Paper book opened from one; it reads each ex-date's bonus shares, which
    # a bare Broker over data without an ex-date table does not have, so the
    # field's own default is off.
    dividend_tax: bool = False
    # The boards (``broker_core.BOARDS``) the account may buy on; a buy on any
    # other is rejected, a sale never is. None, no restriction, so that a
    # profile recorded before the field existed replays as it was recorded;
    # creation stamps every new arm's boards (``default_permitted_boards``).
    permitted_boards: tuple[str, ...] | None = None
    profile_id: str = "gjzq_cash"
    source: str = "docs/environment-design.md §3.4"

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
        # Building the cost model validates the cost fields at construction.
        self.costs  # noqa: B018

    def to_record(self) -> dict[str, object]:
        return {
            "profile_id": self.profile_id,
            "source": self.source,
            "initial_cash": self.initial_cash,
            "commission_bps": self.commission_bps,
            "min_commission_cny": self.min_commission_cny,
            "stamp_duty_sell_bps_before_cutover": self.stamp_duty_sell_bps_before_cutover,
            "stamp_duty_sell_bps_from_cutover": self.stamp_duty_sell_bps_from_cutover,
            "stamp_duty_cutover_date": STAMP_DUTY_CUTOVER,
            "transfer_fee_bps": self.transfer_fee_bps,
            "slippage_bps": self.slippage_bps,
            "max_total_holdings": self.max_total_holdings,
            "max_single_name_weight": self.max_single_name_weight,
            "dividend_tax": self.dividend_tax,
            # Absent, like the parameter itself, on an unrestricted account.
            **({} if self.permitted_boards is None else {"permitted_boards": list(self.permitted_boards)}),
        }

    @cached_property
    def costs(self) -> CostModel:
        return CostModel(
            commission_bps=self.commission_bps,
            min_commission_cny=self.min_commission_cny,
            stamp_duty_sell_bps_before_cutover=self.stamp_duty_sell_bps_before_cutover,
            stamp_duty_sell_bps_from_cutover=self.stamp_duty_sell_bps_from_cutover,
            transfer_fee_bps=self.transfer_fee_bps,
            slippage_bps=self.slippage_bps,
        )


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
    # Dividend income these shares received while held (gross cash plus the
    # par value of bonus shares, CNY), taxed when they are sold.
    dividend_income: float = 0.0


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

    def to_record(self) -> dict[str, object]:
        return {
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
        bonus_shares: Mapping[str, float] | None = None,
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
        same-day buy.

        ``bonus_shares`` maps a symbol to its bonus shares (送股) per share
        going ex on this day, apart from capital-reserve conversions (转增).
        Only the dividend tax reads it, so a profile that charges the tax
        refuses a day without it rather than treat every share leg as tax-free.
        """
        if self._current_day == trade_date:
            return
        if self.profile.dividend_tax and bonus_shares is None:
            raise ValueError(
                f"{trade_date}: the dividend tax needs each ex-date's bonus shares apart from "
                "capital-reserve conversions, and this replay's ex-date table does not carry "
                "bonus_per_share; a replay slot built before that column existed must be rebuilt"
            )
        self._current_day = str(trade_date)
        dividends = cash_dividends or {}
        bonuses = bonus_shares or {}
        for symbol, position in self.positions.items():
            position.available_quantity = position.quantity
            bar = bars.get(symbol)
            # No bar today (full-day suspension or delisting): nothing marks or
            # resets the name, so there is no ex-date to settle either.
            if bar is not None:
                self._settle_ex_date(
                    position, bar, dividends.get(symbol, 0.0), bonuses.get(symbol, 0.0)
                )

    def _settle_ex_date(
        self,
        position: Position,
        bar: Mapping[str, object],
        cash_per_share: object,
        bonus_per_share: object,
    ) -> None:
        symbol, day = position.symbol, self._current_day
        pre_close = _price(bar.get("pre_close"))
        if pre_close is None:
            raise ValueError(
                f"{symbol} on {day}: the day's bar has no usable pre_close, so a held "
                "position cannot be carried across the day"
            )
        for name, value in (("cash dividend", cash_per_share), ("bonus shares", bonus_per_share)):
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
        if ratio is None:
            # No share change: no exchange price reset, or a reset that is a
            # cash dividend alone. A cash dividend the table records for today
            # is credited in full either way.
            if cash_per_share == 0.0:
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
            position.quantity = quantity_after
            position.available_quantity = min(position.available_quantity, quantity_after)
            position.last_price = pre_close
        # The cost basis carries over and the cash paid out is a return of
        # capital, so a later sale's realized P&L still closes the loop.
        position.average_cost = (
            position.average_cost * quantity_before - credit
        ) / quantity_after
        self.cash += credit
        if self.profile.dividend_tax:
            # Bonus shares are income only where the share leg created shares.
            bonus = float(bonus_per_share) if ratio is not None else 0.0
            self._settle_lots(symbol, cash_per_share + bonus * BONUS_SHARE_PAR_CNY, ratio)
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
            )
        )

    def _settle_lots(self, symbol: str, income_per_share: float, ratio: float | None) -> None:
        """Credit one ex-date's dividend income to each lot of ``symbol`` and
        carry the share leg into the lots.

        The lots are resized on their running totals, so they always add up to
        the whole shares the position became; a lot rounded to nothing (only a
        reverse split can do that) leaves with its income.
        """

        held = settled = 0
        kept: list[Lot] = []
        for lot in self.lots[symbol]:
            lot.dividend_income += lot.quantity * income_per_share
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
            income = lot.dividend_income * taken / lot.quantity
            if income:
                tax += income * dividend_tax_rate(lot.acquired, str(self._current_day))
            lot.quantity -= taken
            lot.dividend_income -= income
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
        """

        lots = {
            str(symbol): [Lot(int(row["quantity"]), str(row["acquired"]), float(row["dividend_income"])) for row in rows]
            for symbol, rows in records.items()
        }
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
        reason = self._reject_reason(order, bar, raw_price=raw_price)
        if reason is not None:
            return self._record(order, matched_at, status="rejected", reason=reason)
        assert bar is not None
        base_price = _price(raw_price)
        assert base_price is not None
        fill_price = self.profile.costs.fill_price(base_price, action=order.action)
        limit_reason = _price_limit_reject(order.action, fill_price, bar)
        if limit_reason is not None:
            return self._record(order, matched_at, status="rejected", reason=limit_reason)
        notional = fill_price * order.quantity
        if self._current_day is None:
            raise RuntimeError("open_day must set the trading day before an order is executed")
        commission, stamp_duty = self.profile.costs.fees(
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
        )
        self.executions.append(execution)
        return execution


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
    panel sizes its exits by.
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


def _price_limit_reject(action: str, price: float, bar: Mapping[str, object]) -> str | None:
    field = "up_limit" if action == "buy" else "down_limit"
    limit = _price(bar.get(field))
    if limit is None:
        return "missing_daily_price_limit"
    blocked = price >= limit if action == "buy" else price <= limit
    return "daily_price_limit" if blocked else None


__all__ = [
    "EX_DATE_PRICE_TOLERANCE",
    "BrokerProfile",
    "CorporateAction",
    "DailyBroker",
    "Execution",
    "Lot",
    "Position",
    "ex_date_share_multiplier",
    "ex_date_shares",
]
