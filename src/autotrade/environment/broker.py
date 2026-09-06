"""Daily stock Broker consuming the shared JSON strategy-order contract."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from functools import cached_property
from types import MappingProxyType

from autotrade.environment.broker_core import (
    STAMP_DUTY_CUTOVER,
    CostModel,
    reduce_amount_reject,
    validate_buy_lot,
)
from autotrade.environment.strategy import StrategyOrder

# A-share prices are quoted to 0.01 CNY, so a ``pre_close`` that differs from
# the last close by more than half a tick is an exchange price reset (an
# ex-date), never rounding.
EX_DATE_PRICE_TOLERANCE = 0.005


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
    profile_id: str = "gjzq_cash"
    source: str = "docs/environment-design.md §3.4"

    def __post_init__(self) -> None:
        if isinstance(self.initial_cash, bool) or not math.isfinite(self.initial_cash) or self.initial_cash <= 0:
            raise ValueError("initial_cash must be a positive finite number")
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
        CostModel(
            commission_bps=self.commission_bps,
            min_commission_cny=self.min_commission_cny,
            stamp_duty_sell_bps_before_cutover=self.stamp_duty_sell_bps_before_cutover,
            stamp_duty_sell_bps_from_cutover=self.stamp_duty_sell_bps_from_cutover,
            transfer_fee_bps=self.transfer_fee_bps,
            slippage_bps=self.slippage_bps,
        )

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
    # Realized P&L of a position-reducing fill, net of the fees on both legs and
    # measured against the released cost basis. ``None`` on buys and rejections:
    # the Broker is the single source of realized P&L, so the return statistics
    # never re-derive a cost basis from the fill stream.
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
        # Cost feedback for the return statistics: gross traded notional drives
        # turnover, the fee totals separate cost from alpha, and the rejection
        # tally names the failure modes a strategy has to fix.
        self.traded_notional = 0.0
        self.fees_paid = 0.0
        self.stamp_duty_paid = 0.0
        self.reject_counts: dict[str, int] = {}
        self._current_day: str | None = None

    def open_day(
        self,
        trade_date: str,
        bars: Mapping[str, Mapping[str, object]],
        cash_dividends: Mapping[str, float] | None = None,
    ) -> None:
        """Enter ``trade_date``: release the T+1 locks, then settle ex-dates.

        ``bars`` is the day's market frame and ``cash_dividends`` maps a symbol
        to the cash dividend per share going ex on this day. The exchange's
        ex-rights reference price (the bar's ``pre_close``) is the truth for the
        share leg: a held name whose ``pre_close`` differs from its last close
        is reset so that its value at ``pre_close`` equals its value at the
        last close, with the cash dividend credited and any fractional share
        paid out in cash. Shares created on the ex-date stay locked until the
        next day, exactly like a same-day buy.
        """
        if self._current_day == trade_date:
            return
        self._current_day = str(trade_date)
        dividends = cash_dividends or {}
        for symbol, position in self.positions.items():
            position.available_quantity = position.quantity
            bar = bars.get(symbol)
            # No bar today (full-day suspension or delisting): nothing marks or
            # resets the name, so there is no ex-date to settle either.
            if bar is not None:
                self._settle_ex_date(position, bar, dividends.get(symbol, 0.0))

    def _settle_ex_date(
        self, position: Position, bar: Mapping[str, object], cash_per_share: object
    ) -> None:
        symbol, day = position.symbol, self._current_day
        pre_close = _price(bar.get("pre_close"))
        if pre_close is None:
            raise ValueError(
                f"{symbol} on {day}: the day's bar has no usable pre_close, so a held "
                "position cannot be carried across the day"
            )
        if (
            isinstance(cash_per_share, bool)
            or not isinstance(cash_per_share, (int, float))
            or not math.isfinite(cash_per_share)
            or cash_per_share < 0
        ):
            raise ValueError(f"{symbol} on {day}: invalid cash dividend per share {cash_per_share!r}")
        cash_per_share = float(cash_per_share)
        last_close = position.last_price
        quantity_before = position.quantity
        # Quotes are two-decimal, so the gap is compared at a precision well
        # below a tick: binary noise (10.005 - 10.0 > 0.005) must not count.
        if round(abs(pre_close - last_close), 6) <= EX_DATE_PRICE_TOLERANCE:
            # No exchange price reset means no share change; a cash dividend
            # the table records for today is still credited.
            if cash_per_share == 0.0:
                return
            quantity_after = quantity_before
            credit = quantity_before * cash_per_share
        else:
            # pre_close = (last_close - cash) / (1 + r): the share multiplier is
            # recovered from the exchange's own reset, never from a ratio table.
            ratio = (last_close - cash_per_share) / pre_close
            if not math.isfinite(ratio) or ratio <= 0.0:
                raise ValueError(
                    f"{symbol} on {day}: ex-date reset from {last_close} to {pre_close} with "
                    f"cash {cash_per_share} implies an invalid share multiplier {ratio!r}"
                )
            quantity_after = math.floor(round(quantity_before * ratio, 6))
            if quantity_after <= 0:
                raise ValueError(
                    f"{symbol} on {day}: ex-date reset leaves no whole share of {quantity_before}"
                )
            # Whole shares only: the account is worth at pre_close exactly what
            # it was worth at the last close once the dividend and the
            # fractional share are paid in cash.
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
        if order.action == "buy":
            required_cash = notional + commission
            if required_cash > self.cash + 1e-9:
                return self._record(order, matched_at, status="rejected", reason="insufficient_cash")
            self.cash -= required_cash
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
            proceeds = notional - commission - stamp_duty
            basis_released = position.average_cost * order.quantity
            realized_pnl = proceeds - basis_released
            self.cash += proceeds
            position.quantity -= order.quantity
            position.available_quantity -= order.quantity
            position.last_price = fill_price
            if position.quantity == 0:
                del self.positions[order.symbol]
        # Counted only once the fill actually settles: a rejected order moves no
        # notional and pays no fee.
        self.traded_notional += notional
        self.fees_paid += commission
        self.stamp_duty_paid += stamp_duty
        return self._record(
            order,
            matched_at,
            status="filled",
            price=fill_price,
            commission=commission,
            stamp_duty=stamp_duty,
            realized_pnl=realized_pnl,
        )

    def _reject_reason(
        self,
        order: StrategyOrder,
        bar: Mapping[str, object] | None,
        *,
        raw_price: object,
    ) -> str | None:
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
            realized_pnl=realized_pnl,
            reason=reason,
            metadata=order.metadata,
        )
        if reason is not None:
            self.reject_counts[reason] = self.reject_counts.get(reason, 0) + 1
        self.executions.append(execution)
        return execution


def _price(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    price = float(value)
    return price if math.isfinite(price) and price > 0 else None


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
    "Position",
]
