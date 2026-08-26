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
        # Cost feedback for the return statistics: gross traded notional drives
        # turnover, the fee totals separate cost from alpha, and the rejection
        # tally names the failure modes a strategy has to fix.
        self.traded_notional = 0.0
        self.fees_paid = 0.0
        self.stamp_duty_paid = 0.0
        self.reject_counts: dict[str, int] = {}
        self._current_day: str | None = None

    def open_day(self, trade_date: str) -> None:
        if self._current_day == trade_date:
            return
        self._current_day = str(trade_date)
        for position in self.positions.values():
            position.available_quantity = position.quantity

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
        if _limit_blocked(order.action, fill_price, bar):
            return self._record(order, matched_at, status="rejected", reason="daily_price_limit")
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


def _limit_blocked(action: str, price: float, bar: Mapping[str, object]) -> bool:
    field = "up_limit" if action == "buy" else "down_limit"
    limit = _price(bar.get(field))
    if limit is None:
        return False
    return price >= limit if action == "buy" else price <= limit


Broker = DailyBroker

__all__ = ["Broker", "BrokerProfile", "DailyBroker", "Execution", "Position"]
