"""Scheduled strategy execution with point-in-time inputs and JSON orders."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, time
from time import perf_counter

from autotrade.environment.broker import BrokerProfile, DailyBroker
from autotrade.environment.executor import FittableStrategyExecutor, StrategyExecutor
from autotrade.environment.strategy import (
    CN_TZ,
    AccountSnapshot,
    NLQuery,
    StrategyContext,
    StrategyOrder,
    StrategySchedule,
    validate_order_payload,
)

from .market import DailyMarketData
from .stats import PhaseTimer, ReplayResult

MARKET_OPEN = time(9, 30)
MARKET_CLOSE = time(15, 0)


class BacktestError(RuntimeError):
    """A strategy cannot produce a truthful replay result."""


@dataclass(frozen=True)
class StrategyDataView:
    """Read-only PIT directories visible for one daily strategy decision."""

    snapshot_dir: str = ""
    asof_dir: str = ""
    asof_version: str = "0"


ContextDataProvider = Callable[[datetime], StrategyDataView]
ExecutionPriceProvider = Callable[[str, datetime], object | None]


class DailyOrderInbox:
    """Environment-owned queue populated only through the JSON order contract."""

    def __init__(self) -> None:
        self._orders: list[StrategyOrder] = []

    def submit(self, payload: object, *, inference_at: datetime) -> None:
        self._orders.extend(validate_order_payload(payload, inference_at=inference_at))
        self._orders.sort(key=lambda order: order.execute_at)

    def take_due(self, cutoff: datetime) -> tuple[StrategyOrder, ...]:
        index = 0
        while index < len(self._orders) and self._orders[index].execute_at <= cutoff:
            index += 1
        due = tuple(self._orders[:index])
        del self._orders[:index]
        return due

    def records(self) -> tuple[dict[str, object], ...]:
        return tuple(order.to_record() for order in self._orders)


StrategyCallable = Callable[[StrategyContext], Sequence[Mapping[str, object]]]
StrategyRunner = StrategyCallable | StrategyExecutor


class DailyReplayEngine:
    def __init__(
        self,
        *,
        schedule: StrategySchedule,
        strategy: StrategyRunner,
        broker: DailyBroker | None = None,
        nl_query: NLQuery | None = None,
        context_data: ContextDataProvider | None = None,
        execution_price: ExecutionPriceProvider | None = None,
        timer: PhaseTimer | None = None,
    ) -> None:
        if not callable(strategy) and not isinstance(strategy, StrategyExecutor):
            raise TypeError("strategy must be callable or implement StrategyExecutor")
        self.schedule = schedule
        self.strategy = strategy
        self.broker = broker or DailyBroker()
        self.nl_query = nl_query
        self.context_data = context_data
        self.execution_price = execution_price
        self.timer = timer if timer is not None else PhaseTimer()
        self.inbox = DailyOrderInbox()
        # A strategy with fit(context) fits at the first decision of the replay
        # and again whenever its declared refit period rolls over.
        self._fittable = (
            strategy if isinstance(strategy, FittableStrategyExecutor) else None
        )
        self._last_fit_date: str | None = None

    def run(self, market: DailyMarketData) -> ReplayResult:
        equity_curve: list[dict[str, object]] = []
        inference_dates: list[str] = []
        previous_trade_date: str | None = None
        initial_equity = self.broker.profile.initial_cash
        started = perf_counter()
        for trade_date in market.trade_dates:
            self.broker.open_day(trade_date)
            day = date.fromisoformat(f"{trade_date[:4]}-{trade_date[4:6]}-{trade_date[6:8]}")
            day_end = datetime.combine(day, time.max, tzinfo=CN_TZ)
            inference_at = self.schedule.at(trade_date)
            bars = market.bars_for_day(trade_date)
            due = self.schedule.is_due(trade_date, previous_trade_date)

            # Existing orders at or before the scheduled inference point settle
            # first, so the strategy receives the account that actually existed
            # at its decision time. Newly returned orders can still request that
            # exact timestamp and are handled by the end-of-day pass below.
            with self.timer.phase("broker"):
                self._match_due(inference_at, market)
            if due:
                self._infer(market, trade_date, inference_at)
                inference_dates.append(inference_at.isoformat())
            with self.timer.phase("broker"):
                self._match_due(day_end, market)
                self.broker.mark(bars)
                cash, positions = self.broker.account_snapshot()
                equity = self.broker.equity()

            equity_curve.append(
                {
                    "trade_date": trade_date,
                    "initial_equity": initial_equity,
                    "equity": equity,
                    "cash": cash,
                    "positions": dict(positions),
                }
            )
            previous_trade_date = trade_date
        return ReplayResult(
            equity_curve=tuple(equity_curve),
            executions=tuple(execution.to_record() for execution in self.broker.executions),
            inference_dates=tuple(inference_dates),
            pending_orders=self.inbox.records(),
            wall_seconds=perf_counter() - started,
            phase_seconds=self.timer.to_record(),
        )

    def _infer(self, market: DailyMarketData, trade_date: str, inference_at: datetime) -> None:
        cash, positions = self.broker.account_snapshot()
        with self.timer.phase("data_view"):
            data_view = (
                self.context_data(inference_at)
                if self.context_data is not None
                else StrategyDataView()
            )
        if not isinstance(data_view, StrategyDataView):
            raise BacktestError("context_data must return StrategyDataView")
        fittable = self._fittable
        context = StrategyContext(
            inference_at=inference_at,
            bars=market.visible_at(inference_at),
            account=AccountSnapshot(cash=cash, positions=positions),
            snapshot_dir=data_view.snapshot_dir,
            asof_dir=data_view.asof_dir,
            asof_version=data_view.asof_version,
            state_dir=fittable.context_state_dir if fittable is not None else "",
            models_dir=fittable.context_models_dir if fittable is not None else "",
            _nl_query=self.nl_query,
        )
        # fit receives the very context object generate_orders gets on this
        # day, so it cannot see one row more than the decision itself.
        if (
            fittable is not None
            and fittable.fit_schedule is not None
            and fittable.fit_schedule.is_due(trade_date, self._last_fit_date)
        ):
            try:
                with self.timer.phase("fit"):
                    fittable.fit(context)
            except Exception as exc:
                raise BacktestError(f"fit failed at {inference_at.isoformat()}: {exc}") from exc
            self._last_fit_date = trade_date
        try:
            # Includes the host NL wait: ctx.nl() blocks inside the strategy
            # call, and phase_seconds["nl"] reports that share separately.
            with self.timer.phase("strategy"):
                if isinstance(self.strategy, StrategyExecutor):
                    payload = self.strategy.execute(context)
                else:
                    payload = self.strategy(context)
            self.inbox.submit(payload, inference_at=inference_at)
        except Exception as exc:
            raise BacktestError(f"generate_orders failed at {inference_at.isoformat()}: {exc}") from exc

    def _match_due(
        self,
        cutoff: datetime,
        market: DailyMarketData,
    ) -> None:
        for order in self.inbox.take_due(cutoff):
            bar, raw_price = resolve_execution_price(
                market,
                order,
                execution_price=self.execution_price,
            )
            self.broker.execute(
                order,
                bar,
                matched_at=order.execute_at,
                raw_price=raw_price,
            )


def resolve_execution_price(
    market: DailyMarketData,
    order: StrategyOrder,
    *,
    execution_price: ExecutionPriceProvider | None,
) -> tuple[Mapping[str, object] | None, object | None]:
    """Resolve only the price at the order's requested timestamp.

    Daily open and close are exact observations at 09:30 and 15:00. Every
    other time requires an injected static price source; the engine never
    rounds or delays an order to one of the daily endpoints.
    """

    when = order.execute_at.astimezone(CN_TZ)
    trade_date = when.strftime("%Y%m%d")
    bar = market.bars_for_day(trade_date).get(order.symbol)
    local_time = when.timetz().replace(tzinfo=None)
    if local_time == MARKET_OPEN:
        return bar, bar.get("open") if bar is not None else None
    if local_time == MARKET_CLOSE:
        return bar, bar.get("close") if bar is not None else None
    if execution_price is None:
        return bar, None
    return bar, execution_price(order.symbol, when)


def run_daily_replay(
    *,
    daily,
    strategy: StrategyRunner,
    schedule: StrategySchedule,
    profile: BrokerProfile | None = None,
    nl_query: NLQuery | None = None,
    context_data: ContextDataProvider | None = None,
    execution_price: ExecutionPriceProvider | None = None,
) -> ReplayResult:
    # One timer spans the market build and the replay loop, so a single
    # phase_seconds breakdown covers the whole call.
    timer = PhaseTimer()
    with timer.phase("market_build"):
        market = daily if isinstance(daily, DailyMarketData) else DailyMarketData(daily)
    return DailyReplayEngine(
        schedule=schedule,
        strategy=strategy,
        broker=DailyBroker(profile),
        nl_query=nl_query,
        context_data=context_data,
        execution_price=execution_price,
        timer=timer,
    ).run(market)


__all__ = [
    "BacktestError",
    "ContextDataProvider",
    "DailyOrderInbox",
    "DailyReplayEngine",
    "ExecutionPriceProvider",
    "PhaseTimer",
    "StrategyDataView",
    "StrategyRunner",
    "resolve_execution_price",
    "run_daily_replay",
]
