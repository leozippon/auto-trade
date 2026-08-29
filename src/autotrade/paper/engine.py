"""Persistent daily Paper engine for the JSON ``generate_orders`` ABI.

The engine advances one explicitly requested local trading day from persisted
end-of-day inputs. Historical minute, auction, fundamental, event, macro, and
text records can be supplied through ``visible_records``; the strategy is invoked
only when its configured ``StrategySchedule`` is due.
"""

from __future__ import annotations

import fcntl
import math
import shutil
import uuid
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from autotrade.environment.broker import BrokerProfile, DailyBroker, Position
from autotrade.environment.executor import (
    DockerStrategyExecutor,
    FittableStrategyExecutor,
    StrategyExecutor,
)
from autotrade.environment.replay.engine import (
    ContextDataProvider,
    ExecutionPriceProvider,
    StrategyDataView,
    resolve_execution_price,
)
from autotrade.environment.replay.market import DailyMarketData
from autotrade.environment.runtime import chmod_tree
from autotrade.environment.sandbox import SandboxConfig
from autotrade.environment.strategy import (
    CN_TZ,
    AccountSnapshot,
    NLQuery,
    StrategyContext,
    StrategyOrder,
    StrategySchedule,
    validate_order_payload,
)

from .storage import append_jsonl_once, read_json, write_json_atomic

PAPER_SOURCE = "paper_engine"
PAPER_STATE_SCHEMA_VERSION = 3
PAPER_STATE_NAME = ".paper_state.json"
PAPER_LOCK_NAME = ".paper_engine.lock"
# Per-day working directory of a strategy that declares ``fit(context)``. It is
# rebuilt empty for every day and discarded with that day's executor, so no
# fitted state ever crosses a Paper day or enters the account journals.
STRATEGY_STATE_NAME = ".strategy_state"
SNAPSHOT_NAME = "account_snapshot.json"


class PaperEngineError(RuntimeError):
    pass


VisibleRecords = Callable[[datetime], Sequence[Mapping[str, object]]]
# (strategy_path, sandbox, this day's empty state directory, revision models
# directory or None) -> executor. The engine owns the two directories so every
# implementation mounts the same ones.
ExecutorFactory = Callable[[Path, SandboxConfig, Path, Path | None], StrategyExecutor]


class DailyPaperEngine:
    def __init__(
        self,
        *,
        strategy_path: str | Path,
        daily: pd.DataFrame | str | Path,
        state_root: str | Path = "data/trading/paper",
        models_dir: str | Path | None = None,
        strategy_revision: str | None = None,
        schedule: StrategySchedule | None = None,
        profile: BrokerProfile | None = None,
        sandbox: SandboxConfig | None = None,
        nl_query: NLQuery | None = None,
        visible_records: VisibleRecords | None = None,
        context_data: ContextDataProvider | None = None,
        execution_price: ExecutionPriceProvider | None = None,
        executor_factory: ExecutorFactory | None = None,
    ) -> None:
        self.strategy_path = Path(strategy_path).resolve()
        if not self.strategy_path.is_file():
            raise ValueError(f"strategy file does not exist: {self.strategy_path}")
        frame = pd.read_parquet(daily) if isinstance(daily, (str, Path)) else daily
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("daily must be a DataFrame or parquet path")
        self.market = DailyMarketData(frame)
        self.state_root = Path(state_root).resolve()
        # The activated revision's frozen models/ tree, mounted read-only for
        # both fit and generate_orders exactly as a replay mounts it.
        self.models_dir = Path(models_dir).resolve() if models_dir is not None else None
        if self.models_dir is not None and not self.models_dir.is_dir():
            raise ValueError(f"models directory does not exist: {self.models_dir}")
        self.strategy_revision = str(strategy_revision or self.strategy_path.name)
        if not self.strategy_revision.strip():
            raise ValueError("strategy_revision must be non-empty")
        self.schedule = schedule or StrategySchedule()
        self.profile = profile or BrokerProfile()
        self.sandbox = sandbox or SandboxConfig()
        self.nl_query = nl_query
        self.visible_records = visible_records
        self.context_data = context_data
        self.execution_price = execution_price
        self.executor_factory = executor_factory
        self._executor: StrategyExecutor | None = None

    def close(self) -> None:
        if self._executor is not None:
            self._executor.close()
            self._executor = None
        self._discard_strategy_state()

    def _strategy_state_dir(self) -> Path:
        """An empty state directory for this day's fit, replacing any leftover.

        World-writable because the sandbox fit worker runs as a non-root user;
        the inference worker binds the same directory read-only.
        """

        state_dir = self.state_root / STRATEGY_STATE_NAME
        self._discard_strategy_state()
        state_dir.mkdir(parents=True)
        state_dir.chmod(0o777)
        return state_dir

    def _discard_strategy_state(self) -> None:
        state_dir = self.state_root / STRATEGY_STATE_NAME
        if not state_dir.exists():
            return
        try:
            chmod_tree(state_dir, file_mode=0o644, dir_mode=0o755)
            shutil.rmtree(state_dir)
        except OSError as exc:
            raise PaperEngineError(
                f"cannot discard strategy state: {state_dir}: {exc}"
            ) from exc

    def run_day(self, trade_date: str) -> dict[str, object]:
        if len(trade_date) != 8 or not trade_date.isdigit():
            raise ValueError("trade_date must be YYYYMMDD")
        if trade_date not in self.market.trade_dates:
            raise PaperEngineError(f"daily market has no trading day {trade_date}")
        self.state_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.state_root.chmod(0o700)
        with self._exclusive_lock():
            try:
                state = self._load_state()
                self._reconcile_emissions(state)
                state = self._prepare_day(state, trade_date)
                if state.get("day_complete"):
                    return self._summary(state)
                broker = self._restore_broker(state)
                pending = self._restore_orders(state)
            except PaperEngineError:
                raise
            except (OSError, TypeError, ValueError) as exc:
                raise PaperEngineError(f"cannot restore Paper account: {exc}") from exc
            try:
                self._advance_day(state, broker, pending)
                return self._summary(state)
            except Exception as exc:
                state["last_error"] = f"{type(exc).__name__}: {exc}"
                self._checkpoint(state, broker, pending)
                self._write_snapshot(state, broker, ok=False, error=str(exc))
                if isinstance(exc, PaperEngineError):
                    raise
                raise PaperEngineError(f"paper day {trade_date} failed: {exc}") from exc
            finally:
                self.close()

    @contextmanager
    def _exclusive_lock(self):
        lock_path = self.state_root / PAPER_LOCK_NAME
        with lock_path.open("a+b") as stream:
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise PaperEngineError("another Paper writer owns this account") from exc
            try:
                yield
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)

    def _new_state(self) -> dict[str, object]:
        return {
            "schema_version": PAPER_STATE_SCHEMA_VERSION,
            "strategy_revision": self.strategy_revision,
            "strategy_path": str(self.strategy_path),
            "schedule": self.schedule.to_record(),
            "profile": asdict(self.profile),
            "trade_date": "",
            "previous_trade_date": "",
            "phase": "done",
            "day_complete": False,
            "inference_dates": [],
            "pending_orders": [],
            "account": None,
            "emissions": [],
            "last_error": "",
        }

    def _load_state(self) -> dict[str, object]:
        path = self.state_root / PAPER_STATE_NAME
        if not path.exists():
            markers = [
                item.name for item in self.state_root.iterdir()
                if item.name == SNAPSHOT_NAME or item.name.startswith(("orders_", "executions_", "events_")) or item.name == "equity_daily.jsonl"
            ] if self.state_root.exists() else []
            if markers:
                raise PaperEngineError(f"Paper journals exist but {PAPER_STATE_NAME} is missing; refusing to fabricate an account")
            return self._new_state()
        try:
            state = read_json(path)
        except (TypeError, ValueError) as exc:
            raise PaperEngineError(str(exc)) from exc
        if state.get("schema_version") != PAPER_STATE_SCHEMA_VERSION:
            raise PaperEngineError("unsupported Paper state schema")
        if state.get("strategy_revision") != self.strategy_revision:
            raise PaperEngineError("strategy revision changed; activate revisions only between accounts")
        if state.get("strategy_path") != str(self.strategy_path):
            raise PaperEngineError("strategy path differs from the persisted Paper account")
        if state.get("schedule") != self.schedule.to_record():
            raise PaperEngineError("strategy schedule differs from the persisted Paper account")
        if state.get("profile") != asdict(self.profile):
            raise PaperEngineError("Broker profile differs from the persisted Paper account")
        return state

    def _prepare_day(self, state: dict[str, object], trade_date: str) -> dict[str, object]:
        current = str(state.get("trade_date") or "")
        if current and current > trade_date:
            raise PaperEngineError(f"persisted account is later than requested date: {current} > {trade_date}")
        if current == trade_date:
            return state
        if current and not state.get("day_complete"):
            raise PaperEngineError(f"previous Paper day {current} is incomplete")
        if current:
            later_market_dates = tuple(day for day in self.market.trade_dates if day > current)
            if not later_market_dates:
                raise PaperEngineError(f"daily market has no trading day after persisted date {current}")
            expected = later_market_dates[0]
            if trade_date != expected:
                raise PaperEngineError(
                    f"Paper must advance to the next market trading day {expected}; requested {trade_date}"
                )
        state.update({
            "previous_trade_date": current,
            "trade_date": trade_date,
            "phase": "before_inference",
            "day_complete": False,
            "inference_dates": [],
            "last_error": "",
        })
        broker = self._restore_broker(state)
        broker.open_day(trade_date)
        self._checkpoint(state, broker, self._restore_orders(state))
        return state

    def _advance_day(self, state: dict[str, object], broker: DailyBroker, pending: list[StrategyOrder]) -> None:
        trade_date = str(state["trade_date"])
        day = date(int(trade_date[:4]), int(trade_date[4:6]), int(trade_date[6:]))
        day_end = datetime.combine(day, datetime.max.time(), tzinfo=CN_TZ)
        inference_at = self.schedule.at(trade_date)
        due = self.schedule.is_due(trade_date, str(state.get("previous_trade_date") or "") or None)
        bars = self.market.bars_for_day(trade_date)
        while state.get("phase") != "done":
            phase = str(state["phase"])
            if phase == "before_inference":
                self._match(state, broker, pending, inference_at)
                state["phase"] = "inference"
            elif phase == "inference":
                if due:
                    self._infer(state, broker, pending, inference_at)
                state["phase"] = "after_inference"
            elif phase == "after_inference":
                self._match(state, broker, pending, day_end)
                broker.mark(bars)
                state["phase"] = "finalize"
            elif phase == "finalize":
                self._queue_emission(state, "equity_daily.jsonl", {
                    "kind": "equity", "trade_date": trade_date, "equity": broker.equity(),
                    "cash": broker.cash, "position_count": len(broker.positions),
                })
                state["phase"] = "done"
                state["day_complete"] = True
            else:
                raise PaperEngineError(f"unknown Paper phase: {phase}")
            self._checkpoint(state, broker, pending)
            self._reconcile_emissions(state)
        self._write_snapshot(state, broker, ok=True, error=None)

    def _infer(self, state: dict[str, object], broker: DailyBroker, pending: list[StrategyOrder], inference_at: datetime) -> None:
        cash, positions = broker.account_snapshot()
        visible = (
            tuple(self.visible_records(inference_at))
            if self.visible_records
            else self.market.visible_at(inference_at)
        )
        data_view = self.context_data(inference_at) if self.context_data is not None else StrategyDataView()
        if not isinstance(data_view, StrategyDataView):
            raise PaperEngineError("context_data must return StrategyDataView")
        if self._executor is None:
            state_dir = self._strategy_state_dir()
            self._executor = (
                self.executor_factory(
                    self.strategy_path, self.sandbox, state_dir, self.models_dir
                )
                if self.executor_factory is not None
                else DockerStrategyExecutor(
                    self.strategy_path,
                    self.sandbox,
                    snapshot_dir=data_view.snapshot_dir or None,
                    asof_dir=data_view.asof_dir or None,
                    models_dir=self.models_dir,
                    state_dir=state_dir,
                )
            )
        fittable = (
            self._executor
            if isinstance(self._executor, FittableStrategyExecutor)
            else None
        )
        context = StrategyContext(
            inference_at=inference_at,
            bars=visible,
            account=AccountSnapshot(cash=cash, positions=positions),
            snapshot_dir=data_view.snapshot_dir,
            asof_dir=data_view.asof_dir,
            asof_version=data_view.asof_version,
            state_dir=fittable.context_state_dir if fittable is not None else "",
            models_dir=fittable.context_models_dir if fittable is not None else "",
            _nl_query=self.nl_query,
        )
        # Paper advances one day at a time and keeps no fitted state between
        # days, so a declared fit runs before every decision on the very context
        # that decision gets. REFIT_PERIOD is a within-replay cadence and cannot
        # stretch across Paper days.
        if fittable is not None and fittable.fit_schedule is not None:
            try:
                fittable.fit(context)
            except Exception as exc:
                raise PaperEngineError(
                    f"fit failed at {inference_at.isoformat()}: {exc}"
                ) from exc
        payload = self._executor.execute(context)
        orders = validate_order_payload(payload, inference_at=inference_at)
        pending.extend(orders)
        pending.sort(key=lambda item: item.execute_at)
        inference_dates = list(state.get("inference_dates") or [])
        inference_dates.append(inference_at.isoformat())
        state["inference_dates"] = inference_dates
        for order in orders:
            self._queue_emission(state, f"orders_{state['trade_date']}.jsonl", {"kind": "order", **order.to_record()})

    def _match(
        self,
        state: dict[str, object],
        broker: DailyBroker,
        pending: list[StrategyOrder],
        cutoff: datetime,
    ) -> None:
        due = [order for order in pending if order.execute_at <= cutoff]
        pending[:] = [order for order in pending if order.execute_at > cutoff]
        for order in due:
            bar, raw_price = resolve_execution_price(
                self.market,
                order,
                execution_price=self.execution_price,
            )
            execution = broker.execute(
                order,
                bar,
                matched_at=order.execute_at,
                raw_price=raw_price,
            )
            self._queue_emission(state, f"executions_{state['trade_date']}.jsonl", {"kind": "execution", **execution.to_record()})

    def _checkpoint(self, state: dict[str, object], broker: DailyBroker, pending: list[StrategyOrder]) -> None:
        state["account"] = self._account_record(broker)
        state["pending_orders"] = [order.to_record() for order in pending]
        write_json_atomic(self.state_root / PAPER_STATE_NAME, state)

    def _restore_broker(self, state: dict[str, object]) -> DailyBroker:
        broker = DailyBroker(self.profile)
        raw = state.get("account")
        if not isinstance(raw, dict):
            return broker
        cash = raw.get("cash")
        if isinstance(cash, bool) or not isinstance(cash, (int, float)) or not math.isfinite(float(cash)):
            raise PaperEngineError("persisted account cash is invalid")
        broker.cash = float(cash)
        positions = raw.get("positions")
        if not isinstance(positions, list):
            raise PaperEngineError("persisted account positions are invalid")
        broker.positions = {}
        for item in positions:
            if not isinstance(item, dict):
                raise PaperEngineError("persisted position is invalid")
            position = Position(**item)
            broker.positions[position.symbol] = position
        broker._current_day = str(raw.get("current_day") or "") or None
        return broker

    @staticmethod
    def _account_record(broker: DailyBroker) -> dict[str, object]:
        return {
            "cash": broker.cash,
            "current_day": broker._current_day,
            "positions": [asdict(position) for _, position in sorted(broker.positions.items())],
        }

    @staticmethod
    def _restore_orders(state: dict[str, object]) -> list[StrategyOrder]:
        raw = state.get("pending_orders") or []
        if not isinstance(raw, list):
            raise PaperEngineError("persisted pending_orders is invalid")
        orders: list[StrategyOrder] = []
        for item in raw:
            if not isinstance(item, dict):
                raise PaperEngineError("persisted pending order is invalid")
            execute_at = datetime.fromisoformat(str(item.get("execute_at")))
            orders.append(StrategyOrder.from_record(item, inference_at=execute_at))
        orders.sort(key=lambda item: item.execute_at)
        return orders

    def _queue_emission(self, state: dict[str, object], name: str, payload: dict[str, object]) -> None:
        emissions = list(state.get("emissions") or [])
        emissions.append({"name": name, "payload": {"event_id": f"paper_{uuid.uuid4().hex}", **payload}})
        state["emissions"] = emissions

    def _reconcile_emissions(self, state: dict[str, object]) -> None:
        emissions = state.get("emissions") or []
        if not isinstance(emissions, list):
            raise PaperEngineError("persisted emissions are invalid")
        for emission in emissions:
            if not isinstance(emission, dict):
                raise PaperEngineError("persisted emission is invalid")
            name = str(emission.get("name") or "")
            payload = emission.get("payload")
            if Path(name).name != name or not isinstance(payload, dict):
                raise PaperEngineError("persisted emission target is invalid")
            append_jsonl_once(self.state_root / name, payload)
        if emissions:
            state["emissions"] = []
            write_json_atomic(self.state_root / PAPER_STATE_NAME, state)

    def _write_snapshot(self, state: dict[str, object], broker: DailyBroker, *, ok: bool, error: str | None) -> None:
        payload: dict[str, object] = {
            # generated_at is the console's staleness clock: a snapshot without
            # it reads as an unusable account state, not as a fresh one.
            "generated_at": datetime.now(CN_TZ).isoformat(),
            "source": PAPER_SOURCE, "ok": ok, "trade_date": state.get("trade_date"),
            "day_complete": state.get("day_complete"), "phase": state.get("phase"),
            "strategy_revision": self.strategy_revision, "cash": broker.cash,
            "equity": broker.equity(), "positions": [asdict(position) for _, position in sorted(broker.positions.items())],
            "pending_order_count": len(state.get("pending_orders") or []),
        }
        if error:
            payload["error"] = error
        write_json_atomic(self.state_root / SNAPSHOT_NAME, payload)

    def _summary(self, state: dict[str, object]) -> dict[str, object]:
        account = state.get("account") if isinstance(state.get("account"), dict) else {}
        return {
            "trade_date": state.get("trade_date"), "day_complete": bool(state.get("day_complete")),
            "phase": state.get("phase"), "strategy_revision": state.get("strategy_revision"),
            "cash": account.get("cash"), "position_count": len(account.get("positions") or []),
            "pending_order_count": len(state.get("pending_orders") or []),
            "inference_dates": list(state.get("inference_dates") or []),
        }


__all__ = ["PAPER_LOCK_NAME", "PAPER_STATE_NAME", "SNAPSHOT_NAME", "DailyPaperEngine", "PaperEngineError"]
