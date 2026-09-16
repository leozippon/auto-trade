"""Persistent daily Paper book for the JSON ``generate_orders`` ABI.

A Paper book is one research replay stretched over the real calendar. Each run
names one target session ``D`` and does two things in order:

1. It settles every earlier session of the book whose daily bars have landed:
   the day opens (T+1 release, ex-date settlement), the orders decided for it
   fill at their ``execute_at`` prices, and the account is marked at the close.
2. It makes the pre-open decision for ``D`` at the scheduled inference time,
   from the account as of the previous close and the PIT view as of that
   inference time, so the orders exist before the market opens. They stay
   pending until the next run settles ``D``.

What a research replay keeps across days, the book keeps too. Fitted state is
persisted and refitted only when the strategy's ``REFIT_PERIOD`` rolls over,
exactly as a replay refits. A replay also keeps one strategy worker alive for
the whole window, so module-level state (a rebalance counter, say) carries from
call to call; a Paper run cannot keep a process alive between mornings, so it
re-establishes that state by re-issuing every earlier decision call of the book
to a fresh worker, in order, with that day's PIT view and journaled account,
and discards their outputs before the real call. The journals, not those
re-issued calls, are the record of what the book decided.
"""

from __future__ import annotations

import fcntl
import math
import os
import shutil
import uuid
from collections.abc import Callable, Iterable
from contextlib import contextmanager
from dataclasses import asdict
from datetime import date, datetime, time
from pathlib import Path
from typing import Protocol

from autotrade.environment.broker import BrokerProfile, DailyBroker, Position
from autotrade.environment.executor import (
    DockerStrategyExecutor,
    FittableStrategyExecutor,
    StrategyExecutor,
)
from autotrade.environment.replay.engine import (
    StrategyDataView,
    resolve_execution_price,
)
from autotrade.environment.replay.market import DailyMarketData
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
from autotrade.environment.strategy_loader import validate_strategy_package

from .storage import append_jsonl_once, read_json, read_jsonl, write_json_atomic

PAPER_SOURCE = "paper_engine"
PAPER_STATE_SCHEMA_VERSION = 4
PAPER_STATE_NAME = ".paper_state.json"
PAPER_LOCK_NAME = ".paper_engine.lock"
# Persisted fitted state: one directory per fit, named by its session; only
# the one ``fit_state`` names is current.
STRATEGY_STATE_NAME = ".strategy_state"
SNAPSHOT_NAME = "account_snapshot.json"
# Journal-row key for the host-side reference quote of an order (last close and
# name). Namespaced so it cannot collide with a strategy's own order metadata.
REFERENCE_KEY = "paper_reference"


class PaperEngineError(RuntimeError):
    pass


class PaperDataNotReady(PaperEngineError):
    """The committed data does not yet reach the sessions a run needs."""


class PaperWriterBusy(PaperEngineError):
    """Another process holds this book's writer lock."""


class PaperData(Protocol):
    """One run's market and PIT inputs for the book window ``[start, D]``."""

    market: DailyMarketData
    # Exchange sessions, sorted; they cover the book window and the session
    # before it.
    sessions: tuple[str, ...]
    # Last session whose daily bars the committed data contains.
    release_end: str
    generation_id: str
    nl_query: NLQuery | None

    def context_data(self, inference_at: datetime) -> StrategyDataView: ...

    def execution_price(self, symbol: str, when: datetime) -> object | None: ...

    def references(self, symbols: Iterable[str], before: str) -> dict[str, dict[str, object]]: ...

    def close(self) -> None: ...


# (book start, target session) -> that run's data. Called only when a run has
# work to do, because building the PIT view is the expensive part.
PaperDataFactory = Callable[[str, str], PaperData]
# (strategy_path, sandbox, first data view, state directory or None, revision
# models directory or None) -> executor.
ExecutorFactory = Callable[
    [Path, SandboxConfig, StrategyDataView, Path | None, Path | None], StrategyExecutor
]


class DailyPaperEngine:
    def __init__(
        self,
        *,
        strategy_path: str | Path,
        strategy_revision: str,
        state_root: str | Path,
        data_factory: PaperDataFactory,
        models_dir: str | Path | None = None,
        schedule: StrategySchedule | None = None,
        profile: BrokerProfile | None = None,
        sandbox: SandboxConfig | None = None,
        executor_factory: ExecutorFactory | None = None,
    ) -> None:
        self.strategy_path = Path(strategy_path).resolve()
        if not self.strategy_path.is_file():
            raise ValueError(f"strategy file does not exist: {self.strategy_path}")
        self.fit_schedule = validate_strategy_package(self.strategy_path)
        self.strategy_revision = str(strategy_revision or "").strip()
        if not self.strategy_revision:
            raise ValueError("strategy_revision must be non-empty")
        self.state_root = Path(state_root).resolve()
        self.data_factory = data_factory
        # The activated revision's frozen models/ tree, mounted read-only for
        # both fit and generate_orders exactly as a replay mounts it.
        self.models_dir = Path(models_dir).resolve() if models_dir is not None else None
        if self.models_dir is not None and not self.models_dir.is_dir():
            raise ValueError(f"models directory does not exist: {self.models_dir}")
        self.schedule = schedule or StrategySchedule()
        self.profile = profile or BrokerProfile()
        self.sandbox = sandbox or SandboxConfig()
        self.executor_factory = executor_factory

    # ------------------------------------------------------------------ run

    def run_day(self, trade_date: str) -> dict[str, object]:
        """Settle every landed session before ``trade_date`` and decide it.

        Idempotent: a session the book already decided returns its summary
        without touching data or the strategy.
        """

        if len(trade_date) != 8 or not trade_date.isdigit():
            raise ValueError("trade_date must be YYYYMMDD")
        self.state_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.state_root.chmod(0o700)
        with self._exclusive_lock():
            try:
                state = self._load_state()
                self._reconcile_emissions(state)
                broker = self._restore_broker(state)
                pending = self._restore_orders(state)
            except PaperEngineError:
                raise
            except (OSError, TypeError, ValueError) as exc:
                raise PaperEngineError(f"cannot restore Paper account: {exc}") from exc
            decided = [str(row["trade_date"]) for row in state["decisions"]]
            if decided and trade_date == decided[-1]:
                return self._summary(state)
            if decided and trade_date < decided[-1]:
                raise PaperEngineError(
                    f"the book has already decided {decided[-1]}; cannot decide the earlier session {trade_date}"
                )
            if self.schedule.at(trade_date).date() > datetime.now(CN_TZ).date():
                # A decision taken the evening before would miss whatever the
                # nightly chain lands for its inference time.
                raise PaperEngineError(f"session {trade_date} has not begun; decide it on its own calendar day")
            data: PaperData | None = None
            try:
                data = self.data_factory(str(state["start_date"] or trade_date), trade_date)
                self._advance(state, data, broker, pending, trade_date)
                self._write_snapshot(state, broker, ok=True, error=None)
                return self._summary(state)
            except Exception as exc:
                # The in-memory state may hold a half-settled day; only the
                # last checkpoint on disk is the account, so record the error
                # there.
                path = self.state_root / PAPER_STATE_NAME
                persisted = read_json(path) if path.exists() else self._new_state()
                persisted["last_error"] = f"{type(exc).__name__}: {exc}"
                write_json_atomic(path, persisted)
                self._write_snapshot(persisted, self._restore_broker(persisted), ok=False, error=str(exc))
                if isinstance(exc, PaperEngineError):
                    raise
                raise PaperEngineError(f"paper session {trade_date} failed: {exc}") from exc
            finally:
                if data is not None:
                    data.close()

    def _advance(
        self,
        state: dict[str, object],
        data: PaperData,
        broker: DailyBroker,
        pending: list[StrategyOrder],
        trade_date: str,
    ) -> None:
        sessions = tuple(data.sessions)
        if trade_date not in sessions:
            raise PaperEngineError(f"{trade_date} is not an exchange session")
        earlier = [day for day in sessions if day < trade_date]
        prior = earlier[-1] if earlier else None
        if prior is None or data.release_end < prior:
            raise PaperDataNotReady(
                f"committed data ends at {data.release_end} (generation {data.generation_id}); "
                f"the decision for {trade_date} needs sessions through {prior}"
            )
        start = str(state["start_date"] or trade_date)
        floor = str(state["settled_through"] or "")
        to_settle = [day for day in sessions if start <= day < trade_date and day > floor]
        missing = [day for day in to_settle if day not in set(data.market.trade_dates)]
        if missing:
            raise PaperDataNotReady(
                f"committed data (generation {data.generation_id}) has no daily bars for session {missing[0]}"
            )
        decided = {str(row["trade_date"]) for row in state["decisions"]}
        for day in to_settle:
            before = [value for value in sessions if value < day]
            due = self.schedule.is_due(day, before[-1] if before else None)
            if due and day not in decided:
                raise PaperEngineError(
                    f"session {day} was never decided; run the book for {day} before {trade_date}"
                )
        for day in to_settle:
            self._settle(state, data, broker, pending, day)
        # The book's first decision is always due, as a replay's first day is.
        first = not state["decisions"]
        if self.schedule.is_due(trade_date, None if first else prior):
            self._decide(state, data, broker, pending, trade_date, prior)

    # --------------------------------------------------------------- settle

    def _settle(
        self,
        state: dict[str, object],
        data: PaperData,
        broker: DailyBroker,
        pending: list[StrategyOrder],
        day: str,
    ) -> None:
        market = data.market
        bars = market.bars_for_day(day)
        settled_actions = len(broker.corporate_actions)
        broker.open_day(day, bars, market.cash_dividends_for_day(day))
        # Opening the day is the only step that changes the account without a
        # fill, so its ex-date settlements are journaled like executions.
        for action in broker.corporate_actions[settled_actions:]:
            self._queue_emission(
                state, f"corporate_actions_{day}.jsonl", {"kind": "corporate_action", **action.to_record()}
            )
        day_end = datetime.combine(_date(day), time.max, tzinfo=CN_TZ)
        due = [order for order in pending if order.execute_at <= day_end]
        pending[:] = [order for order in pending if order.execute_at > day_end]
        for order in due:
            bar, raw_price = resolve_execution_price(market, order, execution_price=data.execution_price)
            execution = broker.execute(order, bar, matched_at=order.execute_at, raw_price=raw_price)
            self._queue_emission(state, f"executions_{day}.jsonl", {"kind": "execution", **execution.to_record()})
        broker.mark(bars)
        self._queue_emission(state, "equity_daily.jsonl", {
            "kind": "equity", "trade_date": day, "equity": broker.equity(),
            "cash": broker.cash, "position_count": len(broker.positions),
        })
        state["settled_through"] = day
        self._checkpoint(state, broker, pending)
        self._reconcile_emissions(state)

    # --------------------------------------------------------------- decide

    def _decide(
        self,
        state: dict[str, object],
        data: PaperData,
        broker: DailyBroker,
        pending: list[StrategyOrder],
        trade_date: str,
        prior: str,
    ) -> None:
        inference_at = self.schedule.at(trade_date)
        cash, positions = broker.account_snapshot()
        fit_due = self.fit_schedule is not None and self.fit_schedule.is_due(
            trade_date, state["last_fit_date"] or None
        )
        state_dir, staged = self._decision_state_dir(state, fit_due=fit_due, trade_date=trade_date)
        executor: StrategyExecutor | None = None
        matched = 0
        try:
            history = list(state["decisions"])
            for row in history:
                at = datetime.fromisoformat(str(row["inference_at"]))
                view = data.context_data(at)
                executor = executor or self._executor(view, state_dir)
                account = AccountSnapshot(cash=float(row["cash"]), positions=dict(row["positions"]))
                replayed = validate_order_payload(
                    executor.execute(self._context(executor, data, at, account, view)), inference_at=at
                )
                matched += [order.to_record() for order in replayed] == self._journaled_orders(str(row["trade_date"]))
            view = data.context_data(inference_at)
            executor = executor or self._executor(view, state_dir)
            context = self._context(executor, data, inference_at, AccountSnapshot(cash=cash, positions=positions), view)
            if fit_due:
                if not isinstance(executor, FittableStrategyExecutor):
                    raise PaperEngineError("the strategy declares fit but its executor cannot run it")
                try:
                    executor.fit(context)
                except Exception as exc:
                    raise PaperEngineError(f"fit failed at {inference_at.isoformat()}: {exc}") from exc
            try:
                orders = validate_order_payload(executor.execute(context), inference_at=inference_at)
            except Exception as exc:
                raise PaperEngineError(f"generate_orders failed at {inference_at.isoformat()}: {exc}") from exc
        except BaseException:
            if executor is not None:
                executor.close()
            if staged:
                _remove_tree(state_dir)
            raise
        executor.close()
        if staged:
            self._publish_fit_state(state_dir, trade_date)
        # Display quotes for the order sheet: a held name at the close the
        # account was marked with, anything else at its last close.
        quotes = data.references({order.symbol for order in orders} | set(positions), trade_date)
        for symbol, position in broker.positions.items():
            quotes[symbol] = {**quotes.get(symbol, {}), "close": position.last_price}
        state["decisions"] = [*state["decisions"], {
            "trade_date": trade_date,
            "inference_at": inference_at.isoformat(),
            # When the run actually wrote this decision, as opposed to the
            # scheduled PIT instant above: the console reads it to say whether
            # the session's sheet exists yet. A same-date rerun returns before
            # deciding, so it is written once.
            "decided_at": datetime.now(CN_TZ).isoformat(),
            "data_through": prior,
            "generation_id": data.generation_id,
            "cash": cash,
            "positions": dict(positions),
            "equity": broker.equity(),
            "quotes": {symbol: quotes.get(symbol, {}) for symbol in sorted(positions)},
            "fitted": fit_due,
            "replayed_calls": len(history),
            "replayed_matching_journal": matched,
        }]
        if fit_due:
            state["last_fit_date"] = trade_date
            state["fit_state"] = trade_date
        if not state["start_date"]:
            state["start_date"] = trade_date
        pending.extend(orders)
        pending.sort(key=lambda item: item.execute_at)
        for order in orders:
            self._queue_emission(state, f"orders_{trade_date}.jsonl", {
                "kind": "order", **order.to_record(), REFERENCE_KEY: quotes.get(order.symbol, {}),
            })
        state["last_error"] = ""
        state["warnings"] = []
        self._checkpoint(state, broker, pending)
        self._reconcile_emissions(state)
        warnings = self._prune_fit_states(state)
        if warnings:
            state["warnings"] = warnings
            write_json_atomic(self.state_root / PAPER_STATE_NAME, state)

    def _context(
        self,
        executor: StrategyExecutor,
        data: PaperData,
        inference_at: datetime,
        account: AccountSnapshot,
        view: StrategyDataView,
    ) -> StrategyContext:
        fittable = executor if isinstance(executor, FittableStrategyExecutor) else None
        return StrategyContext(
            inference_at=inference_at,
            bars=data.market.visible_at(inference_at),
            account=account,
            snapshot_dir=view.snapshot_dir,
            asof_dir=view.asof_dir,
            asof_version=view.asof_version,
            state_dir=fittable.context_state_dir if fittable is not None else "",
            models_dir=fittable.context_models_dir if fittable is not None else "",
            _nl_query=data.nl_query,
        )

    def _executor(self, view: StrategyDataView, state_dir: Path | None) -> StrategyExecutor:
        if self.executor_factory is not None:
            return self.executor_factory(self.strategy_path, self.sandbox, view, state_dir, self.models_dir)
        return DockerStrategyExecutor(
            self.strategy_path,
            self.sandbox,
            snapshot_dir=view.snapshot_dir or None,
            asof_dir=view.asof_dir or None,
            models_dir=self.models_dir,
            state_dir=state_dir,
            # The book replays a frozen strategy with no Agent and no contract
            # text in the loop, so its pinned image has to bake this checkout's
            # strategy runtime and nothing more: a README-only rebuild must not
            # stop a book from deciding its session.
            agent_contract=False,
        )

    def _journaled_orders(self, trade_date: str) -> list[dict[str, object]]:
        rows, _skipped = read_jsonl(self.state_root / f"orders_{trade_date}.jsonl")
        hidden = {"kind", "event_id", REFERENCE_KEY}
        return [{key: value for key, value in row.items() if key not in hidden} for row in rows]

    # ---------------------------------------------------------- fit state

    def _decision_state_dir(
        self, state: dict[str, object], *, fit_due: bool, trade_date: str
    ) -> tuple[Path | None, bool]:
        """(state directory the worker mounts, whether it is a staged copy).

        A due fit writes into a staged copy of the current state, published
        only after the decision succeeds, so a failed fit never damages the
        state the book's earlier decisions were taken with.
        """

        if self.fit_schedule is None:
            return None, False
        root = self.state_root / STRATEGY_STATE_NAME
        current = root / str(state["fit_state"]) if state["fit_state"] else None
        if current is not None and not current.is_dir():
            raise PaperEngineError(f"persisted fitted state is missing: {current}")
        if not fit_due:
            if current is None:
                raise PaperEngineError("the book has no fitted state and no fit is due")
            return current, False
        root.mkdir(parents=True, exist_ok=True)
        staged = root / f".{trade_date}.{uuid.uuid4().hex}.tmp"
        if current is not None:
            shutil.copytree(current, staged)
        else:
            staged.mkdir()
        # The sandbox fit worker runs as a non-root user and overwrites files
        # the host copied; the inference worker still binds it read-only.
        _open_tree(staged)
        return staged, True

    def _publish_fit_state(self, staged: Path, trade_date: str) -> None:
        target = self.state_root / STRATEGY_STATE_NAME / trade_date
        if target.exists():
            _remove_tree(target)  # left by a run that crashed before its checkpoint
        staged.rename(target)

    def _prune_fit_states(self, state: dict[str, object]) -> list[str]:
        """Remove superseded fitted states; a failure is reported, not fatal,
        because the decision it follows is already journaled."""

        root = self.state_root / STRATEGY_STATE_NAME
        if not root.is_dir():
            return []
        warnings = []
        for entry in root.iterdir():
            if entry.name == state["fit_state"]:
                continue
            try:
                _remove_tree(entry)
            except OSError as exc:
                warnings.append(f"cannot remove superseded fitted state {entry.name}: {exc}")
        return warnings

    # ------------------------------------------------------------ storage

    @contextmanager
    def _exclusive_lock(self):
        lock_path = self.state_root / PAPER_LOCK_NAME
        with lock_path.open("a+b") as stream:
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise PaperWriterBusy("another Paper writer owns this account") from exc
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
            "start_date": "",
            "settled_through": "",
            "decisions": [],
            "last_fit_date": "",
            "fit_state": "",
            "pending_orders": [],
            "account": None,
            "emissions": [],
            "warnings": [],
            "last_error": "",
        }

    def _load_state(self) -> dict[str, object]:
        path = self.state_root / PAPER_STATE_NAME
        if not path.exists():
            markers = [
                item.name for item in self.state_root.iterdir()
                if item.name == SNAPSHOT_NAME
                or item.name.startswith(("orders_", "executions_", "corporate_actions_"))
                or item.name == "equity_daily.jsonl"
            ]
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
        decisions = state.get("decisions") or []
        payload: dict[str, object] = {
            # generated_at is the console's staleness clock: a snapshot without
            # it reads as an unusable account state, not as a fresh one.
            "generated_at": datetime.now(CN_TZ).isoformat(),
            "source": PAPER_SOURCE, "ok": ok,
            "trade_date": decisions[-1]["trade_date"] if decisions else None,
            "settled_through": state.get("settled_through") or None,
            # The latest decided session's orders wait for its bars.
            "day_complete": False,
            "phase": "decided" if decisions else "not_started",
            "strategy_revision": self.strategy_revision, "cash": broker.cash,
            "equity": broker.equity(), "positions": [asdict(position) for _, position in sorted(broker.positions.items())],
            "pending_order_count": len(state.get("pending_orders") or []),
        }
        if error:
            payload["error"] = error
        write_json_atomic(self.state_root / SNAPSHOT_NAME, payload)

    def _summary(self, state: dict[str, object]) -> dict[str, object]:
        account = state.get("account") if isinstance(state.get("account"), dict) else {}
        positions = account.get("positions") or []
        decisions = state.get("decisions") or []
        return {
            "trade_date": decisions[-1]["trade_date"] if decisions else None,
            "start_date": state.get("start_date") or None,
            "settled_through": state.get("settled_through") or None,
            "strategy_revision": state.get("strategy_revision"),
            "cash": account.get("cash", self.profile.initial_cash),
            "equity": account.get("cash", self.profile.initial_cash)
            + sum(float(row["quantity"]) * float(row["last_price"]) for row in positions),
            "position_count": len(positions),
            "pending_order_count": len(state.get("pending_orders") or []),
            "last_fit_date": state.get("last_fit_date") or None,
            "decision": decisions[-1] if decisions else None,
            "warnings": list(state.get("warnings") or []),
        }


def _date(value: str) -> date:
    return date(int(value[:4]), int(value[4:6]), int(value[6:]))


def _open_tree(root: Path) -> None:
    for directory, _dirnames, filenames in os.walk(root):
        Path(directory).chmod(0o777)
        for name in filenames:
            (Path(directory) / name).chmod(0o666)


def _remove_tree(root: Path) -> None:
    """Remove a tree this book owns. Only directories are made writable: a
    file's mode is its inode's, so it is never changed on the way out."""

    for directory, _dirnames, _filenames in os.walk(root):
        try:
            Path(directory).chmod(0o755)
        except OSError:
            pass  # a directory the sandbox user created; rmtree reports it
    shutil.rmtree(root)


__all__ = [
    "PAPER_LOCK_NAME",
    "PAPER_STATE_NAME",
    "REFERENCE_KEY",
    "SNAPSHOT_NAME",
    "DailyPaperEngine",
    "PaperData",
    "PaperDataNotReady",
    "PaperEngineError",
    "PaperWriterBusy",
]
