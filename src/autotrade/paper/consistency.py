"""Paper against a historical replay of the same book: the consistency check.

A book is one research replay stretched over the real calendar
(``paper.engine``), so each session it settled should open, order, fill and
close as a replay of its frozen strategy does from its first decision, with
its schedule, Broker profile, sandbox and models, through the function behind
its Held-out record (``PITDailyEvaluationBackend.evaluate``, on the newest
research release or a named one). The replay reads history as any replay
reads it today, so a difference is what history-replays see that Paper did not.

Each settled session is compared at four levels, in order (``LEVELS``): the
ex-date settlements of its opening, the orders that fell due (the order half
of each execution, as a multiset), the fills of the orders both sides hold,
and its close (cash, equity, the holdings rebuilt from the journals). Prices
and quantities must be equal, money within ``fills.CENT``. Only the first
difference is explained, and every later session is downstream of it:

- an opening, fill or close is ``data_revised`` when what it was settled from
  moved (the reference price or dividend, the fill's bar, a close) and
  ``rule`` otherwise;
- orders are ``ex_date_timing`` when the replay settled an ex-date of a held
  name before that decision (a replay opens the day before it decides, Paper
  after: accepted), ``nondeterministic`` when a second replay disagrees with
  the first there, and with ``paper_mode`` ``paper_decision_path`` when the
  Paper engine over the same decisions and release disagrees with the replay
  there, ``data_revised`` when it agrees; else ``unresolved``, with the mounted
  datasets restated since Paper decided;
- a cause reached by elimination (``data_revised``, ``rule``, ``unresolved``)
  is ``code_changed`` unless the book decided and settled every session
  through that day on the check's own commit (``engine.code_revision``): a
  replay on today's code cannot say what other code did. A dirty tree is no
  commit's code, and a session that names none (every session Paper ran
  before it recorded its code) ran on other code as far as the check can tell.

A real-fill book is compared until its account leaves the simulation
(``partial``). A check that cannot run is ``failed`` with its reason; none
that compared no session is ``consistent``. The check holds the writer lock
only to read the journals, works under ``PAPER_CHECK_DIR``, writes only the
book's ``consistency.json`` (the latest result), nothing for a book deleted
while it ran, and never feeds the verdict.
"""

from __future__ import annotations

import json
from bisect import bisect_left
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter

from autotrade.common.records import read_json_object, write_json_atomic
from autotrade.common.time import CN_TZ
from autotrade.environment.artifacts import ArtifactRevision
from autotrade.environment.data.contracts import (
    dataset_vintages,
    read_committed_raw_generation,
)
from autotrade.environment.executor import TrustedStrategyExecutor
from autotrade.environment.sandbox import remove_sandbox_tree
from autotrade.environment.strategy import StrategyOrder
from autotrade.pipelines.calendar import anchor_before
from autotrade.pipelines.config import FORWARD_PHASE, EvaluationRequest
from autotrade.pipelines.pit_backend import (
    PITDailyEvaluationBackend,
    ResearchPITSnapshotProvider,
    required_release_raw_datasets,
)

from .book import BOOK_NAME, Book, verify_book_artifact
from .engine import (
    DIRTY,
    PAPER_STATE_NAME,
    REFERENCE_KEY,
    DailyPaperEngine,
    code_revision,
    docker_executor,
    writer_lock,
)
from .fills import CENT, CORRECTIONS_NAME, _same, mode
from .pit import PIT_CACHE_NAME, book_data_factory
from .storage import (
    CORPORATE_ACTIONS_PREFIX,
    EQUITY_JOURNAL_NAME,
    EXECUTIONS_PREFIX,
    ORDERS_PREFIX,
    journal_dates,
    journal_name,
    read_jsonl,
)

CONSISTENCY_NAME = "consistency.json"
CONSISTENCY_SCHEMA_VERSION = 1
# Each book's work directory, under the repository root: the pinned release
# and its views (``pit``, removed after every check, kept by the release
# census while it is there), the replays' results (kept until the next check)
# and the Paper-mode run's account (removed).
PAPER_CHECK_DIR = Path(".runtime/paper_check")
PAPER_MODE_NAME = "paper_mode"
LEVELS = ("opening", "orders", "fills", "close")
# The causes the check reaches by elimination, each assuming Paper ran the
# code the check runs.
_ELIMINATED = frozenset({"data_revised", "rule", "unresolved"})
# Journal bookkeeping, not account content.
_BOOKKEEPING = frozenset({"kind", "event_id", REFERENCE_KEY, "recorded"})
_SETTLEMENT_INPUTS = ("last_close", "pre_close", "cash_per_share")
_MONEY = frozenset(
    {"cash_credit", "dividend_tax_withheld", "commission", "stamp_duty", "dividend_tax", "realized_pnl"}
)
# The orders a probe has due on one session.
OrdersDue = Callable[[str], list[dict[str, object]]]


class CheckFailed(RuntimeError):
    """The book cannot be compared; the message is the result's reason."""


@dataclass(frozen=True)
class BookRecords:
    """One snapshot of a book's state and journals, bookkeeping removed."""

    state: Mapping[str, object]
    orders: Mapping[str, list[dict[str, object]]]
    executions: Mapping[str, list[dict[str, object]]]
    corporate_actions: Mapping[str, list[dict[str, object]]]
    equity: list[dict[str, object]]
    corrections: list[dict[str, object]]


def read_book_records(root: Path) -> BookRecords:
    """The book's state and journals, read whole under its writer lock."""

    root = Path(root)
    with writer_lock(root):
        state = read_json_object(root / PAPER_STATE_NAME, missing_ok=True)
        families = {
            prefix: {day: _journal(root / journal_name(prefix, day)) for day in journal_dates(root, prefix)}
            for prefix in (ORDERS_PREFIX, EXECUTIONS_PREFIX, CORPORATE_ACTIONS_PREFIX)
        }
        equity = _journal(root / EQUITY_JOURNAL_NAME)
        corrections = _journal(root / CORRECTIONS_NAME)
    if state.get("emissions"):
        raise CheckFailed("the book's journals are behind its state; its next Paper run completes them")
    return BookRecords(
        state, families[ORDERS_PREFIX], families[EXECUTIONS_PREFIX], families[CORPORATE_ACTIONS_PREFIX],
        equity, corrections,
    )


def _journal(path: Path) -> list[dict[str, object]]:
    rows, skipped = read_jsonl(path)
    if skipped:
        raise CheckFailed(f"{path.name} has {skipped} unreadable line(s)")
    return [{key: value for key, value in row.items() if key not in _BOOKKEEPING} for row in rows]


# ------------------------------------------------------------- comparison


def compare(
    records: BookRecords,
    replay: Mapping[str, object],
    *,
    vintages: Mapping[str, str],
    code: str | None,
    repeat: OrdersDue | None = None,
    paper_mode: OrdersDue | None = None,
) -> dict[str, object]:
    """The book's settled sessions against ``replay`` (a ``result.json``
    record): the status, a row per session compared, and the first difference
    with its cause. ``vintages`` are the check's release's and ``code`` its
    ``code_revision``; ``repeat`` and ``paper_mode`` are asked only for a
    first difference in the orders."""

    curve = list(replay["equity_curve"])
    days = [str(row["trade_date"]) for row in curve]
    settled = [str(row["trade_date"]) for row in records.equity]
    if not settled:
        raise CheckFailed("the book has no settled session to compare")
    if days != settled:
        raise CheckFailed(f"the replay covers {_span(days)} but the book settled {_span(settled)}")
    diverged = _real_fills_diverge(records)
    actions = _by_day(replay["corporate_actions"], lambda row: str(row["trade_date"]))
    due = _by_day(replay["executions"], lambda row: _due_session(str(row["execute_at"]), days))
    held: dict[str, int] = {}
    rows: list[dict[str, object]] = []
    first: dict[str, object] | None = None
    probes = {"repeat": "not_run", "paper_mode": "not_run"}
    for day, close, equity in zip(days, curve, records.equity, strict=True):
        if diverged is not None and day >= diverged:
            break
        paper_actions, paper_fills = records.corporate_actions.get(day, []), records.executions.get(day, [])
        _rebuild(held, paper_actions, paper_fills)
        if len(held) != equity.get("position_count"):
            raise CheckFailed(
                f"{day}: the journals rebuild {len(held)} holdings, the equity journal records "
                f"{equity.get('position_count')}"
            )
        replay_fills = due.get(day, [])
        paper_opening, replay_opening = _by_symbol(paper_actions), _by_symbol(actions.get(day, []))
        pairs = _pairs(paper_fills, replay_fills)
        replay_held = {str(symbol): int(quantity) for symbol, quantity in dict(close["positions"]).items()}
        # Each delta is the replay's figure less Paper's.
        row = {
            "trade_date": day,
            "opening": _same_rows(paper_opening, replay_opening),
            "orders": _orders(paper_fills) == _orders(replay_fills),
            "fills": all(_same_record(paper, other) for paper, other in pairs),
            "cash_delta": float(close["cash"]) - float(equity["cash"]),
            "equity_delta": float(close["equity"]) - float(equity["equity"]),
            "holdings": sorted(s for s in held.keys() | replay_held.keys() if held.get(s) != replay_held.get(s)),
        }
        row["close"] = abs(row["cash_delta"]) < CENT and abs(row["equity_delta"]) < CENT and not row["holdings"]
        row["downstream"] = first is not None
        level = next((name for name in LEVELS if not row[name]), None)
        if first is None and level is not None:
            if level == "opening":
                explained = _opening_cause(paper_opening, replay_opening)
            elif level == "orders":
                explained = _orders_cause(
                    records, day, paper_fills, replay_fills, actions, vintages, repeat, paper_mode, probes
                )
            elif level == "fills":
                paper, other = next(pair for pair in pairs if not _same_record(*pair))
                met = paper["price"] != other["price"] or paper["status"] != other["status"]
                explained = {"cause": "data_revised" if met else "rule", "evidence": {"paper": paper, "replay": other}}
            else:
                restated = not row["holdings"] and abs(row["cash_delta"]) < CENT
                explained = {"cause": "data_revised" if restated else "rule", "evidence": {
                    "cash": [equity["cash"], close["cash"]], "equity": [equity["equity"], close["equity"]],
                    "holdings": {s: [held.get(s), replay_held.get(s)] for s in row["holdings"]},
                }}
            first = {"trade_date": day, "level": level, **_code_changed(explained, records, day, code)}
        rows.append(row)
    if first is not None:
        status, reason = "differs", None
    elif diverged is not None:
        status = "partial"
        reason = f"its account follows the owner's fills, which leave the simulation on {diverged}"
    else:
        status, reason = "consistent", None
    return {
        "status": status,
        "reason": reason,
        "sessions_compared": len(rows),
        "sessions": rows,
        "first_difference": first,
        "probes": probes,
    }


def _code_changed(
    explained: dict[str, object], records: BookRecords, day: str, code: str | None
) -> dict[str, object]:
    """``explained``, unless its cause was reached by elimination and the book
    decided or settled a session through ``day`` on anything but ``code`` as a
    clean commit: then ``code_changed``, with the cause it would otherwise
    have and the revisions Paper ran (None where it recorded none)."""

    ran_on = {
        row.get("code_revision")
        for row in (*(records.state.get("decisions") or ()), *records.equity)
        if str(row["trade_date"]) <= day
    }
    clean = code is not None and not code.endswith(DIRTY)
    other = sorted((revision for revision in ran_on if not clean or revision != code), key=lambda name: name or "")
    if explained["cause"] not in _ELIMINATED or not other:
        return explained
    return {"cause": "code_changed", "evidence": {
        **dict(explained["evidence"]), "otherwise": explained["cause"], "paper_code_revisions": other,
    }}


def _opening_cause(paper: dict[str, dict], replay: dict[str, dict]) -> dict[str, object]:
    inputs = paper.keys() != replay.keys() or any(
        paper[symbol][field] != replay[symbol][field] for symbol in paper for field in _SETTLEMENT_INPUTS
    )
    differing = sorted(symbol for symbol in paper.keys() | replay.keys() if paper.get(symbol) != replay.get(symbol))
    return {
        "cause": "data_revised" if inputs else "rule",
        "evidence": {"paper": [paper.get(s) for s in differing], "replay": [replay.get(s) for s in differing]},
    }


def _orders_cause(
    records: BookRecords,
    day: str,
    paper_fills: list[dict[str, object]],
    replay_fills: list[dict[str, object]],
    actions: Mapping[str, list[dict[str, object]]],
    vintages: Mapping[str, str],
    repeat: OrdersDue | None,
    paper_mode: OrdersDue | None,
    probes: dict[str, str],
) -> dict[str, object]:
    paper, replay = _orders(paper_fills), _orders(replay_fills)
    paper_only, replay_only = paper - replay, replay - paper
    decisions = {str(row["trade_date"]): row for row in records.state.get("decisions") or ()}
    # The decision behind the first order Paper alone holds, or else the last
    # one before the session.
    journaled = [
        decided for decided, rows in sorted(records.orders.items())
        if decided <= day and paper_only & _orders(_order_parts(rows))
    ]
    decision = journaled[0] if journaled else max((d for d in decisions if d <= day), default=day)
    row = decisions.get(decision, {})
    evidence = {
        "decision": decision,
        "decided_at": row.get("decided_at"),
        "generation_id": row.get("generation_id"),
        "paper_only": _listed(paper_only),
        "replay_only": _listed(replay_only),
    }
    if actions.get(decision):
        return {"cause": "ex_date_timing", "evidence": {**evidence, "corporate_actions": actions[decision]}}
    if repeat is not None:
        again = _orders(repeat(day))
        probes["repeat"] = "equal" if again == replay else "differs"
        if again != replay:
            return {"cause": "nondeterministic", "evidence": {
                **evidence, "repeat_only": _listed(again - replay),
            }}
    if paper_mode is not None:
        engine = _orders(paper_mode(day))
        probes["paper_mode"] = "equal" if engine == replay else "differs"
        return {"cause": "data_revised" if engine == replay else "paper_decision_path", "evidence": {
            **evidence, "paper_mode_only": _listed(engine - replay),
        }}
    paper_vintages = dict(row.get("vintages") or {})
    restated = sorted(name for name, vintage in paper_vintages.items() if vintages.get(name, vintage) != vintage)
    return {"cause": "unresolved", "evidence": {**evidence, "restated": restated}}


def _rebuild(held: dict[str, int], actions: Iterable[Mapping], executions: Iterable[Mapping]) -> None:
    """Carry the holdings through one session's journals."""

    for action in actions:
        held[str(action["symbol"])] = int(action["quantity_after"])
    for execution in executions:
        if execution["status"] == "filled":
            sign = 1 if execution["action"] == "buy" else -1
            held[str(execution["symbol"])] = held.get(str(execution["symbol"]), 0) + sign * int(execution["quantity"])
    for symbol in [symbol for symbol, quantity in held.items() if quantity == 0]:
        del held[symbol]


def _real_fills_diverge(records: BookRecords) -> str | None:
    """The first session a real-fill book's account left the simulation: a
    recorded outcome other than the simulated one, or a correction."""

    current = mode(records.state)
    if current is None:
        return None
    days = [
        day for day, ledger in dict(current["sessions"]).items()
        if any(not _same(entry["real"], entry["simulated"]) for entry in ledger.values())
    ]
    return min([*days, *(str(row["trade_date"]) for row in records.corrections)], default=None)


def _order_part(execution: Mapping[str, object]) -> dict[str, object]:
    """The order an execution carries out, field for field."""

    return {
        "symbol": execution["symbol"], "action": execution["action"], "quantity": execution["quantity"],
        "execute_at": execution["execute_at"], "session": execution.get("session"),
        "limit_price": execution.get("limit_price"), "metadata": dict(execution.get("metadata") or {}),
    }


def _order_parts(journal: Iterable[Mapping[str, object]]) -> list[dict[str, object]]:
    """The orders of an orders journal, in the form an execution carries them."""

    orders = [
        StrategyOrder.from_record(row, inference_at=datetime.fromisoformat(str(row["execute_at"]))) for row in journal
    ]
    return [_order_part({**order.to_record(), "metadata": dict(order.metadata)}) for order in orders]


def _key(row: Mapping[str, object]) -> str:
    return json.dumps(_order_part(row), sort_keys=True, ensure_ascii=False)


def _orders(rows: Iterable[Mapping[str, object]]) -> Counter[str]:
    return Counter(_key(row) for row in rows)


def _listed(orders: Counter[str]) -> list[dict[str, object]]:
    return [json.loads(key) for key in sorted(orders.elements())]


def _pairs(paper: list[dict], replay: list[dict]) -> list[tuple[dict, dict]]:
    """Each order both sides hold, Paper's execution beside the replay's, in order."""

    pending: dict[str, list[dict]] = {}
    for row in replay:
        pending.setdefault(_key(row), []).append(row)
    return [(row, pending[_key(row)].pop(0)) for row in paper if pending.get(_key(row))]


def _same_record(paper: Mapping[str, object], replay: Mapping[str, object]) -> bool:
    if paper.keys() != replay.keys():
        return False
    for field, value in paper.items():
        other = replay[field]
        if field in _MONEY and value is not None and other is not None:
            if abs(float(value) - float(other)) >= CENT:
                return False
        elif value != other:
            return False
    return True


def _same_rows(paper: Mapping[str, dict], replay: Mapping[str, dict]) -> bool:
    return paper.keys() == replay.keys() and all(_same_record(paper[key], replay[key]) for key in paper)


def _by_symbol(rows: Iterable[Mapping[str, object]]) -> dict[str, dict]:
    return {str(row["symbol"]): dict(row) for row in rows}


def _by_day(rows: Iterable[Mapping[str, object]], day: Callable[[Mapping], str]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(day(row), []).append(dict(row))
    return grouped


def _due_session(execute_at: str, sessions: Sequence[str]) -> str:
    """The session that settles an order: its own day's, or the next one's."""

    day = datetime.fromisoformat(execute_at).astimezone(CN_TZ).strftime("%Y%m%d")
    index = bisect_left(sessions, day)
    return sessions[index] if index < len(sessions) else day


def _span(days: Sequence[str]) -> str:
    return f"{days[0]}..{days[-1]} ({len(days)} sessions)" if days else "no session"


# ------------------------------------------------------------------- check


def check_book(
    book: Book,
    *,
    work_dir: Path,
    take_devices: Callable[[], tuple[int, ...] | None] = lambda: None,
    generation_id: str | None = None,
    paper_mode: bool = False,
    updater_schedule: Mapping[str, object] | None = None,
    execution_mode: str = "sandbox",
) -> dict[str, object]:
    """Check ``book``, write the result to its ``consistency.json`` and return
    it; a book deleted meanwhile gets no file and a ``failed`` result.
    ``take_devices`` gives the cards the replays attach, asked once the
    views are built; ``generation_id`` names a published release to replay on
    instead of the newest; ``paper_mode`` runs the Paper engine, under the
    updater's ``updater_schedule``, when a decision differs."""

    started, work_dir = perf_counter(), Path(work_dir)
    record: dict[str, object] = {
        "schema_version": CONSISTENCY_SCHEMA_VERSION,
        "checked_at": datetime.now(CN_TZ).isoformat(timespec="seconds"),
        "code_revision": code_revision(),
        "artifact_id": book.artifact_id,
    }
    try:
        records = read_book_records(book.root)
        start, end = str(records.state.get("start_date") or ""), str(records.state.get("settled_through") or "")
        if not end:
            raise CheckFailed("the book has no settled session to compare")
        verify_book_artifact(book)
        _clear(work_dir, book)
        try:
            provider = ResearchPITSnapshotProvider(
                experiment_dir=work_dir / PIT_CACHE_NAME, raw_dir=book.raw_dir,
                fundamental_events_root=book.fundamental_events_root,
                fundamental_events_status=book.fundamental_events_status,
                config=book.snapshot_config, generation_id=generation_id,
            )
            release = provider.release.generation_id
            vintages = dataset_vintages(
                read_committed_raw_generation(provider.release.raw_dir),
                required_release_raw_datasets(book.snapshot_config),
            )
            bundle = provider.prepare(phase=FORWARD_PHASE, start=start, end=end, decision_time=anchor_before(start))
            devices = take_devices()
            backend = PITDailyEvaluationBackend(
                work_dir / "results", execution_mode=execution_mode, sandbox=book.sandbox,
                max_intraday_row_group_rows=book.max_intraday_row_group_rows, benchmark_index=book.benchmark_index,
            )
            request = EvaluationRequest(
                ArtifactRevision(book.artifact_id, book.strategy_path.parent, book.models_dir),
                bundle, FORWARD_PHASE, start, end, book.schedule, book.profile, gpu_devices=devices,
            )
            result = Path(backend.evaluate(request).result_ref)
            replay = read_json_object(result)
            days = [str(row["trade_date"]) for row in replay["equity_curve"]]

            def due(rows: Iterable[Mapping[str, object]], day: str) -> list[dict[str, object]]:
                return [dict(row) for row in rows if _due_session(str(row["execute_at"]), days) == day]

            def repeat(day: str) -> list[dict[str, object]]:
                return due(read_json_object(backend.evaluate(request, end_day=day).result_ref)["executions"], day)

            def engine(day: str) -> list[dict[str, object]]:
                # Paper's decision path on the replay's data: the book's
                # decisions through ``day`` again, from an empty account.
                root = work_dir / PAPER_MODE_NAME

                def executor(path, sandbox, view, state_dir, models_dir):
                    if execution_mode == "trusted":
                        return TrustedStrategyExecutor.from_path(path, state_dir=state_dir, models_dir=models_dir)
                    return docker_executor(path, sandbox.on_devices(devices), view, state_dir, models_dir)

                paper = DailyPaperEngine(
                    strategy_path=book.strategy_path, strategy_revision=book.artifact_id, state_root=root,
                    data_factory=book_data_factory(
                        book, updater_schedule or {}, state_root=root, generation_id=release or None
                    ),
                    models_dir=book.models_dir, schedule=book.schedule, profile=book.profile, sandbox=book.sandbox,
                    executor_factory=executor,
                )
                for row in records.state["decisions"]:
                    if str(row["trade_date"]) <= day:
                        paper.run_day(str(row["trade_date"]))
                journals = (_journal(root / journal_name(ORDERS_PREFIX, d)) for d in journal_dates(root, ORDERS_PREFIX))
                return due((order for rows in journals for order in _order_parts(rows)), day)

            record.update(
                compare(
                    records, replay, vintages=vintages, code=record["code_revision"], repeat=repeat,
                    paper_mode=engine if paper_mode else None,
                ),
                window={"start": start, "end": end},
                release={"generation_id": release, "vintages": vintages},
                result=str(result),
            )
        finally:
            for name in (PIT_CACHE_NAME, PAPER_MODE_NAME):
                _clear(work_dir / name, book)
    except Exception as exc:  # noqa: BLE001 - every failure is the result's reason
        record.update(status="failed", reason=f"{type(exc).__name__}: {exc}")
    record["wall_seconds"] = round(perf_counter() - started, 1)
    if not (book.root / BOOK_NAME).is_file():
        # The write would re-create the deleted book's directory, and a new
        # book cannot take its name while that holds a file (``create_book``).
        record.update(
            status="failed", reason=f"the book {book.root.name} was deleted during its check; no result is written"
        )
        return record
    write_json_atomic(book.root / CONSISTENCY_NAME, record, private=True, sort_keys=False)
    return record


def _clear(path: Path, book: Book) -> None:
    if not remove_sandbox_tree(path, book.sandbox.spec):
        raise CheckFailed(f"cannot remove the check's work directory {path}")


__all__ = [
    "CONSISTENCY_NAME",
    "PAPER_CHECK_DIR",
    "BookRecords",
    "CheckFailed",
    "check_book",
    "compare",
    "read_book_records",
]
