"""The Paper verdict: does a book's active return hold up on days nobody saw?

The evidence is the book's daily unregressed active return on Paper: the
account's daily return minus the mean of the zero-skill panel drawn on the
book's own Paper fills -- for a real-fill book the fills its owner recorded,
corrections included (``replay/null_control.run_null_control``, the panel
research grades against), over the bars, ex-dates, index and membership of the
book's newest replay slot. The days begin at the book's first Paper settlement:
the forward year selected the book, so it is never read into the statistic.

A book is ``observing`` until one terminal transition, ``confirmed`` or
``killed``, appended once to ``verdict.jsonl`` (``book.book_status``). The
statistical checks run only at checkpoints, every ``checkpoint_days`` settled
sessions, each on the days through it and each once: kill when the
``kill_confidence`` upper bound of the annualised active mean is below zero,
confirm when the ``confirm_confidence`` lower bound is above it (F8's bootstrap,
``pipelines/verdict.bootstrap_lower_bound``, seeded by the artifact id). A
drawdown past the arm's pinned ``max_drawdown`` (account) or
``active_max_drawdown`` (active series) kills on any day. ``verdict.json``
holds the latest reading. The rules are the ones the book pinned at creation.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from autotrade.environment.data.snapshot import load_snapshot_manifest
from autotrade.environment.replay.null_control import run_null_control
from autotrade.environment.replay.stats import (
    TRADING_DAYS_PER_YEAR,
    ReplayResult,
    compounded_path,
    compounded_return,
)
from autotrade.environment.replay.style import (
    daily_returns_from_curve,
    slot_benchmark,
    slot_membership,
)
from autotrade.environment.strategy import CN_TZ
from autotrade.pipelines.pit_backend import _span_daily, load_slot_corporate_actions
from autotrade.pipelines.verdict import (
    BOOTSTRAP_BLOCK_DAYS,
    bootstrap_lower_bound,
    daily_mean,
)

from .book import VERDICT_LOG_NAME, Book, book_status
from .engine import PAPER_STATE_NAME, writer_lock
from .fills import CORRECTIONS_NAME, key_text, mode, parse_key
from .pit import newest_replay_slot
from .storage import append_jsonl_once, read_json, read_jsonl, write_json_atomic

VERDICT_NAME = "verdict.json"
VERDICT_SCHEMA_VERSION = 1


def paper_reading(book: Book) -> dict[str, object]:
    """The book's verdict reading from its journals and newest replay slot.

    Pure: nothing is written (``record_verdict`` writes). A book with no
    settled session reads as observing with nothing measured.
    """

    equity, executions = _journals(book.root)
    previous = read_json(book.root / VERDICT_NAME)
    if not equity:
        return evidence_reading(book, equity, executions, None, {}, previous=previous)
    slot = newest_replay_slot(book.root)
    if slot is None:
        raise FileNotFoundError(f"{book.root.name} has settled sessions but no replay slot to read them by")
    first, last = str(equity[0]["trade_date"]), str(equity[-1]["trade_date"])
    # The decision view beside the slot holds the index sections dated before
    # the book's first session, which its first round trips are matched on.
    decision = (slot.parents[2] / "decision" / slot.name.split("_", 2)[2]).resolve(strict=True)
    return evidence_reading(
        book,
        equity,
        executions,
        _span_daily([slot], first, last),
        slot_benchmark(slot, benchmark_index=book.benchmark_index),
        corporate_actions=load_slot_corporate_actions(slot, load_snapshot_manifest(slot)),
        membership=slot_membership((decision, slot), benchmark_index=book.benchmark_index),
        previous=previous,
    )


def evidence_reading(
    book: Book,
    equity: Sequence[Mapping[str, object]],
    executions: Sequence[Mapping[str, object]],
    frame: pd.DataFrame | None,
    benchmark: Mapping[str, float],
    *,
    corporate_actions: pd.DataFrame | None = None,
    membership: Mapping[str, Collection[str]] | None = None,
    previous: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """The reading of ``equity`` (the book's settled days, in order) and its
    fills against ``frame``'s bars of exactly those days.

    ``previous`` is the last reading: a checkpoint it already checked is never
    checked again, though the panel drawn on a longer window differs.
    """

    rules = book.verdict_rules
    checkpoint = int(rules["checkpoint_days"])
    stored = book_status(book.root)
    transition = _stored_transition(book.root) if stored != "observing" else None
    checked = int((previous or {}).get("checked_through") or 0)
    dates = [str(row["trade_date"]) for row in equity]
    reading: dict[str, object] = {
        "schema_version": VERDICT_SCHEMA_VERSION,
        "computed_at": datetime.now(CN_TZ).isoformat(timespec="seconds"),
        "book_id": book.root.name,
        "experiment_id": book.experiment_id,
        "artifact_id": book.artifact_id,
        "track": book.candidate_source,
        "rules": dict(rules),
        "first_day": dates[0] if dates else None,
        "last_day": dates[-1] if dates else None,
        "days": len(dates),
    }
    if not dates:
        return {
            **reading, "status": stored, "transition": transition, "checked_through": checked,
            "next_checkpoint": checkpoint, "active_mean": None, "confirm_lower_bound": None,
            "kill_upper_bound": None, "account_return": None, "index_return": None,
            "account_max_drawdown": None, "active_max_drawdown": None, "panel": None,
        }
    if len(set(dates)) != len(dates) or dates != sorted(dates):
        raise ValueError(f"{book.root.name}'s equity journal does not hold each settled day once, in order")
    # A replay's first row carries the capital it opened with; the account's
    # first return is measured from it as the panel's is.
    curve = [
        {"trade_date": day, "equity": float(row["equity"]), **({"initial_equity": book.profile.initial_cash} if index == 0 else {})}
        for index, (day, row) in enumerate(zip(dates, equity, strict=True))
    ]
    panel = run_null_control(
        ReplayResult(equity_curve=tuple(curve), executions=tuple(executions), inference_dates=(), pending_orders=()),
        frame,
        benchmark,
        book.profile,
        book.schedule,
        seed=int(rules["panel_seed"]),
        corporate_actions=corporate_actions,
        membership=membership,
        panel=True,
    )
    account = daily_returns_from_curve(curve)
    zero_skill = {str(day): float(value) for day, value in panel["panel_daily"]}
    if [day for day, _value in account] != dates or list(zero_skill) != dates:
        raise ValueError(f"{book.root.name}'s account and zero-skill panel measured different days")
    missing = [day for day in dates if day not in benchmark]
    if missing:
        raise ValueError(f"the benchmark {book.benchmark_index} has no return on {missing[0]}")
    account_returns = np.asarray([value for _day, value in account])
    active = account_returns - np.asarray([zero_skill[day] for day in dates])

    if transition is None:
        transition = _first_transition(book, dates, account_returns, active, checked)
    last_checkpoint = len(dates) // checkpoint * checkpoint
    lower, upper = _bounds(active, book)
    return {
        **reading,
        "status": transition["status"] if transition else "observing",
        "transition": transition,
        "checked_through": max(checked, last_checkpoint),
        "next_checkpoint": last_checkpoint + checkpoint,
        "active_mean": float(active.mean()) * TRADING_DAYS_PER_YEAR,
        "confirm_lower_bound": lower,
        "kill_upper_bound": upper,
        "account_return": float(compounded_return(account_returns)),
        "index_return": float(compounded_return(benchmark[day] for day in dates)),
        "account_max_drawdown": float(max(drawdown for _equity, drawdown in compounded_path(account_returns))),
        "active_max_drawdown": float(max(drawdown for _equity, drawdown in compounded_path(active))),
        "panel": {
            key: panel.get(key) for key in ("k", "seed", "matched", "round_trips", "dropped_trips_mean", "reason")
        },
    }


def record_verdict(book: Book) -> dict[str, object]:
    """Read the book's verdict under its writer lock, append a new terminal
    transition to ``verdict.jsonl`` and write the reading to ``verdict.json``."""

    with writer_lock(book.root):
        reading = paper_reading(book)
        transition = reading["transition"]
        if transition is not None and book_status(book.root) == "observing":
            append_jsonl_once(book.root / VERDICT_LOG_NAME, transition)
        write_json_atomic(book.root / VERDICT_NAME, reading)
    return reading


def _first_transition(
    book: Book, dates: Sequence[str], account: np.ndarray, active: np.ndarray, checked: int
) -> dict[str, object] | None:
    """The first terminal event on the book's path, or None.

    Drawdowns are read every day; a checkpoint only once, past ``checked``.
    """

    rules = book.verdict_rules
    checkpoint = int(rules["checkpoint_days"])
    account_path = compounded_path(account)
    active_path = compounded_path(active)
    for index, day in enumerate(dates):
        days = index + 1
        status = reason = None
        if account_path[index][1] > float(rules["max_drawdown"]):
            status, reason = "killed", "account_drawdown_exceeded"
        elif active_path[index][1] > float(rules["active_max_drawdown"]):
            status, reason = "killed", "active_drawdown_exceeded"
        elif days % checkpoint == 0 and days > checked:
            lower, upper = _bounds(active[:days], book)
            if upper < 0:
                status, reason = "killed", "kill_upper_bound_below_zero"
            elif lower > 0:
                status, reason = "confirmed", "confirm_lower_bound_above_zero"
        if status is not None:
            lower, upper = _bounds(active[:days], book)
            return {
                "event_id": "terminal",
                "status": status,
                "reason": reason,
                "date": day,
                "days": days,
                "recorded_at": datetime.now(CN_TZ).isoformat(timespec="seconds"),
                "active_mean": float(active[:days].mean()) * TRADING_DAYS_PER_YEAR,
                "confirm_lower_bound": lower,
                "kill_upper_bound": upper,
                "account_drawdown": float(account_path[index][1]),
                "active_drawdown": float(active_path[index][1]),
            }
    return None


def _bounds(active: np.ndarray, book: Book) -> tuple[float | None, float | None]:
    """The confirm lower bound and the kill upper bound of the annualised
    active mean; None before one bootstrap block of days. The upper bound is
    the lower bound of the negated series on the same resampled days."""

    if len(active) < BOOTSTRAP_BLOCK_DAYS:
        return None, None
    rules = book.verdict_rules
    lower = bootstrap_lower_bound(
        active, book.artifact_id, confidence=float(rules["confirm_confidence"]), statistic=daily_mean
    )
    upper = -bootstrap_lower_bound(
        -active, book.artifact_id, confidence=float(rules["kill_confidence"]), statistic=daily_mean
    )
    return lower, upper


def _journals(root: Path) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """The settled days and the fills the book holds, from its journals.

    A real-fill book journals each session's recorded outcomes. A later
    correction of a settled session is booked on the session that settles it
    and leaves that session's journal as it was (``paper/fills.py``), so the
    fills of each corrected key are replaced by the outcome the book's ledger
    now holds, at that session's own time. The account's marks between the two
    sessions are not restated, as the account itself never restates them.
    """

    equity, skipped = read_jsonl(root / "equity_daily.jsonl")
    corrections, damaged = read_jsonl(root / CORRECTIONS_NAME)
    skipped += damaged
    executions: list[dict[str, object]] = []
    for path in sorted(root.glob("executions_*.jsonl")):
        rows, damaged = read_jsonl(path)
        skipped += damaged
        executions += [row for row in rows if row.get("kind") == "execution"]
    if skipped:
        raise ValueError(f"{root.name} has {skipped} damaged journal lines; its verdict cannot be read")
    if corrections:
        ledger = mode(read_json(root / PAPER_STATE_NAME))["sessions"]
        executions = _corrected(executions, {(str(row["corrects"]), str(row["key"])) for row in corrections}, ledger)
    return sorted(equity, key=lambda row: str(row["trade_date"])), executions


def _corrected(
    executions: Sequence[Mapping[str, object]],
    corrected: Collection[tuple[str, str]],
    ledger: Mapping[str, Mapping[str, Mapping[str, object]]],
) -> list[dict[str, object]]:
    """``executions`` with the fills of each corrected ``(session, key)`` replaced,
    where its first row stood, by the one fill the ledger's outcome holds."""

    held: dict[tuple[str, str], list[dict[str, object]]] = {}
    for session, key in corrected:
        outcome = ledger[session][key]["real"]
        clock, action, symbol = parse_key(key)
        at = datetime.strptime(f"{session} {clock}", "%Y%m%d %H:%M").replace(tzinfo=CN_TZ).isoformat()
        held[(session, key)] = [
            {
                "kind": "execution", "symbol": symbol, "action": action, "quantity": int(outcome["quantity"]),
                "price": float(outcome["price"]), "execute_at": at, "matched_at": at, "status": "filled",
            }
        ] if outcome["quantity"] else []
    result: list[dict[str, object]] = []
    for row in executions:
        at = datetime.fromisoformat(str(row["execute_at"])).astimezone(CN_TZ)
        identity = (at.strftime("%Y%m%d"), key_text(at.strftime("%H:%M"), str(row["action"]), str(row["symbol"])))
        if identity in held:
            result += held.pop(identity)
        elif identity not in corrected:
            result.append(dict(row))
    # A key the session's journal never held (a trade recorded only later).
    return result + [row for rows in held.values() for row in rows]


def _stored_transition(root: Path) -> dict[str, object]:
    rows, _skipped = read_jsonl(root / VERDICT_LOG_NAME)
    return rows[0]


__all__ = ["VERDICT_NAME", "evidence_reading", "paper_reading", "record_verdict"]
