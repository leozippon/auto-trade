"""Read-only projection of the local daily Paper books, one function per page panel.

Books sit side by side under the Paper state root, one directory each
(``paper.books``). ``books_payload`` is the overview, one row per book. A book's
page reads it through eight projections, each from the book's own files —
``fills_payload`` (whether the book follows its owner's recorded fills, the
sessions still to record and every settled one against its simulated fill)
and these seven:
``book_status`` (the status ladder, and whether the session ahead already has
its order sheet), ``book_payload`` (identity, ``book.json``, and the stored Paper verdict),
``signal_payload`` (the latest decision's order sheet), ``history_payload``
(every earlier day's order sheet and fills), ``performance_payload`` (return against
the book's benchmark, equity and cash tracks, statistics, and the source experiment's
out-of-sample curve the book copied at creation), ``snapshot_payload``
(account and positions) and ``pnl_payload`` (the account's profit and loss and
each traded name's share of it). ``health_payload`` is the external probe over
every book.

Every function is total: degradation is a structured payload state
(absent / no_snapshot / unreadable / export_error / stale / ok), never a 500.
Redaction is whitelist projection — payloads are assembled from named scalar
fields only, so nothing the writer happens to add can leak through. Each
figure is computed once, by the code that already defines it: the order sheet
by ``paper.orders.order_sheet``, returns and statistics by the replay reducer,
the benchmark series by the style sidecar's slot reader.

Timestamps: the Paper engine persists Asia/Shanghai stamps (frozen contract).
This module is the single normalization boundary for the snapshot clock — a
naive stamp is CN-local and is converted, never relabelled, so ``generated_at``
and ``age_seconds`` leave as UTC and the SPA renders UTC+8. Order/fill stamps
are already offset-aware and pass through as opaque display text.
"""

from __future__ import annotations

import json
import math
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path

import pyarrow as pa

from autotrade.environment.data.contracts import benchmark_index_label
from autotrade.environment.replay.curve import curve_entry
from autotrade.environment.replay.stats import ReplayResult, compute_return_stats
from autotrade.environment.replay.style import (
    BENCHMARK_TS_CODE,
    daily_returns_from_curve,
    slot_benchmark,
)
from autotrade.environment.strategy import CN_TZ
from autotrade.paper import fills
from autotrade.paper.book import (
    BOOK_NAME,
    SOURCE_HISTORY_NAME,
    SOURCE_HISTORY_SCHEMA_VERSION,
)
from autotrade.paper.books import BOOK_ID_PATTERN, PAPER_STATE_DIR, list_books
from autotrade.paper.engine import PAPER_STATE_NAME, REFERENCE_KEY, SNAPSHOT_NAME
from autotrade.paper.orders import (
    CONTINUOUS,
    LIMIT_GUIDANCE,
    order_sheet,
    order_window,
    window_text,
)
from autotrade.paper.pit import newest_replay_slot
from autotrade.paper.storage import read_jsonl
from autotrade.paper.verdict import VERDICT_NAME

TRADING_ENVS = ("paper",)
# A snapshot older than this is served but flagged: the account data is the
# last one the engine wrote, not the current one. The book writes once per
# weekday before the open, so a weekend gap (~72 h) is normal and four days
# means a scheduled run did not happen (a failed run shows export_error).
STALE_SNAPSHOT_ALERT_SECONDS = 4 * 24 * 3600.0
# Annualised return, Sharpe and maximum drawdown are served from this many
# settled days; a shorter book reads "—" with its day count instead.
MIN_STATISTICS_DAYS = 20
EQUITY_JOURNAL_NAME = "equity_daily.jsonl"


def env_dir(repo_root: Path, env: str) -> Path:
    if env not in TRADING_ENVS:
        raise KeyError(f"unknown trading environment: {env}")
    return Path(repo_root) / PAPER_STATE_DIR


def book_dir(repo_root: Path, book: str, env: str = "paper") -> Path:
    """One book's directory; anything that is not a book under the root is unknown."""
    root = env_dir(repo_root, env)
    if not BOOK_ID_PATTERN.fullmatch(book) or ".." in book or not (root / book / BOOK_NAME).is_file():
        raise KeyError(f"unknown Paper book: {book}")
    return root / book


def _valid_date(value: str) -> bool:
    return len(value) == 8 and value.isdigit()


def _dates(root: Path, prefix: str) -> list[str]:
    return sorted(path.stem.removeprefix(prefix) for path in root.glob(f"{prefix}[0-9]*.jsonl") if _valid_date(path.stem.removeprefix(prefix)))


def _text(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _quantity(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None


def _count(value: object) -> int | None:
    """Position counters, where zero is meaningful (a fully T+1-locked line)."""
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _to_utc(value: object) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        moment = datetime.fromisoformat(value)
    except ValueError:
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=CN_TZ)  # writers store naive CN time
    return moment.astimezone(UTC)


def _utc_iso(moment: datetime | None) -> str | None:
    return None if moment is None else moment.isoformat().replace("+00:00", "Z")


def _age_seconds(moment: datetime | None) -> float | None:
    if moment is None:
        return None
    return round((datetime.now(UTC) - moment).total_seconds(), 1)


def _read_json(path: Path) -> tuple[dict[str, object] | None, str | None]:
    """(payload, error): (None, None) = missing, (None, msg) = unreadable."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None, None
    except (OSError, json.JSONDecodeError) as exc:
        return None, str(exc)
    if not isinstance(value, dict):
        return None, f"{path.name} must be a JSON object"
    return value, None


def _mapping(value: object) -> dict[str, object]:
    return value if isinstance(value, dict) else {}


def _decisions(state: dict[str, object] | None) -> list[dict[str, object]]:
    decisions = (state or {}).get("decisions")
    return [
        row
        for row in (decisions if isinstance(decisions, list) else ())
        if isinstance(row, dict) and isinstance(row.get("trade_date"), str)
    ]


def _decision_dates(state: dict[str, object] | None) -> list[str]:
    return [str(row["trade_date"]) for row in _decisions(state)]


def _today(state: dict[str, object] | None) -> dict[str, object]:
    """The book's latest order sheet, and whether it is the session ahead's.

    The book decides a session on that session's own morning, so a decision for
    the current Asia/Shanghai calendar day is the sheet the operator places at
    the next open. ``ready`` is that record's existence and nothing else: no
    deadline, no judgement about how long a run may still take. ``decided_at``
    is absent for a decision the engine took before it recorded one."""
    latest = (_decisions(state) or [{}])[-1]
    return {
        "trade_date": _text(latest.get("trade_date")),
        "decided_at": _utc_iso(_to_utc(latest.get("decided_at"))),
        "ready": latest.get("trade_date") == datetime.now(CN_TZ).strftime("%Y%m%d"),
    }


# ---------------------------------------------------------------- book

def book_payload(repo_root: Path, book: str, env: str = "paper") -> dict[str, object]:
    """The book's frozen identity and where its calendar stands, with the
    Paper verdict ``run_paper`` last stored, as written (None before the first
    run); the console reads it and never computes one."""
    root = book_dir(repo_root, book, env)
    record, error = _read_json(root / BOOK_NAME)
    state, state_error = _read_json(root / PAPER_STATE_NAME)
    verdict, verdict_error = _read_json(root / VERDICT_NAME)
    error = error or verdict_error
    status = "unreadable" if error or state_error else "ok" if record else "absent"
    identity = None
    if record:
        identity = {
            "experiment_id": _text(record.get("experiment_id")),
            "artifact_id": _text(record.get("artifact_id")),
            "candidate_source": _text(record.get("candidate_source")),
            "note": _text(record.get("note")),
            "initial_cash": _number(_mapping(record.get("profile")).get("initial_cash")),
        }
    state = state or {}
    return {
        "env": env,
        "state": status,
        "error": error or state_error,
        "book": identity,
        "verdict": verdict,
        "start_date": _text(state.get("start_date")),
        "settled_through": _text(state.get("settled_through")),
        "last_fit_date": _text(state.get("last_fit_date")),
    }


# ------------------------------------------------------- signal, history

def _sheet(root: Path, state: dict[str, object], trade_date: str) -> dict[str, object]:
    """One decision's order sheet, whitelisted."""
    sheet = order_sheet(root, trade_date, state=state)
    decision = sheet["decision"]
    return {
        "trade_date": trade_date,
        "inference_at": _text(decision.get("inference_at")),
        "data_through": _text(decision.get("data_through")),
        "generation_id": _text(decision.get("generation_id")),
        "fitted": decision.get("fitted") is True,
        "replayed_calls": _count(decision.get("replayed_calls")),
        "replayed_matching_journal": _count(decision.get("replayed_matching_journal")),
        "orders": [
            {
                "execute_at": _text(row["execute_at"]),
                "session": _text(row["session"]),
                "time": _text(row["time"]),
                # Where the operator places it, worded once by the sheet: the
                # opening or closing call auction that sets the fill price, or
                # continuous trading at its minute.
                "window": _text(row["window"]),
                "window_label": _text(row["window_label"]),
                "symbol": _text(row["symbol"]),
                "name": _text(row["name"]),
                "action": _text(row["action"]),
                "quantity": _quantity(row["quantity"]),
                "reference_price": _number(row["reference_price"]),
                "notional": _number(row["notional"]),
            }
            for row in sheet["orders"]
        ],
        # The batches placed at one sitting each, in placement order.
        "groups": [
            {
                "window": _text(group["window"]),
                "title": _text(group["title"]),
                "how": _text(group["how"]),
                "orders": list(group["orders"]),
                "buy": _number(group["flows"]["buy"]),
                "sell": _number(group["flows"]["sell"]),
            }
            for group in sheet["groups"]
        ],
        "limit_guidance": LIMIT_GUIDANCE if any(row["window"] != CONTINUOUS for row in sheet["orders"]) else None,
        "target": [
            {
                "symbol": _text(row["symbol"]),
                "name": _text(row["name"]),
                "quantity": _quantity(row["quantity"]),
                "reference_price": _number(row["reference_price"]),
                "value": _number(row["value"]),
                "weight": _number(row["weight"]),
            }
            for row in sheet["target"]
        ],
        "cash_after": _number(sheet["cash_after"]),
        "cash_weight": _number(sheet["cash_weight"]),
        "skipped_lines": int(sheet["skipped_lines"]),
    }


def signal_payload(repo_root: Path, book: str, env: str = "paper") -> dict[str, object]:
    """The latest decision: its orders and the holdings once they fill."""
    root = book_dir(repo_root, book, env)
    state, error = _read_json(root / PAPER_STATE_NAME)
    decided = _decision_dates(state)
    if error or not decided:
        return {"env": env, "state": "unreadable" if error else "absent", "error": error, "signal": None}
    try:
        signal = _sheet(root, state, decided[-1])
    except (KeyError, TypeError, ValueError) as exc:
        return {"env": env, "state": "unreadable", "error": f"decision {decided[-1]}: {exc}", "signal": None}
    return {"env": env, "state": "ok", "error": None, "signal": signal}


def _dividend_tax(row: dict[str, object]) -> float | None:
    """The dividend tax a sale paid; a row journaled before the Broker charged
    it carries no key and paid none."""

    return _number(row.get("dividend_tax", 0.0))


def _fill(row: dict[str, object], names: dict[str, str | None]) -> dict[str, object]:
    symbol = _text(row.get("symbol"))
    commission, stamp_duty = _number(row.get("commission")), _number(row.get("stamp_duty"))
    dividend_tax = _dividend_tax(row)
    return {
        "symbol": symbol,
        "name": names.get(symbol or ""),
        "action": _text(row.get("action")),
        "quantity": _quantity(row.get("quantity")),
        "matched_at": _text(row.get("matched_at")),
        "status": _text(row.get("status")),
        "price": _number(row.get("price")),
        "cost": None
        if commission is None and stamp_duty is None
        else (commission or 0.0) + (stamp_duty or 0.0) + (dividend_tax or 0.0),
        # Inside ``cost``; stated apart because it is the holding-period tax on
        # dividends the sold shares received, not a trading fee.
        "dividend_tax": dividend_tax,
        "reason": _text(row.get("reason")),
        # A trade the owner recorded that no order asked for.
        "off_sheet": row.get("off_sheet") is True,
    }


# ------------------------------------------------------- real fills

def _outcome(value: object) -> dict[str, object]:
    row = _mapping(value)
    commission, stamp = _number(row.get("commission")) or 0.0, _number(row.get("stamp_duty")) or 0.0
    tax = _dividend_tax(row) or 0.0
    return {
        "quantity": _count(row.get("quantity")),
        "price": _number(row.get("price")),
        "fees": commission + stamp,
        "dividend_tax": tax,
        "reason": _text(row.get("reason")),
    }


def _key_view(day: str, key: str, names: dict[str, str], references: dict[str, object]) -> dict[str, object]:
    clock, action, symbol = fills.parse_key(key)
    execute_at = datetime.combine(date(int(day[:4]), int(day[4:6]), int(day[6:])), time.fromisoformat(clock), tzinfo=CN_TZ)
    return {
        "key": key, "time": clock, "window": order_window(execute_at),
        "window_label": window_text(execute_at, day)[0], "symbol": symbol,
        "name": names.get(symbol), "action": action, "reference_price": _number(references.get(symbol)),
    }


def _record_view(record: fills.Record | None) -> dict[str, object] | None:
    if record is None:
        return None
    return {
        "outcome": record.outcome, "quantity": record.quantity or None, "price": _number(record.price),
        "commission": _number(record.commission), "recorded_at": _utc_iso(_to_utc(record.recorded_at)),
    }


def _settled_rows(day: str, ledger: dict[str, object], names: dict[str, str]) -> list[dict[str, object]]:
    """One settled real-fill session, key by key: what the simulator filled,
    what the owner recorded, and what the difference cost."""
    rows = []
    for key, entry in ledger.items():
        compared = fills.compare(key, entry)
        rows.append({
            **_key_view(day, key, names, {}),
            "ordered": _count(_mapping(entry).get("ordered")),
            "simulated": _outcome(_mapping(entry).get("simulated")),
            "real": _outcome(_mapping(entry).get("real")),
            "status": compared["status"],
            "price_bp": _number(compared["price_bp"]),
            # To the cent: a sub-cent difference is no difference.
            **{
                name: None if compared[name] is None else round(compared[name], 2) + 0.0
                for name in ("price_cost", "fee_cost", "missed_notional")
            },
        })
    return sorted(rows, key=lambda row: row["time"])


def _equity_gaps(root: Path) -> dict[str, float]:
    """Per real-fill session: the simulated account's close less the real one."""
    gaps = {}
    for row in read_jsonl(root / EQUITY_JOURNAL_NAME)[0]:
        day, simulated, real = _text(row.get("trade_date")), _number(row.get("simulated_equity")), _number(row.get("equity"))
        if day and simulated is not None and real is not None:
            gaps[day] = simulated - real
    return gaps


def _session_closed(day: str) -> bool:
    """A session's outcome can be recorded once its closing auction is over."""
    now = datetime.now(CN_TZ)
    today = now.strftime("%Y%m%d")
    return day < today or (day == today and now.time() >= time(15, 0))


def _open_sessions(root: Path, state: dict[str, object], records: fills.Records) -> list[dict[str, object]]:
    """The sessions the next run settles from records, oldest first: each
    order key due on it (and any trade recorded off the sheet) with what the
    owner recorded and what it resolves to."""
    current = fills.mode(state)
    settled_through = str(state.get("settled_through") or "")
    after = max(str(current["after"]) if current else settled_through, settled_through)
    ordered: dict[str, dict[str, int]] = {}
    for order in fills.pending_orders(state):
        day = fills.order_session(order)
        if day > after:
            keys = ordered.setdefault(day, {})
            keys[fills.order_key(order)] = keys.get(fills.order_key(order), 0) + order.quantity
    days = set(ordered) | {day for day in _decision_dates(state) if day > after}
    days |= {day for day, _key in records.fills if day > after} | {day for day in records.sessions if day > after}
    late = fills.post_hoc_sessions(state)
    names = _names(root)
    sessions = []
    for day in sorted(days):
        references = {
            str(row.get("symbol")): _mapping(row.get(REFERENCE_KEY)).get("close")
            for row in read_jsonl(root / f"orders_{day}.jsonl")[0]
        }
        keys = ordered.get(day, {})
        rows = []
        for key in [*keys, *(key for key in records.keys(day) if key not in keys)]:
            resolved = fills.resolve(records, day, key, ordered=key in keys, post_hoc=day in late)
            rows.append({
                **_key_view(day, key, names, references),
                "ordered": keys.get(key, 0),
                "record": _record_view(records.fills.get((day, key))),
                "resolved": resolved.outcome if resolved is not None else None,
            })
        session = records.sessions.get(day)
        unresolved = sum(1 for row in rows if row["ordered"] and row["resolved"] is None)
        sessions.append({
            "trade_date": day, "settled": False, "closed": _session_closed(day), "post_hoc": day in late,
            "default": session.outcome if session is not None else None,
            "unresolved": unresolved, "rows": sorted(rows, key=lambda row: row["time"]),
        })
    return sessions


def _awaiting(sessions: list[dict[str, object]]) -> list[str]:
    """Closed sessions with orders the owner has not recorded: the next run
    stops on them."""
    return [row["trade_date"] for row in sessions if row["closed"] and row["unresolved"]]


def _awaiting_sessions(root: Path, state: dict[str, object] | None) -> list[str]:
    try:
        records = fills.read_records(root)
    except ValueError:
        return []  # the fills panel says why
    state = state or {}
    if fills.mode(state) is None and records.enabled_at is None:
        return []
    return _awaiting(_open_sessions(root, state, records))


def fills_payload(repo_root: Path, book: str, env: str = "paper") -> dict[str, object]:
    """Whether the book follows real fills, and the record behind it: the
    sessions still to record, every settled one key by key against its
    simulated fill, and what manual execution has cost in total."""
    root = book_dir(repo_root, book, env)
    base: dict[str, object] = {
        "env": env, "state": "ok", "error": None, "mode": "off", "after": None, "enabled_at": None,
        "awaiting": [], "sessions": [], "totals": None, "skipped_lines": 0,
    }
    state, error = _read_json(root / PAPER_STATE_NAME)
    try:
        records = fills.read_records(root)
    except ValueError as exc:
        return {**base, "state": "unreadable", "error": str(exc)}
    if error:
        return {**base, "state": "unreadable", "error": error}
    state = state or {}
    current = fills.mode(state)
    if current is None and records.enabled_at is None:
        return base
    open_sessions = _open_sessions(root, state, records)
    names = _names(root)
    gaps = _equity_gaps(root)
    settled = [
        {"trade_date": day, "settled": True, "gap": gaps.get(day), "rows": _settled_rows(day, ledger, names)}
        for day, ledger in sorted(_mapping((current or {}).get("sessions")).items(), reverse=True)
        if ledger
    ]
    compared = [row for session in settled for row in session["rows"]]
    corrections = read_jsonl(root / fills.CORRECTIONS_NAME)[0]
    return {
        **base,
        "mode": "on" if current is not None else "pending",
        "after": _text((current or {}).get("after")) if current is not None else _text(state.get("settled_through")),
        "enabled_at": _utc_iso(_to_utc(records.enabled_at)),
        "awaiting": _awaiting(open_sessions),
        "sessions": [*reversed(open_sessions), *settled],
        "totals": {
            "sessions": len(gaps),
            # The two curves' gap summed over the sessions: what every
            # difference (price, fee, missed or extra trade, correction) cost.
            "execution_cost": sum(gaps.values()),
            "price_cost": sum(row["price_cost"] or 0.0 for row in compared),
            "fee_cost": sum(row["fee_cost"] or 0.0 for row in compared),
            "missed": sum(1 for row in compared if row["status"] in {"missed", "partial"}),
            "missed_notional": sum(row["missed_notional"] or 0.0 for row in compared),
            "off_sheet": sum(1 for row in compared if row["status"] == "off_sheet"),
            "corrections": len(corrections),
        },
        "skipped_lines": records.skipped,
    }


def history_payload(repo_root: Path, book: str, env: str = "paper") -> dict[str, object]:
    """Every trading day the book has acted on, newest first: that morning's
    order sheet — its orders and the holdings they fill into — and the fills it
    settled into (both keyed by the session). The latest decision is the signal
    panel's until its fills exist."""
    root = book_dir(repo_root, book, env)
    state, error = _read_json(root / PAPER_STATE_NAME)
    if error:
        return {"env": env, "state": "unreadable", "error": error, "days": []}
    decided = _decision_dates(state)
    days = sorted(set(decided[:-1]) | set(_dates(root, "executions_")), reverse=True)
    # A real-fill book's settled sessions, key by key against the simulator,
    # and the corrections each later session booked.
    ledgers = _mapping(_mapping(fills.mode(state or {})).get("sessions"))
    gaps = _equity_gaps(root) if ledgers else {}
    booked: dict[str, list[dict[str, object]]] = {}
    for row in read_jsonl(root / fills.CORRECTIONS_NAME)[0] if ledgers else ():
        booked.setdefault(str(row.get("trade_date")), []).append({
            "corrects": _text(row.get("corrects")), "symbol": _text(row.get("symbol")),
            "action": _text(row.get("action")), "quantity": _number(row.get("quantity")),
            "cash": _number(row.get("cash")), "realized_pnl": _number(row.get("realized_pnl")),
        })
    all_names = _names(root) if ledgers else {}
    rows = []
    for day in days:
        try:
            sheet = _sheet(root, state, day) if day in decided else None
        except (KeyError, TypeError, ValueError) as exc:
            return {"env": env, "state": "unreadable", "error": f"decision {day}: {exc}", "days": []}
        orders = sheet["orders"] if sheet else []
        names = {**all_names, **{row["symbol"]: row["name"] for row in orders if row["symbol"]}}
        executed, skipped = read_jsonl(root / f"executions_{day}.jsonl")
        ledger = ledgers.get(day)
        rows.append({
            "trade_date": day,
            "orders": orders,
            "groups": sheet["groups"] if sheet else [],
            "limit_guidance": sheet["limit_guidance"] if sheet else None,
            # The same post-trade block the signal panel carries for today, so
            # a past day reads at the same level of detail. A day the book only
            # settled has no order sheet, and says so with null.
            "target": sheet["target"] if sheet else None,
            "cash_after": sheet["cash_after"] if sheet else None,
            "cash_weight": sheet["cash_weight"] if sheet else None,
            "fills": [_fill(row, names) for row in executed],
            "reconciliation": (
                {"gap": gaps.get(day), "rows": _settled_rows(day, _mapping(ledger), names)}
                if isinstance(ledger, dict) else None
            ),
            "corrections": booked.get(day, []),
            "skipped_lines": skipped + (sheet["skipped_lines"] if sheet else 0),
        })
    return {"env": env, "state": "ok" if rows else "absent", "error": None, "days": rows}


# ---------------------------------------------------------- performance

def _day_before(yyyymmdd: str) -> str:
    return (date(int(yyyymmdd[:4]), int(yyyymmdd[4:6]), int(yyyymmdd[6:8])) - timedelta(days=1)).strftime("%Y%m%d")


def _book_benchmark_index(root: Path) -> str:
    """The index this book is measured against, from its own record.

    Frozen at creation from the source arm's ``benchmark_index``; a book
    created before the parameter existed carries none and reads as the
    default, which is the index those arms ran on.
    """
    record, _error = _read_json(root / BOOK_NAME)
    code = _mapping(record).get("benchmark_index")
    return str(code) if isinstance(code, str) and code else BENCHMARK_TS_CODE


def _benchmark(root: Path) -> tuple[dict[str, float], str | None]:
    """The book's benchmark daily returns from the release its latest run
    pinned: the replay slot of that run, read as the research style sidecar
    reads it, keyed on the book's own index."""
    slot = newest_replay_slot(root)
    if slot is None:
        return {}, None
    index = _book_benchmark_index(root)
    try:
        daily = slot_benchmark(slot, benchmark_index=index)
    except (OSError, ValueError, pa.ArrowException) as exc:  # a damaged cache file degrades the benchmark only
        return {}, f"{type(exc).__name__}: {exc}"
    # A slot that carries no index row at all is a broken read, not an empty
    # book: say so instead of leaving the panel to guess "no data".
    if not daily:
        return {}, f"replay slot {slot.name} carries no {benchmark_index_label(index)} rows"
    return daily, None


def _curve_entry(value: object) -> dict[str, object] | None:
    """One compounded curve of the book's source history, whitelisted."""
    row = _mapping(value)
    dates = [day for day in row.get("dates") or () if isinstance(day, str) and _valid_date(day)]
    cumulative = [_number(item) for item in row.get("cum") or ()]
    drawdown = [_number(item) for item in row.get("drawdown") or ()]
    if not dates or len(cumulative) != len(dates) or len(drawdown) != len(dates):
        return None
    if any(item is None for item in (*cumulative, *drawdown)):
        return None
    return {
        "key": _text(row.get("key")),
        "label": _text(row.get("label")),
        "dates": dates,
        "cum": cumulative,
        "drawdown": drawdown,
        "final": _number(row.get("final")),
    }


def _source_history(root: Path) -> tuple[dict[str, object] | None, str | None]:
    """The out-of-sample curve the book copied out of its source experiment at
    creation. Absent for a book created before the copy existed, which is a
    missing history and not an error; anything else is reported."""
    record, error = _read_json(root / SOURCE_HISTORY_NAME)
    if record is None:
        return None, error
    if record.get("schema_version") != SOURCE_HISTORY_SCHEMA_VERSION:
        return None, f"unsupported {SOURCE_HISTORY_NAME} schema: {record.get('schema_version')}"
    series = [entry for value in record.get("series") or () if (entry := _curve_entry(value))]
    if not series:
        return None, f"{SOURCE_HISTORY_NAME} carries no daily returns"
    return {
        "experiment_id": _text(record.get("experiment_id")),
        "result": _text(record.get("result")),
        "heldout_start": _text(record.get("heldout_start")),
        "series": series,
        "benchmark": _curve_entry(record.get("benchmark")),
    }, None


def _simulated_returns(rows: list[dict[str, object]], initial: float) -> list[tuple[str, float]] | None:
    """The simulated track's daily returns, or None for a simulated book.

    Each real-fill session is simulated from the real account it opened with,
    so its return is that session's simulated close over the real close
    before it; a session settled before the switch is the account's own."""
    if not any(_number(row.get("simulated_equity")) is not None for row in rows):
        return None
    result, previous = [], initial
    for row in rows:
        day, equity = _text(row.get("trade_date")), _number(row.get("equity"))
        if not day or not _valid_date(day) or equity is None:
            continue
        simulated = _number(row.get("simulated_equity"))
        result.append((day, (simulated if simulated is not None else equity) / previous - 1.0))
        previous = equity
    return result


def performance_payload(repo_root: Path, book: str, env: str = "paper") -> dict[str, object]:
    """The book's return against its benchmark over the same settled days, its
    end-of-day equity and cash on those days, the replay statistics, and the
    source experiment's out-of-sample curve the book's own days continue."""
    root = book_dir(repo_root, book, env)
    record, error = _read_json(root / BOOK_NAME)
    initial = _number(_mapping(_mapping(record).get("profile")).get("initial_cash"))
    rows, _skipped = read_jsonl(root / EQUITY_JOURNAL_NAME)
    # One row per settled day: equity marked at that day's close and the cash
    # after that day's fills, as the engine journals them together.
    curve = [
        {"trade_date": day, "equity": equity, "cash": _number(row.get("cash")), "initial_equity": initial}
        for row in rows
        if (day := _text(row.get("trade_date"))) and _valid_date(day)
        and (equity := _number(row.get("equity"))) is not None
    ]
    base = {"env": env, "min_days": MIN_STATISTICS_DAYS}
    source, source_error = _source_history(root)
    if error or initial is None or initial <= 0 or not curve:
        state = "unreadable" if error else "absent"
        return {
            **base, "state": state, "error": error, "chart": None, "statistics": None,
            "benchmark_label": None, "benchmark_error": None, "source": source, "source_error": source_error,
        }
    returns = daily_returns_from_curve(curve)
    daily, benchmark_error = _benchmark(root)
    benchmark_rows = [(day, daily[day]) for day, _value in returns if day in daily]
    # The book's starting point — initial cash, no return, the calendar day
    # before its first settled day — opens the curve, so one settled day is
    # already a segment and the benchmark starts at zero beside it.
    anchor = _day_before(str(curve[0]["trade_date"]))
    label = benchmark_index_label(_book_benchmark_index(root))
    benchmark = (
        curve_entry("benchmark", label, [(anchor, 0.0), *benchmark_rows])
        if benchmark_rows
        else None
    )
    executions = [row for day in _dates(root, "executions_") for row in read_jsonl(root / f"executions_{day}.jsonl")[0]]
    stats = compute_return_stats(
        ReplayResult(equity_curve=tuple(curve), executions=tuple(executions), inference_dates=(), pending_orders=())
    )
    covered = benchmark is not None and len(benchmark_rows) == len(returns)
    enough = len(curve) >= MIN_STATISTICS_DAYS
    series = [curve_entry("strategy", "模拟账户", [(anchor, 0.0), *returns])]
    simulated = _simulated_returns(rows, initial)
    if simulated is not None:
        # A real-fill book: the account is the recorded one, and the second
        # line compounds what each session would have made filled as simulated.
        series = [
            curve_entry("strategy", "实际成交", [(anchor, 0.0), *returns]),
            curve_entry("simulated", "按模拟成交", [(anchor, 0.0), *simulated]),
        ]
    total_return = _number(stats["total_return"])
    benchmark_return = _number(benchmark["final"]) if covered else None
    return {
        **base,
        "state": "ok",
        "error": None,
        "chart": {
            "series": series,
            "benchmark": benchmark,
            "account": {
                "dates": [anchor, *(row["trade_date"] for row in curve)],
                "equity": [initial, *(row["equity"] for row in curve)],
                "cash": [initial, *(row["cash"] for row in curve)],
            },
        },
        # The book's own index by name, for the tiles and notes that speak of
        # it even when no benchmark row was read.
        "benchmark_label": label,
        "benchmark_days": len(benchmark_rows),
        "benchmark_error": benchmark_error,
        # The artifact's forward and Held-out replay, copied into the book when
        # it was created: the history its own days continue.
        "source": source,
        "source_error": source_error,
        "statistics": {
            "days": len(curve),
            "total_return": total_return,
            "benchmark_return": benchmark_return,
            "excess_return": total_return - benchmark_return
            if total_return is not None and benchmark_return is not None
            else None,
            "annualized_return": _number(stats["annualized_return"]) if enough else None,
            "sharpe": _number(stats["sharpe"]) if enough else None,
            "max_drawdown": _number(stats["max_drawdown"]) if enough else None,
            "turnover": _number(stats["turnover"]),
            "fees": _number(stats["fees_paid"]),
            "stamp_duty": _number(stats["stamp_duty_paid"]),
            "dividend_tax": _number(stats.get("dividend_tax_paid", 0.0)),
            "fills": int(_mapping(stats["order_status_counts"]).get("filled", 0)),
        },
    }


# ------------------------------------------------------------- snapshot

# Position rows are projected through one candidate list; a row where nothing
# maps is served as {"unmapped": true}, visible and never silently dropped.
_POSITION_FIELDS = (
    ("symbol", "symbol", _text),
    ("quantity", "quantity", _count),
    ("available_quantity", "available_quantity", _count),
    ("average_cost", "average_cost", _number),
    ("last_price", "last_price", _number),
)


def _project_position(record: dict[str, object], equity: float | None) -> dict[str, object]:
    row: dict[str, object] = {name: cast(record.get(key)) for name, key, cast in _POSITION_FIELDS}
    row["unmapped"] = all(value is None for value in row.values())
    quantity, cost, last = row["quantity"], row["average_cost"], row["last_price"]
    value = quantity * last if quantity is not None and last is not None else None
    row["market_value"] = value
    row["pnl"] = quantity * (last - cost) if value is not None and cost is not None else None
    row["weight"] = value / equity if value is not None and equity else None
    return row


def _project_snapshot(raw: dict[str, object]) -> dict[str, object]:
    positions = raw.get("positions") if isinstance(raw.get("positions"), list) else []
    equity = _number(raw.get("equity"))
    rows = [_project_position(row, equity) for row in positions if isinstance(row, dict)]
    return {
        "settled_through": _text(raw.get("settled_through")),
        "cash": _number(raw.get("cash")),
        "equity": equity,
        "market_value": sum(row["market_value"] or 0.0 for row in rows),
        "pending_order_count": _count(raw.get("pending_order_count")),
        "positions": rows,
    }


def _snapshot_status(directory: Path) -> dict[str, object]:
    """State precedence per the design: absent -> no_snapshot -> unreadable ->
    export_error -> stale -> ok. ``raw`` is the parsed snapshot kept for
    further whitelist projection; it is never serialized directly."""
    empty: dict[str, object] = {"raw": None, "generated_at": None, "age_seconds": None}
    if not directory.is_dir():
        return {"state": "absent", "error": None, **empty}
    raw, error = _read_json(directory / SNAPSHOT_NAME)
    if raw is None and error is None:
        return {"state": "no_snapshot", "error": None, **empty}
    if raw is None:
        return {"state": "unreadable", "error": error, **empty}
    generated = _to_utc(raw.get("generated_at"))
    base: dict[str, object] = {
        "raw": raw,
        "generated_at": _utc_iso(generated),
        "age_seconds": _age_seconds(generated),
    }
    if not raw.get("ok", True):
        # Only the first line of the writer's error leaves the API (the rest
        # may quote payloads); the SPA shows it in the red banner.
        lines = (_text(raw.get("error")) or "").splitlines()
        first = lines[0].strip() if lines else ""
        return {"state": "export_error", "error": first or "writer reported ok=false", **base}
    if generated is None:
        return {"state": "unreadable", "error": "snapshot generated_at missing or unparseable", **base}
    age = base["age_seconds"]
    if isinstance(age, float) and age > STALE_SNAPSHOT_ALERT_SECONDS:
        return {"state": "stale", "error": None, **base}
    return {"state": "ok", "error": None, **base}


def snapshot_payload(repo_root: Path, book: str, env: str = "paper") -> dict[str, object]:
    status = _snapshot_status(book_dir(repo_root, book, env))
    raw = status["raw"]
    return {
        "env": env,
        "state": status["state"],
        "error": status["error"],
        "generated_at": status["generated_at"],
        "age_seconds": status["age_seconds"],
        "stale_threshold_seconds": STALE_SNAPSHOT_ALERT_SECONDS,
        "snapshot": _project_snapshot(raw) if isinstance(raw, dict) else None,
    }


# ------------------------------------------------------------------ pnl

# What a name's fills and ex-dates moved, summed per symbol from the journals.
_FLOW_FIELDS = ("commission", "stamp_duty", "dividend_tax", "realized_pnl", "dividends")


def _journal_rows(root: Path, prefix: str) -> list[tuple[str, dict[str, object]]]:
    """Every row of one journal family, each with its file name. A damaged line
    raises: here it would silently shorten one name's P&L, not just a list."""
    rows = []
    for day in _dates(root, prefix):
        name = f"{prefix}{day}.jsonl"
        records, skipped = read_jsonl(root / name)
        if skipped:
            raise ValueError(f"{name} has {skipped} unreadable line(s)")
        rows.extend((name, record) for record in records)
    return rows


def _instrument_flows(root: Path) -> dict[str, dict[str, float]]:
    """Per symbol: the fees and stamp duty its fills paid, the realized P&L its
    sells journaled and the cash its ex-dates credited. Only a fill moves money;
    a filled row without a money field raises instead of counting as zero."""
    flows: dict[str, dict[str, float]] = {}

    def add(name: str, row: dict[str, object], **amounts: float | None) -> None:
        symbol = _text(row.get("symbol"))
        if symbol is None or None in amounts.values():
            raise ValueError(f"{name} has a row without its symbol or money fields")
        entry = flows.setdefault(symbol, dict.fromkeys(_FLOW_FIELDS, 0.0))
        for key, value in amounts.items():
            entry[key] += value

    for name, row in _journal_rows(root, "executions_"):
        if row.get("status") != "filled":
            continue
        realized = _number(row.get("realized_pnl")) if row.get("action") == "sell" else 0.0
        add(name, row, commission=_number(row.get("commission")),
            stamp_duty=_number(row.get("stamp_duty")), dividend_tax=_dividend_tax(row),
            realized_pnl=realized)
    for name, row in _journal_rows(root, "corporate_actions_"):
        add(name, row, dividends=_number(row.get("cash_credit")))
    # A real-fill book's corrections move the same money a fill does.
    corrections, skipped = read_jsonl(root / fills.CORRECTIONS_NAME)
    if skipped:
        raise ValueError(f"{fills.CORRECTIONS_NAME} has {skipped} unreadable line(s)")
    for row in corrections:
        add(fills.CORRECTIONS_NAME, row, commission=_number(row.get("commission")),
            stamp_duty=_number(row.get("stamp_duty")), dividend_tax=_dividend_tax(row),
            realized_pnl=_number(row.get("realized_pnl")))
    return flows


def _names(root: Path) -> dict[str, str]:
    """Security names from the reference quote on each order row: every name
    the book ever traded was ordered first."""
    names = {}
    for day in _dates(root, "orders_"):
        for row in read_jsonl(root / f"orders_{day}.jsonl")[0]:
            symbol, name = _text(row.get("symbol")), _text(_mapping(row.get(REFERENCE_KEY)).get("name"))
            if symbol and name:
                names[symbol] = name
    return names


def pnl_payload(repo_root: Path, book: str, env: str = "paper") -> dict[str, object]:
    """The account's profit and loss since the book opened, and each traded
    name's share of it, all from the Broker's own bookkeeping.

    The account total is the checkpoint's equity — cash plus every holding at
    the close of ``mark_date`` (a name with no bar that day keeps its last
    close) — less the initial cash; the engine has no deposits or withdrawals.
    A name's total is its unrealized P&L against the Broker's average cost plus
    the realized P&L its sells journaled. That cost carries the buy fees and is
    lowered by ex-date cash, and a sale's realized P&L is net of its fees, so
    the fees, stamp duty and dividends listed per name are already inside the
    total, and the names sum to the account total. ``residual`` is the cent
    amount they do not reconcile by: the engine books no cash that belongs to
    no name (no interest), so it is zero unless a journal and the checkpoint
    disagree. A journal that cannot be read in full withholds the per-name view
    with its reason; the account total does not depend on it."""
    root = book_dir(repo_root, book, env)
    record, error = _read_json(root / BOOK_NAME)
    state, state_error = _read_json(root / PAPER_STATE_NAME)
    initial = _number(_mapping(_mapping(record).get("profile")).get("initial_cash"))
    mark_date = _text(_mapping(state).get("settled_through"))
    base: dict[str, object] = {
        "env": env, "state": "ok", "error": None, "mark_date": mark_date, "initial_cash": initial,
        "equity": None, "total_pnl": None, "total_return": None,
        "instruments": None, "instruments_error": None, "instruments_pnl": None, "residual": None,
    }
    if error or state_error:
        return {**base, "state": "unreadable", "error": error or state_error}
    if not mark_date:  # never run, or nothing settled yet: nothing to measure
        return {**base, "state": "absent"}
    account = _mapping(_mapping(state).get("account"))
    cash, raw = _number(account.get("cash")), account.get("positions")
    if initial is None or initial <= 0 or cash is None or not isinstance(raw, list):
        return {**base, "state": "unreadable", "error": "no valid initial cash or account checkpoint"}
    positions = {}
    for item in raw:
        row = _mapping(item)
        symbol = _text(row.get("symbol"))
        held = (_quantity(row.get("quantity")), _number(row.get("average_cost")), _number(row.get("last_price")))
        if symbol is None or None in held:
            return {**base, "state": "unreadable", "error": "the account checkpoint has an invalid position"}
        positions[symbol] = held
    equity = cash + sum(quantity * last for quantity, _cost, last in positions.values())
    total = equity - initial
    account_view = {**base, "equity": equity, "total_pnl": total, "total_return": total / initial}
    try:
        flows = _instrument_flows(root)
    except ValueError as exc:
        return {**account_view, "instruments_error": str(exc)}
    names = _names(root)
    rows = []
    for symbol in sorted(positions.keys() | flows.keys()):
        flow = flows.get(symbol) or dict.fromkeys(_FLOW_FIELDS, 0.0)
        quantity, cost, last = positions.get(symbol) or (0, None, None)
        # A closed name has realized P&L only: no cost, mark or holding left.
        unrealized = quantity * (last - cost) if quantity else None
        rows.append({
            "symbol": symbol, "name": names.get(symbol), "quantity": quantity,
            "average_cost": cost, "last_price": last,
            "market_value": quantity * last if quantity else None, "unrealized_pnl": unrealized,
            **flow, "total_pnl": (unrealized or 0.0) + flow["realized_pnl"],
        })
    rows.sort(key=lambda row: row["total_pnl"], reverse=True)
    summed = sum(row["total_pnl"] for row in rows)
    # + 0.0 turns round()'s -0.0 into 0.0, which the page prints unsigned.
    return {**account_view, "instruments": rows, "instruments_pnl": summed, "residual": round(total - summed, 2) + 0.0}


# --------------------------------------------------------------- status

def _environment_state(snapshot: str, state_error: str | None, latest: str | None) -> str:
    """Snapshot precedence, widened to the book state the signal panels read,
    so nothing hides behind a healthy snapshot.

    Damaged journal lines are counted (``skipped_lines``) and the good records
    around them are still served, so they never reach this ladder: a truncated
    append is a report, not an unreadable environment."""
    if state_error or snapshot == "unreadable":
        return "unreadable"
    if snapshot in {"export_error", "stale"}:
        return snapshot
    if latest or snapshot == "ok":
        return "ok"
    # Initialized but never run: the directory exists, nothing has been written.
    return "no_snapshot" if snapshot == "no_snapshot" else "absent"


def book_status(repo_root: Path, book: str, env: str = "paper") -> dict[str, object]:
    root = book_dir(repo_root, book, env)
    snapshot = snapshot_payload(repo_root, book, env)
    state, state_error = _read_json(root / PAPER_STATE_NAME)
    latest = max([*_dates(root, "orders_")[-1:], *_dates(root, "executions_")[-1:]], default=None)
    environment = _environment_state(str(snapshot["state"]), state_error, latest)
    # A real-fill book whose closed session is not recorded yet: the next run
    # stops there, so it outranks every state but an unreadable one.
    awaiting = _awaiting_sessions(root, state) if environment != "unreadable" else []
    return {
        "env": env,
        "book_id": book,
        "state": "awaiting_fills" if awaiting else environment,
        "awaiting_fills": awaiting,
        "error": snapshot["error"] or state_error,
        "generated_at": snapshot["generated_at"],
        "age_seconds": snapshot["age_seconds"],
        # Whether the session ahead already has its order sheet, and when it
        # was written; the page shows it before the operator's own deadline.
        "today": _today(state),
        # Exported so the SPA can quote the alert threshold without
        # duplicating the constant client-side.
        "stale_threshold_seconds": STALE_SNAPSHOT_ALERT_SECONDS,
    }


def books_payload(repo_root: Path, env: str = "paper") -> dict[str, object]:
    """The overview: one row per book, each figure read off that book's panels."""
    try:
        books = list_books(env_dir(repo_root, env))
    except RuntimeError as exc:  # a root still in the single-book layout
        return {"env": env, "state": "unreadable", "error": str(exc), "books": []}
    rows = []
    for book in books:
        identity = book_payload(repo_root, book, env)
        signal = signal_payload(repo_root, book, env)["signal"]
        performance = performance_payload(repo_root, book, env)
        statistics = performance["statistics"] or {}
        chart = performance["chart"]
        account = snapshot_payload(repo_root, book, env)["snapshot"] or {}
        # The lines the book's own 当前持仓 panel lists; absent, not zero, when
        # no snapshot has been written yet.
        positions = account.get("positions")
        status = book_status(repo_root, book, env)
        rows.append({
            "book_id": book,
            "experiment_id": (identity["book"] or {}).get("experiment_id"),
            "artifact_id": (identity["book"] or {}).get("artifact_id"),
            "candidate_source": (identity["book"] or {}).get("candidate_source"),
            "verdict": identity["verdict"],
            "start_date": identity["start_date"],
            "initial_cash": (identity["book"] or {}).get("initial_cash"),
            "equity": account.get("equity"),
            "cash": account.get("cash"),
            "position_count": len(positions) if positions is not None else None,
            "total_return": statistics.get("total_return"),
            "excess_return": statistics.get("excess_return"),
            "benchmark_label": performance.get("benchmark_label"),
            # The card's miniature of the book's own return curve, the same
            # series its performance panel draws and absent on the same rule.
            "curve": {"series": chart["series"], "benchmark": chart["benchmark"]} if chart else None,
            # The card draws the same chained line the page does.
            "source": performance["source"],
            "signal_date": signal["trade_date"] if signal else None,
            "order_count": len(signal["orders"]) if signal else None,
            "today": status["today"],
            "state": status["state"],
            "error": status["error"],
        })
    return {"env": env, "state": "ok" if rows else "absent", "error": None, "books": rows}


def health_payload(repo_root: Path, env: str = "paper") -> dict[str, object]:
    """For external monitors: every book's status, ``ok`` unless one is unreadable."""
    try:
        books = [book_status(repo_root, book, env) for book in list_books(env_dir(repo_root, env))]
    except RuntimeError as exc:
        return {"ok": False, "env": env, "error": str(exc), "books": []}
    return {"ok": all(row["state"] != "unreadable" for row in books), "env": env, "error": None, "books": books}
