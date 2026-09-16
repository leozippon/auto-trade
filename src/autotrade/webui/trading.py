"""Read-only projection of the local daily Paper books, one function per page panel.

Books sit side by side under the Paper state root, one directory each
(``paper.books``). ``books_payload`` is the overview, one row per book. A book's
page reads it through six projections, each from the book's own files:
``book_status`` (the status ladder, and whether the session ahead already has
its order sheet), ``book_payload`` (identity, ``book.json``),
``signal_payload`` (the latest decision's order sheet), ``history_payload``
(every earlier day's order sheet and fills), ``performance_payload`` (return against
CSI 300, equity and cash tracks, statistics, and the source experiment's
out-of-sample curve the book copied at creation) and ``snapshot_payload``
(account and positions). ``health_payload`` is the external probe over every book.

Every function is total: degradation is a structured payload state
(absent / no_snapshot / unreadable / export_error / stale / ok), never a 500.
Redaction is whitelist projection — payloads are assembled from named scalar
fields only, so nothing the writer happens to add can leak through. Each
figure is computed once, by the code that already defines it: the order sheet
by ``paper.orders.order_sheet``, returns and statistics by the replay reducer,
the CSI 300 series by the style sidecar's slot reader.

Timestamps: the Paper engine persists Asia/Shanghai stamps (frozen contract).
This module is the single normalization boundary for the snapshot clock — a
naive stamp is CN-local and is converted, never relabelled, so ``generated_at``
and ``age_seconds`` leave as UTC and the SPA renders UTC+8. Order/fill stamps
are already offset-aware and pass through as opaque display text.
"""

from __future__ import annotations

import json
import math
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pyarrow as pa

from autotrade.environment.replay.curve import curve_entry
from autotrade.environment.replay.stats import ReplayResult, compute_return_stats
from autotrade.environment.replay.style import (
    BENCHMARK_LABEL,
    daily_returns_from_curve,
    slot_benchmark,
)
from autotrade.environment.strategy import CN_TZ
from autotrade.paper.book import (
    BOOK_NAME,
    SOURCE_HISTORY_NAME,
    SOURCE_HISTORY_SCHEMA_VERSION,
)
from autotrade.paper.books import BOOK_ID_PATTERN, list_books
from autotrade.paper.engine import PAPER_STATE_NAME, SNAPSHOT_NAME
from autotrade.paper.orders import order_sheet
from autotrade.paper.pit import newest_replay_slot
from autotrade.paper.storage import read_jsonl

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
    if env not in TRADING_ENVS: raise KeyError(f"unknown trading environment: {env}")
    return Path(repo_root) / "data/trading/paper"


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
    if isinstance(value, bool) or not isinstance(value, (int, float)): return None
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
    """The book's frozen identity and where its calendar stands."""
    root = book_dir(repo_root, book, env)
    record, error = _read_json(root / BOOK_NAME)
    state, state_error = _read_json(root / PAPER_STATE_NAME)
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
                # Where the operator declares it: the opening call auction that
                # produces the fill price, or continuous trading.
                "window": _text(row["window"]),
                "symbol": _text(row["symbol"]),
                "name": _text(row["name"]),
                "action": _text(row["action"]),
                "quantity": _quantity(row["quantity"]),
                "reference_price": _number(row["reference_price"]),
                "notional": _number(row["notional"]),
            }
            for row in sheet["orders"]
        ],
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


def _fill(row: dict[str, object], names: dict[str, str | None]) -> dict[str, object]:
    symbol = _text(row.get("symbol"))
    commission, stamp_duty = _number(row.get("commission")), _number(row.get("stamp_duty"))
    return {
        "symbol": symbol,
        "name": names.get(symbol or ""),
        "action": _text(row.get("action")),
        "quantity": _quantity(row.get("quantity")),
        "matched_at": _text(row.get("matched_at")),
        "status": _text(row.get("status")),
        "price": _number(row.get("price")),
        "cost": None if commission is None and stamp_duty is None else (commission or 0.0) + (stamp_duty or 0.0),
        "reason": _text(row.get("reason")),
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
    rows = []
    for day in days:
        try:
            sheet = _sheet(root, state, day) if day in decided else None
        except (KeyError, TypeError, ValueError) as exc:
            return {"env": env, "state": "unreadable", "error": f"decision {day}: {exc}", "days": []}
        orders = sheet["orders"] if sheet else []
        names = {row["symbol"]: row["name"] for row in orders if row["symbol"]}
        fills, skipped = read_jsonl(root / f"executions_{day}.jsonl")
        rows.append({
            "trade_date": day,
            "orders": orders,
            # The same post-trade block the signal panel carries for today, so
            # a past day reads at the same level of detail. A day the book only
            # settled has no order sheet, and says so with null.
            "target": sheet["target"] if sheet else None,
            "cash_after": sheet["cash_after"] if sheet else None,
            "cash_weight": sheet["cash_weight"] if sheet else None,
            "fills": [_fill(row, names) for row in fills],
            "skipped_lines": skipped + (sheet["skipped_lines"] if sheet else 0),
        })
    return {"env": env, "state": "ok" if rows else "absent", "error": None, "days": rows}


# ---------------------------------------------------------- performance

def _day_before(yyyymmdd: str) -> str:
    return (date(int(yyyymmdd[:4]), int(yyyymmdd[4:6]), int(yyyymmdd[6:8])) - timedelta(days=1)).strftime("%Y%m%d")


def _benchmark(root: Path) -> tuple[dict[str, float], str | None]:
    """CSI 300 daily returns from the release the book's latest run pinned:
    the replay slot of that run, read as the research style sidecar reads it."""
    slot = newest_replay_slot(root)
    if slot is None:
        return {}, None
    try:
        daily = slot_benchmark(slot)
    except (OSError, ValueError, pa.ArrowException) as exc:  # a damaged cache file degrades the benchmark only
        return {}, f"{type(exc).__name__}: {exc}"
    # A slot that carries no index row at all is a broken read, not an empty
    # book: say so instead of leaving the panel to guess "no data".
    if not daily:
        return {}, f"replay slot {slot.name} carries no {BENCHMARK_LABEL} rows"
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


def performance_payload(repo_root: Path, book: str, env: str = "paper") -> dict[str, object]:
    """The book's return against CSI 300 over the same settled days, its
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
            "benchmark_error": None, "source": source, "source_error": source_error,
        }
    returns = daily_returns_from_curve(curve)
    daily, benchmark_error = _benchmark(root)
    benchmark_rows = [(day, daily[day]) for day, _value in returns if day in daily]
    # The book's starting point — initial cash, no return, the calendar day
    # before its first settled day — opens the curve, so one settled day is
    # already a segment and the benchmark starts at zero beside it.
    anchor = _day_before(str(curve[0]["trade_date"]))
    benchmark = (
        curve_entry("benchmark", BENCHMARK_LABEL, [(anchor, 0.0), *benchmark_rows])
        if benchmark_rows
        else None
    )
    executions = [row for day in _dates(root, "executions_") for row in read_jsonl(root / f"executions_{day}.jsonl")[0]]
    stats = compute_return_stats(
        ReplayResult(equity_curve=tuple(curve), executions=tuple(executions), inference_dates=(), pending_orders=())
    )
    covered = benchmark is not None and len(benchmark_rows) == len(returns)
    enough = len(curve) >= MIN_STATISTICS_DAYS
    total_return = _number(stats["total_return"])
    benchmark_return = _number(benchmark["final"]) if covered else None
    return {
        **base,
        "state": "ok",
        "error": None,
        "chart": {
            "series": [curve_entry("strategy", "模拟账户", [(anchor, 0.0), *returns])],
            "benchmark": benchmark,
            "account": {
                "dates": [anchor, *(row["trade_date"] for row in curve)],
                "equity": [initial, *(row["equity"] for row in curve)],
                "cash": [initial, *(row["cash"] for row in curve)],
            },
        },
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
    return {
        "env": env,
        "book_id": book,
        "state": _environment_state(str(snapshot["state"]), state_error, latest),
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
            "start_date": identity["start_date"],
            "initial_cash": (identity["book"] or {}).get("initial_cash"),
            "equity": account.get("equity"),
            "cash": account.get("cash"),
            "position_count": len(positions) if positions is not None else None,
            "total_return": statistics.get("total_return"),
            "excess_return": statistics.get("excess_return"),
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
