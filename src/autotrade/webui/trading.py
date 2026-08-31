"""Read-only projection of the local daily Paper account.

Every function is total: degradation is a structured payload state
(absent / no_snapshot / unreadable / export_error / stale / ok), never a 500.
Redaction is whitelist projection — payloads are assembled from named scalar
fields only, so nothing the writer happens to add can leak through.

Timestamps: the Paper engine persists Asia/Shanghai stamps (frozen contract).
This module is the single normalization boundary for the snapshot clock — a
naive stamp is CN-local and is converted, never relabelled, so ``generated_at``
and ``age_seconds`` leave as UTC and the SPA renders UTC+8. Order/fill stamps
are already offset-aware and pass through as opaque display text.
"""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from pathlib import Path

from autotrade.environment.strategy import CN_TZ
from autotrade.paper.engine import SNAPSHOT_NAME
from autotrade.paper.storage import read_jsonl

TRADING_ENVS = ("paper",)
ENV_LABELS = {"paper": "Paper 模拟"}
# A snapshot older than this is served but flagged: the account data is the
# last one the engine wrote, not the current one.
STALE_SNAPSHOT_ALERT_SECONDS = 180.0


def env_dir(repo_root: Path, env: str) -> Path:
    if env not in TRADING_ENVS: raise KeyError(f"unknown trading environment: {env}")
    return Path(repo_root) / "data/trading/paper"


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
        return None, "account snapshot must be a JSON object"
    return value, None


def _payload(repo_root: Path, env: str, kind: str, date: str | None) -> dict[str, object]:
    root = env_dir(repo_root, env)
    # The Paper engine journals matched fills as executions_<date>.jsonl;
    # the console serves them under the deals contract.
    prefix = "orders_" if kind == "orders" else "executions_"
    dates = _dates(root, prefix)
    selected = date or (dates[-1] if dates else None)
    rows, skipped = read_jsonl(root / f"{prefix}{selected}.jsonl") if selected else ([], 0)
    projected = []
    for row in rows:
        item = {
            "symbol": _text(row.get("symbol")), "action": _text(row.get("action")),
            "quantity": _quantity(row.get("quantity")), "execute_at": _text(row.get("execute_at")),
        }
        if kind == "deals":
            item.update({
                "matched_at": _text(row.get("matched_at")), "status": _text(row.get("status")),
                "price": _number(row.get("price")), "commission": _number(row.get("commission")),
                "stamp_duty": _number(row.get("stamp_duty")), "reason": _text(row.get("reason")),
            })
        projected.append(item)
    # A journal is an append-only stream a crash can truncate mid-line: the
    # damaged lines are counted and the good records before them still served,
    # so a skipped line is never a state.
    return {"env": env, "trade_date": selected, "available_dates": dates, "state": "ok" if selected else "absent", "skipped_lines": skipped, kind: projected, "count": len(projected)}


def orders_payload(repo_root: Path, env: str = "paper", date: str | None = None) -> dict[str, object]:
    return _payload(repo_root, env, "orders", date)


def deals_payload(repo_root: Path, env: str = "paper", date: str | None = None) -> dict[str, object]:
    return _payload(repo_root, env, "deals", date)


# Position rows are projected through one candidate list; a row where nothing
# maps is served as {"unmapped": true}, visible and never silently dropped.
_POSITION_FIELDS = (
    ("symbol", "symbol", _text),
    ("quantity", "quantity", _count),
    ("available_quantity", "available_quantity", _count),
    ("average_cost", "average_cost", _number),
    ("last_price", "last_price", _number),
)


def _project_position(record: dict[str, object]) -> dict[str, object]:
    row: dict[str, object] = {name: cast(record.get(key)) for name, key, cast in _POSITION_FIELDS}
    row["unmapped"] = all(value is None for value in row.values())
    return row


def _project_snapshot(raw: dict[str, object]) -> dict[str, object]:
    positions = raw.get("positions") if isinstance(raw.get("positions"), list) else []
    return {
        "source": _text(raw.get("source")),
        "trade_date": _text(raw.get("trade_date")),
        "day_complete": raw.get("day_complete") if isinstance(raw.get("day_complete"), bool) else None,
        "phase": _text(raw.get("phase")),
        "strategy_revision": _text(raw.get("strategy_revision")),
        "cash": _number(raw.get("cash")),
        "equity": _number(raw.get("equity")),
        "pending_order_count": _count(raw.get("pending_order_count")),
        "positions": [_project_position(row) for row in positions if isinstance(row, dict)],
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


def snapshot_payload(repo_root: Path, env: str = "paper") -> dict[str, object]:
    status = _snapshot_status(env_dir(repo_root, env))
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


def series_payload(repo_root: Path, env: str = "paper") -> dict[str, object]:
    rows, skipped = read_jsonl(env_dir(repo_root, env) / "equity_daily.jsonl")
    points = [{"trade_date": row.get("trade_date"), "equity": _number(row.get("equity")), "cash": _number(row.get("cash"))} for row in rows]
    return {"env": env, "state": "ok" if points else "absent", "skipped_lines": skipped, "series": points}


def _environment_state(orders: dict[str, object], deals: dict[str, object], snapshot: dict[str, object], latest: str | None) -> str:
    """Snapshot precedence, widened to any degraded reader rather than the
    snapshot alone, so nothing hides behind a healthy snapshot.

    Damaged journal lines are counted (``skipped_lines``) and the good records
    around them are still served, so they never reach this ladder: a truncated
    append is a report, not an unreadable environment."""
    if "unreadable" in {orders["state"], deals["state"], snapshot["state"]}:
        return "unreadable"
    if snapshot["state"] in {"export_error", "stale"}:
        return str(snapshot["state"])
    if latest or snapshot["state"] == "ok":
        return "ok"
    # Initialized but never run: the directory exists, nothing has been written.
    return "no_snapshot" if snapshot["state"] == "no_snapshot" else "absent"


def environment_summary(repo_root: Path, env: str = "paper") -> dict[str, object]:
    orders, deals = orders_payload(repo_root, env), deals_payload(repo_root, env)
    latest = max([date for date in (orders.get("trade_date"), deals.get("trade_date")) if isinstance(date, str)], default=None)
    snapshot = snapshot_payload(repo_root, env)
    return {
        "env": env,
        "label": ENV_LABELS[env],
        "state": _environment_state(orders, deals, snapshot, latest),
        "error": snapshot["error"],
        "generated_at": snapshot["generated_at"],
        "age_seconds": snapshot["age_seconds"],
        # Exported so the SPA can quote the alert threshold without
        # duplicating the constant client-side.
        "stale_threshold_seconds": STALE_SNAPSHOT_ALERT_SECONDS,
        "snapshot": snapshot["snapshot"],
        "trade_date": latest,
        "order_count": orders["count"],
        "deal_count": deals["count"],
        # Damaged journal lines are reported, not promoted to a state.
        "skipped_lines": int(orders["skipped_lines"]) + int(deals["skipped_lines"]),
    }


def environments_payload(repo_root: Path) -> dict[str, object]:
    return {"environments": [environment_summary(repo_root, "paper")]}


def health_payload(repo_root: Path, env: str = "paper") -> dict[str, object]:
    summary = environment_summary(repo_root, env)
    return {"ok": summary["state"] != "unreadable", **summary}
