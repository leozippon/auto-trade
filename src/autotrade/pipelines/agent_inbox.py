"""Append-only HITL agent inbox sidecar.

Users enqueue messages to the current Agent session through the Python
control plane. Fold and Meta runners consume them at documented safe points.
"""

from __future__ import annotations

import json
import os
import uuid
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from pathlib import Path

from autotrade.environment.runtime import utc_now_iso
from autotrade.pipelines.hitl_state import HITL_DIR_NAME, control_lock

INBOX_NAME = "agent_inbox.jsonl"
INBOX_SCHEMA_VERSION = 1
INBOX_MAX_TEXT_CHARS = 8192
INBOX_MAX_PENDING = 32
QUEUED_EVENT = "queued"
CONSUMED_EVENT = "consumed"
EXPIRED_EVENT = "expired"
REOPENED_EVENT = "reopened"
SESSION_EXPIRED_EVENT = "session_expired"
SESSION_OPENED_EVENT = "session_opened"
CONSUMED = "consumed"
ALREADY_CONSUMED = "already_consumed"
_SUCCESS_RECORD_TYPES = frozenset({"fold", "meta_learning"})


class InboxError(ValueError):
    """Durable inbox contract violation."""


@dataclass(frozen=True)
class InboxMessage:
    schema_version: int
    message_id: str
    session_key: str
    text: str
    interrupt: bool
    created_at: str
    consumed_at: str | None = None
    consumed_by: str | None = None
    expired_at: str | None = None
    expired_by: str | None = None


def inbox_path(experiment_dir: str | Path) -> Path:
    return Path(experiment_dir) / HITL_DIR_NAME / INBOX_NAME


def enqueue_inbox_message(
    path: str | Path,
    *,
    session_key: str,
    text: object,
    interrupt: bool = False,
) -> dict[str, object]:
    """Append one message and return a queued receipt without the body."""

    target = Path(path)
    key = _require_token(session_key, "session_key")
    body = _validate_text(text)
    if type(interrupt) is not bool:
        raise InboxError("inbox interrupt must be a boolean")
    message_id = uuid.uuid4().hex
    created_at = utc_now_iso()
    record = {
        "schema_version": INBOX_SCHEMA_VERSION,
        "event": QUEUED_EVENT,
        "message_id": message_id,
        "session_key": key,
        "text": body,
        "interrupt": interrupt,
        "created_at": created_at,
        "consumed_at": None,
        "consumed_by": None,
    }
    with control_lock(target):
        state = _load_state(target)
        if message_id in state.messages or message_id in state.consumed:
            raise InboxError("inbox message_id collision")
        if key in state.expired_sessions:
            raise InboxError("inbox session is expired")
        pending = sum(
            1
            for item in state.messages.values()
            if item.consumed_at is None and item.expired_at is None
        )
        if pending >= INBOX_MAX_PENDING:
            raise InboxError(f"agent inbox pending cap reached ({INBOX_MAX_PENDING})")
        _append_record(target, record)
    return {
        "status": "queued",
        "message_id": message_id,
        "session_key": key,
        "interrupt": interrupt,
        "created_at": created_at,
    }


def list_unconsumed_messages(
    path: str | Path, session_key: str
) -> tuple[InboxMessage, ...]:
    """Return this session's unconsumed messages in enqueue order.

    Phase 2B Runner entry point. Missing files are empty; other sessions are
    omitted. Worker restart re-reads this view and will not see consumed ids.
    """

    key = _require_token(session_key, "session_key")
    target = Path(path)
    with control_lock(target):
        state = _load_state(target)
    return tuple(
        item
        for item in state.messages.values()
        if item.session_key == key
        and item.consumed_at is None
        and item.expired_at is None
    )


def consume_inbox_message(
    path: str | Path,
    message_id: str,
    *,
    session_key: str,
    consumed_by: str,
) -> str:
    """Atomically mark one message consumed for ``session_key``.

    Phase 2B Runner entry point. Returns ``consumed`` or ``already_consumed``.
    A reread or worker restart that repeats the same id is idempotent. A
    message belonging to another session is refused and left untouched.
    """

    target = Path(path)
    mid = _require_token(message_id, "message_id")
    key = _require_token(session_key, "session_key")
    actor = _require_token(consumed_by, "consumed_by")
    consumed_at = utc_now_iso()
    with control_lock(target):
        state = _load_state(target)
        message = state.messages.get(mid)
        if message is None:
            raise InboxError(f"unknown inbox message_id: {mid}")
        if message.session_key != key:
            raise InboxError("inbox message does not belong to session")
        if message.expired_at is not None:
            raise InboxError("inbox message is expired")
        if message.consumed_at is not None:
            return ALREADY_CONSUMED
        _append_record(
            target,
            {
                "schema_version": INBOX_SCHEMA_VERSION,
                "event": CONSUMED_EVENT,
                "message_id": mid,
                "session_key": key,
                "consumed_at": consumed_at,
                "consumed_by": actor,
            },
        )
    return CONSUMED


def expire_session_inbox(
    path: str | Path,
    session_key: str,
    *,
    expired_by: str,
) -> tuple[str, ...]:
    """Expire leftovers and close the session to further enqueue."""

    target = Path(path)
    key = _require_token(session_key, "session_key")
    actor = _require_token(expired_by, "expired_by")
    expired_at = utc_now_iso()
    expired_ids: list[str] = []
    with control_lock(target):
        state = _load_state(target)
        for message in state.messages.values():
            if message.session_key != key or message.expired_at is not None:
                continue
            if message.consumed_at is not None and message.consumed_by == actor:
                continue
            _append_record(
                target,
                {
                    "schema_version": INBOX_SCHEMA_VERSION,
                    "event": EXPIRED_EVENT,
                    "message_id": message.message_id,
                    "session_key": key,
                    "expired_at": expired_at,
                    "expired_by": actor,
                },
            )
            expired_ids.append(message.message_id)
        if key not in state.expired_sessions:
            _append_record(
                target,
                {
                    "schema_version": INBOX_SCHEMA_VERSION,
                    "event": SESSION_EXPIRED_EVENT,
                    "session_key": key,
                    "expired_at": expired_at,
                    "expired_by": actor,
                },
            )
    return tuple(expired_ids)


def expire_experiment_session_inbox(
    experiment_dir: str | Path,
    session_key: str,
    *,
    expired_by: str,
) -> tuple[str, ...]:
    key = str(session_key or "").strip()
    actor = str(expired_by or "").strip()
    if not key or not actor:
        return ()
    return expire_session_inbox(
        inbox_path(experiment_dir), key, expired_by=actor
    )


def reopen_uncommitted_inbox(
    path: str | Path,
    session_key: str,
    *,
    committed_run_ids: Collection[str],
) -> tuple[str, ...]:
    """Reopen messages consumed only by runs that never landed in the ledger."""

    target = Path(path)
    key = _require_token(session_key, "session_key")
    committed = {
        str(item).strip() for item in committed_run_ids if str(item).strip()
    }
    reopened_at = utc_now_iso()
    reopened_ids: list[str] = []
    with control_lock(target):
        state = _load_state(target)
        for message in state.messages.values():
            if message.session_key != key or message.expired_at is not None:
                continue
            if message.consumed_at is None:
                continue
            if str(message.consumed_by or "") in committed:
                continue
            _append_record(
                target,
                {
                    "schema_version": INBOX_SCHEMA_VERSION,
                    "event": REOPENED_EVENT,
                    "message_id": message.message_id,
                    "session_key": key,
                    "reopened_at": reopened_at,
                },
            )
            reopened_ids.append(message.message_id)
    return tuple(reopened_ids)


def committed_session_run_ids(
    experiment_dir: str | Path, session_key: str
) -> frozenset[str]:
    """Successful Fold/Meta run_ids recorded for this session_key."""

    from autotrade.pipelines.ledger import ExperimentLedger

    key = str(session_key or "").strip()
    if not key:
        return frozenset()
    path = Path(experiment_dir) / "ledgers" / "experiment_ledger.jsonl"
    if not path.exists():
        return frozenset()
    ids: set[str] = set()
    for row in ExperimentLedger(path).read():
        if row.get("record_type") not in _SUCCESS_RECORD_TYPES:
            continue
        if str(row.get("session_key") or "") != key:
            continue
        run_id = str(row.get("run_id") or "").strip()
        if run_id:
            ids.add(run_id)
    return frozenset(ids)


@dataclass(frozen=True)
class SessionInbox:
    """Runner hook: current-session pending notices plus consume-by-run."""

    path: Path
    session_key: str
    run_id: str

    def pending(self) -> tuple[InboxMessage, ...]:
        return list_unconsumed_messages(self.path, self.session_key)

    def consume(self, message_id: str) -> str:
        return consume_inbox_message(
            self.path,
            message_id,
            session_key=self.session_key,
            consumed_by=self.run_id,
        )


def bind_session_inbox(
    experiment_dir: str | Path,
    *,
    session_key: str,
    run_id: str,
    committed_run_ids: Collection[str] | None = None,
) -> SessionInbox | None:
    """Reopen uncommitted consumes and return a runner hook, or None."""

    key = str(session_key or "").strip()
    actor = str(run_id or "").strip()
    if not key or not actor:
        return None
    path = inbox_path(experiment_dir)
    committed = (
        frozenset(str(item).strip() for item in committed_run_ids if str(item).strip())
        if committed_run_ids is not None
        else committed_session_run_ids(experiment_dir, key)
    )
    reopen_uncommitted_inbox(path, key, committed_run_ids=committed)
    with control_lock(path):
        state = _load_state(path)
        if key in state.expired_sessions:
            _append_record(
                path,
                {
                    "schema_version": INBOX_SCHEMA_VERSION,
                    "event": SESSION_OPENED_EVENT,
                    "session_key": key,
                    "opened_at": utc_now_iso(),
                    "opened_by": actor,
                },
            )
    return SessionInbox(path=path, session_key=key, run_id=actor)


def inbox_public_view(
    path: str | Path, *, session_key: str | None
) -> dict[str, object]:
    """Pending count and queued ids for a session; never includes bodies."""

    if not session_key:
        return {"pending_count": 0, "queued_ids": []}
    messages = list_unconsumed_messages(path, session_key)
    return {
        "pending_count": len(messages),
        "queued_ids": [item.message_id for item in messages],
    }


@dataclass
class _InboxState:
    messages: dict[str, InboxMessage]
    consumed: dict[str, tuple[str, str]]
    expired_sessions: set[str]


def _load_state(path: Path) -> _InboxState:
    messages: dict[str, InboxMessage] = {}
    consumed: dict[str, tuple[str, str]] = {}
    expired_sessions: set[str] = set()
    if not path.exists():
        return _InboxState(
            messages=messages, consumed=consumed, expired_sessions=expired_sessions
        )
    text = path.read_text(encoding="utf-8")
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise InboxError(f"corrupt agent inbox in {path}") from exc
        record = _parse_record(payload, path)
        event = str(record["event"])
        if event == SESSION_EXPIRED_EVENT:
            expired_sessions.add(str(record["session_key"]))
            continue
        if event == SESSION_OPENED_EVENT:
            expired_sessions.discard(str(record["session_key"]))
            continue
        message_id = str(record["message_id"])
        if event == QUEUED_EVENT:
            if message_id in messages:
                raise InboxError(f"duplicate inbox message_id in {path}")
            messages[message_id] = InboxMessage(
                schema_version=INBOX_SCHEMA_VERSION,
                message_id=message_id,
                session_key=str(record["session_key"]),
                text=str(record["text"]),
                interrupt=bool(record["interrupt"]),
                created_at=str(record["created_at"]),
                consumed_at=None,
                consumed_by=None,
            )
            continue
        if message_id not in messages:
            raise InboxError(f"consume record for unknown inbox message in {path}")
        if messages[message_id].session_key != record["session_key"]:
            raise InboxError(f"consume record session mismatch in {path}")
        queued = messages[message_id]
        if event == REOPENED_EVENT:
            if queued.expired_at is not None:
                continue
            consumed.pop(message_id, None)
            messages[message_id] = InboxMessage(
                schema_version=queued.schema_version,
                message_id=queued.message_id,
                session_key=queued.session_key,
                text=queued.text,
                interrupt=queued.interrupt,
                created_at=queued.created_at,
            )
            continue
        if event == EXPIRED_EVENT:
            if queued.expired_at is not None:
                continue
            messages[message_id] = InboxMessage(
                schema_version=queued.schema_version,
                message_id=queued.message_id,
                session_key=queued.session_key,
                text=queued.text,
                interrupt=queued.interrupt,
                created_at=queued.created_at,
                consumed_at=queued.consumed_at,
                consumed_by=queued.consumed_by,
                expired_at=str(record["expired_at"]),
                expired_by=str(record["expired_by"]),
            )
            continue
        if message_id in consumed:
            continue
        consumed_at = str(record["consumed_at"])
        consumed_by = str(record["consumed_by"])
        consumed[message_id] = (consumed_at, consumed_by)
        messages[message_id] = InboxMessage(
            schema_version=queued.schema_version,
            message_id=queued.message_id,
            session_key=queued.session_key,
            text=queued.text,
            interrupt=queued.interrupt,
            created_at=queued.created_at,
            consumed_at=consumed_at,
            consumed_by=consumed_by,
            expired_at=queued.expired_at,
            expired_by=queued.expired_by,
        )
    return _InboxState(
        messages=messages, consumed=consumed, expired_sessions=expired_sessions
    )


def _parse_record(payload: object, path: Path) -> dict[str, object]:
    if not isinstance(payload, dict):
        raise InboxError(f"corrupt agent inbox in {path}")
    version = payload.get("schema_version")
    if type(version) is not int or version != INBOX_SCHEMA_VERSION:
        raise InboxError(f"incompatible agent inbox in {path}")
    event = payload.get("event")
    if event not in {
        QUEUED_EVENT,
        CONSUMED_EVENT,
        EXPIRED_EVENT,
        REOPENED_EVENT,
        SESSION_EXPIRED_EVENT,
        SESSION_OPENED_EVENT,
    }:
        raise InboxError(f"corrupt agent inbox in {path}")
    session_key = payload.get("session_key")
    if not isinstance(session_key, str) or not session_key.strip():
        raise InboxError(f"corrupt agent inbox in {path}")
    if event == SESSION_EXPIRED_EVENT:
        if not isinstance(payload.get("expired_at"), str) or not payload["expired_at"]:
            raise InboxError(f"corrupt agent inbox in {path}")
        if not isinstance(payload.get("expired_by"), str) or not str(
            payload["expired_by"]
        ).strip():
            raise InboxError(f"corrupt agent inbox in {path}")
        return payload
    if event == SESSION_OPENED_EVENT:
        if not isinstance(payload.get("opened_at"), str) or not payload["opened_at"]:
            raise InboxError(f"corrupt agent inbox in {path}")
        if not isinstance(payload.get("opened_by"), str) or not str(
            payload["opened_by"]
        ).strip():
            raise InboxError(f"corrupt agent inbox in {path}")
        return payload
    message_id = payload.get("message_id")
    if not isinstance(message_id, str) or not message_id.strip():
        raise InboxError(f"corrupt agent inbox in {path}")
    if event == QUEUED_EVENT:
        if type(payload.get("interrupt")) is not bool:
            raise InboxError(f"corrupt agent inbox in {path}")
        if not isinstance(payload.get("text"), str):
            raise InboxError(f"corrupt agent inbox in {path}")
        if not isinstance(payload.get("created_at"), str) or not payload["created_at"]:
            raise InboxError(f"corrupt agent inbox in {path}")
        return payload
    if event == EXPIRED_EVENT:
        if not isinstance(payload.get("expired_at"), str) or not payload["expired_at"]:
            raise InboxError(f"corrupt agent inbox in {path}")
        if not isinstance(payload.get("expired_by"), str) or not str(
            payload["expired_by"]
        ).strip():
            raise InboxError(f"corrupt agent inbox in {path}")
        return payload
    if event == REOPENED_EVENT:
        if not isinstance(payload.get("reopened_at"), str) or not payload["reopened_at"]:
            raise InboxError(f"corrupt agent inbox in {path}")
        return payload
    if not isinstance(payload.get("consumed_at"), str) or not payload["consumed_at"]:
        raise InboxError(f"corrupt agent inbox in {path}")
    if not isinstance(payload.get("consumed_by"), str) or not str(
        payload["consumed_by"]
    ).strip():
        raise InboxError(f"corrupt agent inbox in {path}")
    return payload


def _append_record(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = (
        json.dumps(
            dict(payload),
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())


def _validate_text(text: object) -> str:
    if not isinstance(text, str) or not text.strip() or "\x00" in text:
        raise InboxError("inbox message text must be a non-empty UTF-8 string")
    if len(text) > INBOX_MAX_TEXT_CHARS:
        raise InboxError(
            f"inbox message text exceeds {INBOX_MAX_TEXT_CHARS} characters"
        )
    try:
        text.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise InboxError("inbox message text must be a non-empty UTF-8 string") from exc
    return text


def _require_token(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise InboxError(f"inbox {label} is required")
    return value.strip()
