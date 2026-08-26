"""Phase 2A agent inbox: durable enqueue, runner consume API, Web queued contract."""

from __future__ import annotations

import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from autotrade.environment.identity import AgentRefStore
from autotrade.environment.runtime import write_json_atomic
from autotrade.pipelines.agent_inbox import (
    ALREADY_CONSUMED,
    CONSUMED,
    INBOX_MAX_PENDING,
    INBOX_MAX_TEXT_CHARS,
    INBOX_NAME,
    INBOX_SCHEMA_VERSION,
    InboxError,
    bind_session_inbox,
    consume_inbox_message,
    enqueue_inbox_message,
    expire_session_inbox,
    inbox_path,
    inbox_public_view,
    list_unconsumed_messages,
    reopen_uncommitted_inbox,
)
from autotrade.pipelines.hitl_state import (
    HITL_STATE_SCHEMA_VERSION,
    ControlState,
    proc_start_ticks,
    read_control,
    write_control,
)
from autotrade.webui.public_identity import PublicIdentity
from autotrade.webui.server import create_app

SESSION_A = "epoch_001/fold_2022Q2"
SESSION_B = "epoch_001/fold_2022Q1"


def _write_inbox_experiment(root: Path, experiment_id: str = "exp_in") -> Path:
    directory = root / "experiments" / experiment_id
    hitl = directory / "hitl"
    hitl.mkdir(parents=True)
    write_json_atomic(hitl / "params.json", {"experiment_id": experiment_id})
    write_control(hitl / "control.json", ControlState(mode="auto"))
    write_json_atomic(
        hitl / "status.json",
        {"schema_version": 1, "pid": 999_999_999, "state": "stopped"},
    )
    write_json_atomic(
        hitl / "schedule.json",
        {
            "schema_version": 1,
            "sessions": [
                {
                    "session_key": SESSION_B,
                    "kind": "fold",
                    "epoch_id": "epoch_001",
                    "fold_id": "fold_2022Q1",
                },
                {
                    "session_key": SESSION_A,
                    "kind": "fold",
                    "epoch_id": "epoch_001",
                    "fold_id": "fold_2022Q2",
                },
            ],
        },
    )
    AgentRefStore(directory)
    return directory


def _public_session(directory: Path, raw_session: str) -> str:
    return PublicIdentity(directory).public_session_key(raw_session)


def _mark_live(
    directory: Path,
    *,
    session_key: str,
    state: str = "running_session",
) -> None:
    write_json_atomic(
        directory / "hitl/status.json",
        {
            "schema_version": 1,
            "pid": os.getpid(),
            "pid_start_ticks": proc_start_ticks(os.getpid()),
            "state": state,
            "session_key": session_key,
        },
    )


def test_hitl_control_schema_version_is_unchanged() -> None:
    assert HITL_STATE_SCHEMA_VERSION == 1
    assert INBOX_SCHEMA_VERSION == 1


def test_enqueue_is_append_only_and_returns_queued_without_body(tmp_path: Path) -> None:
    path = tmp_path / INBOX_NAME
    first = enqueue_inbox_message(
        path, session_key=SESSION_A, text="先看回撤", interrupt=False
    )
    second = enqueue_inbox_message(
        path, session_key=SESSION_A, text="再看换手", interrupt=True
    )
    assert first["status"] == second["status"] == "queued"
    assert "text" not in first and "text" not in second
    first_id = str(first["message_id"])
    second_id = str(second["message_id"])
    assert first_id != second_id
    assert len(first_id) == 32
    assert set(first_id) <= set("0123456789abcdef")
    pending = list_unconsumed_messages(path, SESSION_A)
    assert [item.text for item in pending] == ["先看回撤", "再看换手"]
    assert pending[1].interrupt is True
    assert pending[0].consumed_at is None
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["schema_version"] == 1
    assert json.loads(lines[1])["interrupt"] is True


def test_concurrent_enqueues_are_all_durable(tmp_path: Path) -> None:
    path = tmp_path / INBOX_NAME

    def _write(index: int) -> str:
        receipt = enqueue_inbox_message(
            path, session_key=SESSION_A, text=f"msg-{index}"
        )
        return str(receipt["message_id"])

    with ThreadPoolExecutor(max_workers=8) as pool:
        ids = list(pool.map(_write, range(8)))
    assert len(set(ids)) == 8
    pending = list_unconsumed_messages(path, SESSION_A)
    assert {item.message_id for item in pending} == set(ids)
    assert {item.text for item in pending} == {f"msg-{index}" for index in range(8)}


def test_concurrent_consume_is_idempotent_and_not_duplicated(tmp_path: Path) -> None:
    path = tmp_path / INBOX_NAME
    queued = enqueue_inbox_message(path, session_key=SESSION_A, text="只消费一次")
    workers = 8
    barrier = threading.Barrier(workers)
    results: list[str] = []
    guard = threading.Lock()

    def _consume() -> None:
        barrier.wait()
        result = consume_inbox_message(
            path,
            str(queued["message_id"]),
            session_key=SESSION_A,
            consumed_by="run_001",
        )
        with guard:
            results.append(result)

    threads = [threading.Thread(target=_consume) for _ in range(workers)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert results.count(CONSUMED) == 1
    assert results.count(ALREADY_CONSUMED) == workers - 1
    assert list_unconsumed_messages(path, SESSION_A) == ()
    lines = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert sum(1 for row in lines if row.get("event") == "consumed") == 1
    replay = consume_inbox_message(
        path,
        str(queued["message_id"]),
        session_key=SESSION_A,
        consumed_by="run_restart",
    )
    assert replay == ALREADY_CONSUMED
    assert list_unconsumed_messages(path, SESSION_A) == ()


def test_sessions_do_not_cross_consume(tmp_path: Path) -> None:
    path = tmp_path / INBOX_NAME
    a = enqueue_inbox_message(path, session_key=SESSION_A, text="A")
    b = enqueue_inbox_message(path, session_key=SESSION_B, text="B")
    assert [item.text for item in list_unconsumed_messages(path, SESSION_A)] == ["A"]
    assert [item.text for item in list_unconsumed_messages(path, SESSION_B)] == ["B"]
    with pytest.raises(InboxError, match="does not belong to session"):
        consume_inbox_message(
            path, str(b["message_id"]), session_key=SESSION_A, consumed_by="run_001"
        )
    assert [item.text for item in list_unconsumed_messages(path, SESSION_B)] == ["B"]
    with pytest.raises(InboxError, match="unknown inbox message_id"):
        consume_inbox_message(
            path, "0" * 32, session_key=SESSION_A, consumed_by="run_001"
        )
    assert consume_inbox_message(
        path, str(a["message_id"]), session_key=SESSION_A, consumed_by="run_001"
    ) == CONSUMED
    assert list_unconsumed_messages(path, SESSION_A) == ()
    assert [item.text for item in list_unconsumed_messages(path, SESSION_B)] == ["B"]


def test_pending_cap_and_text_limits(tmp_path: Path) -> None:
    path = tmp_path / INBOX_NAME
    for index in range(INBOX_MAX_PENDING):
        enqueue_inbox_message(path, session_key=SESSION_A, text=f"n{index}")
    with pytest.raises(InboxError, match="pending cap"):
        enqueue_inbox_message(path, session_key=SESSION_A, text="overflow")
    first = list_unconsumed_messages(path, SESSION_A)[0]
    consume_inbox_message(
        path, first.message_id, session_key=SESSION_A, consumed_by="run_001"
    )
    enqueue_inbox_message(path, session_key=SESSION_A, text="after-consume")
    with pytest.raises(InboxError, match="non-empty UTF-8"):
        enqueue_inbox_message(path, session_key=SESSION_A, text="")
    with pytest.raises(InboxError, match="non-empty UTF-8"):
        enqueue_inbox_message(path, session_key=SESSION_A, text=" \n\t")
    with pytest.raises(InboxError, match=str(INBOX_MAX_TEXT_CHARS)):
        enqueue_inbox_message(
            path, session_key=SESSION_A, text="x" * (INBOX_MAX_TEXT_CHARS + 1)
        )
    with pytest.raises(InboxError, match="non-empty UTF-8"):
        enqueue_inbox_message(path, session_key=SESSION_A, text="\ud800")


def test_public_view_omits_bodies(tmp_path: Path) -> None:
    path = tmp_path / INBOX_NAME
    queued = enqueue_inbox_message(path, session_key=SESSION_A, text="secret-body")
    view = inbox_public_view(path, session_key=SESSION_A)
    assert view == {
        "pending_count": 1,
        "queued_ids": [queued["message_id"]],
    }
    assert "text" not in view
    assert "secret-body" not in json.dumps(view)
    assert inbox_public_view(path, session_key=None) == {
        "pending_count": 0,
        "queued_ids": [],
    }
    assert inbox_public_view(path, session_key=SESSION_B) == {
        "pending_count": 0,
        "queued_ids": [],
    }


def test_old_control_schema_is_unchanged_by_inject(tmp_path: Path) -> None:
    directory = _write_inbox_experiment(tmp_path)
    public_session = _public_session(directory, SESSION_A)
    control_path = directory / "hitl/control.json"
    control_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "mode": "auto",
                "request": None,
                "approved_sessions": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    before = control_path.read_bytes()
    state = read_control(control_path)
    assert state.mode == "auto"
    assert state.gpu_counts == {}
    _mark_live(directory, session_key=SESSION_A)
    client = TestClient(create_app(tmp_path))
    response = client.post(
        "/api/experiments/exp_in/control",
        json={
            "action": "inject_message",
            "session_key": public_session,
            "text": "继续验证父策略",
        },
    )
    assert response.status_code == 200, response.text
    assert control_path.read_bytes() == before
    assert HITL_STATE_SCHEMA_VERSION == 1


def test_api_queues_without_claiming_interrupt_and_hides_bodies(
    tmp_path: Path,
) -> None:
    directory = _write_inbox_experiment(tmp_path)
    public_session = _public_session(directory, SESSION_A)
    _mark_live(directory, session_key=SESSION_A)
    client = TestClient(create_app(tmp_path))
    secret = "unique-inbox-secret-body"
    response = client.post(
        "/api/experiments/exp_in/control",
        json={
            "action": "inject_message",
            "session_key": public_session,
            "text": secret,
            "interrupt": True,
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "queued"
    assert body["interrupt"] is True
    assert body["session_key"] == public_session
    assert "text" not in body
    assert secret not in response.text
    assert "interrupted" not in response.text
    detail = client.get("/api/experiments/exp_in").json()
    status = client.get("/api/experiments/exp_in/status").json()
    listing = client.get("/api/experiments").json()
    assert detail["inbox"]["pending_count"] == 1
    assert detail["inbox"]["queued_ids"] == [body["message_id"]]
    public = json.dumps({"detail": detail, "status": status, "listing": listing})
    assert secret not in public
    assert "text" not in detail["inbox"]
    assert "inbox" not in listing["experiments"][0]
    pending = list_unconsumed_messages(inbox_path(directory), SESSION_A)
    assert pending[0].text == secret
    assert pending[0].interrupt is True


def test_api_refuses_empty_text_bad_id_dead_worker_and_stale_session(
    tmp_path: Path,
) -> None:
    directory = _write_inbox_experiment(tmp_path)
    public_a = _public_session(directory, SESSION_A)
    public_b = _public_session(directory, SESSION_B)
    client = TestClient(create_app(tmp_path))
    dead = client.post(
        "/api/experiments/exp_in/control",
        json={
            "action": "inject_message",
            "session_key": public_a,
            "text": "hello",
        },
    )
    assert dead.status_code == 400
    assert "live worker" in dead.json()["detail"]

    _mark_live(directory, session_key=SESSION_A)
    empty = client.post(
        "/api/experiments/exp_in/control",
        json={"action": "inject_message", "session_key": public_a, "text": ""},
    )
    assert empty.status_code == 400
    oversized = client.post(
        "/api/experiments/exp_in/control",
        json={
            "action": "inject_message",
            "session_key": public_a,
            "text": "x" * (INBOX_MAX_TEXT_CHARS + 1),
        },
    )
    assert oversized.status_code == 400
    dated = client.post(
        "/api/experiments/exp_in/control",
        json={
            "action": "inject_message",
            "session_key": public_a,
            "text": "在 2022Q1 减仓",
        },
    )
    assert dated.status_code == 400
    assert "日历日期" in dated.json()["detail"]
    wrong = client.post(
        "/api/experiments/exp_in/control",
        json={
            "action": "inject_message",
            "session_key": public_b,
            "text": "hello",
        },
    )
    assert wrong.status_code == 400
    assert "current Agent session" in wrong.json()["detail"]

    _mark_live(directory, session_key=SESSION_A, state="failed")
    failed = client.post(
        "/api/experiments/exp_in/control",
        json={
            "action": "inject_message",
            "session_key": public_a,
            "text": "hello",
        },
    )
    assert failed.status_code == 400
    assert "finished or failed" in failed.json()["detail"]

    _mark_live(directory, session_key=SESSION_A, state="completed")
    completed = client.post(
        "/api/experiments/exp_in/control",
        json={
            "action": "inject_message",
            "session_key": public_a,
            "text": "hello",
        },
    )
    assert completed.status_code == 400

    bad_id = client.post(
        "/api/experiments/inv@lid/control",
        json={
            "action": "inject_message",
            "session_key": public_a,
            "text": "hello",
        },
    )
    assert bad_id.status_code == 400
    missing = client.post(
        "/api/experiments/missing_exp/control",
        json={
            "action": "inject_message",
            "session_key": public_a,
            "text": "hello",
        },
    )
    assert missing.status_code == 404
    assert list_unconsumed_messages(inbox_path(directory), SESSION_A) == ()


def test_uncommitted_consume_is_reopened_committed_consume_stays(tmp_path: Path) -> None:
    path = inbox_path(tmp_path)
    queued = enqueue_inbox_message(path, session_key=SESSION_A, text="引导减仓")
    mid = str(queued["message_id"])
    assert consume_inbox_message(
        path, mid, session_key=SESSION_A, consumed_by="run_dead"
    ) == CONSUMED
    assert list_unconsumed_messages(path, SESSION_A) == ()
    reopened = reopen_uncommitted_inbox(
        path, SESSION_A, committed_run_ids=()
    )
    assert reopened == (mid,)
    pending = list_unconsumed_messages(path, SESSION_A)
    assert [item.text for item in pending] == ["引导减仓"]
    hook = bind_session_inbox(
        tmp_path,
        session_key=SESSION_A,
        run_id="run_live",
        committed_run_ids=(),
    )
    assert hook is not None
    assert hook.consume(mid) == CONSUMED
    assert list_unconsumed_messages(path, SESSION_A) == ()
    assert reopen_uncommitted_inbox(
        path, SESSION_A, committed_run_ids={"run_live"}
    ) == ()
    assert list_unconsumed_messages(path, SESSION_A) == ()


def test_expire_blocks_next_session_and_reopen(tmp_path: Path) -> None:
    path = tmp_path / INBOX_NAME
    leftover = enqueue_inbox_message(path, session_key=SESSION_A, text="未消费")
    other = enqueue_inbox_message(path, session_key=SESSION_B, text="别的会话")
    dead = enqueue_inbox_message(path, session_key=SESSION_A, text="旧 run 已消费")
    consume_inbox_message(
        path,
        str(dead["message_id"]),
        session_key=SESSION_A,
        consumed_by="run_dead",
    )
    applied = enqueue_inbox_message(path, session_key=SESSION_A, text="本 run 已用")
    consume_inbox_message(
        path,
        str(applied["message_id"]),
        session_key=SESSION_A,
        consumed_by="run_done",
    )
    expired = expire_session_inbox(path, SESSION_A, expired_by="run_done")
    assert set(expired) == {leftover["message_id"], dead["message_id"]}
    assert list_unconsumed_messages(path, SESSION_A) == ()
    assert [item.text for item in list_unconsumed_messages(path, SESSION_B)] == [
        "别的会话"
    ]
    assert reopen_uncommitted_inbox(
        path, SESSION_A, committed_run_ids={"run_done"}
    ) == ()
    with pytest.raises(InboxError, match="expired"):
        consume_inbox_message(
            path,
            str(leftover["message_id"]),
            session_key=SESSION_A,
            consumed_by="run_next",
        )
    assert other["message_id"]


def test_pending_cap_excludes_expired_messages(tmp_path: Path) -> None:
    path = inbox_path(tmp_path)
    for index in range(INBOX_MAX_PENDING):
        enqueue_inbox_message(path, session_key=SESSION_A, text=f"n{index}")
    with pytest.raises(InboxError, match="pending cap"):
        enqueue_inbox_message(path, session_key=SESSION_B, text="blocked")
    expire_session_inbox(path, SESSION_A, expired_by="run_done")
    new_session = enqueue_inbox_message(
        path, session_key=SESSION_B, text="new-session"
    )
    assert new_session["status"] == "queued"
    bind_session_inbox(
        tmp_path,
        session_key=SESSION_A,
        run_id="run_next",
        committed_run_ids={"run_done"},
    )
    legal = enqueue_inbox_message(path, session_key=SESSION_A, text="reopened-session")
    assert legal["status"] == "queued"
    assert [item.text for item in list_unconsumed_messages(path, SESSION_B)] == [
        "new-session"
    ]
    assert [item.text for item in list_unconsumed_messages(path, SESSION_A)] == [
        "reopened-session"
    ]


def test_expire_then_enqueue_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / INBOX_NAME
    assert expire_session_inbox(path, SESSION_A, expired_by="run_done") == ()
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert records == [
        {
            "event": "session_expired",
            "expired_at": records[0]["expired_at"],
            "expired_by": "run_done",
            "schema_version": 1,
            "session_key": SESSION_A,
        }
    ]
    with pytest.raises(InboxError, match="session is expired"):
        enqueue_inbox_message(path, session_key=SESSION_A, text="迟到")
    assert list_unconsumed_messages(path, SESSION_A) == ()
    other = enqueue_inbox_message(path, session_key=SESSION_B, text="别的会话")
    assert other["status"] == "queued"


def test_enqueue_then_expire_covers_the_message(tmp_path: Path) -> None:
    path = tmp_path / INBOX_NAME
    queued = enqueue_inbox_message(path, session_key=SESSION_A, text="先入队")
    expired = expire_session_inbox(path, SESSION_A, expired_by="run_done")
    assert expired == (queued["message_id"],)
    assert list_unconsumed_messages(path, SESSION_A) == ()
    with pytest.raises(InboxError, match="session is expired"):
        enqueue_inbox_message(path, session_key=SESSION_A, text="再入队")


def test_concurrent_enqueue_and_expire_leave_no_unconsumed(tmp_path: Path) -> None:
    path = tmp_path / INBOX_NAME
    barrier = threading.Barrier(2)
    queued_id: str | None = None
    refused: str | None = None
    expired_ids: tuple[str, ...] = ()
    guard = threading.Lock()

    def _enqueue() -> None:
        nonlocal queued_id, refused
        barrier.wait()
        try:
            receipt = enqueue_inbox_message(path, session_key=SESSION_A, text="迟到")
            with guard:
                queued_id = str(receipt["message_id"])
        except InboxError as exc:
            with guard:
                refused = str(exc)

    def _expire() -> None:
        nonlocal expired_ids
        barrier.wait()
        result = expire_session_inbox(path, SESSION_A, expired_by="run_done")
        with guard:
            expired_ids = result

    threads = [threading.Thread(target=_enqueue), threading.Thread(target=_expire)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert list_unconsumed_messages(path, SESSION_A) == ()
    if queued_id is not None:
        assert queued_id in expired_ids
        assert refused is None
    else:
        assert refused is not None and "session is expired" in refused
    with pytest.raises(InboxError, match="session is expired"):
        enqueue_inbox_message(path, session_key=SESSION_A, text="再来")


def test_api_refuses_inject_after_expire_while_status_still_live(
    tmp_path: Path,
) -> None:
    directory = _write_inbox_experiment(tmp_path)
    public_session = _public_session(directory, SESSION_A)
    _mark_live(directory, session_key=SESSION_A)
    expire_session_inbox(inbox_path(directory), SESSION_A, expired_by="run_done")
    client = TestClient(create_app(tmp_path))
    response = client.post(
        "/api/experiments/exp_in/control",
        json={
            "action": "inject_message",
            "session_key": public_session,
            "text": "迟到注入",
        },
    )
    assert response.status_code == 400, response.text
    assert "session is expired" in response.json()["detail"]
    assert list_unconsumed_messages(inbox_path(directory), SESSION_A) == ()
