"""Trading console read-model + route tests (negative paths first).

Every fixture is synthesized in a tempfile repo root under
``data/trading/paper/`` exactly as the Paper engine writes it. Invariants under
test: whitelist projection, structured degradation (never 500), the environment
whitelist, date validation, and non-finite numbers degrading to null instead of
exploding at the serializer.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest
from fastapi.testclient import TestClient

from autotrade.webui import trading
from autotrade.webui.server import create_app

DETAIL_ROUTES = ("snapshot", "orders", "deals", "series", "health")


def _jsonl(path: Path, *payloads: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            (line if isinstance(line, str) else json.dumps(line, ensure_ascii=False)) + "\n"
            for line in payloads
        ),
        encoding="utf-8",
    )


def _order(**overrides: object) -> dict[str, object]:
    row = {
        "event_id": "o1",
        "symbol": "000001.SZ",
        "action": "buy",
        "quantity": 100,
        "execute_at": "2026-01-02T09:30:00+08:00",
        "metadata": {"reason": "rebalance"},
    }
    row.update(overrides)
    return row


def _execution(**overrides: object) -> dict[str, object]:
    row = {
        **_order(event_id="e1"),
        "matched_at": "2026-01-02T09:30:00+08:00",
        "status": "filled",
        "price": 10.25,
        "commission": 5.0,
        "stamp_duty": 0.0,
    }
    row.update(overrides)
    return row


def test_daily_paper_projection(tmp_path: Path):
    root = tmp_path / "data/trading/paper"
    _jsonl(root / "orders_20260102.jsonl", _order())
    _jsonl(root / "executions_20260102.jsonl", _execution())
    orders = trading.orders_payload(tmp_path)
    assert orders["orders"][0]["symbol"] == "000001.SZ"
    # The engine journals matched fills as executions_<date>.jsonl; the console
    # serves them under the deals contract.
    deals = trading.deals_payload(tmp_path)
    assert deals["deals"][0]["status"] == "filled"
    assert deals["deals"][0]["price"] == 10.25
    summary = trading.environment_summary(tmp_path)
    assert summary["deal_count"] == 1
    assert summary["order_count"] == 1
    assert summary["trade_date"] == "20260102"
    assert summary["label"] == "Paper 模拟"
    assert trading.health_payload(tmp_path)["ok"] is True


def test_order_projection_is_a_whitelist_and_never_echoes_extra_fields(tmp_path: Path):
    _jsonl(
        tmp_path / "data/trading/paper/orders_20260102.jsonl",
        _order(internal_note="must not surface", account_id="ACCT-PRIVATE"),
    )
    row = trading.orders_payload(tmp_path)["orders"][0]
    assert set(row) == {"symbol", "action", "quantity", "execute_at"}
    # Strategy-authored order metadata is writer content, not a projected
    # scalar: it never reaches the payload either.
    assert "rebalance" not in json.dumps(trading.orders_payload(tmp_path))
    assert "ACCT-PRIVATE" not in json.dumps(trading.orders_payload(tmp_path))


def test_latest_date_is_selected_and_available_dates_are_listed(tmp_path: Path):
    root = tmp_path / "data/trading/paper"
    _jsonl(root / "orders_20260102.jsonl", _order(event_id="a"))
    _jsonl(root / "orders_20260105.jsonl", _order(event_id="b"), _order(event_id="c"))
    payload = trading.orders_payload(tmp_path)
    assert payload["available_dates"] == ["20260102", "20260105"]
    assert payload["trade_date"] == "20260105"
    assert payload["count"] == 2
    assert trading.orders_payload(tmp_path, date="20260102")["count"] == 1


def test_missing_and_invalid_paper_files_degrade(tmp_path: Path):
    assert trading.orders_payload(tmp_path)["state"] == "absent"
    assert trading.series_payload(tmp_path)["state"] == "absent"
    assert trading.snapshot_payload(tmp_path)["state"] == "absent"
    assert trading.environment_summary(tmp_path)["state"] == "absent"
    path = tmp_path / "data/trading/paper/orders_20260102.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text("not-json\n", encoding="utf-8")
    payload = trading.orders_payload(tmp_path)
    # A journal is an append-only stream a crash can truncate: damaged lines
    # are counted and the good records around them are still served, so a
    # skipped line is reported rather than promoted to an environment state.
    assert payload["state"] == "ok"
    assert payload["orders"] == []
    assert payload["skipped_lines"] == 1
    assert "error" not in payload
    assert trading.environment_summary(tmp_path)["state"] == "ok"
    assert trading.environment_summary(tmp_path)["skipped_lines"] == 1
    assert trading.health_payload(tmp_path)["ok"] is True
    snapshot = tmp_path / "data/trading/paper/account_snapshot.json"
    snapshot.write_text("[]\n", encoding="utf-8")
    snapshot_payload = trading.snapshot_payload(tmp_path)
    assert snapshot_payload["state"] == "unreadable"
    assert snapshot_payload["snapshot"] is None
    snapshot.write_text("{broken", encoding="utf-8")
    assert trading.snapshot_payload(tmp_path)["state"] == "unreadable"
    # The snapshot's unreadable rung is untouched and still degrades health.
    assert trading.environment_summary(tmp_path)["state"] == "unreadable"
    assert trading.health_payload(tmp_path)["ok"] is False


def test_good_rows_around_a_damaged_line_are_all_served(tmp_path: Path):
    """The regression the counting revert is about: a fail-fast reader returned
    only the rows BEFORE the break and called the environment unreadable."""
    root = tmp_path / "data/trading/paper"
    _jsonl(
        root / "orders_20260102.jsonl",
        _order(event_id="a", symbol="000001.SZ"),
        "{ truncated",
        _order(event_id="b", symbol="600000.SH"),
        "[]",
    )
    payload = trading.orders_payload(tmp_path)
    assert payload["state"] == "ok"
    assert payload["count"] == 2
    assert [row["symbol"] for row in payload["orders"]] == ["000001.SZ", "600000.SH"]
    assert payload["skipped_lines"] == 2
    summary = trading.environment_summary(tmp_path)
    assert summary["state"] == "ok"
    assert summary["skipped_lines"] == 2
    assert summary["order_count"] == 2


def test_non_finite_numbers_degrade_to_null_instead_of_failing_the_serializer(tmp_path: Path):
    # json.loads accepts NaN/Infinity tokens; starlette's serializer does not
    # (allow_nan=False) — the read model must project them to null.
    root = tmp_path / "data/trading/paper"
    _jsonl(
        root / "executions_20260102.jsonl",
        '{"symbol": "000001.SZ", "action": "buy", "quantity": 100,'
        ' "execute_at": "2026-01-02T09:30:00+08:00", "matched_at": null,'
        ' "status": "filled", "price": NaN, "commission": Infinity, "stamp_duty": -Infinity}',
    )
    _jsonl(
        root / "equity_daily.jsonl",
        '{"trade_date": "20260102", "equity": NaN, "cash": Infinity}',
    )
    row = trading.deals_payload(tmp_path)["deals"][0]
    assert row["price"] is None
    assert row["commission"] is None
    assert row["stamp_duty"] is None
    assert row["matched_at"] is None
    point = trading.series_payload(tmp_path)["series"][0]
    assert point["equity"] is None and point["cash"] is None
    client = TestClient(create_app(tmp_path))
    for route in ("deals", "series"):
        assert client.get(f"/api/trading/paper/{route}").status_code == 200


def test_invalid_quantity_and_blank_text_project_to_null(tmp_path: Path):
    _jsonl(
        tmp_path / "data/trading/paper/orders_20260102.jsonl",
        _order(quantity=True, symbol=""),
        _order(quantity=-5),
    )
    rows = trading.orders_payload(tmp_path)["orders"]
    assert rows[0]["quantity"] is None and rows[0]["symbol"] is None
    assert rows[1]["quantity"] is None


def test_equity_series_projection(tmp_path: Path):
    _jsonl(
        tmp_path / "data/trading/paper/equity_daily.jsonl",
        {"trade_date": "20260102", "equity": 100_000.0, "cash": 40_000.0},
        {"trade_date": "20260105", "equity": 101_000.0, "cash": 39_000.0},
    )
    payload = trading.series_payload(tmp_path)
    assert payload["state"] == "ok"
    assert [row["trade_date"] for row in payload["series"]] == ["20260102", "20260105"]
    assert payload["series"][1]["equity"] == 101_000.0


def test_env_whitelist_rejects_everything_else(tmp_path: Path):
    client = TestClient(create_app(tmp_path))
    for env in ("live", "sim", "prod", "%2E%2E%2Fpaper", "%2E"):
        for route in DETAIL_ROUTES:
            assert client.get(f"/api/trading/{env}/{route}").status_code == 404, (env, route)
    # Defense in depth below the routes: no path is built from bad input.
    for env in ("live", "sim", "prod", "../paper", ".", ""):
        with pytest.raises(KeyError):
            trading.env_dir(tmp_path, env)
    assert trading.env_dir(tmp_path, "paper") == tmp_path / "data/trading/paper"


def test_trading_api_exposes_only_paper_and_validates_date(tmp_path: Path):
    client = TestClient(create_app(tmp_path))
    roster = client.get("/api/trading/environments").json()["environments"]
    assert [entry["env"] for entry in roster] == ["paper"]
    assert client.get("/api/trading/live/orders").status_code == 404
    for bad in ("2026-01-02", "abc", "202601021", "2026010"):
        for route in ("orders", "deals"):
            assert client.get(f"/api/trading/paper/{route}?date={bad}").status_code == 400, (route, bad)
    for route in DETAIL_ROUTES:
        response = client.get(f"/api/trading/paper/{route}")
        assert response.status_code == 200
        assert response.json()["env"] == "paper"


# ---- the snapshot state machine ---------------------------------------------

SNAPSHOT_NAME = "account_snapshot.json"


def _cn_iso(age_seconds: float = 0.0) -> str:
    """The Paper engine persists Asia/Shanghai stamps; the API normalizes."""
    moment = datetime.now(ZoneInfo("Asia/Shanghai")) - timedelta(seconds=age_seconds)
    return moment.replace(microsecond=0).isoformat()


def _write_snapshot(
    tmp_path: Path,
    *,
    age_seconds: float = 0.0,
    ok: bool = True,
    error: str | None = None,
    generated_at: str | None = None,
    positions: list[dict] | None = None,
    **extra: object,
) -> Path:
    root = tmp_path / "data/trading/paper"
    root.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": generated_at if generated_at is not None else _cn_iso(age_seconds),
        "ok": ok,
        "error": error,
        "source": "paper_engine",
        "trade_date": "20260102",
        "day_complete": True,
        "phase": "closed",
        "strategy_revision": "revision_001",
        "cash": 400_000.0,
        "equity": 1_000_000.0,
        "pending_order_count": 0,
        "positions": positions if positions is not None else [
            {
                "symbol": "000001.SZ",
                "quantity": 100,
                "available_quantity": 0,  # fully T+1 locked
                "average_cost": 10.1,
                "last_price": 10.5,
            }
        ],
        **extra,
    }
    (root / SNAPSHOT_NAME).write_text(json.dumps(payload), encoding="utf-8")
    return root


def test_absent_and_no_snapshot_are_distinguishable(tmp_path: Path):
    # No env directory at all.
    assert trading.snapshot_payload(tmp_path)["state"] == "absent"
    assert trading.environment_summary(tmp_path)["state"] == "absent"
    # Directory created by a deployment, engine never run.
    (tmp_path / "data/trading/paper").mkdir(parents=True)
    assert trading.snapshot_payload(tmp_path)["state"] == "no_snapshot"
    assert trading.environment_summary(tmp_path)["state"] == "no_snapshot"


def test_corrupt_and_non_object_snapshots_are_unreadable(tmp_path: Path):
    root = tmp_path / "data/trading/paper"
    root.mkdir(parents=True)
    for content in ("{not json", "[1, 2, 3]"):
        (root / SNAPSHOT_NAME).write_text(content, encoding="utf-8")
        payload = trading.snapshot_payload(tmp_path)
        assert payload["state"] == "unreadable", content
        assert payload["snapshot"] is None
        assert payload["error"]
        assert trading.health_payload(tmp_path)["ok"] is False


def test_writer_error_surfaces_only_its_first_line(tmp_path: Path):
    _write_snapshot(
        tmp_path,
        ok=False,
        error="RuntimeError: writer down\nTraceback (most recent call last):\n  secret payload",
    )
    payload = trading.snapshot_payload(tmp_path)
    assert payload["state"] == "export_error"
    assert payload["error"] == "RuntimeError: writer down"
    assert "secret payload" not in json.dumps(payload)
    assert trading.environment_summary(tmp_path)["state"] == "export_error"


def test_writer_error_without_a_message_still_reports_the_state(tmp_path: Path):
    _write_snapshot(tmp_path, ok=False, error=None)
    payload = trading.snapshot_payload(tmp_path)
    assert payload["state"] == "export_error"
    assert payload["error"] == "writer reported ok=false"


def test_a_stale_snapshot_is_served_and_flagged_against_the_exported_threshold(tmp_path: Path):
    _write_snapshot(tmp_path, age_seconds=600.0)
    payload = trading.snapshot_payload(tmp_path)
    assert payload["state"] == "stale"
    assert payload["stale_threshold_seconds"] == trading.STALE_SNAPSHOT_ALERT_SECONDS == 180.0
    assert 590.0 <= payload["age_seconds"] <= 630.0
    # Stale-but-visible: the last written account data still reaches the page.
    assert payload["snapshot"]["equity"] == 1_000_000.0
    summary = trading.environment_summary(tmp_path)
    assert summary["state"] == "stale"
    assert trading.health_payload(tmp_path)["ok"] is True  # degraded, not broken


def test_a_fresh_snapshot_is_ok(tmp_path: Path):
    _write_snapshot(tmp_path, age_seconds=1.0)
    payload = trading.snapshot_payload(tmp_path)
    assert payload["state"] == "ok"
    assert payload["age_seconds"] < trading.STALE_SNAPSHOT_ALERT_SECONDS


def test_a_missing_or_unparseable_generated_at_is_unreadable_not_a_crash(tmp_path: Path):
    for stamp in ("not-a-timestamp", ""):
        _write_snapshot(tmp_path, generated_at=stamp)
        payload = trading.snapshot_payload(tmp_path)
        assert payload["state"] == "unreadable", stamp
        assert "generated_at" in payload["error"]


def test_naive_china_local_stamps_normalize_to_utc(tmp_path: Path):
    """The engine persists naive Asia/Shanghai stamps and this module is the
    single normalization boundary, so a naive stamp must be CONVERTED, not
    relabelled. Relabelling puts the staleness clock 8 hours out: a snapshot
    written now reports a negative age, and one 8 hours old reads as fresh, so
    STALE_SNAPSHOT_ALERT_SECONDS never fires."""
    _write_snapshot(tmp_path, generated_at="2026-07-30T14:01:26")
    payload = trading.snapshot_payload(tmp_path)
    assert payload["generated_at"] == "2026-07-30T06:01:26Z"

    fresh = datetime.now(ZoneInfo("Asia/Shanghai")).replace(microsecond=0, tzinfo=None)
    _write_snapshot(tmp_path, generated_at=fresh.isoformat())
    age = trading.snapshot_payload(tmp_path)["age_seconds"]
    assert 0.0 <= age < 60.0, f"a snapshot written now reports age {age}"


def test_an_offset_aware_stamp_is_converted_not_relabelled(tmp_path: Path):
    _write_snapshot(tmp_path, generated_at="2026-07-30T14:01:26+08:00")
    assert trading.snapshot_payload(tmp_path)["generated_at"] == "2026-07-30T06:01:26Z"


def test_snapshot_is_whitelist_projected_and_never_echoes_the_raw_dict(tmp_path: Path):
    _write_snapshot(tmp_path, secret="LEAK", account_id="ACCT-PRIVATE")
    payload = trading.snapshot_payload(tmp_path)
    assert payload["state"] == "ok"
    assert set(payload["snapshot"]) == {
        "source", "trade_date", "day_complete", "phase", "strategy_revision",
        "cash", "equity", "pending_order_count", "positions",
    }
    rendered = json.dumps(payload)
    assert "LEAK" not in rendered and "ACCT-PRIVATE" not in rendered


def test_a_fully_locked_position_keeps_its_zero_counters(tmp_path: Path):
    _write_snapshot(tmp_path)
    row = trading.snapshot_payload(tmp_path)["snapshot"]["positions"][0]
    # A T+1-locked line has available_quantity 0. Projecting it as None would
    # read as "unknown" and make the row look unmappable.
    assert row["available_quantity"] == 0
    assert row["quantity"] == 100
    assert row["unmapped"] is False


def test_an_unmappable_position_row_is_flagged_never_dropped(tmp_path: Path):
    _write_snapshot(tmp_path, positions=[{"m_unknownField": 1, "raw": {"blob": "x"}}])
    positions = trading.snapshot_payload(tmp_path)["snapshot"]["positions"]
    assert len(positions) == 1
    assert positions[0]["unmapped"] is True
    assert all(positions[0][key] is None for key in ("symbol", "quantity", "last_price"))


def test_non_finite_snapshot_numbers_degrade_to_null(tmp_path: Path):
    root = tmp_path / "data/trading/paper"
    root.mkdir(parents=True)
    (root / SNAPSHOT_NAME).write_text(
        '{"generated_at": "' + _cn_iso() + '", "ok": true, "cash": NaN,'
        ' "equity": Infinity, "pending_order_count": 0,'
        ' "positions": [{"symbol": "000001.SZ", "quantity": 100,'
        '                "available_quantity": 0, "last_price": NaN}]}',
        encoding="utf-8",
    )
    snapshot = trading.snapshot_payload(tmp_path)["snapshot"]
    assert snapshot["cash"] is None and snapshot["equity"] is None
    assert snapshot["positions"][0]["last_price"] is None
    assert snapshot["positions"][0]["unmapped"] is False  # symbol still mapped
    assert TestClient(create_app(tmp_path)).get("/api/trading/paper/snapshot").status_code == 200


def test_environment_state_precedence_puts_the_worst_reader_first(tmp_path: Path):
    root = tmp_path / "data/trading/paper"
    # A stale snapshot beats a healthy journal.
    _write_snapshot(tmp_path, age_seconds=600.0)
    _jsonl(root / "orders_20260102.jsonl", _order())
    assert trading.environment_summary(tmp_path)["state"] == "stale"
    # A damaged journal line does NOT reach the ladder; stale still wins.
    _jsonl(root / "orders_20260102.jsonl", _order(), "broken")
    summary = trading.environment_summary(tmp_path)
    assert summary["state"] == "stale"
    assert summary["skipped_lines"] == 1
    # An unreadable snapshot outranks everything.
    (root / SNAPSHOT_NAME).write_text("{broken", encoding="utf-8")
    assert trading.environment_summary(tmp_path)["state"] == "unreadable"


def test_the_roster_and_health_carry_the_snapshot_block(tmp_path: Path):
    _write_snapshot(tmp_path, age_seconds=1.0)
    entry = trading.environments_payload(tmp_path)["environments"][0]
    assert entry["env"] == "paper" and entry["label"] == "Paper 模拟"
    assert entry["state"] == "ok"
    assert entry["snapshot"]["equity"] == 1_000_000.0
    assert entry["generated_at"].endswith("Z")
    assert entry["stale_threshold_seconds"] == 180.0
    health = trading.health_payload(tmp_path)
    assert health["ok"] is True and health["state"] == "ok"


# ---- the client renders the states the server emits -------------------------

def _app_js() -> str:
    return (
        Path(__file__).resolve().parents[2]
        / "src/autotrade/webui/static/app.js"
    ).read_text(encoding="utf-8")


def test_every_server_snapshot_state_has_a_client_label_and_tone():
    script = _app_js()
    block = script.split("const TRADING_STATE", 1)[1].split("};", 1)[0]
    for state in ("ok", "stale", "no_snapshot", "export_error", "unreadable", "absent"):
        assert f"{state}:" in block, state
    # Emitting a state nothing renders is the same defect in a third direction:
    # every state carries a badge tone and a Chinese label.
    for tone in ("completed", "paused", "failed", "stopped"):
        assert tone in block
    for label in ("正常", "数据陈旧", "等待首次运行", "写入错误", "数据不可读", "等待数据"):
        assert label in block


def test_degraded_states_raise_a_banner_and_skipped_lines_raise_a_chip():
    script = _app_js()
    assert "function paperBanners(" in script
    banners = script.split("function paperBanners(", 1)[1].split("\nfunction ", 1)[0]
    for state in ("export_error", "unreadable", "stale"):
        assert state in banners, state
    # Item 10: closed's stale banner ends 「与飞书告警一致」; that clause is out.
    assert "飞书" not in script
    assert "function skippedChip(" in script
    assert "行无法解析" in script
    # The unmapped flag reaches the researcher rather than sitting in the payload.
    assert "无法映射" in script
