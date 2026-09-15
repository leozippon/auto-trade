"""Trading console read-model + route tests (negative paths first).

Every fixture is synthesized in a tempfile repo root under
``data/trading/paper/`` exactly as the Paper engine writes it — the book panels
read a book the real engine ran over synthetic bars. Invariants under test: each
panel's figures are the ones the book's own code defines, whitelist projection,
structured degradation (never 500), the environment whitelist, and non-finite
numbers degrading to null instead of exploding at the serializer.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from autotrade.paper.orders import order_sheet
from autotrade.webui import trading
from autotrade.webui.server import create_app
from tests.unit.paper_book_fixture import engine_book, paper_root, write_book_record

BOOK = "exp"
# The panels one book's page reads, each under /api/trading/<env>/books/<book>/.
BOOK_ROUTES = ("status", "book", "signal", "history", "performance", "snapshot")


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


def _csi300_slot(root: Path, name: str, pct_chg: dict[str, float]) -> None:
    """One replay slot of the book's PIT cache carrying CSI 300 (and another index)."""
    slot = root / "pit/gen/pit_views/replay/paper" / name
    slot.mkdir(parents=True, exist_ok=True)
    rows = [
        {"dataset": "index_daily", "ts_code": "000300.SH", "trade_date": day, "pct_chg": value}
        for day, value in pct_chg.items()
    ]
    rows.append({"dataset": "index_daily", "ts_code": "000905.SH", "trade_date": "20260105", "pct_chg": 9.0})
    pd.DataFrame(rows).to_parquet(slot / "macro.parquet")


def test_the_book_panel_is_a_whitelist_of_the_frozen_identity(tmp_path: Path):
    engine_book(tmp_path, "20260105", "20260106")
    payload = trading.book_payload(tmp_path, BOOK)
    assert payload["state"] == "ok"
    assert payload["book"] == {
        "experiment_id": "exp", "artifact_id": "art", "candidate_source": "graduated",
        "note": "参考簿（观察中）", "initial_cash": 100_000.0,
    }
    assert (payload["start_date"], payload["settled_through"]) == ("20260105", "20260105")
    assert "/private/lake" not in json.dumps(payload)


def test_the_signal_is_the_latest_order_sheet_and_history_keeps_every_earlier_day(tmp_path: Path):
    engine_book(tmp_path, "20260105")
    signal = trading.signal_payload(tmp_path, BOOK)["signal"]
    # The first morning buys 100 at the 10.50 reference; the post-trade account
    # is that holding plus the cash left, so the weights sum to one.
    [target] = signal["target"]
    assert (target["quantity"], target["value"]) == (100, 1050.0)
    assert signal["cash_after"] == 100_000.0 - 1050.0
    assert target["weight"] + signal["cash_weight"] == pytest.approx(1.0)
    assert set(signal["orders"][0]) == {
        "execute_at", "symbol", "name", "action", "quantity", "reference_price", "notional",
    }
    # The strategy's own order metadata ("call") is writer content, never served.
    assert '"call"' not in json.dumps(signal)
    assert trading.history_payload(tmp_path, BOOK) == {"env": "paper", "state": "absent", "error": None, "days": []}

    root = engine_book(tmp_path, "20260106", "20260107")
    signal = trading.signal_payload(tmp_path, BOOK)["signal"]
    assert (signal["trade_date"], signal["data_through"]) == ("20260107", "20260106")
    assert [(row["action"], row["quantity"], row["reference_price"], row["notional"]) for row in signal["orders"]] == [
        ("sell", 100, 12.5, 1250.0)
    ]
    assert signal["target"] == [] and signal["cash_after"] == order_sheet(root, "20260107")["cash_after"]
    # Every earlier day, newest first, its orders and fills under one session
    # key; the latest decision stays the signal's until its fills exist.
    days = trading.history_payload(tmp_path, BOOK)["days"]
    assert [day["trade_date"] for day in days] == ["20260106", "20260105"]
    assert days[0]["orders"] == [] and days[0]["fills"] == []
    assert [row["action"] for row in days[1]["orders"]] == ["buy"]
    [fill] = days[1]["fills"]
    assert (fill["status"], fill["name"], fill["quantity"], fill["price"]) == ("filled", "平安银行", 100, 11.0)
    # A past day carries the same post-trade block the signal panel shows for
    # today; dropping it server-side left the two panels at different detail.
    sheet = order_sheet(root, "20260105")
    assert [(row["symbol"], row["quantity"], row["value"]) for row in days[1]["target"]] == [
        (row["symbol"], row["quantity"], row["value"]) for row in sheet["target"]
    ]
    assert (days[1]["cash_after"], days[1]["cash_weight"]) == (sheet["cash_after"], sheet["cash_weight"])


def test_a_settlement_only_day_says_it_has_no_order_sheet(tmp_path: Path):
    """A day with fills but no decision of its own has no post-trade holdings
    to show: null says so, where an empty list would read as a flat book."""
    root = engine_book(tmp_path, "20260105", "20260106")
    _jsonl(root / "executions_20260107.jsonl", _execution(matched_at="2026-01-07T09:30:00+08:00"))
    [extra] = [day for day in trading.history_payload(tmp_path, BOOK)["days"] if day["trade_date"] == "20260107"]
    assert extra["orders"] == [] and extra["target"] is None
    assert extra["cash_after"] is None and extra["cash_weight"] is None
    assert len(extra["fills"]) == 1


def test_performance_keys_return_equity_cash_and_csi300_by_the_same_settled_days(tmp_path: Path):
    root = engine_book(tmp_path, "20260105", "20260106", "20260107", "20260108")
    journal = [json.loads(line) for line in (root / "equity_daily.jsonl").read_text(encoding="utf-8").splitlines()]
    settled = [row["trade_date"] for row in journal]
    assert settled == ["20260105", "20260106", "20260107"]
    # An older, narrower slot of an earlier run must not be the one read.
    _csi300_slot(root, "20260102_20260107_20251231T235959+0800", {"20260105": 50.0})
    _csi300_slot(
        root,
        "20260102_20260108_20251231T235959+0800",
        {"20260102": 5.0, "20260105": 1.0, "20260106": -0.5, "20260107": 0.2},
    )
    payload = trading.performance_payload(tmp_path, BOOK)
    chart, stats = payload["chart"], payload["statistics"]
    assert chart["series"][0]["dates"] == chart["account"]["dates"] == chart["benchmark"]["dates"] == settled
    assert chart["account"]["equity"] == [row["equity"] for row in journal]
    assert chart["account"]["cash"] == [row["cash"] for row in journal]
    # Day-0 baseline: the first day's return is measured from the initial cash.
    assert chart["series"][0]["cum"][0] == round(journal[0]["equity"] / 100_000.0 - 1.0, 6)
    assert stats["total_return"] == pytest.approx(journal[-1]["equity"] / 100_000.0 - 1.0)
    assert stats["benchmark_return"] == pytest.approx(1.01 * 0.995 * 1.002 - 1.0, abs=1e-6)
    assert stats["excess_return"] == pytest.approx(stats["total_return"] - stats["benchmark_return"])
    assert (stats["fills"], stats["days"], payload["min_days"]) == (2, 3, trading.MIN_STATISTICS_DAYS)
    assert stats["turnover"] == pytest.approx((1100.0 + 1300.0) / 100_000.0)
    # Too few settled days for annualised figures: "—", never a number.
    assert stats["annualized_return"] is None and stats["sharpe"] is None and stats["max_drawdown"] is None

    # CSI 300 missing one settled day: its curve keeps the days it has, the
    # excess is not computed over a different window.
    _csi300_slot(root, "20260102_20260108_20251231T235959+0800", {"20260105": 1.0, "20260106": -0.5})
    partial = trading.performance_payload(tmp_path, BOOK)
    assert partial["benchmark_days"] == 2 and partial["chart"]["benchmark"]["dates"] == settled[:2]
    assert partial["statistics"]["benchmark_return"] is None and partial["statistics"]["excess_return"] is None


def test_one_settled_day_has_statistics_but_no_curve(tmp_path: Path):
    """A curve needs two points. Served as a chart, one settled day drew a lone
    dot under an empty drawdown band; the day's statistics still stand, so the
    rule lives here and every surface reads the same absent chart."""
    engine_book(tmp_path, "20260105", "20260106")
    payload = trading.performance_payload(tmp_path, BOOK)
    assert payload["state"] == "ok" and payload["chart"] is None
    assert payload["statistics"]["days"] == 1
    assert payload["statistics"]["total_return"] is not None
    assert trading.books_payload(tmp_path)["books"][0]["curve"] is None

    engine_book(tmp_path, "20260107")
    payload = trading.performance_payload(tmp_path, BOOK)
    chart = payload["chart"]
    assert payload["statistics"]["days"] == 2
    assert chart["series"][0]["dates"] == chart["account"]["dates"] == ["20260105", "20260106"]
    assert len(chart["account"]["equity"]) == len(chart["account"]["cash"]) == 2
    # The overview card draws the same curve, and only the return series of it.
    curve = trading.books_payload(tmp_path)["books"][0]["curve"]
    assert curve == {"series": chart["series"], "benchmark": chart["benchmark"]}


def test_annualised_statistics_appear_once_the_book_has_enough_days(tmp_path: Path, monkeypatch):
    engine_book(tmp_path, "20260105", "20260106", "20260107", "20260108")
    monkeypatch.setattr(trading, "MIN_STATISTICS_DAYS", 3)
    stats = trading.performance_payload(tmp_path, BOOK)["statistics"]
    assert None not in (stats["annualized_return"], stats["sharpe"], stats["max_drawdown"])


def test_missing_and_damaged_book_files_degrade_per_panel(tmp_path: Path):
    assert trading.books_payload(tmp_path)["state"] == "absent"
    with pytest.raises(KeyError):
        trading.signal_payload(tmp_path, BOOK)  # not a book under the root
    write_book_record(paper_root(tmp_path) / BOOK)  # created, never run
    for projection in (trading.signal_payload, trading.history_payload, trading.performance_payload):
        assert projection(tmp_path, BOOK)["state"] == "absent", projection.__name__
    assert trading.book_status(tmp_path, BOOK)["state"] == "no_snapshot"
    # Created, never run: nothing has settled, so the card counts no holdings.
    assert trading.books_payload(tmp_path)["books"][0]["position_count"] is None
    root = engine_book(tmp_path, "20260105", "20260106")
    (root / ".paper_state.json").write_text("{broken", encoding="utf-8")
    for projection in (trading.book_payload, trading.signal_payload, trading.history_payload):
        payload = projection(tmp_path, BOOK)
        assert payload["state"] == "unreadable" and payload["error"], projection.__name__
    assert trading.book_status(tmp_path, BOOK)["state"] == "unreadable"
    assert trading.health_payload(tmp_path)["ok"] is False
    client = TestClient(create_app(tmp_path))
    for route in BOOK_ROUTES:
        assert client.get(f"/api/trading/paper/books/{BOOK}/{route}").status_code == 200, route
        assert client.get(f"/api/trading/paper/books/nobook/{route}").status_code == 404, route


def test_damaged_fill_and_equity_lines_are_counted_and_projected_to_null(tmp_path: Path):
    """A journal is an append-only stream a crash can truncate: the good rows
    around a damaged line are served and the line is counted, and json.loads'
    NaN/Infinity tokens project to null instead of failing the serializer."""
    root = engine_book(tmp_path, "20260105", "20260106", "20260107")
    _jsonl(
        root / "executions_20260106.jsonl",
        _execution(event_id="a", matched_at="2026-01-06T09:30:00+08:00"),
        "{ truncated",
        '{"symbol": "", "action": "buy", "quantity": true, "status": "rejected",'
        ' "price": NaN, "commission": Infinity, "reason": "limit_up"}',
    )
    with (root / "equity_daily.jsonl").open("a", encoding="utf-8") as stream:
        stream.write('{"trade_date": "20260106", "equity": NaN, "cash": 1.0}\n[]\n')
    [day] = [day for day in trading.history_payload(tmp_path, BOOK)["days"] if day["trade_date"] == "20260106"]
    assert day["skipped_lines"] == 1
    good, bad = day["fills"]
    assert (good["status"], good["price"], good["cost"]) == ("filled", 10.25, 5.0)
    assert (bad["symbol"], bad["quantity"], bad["price"], bad["cost"]) == (None, None, None, None)
    performance = trading.performance_payload(tmp_path, BOOK)
    assert performance["chart"]["account"]["dates"] == ["20260105", "20260106"]
    client = TestClient(create_app(tmp_path))
    for route in ("history", "performance"):
        assert client.get(f"/api/trading/paper/books/{BOOK}/{route}").status_code == 200, route


def test_env_and_book_whitelists_reject_everything_else(tmp_path: Path):
    engine_book(tmp_path)
    client = TestClient(create_app(tmp_path))
    for env in ("live", "sim", "prod", "%2E%2E%2Fpaper", "%2E"):
        assert client.get(f"/api/trading/{env}/books").status_code == 404, env
        assert client.get(f"/api/trading/{env}/health").status_code == 404, env
        for route in BOOK_ROUTES:
            assert client.get(f"/api/trading/{env}/books/{BOOK}/{route}").status_code == 404, (env, route)
    for book in ("..", "%2E%2E", ".hidden", "not-a-book"):
        assert client.get(f"/api/trading/paper/books/{book}/signal").status_code == 404, book
    # Defense in depth below the routes: no path is built from bad input.
    for env in ("live", "sim", "prod", "../paper", ".", ""):
        with pytest.raises(KeyError):
            trading.env_dir(tmp_path, env)
    for book in ("..", "../exp", "", "exp/..", "missing"):
        with pytest.raises(KeyError):
            trading.book_dir(tmp_path, book)
    assert trading.book_dir(tmp_path, BOOK) == tmp_path / "data/trading/paper" / BOOK


def test_the_overview_has_one_row_per_book_read_off_its_panels(tmp_path: Path):
    engine_book(tmp_path, "20260105", "20260106", "20260107", book="alpha")
    engine_book(tmp_path, "20260105", book="beta")
    overview = trading.books_payload(tmp_path)
    assert overview["state"] == "ok"
    alpha, beta = overview["books"]
    assert (alpha["book_id"], beta["book_id"]) == ("alpha", "beta")
    performance = trading.performance_payload(tmp_path, "alpha")["statistics"]
    performance_chart = trading.performance_payload(tmp_path, "alpha")["chart"]
    snapshot = trading.snapshot_payload(tmp_path, "alpha")["snapshot"]
    assert alpha["total_return"] == performance["total_return"]
    assert alpha["equity"] == snapshot["equity"]
    assert (alpha["signal_date"], alpha["order_count"]) == ("20260107", 1)
    assert (alpha["start_date"], alpha["initial_cash"], alpha["state"]) == ("20260105", 100_000.0, "ok")
    # The card names where the candidate came from and draws the book's own
    # curve; the drawdown the day count still gates has no tile.
    assert (alpha["candidate_source"], alpha["max_drawdown"]) == ("graduated", None)
    # 持仓 is the one figure every settled book has: the lines its own 当前持仓
    # panel lists, absent rather than zero before the first snapshot.
    assert alpha["position_count"] == len(snapshot["positions"])
    assert trading.books_payload(tmp_path)["books"][1]["position_count"] == len(
        trading.snapshot_payload(tmp_path, "beta")["snapshot"]["positions"]
    )
    assert alpha["curve"]["series"][0]["dates"] == performance_chart["account"]["dates"]
    # A book whose first decision has not settled yet has no return to show.
    assert (beta["total_return"], beta["order_count"]) == (None, 1)
    assert beta["curve"] is None
    health = trading.health_payload(tmp_path)
    assert health["ok"] is True and [row["book_id"] for row in health["books"]] == ["alpha", "beta"]
    # A root still in the single-book layout is reported, not read as a book.
    write_book_record(paper_root(tmp_path))
    assert trading.books_payload(tmp_path)["state"] == "unreadable"
    assert trading.health_payload(tmp_path)["ok"] is False


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
    root = paper_root(tmp_path) / BOOK
    write_book_record(root)
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


def test_no_snapshot_and_unreadable_snapshots_are_distinguishable(tmp_path: Path):
    root = paper_root(tmp_path) / BOOK
    write_book_record(root)  # created, engine never run
    assert trading.snapshot_payload(tmp_path, BOOK)["state"] == "no_snapshot"
    assert trading.book_status(tmp_path, BOOK)["state"] == "no_snapshot"
    for content in ("{not json", "[1, 2, 3]"):
        (root / SNAPSHOT_NAME).write_text(content, encoding="utf-8")
        payload = trading.snapshot_payload(tmp_path, BOOK)
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
    payload = trading.snapshot_payload(tmp_path, BOOK)
    assert payload["state"] == "export_error"
    assert payload["error"] == "RuntimeError: writer down"
    assert "secret payload" not in json.dumps(payload)
    assert trading.book_status(tmp_path, BOOK)["state"] == "export_error"


def test_writer_error_without_a_message_still_reports_the_state(tmp_path: Path):
    _write_snapshot(tmp_path, ok=False, error=None)
    payload = trading.snapshot_payload(tmp_path, BOOK)
    assert payload["state"] == "export_error"
    assert payload["error"] == "writer reported ok=false"


def test_a_weekend_between_daily_runs_is_not_stale(tmp_path: Path):
    """The book writes once per weekday before the open: Friday's snapshot read
    on Monday before that day's run is the normal cadence, not a stale book."""
    _write_snapshot(tmp_path, age_seconds=75 * 3600.0)
    assert trading.snapshot_payload(tmp_path, BOOK)["state"] == "ok"


def test_a_stale_snapshot_is_served_and_flagged_against_the_exported_threshold(tmp_path: Path):
    age = trading.STALE_SNAPSHOT_ALERT_SECONDS + 600.0
    _write_snapshot(tmp_path, age_seconds=age)
    payload = trading.snapshot_payload(tmp_path, BOOK)
    assert payload["state"] == "stale"
    assert payload["stale_threshold_seconds"] == trading.STALE_SNAPSHOT_ALERT_SECONDS
    assert age - 10.0 <= payload["age_seconds"] <= age + 30.0
    # Stale-but-visible: the last written account data still reaches the page.
    assert payload["snapshot"]["equity"] == 1_000_000.0
    summary = trading.book_status(tmp_path, BOOK)
    assert summary["state"] == "stale"
    assert trading.health_payload(tmp_path)["ok"] is True  # degraded, not broken


def test_a_fresh_snapshot_is_ok(tmp_path: Path):
    _write_snapshot(tmp_path, age_seconds=1.0)
    payload = trading.snapshot_payload(tmp_path, BOOK)
    assert payload["state"] == "ok"
    assert payload["age_seconds"] < trading.STALE_SNAPSHOT_ALERT_SECONDS


def test_a_missing_or_unparseable_generated_at_is_unreadable_not_a_crash(tmp_path: Path):
    for stamp in ("not-a-timestamp", ""):
        _write_snapshot(tmp_path, generated_at=stamp)
        payload = trading.snapshot_payload(tmp_path, BOOK)
        assert payload["state"] == "unreadable", stamp
        assert "generated_at" in payload["error"]


def test_naive_china_local_stamps_normalize_to_utc(tmp_path: Path):
    """The engine persists naive Asia/Shanghai stamps and this module is the
    single normalization boundary, so a naive stamp must be CONVERTED, not
    relabelled. Relabelling puts the staleness clock 8 hours out: a snapshot
    written now reports a negative age, and one 8 hours old reads as fresh, so
    STALE_SNAPSHOT_ALERT_SECONDS never fires."""
    _write_snapshot(tmp_path, generated_at="2026-07-30T14:01:26")
    payload = trading.snapshot_payload(tmp_path, BOOK)
    assert payload["generated_at"] == "2026-07-30T06:01:26Z"

    fresh = datetime.now(ZoneInfo("Asia/Shanghai")).replace(microsecond=0, tzinfo=None)
    _write_snapshot(tmp_path, generated_at=fresh.isoformat())
    age = trading.snapshot_payload(tmp_path, BOOK)["age_seconds"]
    assert 0.0 <= age < 60.0, f"a snapshot written now reports age {age}"


def test_an_offset_aware_stamp_is_converted_not_relabelled(tmp_path: Path):
    _write_snapshot(tmp_path, generated_at="2026-07-30T14:01:26+08:00")
    assert trading.snapshot_payload(tmp_path, BOOK)["generated_at"] == "2026-07-30T06:01:26Z"


def test_snapshot_is_whitelist_projected_and_never_echoes_the_raw_dict(tmp_path: Path):
    _write_snapshot(tmp_path, secret="LEAK", account_id="ACCT-PRIVATE")
    payload = trading.snapshot_payload(tmp_path, BOOK)
    assert payload["state"] == "ok"
    assert set(payload["snapshot"]) == {
        "settled_through", "cash", "equity", "market_value", "pending_order_count", "positions",
    }
    rendered = json.dumps(payload)
    assert "LEAK" not in rendered and "ACCT-PRIVATE" not in rendered


def test_a_fully_locked_position_keeps_its_zero_counters(tmp_path: Path):
    _write_snapshot(tmp_path)
    row = trading.snapshot_payload(tmp_path, BOOK)["snapshot"]["positions"][0]
    # A T+1-locked line has available_quantity 0. Projecting it as None would
    # read as "unknown" and make the row look unmappable.
    assert row["available_quantity"] == 0
    assert row["quantity"] == 100
    assert row["unmapped"] is False
    # Value, P&L and weight are served, computed once from the projected figures.
    assert row["market_value"] == pytest.approx(1050.0)
    assert row["pnl"] == pytest.approx(100 * (10.5 - 10.1))
    assert row["weight"] == pytest.approx(1050.0 / 1_000_000.0)


def test_an_unmappable_position_row_is_flagged_never_dropped(tmp_path: Path):
    _write_snapshot(tmp_path, positions=[{"m_unknownField": 1, "raw": {"blob": "x"}}])
    positions = trading.snapshot_payload(tmp_path, BOOK)["snapshot"]["positions"]
    assert len(positions) == 1
    assert positions[0]["unmapped"] is True
    assert all(positions[0][key] is None for key in ("symbol", "quantity", "last_price"))


def test_non_finite_snapshot_numbers_degrade_to_null(tmp_path: Path):
    root = paper_root(tmp_path) / BOOK
    write_book_record(root)
    (root / SNAPSHOT_NAME).write_text(
        '{"generated_at": "' + _cn_iso() + '", "ok": true, "cash": NaN,'
        ' "equity": Infinity, "pending_order_count": 0,'
        ' "positions": [{"symbol": "000001.SZ", "quantity": 100,'
        '                "available_quantity": 0, "last_price": NaN}]}',
        encoding="utf-8",
    )
    snapshot = trading.snapshot_payload(tmp_path, BOOK)["snapshot"]
    assert snapshot["cash"] is None and snapshot["equity"] is None
    assert snapshot["positions"][0]["last_price"] is None
    assert snapshot["positions"][0]["unmapped"] is False  # symbol still mapped
    assert TestClient(create_app(tmp_path)).get(f"/api/trading/paper/books/{BOOK}/snapshot").status_code == 200


def test_environment_state_precedence_puts_the_worst_reader_first(tmp_path: Path):
    root = paper_root(tmp_path) / BOOK
    # A stale snapshot beats a healthy journal.
    _write_snapshot(tmp_path, age_seconds=trading.STALE_SNAPSHOT_ALERT_SECONDS + 600.0)
    _jsonl(root / "orders_20260102.jsonl", _order())
    assert trading.book_status(tmp_path, BOOK)["state"] == "stale"
    # A damaged journal line does NOT reach the ladder; stale still wins.
    _jsonl(root / "orders_20260102.jsonl", _order(), "broken")
    assert trading.book_status(tmp_path, BOOK)["state"] == "stale"
    # An unreadable snapshot outranks everything.
    (root / SNAPSHOT_NAME).write_text("{broken", encoding="utf-8")
    assert trading.book_status(tmp_path, BOOK)["state"] == "unreadable"


def test_the_status_and_health_carry_the_status_ladder(tmp_path: Path):
    _write_snapshot(tmp_path, age_seconds=1.0)
    status = trading.book_status(tmp_path, BOOK)
    assert (status["env"], status["book_id"], status["state"]) == ("paper", BOOK, "ok")
    assert status["generated_at"].endswith("Z")
    assert status["stale_threshold_seconds"] == trading.STALE_SNAPSHOT_ALERT_SECONDS
    health = trading.health_payload(tmp_path)
    assert health["ok"] is True and [(row["book_id"], row["state"]) for row in health["books"]] == [(BOOK, "ok")]


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
    assert "function skippedChip(" in script
    assert "行无法解析" in script
    # The unmapped flag reaches the researcher rather than sitting in the payload.
    assert "无法映射" in script


def _js_top_level(script: str, opening: str) -> str:
    """One top-level declaration of app.js, up to the next one."""
    start = script.index(opening)
    ends = [
        index
        for marker in ("\nfunction ", "\nasync function ", "\nconst ", "\n/*")
        if (index := script.find(marker, start + 1)) != -1
    ]
    return script[start : min(ends)]


@pytest.mark.skipif(shutil.which("node") is None, reason="node required for the JS formatters")
def test_prices_keep_their_cents_and_only_large_amounts_abbreviate():
    """A per-share price is never a money amount: the amount formatter rounded a
    38.389185 fill to ¥38. Prices print at cent resolution, and an amount below
    ¥10,000 keeps its cents, as the Paper orders sheet prints both."""
    script = _app_js()
    harness = "\n".join(
        [
            _js_top_level(script, "const CENTS_FMT ="),
            _js_top_level(script, "function fmtAmount("),
            _js_top_level(script, "function fmtPrice("),
            _js_top_level(script, "function fmtAmountOpt("),
            (
                "console.log(JSON.stringify(["
                "fmtPrice(38.389185), fmtPrice(1450.123456), fmtPrice(0), fmtPrice(null),"
                " fmtPrice(undefined), fmtPrice('n/a'), fmtAmount(5922), fmtAmount(43.5),"
                " fmtAmount(-881.4), fmtAmount(146250), fmtAmount(2.5e8), fmtAmountOpt(null)]));"
            ),
        ]
    )
    result = subprocess.run(
        ["node", "-e", harness], capture_output=True, text=True, timeout=60, check=False
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == [
        "38.39", "1,450.12", "0.00", "—", "—", "—",
        "¥5,922.00", "¥43.50", "¥-881.40", "¥14.6万", "¥2.50亿", "—",
    ]
    # Every price column goes through the price formatter.
    for name in ("paperPositionsPanel", "paperSheetBody", "paperOrdersTable", "paperHistoryDay", "ordersNode"):
        assert "fmtPrice(" in _js_top_level(script, f"function {name}("), name
    assert not re.search(r"fmtAmount(Opt)?\(row\.(price|average_cost|last_price|reference_price)\)", script)


def test_the_paper_performance_chart_is_the_research_chart_on_one_date_axis():
    """Equity and cash were once two charts with their own widths, pads and date
    ticks, so one trading day sat at different x positions. The page now feeds
    the book's days to the research return chart, whose panes share one x-scale,
    and draws no second chart implementation of its own."""
    script = _app_js()
    panel = _js_top_level(script, "function paperPerformancePanel(")
    assert "equityChart(payload.chart" in panel
    trading_section = script.split("let tradingView = null;", 1)[1].split("function renderQmtPage(", 1)[0]
    assert "singleSeriesBarChart" not in trading_section
    assert "payload.account" in _js_top_level(script, "function equityChart(")


def test_the_page_draws_a_figure_only_where_the_book_measured_one():
    """The panel printed 「—」 for every statistic the day count still gates and
    for a CSI 300 the book has no data for. Tiles now come from presentTiles,
    which drops an absent figure, and the reason moves into the caption."""
    script = _app_js()
    for name in ("paperPerformancePanel", "bookCard"):
        body = _js_top_level(script, f"function {name}(")
        assert "presentTiles(" in body, name
        assert "—" not in body, name
    panel = _js_top_level(script, "function paperPerformancePanel(")
    assert "min_days" in panel and "benchmark_days" in panel


def test_today_and_every_past_day_render_through_one_order_sheet_renderer():
    """The signal panel showed the post-trade holdings and a history day could
    not, because the payload dropped them. One renderer now draws both, so the
    two levels of detail cannot drift apart again."""
    script = _app_js()
    for name in ("paperSignalPanel", "paperHistoryDay"):
        assert "paperSheetBody(" in _js_top_level(script, f"function {name}("), name
