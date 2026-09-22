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

from autotrade.paper.book import SOURCE_HISTORY_NAME, copy_source_history
from autotrade.paper.orders import order_sheet
from autotrade.paper.pit import newest_replay_slot
from autotrade.pipelines.ledger import ExperimentLedger, forward_record
from autotrade.webui import trading
from autotrade.webui.equity import result_equity_payload
from autotrade.webui.server import create_app
from tests.unit.paper_book_fixture import engine_book, paper_root, write_book_record
from tests.unit.webui_research_arm import REPLAY, build_arm

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
    """One replay slot of the book's PIT cache carrying CSI 300 (and another index).

    The builder leaves a permanent ``<slot>.lock`` beside every slot, so the
    fixture writes one too: a slot picker that does not skip it reads the lock
    file and finds no index at all.
    """
    slot = root / "pit/gen/pit_views/replay/paper" / name
    slot.mkdir(parents=True, exist_ok=True)
    slot.with_suffix(".lock").touch()
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
        "execute_at", "window", "symbol", "name", "action", "quantity", "reference_price", "notional",
    }
    # The fixture's orders fill at the open, so the operator declares them in
    # the opening call auction.
    assert signal["orders"][0]["window"] == "open_auction"
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
    # The curve opens at the book's starting point — initial cash, no return,
    # the calendar day before its first settled day — and the benchmark with it.
    assert chart["series"][0]["dates"] == chart["account"]["dates"] == chart["benchmark"]["dates"] == ["20260104", *settled]
    assert chart["account"]["equity"] == [100_000.0, *(row["equity"] for row in journal)]
    assert chart["account"]["cash"] == [100_000.0, *(row["cash"] for row in journal)]
    assert chart["series"][0]["cum"][0] == 0.0 and chart["benchmark"]["cum"][0] == 0.0
    # Day-0 baseline: the first day's return is measured from the initial cash.
    assert chart["series"][0]["cum"][1] == round(journal[0]["equity"] / 100_000.0 - 1.0, 6)
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
    assert partial["benchmark_days"] == 2 and partial["chart"]["benchmark"]["dates"] == ["20260104", *settled[:2]]
    assert partial["statistics"]["benchmark_return"] is None and partial["statistics"]["excess_return"] is None


def test_a_book_names_the_index_it_froze_wherever_the_benchmark_is_spoken_of(tmp_path: Path):
    """A book is measured against its source arm's index, so its tiles, notes
    and curve all name that index; none of them may say CSI 300 for it."""
    root = engine_book(tmp_path, "20260105", "20260106", "20260107")
    write_book_record(root, benchmark_index="000905.SH")
    _csi300_slot(root, "20260102_20260108_20251231T235959+0800", {"20260105": 1.0, "20260106": 1.0})
    payload = trading.performance_payload(tmp_path, BOOK)
    assert payload["benchmark_label"] == payload["chart"]["benchmark"]["label"] == "中证500"
    # The slot carries 中证500 on one of the two settled days: a partial cover.
    assert payload["benchmark_days"] == 1 and payload["statistics"]["excess_return"] is None
    assert trading.books_payload(tmp_path)["books"][0]["benchmark_label"] == "中证500"


def _forward_result(experiment: Path) -> str:
    record = forward_record(ExperimentLedger(experiment / "ledgers/experiment_ledger.jsonl").read())
    return Path(str(record["result_ref"])).parent.name


def test_a_book_carries_the_source_curve_the_console_projects_for_that_result(tmp_path: Path):
    """The book's page continues the artifact's out-of-sample replay, and the
    experiment is archived out of experiments/ once it retires. The curve is
    therefore copied into the book when the book is created — projected by the
    same code the console projects that very result with."""
    experiments = tmp_path / "experiments"
    arm = build_arm(experiments, "exp", "graduated")
    root = engine_book(tmp_path, "20260105", "20260106")
    copy_source_history(root, arm)
    payload = trading.performance_payload(tmp_path, BOOK)
    source, console = payload["source"], result_equity_payload(experiments, "exp", _forward_result(arm))
    assert payload["source_error"] is None
    assert source["experiment_id"] == "exp" and source["heldout_start"] == REPLAY["heldout_start"]
    assert source["series"][0]["dates"] == console["series"][0]["dates"]
    assert source["series"][0]["cum"] == console["series"][0]["cum"]
    assert source["benchmark"]["final"] == console["benchmark"]["final"]
    # The overview card draws the same chained line the page does.
    assert trading.books_payload(tmp_path)["books"][0]["source"] == source
    # A book created before the copy existed has no history, which is not an
    # error: the page says so in one label instead of a red banner.
    (root / SOURCE_HISTORY_NAME).unlink()
    plain = trading.performance_payload(tmp_path, BOOK)
    assert plain["source"] is None and plain["source_error"] is None
    # A damaged or newer copy is reported rather than drawn.
    (root / SOURCE_HISTORY_NAME).write_text('{"schema_version": 99}', encoding="utf-8")
    assert trading.performance_payload(tmp_path, BOOK)["source_error"] == (
        f"unsupported {SOURCE_HISTORY_NAME} schema: 99"
    )


def test_an_arm_without_a_forward_record_has_no_history_to_copy(tmp_path: Path):
    """The forward record names the replay that carries the verdict's slices;
    without it there is nothing to copy, and init must say so rather than
    leaving a book that silently has no history."""
    sealed = build_arm(tmp_path / "experiments", "sealed_arm", "sealed")
    root = engine_book(tmp_path, "20260105")
    with pytest.raises(ValueError, match="no forward verdict record"):
        copy_source_history(root, sealed)
    assert not (root / SOURCE_HISTORY_NAME).exists()


def test_the_newest_replay_slot_is_a_directory_never_its_lock_sibling(tmp_path: Path):
    """The builder leaves a permanent ``<slot>.lock`` beside every slot, and
    that name sorts after the slot it guards. Picking it up left every reader
    of the slot — the CSI 300 benchmark — reading a file that is not a view."""
    root = tmp_path / BOOK
    for name in ("20260102_20260107_20251231T235959+0800", "20260102_20260108_20251231T235959+0800"):
        _csi300_slot(root, name, {"20260105": 1.0})
    slot = newest_replay_slot(root)
    assert slot is not None and slot.is_dir()
    assert slot.name == "20260102_20260108_20251231T235959+0800"
    assert newest_replay_slot(tmp_path / "never_run") is None


def test_a_slot_without_csi300_rows_reports_the_read_instead_of_no_data(tmp_path: Path):
    """Zero rows out of a slot that was selected is a broken read: the panel
    says which slot, rather than claiming the index has no data."""
    root = engine_book(tmp_path, "20260105", "20260106")
    _csi300_slot(root, "20260102_20260107_20251231T235959+0800", {})
    payload = trading.performance_payload(tmp_path, BOOK)
    assert payload["benchmark_days"] == 0 and payload["chart"]["benchmark"] is None
    assert "20260102_20260107" in payload["benchmark_error"]
    assert "沪深300" in payload["benchmark_error"]


def test_one_settled_day_is_already_a_curve_from_the_starting_point(tmp_path: Path):
    """The book's starting point opens the curve, so the first settled day is
    a segment, not a lone dot; a book with no settled day has no curve, and
    every surface reads the same rule."""
    engine_book(tmp_path, "20260105", "20260106")
    payload = trading.performance_payload(tmp_path, BOOK)
    chart = payload["chart"]
    assert payload["state"] == "ok" and payload["statistics"]["days"] == 1
    assert chart["series"][0]["dates"] == chart["account"]["dates"] == ["20260104", "20260105"]
    assert chart["series"][0]["cum"][0] == 0.0 and chart["series"][0]["drawdown"][0] == 0.0
    assert chart["account"]["equity"][0] == chart["account"]["cash"][0] == 100_000.0
    # The overview card draws the same curve, and only the return series of it.
    curve = trading.books_payload(tmp_path)["books"][0]["curve"]
    assert curve == {"series": chart["series"], "benchmark": chart["benchmark"]}

    engine_book(tmp_path, "20260107")
    chart = trading.performance_payload(tmp_path, BOOK)["chart"]
    assert chart["series"][0]["dates"] == chart["account"]["dates"] == ["20260104", "20260105", "20260106"]


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
    assert performance["chart"]["account"]["dates"] == ["20260104", "20260105", "20260106"]
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
    # curve; the maximum drawdown the day count gates belongs to the book page.
    assert alpha["candidate_source"] == "graduated" and "max_drawdown" not in alpha
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
    # The card's six figures are one set: they arrive together with the first
    # settled day, so the card never draws a half-filled row of tiles.
    _csi300_slot(
        paper_root(tmp_path) / "alpha",
        "20260102_20260107_20251231T235959+0800",
        {"20260105": 1.0, "20260106": -0.5},
    )
    alpha, beta = trading.books_payload(tmp_path)["books"]
    card = ("equity", "total_return", "excess_return", "cash", "position_count", "order_count")
    assert [name for name in card if alpha[name] is None] == []
    assert alpha["cash"] == snapshot["cash"]
    # beta has not settled a day: the figure the card gates on is absent, and
    # with it every tile.
    assert beta["total_return"] is None
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


def test_the_status_says_whether_the_session_ahead_has_its_order_sheet(tmp_path: Path):
    """The operator has to place the sheet's orders before the 09:25 auction
    closes, so the page asks one question: does a decision for the current
    Asia/Shanghai session exist, and when was it written. No deadline is
    encoded — a book whose latest decision is an older session simply is not
    ready."""
    root = engine_book(tmp_path, "20260105")
    earlier = trading.book_status(tmp_path, BOOK)["today"]
    assert (earlier["trade_date"], earlier["ready"]) == ("20260105", False)
    assert earlier["decided_at"].endswith("Z")

    state = json.loads((root / ".paper_state.json").read_text(encoding="utf-8"))
    state["decisions"][-1]["trade_date"] = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d")
    (root / ".paper_state.json").write_text(json.dumps(state), encoding="utf-8")
    current = trading.book_status(tmp_path, BOOK)["today"]
    assert current["ready"] is True and current["decided_at"] == earlier["decided_at"]
    # The overview card reads the same block, computed once.
    assert trading.books_payload(tmp_path)["books"][0]["today"] == current

    # A decision the engine took before it stamped one, and a book that has
    # never run, both project instead of failing.
    del state["decisions"][-1]["decided_at"]
    (root / ".paper_state.json").write_text(json.dumps(state), encoding="utf-8")
    assert trading.book_status(tmp_path, BOOK)["today"]["decided_at"] is None
    write_book_record(paper_root(tmp_path) / "fresh")
    assert trading.book_status(tmp_path, "fresh")["today"] == {
        "trade_date": None, "decided_at": None, "ready": False,
    }


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


def test_a_book_card_is_a_name_a_badge_six_figures_and_a_curve():
    """The overview is scanned, so a card carries only what distinguishes one
    book from another: its name with a state badge — the writer's error in that
    badge's tooltip alone — the six figures and the miniature curve. The book's
    frozen identity and the error in full belong to its own page."""

    script = _app_js()
    card = _js_top_level(script, "function bookCard(")
    assert "tradingBadge(row.state, row.error)" in card
    assert "row.error" not in card.replace("tradingBadge(row.state, row.error)", "")
    assert "title: error || null" in _js_top_level(script, "function tradingBadge(")
    # The identity line the page head draws as chips is not repeated here.
    for field in ("candidate_source", "artifact_id", "start_date", "initial_cash"):
        assert field not in card, field
    assert "meta-line" not in card
    assert "bookCurveChart(" in card


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


def test_the_book_curve_continues_its_source_experiment_on_one_date_axis():
    """Equity and cash were once two charts with their own widths, pads and date
    ticks, so one trading day sat at different x positions. The book's days now
    feed the research return chart, whose panes share one x-scale, with the
    source experiment's out-of-sample replay — the book's own copy of it —
    chained in front through the same chainEquity the research pages use. The
    card's miniature and the page's chart go through that one builder, so the
    two cannot drift apart."""
    script = _app_js()
    chart = _js_top_level(script, "function bookCurveChart(")
    for piece in ("chainEquity(", "equityChart(", "Paper 起始", "source.heldout_start", "chart.account"):
        assert piece in chart, piece
    for name in ("paperEquityPanel", "bookCard"):
        assert "bookCurveChart(" in _js_top_level(script, f"function {name}("), name
    # The page reads the curve off the book's own payload; nothing on this page
    # asks experiments/ for it.
    assert "payload.source" in _js_top_level(script, "function paperEquityPanel(")
    trading_section = script.split("let tradingView = null;", 1)[1].split("function renderQmtPage(", 1)[0]
    assert "singleSeriesBarChart" not in trading_section
    assert "/api/experiments/" not in trading_section
    assert "payload.account" in _js_top_level(script, "function equityChart(")
    # A book with no copied history says so in one short label.
    assert "无源实验历史" in _js_top_level(script, "function paperEquityPanel(")


def test_the_page_draws_a_figure_only_where_the_book_measured_one():
    """The panel printed 「—」 for every statistic the day count still gates and
    for a CSI 300 the book has no data for. Tiles now come from presentTiles,
    which drops an absent figure, and the reason moves into the caption."""
    script = _app_js()
    for name in ("paperEquityPanel", "bookCard"):
        body = _js_top_level(script, f"function {name}(")
        assert "presentTiles(" in body, name
        assert "—" not in body, name
    panel = _js_top_level(script, "function paperEquityPanel(")
    assert "min_days" in panel and "benchmark_days" in panel


def test_the_book_page_leads_with_what_the_operator_acts_on():
    """The operator executes the day's orders by hand, so the page is ordered by
    use: today's order sheet, then where the book stands, then what it holds
    now, and only then the history."""
    script = _app_js()
    panels = _js_top_level(script, "function renderBookBundle(")
    order = [
        panels.index(f"{name}(bundle.")
        for name in ("paperSignalPanel", "paperEquityPanel", "paperPositionsPanel", "paperHistoryPanel")
    ]
    assert order == sorted(order)
    # The sheet is copyable as plain text, for a manual order entry screen.
    assert "复制订单" in _js_top_level(script, "function copyOrdersButton(")
    assert "navigator.clipboard" in _js_top_level(script, "async function copyToClipboard(")


def test_today_and_every_past_day_render_through_one_order_sheet_renderer():
    """The signal panel showed the post-trade holdings and a history day could
    not, because the payload dropped them. One renderer now draws both, so the
    two levels of detail cannot drift apart again."""
    script = _app_js()
    for name in ("paperSignalPanel", "paperHistoryDay"):
        assert "paperSheetBody(" in _js_top_level(script, f"function {name}("), name


@pytest.mark.skipif(shutil.which("node") is None, reason="node required for the JS formatters")
def test_the_chip_says_whether_the_session_ahead_has_its_order_sheet():
    """The page answers the operator's first question before they read a number:
    is today's sheet in hand, and when was it written (UTC+8). Not ready is the
    ordinary state before the morning run, so it is muted and quotes no
    deadline; a decision the engine took before it stamped one still reads as
    ready. The overview card and the book page say it with the same chip."""
    script = _app_js()
    harness = "\n".join(
        [
            (
                "const el = (tag, attrs, ...kids) => ({ class: attrs.class, title: attrs.title || null,"
                " text: kids.filter((kid) => kid !== null && kid !== undefined).join('') });"
            ),
            _js_top_level(script, "function fmtDate("),
            _js_top_level(script, "const TS_FMT ="),
            _js_top_level(script, "function fmtTs("),
            _js_top_level(script, "function fmtClock("),
            _js_top_level(script, "function todayChip("),
            (
                "console.log(JSON.stringify(["
                "todayChip({ trade_date: '20260916', decided_at: '2026-09-16T00:32:00Z', ready: true }),"
                "todayChip({ trade_date: '20260916', decided_at: null, ready: true }),"
                "todayChip({ trade_date: '20260915', decided_at: '2026-09-15T00:32:00Z', ready: false }),"
                "todayChip({ trade_date: null, decided_at: null, ready: false }),"
                "todayChip(null)]));"
            ),
        ]
    )
    result = subprocess.run(
        ["node", "-e", harness], capture_output=True, text=True, timeout=60, check=False
    )
    assert result.returncode == 0, result.stderr
    stamped, unstamped, stale, fresh, absent = json.loads(result.stdout)
    assert stamped == {"class": "stat-chip", "title": None, "text": "今日订单已就绪 · 08:32"}
    assert unstamped["text"] == "今日订单已就绪"
    assert stale == {
        "class": "stat-chip muted",
        "title": "最新一张是 2026-09-15 的订单单",
        "text": "今日订单未生成",
    }
    assert (fresh["text"], fresh["title"]) == ("今日订单未生成", None)
    assert absent is None
    for name in ("bookCard", "paperHead"):
        assert "todayChip(" in _js_top_level(script, f"function {name}("), name


@pytest.mark.skipif(shutil.which("node") is None, reason="node required for the JS formatters")
def test_every_order_row_names_the_window_that_declares_it():
    """One sheet can mix an opening-auction order with a continuous-trading one,
    and the two have different deadlines, so the window rides on the row rather
    than in a heading. The badge and the copied text read it off one wording,
    and every row of the table fills every column the header declares."""
    script = _app_js()
    harness = "\n".join(
        [
            _js_top_level(script, "const ORDER_WINDOW_LABELS ="),
            _js_top_level(script, "function orderWindowLabel("),
            _js_top_level(script, "function largestWeight("),
            _js_top_level(script, "function paperOrdersTable("),
            "const actionCell = (v) => v, windowCell = (v) => v, weightCell = (v) => v;",
            "const fmtShares = (v) => v, fmtPrice = (v) => v, fmtAmountOpt = (v) => v;",
            (
                "const dataTable = (columns, rows) => ({ columns: columns.map((c) => c.label),"
                " widths: rows.map((row) => row.length) });"
            ),
            (
                "const sheet = { orders: ["
                "{ symbol: '000001.SZ', name: 'A', action: 'buy', quantity: 100, reference_price: 10,"
                " notional: 1000, window: 'open_auction' },"
                "{ symbol: '000002.SZ', name: 'B', action: 'sell', quantity: 200, reference_price: 20,"
                " notional: 4000, window: 'continuous' }],"
                " target: [{ symbol: '000001.SZ', weight: 0.5 }], cash_after: 1, cash_weight: 0.5 };"
            ),
            (
                "console.log(JSON.stringify([paperOrdersTable(sheet),"
                " orderWindowLabel('open_auction'), orderWindowLabel('continuous'),"
                " orderWindowLabel('')]));"
            ),
        ]
    )
    result = subprocess.run(
        ["node", "-e", harness], capture_output=True, text=True, timeout=60, check=False
    )
    assert result.returncode == 0, result.stderr
    table, auction, continuous, unknown = json.loads(result.stdout)
    assert "窗口" in table["columns"]
    # Orders and the closing cash line alike: one cell per declared column.
    assert set(table["widths"]) == {len(table["columns"])}
    assert (auction, continuous) == ("集合竞价（09:15–09:25）", "连续竞价（09:30 起）")
    assert unknown == "—"
    assert "windowCell(row.window)" in _js_top_level(script, "function paperOrdersTable(")
    # The copied text carries the same wording, from the same table.
    assert "orderWindowLabel(row.window)" in _js_top_level(script, "function orderClipboardText(")
