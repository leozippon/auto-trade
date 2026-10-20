"""Paper run boundaries over real local books, engine, verdict and projections."""
from __future__ import annotations

import fcntl
from dataclasses import asdict

import pytest

from autotrade.environment.artifacts import artifact_fingerprint
from autotrade.environment.executor import TrustedStrategyExecutor
from autotrade.environment.replay.market import DailyMarketData
from autotrade.environment.strategy_loader import StrategyLoadError
from autotrade.paper.engine import (
    PAPER_STATE_NAME,
    RUN_STATUS_NAME,
    PaperEngineError,
    PaperWriterBusy,
)
from autotrade.paper.storage import read_json, read_jsonl, write_json_atomic
from autotrade.webui import trading
from tests.unit.paper_book_fixture import BOOK_SESSIONS
from tests.unit.test_paper_trading import COUNTER_STRATEGY, PROFILE, _bars, _Data
from tests.unit.test_paper_verdict import _graduated_book, _run_paper, _write_slot


def _setup(tmp_path, monkeypatch, *, losing=False):
    cli = _run_paper()
    book = _graduated_book(tmp_path)
    book.strategy_path.write_text(COUNTER_STRATEGY, encoding="utf-8")
    models = book.root / "strategy/models"
    models.mkdir()
    (models / "weights.json").write_text('{"weight": 1}', encoding="utf-8")
    record = read_json(book.root / "book.json")
    record.update(profile=asdict(PROFILE), models_dir="strategy/models",
                  artifact_fingerprint=artifact_fingerprint(book.strategy_path.parent, models))
    record["verdict_rules"]["max_drawdown"] = 0.005 if losing else 0.45
    write_json_atomic(book.root / "book.json", record)
    _write_slot(book.root, book.benchmark_index, {day: 0.0 for day in BOOK_SESSIONS})
    calls = []

    def data(**kwargs):
        target = kwargs["trade_date"]
        release = max(day for day in BOOK_SESSIONS if day < target)
        result = _Data(BOOK_SESSIONS, release)
        if losing:
            bars = _bars([day for day in BOOK_SESSIONS if day <= release])
            bars["close"] = 1.0
            result.market = DailyMarketData(bars)
        return result

    def executor(path, _sandbox, _view, state_dir, models_dir):
        calls.append(path)
        return TrustedStrategyExecutor.from_path(path, state_dir=state_dir, models_dir=models_dir)

    monkeypatch.setattr(cli, "load_sse_trading_days", lambda _raw: BOOK_SESSIONS)
    monkeypatch.setattr(cli, "BookPITData", data)
    monkeypatch.setattr(cli, "_executor_on_free_gpus", executor)
    return cli, book, calls


def _run(cli, book, tmp_path, day):
    cli.run_book("exp", book.root, day, tmp_path / "orders")


@pytest.mark.parametrize("losing", [False, True])
def test_settlement_verdict_precedes_new_decision_and_publication(tmp_path, monkeypatch, losing):
    cli, book, calls = _setup(tmp_path, monkeypatch, losing=losing)
    _run(cli, book, tmp_path, "20260105")
    historical = (book.root / "orders_20260105.jsonl").read_bytes()
    _run(cli, book, tmp_path, "20260106")
    state = read_json(book.root / PAPER_STATE_NAME)
    assert state["settled_through"] == "20260105"
    assert (book.root / "orders_20260105.jsonl").read_bytes() == historical
    assert len(read_jsonl(book.root / "equity_daily.jsonl")[0]) == 1
    sheet = (tmp_path / "orders/exp/latest_orders.md").read_text(encoding="utf-8")
    signal = trading.signal_payload(tmp_path, "exp")["signal"]
    if losing:
        assert len(calls) == 1  # no historical replay or new strategy call
        assert [row["trade_date"] for row in state["decisions"]] == ["20260105"]
        assert not (book.root / "orders_20260106.jsonl").exists()
        assert "killed — exit these holdings by hand" in sheet
        assert signal["instruction"] == "exit" and signal["orders"] == []
        assert signal["target"][0]["quantity"] == 100
        assert len(read_jsonl(book.root / "verdict.jsonl")[0]) == 1
        assert trading.history_payload(tmp_path, "exp")["days"]  # evidence remains available
        before = (book.root / PAPER_STATE_NAME).read_bytes()
        _run(cli, book, tmp_path, "20260106")
        assert (book.root / PAPER_STATE_NAME).read_bytes() == before
    else:
        assert len(calls) == 2
        assert [row["trade_date"] for row in state["decisions"]] == ["20260105", "20260106"]
        assert "已终止" not in sheet and signal["trade_date"] == "20260106"


@pytest.mark.parametrize("file", ["main.py", "helper.py", "models"])
def test_tampered_complete_artifact_refused_visibly_before_execution(tmp_path, monkeypatch, file):
    cli, book, calls = _setup(tmp_path, monkeypatch)
    _run(cli, book, tmp_path, "20260105")
    before = (book.root / PAPER_STATE_NAME).read_bytes()
    target = book.root / "strategy/models/weights.json" if file == "models" else book.strategy_path.parent / file
    target.write_text('tampered = True\n', encoding="utf-8")
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        _run(cli, book, tmp_path, "20260106")
    assert len(calls) == 1
    assert (book.root / PAPER_STATE_NAME).read_bytes() == before
    assert "fingerprint mismatch" in (tmp_path / "orders/exp/latest_orders.md").read_text(encoding="utf-8")
    assert trading.book_status(tmp_path, "exp")["state"] == "export_error"
    assert trading.signal_payload(tmp_path, "exp")["signal"] is None
    assert trading.health_payload(tmp_path)["ok"] is False


@pytest.mark.parametrize(
    ("failure", "expected"),
    [
        ("book", ValueError),  # malformed book.json
        ("calendar", FileNotFoundError),
        ("strategy", StrategyLoadError),
        ("restoration", PaperEngineError),
        ("verdict", PaperEngineError),
    ],
)
def test_startup_and_restoration_failures_have_same_visible_boundary(tmp_path, monkeypatch, failure, expected):
    cli, book, calls = _setup(tmp_path, monkeypatch)
    checkpoint = book.root / PAPER_STATE_NAME
    if failure == "book":
        (book.root / "book.json").write_text("{", encoding="utf-8")
    elif failure == "calendar":
        def missing(_raw):
            raise FileNotFoundError("calendar missing")
        monkeypatch.setattr(cli, "load_sse_trading_days", missing)
    elif failure == "strategy":
        book.strategy_path.write_text("not valid python!", encoding="utf-8")
        record = read_json(book.root / "book.json")
        record["artifact_fingerprint"] = artifact_fingerprint(book.strategy_path.parent, book.root / "strategy/models")
        write_json_atomic(book.root / "book.json", record)
    elif failure == "restoration":
        _run(cli, book, tmp_path, "20260105")
        state = read_json(checkpoint)
        state["account"]["cash"] = "invalid"
        write_json_atomic(checkpoint, state)
    else:
        _run(cli, book, tmp_path, "20260105")
        # Settlement succeeds, but the persistent evidence is damaged.
        (book.root / "equity_daily.jsonl").write_text("damaged\n", encoding="utf-8")
    before = checkpoint.read_bytes() if checkpoint.exists() else None
    with pytest.raises(expected):
        _run(cli, book, tmp_path, "20260106")
    assert read_json(book.root / RUN_STATUS_NAME)["ok"] is False
    assert "未生成" in (tmp_path / "orders/exp/latest_orders.md").read_text(encoding="utf-8")
    assert trading.book_status(tmp_path, "exp")["state"] == "export_error"
    assert trading.signal_payload(tmp_path, "exp")["signal"] is None
    if failure != "verdict":
        assert (checkpoint.read_bytes() if checkpoint.exists() else None) == before
    else:
        assert len(read_json(checkpoint)["decisions"]) == 1
    if failure in {"book", "calendar", "strategy"}:
        assert calls == [] and not checkpoint.exists()


def test_busy_writer_does_not_overwrite_health_or_sheet(tmp_path, monkeypatch):
    cli, book, calls = _setup(tmp_path, monkeypatch)
    _run(cli, book, tmp_path, "20260105")
    status = (book.root / RUN_STATUS_NAME).read_bytes()
    sheet = (tmp_path / "orders/exp/latest_orders.md").read_bytes()
    with (book.root / ".paper_engine.lock").open("a+b") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(PaperWriterBusy):
            _run(cli, book, tmp_path, "20260106")
    assert (book.root / RUN_STATUS_NAME).read_bytes() == status
    assert (tmp_path / "orders/exp/latest_orders.md").read_bytes() == sheet
    assert len(calls) == 1


@pytest.mark.parametrize("damage", ["missing_state", "malformed_state", "missing_account", "invalid_account",
                                   "missing_positions", "invalid_positions"])
def test_killed_book_requires_known_account_and_explicit_positions(tmp_path, monkeypatch, damage):
    cli, book, _calls = _setup(tmp_path, monkeypatch, losing=True)
    _run(cli, book, tmp_path, "20260105")
    _run(cli, book, tmp_path, "20260106")
    checkpoint = book.root / PAPER_STATE_NAME
    state = read_json(checkpoint)
    if damage == "missing_state":
        checkpoint.unlink()
    elif damage == "malformed_state":
        checkpoint.write_text("{", encoding="utf-8")
    else:
        if damage == "missing_account":
            state.pop("account")
        elif damage == "invalid_account":
            state["account"] = None
        elif damage == "missing_positions":
            state["account"].pop("positions")
        else:
            state["account"]["positions"] = {}
        write_json_atomic(checkpoint, state)
    payload = trading.signal_payload(tmp_path, "exp")
    assert payload["state"] == "unreadable" and payload["signal"] is None and payload["error"]
    with pytest.raises(ValueError):
        _run(cli, book, tmp_path, "20260106")  # render_killed uses the same refusal
    assert "未生成" in (tmp_path / "orders/exp/latest_orders.md").read_text(encoding="utf-8")
    assert read_json(book.root / RUN_STATUS_NAME)["ok"] is False


def test_valid_empty_killed_account_still_exits_despite_prior_operational_error(tmp_path, monkeypatch):
    cli, book, _calls = _setup(tmp_path, monkeypatch, losing=True)
    _run(cli, book, tmp_path, "20260105")
    _run(cli, book, tmp_path, "20260106")
    state = read_json(book.root / PAPER_STATE_NAME)
    state["account"]["positions"] = []
    write_json_atomic(book.root / PAPER_STATE_NAME, state)
    write_json_atomic(book.root / RUN_STATUS_NAME, {"ok": False, "error": "prior operational error"})
    payload = trading.signal_payload(tmp_path, "exp")
    assert payload["state"] == "ok"
    assert payload["signal"]["instruction"] == "exit" and payload["signal"]["target"] == []
    _run(cli, book, tmp_path, "20260106")
    assert "账户没有持仓。" in (tmp_path / "orders/exp/latest_orders.md").read_text(encoding="utf-8")


def test_unreadable_run_status_suppresses_historical_buy_signal(tmp_path, monkeypatch):
    cli, book, _calls = _setup(tmp_path, monkeypatch)
    _run(cli, book, tmp_path, "20260105")
    assert trading.signal_payload(tmp_path, "exp")["signal"]["orders"][0]["action"] == "buy"
    (book.root / RUN_STATUS_NAME).write_text("{", encoding="utf-8")
    payload = trading.signal_payload(tmp_path, "exp")
    status = trading.snapshot_payload(tmp_path, "exp")
    assert payload["state"] == status["state"] == "unreadable"
    assert payload["error"] == status["error"] and payload["signal"] is None
