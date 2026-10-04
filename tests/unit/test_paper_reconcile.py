"""A Paper book that follows its owner's recorded fills (``paper.fills``).

Each book here runs the real engine over two synthetic names with the default
cost model (1 bp commission, 5 CNY minimum, transfer fee, stamp duty, 5 bp
slippage), so every figure is the Broker's own arithmetic. The wall clock is
pinned to the morning of each decided session, as the cron runs it.
"""

from __future__ import annotations

import json
from datetime import datetime as real_datetime
from pathlib import Path

import pandas as pd
import pytest

from autotrade.environment.broker import BrokerProfile
from autotrade.environment.executor import TrustedStrategyExecutor
from autotrade.environment.replay.engine import StrategyDataView
from autotrade.environment.replay.market import DailyMarketData
from autotrade.environment.strategy import CN_TZ
from autotrade.paper import engine as engine_module
from autotrade.paper.engine import (
    DailyPaperEngine,
    PaperAwaitingFills,
    PaperEngineError,
)
from autotrade.paper.fills import append_records
from autotrade.paper.orders import order_sheet, render_failure

SESSIONS = ("20260102", "20260105", "20260106", "20260107", "20260108", "20260109", "20260112")
A, B = "000001.SZ", "600000.SH"
PROFILE = BrokerProfile(initial_cash=100_000.0)

PLAN_STRATEGY = """PLAN = {plan!r}


def generate_orders(context):
    day = context.inference_at.strftime("%Y%m%d")
    stamp = context.inference_at.strftime("%Y-%m-%d")
    return [
        {{"symbol": symbol, "action": action, "quantity": quantity, "execute_at": stamp + "T" + clock + ":00+08:00"}}
        for symbol, action, quantity, clock in PLAN.get(day, [])
    ]
"""

# Stateless, like the graduated books: one seat in B, bought at the close
# whenever the account it is shown does not hold B.
SEAT_STRATEGY = """def generate_orders(context):
    stamp = context.inference_at.strftime("%Y-%m-%d")
    if "600000.SH" in context.account.positions:
        return []
    return [{"symbol": "600000.SH", "action": "buy", "quantity": 300, "execute_at": stamp + "T15:00:00+08:00"}]
"""


def _bars(days) -> pd.DataFrame:
    rows = []
    for index, day in enumerate(days):
        for offset, symbol in enumerate((A, B)):
            base = 10.0 + 5 * offset + 0.1 * index
            rows.append({
                "trade_date": day, "symbol": symbol, "open": base, "close": base + 0.05,
                "pre_close": base - 0.05, "up_limit": round(base * 1.1, 2), "down_limit": round(base * 0.9, 2),
            })
    return pd.DataFrame(rows)


class _Data:
    def __init__(self, release_end: str) -> None:
        self.sessions = SESSIONS
        self.release_end = release_end
        self.generation_id = f"gen_{release_end}"
        self.market = DailyMarketData(_bars([day for day in SESSIONS if day <= release_end]))
        self.nl_query = None

    def context_data(self, inference_at):
        return StrategyDataView("", "", inference_at.isoformat())

    def execution_price(self, symbol, when):
        return None

    def references(self, symbols, before):
        return {symbol: {"name": {A: "平安银行", B: "浦发银行"}[symbol], "close": 10.0} for symbol in symbols}

    def close(self) -> None:
        pass


class _Book:
    def __init__(self, root: Path, source: str, monkeypatch) -> None:
        self.root = root
        strategy = root.parent / f"{root.name}_strategy" / "main.py"
        strategy.parent.mkdir(parents=True, exist_ok=True)
        strategy.write_text(source, encoding="utf-8")
        self.builds = 0
        self.monkeypatch = monkeypatch
        self.engine = DailyPaperEngine(
            strategy_path=strategy, strategy_revision="revision_1", state_root=root,
            data_factory=self._data, profile=PROFILE,
            executor_factory=lambda path, _sandbox, _view, state_dir, models_dir: TrustedStrategyExecutor.from_path(
                path, state_dir=state_dir, models_dir=models_dir
            ),
        )

    def _data(self, _start: str, trade_date: str) -> _Data:
        self.builds += 1
        return _Data(max(day for day in SESSIONS if day < trade_date))

    def run(self, day: str) -> dict[str, object]:
        """Decide ``day`` at 05:45 that morning, as the cron does."""

        morning = real_datetime(int(day[:4]), int(day[4:6]), int(day[6:]), 5, 45, tzinfo=CN_TZ)

        class Clock(real_datetime):
            @classmethod
            def now(cls, tz=None):
                return morning.astimezone(tz) if tz else morning

        self.monkeypatch.setattr(engine_module, "datetime", Clock)
        return self.engine.run_day(day)

    def record(self, *records: dict[str, object]) -> None:
        append_records(self.root, records)

    def journal(self, name: str) -> list[dict[str, object]]:
        path = self.root / name
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []

    def state(self) -> dict[str, object]:
        return json.loads((self.root / ".paper_state.json").read_text(encoding="utf-8"))


def _book(tmp_path: Path, monkeypatch, *, plan=None, source=None, real=True, name="book") -> _Book:
    book = _Book(tmp_path / name, source or PLAN_STRATEGY.format(plan=plan or {}), monkeypatch)
    if real:
        book.record({"kind": "enable"})
    return book


def _without_ids(rows):
    return [{key: value for key, value in row.items() if key != "event_id"} for row in rows]


def _session(day: str, outcome: str = "simulated") -> dict[str, object]:
    return {"kind": "session", "trade_date": day, "outcome": outcome}


def _fill(day: str, clock: str, symbol: str, action: str, outcome: str = "filled", **fields) -> dict[str, object]:
    return {"kind": "fill", "trade_date": day, "time": clock, "symbol": symbol, "action": action,
            "outcome": outcome, **fields}


MIXED_PLAN = {
    "20260105": [(A, "buy", 1000, "09:30"), (B, "buy", 600, "15:00")],
    "20260106": [(A, "sell", 400, "15:00"), (B, "sell", 200, "09:30")],
    "20260107": [(B, "buy", 100, "09:30"), (A, "sell", 600, "09:30")],
}


def test_confirming_every_session_as_simulated_reproduces_the_simulated_book(tmp_path: Path, monkeypatch):
    simulated = _book(tmp_path, monkeypatch, plan=MIXED_PLAN, real=False, name="simulated")
    real = _book(tmp_path, monkeypatch, plan=MIXED_PLAN, name="real")
    for day in SESSIONS[1:6]:
        simulated.run(day)
        real.run(day)
        real.record(_session(day))
    assert real.state()["real_fills"]["after"] == ""
    assert simulated.state()["account"] == real.state()["account"]
    assert [row["positions"] for row in simulated.state()["decisions"]] == [
        row["positions"] for row in real.state()["decisions"]
    ]
    assert [row["cash"] for row in simulated.state()["decisions"]] == [row["cash"] for row in real.state()["decisions"]]
    for day in SESSIONS[1:5]:
        expected = _without_ids(simulated.journal(f"executions_{day}.jsonl"))
        assert _without_ids(real.journal(f"executions_{day}.jsonl")) == expected
        # The simulated track of a session confirmed as simulated is that session.
        assert _without_ids(real.journal(f"simulated_executions_{day}.jsonl")) == expected
    equity = _without_ids(real.journal("equity_daily.jsonl"))
    assert [row["simulated_equity"] for row in equity] == [row["equity"] for row in equity]
    assert [{key: value for key, value in row.items() if key != "simulated_equity"} for row in equity] == (
        _without_ids(simulated.journal("equity_daily.jsonl"))
    )
    assert real.journal("corrections.jsonl") == []


def test_a_recorded_price_and_quantity_move_cash_and_position_as_the_arithmetic_says(tmp_path: Path, monkeypatch):
    plan = {"20260105": [(B, "buy", 300, "15:00")], "20260106": [(B, "sell", 200, "09:30")]}
    book = _book(tmp_path, monkeypatch, plan=plan)
    book.run("20260105")
    # A board lot is the exchange's rule, so a record that breaks it is refused
    # before anything is written.
    with pytest.raises(ValueError, match="multiple of 100"):
        book.record(_fill("20260105", "15:00", B, "buy", quantity=250, price=15.3))
    assert len(book.journal("fills.jsonl")) == 1
    book.record(_fill("20260105", "15:00", B, "buy", quantity=200, price=15.3, commission=6.0))
    book.run("20260106")
    [position] = book.state()["account"]["positions"]
    assert (position["symbol"], position["quantity"]) == (B, 200)
    assert position["average_cost"] == pytest.approx((200 * 15.3 + 6.0) / 200)
    assert book.state()["account"]["cash"] == pytest.approx(100_000 - 200 * 15.3 - 6.0)
    # The strategy decided the next morning from the 200 shares it holds.
    assert book.state()["decisions"][-1]["positions"] == {B: 200}
    [row] = book.journal("executions_20260105.jsonl")
    assert (row["status"], row["quantity"], row["price"], row["commission"]) == ("filled", 200, 15.3, 6.0)
    assert row["recorded"]

    # No commission stated: the model's fee on the actual amount, stamp duty
    # by law, and the realized P&L against the cost the buy carried.
    book.record(_fill("20260106", "09:30", B, "sell", quantity=200, price=15.5))
    book.run("20260107")
    [sale] = book.journal("executions_20260106.jsonl")
    notional = 200 * 15.5
    commission = max(notional * 1e-4, 5.0) + notional * 1e-5
    stamp = notional * 5e-4
    assert (sale["commission"], sale["stamp_duty"]) == (pytest.approx(commission), pytest.approx(stamp))
    assert sale["realized_pnl"] == pytest.approx(notional - commission - stamp - (200 * 15.3 + 6.0))
    assert book.state()["account"]["positions"] == []
    assert book.state()["account"]["cash"] == pytest.approx(100_000 + sale["realized_pnl"])


def test_an_unfilled_buy_leaves_the_seat_empty_and_the_next_sheet_buys_again(tmp_path: Path, monkeypatch):
    book = _book(tmp_path, monkeypatch, source=SEAT_STRATEGY)
    book.run("20260105")
    book.record(_fill("20260105", "15:00", B, "buy", outcome="not_traded"))
    book.run("20260106")
    assert book.state()["account"]["positions"] == []
    assert book.state()["account"]["cash"] == 100_000.0
    [row] = book.journal("executions_20260105.jsonl")
    assert (row["status"], row["reason"], row["quantity"]) == ("unfilled", "not_traded", 300)
    # The simulated track bought it; the account did not.
    [simulated] = book.journal("simulated_executions_20260105.jsonl")
    assert simulated["status"] == "filled"
    [equity] = book.journal("equity_daily.jsonl")
    assert equity["equity"] == 100_000.0 and equity["simulated_equity"] < equity["equity"]
    ledger = book.state()["real_fills"]["sessions"]["20260105"]["15:00 buy 600000.SH"]
    assert (ledger["ordered"], ledger["simulated"]["quantity"], ledger["real"]["quantity"]) == (300, 300, 0)
    # The next sheet starts from the empty seat and buys it again.
    sheet = order_sheet(book.root, "20260106")
    assert sheet["decision"]["positions"] == {}
    assert [(row["symbol"], row["action"], row["quantity"], row["window"]) for row in sheet["orders"]] == [
        (B, "buy", 300, "close_auction")
    ]


def test_an_unrecorded_session_stops_the_run_before_any_data_is_built(tmp_path: Path, monkeypatch):
    book = _book(tmp_path, monkeypatch, source=SEAT_STRATEGY)
    book.run("20260105")
    builds = book.builds
    with pytest.raises(PaperAwaitingFills, match="20260105 has 1 order") as caught:
        book.run("20260106")
    assert book.builds == builds
    snapshot = json.loads((book.root / "account_snapshot.json").read_text(encoding="utf-8"))
    assert snapshot["ok"] is False and "fills not recorded" in snapshot["error"]
    # The failure sheet tells the owner where to record them.
    assert "成交记录" in render_failure(type("SheetBook", (), {"note": ""})(), "20260106", caught.value)
    # One click for the whole sheet settles it.
    book.record(_session("20260105"))
    book.run("20260106")
    assert book.state()["settled_through"] == "20260105"


def test_a_back_filled_session_was_never_tradeable_and_reads_as_not_traded(tmp_path: Path, monkeypatch):
    book = _book(tmp_path, monkeypatch, source=SEAT_STRATEGY)
    book.run("20260105")
    book.record(_session("20260105", "not_traded"))
    # 20260106 is decided only on 20260107's morning: nobody held that sheet.
    morning = real_datetime(2026, 1, 7, 5, 45, tzinfo=CN_TZ)

    class Late(real_datetime):
        @classmethod
        def now(cls, tz=None):
            return morning.astimezone(tz) if tz else morning

    monkeypatch.setattr(engine_module, "datetime", Late)
    book.engine.run_day("20260106")
    # Settled with nothing recorded for 20260106, as not traded.
    book.run("20260107")
    [row] = book.journal("executions_20260106.jsonl")
    assert (row["status"], row["recorded"]) == ("unfilled", None)
    assert book.state()["account"]["positions"] == []
    # A record still overrides it, as a correction of the settled session.
    book.record(_fill("20260106", "15:00", B, "buy", quantity=300, price=15.15))
    book.record(_session("20260107", "not_traded"))
    book.run("20260108")
    assert [row["quantity"] for row in book.state()["account"]["positions"]] == [300]


def test_a_trade_off_the_sheet_reaches_the_account_and_the_next_decision(tmp_path: Path, monkeypatch):
    book = _book(tmp_path, monkeypatch, plan={"20260105": [(B, "buy", 100, "15:00")]})
    book.run("20260105")
    book.record(_session("20260105"), _fill("20260105", "09:30", A, "buy", quantity=500, price=10.02))
    book.run("20260106")
    holdings = {row["symbol"]: row["quantity"] for row in book.state()["account"]["positions"]}
    assert holdings == {A: 500, B: 100}
    assert book.state()["decisions"][-1]["positions"] == {A: 500, B: 100}
    rows = book.journal("executions_20260105.jsonl")
    assert [(row["symbol"], row.get("off_sheet", False)) for row in rows] == [(A, True), (B, False)]
    ledger = book.state()["real_fills"]["sessions"]["20260105"]
    assert ledger["09:30 buy 000001.SZ"]["ordered"] == 0


def test_a_correction_of_a_settled_session_is_booked_on_the_next_one(tmp_path: Path, monkeypatch):
    plan = {"20260105": [(B, "buy", 300, "15:00")]}
    book = _book(tmp_path, monkeypatch, plan=plan)
    book.run("20260105")
    book.record(_session("20260105"))
    book.run("20260106")
    before = book.state()["account"]
    [held] = before["positions"]
    applied = book.state()["real_fills"]["sessions"]["20260105"]["15:00 buy 600000.SH"]["real"]
    # Only 200 filled, at 15.20, for a 5.00 commission.
    book.record(_fill("20260105", "15:00", B, "buy", quantity=200, price=15.2, commission=5.0))
    book.run("20260107")
    [correction] = book.journal("corrections.jsonl")
    assert (correction["corrects"], correction["trade_date"], correction["quantity"]) == ("20260105", "20260106", -100)
    refund = applied["notional"] + applied["commission"] - (200 * 15.2 + 5.0)
    assert correction["cash"] == pytest.approx(refund)
    after = book.state()["account"]
    [position] = after["positions"]
    assert position["quantity"] == 200
    assert position["average_cost"] == pytest.approx((200 * 15.2 + 5.0) / 200)
    # Cash is exactly what the corrected fill leaves; 20260106 had no orders.
    assert after["cash"] == pytest.approx(before["cash"] + refund)
    assert held["quantity"] == 300
    # The ledger now holds the corrected fill, so the next run books nothing.
    book.run("20260108")
    assert len(book.journal("corrections.jsonl")) == 1


def test_a_correction_that_takes_back_shares_already_sold_stops_the_run(tmp_path: Path, monkeypatch):
    plan = {"20260105": [(B, "buy", 300, "15:00")], "20260106": [(B, "sell", 300, "15:00")]}
    book = _book(tmp_path, monkeypatch, plan=plan)
    book.run("20260105")
    book.record(_session("20260105"))
    book.run("20260106")
    book.record(_session("20260106"))
    book.run("20260107")
    book.record(_fill("20260105", "15:00", B, "buy", outcome="not_traded"))
    with pytest.raises(PaperEngineError, match="correct the later sale first"):
        book.run("20260108")
    assert book.state()["settled_through"] == "20260106"


def test_a_book_never_switched_writes_exactly_what_it_wrote_before(tmp_path: Path, monkeypatch):
    """The mode is opt-in: without an ``enable`` record no reconciliation file,
    state key or journal field exists, so a simulated book is byte for byte the
    book it always was."""

    book = _book(tmp_path, monkeypatch, plan=MIXED_PLAN, real=False)
    for day in SESSIONS[1:5]:
        book.run(day)
    names = {path.name for path in book.root.iterdir()}
    assert not {"fills.jsonl", "corrections.jsonl"} & names
    assert not any(name.startswith("simulated_") for name in names)
    state = book.state()
    assert set(state) == {
        "schema_version", "strategy_revision", "strategy_path", "schedule", "profile", "start_date",
        "settled_through", "decisions", "last_fit_date", "fit_state", "pending_orders", "account",
        "emissions", "warnings", "last_error",
    }
    assert {key for row in book.journal("equity_daily.jsonl") for key in row} == {
        "event_id", "kind", "trade_date", "equity", "cash", "position_count",
    }
    assert all("recorded" not in row for day in SESSIONS[1:4] for row in book.journal(f"executions_{day}.jsonl"))
