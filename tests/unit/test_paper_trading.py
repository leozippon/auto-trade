from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from autotrade.environment.broker import BrokerProfile
from autotrade.environment.data.snapshot import SnapshotConfig
from autotrade.environment.executor import TrustedStrategyExecutor
from autotrade.environment.replay.engine import StrategyDataView
from autotrade.environment.replay.market import DailyMarketData
from autotrade.environment.strategy import StrategySchedule
from autotrade.paper import DailyPaperEngine, PaperEngineError
from autotrade.paper.engine import PaperDataNotReady
from autotrade.paper.orders import (
    LATEST_NAME,
    order_sheet,
    render_failure,
    render_orders,
    write_orders,
)

SESSIONS = ("20260105", "20260106", "20260107", "20260108", "20260109", "20260112")
SYMBOL = "000001.SZ"
PROFILE = BrokerProfile(initial_cash=100_000.0, min_commission_cny=0, slippage_bps=0, transfer_fee_bps=0)

# A module-level decision counter, the way the graduated arm paces its
# rebalance: it buys on the first call of a worker and sells on the third.
COUNTER_STRATEGY = """_CALLS = 0


def generate_orders(context):
    global _CALLS
    call = _CALLS
    _CALLS += 1
    day = context.inference_at.strftime("%Y-%m-%d")
    if call % 3 == 0 and not context.account.positions:
        return [{"symbol": "000001.SZ", "action": "buy", "quantity": 100, "execute_at": day + "T09:30:00+08:00", "call": call}]
    if call % 3 == 2 and context.account.positions:
        return [{"symbol": "000001.SZ", "action": "sell", "quantity": 100, "execute_at": day + "T09:30:00+08:00", "call": call}]
    return []
"""

FIT_STRATEGY = """import numpy as np

REFIT_PERIOD = "month"


def fit(context):
    if context.inference_at.strftime("%Y%m%d") == "20260202":
        raise RuntimeError("fit refuses this day")
    np.save(context.state_dir + "/fitted_on.npy", np.array([int(context.inference_at.strftime("%Y%m%d"))]))


def generate_orders(context):
    fitted_on = int(np.load(context.state_dir + "/fitted_on.npy")[0])
    day = context.inference_at.strftime("%Y-%m-%d")
    return [{"symbol": "000001.SZ", "action": "buy", "quantity": 100, "execute_at": day + "T09:30:00+08:00", "fitted_on": fitted_on}]
"""


def _bars(days) -> pd.DataFrame:
    return pd.DataFrame([
        {"trade_date": day, "symbol": SYMBOL, "open": 10.0 + index, "close": 10.5 + index,
         "pre_close": 10.5 + index - 1, "up_limit": 30.0, "down_limit": 1.0}
        for index, day in enumerate(days)
    ])


class _Data:
    """The committed data of one run: bars through ``release_end``."""

    def __init__(self, sessions, release_end: str, *, execution_price=None) -> None:
        self.sessions = tuple(sessions)
        self.release_end = release_end
        self.generation_id = f"gen_{release_end}"
        self.market = DailyMarketData(_bars([day for day in self.sessions if day <= release_end]))
        self.nl_query = None
        self.views: list[str] = []
        self.closed = False
        self._execution_price = execution_price

    def context_data(self, inference_at):
        self.views.append(inference_at.isoformat())
        return StrategyDataView("", "", str(len(self.views)))

    def execution_price(self, symbol, when):
        return self._execution_price(symbol, when) if self._execution_price else None

    def references(self, symbols, before):
        return {symbol: {"name": "平安银行", "close": 10.5} for symbol in symbols}

    def close(self) -> None:
        self.closed = True


class _Book:
    """A book whose committed data ends wherever the test says it does."""

    def __init__(self, tmp_path: Path, source: str, *, sessions=SESSIONS, schedule=None) -> None:
        self.strategy = tmp_path / "strategy" / "main.py"
        self.strategy.parent.mkdir(parents=True, exist_ok=True)
        self.strategy.write_text(source, encoding="utf-8")
        self.root = tmp_path / "paper"
        self.sessions = sessions
        self.release_end = sessions[0]
        self.built: list[_Data] = []
        self.execution_price = None
        self.engine = DailyPaperEngine(
            strategy_path=self.strategy,
            strategy_revision="revision_1",
            state_root=self.root,
            data_factory=self._data,
            schedule=schedule,
            profile=PROFILE,
            executor_factory=lambda path, _sandbox, _view, state_dir, models_dir: TrustedStrategyExecutor.from_path(
                path, state_dir=state_dir, models_dir=models_dir
            ),
        )

    def _data(self, start: str, trade_date: str) -> _Data:
        data = _Data(self.sessions, self.release_end, execution_price=self.execution_price)
        self.built.append(data)
        return data

    def run(self, trade_date: str) -> dict[str, object]:
        """Run the pre-open decision for ``trade_date`` once the prior session landed."""

        self.release_end = max(day for day in self.sessions if day < trade_date)
        return self.engine.run_day(trade_date)

    def journal(self, name: str) -> list[dict[str, object]]:
        path = self.root / name
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []

    def state(self) -> dict[str, object]:
        return json.loads((self.root / ".paper_state.json").read_text(encoding="utf-8"))


def _sheet_book(book: _Book, note: str = "") -> object:
    """The frozen identity ``render_orders`` reads beside the book's own files."""

    return type("SheetBook", (), {
        "root": book.root.resolve(), "note": note,
        "experiment_id": "exp", "artifact_id": "art", "candidate_source": "graduated",
        "profile": PROFILE, "schedule": StrategySchedule(),
    })()


def test_orders_exist_before_the_session_bars_and_fill_on_the_next_run(tmp_path: Path):
    book = _Book(tmp_path, COUNTER_STRATEGY, sessions=("20260102", *SESSIONS))
    first = book.run("20260105")
    assert first["trade_date"] == "20260105"
    assert [row["action"] for row in book.journal("orders_20260105.jsonl")] == ["buy"]
    # The session's own bars have not landed: nothing filled yet.
    assert book.journal("executions_20260105.jsonl") == []
    assert first["pending_order_count"] == 1 and first["position_count"] == 0
    assert book.built[-1].closed

    # The wall clock the decision was written at, beside the scheduled PIT
    # instant: an Asia/Shanghai stamp, as every writer's stamp is.
    decided_at = book.state()["decisions"][-1]["decided_at"]
    assert decided_at.endswith("+08:00")

    # Idempotent: a decided session is answered from the state alone, so the
    # later runs of the same morning never re-decide or re-stamp it.
    assert book.engine.run_day("20260105") == first
    assert len(book.built) == 1
    assert book.state()["decisions"][-1]["decided_at"] == decided_at

    book.run("20260106")
    [fill] = book.journal("executions_20260105.jsonl")
    assert (fill["status"], fill["price"]) == ("filled", 11.0)
    assert [row["trade_date"] for row in book.journal("equity_daily.jsonl")] == ["20260105"]
    state = book.state()
    assert state["settled_through"] == "20260105"
    assert state["decisions"][-1]["positions"] == {SYMBOL: 100}
    assert state["decisions"][-1]["data_through"] == "20260105"


def test_earlier_decisions_are_replayed_so_module_state_spans_the_book(tmp_path: Path):
    """One research worker buys on calls 0 and 3 and sells on call 2; a Paper
    run is a fresh worker every morning, which on its own would never sell."""

    book = _Book(tmp_path, COUNTER_STRATEGY, sessions=("20260102", *SESSIONS))
    for day in SESSIONS[:4]:
        book.run(day)
    actions = {day: [row["action"] for row in book.journal(f"orders_{day}.jsonl")] for day in SESSIONS[:4]}
    assert actions == {"20260105": ["buy"], "20260106": [], "20260107": ["sell"], "20260108": ["buy"]}
    decisions = book.state()["decisions"]
    assert [row["replayed_calls"] for row in decisions] == [0, 1, 2, 3]
    # Same data and account, so every replayed call reproduced its journal.
    assert [row["replayed_matching_journal"] for row in decisions] == [0, 1, 2, 3]
    # One Timeview refresh per replayed and real call, in time order.
    assert book.built[-1].views == [f"{day[:4]}-{day[4:6]}-{day[6:]}T08:30:00+08:00" for day in SESSIONS[:4]]


def test_fitted_state_persists_and_refits_only_when_the_period_rolls(tmp_path: Path):
    sessions = ("20260128", "20260129", "20260130", "20260202", "20260203")
    book = _Book(tmp_path, FIT_STRATEGY, sessions=sessions)
    book.run("20260129")
    book.run("20260130")
    fitted = [row["fitted_on"] for day in ("20260129", "20260130") for row in book.journal(f"orders_{day}.jsonl")]
    assert fitted == [20260129, 20260129]
    state = book.state()
    assert (state["last_fit_date"], state["fit_state"]) == ("20260129", "20260129")

    # The February refit fails: the run fails, the January state is intact
    # and no staged copy is left behind.
    with pytest.raises(PaperEngineError, match="fit failed"):
        book.run("20260202")
    state = book.state()
    assert (state["last_fit_date"], len(state["decisions"])) == ("20260129", 2)
    assert [entry.name for entry in (book.root / ".strategy_state").iterdir()] == ["20260129"]
    snapshot = json.loads((book.root / "account_snapshot.json").read_text(encoding="utf-8"))
    assert snapshot["ok"] is False and "fit failed" in snapshot["error"]


def test_a_month_rollover_refits_into_a_new_state_and_drops_the_old_one(tmp_path: Path):
    sessions = ("20260129", "20260130", "20260202")
    book = _Book(tmp_path, FIT_STRATEGY.replace('"20260202"', '"never"'), sessions=sessions)
    book.run("20260130")
    book.run("20260202")
    assert [row["fitted_on"] for row in book.journal("orders_20260202.jsonl")] == [20260202]
    assert [entry.name for entry in (book.root / ".strategy_state").iterdir()] == ["20260202"]
    assert [row["fitted"] for row in book.state()["decisions"]] == [True, True]


def test_a_run_fails_fast_until_the_prior_session_is_committed(tmp_path: Path):
    book = _Book(tmp_path, COUNTER_STRATEGY, sessions=("20260102", *SESSIONS))
    book.run("20260105")
    book.release_end = "20260102"  # the evening update has not committed 20260105
    with pytest.raises(PaperDataNotReady, match="needs sessions through 20260105"):
        book.engine.run_day("20260106")
    state = book.state()
    assert [row["trade_date"] for row in state["decisions"]] == ["20260105"]
    assert state["settled_through"] == "" and "PaperDataNotReady" in state["last_error"]
    assert book.built[-1].closed


def test_a_skipped_session_must_be_decided_before_a_later_one(tmp_path: Path):
    book = _Book(tmp_path, COUNTER_STRATEGY, sessions=("20260102", *SESSIONS))
    book.run("20260105")
    with pytest.raises(PaperEngineError, match="session 20260106 was never decided"):
        book.run("20260107")
    assert book.journal("executions_20260105.jsonl") == []
    with pytest.raises(PaperEngineError, match="cannot decide the earlier session"):
        book.engine.run_day("20260102")


def test_t_plus_one_and_exact_minute_prices_hold_across_the_split_day(tmp_path: Path):
    source = """def generate_orders(context):
    day = context.inference_at.strftime("%Y-%m-%d")
    if context.account.positions:
        return [{"symbol": "000001.SZ", "action": "sell", "quantity": 100, "execute_at": day + "T10:00:00+08:00"}]
    return [
        {"symbol": "000001.SZ", "action": "buy", "quantity": 100, "execute_at": day + "T09:30:00+08:00"},
        {"symbol": "000001.SZ", "action": "sell", "quantity": 100, "execute_at": day + "T15:00:00+08:00"},
    ]
"""
    book = _Book(tmp_path, source, sessions=("20260102", *SESSIONS))
    quotes: list[str] = []
    book.execution_price = lambda symbol, when: quotes.append(when.isoformat()) or 11.25
    book.run("20260105")
    book.run("20260106")
    book.run("20260107")
    first = book.journal("executions_20260105.jsonl")
    assert [(row["action"], row["status"], row["reason"]) for row in first] == [
        ("buy", "filled", None),
        ("sell", "rejected", "insufficient_available_position"),
    ]
    [sold] = book.journal("executions_20260106.jsonl")
    assert (sold["status"], sold["price"], sold["matched_at"]) == ("filled", 11.25, "2026-01-06T10:00:00+08:00")
    assert quotes == ["2026-01-06T10:00:00+08:00"]


def test_the_account_identity_is_frozen_with_the_book(tmp_path: Path):
    book = _Book(tmp_path, COUNTER_STRATEGY, sessions=("20260102", *SESSIONS))
    book.run("20260105")
    assert book.state()["schema_version"] == 4

    def reopened(**changes):
        arguments = {
            "strategy_path": book.strategy, "strategy_revision": "revision_1", "state_root": book.root,
            "data_factory": book._data, "profile": PROFILE, **changes,
        }
        return DailyPaperEngine(**arguments)

    with pytest.raises(PaperEngineError, match="strategy schedule differs"):
        reopened(schedule=StrategySchedule("day", "09:00")).run_day("20260106")
    with pytest.raises(PaperEngineError, match="Broker profile differs"):
        reopened(profile=BrokerProfile(initial_cash=5_000_000.0)).run_day("20260106")
    with pytest.raises(PaperEngineError, match="strategy revision changed"):
        reopened(strategy_revision="revision_2").run_day("20260106")


def test_the_book_refuses_orphan_journals_and_foreign_state(tmp_path: Path):
    book = _Book(tmp_path, COUNTER_STRATEGY, sessions=("20260102", *SESSIONS))
    book.root.mkdir()
    (book.root / "orders_20260105.jsonl").write_text("{}\n", encoding="utf-8")
    with pytest.raises(PaperEngineError, match="state.*missing"):
        book.run("20260105")
    (book.root / "orders_20260105.jsonl").unlink()
    (book.root / ".paper_state.json").write_text(json.dumps({"schema_version": 3}), encoding="utf-8")
    with pytest.raises(PaperEngineError, match="unsupported Paper state schema"):
        book.run("20260105")
    assert book.built == []


def test_the_order_sheet_matches_the_journal(tmp_path: Path):
    book = _Book(tmp_path, COUNTER_STRATEGY, sessions=("20260102", *SESSIONS))
    book.run("20260105")
    book.run("20260106")
    book.run("20260107")
    sheet_book = _sheet_book(book, note="参考簿（观察中）：graduated, unconfirmed — monitor")
    text = render_orders(sheet_book, "20260107")
    assert text.startswith("# Paper 订单 · 2026-01-07（周三）\n\n> 参考簿（观察中）：graduated, unconfirmed — monitor")
    assert "数据截至 2026-01-06（周二） 收盘" in text
    # A held name is quoted at the close the account was marked with (20260106: 12.50).
    assert "总资产 ¥100,149.89，现金 ¥98,899.89，持仓 1 只" in text
    assert "| 09:30 | 000001.SZ | 平安银行 | 卖出 | 100 | 12.50 | ¥1,250.00 |" in text
    assert "下单窗口：全部 1 笔以当日开盘价成交，请在集合竞价（09:15–09:25）内申报。" in text
    assert "## 成交后目标持仓（0 只）" in text
    assert "重放此前决策 2 次，其中 2 次与账簿记录的订单一致" in text
    idle = render_orders(sheet_book, "20260106")
    # An old sheet still renders from its own morning's record (20260105 close: 11.50).
    assert "今日无订单，持仓不变。" in idle and "| 000001.SZ | 平安银行 | 100 | 11.50 | ¥1,150.00 |" in idle

    path = write_orders(tmp_path / "logs", "20260107", text)
    assert path.read_text(encoding="utf-8") == (tmp_path / "logs" / LATEST_NAME).read_text(encoding="utf-8") == text
    failure = render_failure(sheet_book, "20260108", PaperDataNotReady("committed data ends at 20260106"))
    assert "未生成" in failure and "PaperDataNotReady: committed data ends at 20260106" in failure


def test_the_sheet_says_where_each_order_is_declared(tmp_path: Path):
    """The operator places these orders by hand, so each one names its window:
    an order filling at the open is matched by the 09:15-09:25 call auction,
    any other execute_at is declared once continuous trading opens."""

    source = """def generate_orders(context):
    day = context.inference_at.strftime("%Y-%m-%d")
    return [
        {"symbol": "000001.SZ", "action": "buy", "quantity": 100, "execute_at": day + "T09:30:00+08:00"},
        {"symbol": "000001.SZ", "action": "buy", "quantity": 100, "execute_at": day + "T14:00:00+08:00"},
    ]
"""
    book = _Book(tmp_path, source, sessions=("20260102", *SESSIONS))
    book.run("20260105")
    assert [row["window"] for row in order_sheet(book.root, "20260105")["orders"]] == [
        "open_auction", "continuous",
    ]
    text = render_orders(_sheet_book(book), "20260105")
    assert (
        "下单窗口：09:30 成交的 1 笔请在集合竞价（09:15–09:25）内申报，"
        "其余 1 笔在连续竞价（09:30 起）按各自时间下单。"
    ) in text


def test_a_pre_open_book_reads_a_real_pit_window_without_the_session_bars(tmp_path: Path):
    """The pre-open state of 20211011 on a synthetic lake: 20211008 has landed,
    20211011 has no bars. The decision sees the prior close of a same-day
    macro table, not its own, and the book's first orders wait for the bars."""

    from autotrade.paper.pit import BookPITData

    from .test_snapshot_builder import (
        build_fundamental_events,
        build_raw,
        write_fundamental_status,
        write_index_daily,
    )

    raw = tmp_path / "data" / "raw"
    build_raw(raw)
    write_index_daily(raw, ("20210930", "20211008"))
    events_root = tmp_path / "data" / "pit" / "fundamental_events"
    build_fundamental_events(events_root)
    status = tmp_path / "results" / "data_quality" / "fundamental_events_status.json"
    status.parent.mkdir(parents=True)
    write_fundamental_status(status)
    config = SnapshotConfig(
        events_datasets=(), macro_datasets=("index_daily",), text_datasets=("cctv_news",),
        fundamental_datasets=("income_vip",), include_intraday=False, include_industry=False,
    )
    seen: dict[str, object] = {}

    class MacroReader:
        def execute(self, context):
            macro = pd.read_parquet(Path(context.asof_dir) / "macro")
            seen["index_dates"] = sorted(macro[macro["dataset"] == "index_daily"]["trade_date"])
            return [{"symbol": "000001.SZ", "action": "buy", "quantity": 100, "execute_at": "2021-10-11T09:30:00+08:00"}]

        def close(self) -> None:
            pass

    strategy = tmp_path / "main.py"
    strategy.write_text("def generate_orders(context):\n    return []\n", encoding="utf-8")
    state_root = tmp_path / "paper"
    engine = DailyPaperEngine(
        strategy_path=strategy,
        strategy_revision="revision_1",
        state_root=state_root,
        data_factory=lambda start, trade_date: BookPITData(
            state_root=state_root, raw_dir=raw, fundamental_events_root=events_root,
            fundamental_events_status=status, snapshot_config=config, start=start, trade_date=trade_date,
        ),
        executor_factory=lambda *_mounts: MacroReader(),
    )
    summary = engine.run_day("20211011")
    assert seen["index_dates"] == ["20210930", "20211008"]
    assert summary["pending_order_count"] == 1
    [order] = [json.loads(line) for line in (state_root / "orders_20211011.jsonl").read_text(encoding="utf-8").splitlines()]
    assert order["paper_reference"]["close"] == 10.5
    assert json.loads((state_root / ".paper_state.json").read_text(encoding="utf-8"))["decisions"][0]["data_through"] == "20211008"
    # The run's as-of view is discarded; the release-scoped cache stays.
    assert [entry.name for entry in (state_root / "pit").iterdir()] == ["live"]
    assert not any((state_root / "pit" / "live" / "runtime").rglob("asof"))


def test_late_data_is_refused_before_any_pit_view_is_built(tmp_path: Path):
    from autotrade.paper.pit import BookPITData

    from .test_snapshot_builder import (
        build_fundamental_events,
        build_raw,
        write_fundamental_status,
    )

    raw = tmp_path / "data" / "raw"
    build_raw(raw)
    (raw / "daily" / "trade_date=20211008.parquet").unlink()  # the evening update has not landed
    events_root = tmp_path / "data" / "pit" / "fundamental_events"
    build_fundamental_events(events_root)
    status = tmp_path / "results" / "data_quality" / "fundamental_events_status.json"
    status.parent.mkdir(parents=True)
    write_fundamental_status(status)
    with pytest.raises(PaperDataNotReady, match="needs sessions through 20211008"):
        BookPITData(
            state_root=tmp_path / "paper", raw_dir=raw, fundamental_events_root=events_root,
            fundamental_events_status=status,
            snapshot_config=SnapshotConfig(
                events_datasets=(), macro_datasets=(), text_datasets=(), fundamental_datasets=("income_vip",),
                include_intraday=False, include_industry=False,
            ),
            start="20211011", trade_date="20211011",
        )
    assert not (tmp_path / "paper" / "pit" / "live" / "pit_views" / "decision").exists()
