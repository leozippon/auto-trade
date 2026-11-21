"""The consistency check between a Paper book and a historical replay of it."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pandas as pd
import pytest

from autotrade.data_sources.tushare.cron_update import DEFAULT_CONFIG, load_config
from autotrade.environment.artifacts import artifact_fingerprint
from autotrade.environment.broker import BrokerProfile
from autotrade.environment.data.contracts import DEFAULT_BENCHMARK_INDEX
from autotrade.environment.data.snapshot import SnapshotConfig
from autotrade.environment.executor import TrustedStrategyExecutor
from autotrade.environment.sandbox import SandboxConfig
from autotrade.environment.strategy import StrategySchedule
from autotrade.paper.book import Book
from autotrade.paper.consistency import (
    CONSISTENCY_NAME,
    BookRecords,
    CheckFailed,
    check_book,
    compare,
)
from autotrade.paper.engine import DailyPaperEngine, writer_lock
from autotrade.paper.pit import book_data_factory
from tests.unit.support.paper_book_fixture import COUNTER_STRATEGY, _Book
from tests.unit.support.snapshot_fixture import (
    build_fundamental_events,
    build_raw,
    write,
    write_fundamental_status,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DAYS = ("20260105", "20260106", "20260107")
SYMBOL = "000001.SZ"


def _execution(day: str, action: str = "buy", price: float = 10.005, **fields: object) -> dict[str, object]:
    stamp = f"{day[:4]}-{day[4:6]}-{day[6:]}T09:30:00+08:00"
    return {
        "symbol": SYMBOL, "action": action, "quantity": 100, "execute_at": stamp, "matched_at": stamp,
        "status": "filled", "price": price, "commission": 5.0, "stamp_duty": 0.0, "dividend_tax": 0.0,
        "realized_pnl": None, "reason": None, "metadata": {}, **fields,
    }


def _matching() -> tuple[BookRecords, dict[str, object]]:
    """A book that bought on the first session and sold on the third, and the
    replay that did exactly the same."""

    buy, sell = _execution(DAYS[0]), _execution(DAYS[2], "sell", 10.2, realized_pnl=9.5)
    closes = [(98994.5, 100044.5, {SYMBOL: 100}), (98994.5, 100094.5, {SYMBOL: 100}), (100009.5, 100009.5, {})]
    decisions = [
        {"trade_date": day, "decided_at": f"{day[:4]}-{day[4:6]}-{day[6:]}T05:45:00+08:00", "generation_id": "g_paper",
         "vintages": {"daily": "v1", "anns_d": "v1"}}
        for day in (*DAYS, "20260108")
    ]
    order = {key: buy[key] for key in ("symbol", "action", "quantity", "execute_at")}
    records = BookRecords(
        state={"start_date": DAYS[0], "settled_through": DAYS[-1], "decisions": decisions},
        orders={DAYS[0]: [order], DAYS[2]: [{**order, "action": "sell", "execute_at": sell["execute_at"]}]},
        executions={DAYS[0]: [buy], DAYS[2]: [sell]},
        corporate_actions={},
        equity=[
            {"trade_date": day, "cash": cash, "equity": equity, "position_count": len(held)}
            for day, (cash, equity, held) in zip(DAYS, closes, strict=True)
        ],
        corrections=[],
    )
    replay = {
        "equity_curve": [
            {"trade_date": day, "cash": cash, "equity": equity, "positions": held}
            for day, (cash, equity, held) in zip(DAYS, closes, strict=True)
        ],
        "executions": [copy.deepcopy(buy), copy.deepcopy(sell)],
        "corporate_actions": [],
    }
    return records, replay


VINTAGES = {"daily": "v1", "anns_d": "v2"}


def test_identical_records_are_consistent_and_nothing_settled_is_never_consistent():
    records, replay = _matching()
    result = compare(records, replay, vintages=VINTAGES)
    assert (result["status"], result["sessions_compared"], result["first_difference"]) == ("consistent", 3, None)
    assert all(row["opening"] and row["orders"] and row["fills"] and row["close"] for row in result["sessions"])

    empty = BookRecords({"decisions": []}, {}, {}, {}, [], [])
    with pytest.raises(CheckFailed, match="no settled session"):
        compare(empty, {"equity_curve": [], "executions": [], "corporate_actions": []}, vintages=VINTAGES)
    # A replay that does not reach every settled session is no comparison.
    replay["equity_curve"] = replay["equity_curve"][:2]
    with pytest.raises(CheckFailed, match=r"the replay covers 20260105\.\.20260106"):
        compare(records, replay, vintages=VINTAGES)


def test_an_order_paper_alone_holds_is_the_first_difference_and_what_follows_is_downstream():
    records, replay = _matching()
    extra = _execution(DAYS[1], "sell")
    records.executions[DAYS[1]] = [{**extra, "status": "rejected", "price": None, "commission": 0.0, "reason": "t1"}]
    records.orders[DAYS[1]] = [{key: extra[key] for key in ("symbol", "action", "quantity", "execute_at")}]
    result = compare(records, replay, vintages=VINTAGES)
    first = result["first_difference"]
    assert (result["status"], first["trade_date"], first["level"], first["cause"]) == ("differs", DAYS[1], "orders", "unresolved")
    evidence = first["evidence"]
    assert (evidence["decision"], len(evidence["paper_only"]), evidence["replay_only"]) == (DAYS[1], 1, [])
    # The mounted datasets restated since Paper's decision, the only lead left.
    assert evidence["restated"] == ["anns_d"]
    assert [row["downstream"] for row in result["sessions"]] == [False, False, True]

    # A second replay that disagrees with the first is the strategy's own noise;
    # one that agrees leaves Paper's decision path to the Paper-mode run.
    rerun = compare(records, replay, vintages=VINTAGES, repeat=lambda day: [extra])
    assert (rerun["first_difference"]["cause"], rerun["probes"]["repeat"]) == ("nondeterministic", "differs")
    same = lambda day: []  # noqa: E731
    path = compare(records, replay, vintages=VINTAGES, repeat=same, paper_mode=lambda day: [extra])
    assert (path["first_difference"]["cause"], path["probes"]) == ("paper_decision_path", {"repeat": "equal", "paper_mode": "differs"})
    data = compare(records, replay, vintages=VINTAGES, repeat=same, paper_mode=same)
    assert (data["first_difference"]["cause"], data["probes"]["paper_mode"]) == ("data_revised", "equal")


def test_a_decision_on_an_ex_date_of_a_held_name_differs_by_the_accepted_timing():
    """A replay opens the day (and credits the dividend) before it decides;
    Paper decides from the previous close."""

    records, replay = _matching()
    action = {
        "trade_date": DAYS[2], "symbol": SYMBOL, "last_close": 10.55, "pre_close": 10.45, "cash_per_share": 0.1,
        "quantity_before": 100, "quantity_after": 100, "cash_credit": 10.0, "dividend_tax_withheld": 0.0,
    }
    records.corporate_actions[DAYS[2]] = [dict(action)]
    replay["corporate_actions"] = [dict(action)]
    replay["executions"][1]["quantity"] = 200
    first = compare(records, replay, vintages=VINTAGES, repeat=lambda day: [])["first_difference"]
    assert (first["trade_date"], first["level"], first["cause"]) == (DAYS[2], "orders", "ex_date_timing")


def test_settlement_differences_are_restated_data_when_their_inputs_moved_and_a_rule_otherwise():
    # A fill of the same order at another price met another bar.
    records, replay = _matching()
    replay["executions"][0]["price"] = 10.015
    first = compare(records, replay, vintages=VINTAGES)["first_difference"]
    assert (first["trade_date"], first["level"], first["cause"]) == (DAYS[0], "fills", "data_revised")
    # The same price with other fees is the rule.
    records, replay = _matching()
    replay["executions"][0]["commission"] = 5.1
    assert compare(records, replay, vintages=VINTAGES)["first_difference"]["cause"] == "rule"
    # Half a cent is the same figure.
    replay["executions"][0]["commission"] = 5.004
    assert compare(records, replay, vintages=VINTAGES)["status"] == "consistent"

    # An ex-date settled from another reference price is restated data.
    records, replay = _matching()
    action = {
        "trade_date": DAYS[1], "symbol": SYMBOL, "last_close": 10.55, "pre_close": 10.45, "cash_per_share": 0.1,
        "quantity_before": 100, "quantity_after": 100, "cash_credit": 10.0, "dividend_tax_withheld": 0.0,
    }
    records.corporate_actions[DAYS[1]] = [dict(action)]
    replay["corporate_actions"] = [{**action, "pre_close": 10.4, "cash_per_share": 0.15, "cash_credit": 15.0}]
    first = compare(records, replay, vintages=VINTAGES)["first_difference"]
    assert (first["trade_date"], first["level"], first["cause"]) == (DAYS[1], "opening", "data_revised")
    # The same inputs settled to another credit is the rule.
    replay["corporate_actions"] = [{**action, "cash_credit": 9.0}]
    assert compare(records, replay, vintages=VINTAGES)["first_difference"]["cause"] == "rule"

    # Equal fills and holdings with another equity met a restated close.
    records, replay = _matching()
    replay["equity_curve"][1]["equity"] += 3.0
    first = compare(records, replay, vintages=VINTAGES)["first_difference"]
    assert (first["trade_date"], first["level"], first["cause"]) == (DAYS[1], "close", "data_revised")
    assert first["evidence"]["equity"] == [100094.5, 100097.5]


def test_a_real_fill_book_is_compared_until_its_account_leaves_the_simulation():
    records, replay = _matching()
    simulated = {"quantity": 100, "price": 10.2, "notional": 1020.0, "commission": 5.0, "stamp_duty": 0.51,
                 "dividend_tax": 0.0, "realized_pnl": 9.5, "reason": None}
    state = dict(records.state)
    state["real_fills"] = {"after": DAYS[0], "sessions": {
        DAYS[1]: {},
        DAYS[2]: {"09:30 sell 000001.SZ": {"ordered": 100, "simulated": simulated, "real": {**simulated, "quantity": 0}}},
    }}
    result = compare(BookRecords(state, *list(records.__dict__.values())[1:]), replay, vintages=VINTAGES)
    assert (result["status"], result["sessions_compared"]) == ("partial", 2)
    assert DAYS[2] in result["reason"]


# ---------------------------------------------------------- the whole check


def _book(root: Path, lake: Path, source: str) -> Book:
    strategy = root / "strategy" / "output" / "main.py"
    strategy.parent.mkdir(parents=True, exist_ok=True)
    strategy.write_text(source, encoding="utf-8")
    return Book(
        root=root, experiment_id="exp", artifact_id="art", note="", strategy_path=strategy, models_dir=None,
        schedule=StrategySchedule("day", "08:30"), profile=BrokerProfile(initial_cash=100_000.0, permitted_boards=("main", "gem")),
        sandbox=SandboxConfig(), snapshot_config=LAKE_CONFIG, max_intraday_row_group_rows=2_000_000,
        raw_dir=lake / "data" / "raw", fundamental_events_root=lake / "data" / "pit" / "fundamental_events",
        fundamental_events_status=lake / "results" / "data_quality" / "fundamental_events_status.json",
        benchmark_index=DEFAULT_BENCHMARK_INDEX, verdict_rules={},
        artifact_fingerprint=artifact_fingerprint(strategy.parent, None),
    )


def _files(root: Path) -> dict[str, bytes]:
    return {str(path.relative_to(root)): path.read_bytes() for path in sorted(root.rglob("*")) if path.is_file()}


def test_a_check_that_cannot_read_the_book_fails_and_writes_nothing_else(tmp_path: Path):
    paper = _Book(tmp_path, COUNTER_STRATEGY)
    for day in ("20260106", "20260107"):
        paper.run(day)
    book = _book(paper.root, tmp_path / "lake", COUNTER_STRATEGY)
    before = _files(paper.root)
    with writer_lock(paper.root):
        busy = check_book(book, work_dir=tmp_path / "work", execution_mode="trusted")
    assert (busy["status"], "another Paper writer" in busy["reason"]) == ("failed", True)
    with (paper.root / "equity_daily.jsonl").open("a", encoding="utf-8") as stream:
        stream.write('{"kind": "equity", "trade\n')
    before["equity_daily.jsonl"] = (paper.root / "equity_daily.jsonl").read_bytes()
    damaged = check_book(book, work_dir=tmp_path / "work", execution_mode="trusted")
    assert damaged["status"] == "failed" and "equity_daily.jsonl has 1 unreadable line" in damaged["reason"]
    after = _files(paper.root)
    assert json.loads(after.pop(CONSISTENCY_NAME)) == damaged
    assert after == before


LAKE_CONFIG = SnapshotConfig(
    events_datasets=(), macro_datasets=(), text_datasets=(), fundamental_datasets=(),
    include_intraday=False, include_industry=False,
)
LAKE_SESSIONS = ("20211011", "20211012", "20211013", "20211014", "20211015")
UPDATER = load_config(REPO_ROOT / DEFAULT_CONFIG)


def _lake(root: Path) -> Path:
    """A committed lake of two names whose sessions run on from the snapshot
    fixture's: 20210930 and 20211008 before a book's first decision on
    20211011, then four more."""

    raw = root / "data" / "raw"
    build_raw(raw)
    write(raw / "trade_cal" / "exchange=SSE" / "year=2021.parquet", pd.DataFrame(
        {"cal_date": ["20210930", "20211008", *LAKE_SESSIONS, "20211018"], "is_open": "1"}
    ))
    symbols = ("000001.SZ", "000002.SZ")
    for index, day in enumerate(LAKE_SESSIONS):
        price = 10.0 + 0.1 * index
        write(raw / "daily" / f"trade_date={day}.parquet", pd.DataFrame([
            {"trade_date": day, "ts_code": symbol, "open": price, "high": price + 0.2, "low": price - 0.2,
             "close": price + 0.05, "pre_close": price - 0.05, "pct_chg": 1.0, "vol": 1000.0, "amount": 10000.0}
            for symbol in symbols
        ]))
        for dataset, fields in (
            ("daily_basic", {"turnover_rate": 2.0, "dv_ttm": 3.0, "pe": 10.0, "total_share": 100.0, "total_mv": 1000.0,
                             "close": 10.5, "circ_mv": 2_000_000.0}),
            ("stk_limit", {"up_limit": 30.0, "down_limit": 5.0}),
            ("adj_factor", {"adj_factor": 1.0}),
        ):
            write(raw / dataset / f"trade_date={day}.parquet", pd.DataFrame(
                [{"trade_date": day, "ts_code": symbol, **fields} for symbol in symbols]
            ))
        write(raw / "suspend_d" / f"trade_date={day}.parquet",
              pd.DataFrame(columns=["trade_date", "ts_code", "suspend_timing", "suspend_type"]))
    build_fundamental_events(root / "data" / "pit" / "fundamental_events")
    status = root / "results" / "data_quality" / "fundamental_events_status.json"
    status.parent.mkdir(parents=True)
    write_fundamental_status(status)
    # Committed as the updater commits: a sidecar per partition, its lock and
    # a generation that refreshed every mounted dataset after the last session.
    for part in raw.rglob("*.parquet"):
        part.with_suffix(".parquet.meta.json").write_text("{}\n", encoding="utf-8")
    lock = root / ".runtime" / "tushare" / "locks" / "tushare_update.lock"
    lock.parent.mkdir(parents=True)
    lock.touch()
    refreshed = {"refreshed_at": "2021-10-15T00:00:00+00:00", "vintage": "v1"}
    (raw / ".raw_generation.json").write_text(json.dumps({
        "schema_version": 3, "state": "committed", "generation_id": "g20211015", "issues": [],
        "datasets": {name: refreshed for name in ("daily", "daily_basic", "adj_factor", "stk_limit", "suspend_d")},
    }), encoding="utf-8")
    return root


def _trade(book: Book) -> None:
    """Paper decides every session, one morning each, as the cron does."""

    engine = DailyPaperEngine(
        strategy_path=book.strategy_path, strategy_revision=book.artifact_id, state_root=book.root,
        data_factory=book_data_factory(book, UPDATER), schedule=book.schedule, profile=book.profile,
        executor_factory=lambda path, _sandbox, _view, state_dir, models_dir: TrustedStrategyExecutor.from_path(
            path, state_dir=state_dir, models_dir=models_dir
        ),
    )
    for day in LAKE_SESSIONS:
        engine.run_day(day)


STATELESS = """def generate_orders(context):
    day = context.inference_at.strftime("%Y-%m-%d")
    if context.account.positions:
        return [{"symbol": "000001.SZ", "action": "sell", "quantity": 100, "execute_at": day + "T15:00:00+08:00"}]
    return [{"symbol": "000001.SZ", "action": "buy", "quantity": 100, "execute_at": day + "T09:30:00+08:00"}]
"""

# Counts its calls in its state directory and buys on every third. Paper
# re-issues a book's earlier calls each morning on the published state, so the
# count runs ahead of a replay's from the second decision on.
STATE_COUNTER = """import numpy as np


def generate_orders(context):
    path = context.state_dir + "/calls.npy"
    try:
        calls = int(np.load(path)[0]) + 1
    except FileNotFoundError:
        calls = 1
    np.save(path, np.array([calls]))
    if calls % 3:
        return []
    day = context.inference_at.strftime("%Y-%m-%d")
    return [{"symbol": "000001.SZ", "action": "buy", "quantity": 100, "execute_at": day + "T09:30:00+08:00"}]
"""


def test_a_stateless_book_is_consistent_with_its_replay_and_only_its_result_is_written(tmp_path: Path):
    book = _book(tmp_path / "paper" / "book", _lake(tmp_path), STATELESS)
    _trade(book)
    before = _files(book.root)
    result = check_book(book, work_dir=tmp_path / "work", execution_mode="trusted", updater_schedule=UPDATER)
    assert result["status"] == "consistent", result.get("reason")
    assert (result["window"], result["sessions_compared"]) == ({"start": "20211011", "end": "20211014"}, 4)
    assert result["release"]["generation_id"] == "g20211015"
    after = _files(book.root)
    assert json.loads(after.pop(CONSISTENCY_NAME)) == result
    assert after == before
    # The views and pin are gone; the replay's result stays until the next check.
    assert [entry.name for entry in (tmp_path / "work").iterdir()] == ["results"]
    assert Path(result["result"]).is_file()


def test_a_book_whose_state_runs_ahead_on_paper_is_caught_where_it_first_orders(tmp_path: Path):
    book = _book(tmp_path / "paper" / "book", _lake(tmp_path), STATE_COUNTER)
    _trade(book)
    plain = check_book(book, work_dir=tmp_path / "work", execution_mode="trusted", updater_schedule=UPDATER)
    first = plain["first_difference"]
    assert (plain["status"], first["trade_date"], first["level"], first["cause"]) == (
        "differs", "20211012", "orders", "unresolved"
    )
    assert (len(first["evidence"]["paper_only"]), first["evidence"]["replay_only"]) == (1, [])
    assert plain["probes"] == {"repeat": "equal", "paper_mode": "not_run"}
    probed = check_book(
        book, work_dir=tmp_path / "work", execution_mode="trusted", updater_schedule=UPDATER, paper_mode=True
    )
    assert probed["first_difference"]["cause"] == "paper_decision_path"
    assert probed["probes"] == {"repeat": "equal", "paper_mode": "differs"}
    assert sorted(entry.name for entry in (tmp_path / "work").iterdir()) == ["results"]
