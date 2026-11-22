"""The consistency check between a Paper book and a historical replay of it."""

from __future__ import annotations

import copy
import dataclasses
import json
import subprocess
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from autotrade.data_sources.tushare.cron_update import DEFAULT_CONFIG, load_config
from autotrade.environment.artifacts import artifact_fingerprint
from autotrade.environment.broker import BrokerProfile
from autotrade.environment.data.contracts import DEFAULT_BENCHMARK_INDEX
from autotrade.environment.data.snapshot import SnapshotConfig, load_snapshot_manifest
from autotrade.environment.executor import TrustedStrategyExecutor
from autotrade.environment.sandbox import SandboxConfig
from autotrade.environment.strategy import CN_TZ, StrategySchedule
from autotrade.paper.book import Book
from autotrade.paper.consistency import (
    CONSISTENCY_NAME,
    BookRecords,
    CheckFailed,
    check_book,
    compare,
    read_book_records,
)
from autotrade.paper.engine import DailyPaperEngine, code_revision, writer_lock
from autotrade.paper.pit import BookPITData, book_data_factory, newest_replay_slot
from autotrade.pipelines.calendar import anchor_before
from autotrade.pipelines.config import FORWARD_PHASE
from autotrade.pipelines.pit_backend import ResearchPITSnapshotProvider
from tests.unit.support.paper_book_fixture import (
    COUNTER_STRATEGY,
    _Book,
    _registered_book,
    _repo,
    _run_paper,
    paper_root,
    write_book_record,
)
from tests.unit.support.snapshot_fixture import (
    build_fundamental_events,
    build_raw,
    write,
    write_fundamental_status,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DAYS = ("20260105", "20260106", "20260107")
SYMBOL = "000001.SZ"
# The one clean commit a book and its check ran on, unless a test says otherwise.
CODE = "c" * 40


def _execution(day: str, action: str = "buy", price: float = 10.005, **fields: object) -> dict[str, object]:
    stamp = f"{day[:4]}-{day[4:6]}-{day[6:]}T09:30:00+08:00"
    return {
        "symbol": SYMBOL, "action": action, "quantity": 100, "execute_at": stamp, "matched_at": stamp,
        "status": "filled", "price": price, "commission": 5.0, "stamp_duty": 0.0, "dividend_tax": 0.0,
        "realized_pnl": None, "reason": None, "metadata": {}, **fields,
    }


def _matching() -> tuple[BookRecords, dict[str, object]]:
    """A book that bought on the first session and sold on the third, all on
    ``CODE``, and the replay that did exactly the same."""

    buy, sell = _execution(DAYS[0]), _execution(DAYS[2], "sell", 10.2, realized_pnl=9.5)
    closes = [(98994.5, 100044.5, {SYMBOL: 100}), (98994.5, 100094.5, {SYMBOL: 100}), (100009.5, 100009.5, {})]
    decisions = [
        {"trade_date": day, "decided_at": f"{day[:4]}-{day[4:6]}-{day[6:]}T05:45:00+08:00", "generation_id": "g_paper",
         "vintages": {"daily": "v1", "anns_d": "v1"}, "code_revision": CODE}
        for day in (*DAYS, "20260108")
    ]
    order = {key: buy[key] for key in ("symbol", "action", "quantity", "execute_at")}
    records = BookRecords(
        state={"start_date": DAYS[0], "settled_through": DAYS[-1], "decisions": decisions},
        orders={DAYS[0]: [order], DAYS[2]: [{**order, "action": "sell", "execute_at": sell["execute_at"]}]},
        executions={DAYS[0]: [buy], DAYS[2]: [sell]},
        corporate_actions={},
        equity=[
            {"trade_date": day, "cash": cash, "equity": equity, "position_count": len(held), "code_revision": CODE}
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
    result = compare(records, replay, vintages=VINTAGES, code=CODE)
    assert (result["status"], result["sessions_compared"], result["first_difference"]) == ("consistent", 3, None)
    assert all(row["opening"] and row["orders"] and row["fills"] and row["close"] for row in result["sessions"])

    empty = BookRecords({"decisions": []}, {}, {}, {}, [], [])
    with pytest.raises(CheckFailed, match="no settled session"):
        compare(empty, {"equity_curve": [], "executions": [], "corporate_actions": []}, vintages=VINTAGES, code=CODE)
    # A replay that does not reach every settled session is no comparison.
    replay["equity_curve"] = replay["equity_curve"][:2]
    with pytest.raises(CheckFailed, match=r"the replay covers 20260105\.\.20260106"):
        compare(records, replay, vintages=VINTAGES, code=CODE)


def test_an_order_paper_alone_holds_is_the_first_difference_and_what_follows_is_downstream():
    records, replay = _matching()
    extra = _execution(DAYS[1], "sell")
    records.executions[DAYS[1]] = [{**extra, "status": "rejected", "price": None, "commission": 0.0, "reason": "t1"}]
    records.orders[DAYS[1]] = [{key: extra[key] for key in ("symbol", "action", "quantity", "execute_at")}]
    result = compare(records, replay, vintages=VINTAGES, code=CODE)
    first = result["first_difference"]
    assert (result["status"], first["trade_date"], first["level"], first["cause"]) == ("differs", DAYS[1], "orders", "unresolved")
    evidence = first["evidence"]
    assert (evidence["decision"], len(evidence["paper_only"]), evidence["replay_only"]) == (DAYS[1], 1, [])
    # The mounted datasets restated since Paper's decision, the only lead left.
    assert evidence["restated"] == ["anns_d"]
    assert [row["downstream"] for row in result["sessions"]] == [False, False, True]

    # A second replay that disagrees with the first is the strategy's own noise;
    # one that agrees leaves Paper's decision path to the Paper-mode run.
    rerun = compare(records, replay, vintages=VINTAGES, code=CODE, repeat=lambda day: [extra])
    assert (rerun["first_difference"]["cause"], rerun["probes"]["repeat"]) == ("nondeterministic", "differs")
    same = lambda day: []  # noqa: E731
    path = compare(records, replay, vintages=VINTAGES, code=CODE, repeat=same, paper_mode=lambda day: [extra])
    assert (path["first_difference"]["cause"], path["probes"]) == ("paper_decision_path", {"repeat": "equal", "paper_mode": "differs"})
    data = compare(records, replay, vintages=VINTAGES, code=CODE, repeat=same, paper_mode=same)
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
    first = compare(records, replay, vintages=VINTAGES, code=CODE, repeat=lambda day: [])["first_difference"]
    assert (first["trade_date"], first["level"], first["cause"]) == (DAYS[2], "orders", "ex_date_timing")


def test_settlement_differences_are_restated_data_when_their_inputs_moved_and_a_rule_otherwise():
    # A fill of the same order at another price met another bar.
    records, replay = _matching()
    replay["executions"][0]["price"] = 10.015
    first = compare(records, replay, vintages=VINTAGES, code=CODE)["first_difference"]
    assert (first["trade_date"], first["level"], first["cause"]) == (DAYS[0], "fills", "data_revised")
    # The same price with other fees is the rule.
    records, replay = _matching()
    replay["executions"][0]["commission"] = 5.1
    assert compare(records, replay, vintages=VINTAGES, code=CODE)["first_difference"]["cause"] == "rule"
    # Half a cent is the same figure.
    replay["executions"][0]["commission"] = 5.004
    assert compare(records, replay, vintages=VINTAGES, code=CODE)["status"] == "consistent"

    # An ex-date settled from another reference price is restated data.
    records, replay = _matching()
    action = {
        "trade_date": DAYS[1], "symbol": SYMBOL, "last_close": 10.55, "pre_close": 10.45, "cash_per_share": 0.1,
        "quantity_before": 100, "quantity_after": 100, "cash_credit": 10.0, "dividend_tax_withheld": 0.0,
    }
    records.corporate_actions[DAYS[1]] = [dict(action)]
    replay["corporate_actions"] = [{**action, "pre_close": 10.4, "cash_per_share": 0.15, "cash_credit": 15.0}]
    first = compare(records, replay, vintages=VINTAGES, code=CODE)["first_difference"]
    assert (first["trade_date"], first["level"], first["cause"]) == (DAYS[1], "opening", "data_revised")
    # The same inputs settled to another credit is the rule.
    replay["corporate_actions"] = [{**action, "cash_credit": 9.0}]
    assert compare(records, replay, vintages=VINTAGES, code=CODE)["first_difference"]["cause"] == "rule"

    # Equal fills and holdings with another equity met a restated close.
    records, replay = _matching()
    replay["equity_curve"][1]["equity"] += 3.0
    first = compare(records, replay, vintages=VINTAGES, code=CODE)["first_difference"]
    assert (first["trade_date"], first["level"], first["cause"]) == (DAYS[1], "close", "data_revised")
    assert first["evidence"]["equity"] == [100094.5, 100097.5]


def _ran_on(records: BookRecords, revision: str | None, through: str = DAYS[-1]) -> BookRecords:
    """``records`` with every decision and settled session through ``through``
    run on ``revision`` (None: before Paper recorded its code, so naming none)."""

    def stamp(row: dict[str, object]) -> dict[str, object]:
        if str(row["trade_date"]) > through:
            return dict(row)
        if revision is None:
            return {key: value for key, value in row.items() if key != "code_revision"}
        return {**row, "code_revision": revision}

    state = {**records.state, "decisions": [stamp(row) for row in records.state["decisions"]]}
    return dataclasses.replace(records, state=state, equity=[stamp(row) for row in records.equity])


def test_a_cause_reached_by_elimination_names_a_code_change_when_the_book_ran_other_code():
    old, new = "a" * 40, "b" * 40
    records, replay = _matching()
    extra = _execution(DAYS[1], "sell")
    records.executions[DAYS[1]] = [{**extra, "status": "rejected", "price": None, "commission": 0.0, "reason": "t1"}]
    records.orders[DAYS[1]] = [{key: extra[key] for key in ("symbol", "action", "quantity", "execute_at")}]

    def cause(book: BookRecords, code: str | None, **probes: object) -> tuple[object, ...]:
        first = compare(book, replay, vintages=VINTAGES, code=code, **probes)["first_difference"]
        return first["cause"], first["evidence"].get("otherwise"), first["evidence"].get("paper_code_revisions")

    assert cause(_ran_on(records, old), new) == ("code_changed", "unresolved", [old])
    # The Paper-mode run agrees on today's code: data or the code change.
    same = lambda day: []  # noqa: E731
    assert cause(_ran_on(records, old), new, repeat=same, paper_mode=same) == ("code_changed", "data_revised", [old])
    # It disagrees: today's decision path differs, whatever the old code did.
    assert cause(_ran_on(records, old), new, repeat=same, paper_mode=lambda day: [extra])[0] == "paper_decision_path"
    # On the check's own clean commit the cause stands.
    assert cause(_ran_on(records, new), new) == ("unresolved", None, None)
    # Sessions Paper ran before it recorded its code ran other code; so does
    # whatever ran when the check cannot read its own.
    assert cause(_ran_on(records, None), new) == ("code_changed", "unresolved", [None])
    assert cause(_ran_on(records, new), None) == ("code_changed", "unresolved", [new])
    # Only what ran through the session counts.
    assert cause(_ran_on(_ran_on(records, new), old, through=DAYS[1]), new)[0] == "code_changed"
    assert cause(_ran_on(_ran_on(records, old), new, through=DAYS[1]), new) == ("unresolved", None, None)
    # A dirty tree is no commit's code, even the check's own.
    assert cause(_ran_on(records, new + "-dirty"), new)[0] == "code_changed"
    assert cause(_ran_on(records, new + "-dirty"), new + "-dirty")[0] == "code_changed"

    # A settlement difference too; an accepted timing is not one.
    records, replay = _matching()
    replay["executions"][0]["price"] = 10.015
    first = compare(_ran_on(records, old), replay, vintages=VINTAGES, code=new)["first_difference"]
    assert (first["level"], first["cause"], first["evidence"]["otherwise"]) == ("fills", "code_changed", "data_revised")
    records, replay = _matching()
    action = {
        "trade_date": DAYS[2], "symbol": SYMBOL, "last_close": 10.55, "pre_close": 10.45, "cash_per_share": 0.1,
        "quantity_before": 100, "quantity_after": 100, "cash_credit": 10.0, "dividend_tax_withheld": 0.0,
    }
    records.corporate_actions[DAYS[2]] = [dict(action)]
    replay["corporate_actions"] = [dict(action)]
    replay["executions"][1]["quantity"] = 200
    first = compare(_ran_on(records, old), replay, vintages=VINTAGES, code=new)["first_difference"]
    assert first["cause"] == "ex_date_timing"


def test_the_code_revision_is_the_checkouts_commit_marked_dirty_when_a_tracked_file_differs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from autotrade.paper import engine

    def git(*args: str) -> str:
        done = subprocess.run(["git", "-C", str(tmp_path), *args], capture_output=True, text=True, check=True)
        return done.stdout.strip()

    monkeypatch.setattr(engine, "REPO_ROOT", tmp_path)
    assert engine.code_revision() is None  # no checkout
    git("init", "-q")
    (tmp_path / "tracked.py").write_text("x = 1\n", encoding="utf-8")
    git("add", "tracked.py")
    git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "init")
    commit = git("rev-parse", "HEAD")
    (tmp_path / "untracked.txt").write_text("log\n", encoding="utf-8")
    assert engine.code_revision() == commit
    (tmp_path / "tracked.py").write_text("x = 2\n", encoding="utf-8")
    assert engine.code_revision() == commit + "-dirty"


def test_a_real_fill_book_is_compared_until_its_account_leaves_the_simulation():
    records, replay = _matching()
    simulated = {"quantity": 100, "price": 10.2, "notional": 1020.0, "commission": 5.0, "stamp_duty": 0.51,
                 "dividend_tax": 0.0, "realized_pnl": 9.5, "reason": None}
    state = dict(records.state)
    state["real_fills"] = {"after": DAYS[0], "sessions": {
        DAYS[1]: {},
        DAYS[2]: {"09:30 sell 000001.SZ": {"ordered": 100, "simulated": simulated, "real": {**simulated, "quantity": 0}}},
    }}
    result = compare(dataclasses.replace(records, state=state), replay, vintages=VINTAGES, code=CODE)
    assert (result["status"], result["sessions_compared"]) == ("partial", 2)
    assert DAYS[2] in result["reason"]


# ---------------------------------------------------------- the whole check


def _book(root: Path, lake: Path, source: str, config: SnapshotConfig | None = None) -> Book:
    strategy = root / "strategy" / "output" / "main.py"
    strategy.parent.mkdir(parents=True, exist_ok=True)
    strategy.write_text(source, encoding="utf-8")
    write_book_record(root)
    return Book(
        root=root, experiment_id="exp", artifact_id="art", note="", strategy_path=strategy, models_dir=None,
        schedule=StrategySchedule("day", "08:30"), profile=BrokerProfile(initial_cash=100_000.0, permitted_boards=("main", "gem")),
        sandbox=SandboxConfig(), snapshot_config=config or LAKE_CONFIG, max_intraday_row_group_rows=2_000_000,
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


def test_the_check_command_keeps_out_of_the_morning_and_exits_2_for_a_book_it_cannot_compare(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    cli = _run_paper()
    # Monday 2026-11-23: the morning run's window is closed to the check, a Saturday is not.
    assert cli.in_morning_window(datetime(2026, 11, 23, 5, 40, tzinfo=CN_TZ))
    assert not cli.in_morning_window(datetime(2026, 11, 23, 9, 10, tzinfo=CN_TZ))
    assert not cli.in_morning_window(datetime(2026, 11, 21, 6, 0, tzinfo=CN_TZ))
    monkeypatch.setattr(cli, "in_morning_window", lambda _now: False)
    monkeypatch.setattr(cli, "REPO_ROOT", _repo(tmp_path))
    book = _registered_book(tmp_path)  # never run: nothing settled yet
    assert cli.main(["check", "--state-root", str(paper_root(tmp_path))]) == 2
    result = json.loads((book.root / CONSISTENCY_NAME).read_text(encoding="utf-8"))
    assert (result["status"], result["reason"]) == ("failed", "CheckFailed: the book has no settled session to compare")
    assert "paper_consistency " in capsys.readouterr().out


def test_a_book_deleted_during_its_check_gets_no_result_and_no_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    """Writing the result would re-create the deleted book's directory, and a
    new book could not take its name; the check writes nothing and exits 2."""

    from autotrade.paper import consistency
    from autotrade.paper.books import delete_book

    cli = _run_paper()
    monkeypatch.setattr(cli, "in_morning_window", lambda _now: False)
    monkeypatch.setattr(cli, "REPO_ROOT", _repo(tmp_path))
    book = _registered_book(tmp_path)
    read_book_records = consistency.read_book_records

    def deleted_first(root: Path) -> BookRecords:
        delete_book(paper_root(tmp_path), "exp")
        return read_book_records(root)

    monkeypatch.setattr(consistency, "read_book_records", deleted_first)
    assert cli.main(["check", "--state-root", str(paper_root(tmp_path))]) == 2
    assert not book.root.exists()
    assert "[exp] the book exp was deleted during its check; no result is written\n" in capsys.readouterr().err


LAKE_CONFIG = SnapshotConfig(
    events_datasets=(), macro_datasets=(), text_datasets=(), fundamental_datasets=(),
    include_intraday=False, include_industry=False,
)
# A month of history, fundamentals and two sessions of minute bars. A book
# whose first decision is 20211011 used to be anchored at 20210930 23:59:59,
# a replay of it at 20211010 23:59:59, so 20210909's bars and the income event
# of 2021-09-10 18:00 lie between where a month back from each would start.
EDGE_CONFIG = dataclasses.replace(
    LAKE_CONFIG, window_months=1, fundamental_datasets=("income_vip",), include_intraday=True,
    replay_include_minutes=True, intraday_trade_days=2,
)
LAKE_SESSIONS = ("20211011", "20211012", "20211013", "20211014", "20211015")
UPDATER = load_config(REPO_ROOT / DEFAULT_CONFIG)


def _lake(root: Path) -> Path:
    """A committed lake of two names whose sessions run on from the snapshot
    fixture's: 20210909, 20210929, 20210930 and 20211008 before a book's
    first decision on 20211011, then four more, with minute bars at 09:30."""

    raw = root / "data" / "raw"
    build_raw(raw)
    write(raw / "trade_cal" / "exchange=SSE" / "year=2021.parquet", pd.DataFrame(
        {"cal_date": ["20210909", "20210929", "20210930", "20211008", *LAKE_SESSIONS, "20211018"], "is_open": "1"}
    ))
    symbols = ("000001.SZ", "000002.SZ")
    for day, price in (
        ("20210909", 9.0), ("20210929", 9.5), *((day, 10.0 + 0.1 * index) for index, day in enumerate(LAKE_SESSIONS))
    ):
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
    for day in ("20210909", "20210929", "20211008", *LAKE_SESSIONS):
        stamp = f"{day[:4]}-{day[4:6]}-{day[6:]}"
        write(raw / "stk_mins_1min_by_date" / f"trade_date={day}.parquet", pd.DataFrame([
            {"ts_code": symbol, "trade_time": f"{stamp} 09:30:00", "open": 10.0, "high": 10.1, "low": 9.9,
             "close": 10.0, "vol": 20000.0, "amount": 200000.0, "trade_date": day,
             "available_at": f"{stamp}T09:30:00+08:00", "available_at_rule": "bar_close"}
            for symbol in symbols
        ]))
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
        "datasets": {
            name: refreshed
            for name in ("daily", "daily_basic", "adj_factor", "stk_limit", "suspend_d", "income_vip", "stk_mins_1min_by_date")
        },
    }), encoding="utf-8")
    return root


def _trade(book: Book, *, pre_open: bool = False) -> None:
    """Paper decides every session, one morning each, as the cron does.

    ``pre_open`` runs the first morning on an earlier generation of the lake,
    one without the bars of that session or any later, as it runs live."""

    engine = DailyPaperEngine(
        strategy_path=book.strategy_path, strategy_revision=book.artifact_id, state_root=book.root,
        data_factory=book_data_factory(book, UPDATER), schedule=book.schedule, profile=book.profile,
        executor_factory=lambda path, _sandbox, _view, state_dir, models_dir: TrustedStrategyExecutor.from_path(
            path, state_dir=state_dir, models_dir=models_dir
        ),
    )
    marker = book.raw_dir / ".raw_generation.json"
    committed = marker.read_text(encoding="utf-8")
    held = book.raw_dir.parent / "held_out"
    landed = [
        path for path in book.raw_dir.rglob("trade_date=*.parquet*")
        if pre_open and path.name.split("=")[1][:8] >= LAKE_SESSIONS[0]
    ]
    for path in landed:
        (held / path.relative_to(book.raw_dir)).parent.mkdir(parents=True, exist_ok=True)
        path.rename(held / path.relative_to(book.raw_dir))
    if pre_open:
        marker.write_text(json.dumps({**json.loads(committed), "generation_id": "g20211008"}), encoding="utf-8")
    engine.run_day(LAKE_SESSIONS[0])
    for path in landed:
        (held / path.relative_to(book.raw_dir)).rename(path)
    marker.write_text(committed, encoding="utf-8")
    for day in LAKE_SESSIONS[1:]:
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


# Records every table of the view it decides on in its order, so the check
# compares what Paper and the replay showed it, decision by decision.
VIEW_RECORDER = """import pandas as pd

TABLES = ("auction", "daily", "events", "fundamentals", "intraday_1min", "macro", "text_index", "universe")


def generate_orders(context):
    view = {}
    for table in TABLES:
        try:
            frame = pd.read_parquet(context.asof_dir + "/" + table)
        except FileNotFoundError:
            continue
        view[table] = sorted("|".join(row) for row in frame[sorted(frame.columns)].astype(str).itertuples(index=False))
    day = context.inference_at.strftime("%Y-%m-%d")
    return [{"symbol": "000001.SZ", "action": "buy", "quantity": 100, "execute_at": day + "T09:30:00+08:00", "view": view}]
"""


def test_a_book_decides_on_the_view_its_replay_shows_down_to_the_oldest_rows(tmp_path: Path):
    book = _book(tmp_path / "paper" / "book", _lake(tmp_path), VIEW_RECORDER, EDGE_CONFIG)
    _trade(book)
    result = check_book(book, work_dir=tmp_path / "work", execution_mode="trusted", updater_schedule=UPDATER)
    assert result["status"] == "consistent", result.get("first_difference") or result.get("reason")
    assert result["sessions_compared"] == 4
    # Neither side shows what lies between the two windows' starts: the bars
    # from 20210929 on, no income event, the replay's last two minute days.
    first = json.loads((book.root / "orders_20211011.jsonl").read_text(encoding="utf-8").splitlines()[0])["view"]
    assert {table: len(first[table]) for table in ("daily", "fundamentals", "intraday_1min")} == {
        "daily": 4, "fundamentals": 0, "intraday_1min": 3
    }


# A book that reads industries and screens ST names out of its universe.
ANCHOR_CONFIG = dataclasses.replace(LAKE_CONFIG, include_industry=True, screen_exclude_st=True)


def _anchor_lake(root: Path) -> Path:
    """``_lake`` whose universe changes after 20210930 23:59:59, where a book
    first deciding on 20211011 used to take it, and before 20211010 23:59:59,
    where a replay of the book takes it: 000002.SZ turns ST (and so leaves the
    screened universe), 000001.SZ moves industry and 000004.SZ lists. The
    rename of 000001.SZ announced on 20211012 comes after both."""

    lake = _lake(root)
    raw = lake / "data" / "raw"
    tables = {
        "stock_basic/list_status=L.parquet": [
            {"ts_code": "000001.SZ", "name": "平安银行", "exchange": "SZSE", "list_date": "19910403", "delist_date": None},
            {"ts_code": "000002.SZ", "name": "万科A", "exchange": "SZSE", "list_date": "19910129", "delist_date": None},
            {"ts_code": "000004.SZ", "name": "新股", "exchange": "SZSE", "list_date": "20211009", "delist_date": None},
        ],
        "namechange/namechange.parquet": [
            {"ts_code": "000001.SZ", "name": "平安银行", "start_date": "20120801", "ann_date": "20120730"},
            {"ts_code": "000001.SZ", "name": "平安新名", "start_date": "20211013", "ann_date": "20211012"},
            {"ts_code": "000002.SZ", "name": "万科A", "start_date": "19910129", "ann_date": "19910129"},
            {"ts_code": "000002.SZ", "name": "ST万科", "start_date": "20211009", "ann_date": "20211008"},
            {"ts_code": "000004.SZ", "name": "新股", "start_date": "20211009", "ann_date": "20211008"},
            {"ts_code": "000005.SZ", "name": "世纪星源", "start_date": "19901210", "ann_date": "19901210"},
        ],
        "index_classify/src=SW2014.parquet": [
            {"index_code": "801780.SI", "industry_name": "银行", "level": "L1"},
            {"index_code": "801790.SI", "industry_name": "非银金融", "level": "L1"},
            {"index_code": "801180.SI", "industry_name": "房地产", "level": "L1"},
        ],
        "index_member/l1_code=801780.SI.parquet": [
            {"index_code": "801780.SI", "con_code": "000001.SZ", "in_date": "19910403", "out_date": "20211009"},
        ],
        "index_member/l1_code=801790.SI.parquet": [
            {"index_code": "801790.SI", "con_code": "000001.SZ", "in_date": "20211009", "out_date": None},
        ],
        "index_member/l1_code=801180.SI.parquet": [
            {"index_code": "801180.SI", "con_code": "000002.SZ", "in_date": "19910129", "out_date": None},
        ],
    }
    for name, rows in tables.items():
        write(raw / name, pd.DataFrame(rows))
        (raw / name).with_suffix(".parquet.meta.json").write_text("{}\n", encoding="utf-8")
    return lake


def _view(book: Book, day: str) -> dict[str, list[str]]:
    """The view ``VIEW_RECORDER`` recorded in the book's order of ``day``."""

    return json.loads((book.root / f"orders_{day}.jsonl").read_text(encoding="utf-8").splitlines()[0])["view"]


def test_a_book_sees_the_universe_and_screen_its_replay_sees_at_its_anchor(tmp_path: Path):
    """The negative control of the anchor rule (``_anchor_lake``); the book's
    first morning runs before its session's bars land, as it does live."""

    book = _book(tmp_path / "paper" / "book", _anchor_lake(tmp_path), VIEW_RECORDER, ANCHOR_CONFIG)
    _trade(book, pre_open=True)
    decisions = json.loads((book.root / ".paper_state.json").read_text(encoding="utf-8"))["decisions"]
    assert [row["generation_id"] for row in decisions[:2]] == ["g20211008", "g20211015"]
    result = check_book(book, work_dir=tmp_path / "work", execution_mode="trusted", updater_schedule=UPDATER)
    assert result["status"] == "consistent", result.get("first_difference") or result.get("reason")
    assert result["sessions_compared"] == 4
    for day in LAKE_SESSIONS[:4]:
        view = _view(book, day)
        # ST at the anchor, so screened out of every per-stock table.
        assert not any("000002.SZ" in row for table in view.values() for row in table)
        [listed] = [row for row in view["universe"] if "000001.SZ" in row]
        # Its industry at the anchor, and its name before the later rename.
        assert ("非银金融" in listed, "平安银行" in listed) == (True, True)
        assert any("000004.SZ" in row for row in view["universe"])


def test_a_book_prepares_the_view_its_replay_prepares_and_research_keys_stay_put(tmp_path: Path):
    lake = _lake(tmp_path)
    provider = ResearchPITSnapshotProvider(
        experiment_dir=tmp_path / "pit", raw_dir=lake / "data" / "raw",
        fundamental_events_root=lake / "data" / "pit" / "fundamental_events",
        fundamental_events_status=lake / "results" / "data_quality" / "fundamental_events_status.json",
        config=EDGE_CONFIG,
    )
    replay = provider.prepare(phase=FORWARD_PHASE, start="20211011", end="20211012", decision_time=anchor_before("20211011"))
    research = load_snapshot_manifest(replay.decision_ref)
    # The names and fields every research view and seed was built under, as
    # 57040db5 writes them.
    assert (Path(replay.decision_ref).name, Path(replay.replay_ref).name) == (
        "20211010T235959+0800", "20211011_20211012_20211010T235959+0800"
    )
    assert sorted(research) == [
        "build_profile", "created_at", "data_profile", "data_quality_warnings", "decision_time", "domain_windows",
        "domains", "kind", "raw_generation", "snapshot_id", "window_config", "window_months", "window_start",
    ]
    book = _book(tmp_path / "paper" / "book", lake, STATELESS, EDGE_CONFIG)
    paper = BookPITData(
        state_root=book.root, raw_dir=book.raw_dir, fundamental_events_root=book.fundamental_events_root,
        fundamental_events_status=book.fundamental_events_status, snapshot_config=EDGE_CONFIG,
        start="20211011", trade_date="20211012", due_at=book.schedule.at("20211012"), schedule=UPDATER,
    )
    try:
        assert (paper.snapshot_dir.name, newest_replay_slot(book.root).name) == (
            "20211010T235959+0800", "20211011_20211012_20211010T235959+0800"
        )
        # The book's decision view is the replay's, file for file.
        assert {
            name: data for name, data in _files(paper.snapshot_dir).items() if name != "manifest.json"
        } == {name: data for name, data in _files(Path(replay.decision_ref)).items() if name != "manifest.json"}
    finally:
        paper.close()


def test_a_stateless_book_is_consistent_with_its_replay_and_only_its_result_is_written(tmp_path: Path):
    book = _book(tmp_path / "paper" / "book", _lake(tmp_path), STATELESS)
    _trade(book)
    before = _files(book.root)
    result = check_book(book, work_dir=tmp_path / "work", execution_mode="trusted", updater_schedule=UPDATER)
    assert result["status"] == "consistent", result.get("reason")
    assert (result["window"], result["sessions_compared"]) == ({"start": "20211011", "end": "20211014"}, 4)
    # Each decision and each settled session names the code it ran on.
    records = read_book_records(book.root)
    assert {row["code_revision"] for row in (*records.state["decisions"], *records.equity)} == {
        result["code_revision"]
    } == {code_revision()}
    assert result["release"]["generation_id"] == "g20211015"
    after = _files(book.root)
    assert json.loads(after.pop(CONSISTENCY_NAME)) == result
    assert after == before
    # The views and pin are gone; the replay's result stays until the next check.
    assert [entry.name for entry in (tmp_path / "work").iterdir()] == ["results"]
    assert Path(result["result"]).is_file()


def test_a_book_whose_state_runs_ahead_on_paper_is_caught_where_it_first_orders(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from autotrade.paper import consistency, engine

    # One clean commit for Paper and the check, whatever state this tree is in.
    for module in (engine, consistency):
        monkeypatch.setattr(module, "code_revision", lambda: CODE)
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
