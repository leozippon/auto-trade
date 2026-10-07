"""Paper books under one state root: creation, listing and the book-by-book run."""

from __future__ import annotations

import fcntl
import json
import os
import stat
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from autotrade.environment.artifacts import artifact_fingerprint
from autotrade.environment.broker import DailyBroker
from autotrade.environment.broker_core import board_of
from autotrade.environment.strategy import CN_TZ
from autotrade.paper import fills
from autotrade.paper.book import (
    BOOK_NAME,
    INCUBATING_BOOK_CAP,
    STRATEGY_COPY_NAME,
    VERDICT_LOG_NAME,
    Book,
    book_status,
    create_book,
    load_book,
    require_incubating_place,
)
from autotrade.paper.books import (
    delete_book,
    follows_real_fills,
    list_books,
    open_graduated_book,
    run_books,
    run_order,
    validate_book_id,
)
from autotrade.paper.engine import PaperWriterBusy
from autotrade.paper.storage import append_jsonl_once, read_json, write_json_atomic
from autotrade.pipelines.ledger import ExperimentLedger, forward_record
from tests.unit.paper_book_fixture import (
    FAILING_STRATEGY,
    engine_book,
    paper_root,
    run_days,
    write_book_record,
)
from tests.unit.test_broker_engine import MATCHED_AT, _bar, _order
from tests.unit.test_null_control import _board_draws
from tests.unit.test_snapshot_builder import write_fundamental_status
from tests.unit.webui_research_arm import build_arm


def _decided(root: Path) -> list[str]:
    state = json.loads((root / ".paper_state.json").read_text(encoding="utf-8"))
    return [row["trade_date"] for row in state["decisions"]]


def test_one_failing_or_busy_book_does_not_stop_the_others(tmp_path: Path):
    for book in ("alpha", "busy", "gamma"):
        engine_book(tmp_path, book=book)
    engine_book(tmp_path, book="broken", strategy=FAILING_STRATEGY)
    root = paper_root(tmp_path)
    (root / "not-a-book").mkdir()  # a directory without book.json is not a book
    assert list_books(root) == ["alpha", "broken", "busy", "gamma"]

    attempted = []

    def run_book(book_id: str, book_root: Path) -> None:
        attempted.append(book_id)
        run_days(book_root, "20260105")

    with (root / "busy" / ".paper_engine.lock").open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)  # another writer owns this book
        failures = run_books(root, list_books(root), run_book)

    assert attempted == ["alpha", "broken", "busy", "gamma"]
    assert sorted(failures) == ["broken", "busy"]
    assert "strategy broke" in str(failures["broken"])
    assert isinstance(failures["busy"], PaperWriterBusy)
    assert _decided(root / "alpha") == _decided(root / "gamma") == ["20260105"]
    assert not (root / "busy" / ".paper_state.json").exists()


def test_a_run_takes_the_books_the_owner_trades_first(tmp_path: Path):
    """The books switched to real fills run before the others, so the sheets
    the owner trades from are ready first; the order is the books' own state,
    so a deleted book leaves nothing behind to name."""

    root = paper_root(tmp_path)
    for book in ("alpha", "beta", "gamma", "traded"):
        write_book_record(root / book, experiment_id=book)
    assert run_order(root) == ["alpha", "beta", "gamma", "traded"]
    fills.enable(root / "traded")
    fills.enable(root / "beta")
    assert [follows_real_fills(root / book) for book in list_books(root)] == [False, True, False, True]
    assert run_order(root) == ["beta", "traded", "alpha", "gamma"]
    delete_book(root, "traded")
    assert run_order(root) == ["beta", "alpha", "gamma"]


def test_a_new_book_records_the_content_address_of_the_artifact_it_trades(tmp_path: Path):
    """Creation copies the graduated artifact into the book and records what it
    copied. The recorded fingerprint is the one content address the pipeline
    uses for a strategy artifact, over the book's own copy, so the bytes the
    book trades are named exactly as research named the artifact it approved."""

    arm = build_arm(tmp_path / "experiments", "exp", "graduated")
    book = create_book(
        paper_root(tmp_path) / "exp",
        experiment_dir=arm,
        artifact_id="strategy_research_abc",
        repo_root=tmp_path,
        track="graduated",
        note="参考簿（观察中）",
    )
    record = json.loads((book.root / "book.json").read_text(encoding="utf-8"))
    copy = book.root / STRATEGY_COPY_NAME
    source = arm / "artifacts/strategy/frozen/strategy_research_abc"
    assert book.strategy_path == copy / "output" / "main.py"
    assert book.models_dir is None
    assert record["artifact_fingerprint"] == artifact_fingerprint(copy / "output")
    assert record["artifact_fingerprint"] == artifact_fingerprint(source / "output")


def test_book_ids_are_single_path_segments():
    assert validate_book_id("arm-graduated_2026.09") == "arm-graduated_2026.09"
    for bad in ("", ".hidden", "a/b", "..", "a..b", "x" * 97):
        with pytest.raises(ValueError):
            validate_book_id(bad)


def test_a_single_book_root_is_refused(tmp_path: Path):
    root = paper_root(tmp_path)
    write_book_record(root, experiment_id="exp")
    with pytest.raises(RuntimeError, match="single-book layout"):
        list_books(root)


def _book_files(root: Path) -> dict[str, bytes]:
    return {str(path.relative_to(root)): path.read_bytes() for path in sorted(root.rglob("*")) if path.is_file()}


def test_a_graduate_opens_its_book_once_and_a_non_graduate_opens_none(tmp_path: Path):
    experiments = tmp_path / "elsewhere"  # any experiments root, not only <repo>/experiments
    arm = build_arm(experiments, "exp", "graduated")
    root = paper_root(tmp_path)
    assert open_graduated_book(tmp_path, arm) == "exp"
    record = json.loads((root / "exp" / "book.json").read_text(encoding="utf-8"))
    assert (record["experiment_id"], record["artifact_id"], record["note"]) == (
        "exp", "strategy_research_abc", "graduated"
    )
    opened = _book_files(root / "exp")
    assert open_graduated_book(tmp_path, arm) == "exp"  # an open book is left as it is
    assert _book_files(root / "exp") == opened

    loser = build_arm(experiments, "loser", "discarded")
    with pytest.raises(ValueError, match="did not graduate"):
        open_graduated_book(tmp_path, loser)
    assert list_books(root) == ["exp"] and not (root / "loser").exists()


def test_deleting_a_book_removes_its_read_only_tree_and_frees_its_id(tmp_path: Path):
    """A real book's strategy copy is read-only and its PIT cache hard-links
    shared release files. The whole tree goes, no shared file changes mode,
    and the id can be opened again."""

    arm = build_arm(tmp_path / "experiments", "exp", "graduated")
    engine_book(tmp_path, book="other")
    open_graduated_book(tmp_path, arm)
    root = paper_root(tmp_path)
    output = root / "exp" / STRATEGY_COPY_NAME / "output"
    (output / "lib").mkdir()
    (output / "lib" / "util.py").write_text("X = 1\n", encoding="utf-8")
    shared = tmp_path / "release.parquet"
    shared.write_bytes(b"release")
    shared.chmod(0o444)
    (root / "exp" / "pit").mkdir()
    os.link(shared, root / "exp" / "pit" / "release.parquet")
    for directory in (output / "lib", output):
        directory.chmod(0o555)

    delete_book(root, "exp")
    assert sorted(entry.name for entry in root.iterdir()) == ["other"]  # nothing left behind
    assert stat.S_IMODE(shared.stat().st_mode) == 0o444 and shared.read_bytes() == b"release"
    assert open_graduated_book(tmp_path, arm) == "exp"
    with pytest.raises(KeyError):
        delete_book(root, "missing")
    with pytest.raises(ValueError):
        delete_book(root, "../other")
    assert list_books(root) == ["exp", "other"]


def test_a_book_a_run_is_writing_is_refused_and_left_whole(tmp_path: Path):
    book = engine_book(tmp_path, "20260105")
    root = paper_root(tmp_path)
    before = _book_files(book)
    with (book / ".paper_engine.lock").open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)  # a Paper run owns the book
        with pytest.raises(PaperWriterBusy):
            delete_book(root, "exp")
    assert list_books(root) == ["exp"] and _book_files(book) == before
    delete_book(root, "exp")  # the run is over
    assert list_books(root) == [] and not any(root.iterdir())


def _graduated_book(tmp_path: Path, book_id: str = "exp") -> Book:
    arm = build_arm(tmp_path / "experiments", book_id, "graduated")
    return create_book(
        paper_root(tmp_path) / book_id,
        experiment_dir=arm,
        artifact_id="strategy_research_abc",
        repo_root=tmp_path,
        track="graduated",
    )


def _incubate(tmp_path: Path, book_id: str, boards: tuple[str, ...] = ("main", "gem")) -> Book:
    """An incubating book of a discarded arm, from the replay its forward
    record names (the shape an incubation record passes), on ``boards``."""

    arm = tmp_path / "experiments" / book_id
    if not arm.exists():
        build_arm(tmp_path / "experiments", book_id, "discarded")
    record = forward_record(ExperimentLedger(arm / "ledgers/experiment_ledger.jsonl").read())
    return create_book(
        paper_root(tmp_path) / book_id,
        experiment_dir=arm,
        artifact_id="strategy_research_abc",
        repo_root=tmp_path,
        track="incubating",
        source_record={"result_ref": record["result_ref"], "replay": record.get("replay") or {}},
        permitted_boards=boards,
    )


def _kill(root: Path, day: str = "20260105") -> None:
    append_jsonl_once(
        root / VERDICT_LOG_NAME,
        {"event_id": "terminal", "status": "killed", "reason": "both_kill_upper_bounds_below_zero", "date": day, "days": 126},
    )


def test_a_new_book_carries_its_track_and_pins_its_verdict_rules(tmp_path: Path):
    book = _graduated_book(tmp_path)
    record = read_json(book.root / BOOK_NAME)
    assert record["schema_version"] == 2 and record["candidate_source"] == book.candidate_source == "graduated"
    rules = record["verdict_rules"]
    assert (rules["checkpoint_days"], rules["kill_confidence"], rules["confirm_confidence"]) == (126, 0.841, 0.977)
    assert (rules["max_drawdown"], rules["active_max_drawdown"]) == (0.45, 0.30)
    assert isinstance(rules["panel_seed"], int) and book.verdict_rules == rules

    incubating = _incubate(tmp_path, "inc")
    assert read_json(incubating.root / BOOK_NAME)["candidate_source"] == "incubating"
    assert read_json(incubating.root / "source_history.json")["series"]
    with pytest.raises(ValueError, match="source_record"):
        create_book(
            paper_root(tmp_path) / "other", experiment_dir=tmp_path / "experiments/inc",
            artifact_id="strategy_research_abc", repo_root=tmp_path, track="incubating",
        )
    with pytest.raises(ValueError, match="unknown Paper book track"):
        create_book(
            paper_root(tmp_path) / "other", experiment_dir=tmp_path / "experiments/exp",
            artifact_id="strategy_research_abc", repo_root=tmp_path, track="promoted",
        )


def test_a_book_pins_the_boards_its_broker_and_its_verdict_panel_buy_on(tmp_path: Path):
    """The fixture arm carries no board stamp: a graduated book pins those its
    initial cash qualifies for, an incubating one the boards its incubation
    replayed on, and that one pinned field is what its Broker enforces and
    what its zero-skill panel draws from."""

    assert load_book(_graduated_book(tmp_path).root).profile.permitted_boards == ("main", "gem", "star", "bj")
    arm = tmp_path / "experiments" / "exp"
    small = create_book(
        paper_root(tmp_path) / "small", experiment_dir=arm, artifact_id="strategy_research_abc",
        repo_root=tmp_path, track="graduated", initial_cash=100_000,
    )
    assert read_json(small.root / BOOK_NAME)["profile"]["permitted_boards"] == ["main", "gem"]

    book = load_book(_incubate(tmp_path, "inc", boards=("main",)).root)
    assert book.profile.permitted_boards == ("main",)
    broker = DailyBroker(book.profile)
    broker.open_day("20260105", {})
    rejected = broker.execute(_order(symbol="300001.SZ"), _bar(), matched_at=MATCHED_AT, raw_price=10.0)
    assert (rejected.status, rejected.reason) == ("rejected", "board_not_permitted")
    assert broker.execute(_order(), _bar(), matched_at=MATCHED_AT, raw_price=10.0).status == "filled"
    drawn = {symbol for draw in _board_draws(book.profile.permitted_boards, 200, 1) for symbol, *_ in draw}
    assert drawn and {board_of(symbol) for symbol in drawn} == {"main"}

    with pytest.raises(ValueError, match="permitted_boards"):
        create_book(
            paper_root(tmp_path) / "other", experiment_dir=arm, artifact_id="strategy_research_abc",
            repo_root=tmp_path, track="graduated", permitted_boards=("main",),
        )


def test_the_incubating_cap_refuses_a_book_and_a_killed_book_frees_its_place(tmp_path: Path):
    root = paper_root(tmp_path)
    for index in range(INCUBATING_BOOK_CAP):
        write_book_record(root / f"inc{index}", experiment_id=f"inc{index}", candidate_source="incubating")
    write_book_record(root / "grad", experiment_id="grad")  # a graduated book never counts
    with pytest.raises(ValueError, match=f"the cap is {INCUBATING_BOOK_CAP}"):
        _incubate(tmp_path, "late")
    assert not (root / "late").exists()
    require_incubating_place(root, "inc0")  # an open book is never a second book of its own arm
    _kill(root / "inc0")
    assert book_status(root / "inc0") == "killed"
    assert _incubate(tmp_path, "late").candidate_source == "incubating"
    with pytest.raises(ValueError, match=f"the cap is {INCUBATING_BOOK_CAP}"):
        require_incubating_place(root, "later")


FUNDAMENTALS_STATUS = Path("results/data_quality/fundamental_events_status.json")
YEAR = datetime.now(CN_TZ).year


def _pit_lake(repo: Path, *, first_month: str, audited_from: str, calendar: bool = True) -> None:
    """The live lake a book's views read fundamentals from: weekday sessions
    around today, PIT fundamental events from ``first_month`` (YYYYMM) and
    their audit from ``audited_from`` (YYYYMMDD)."""

    if calendar:
        sessions = pd.bdate_range(f"{YEAR - 1}0101", f"{YEAR + 1}1231").strftime("%Y%m%d")
        path = repo / "data/raw/trade_cal/exchange=SSE" / f"year={YEAR}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"cal_date": sessions, "is_open": "1"}).to_parquet(path, index=False)
    events = repo / "data/pit/fundamental_events/income_vip"
    events.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"ts_code": []}).to_parquet(events / f"available_month={first_month}.parquet", index=False)
    (repo / FUNDAMENTALS_STATUS).parent.mkdir(parents=True, exist_ok=True)
    write_fundamental_status(repo / FUNDAMENTALS_STATUS, scope_start_date=audited_from)


def _fundamentals_arm(tmp_path: Path, name: str, window_months: int) -> Path:
    arm = build_arm(tmp_path / "experiments", name, "graduated")
    params = read_json(arm / "hitl/params.json")
    write_json_atomic(
        arm / "hitl/params.json", {**params, "include_fundamentals": True, "window_months": window_months}
    )
    return arm


def test_a_book_reading_fundamentals_opens_only_on_an_audit_covering_its_view(tmp_path: Path):
    """Every run builds the book's decision view behind the snapshot's audit
    gate, so a book whose view that gate refuses could never decide: it is
    refused when it is opened, on either track, with the rebuild that fixes
    it. The months the view loads follow the book's own window: events from
    six years back audited from three years back cover a 24-month window and
    not a 108-month one."""

    first, audited = f"{YEAR - 6}01", f"{YEAR - 3}0101"
    _pit_lake(tmp_path, first_month=first, audited_from=audited)
    assert open_graduated_book(tmp_path, _fundamentals_arm(tmp_path, "short", 24)) == "short"

    arm = _fundamentals_arm(tmp_path, "long", 108)
    with pytest.raises(ValueError, match="cannot open") as refused:
        open_graduated_book(tmp_path, arm)
    message = str(refused.value)
    assert f"loads PIT fundamental events from {first}, but their audit" in message
    assert f"starts at {audited}" in message
    command = f"python scripts/data/tushare_cron_update.py --job cn_nightly_pit_event_build --start-date {first}01 --force-run"
    assert command in message
    record = forward_record(ExperimentLedger(arm / "ledgers/experiment_ledger.jsonl").read())
    with pytest.raises(ValueError, match="cannot open"):
        create_book(
            paper_root(tmp_path) / "long-incubating", experiment_dir=arm, artifact_id="strategy_research_abc",
            repo_root=tmp_path, track="incubating", source_record=record, permitted_boards=("main",),
        )
    assert sorted(entry.name for entry in paper_root(tmp_path).iterdir()) == ["short"]  # nothing written

    # The rebuild the refusal names audits from that month: the book opens.
    write_fundamental_status(tmp_path / FUNDAMENTALS_STATUS, scope_start_date=f"{first}01")
    assert open_graduated_book(tmp_path, arm) == "long"


def test_a_book_without_fundamentals_never_asks_the_audit(tmp_path: Path):
    # An audit that covers nothing and no exchange calendar: either one read
    # would refuse the book.
    _pit_lake(tmp_path, first_month="201501", audited_from="20990101", calendar=False)
    assert open_graduated_book(tmp_path, build_arm(tmp_path / "experiments", "exp", "graduated")) == "exp"


def test_a_book_of_an_older_schema_is_refused(tmp_path: Path):
    book = _graduated_book(tmp_path)
    record = read_json(book.root / BOOK_NAME)
    (book.root / BOOK_NAME).write_text(json.dumps({**record, "schema_version": 1}), encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported Paper book schema 1"):
        load_book(book.root)
