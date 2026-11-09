"""Paper accounts under one state root: creation, listing and the book-by-book run."""

from __future__ import annotations

import fcntl
import json
import os
import stat
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from autotrade.common.records import (
    read_json_object,
    write_json_atomic,
)
from autotrade.data_sources.tushare.cron_update import resolve_job_start_date
from autotrade.environment.artifacts import artifact_fingerprint
from autotrade.environment.broker import (
    DailyBroker,
    default_permitted_boards,
)
from autotrade.environment.broker_core import board_of
from autotrade.environment.sandbox import SandboxConfig
from autotrade.environment.strategy import CN_TZ
from autotrade.paper import fills
from autotrade.paper.book import (
    BOOK_CAP,
    BOOK_NAME,
    PAPER_DECISION_TIMEOUT_SECONDS,
    PAPER_INITIAL_CASH,
    STRATEGY_COPY_NAME,
    VERDICT_LOG_NAME,
    Book,
    book_status,
    create_book,
    load_book,
    require_book_place,
)
from autotrade.paper.books import (
    delete_book,
    follows_real_fills,
    list_books,
    open_registered_book,
    run_books,
    run_order,
    validate_book_id,
)
from autotrade.paper.engine import PaperWriterBusy
from autotrade.paper.storage import append_jsonl_once
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
from tests.unit.webui_research_experiment import build_experiment


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
    the owner trades from are ready first; a book that asks for a GPU runs
    after every book that asks for none, whatever its fills, so its wait for a
    card never holds one up. The order is the books' own state, so a deleted
    book leaves nothing behind to name."""

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
    _registered_book(tmp_path, "aaa")  # asks for a GPU, as its experiment did
    fills.enable(root / "aaa")
    assert run_order(root) == ["beta", "alpha", "gamma", "aaa"]


def test_a_new_book_records_the_content_address_of_the_artifact_it_trades(tmp_path: Path):
    """Creation copies the registered artifact into the book and records what it
    copied. The recorded fingerprint is the one content address the pipeline
    uses for a strategy artifact, over the book's own copy, so the bytes the
    book trades are named exactly as research named the artifact it approved."""

    experiment = build_experiment(tmp_path / "experiments", "exp", "registered_with_record")
    book = create_book(
        paper_root(tmp_path) / "exp",
        experiment_dir=experiment,
        artifact_id="strategy_research_abc",
        repo_root=tmp_path,
        note="参考簿（观察中）",
    )
    record = json.loads((book.root / "book.json").read_text(encoding="utf-8"))
    copy = book.root / STRATEGY_COPY_NAME
    source = experiment / "artifacts/strategy/registered/strategy_research_abc"
    assert book.strategy_path == copy / "output" / "main.py"
    assert book.models_dir is None
    assert record["artifact_fingerprint"] == artifact_fingerprint(copy / "output")
    assert record["artifact_fingerprint"] == artifact_fingerprint(source / "output")


def test_book_ids_are_single_path_segments():
    assert validate_book_id("experiment-registered_2026.09") == "experiment-registered_2026.09"
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


def test_a_registered_strategy_opens_its_book_once_and_an_experiment_without_one_opens_none(tmp_path: Path):
    experiments = tmp_path / "elsewhere"  # any experiments root, not only <repo>/experiments
    experiment = build_experiment(experiments, "exp", "registered_with_record")
    root = paper_root(tmp_path)
    assert open_registered_book(tmp_path, experiment) == "exp"
    record = json.loads((root / "exp" / "book.json").read_text(encoding="utf-8"))
    assert (record["experiment_id"], record["artifact_id"], record["note"]) == (
        "exp", "strategy_research_abc", "registered"
    )
    assert "candidate_source" not in record
    opened = _book_files(root / "exp")
    assert open_registered_book(tmp_path, experiment) == "exp"  # an open book is left as it is
    assert _book_files(root / "exp") == opened

    loser = build_experiment(experiments, "loser", "deadline")
    with pytest.raises(ValueError, match="no registered strategy with a forward record"):
        open_registered_book(tmp_path, loser)
    assert list_books(root) == ["exp"] and not (root / "loser").exists()


def test_deleting_a_book_removes_its_read_only_tree_and_frees_its_id(tmp_path: Path):
    """A real book's strategy copy is read-only and its PIT cache hard-links
    shared release files. The whole tree goes, no shared file changes mode,
    and the id can be opened again."""

    experiment = build_experiment(tmp_path / "experiments", "exp", "registered_with_record")
    engine_book(tmp_path, book="other")
    open_registered_book(tmp_path, experiment)
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
    assert open_registered_book(tmp_path, experiment) == "exp"
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


def _registered_book(tmp_path: Path, book_id: str = "exp", **params: object) -> Book:
    """The book of a registered strategy, its experiment's params overlaid with ``params``."""

    experiment = build_experiment(tmp_path / "experiments", book_id, "registered_with_record")
    if params:
        stamped = read_json_object(experiment / "hitl" / "params.json")
        write_json_atomic(experiment / "hitl" / "params.json", {**stamped, **params})
    return create_book(
        paper_root(tmp_path) / book_id,
        experiment_dir=experiment,
        artifact_id="strategy_research_abc",
        repo_root=tmp_path,
    )


def _kill(root: Path, day: str = "20260105") -> None:
    append_jsonl_once(
        root / VERDICT_LOG_NAME,
        {"event_id": "terminal", "status": "killed", "reason": "kill_upper_bound_below_zero", "date": day, "days": 126},
    )


def test_a_new_book_pins_its_verdict_rules(tmp_path: Path):
    book = _registered_book(tmp_path)
    record = read_json_object(book.root / BOOK_NAME)
    assert record["schema_version"] == 4
    rules = record["verdict_rules"]
    assert (rules["checkpoint_days"], rules["kill_confidence"], rules["confirm_confidence"]) == (126, 0.841, 0.977)
    assert (rules["max_drawdown"], rules["active_max_drawdown"]) == (0.45, 0.30)
    assert isinstance(rules["baseline_seed"], int) and book.verdict_rules == rules
    assert read_json_object(book.root / "source_history.json")["series"]


def test_a_book_pins_the_boards_its_broker_and_its_verdict_random_baseline_buy_on(tmp_path: Path):
    """The fixture experiment carries no board stamp: a book pins those its
    initial cash qualifies for -- CNY 200,000 unless its creator names other
    capital, whatever the experiment's own -- and of an experiment stamped with
    boards only those, and that one pinned field is what its Broker enforces and
    what its random baseline draws from."""

    default = load_book(_registered_book(tmp_path).root).profile
    assert (default.initial_cash, default.permitted_boards) == (PAPER_INITIAL_CASH, ("main", "gem"))
    assert PAPER_INITIAL_CASH == 200_000
    experiment = tmp_path / "experiments" / "exp"
    large = create_book(
        paper_root(tmp_path) / "large", experiment_dir=experiment, artifact_id="strategy_research_abc",
        repo_root=tmp_path, initial_cash=1_000_000,
    )
    assert read_json_object(large.root / BOOK_NAME)["profile"]["permitted_boards"] == ["main", "gem", "star", "bj"]

    book = load_book(_registered_book(tmp_path, "main_only", permitted_boards=["main"]).root)
    assert book.profile.permitted_boards == ("main",)
    broker = DailyBroker(book.profile)
    broker.open_day("20260105", {}, distributions={})  # every experiment's account is taxed
    rejected = broker.execute(_order(symbol="300001.SZ"), _bar(), matched_at=MATCHED_AT, raw_price=10.0)
    assert (rejected.status, rejected.reason) == ("rejected", "board_not_permitted")
    assert broker.execute(_order(), _bar(), matched_at=MATCHED_AT, raw_price=10.0).status == "filled"
    drawn = {symbol for draw in _board_draws(book.profile.permitted_boards, 200, 1) for symbol, *_ in draw}
    assert drawn and {board_of(symbol) for symbol in drawn} == {"main"}


def test_a_paper_account_keeps_only_the_boards_its_own_capital_qualifies_for(tmp_path: Path):
    """An experiment stamped with every board at CNY 1,000,000 opens a CNY 200,000
    Paper account that may not buy on the STAR Market or the Beijing exchange: both ask
    CNY 500,000 of assets, whatever the experiment could buy."""

    experiment = build_experiment(tmp_path / "experiments", "exp", "registered_with_record")
    params = read_json_object(experiment / "hitl" / "params.json")
    stamp = {"initial_cash": 1_000_000.0, "permitted_boards": ["main", "gem", "star", "bj"]}
    write_json_atomic(experiment / "hitl" / "params.json", {**params, **stamp})
    book = create_book(
        paper_root(tmp_path) / "exp", experiment_dir=experiment, artifact_id="strategy_research_abc",
        repo_root=tmp_path,
    )
    assert read_json_object(book.root / BOOK_NAME)["profile"]["permitted_boards"] == ["main", "gem"]
    assert load_book(book.root).profile.permitted_boards == default_permitted_boards(PAPER_INITIAL_CASH)


def test_the_book_cap_refuses_a_book_and_a_killed_book_frees_its_place(tmp_path: Path):
    root = paper_root(tmp_path)
    for index in range(BOOK_CAP):
        write_book_record(root / f"fz{index}", experiment_id=f"fz{index}")
    late = build_experiment(tmp_path / "experiments", "late", "registered_with_record")
    with pytest.raises(ValueError, match=f"the cap is {BOOK_CAP}"):
        open_registered_book(tmp_path, late)
    assert not (root / "late").exists()
    require_book_place(root, "fz0")  # an open book is never a second book of its own experiment
    _kill(root / "fz0")
    assert book_status(root / "fz0") == "killed"
    assert open_registered_book(tmp_path, late) == "late"
    with pytest.raises(ValueError, match=f"the cap is {BOOK_CAP}"):
        require_book_place(root, "later")


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


def _fundamentals_experiment(tmp_path: Path, name: str, window_months: int) -> Path:
    experiment = build_experiment(tmp_path / "experiments", name, "registered_with_record")
    params = read_json_object(experiment / "hitl/params.json")
    write_json_atomic(
        experiment / "hitl/params.json", {**params, "include_fundamentals": True, "window_months": window_months}
    )
    return experiment


def test_a_book_reading_fundamentals_opens_only_on_an_audit_covering_its_view(tmp_path: Path):
    """Every run builds the book's decision view behind the snapshot's audit
    gate, so a book whose view that gate refuses could never decide: it is
    refused when it is opened, naming the schedule setting
    that makes the nightly audit cover it. The months the view loads follow
    the book's own window: events from six years back audited from three
    years back cover a 24-month window and not a 108-month one."""

    first, audited = f"{YEAR - 6}01", f"{YEAR - 3}0101"
    _pit_lake(tmp_path, first_month=first, audited_from=audited)
    assert open_registered_book(tmp_path, _fundamentals_experiment(tmp_path, "short", 24)) == "short"

    experiment = _fundamentals_experiment(tmp_path, "long", 108)
    with pytest.raises(ValueError, match="cannot open") as refused:
        open_registered_book(tmp_path, experiment)
    message = str(refused.value)
    assert f"loads PIT fundamental events from {first}, but their audit" in message
    assert f"starts at {audited}" in message
    setting = f'Set "start_date": "{first}01" on job cn_nightly_pit_event_build in configs/tushare_update_schedule.json'
    assert setting in message
    assert "`python scripts/data/tushare_cron_update.py --job cn_nightly_pit_event_build --force-run` once" in message
    # The key it names is the one the nightly runner reads as that job's start.
    assert resolve_job_start_date({"start_date": f"{first}01"}, {}, "20991231") == f"{first}01"
    assert sorted(entry.name for entry in paper_root(tmp_path).iterdir()) == ["short"]  # nothing written

    # The nightly run the refusal configures audits from that month: the book opens.
    write_fundamental_status(tmp_path / FUNDAMENTALS_STATUS, scope_start_date=f"{first}01")
    assert open_registered_book(tmp_path, experiment) == "long"


def test_a_book_without_fundamentals_never_asks_the_audit(tmp_path: Path):
    # An audit that covers nothing and no exchange calendar: either one read
    # would refuse the book.
    _pit_lake(tmp_path, first_month="201501", audited_from="20990101", calendar=False)
    assert open_registered_book(tmp_path, build_experiment(tmp_path / "experiments", "exp", "registered_with_record")) == "exp"


def test_a_book_of_an_older_schema_is_refused(tmp_path: Path):
    """Schema 3 recorded a candidate source every book shared; such a book is
    refused, not read without it."""
    book = _registered_book(tmp_path)
    record = read_json_object(book.root / BOOK_NAME)
    for older in (1, 3):
        stale = {**record, "schema_version": older, "candidate_source": "registered"}
        (book.root / BOOK_NAME).write_text(json.dumps(stale), encoding="utf-8")
        with pytest.raises(ValueError, match=f"unsupported Paper account schema {older}"):
            load_book(book.root)


def test_every_book_decides_under_the_paper_decision_clock(tmp_path: Path):
    """A new book carries no decision clock: every Paper decision call runs under
    PAPER_DECISION_TIMEOUT_SECONDS, whatever the experiment's."""
    book = _registered_book(tmp_path, strategy_inference_timeout_seconds=7200)
    record = read_json_object(book.root / BOOK_NAME)
    assert "timeout_seconds" not in record["sandbox"]
    assert book.sandbox.timeout_seconds == PAPER_DECISION_TIMEOUT_SECONDS != 7200
    assert SandboxConfig.from_record({**record["sandbox"], "timeout_seconds": PAPER_DECISION_TIMEOUT_SECONDS}) == book.sandbox


def test_the_vintage_report_names_what_was_restated_since_each_pin(tmp_path: Path):
    """The nightly audit's Paper vintage report: for each live book, the
    datasets it mounts whose vintage moved since the release it last pinned."""

    from autotrade.environment.data.contracts import RAW_GENERATION_FILENAME
    from autotrade.paper.vintages import vintage_report
    from autotrade.pipelines.pit_backend import required_release_raw_datasets

    book = _registered_book(tmp_path)
    mounted = required_release_raw_datasets(book.snapshot_config)
    release_raw = book.raw_dir.parent / "research_releases" / "gen_pinned" / "raw"
    release_raw.mkdir(parents=True)

    def record(path: Path, vintages: dict[str, str]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"schema_version": 3, "state": "committed", "datasets": {
            name: {"vintage": value} for name, value in vintages.items()
        }}), encoding="utf-8")

    record(release_raw / RAW_GENERATION_FILENAME, {name: "v1" for name in mounted})
    record(book.raw_dir / RAW_GENERATION_FILENAME, {**{name: "v1" for name in mounted}, mounted[0]: "v2", "unmounted": "v9"})
    (book.root / "pit" / "gen_pinned").mkdir(parents=True)
    assert vintage_report(paper_root(tmp_path)) == {
        "exp": {"generation": "gen_pinned", "changed": [mounted[0]], "not_recorded": []},
    }
    # A release published before vintages were recorded cannot tell.
    record(release_raw / RAW_GENERATION_FILENAME, {})
    assert vintage_report(paper_root(tmp_path))["exp"]["not_recorded"] == sorted(mounted)
