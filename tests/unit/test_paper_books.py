"""Paper books under one state root: creation, listing and the book-by-book run."""

from __future__ import annotations

import fcntl
import json
import os
import stat
from pathlib import Path

import pytest

from autotrade.environment.artifacts import artifact_fingerprint
from autotrade.paper import fills
from autotrade.paper.book import STRATEGY_COPY_NAME, create_book
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
from tests.unit.paper_book_fixture import (
    FAILING_STRATEGY,
    engine_book,
    paper_root,
    run_days,
    write_book_record,
)
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
