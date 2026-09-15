"""Several Paper books under one state root: listing, the book-by-book run, and
moving a single-book root into its book directory."""

from __future__ import annotations

import fcntl
import hashlib
import json
from pathlib import Path

import pytest

from autotrade.paper.books import (
    list_books,
    migrate_single_root,
    run_books,
    validate_book_id,
)
from autotrade.paper.engine import PaperWriterBusy
from autotrade.paper.orders import order_sheet
from tests.unit.paper_book_fixture import (
    FAILING_STRATEGY,
    engine_book,
    paper_root,
    run_days,
    write_book_record,
)


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


def test_book_ids_are_single_path_segments():
    assert validate_book_id("arm-graduated_2026.09") == "arm-graduated_2026.09"
    for bad in ("", ".hidden", "a/b", "..", "a..b", "x" * 97):
        with pytest.raises(ValueError):
            validate_book_id(bad)


def _tree(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_a_single_book_root_moves_into_its_book_directory_and_keeps_running(tmp_path: Path):
    root = paper_root(tmp_path)
    write_book_record(root, experiment_id="exp")
    run_days(root, "20260105", "20260106")
    before = _tree(root)
    sheet = order_sheet(root, "20260106")
    with pytest.raises(RuntimeError, match="single-book layout"):
        list_books(root)

    target = migrate_single_root(root)

    assert target == root.resolve() / "exp" and list_books(root) == ["exp"]
    after = _tree(target)
    # Every file moved as it was; the state differs only in the strategy path,
    # which names the same copy at its new location.
    assert after.keys() == before.keys()
    assert {name for name in before if before[name] != after[name]} == {".paper_state.json"}
    state = json.loads((target / ".paper_state.json").read_text(encoding="utf-8"))
    assert state["strategy_path"] == str((target / "strategy/main.py").resolve())
    # The moved book reprints its decided day unchanged and decides the next one.
    assert order_sheet(target, "20260106") == sheet
    run_days(target, "20260106", "20260107")
    assert _decided(target) == ["20260105", "20260106", "20260107"]


def test_a_single_book_root_is_not_moved_while_its_writer_runs(tmp_path: Path):
    root = paper_root(tmp_path)
    write_book_record(root)
    run_days(root, "20260105")
    before = _tree(root)
    with (root / ".paper_engine.lock").open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="writer is running"):
            migrate_single_root(root)
    assert _tree(root) == before
