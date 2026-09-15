"""Several Paper books under one state root: listing and the book-by-book run."""

from __future__ import annotations

import fcntl
import json
from pathlib import Path

import pytest

from autotrade.paper.books import (
    list_books,
    run_books,
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


def test_a_single_book_root_is_refused(tmp_path: Path):
    root = paper_root(tmp_path)
    write_book_record(root, experiment_id="exp")
    with pytest.raises(RuntimeError, match="single-book layout"):
        list_books(root)
