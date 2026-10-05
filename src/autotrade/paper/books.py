"""Several Paper books side by side under one Paper state root.

Each book is an independent account in its own directory,
``<state root>/<book id>/``, holding everything ``create_book`` and the engine
write for it (``book.json``, state, journals, artifact copy, PIT cache) and its
own writer lock. A run goes through the books one at a time, the ones the owner
trades by hand first; one book's failure is reported for that book and never
stops the others.
"""

from __future__ import annotations

import re
import uuid
from collections.abc import Callable
from pathlib import Path

from autotrade.environment.sandbox import remove_sandbox_tree
from autotrade.pipelines.ledger import ExperimentLedger, paper_candidate

from .book import BOOK_NAME, create_book
from .engine import writer_lock
from .fills import FILLS_NAME

# The Paper state root, relative to the repository root.
PAPER_STATE_DIR = Path("data/trading/paper")
# One path segment: experiment ids already satisfy it, so a book created from
# an experiment is named after it by default.
BOOK_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}")


def validate_book_id(book_id: str) -> str:
    if not BOOK_ID_PATTERN.fullmatch(book_id) or ".." in book_id:
        raise ValueError(f"invalid Paper book id {book_id!r}: letters, digits, '_', '.', '-'")
    return book_id


def list_books(state_root: str | Path) -> list[str]:
    """The ids of the books under ``state_root``, sorted.

    A root that is itself a book is the single-book layout from before books
    had directories; it is refused rather than read as one unnamed book.
    """

    root = Path(state_root)
    if (root / BOOK_NAME).exists():
        raise RuntimeError(
            f"{root} holds one book in the single-book layout; books live in "
            "<state root>/<book id>/"
        )
    if not root.is_dir():
        return []
    return sorted(
        entry.name
        for entry in root.iterdir()
        if entry.is_dir() and BOOK_ID_PATTERN.fullmatch(entry.name) and (entry / BOOK_NAME).is_file()
    )


def follows_real_fills(book_root: str | Path) -> bool:
    """Whether the owner switched this book to real fills.

    He did so because he trades it by hand, and its ``fills.jsonl`` -- the
    switch and every outcome he recorded -- is his record of what he actually
    traded.
    """

    return (Path(book_root) / FILLS_NAME).is_file()


def run_order(state_root: str | Path) -> list[str]:
    """Every book in the order a run takes them: the books that follow real
    fills first, so the sheets the owner trades from are ready first."""

    root = Path(state_root)
    return sorted(list_books(root), key=lambda book_id: not follows_real_fills(root / book_id))


def run_books(
    state_root: str | Path, book_ids: list[str], run_book: Callable[[str, Path], None]
) -> dict[str, BaseException]:
    """Run each book in turn; the failures, by book id. Every book is attempted."""

    failures: dict[str, BaseException] = {}
    for book_id in book_ids:
        try:
            run_book(book_id, Path(state_root) / validate_book_id(book_id))
        except Exception as exc:  # noqa: BLE001 - one book's failure must not stop the others
            failures[book_id] = exc
    return failures


def open_graduated_book(repo_root: str | Path, experiment_dir: str | Path) -> str:
    """Create a graduate's Paper book when none exists yet; the book id.

    The book is named after the experiment and trades its Paper candidate. A
    book already on disk is left as it is.
    """
    repo_root = Path(repo_root).resolve()
    experiment = Path(experiment_dir).resolve(strict=True)
    book_id = validate_book_id(experiment.name)
    state_root = repo_root / PAPER_STATE_DIR
    if (state_root / book_id / BOOK_NAME).is_file():
        return book_id
    records = ExperimentLedger(experiment / "ledgers" / "experiment_ledger.jsonl").read()
    candidate = paper_candidate(records)
    if candidate is None:
        raise ValueError(f"{book_id} has no Paper candidate: it did not graduate")
    list_books(state_root)
    create_book(
        state_root / book_id,
        experiment_dir=experiment,
        artifact_id=str(candidate["artifact_id"]),
        repo_root=repo_root,
        note="graduated",
    )
    return book_id


def delete_book(state_root: str | Path, book_id: str) -> None:
    """Remove one Paper book whole, or leave it listed as it was.

    An unknown id raises ``KeyError`` and a book a Paper run is writing raises
    ``PaperWriterBusy``, both before anything changes. Under the book's writer
    lock the directory leaves the roster in one rename to a hidden name, and
    that tree is removed as a finished sandbox is: its read-only directories
    (the frozen strategy copy) are unlocked, and files a strategy container
    wrote are removed through the container. Files that still remain are
    named in the error; the book itself is gone either way.
    """
    book_id = validate_book_id(book_id)
    root = Path(state_root).resolve()
    if book_id not in list_books(root):
        raise KeyError(f"unknown Paper book: {book_id}")
    path = root / book_id
    with writer_lock(path):
        removed = root / f".deleted-{book_id}-{uuid.uuid4().hex[:8]}"
        path.rename(removed)
        if not remove_sandbox_tree(removed):
            raise OSError(f"Paper book {book_id} is deleted, but files remain under {removed}")


__all__ = [
    "BOOK_ID_PATTERN",
    "PAPER_STATE_DIR",
    "delete_book",
    "follows_real_fills",
    "list_books",
    "open_graduated_book",
    "run_books",
    "run_order",
    "validate_book_id",
]
