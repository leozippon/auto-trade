"""Several Paper books side by side under one Paper state root.

Each book is an independent account in its own directory,
``<state root>/<book id>/``, holding everything ``create_book`` and the engine
write for it (``book.json``, state, journals, artifact copy, PIT cache) and its
own writer lock. A run goes through the books one at a time; one book's failure
is reported for that book and never stops the others.
"""

from __future__ import annotations

import fcntl
import os
import re
from collections.abc import Callable
from pathlib import Path

from .book import BOOK_NAME
from .engine import PAPER_LOCK_NAME, PAPER_STATE_NAME
from .storage import read_json, write_json_atomic

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
            f"{root} holds one book in the single-book layout; move it into its own directory "
            "with `run_paper.py migrate`"
        )
    if not root.is_dir():
        return []
    return sorted(
        entry.name
        for entry in root.iterdir()
        if entry.is_dir() and BOOK_ID_PATTERN.fullmatch(entry.name) and (entry / BOOK_NAME).is_file()
    )


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


def migrate_single_root(state_root: str | Path, book_id: str | None = None) -> Path:
    """Move a single-book root into ``<state_root>/<book_id>/``, in place.

    The book's files move as they are: journals, state, artifact copy, fitted
    state and PIT cache. The one recorded absolute path, the strategy file the
    state was created with, is rewritten to the same file at its new location.
    The move is refused while the book's writer holds its lock.
    """

    root = Path(state_root).resolve()
    record = read_json(root / BOOK_NAME)
    if not record:
        raise FileNotFoundError(f"no single-book layout at {root}")
    book_id = validate_book_id(book_id or str(record["experiment_id"]))
    staging = root.with_name(f".{root.name}.migrating")
    with (root / PAPER_LOCK_NAME).open("a+b") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("the book's writer is running; migrate after it finishes") from exc
        state = read_json(root / PAPER_STATE_NAME)
        old_strategy = str((root / str(record["strategy_path"])).resolve())
        if state and state.get("strategy_path") != old_strategy:
            raise RuntimeError(
                f"state records strategy {state.get('strategy_path')!r}, not the book's copy {old_strategy!r}"
            )
        # Two renames on one filesystem: the book directory becomes the book's
        # own directory under a fresh root, every inode untouched. The lock
        # file moves with it, so the writer lock is held until the state
        # names the strategy at its new path.
        os.rename(root, staging)
        root.mkdir(mode=0o700)
        target = root / book_id
        os.rename(staging, target)
        if state:
            write_json_atomic(
                target / PAPER_STATE_NAME,
                {**state, "strategy_path": str((target / str(record["strategy_path"])).resolve())},
            )
    return target


__all__ = ["BOOK_ID_PATTERN", "list_books", "migrate_single_root", "run_books", "validate_book_id"]
