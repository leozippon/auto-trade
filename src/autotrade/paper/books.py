"""Several Paper books side by side under one Paper state root.

Each book is an independent account in its own directory,
``<state root>/<book id>/``, holding everything ``create_book`` and the engine
write for it (``book.json``, state, journals, artifact copy, PIT cache) and its
own writer lock. A run goes through the books one at a time; one book's failure
is reported for that book and never stops the others.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path

from .book import BOOK_NAME

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


__all__ = ["BOOK_ID_PATTERN", "list_books", "run_books", "validate_book_id"]
