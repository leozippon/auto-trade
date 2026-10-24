"""Which datasets a live Paper book mounts have changed vintage since its pin.

A dataset's vintage id moves when the vendor's history of it was restated (a
restatement, promotion or revert in the raw lake), not when rows are appended.
Paper re-issues every earlier decision of a book against the newest release
and compares the orders with its journal; a dataset whose vintage moved since
the release the book last pinned is where a mismatch on the next run comes
from. The nightly full audit writes this report before the morning runs.
"""

from __future__ import annotations

import json
from pathlib import Path

from autotrade.environment.data.contracts import (
    RAW_GENERATION_FILENAME,
    dataset_vintages,
)
from autotrade.pipelines.pit_backend import required_release_raw_datasets

from .book import book_status, load_book
from .books import list_books
from .pit import LIVE_GENERATION, PIT_CACHE_NAME


def _record(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def book_vintage_changes(book_root: str | Path) -> dict[str, object]:
    """The book's pinned generation, the mounted datasets whose vintage
    changed since, and those its pinned release records no vintage for."""

    book = load_book(book_root)
    cache = book.root / PIT_CACHE_NAME
    pinned = sorted(
        entry.name for entry in cache.iterdir() if entry.is_dir() and not entry.name.startswith(".")
    ) if cache.is_dir() else []
    mounted = required_release_raw_datasets(book.snapshot_config)
    current = dataset_vintages(_record(book.raw_dir / RAW_GENERATION_FILENAME), mounted)
    generation = pinned[-1] if pinned else ""
    release_record = (
        book.raw_dir.parent / "research_releases" / generation / "raw" / RAW_GENERATION_FILENAME
        if generation and generation != LIVE_GENERATION
        else None
    )
    before = dataset_vintages(_record(release_record), mounted) if release_record is not None else {}
    return {
        "generation": generation,
        "changed": sorted(name for name in current if name in before and before[name] != current[name]),
        "not_recorded": sorted(name for name in mounted if name not in before),
    }


def vintage_report(state_root: str | Path) -> dict[str, dict[str, object]]:
    """``book_vintage_changes`` of every book that is not killed."""

    root = Path(state_root)
    return {
        book_id: book_vintage_changes(root / book_id)
        for book_id in list_books(root)
        if book_status(root / book_id) != "killed"
    }


__all__ = ["book_vintage_changes", "vintage_report"]
