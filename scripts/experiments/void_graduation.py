#!/usr/bin/env python3
"""Withdraw one arm's recorded graduation: delete its Paper books, then append
a void to its ledger.

The ledger is append-only: this writes one ``verdict_void`` record naming who
voided the graduation, why and where the evidence is, and never edits the
forward record the arm produced. From then on the arm reads as voided
(``ledger.experiment_verdict``): the console shows it voided with the reason,
the graduated memory tier does not mount its skills into new arms, and no Paper
book is opened or offered for it. Its trials still count for arms that inherit
it as lineage.

A failed strategy does not stay in Paper: every book trading the arm (its
``book.json`` names the experiment, whatever the book id) is deleted in the
same operation. Every check runs before anything changes. A book that follows
real fills holds the owner's record of what he actually traded, so it is
refused unless ``--delete-real-fill-book`` says to delete it too. The books go
first and the void last: a run stopped in between -- by a book a Paper run is
writing, say -- leaves the graduation standing, and the same command run again
finishes it. ``--dry-run`` runs every check and prints the record it would
write and the books it would delete, changing nothing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[1]
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from _bootstrap import add_repo_src

add_repo_src(__file__)

from autotrade.paper.book import BOOK_NAME
from autotrade.paper.books import (
    PAPER_STATE_DIR,
    delete_book,
    follows_real_fills,
    list_books,
)
from autotrade.paper.engine import PaperWriterBusy
from autotrade.paper.storage import read_json
from autotrade.pipelines.ledger import ExperimentLedger, require_voidable, verdict_void

REPO_ROOT = Path(__file__).resolve().parents[2]


def arm_books(state_root: Path, experiment_id: str) -> list[str]:
    """The ids of the Paper books trading ``experiment_id``."""

    return [
        book_id
        for book_id in list_books(state_root)
        if read_json(state_root / book_id / BOOK_NAME).get("experiment_id") == experiment_id
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True, help="Experiment id.")
    parser.add_argument("--by", required=True, help="Who voids the graduation.")
    parser.add_argument("--reason", required=True, help="Why, in a sentence or two.")
    parser.add_argument(
        "--evidence",
        required=True,
        help="Repository-relative path of the evidence (a note or a directory).",
    )
    parser.add_argument("--experiments-root", type=Path, default=REPO_ROOT / "experiments")
    parser.add_argument("--paper-state-root", type=Path, default=REPO_ROOT / PAPER_STATE_DIR)
    parser.add_argument(
        "--delete-real-fill-book",
        action="store_true",
        help="Also delete a book that follows real fills, and with it the owner's record of what he traded.",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Check and print the record and the books; change nothing."
    )
    args = parser.parse_args()

    experiment_dir = args.experiments_root / args.experiment
    if not experiment_dir.is_dir():
        parser.error(f"unknown experiment: {experiment_dir}")
    evidence = Path(args.evidence)
    if evidence.is_absolute() or not (REPO_ROOT / evidence).exists():
        parser.error(f"evidence must be an existing repository-relative path: {args.evidence}")
    ledger = ExperimentLedger(experiment_dir / "ledgers" / "experiment_ledger.jsonl")
    record = verdict_void(
        args.experiment,
        voided_by=args.by,
        reason=args.reason,
        evidence_ref=str(evidence),
    )
    paper_root = args.paper_state_root
    try:
        # The checks the append runs, then the books', before anything changes.
        require_voidable(ledger.read(), record)
        books = arm_books(paper_root, args.experiment)
        real = [book_id for book_id in books if follows_real_fills(paper_root / book_id)]
        if real and not args.delete_real_fill_book:
            raise ValueError(
                f"Paper book {', '.join(real)} follows real fills: its fills.jsonl is the owner's record "
                "of what he actually traded. Nothing was changed; keep a copy if it is wanted, then pass "
                "--delete-real-fill-book to delete the book with the void"
            )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if not args.dry_run:
        try:
            for book_id in books:
                try:
                    delete_book(paper_root, book_id)
                except PaperWriterBusy as exc:
                    raise PaperWriterBusy(f"a Paper run is writing book {book_id}; it is left whole") from exc
            ledger.append(record)
        except (KeyError, OSError, RuntimeError, ValueError) as exc:
            print(
                f"error: {exc}. The graduation is not voided yet; the same command, run again, finishes it",
                file=sys.stderr,
            )
            return 2
    written = record if args.dry_run else ledger.read(str(record["record_type"]))[-1]
    print(
        json.dumps(
            {"dry_run": args.dry_run, "paper_books_deleted": books, **written}, ensure_ascii=False, sort_keys=True
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
