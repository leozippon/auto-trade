#!/usr/bin/env python3
"""Withdraw one arm's recorded graduation by appending a void to its ledger.

The ledger is append-only: this writes one ``verdict_void`` record naming who
voided the graduation, why and where the evidence is, and never edits the
forward record the arm produced. From then on the arm reads as voided
(``ledger.experiment_verdict``): the console shows it voided with the reason,
the graduated memory tier does not mount its skills into new arms, and no Paper
book is opened or offered for it. Its trials still count for arms that inherit
it as lineage. ``--dry-run`` runs every check and prints the record it would
write without writing it.
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

from autotrade.pipelines.ledger import ExperimentLedger, require_voidable, verdict_void

REPO_ROOT = Path(__file__).resolve().parents[2]


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
    parser.add_argument(
        "--dry-run", action="store_true", help="Check and print the record; write nothing."
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
    try:
        if args.dry_run:
            # The checks the append runs, without writing.
            require_voidable(ledger.read(), record)
        else:
            ledger.append(record)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    written = record if args.dry_run else ledger.read(str(record["record_type"]))[-1]
    print(json.dumps({"dry_run": args.dry_run, **written}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
