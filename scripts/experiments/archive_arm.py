#!/usr/bin/env python3
"""Archive arms out of the console's home list, or restore them.

An arm is archived by one marker file inside its own directory,
``hitl/archived.json``, which records when and why; ``--restore`` removes it.
The directory never moves, so only the console's home list changes: it leaves
the arm out (and never summarizes it) and offers it under 已归档 instead, and
its experiment page still opens. Lineage, the consistency check, the round
fill queue, the running caps, operating memory and Paper read the arm exactly
as before. The console need not be running.

Only an arm that is over and holds nothing live is archived. Every arm named
is checked before anything changes, and the command refuses them all when one
has a live or launching worker, a graduation that has not been voided
(``void_graduation.py``), or a Paper book. Archiving an archived arm keeps its
marker, and restoring an arm that is not archived changes nothing.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[1]
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from _bootstrap import add_repo_src

add_repo_src(__file__)

from autotrade.environment.runtime import write_json_atomic
from autotrade.paper.books import PAPER_STATE_DIR, experiment_books
from autotrade.pipelines.hitl_state import HITL_DIR_NAME
from autotrade.pipelines.ledger import experiment_verdict
from autotrade.webui.registry import (
    ARCHIVED_NAME,
    experiment_state,
    read_ledger_records,
    resolve_experiment_dir,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def refusal(directory: Path, paper_root: Path) -> str | None:
    """Why ``directory`` must not be archived, or None."""

    state = experiment_state(directory)
    if state.get("worker_alive") or state.get("state") == "launching":
        return "has a live worker; stop it first"
    verdict = experiment_verdict(read_ledger_records(directory))
    if verdict is not None and verdict["status"] == "graduated":
        return "is a graduate; void the graduation first (void_graduation.py)"
    books = experiment_books(paper_root, directory.name)
    if books:
        return f"has Paper book {', '.join(books)}"
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiments", nargs="+", metavar="EXPERIMENT", help="Experiment id.")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--reason", help="Why the arms are archived, in a sentence.")
    action.add_argument("--restore", action="store_true", help="Remove the arms' archive markers.")
    parser.add_argument("--experiments-root", type=Path, default=REPO_ROOT / "experiments")
    parser.add_argument("--paper-state-root", type=Path, default=REPO_ROOT / PAPER_STATE_DIR)
    args = parser.parse_args()
    if not args.restore and not args.reason.strip():
        parser.error("--reason must say why")

    errors: list[str] = []
    markers: list[Path] = []
    for experiment_id in args.experiments:
        try:
            directory = resolve_experiment_dir(args.experiments_root, experiment_id)
            reason = None if args.restore else refusal(directory, args.paper_state_root)
        except (KeyError, OSError, RuntimeError, ValueError) as exc:
            # A KeyError's str() is its quoted repr; say the message itself.
            errors.append(f"{experiment_id}: {exc.args[0] if isinstance(exc, KeyError) else exc}")
            continue
        if reason is not None:
            errors.append(f"{experiment_id} {reason}")
        markers.append(directory / HITL_DIR_NAME / ARCHIVED_NAME)
    if errors:
        for line in errors:
            print(f"error: {line}", file=sys.stderr)
        print("error: nothing was changed", file=sys.stderr)
        return 2

    changed: list[str] = []
    stamp = datetime.now(UTC).isoformat(timespec="seconds")
    for marker in markers:
        if args.restore and marker.is_file():
            marker.unlink()
        elif not args.restore and not marker.is_file():
            write_json_atomic(marker, {"archived_at": stamp, "reason": args.reason.strip()})
        else:
            continue
        changed.append(marker.parents[1].name)
    print(
        json.dumps(
            {"action": "restore" if args.restore else "archive", "changed": changed},
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
