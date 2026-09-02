#!/usr/bin/env python3
"""Mark one Agent-filed issue report resolved in its experiment's issue log.

The log is append-only: this writes a resolution line naming the report, the
outcome and a short note (name the commit when the outcome is a fix), and never
edits the report the session filed. The console's 问题反馈 page then shows the
report as resolved and drops it from the default open list.
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

from autotrade.environment.tools.report_issue import (
    ISSUE_OUTCOMES,
    append_issue_resolution,
    issue_reports_path,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True, help="Experiment id.")
    parser.add_argument("--report", required=True, help="report_id from the log.")
    parser.add_argument(
        "--outcome",
        required=True,
        choices=list(ISSUE_OUTCOMES),
        help="fixed names a repair; the other two close a report that needed none.",
    )
    parser.add_argument(
        "--note", required=True, help="Short disposition, e.g. the fixing commit."
    )
    parser.add_argument("--experiments-root", type=Path, default=Path("experiments"))
    args = parser.parse_args()

    experiment_dir = args.experiments_root / args.experiment
    if not experiment_dir.is_dir():
        parser.error(f"unknown experiment: {experiment_dir}")
    try:
        record = append_issue_resolution(
            issue_reports_path(experiment_dir),
            report_id=args.report,
            outcome=args.outcome,
            note=args.note,
        )
    except (OSError, ValueError) as exc:
        # An unknown report id, a second resolution and an unreadable log all
        # land here: say which and write nothing.
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {"experiment": args.experiment, **record}, ensure_ascii=False, sort_keys=True
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
