#!/usr/bin/env python3
"""Mark one Agent-filed report resolved in its experiment's feedback log.

``issue`` answers a ``report_issue`` report (the console's 问题反馈 page),
``skill`` a ``skill_feedback`` report (技能反馈). The log is append-only: this
writes a resolution line naming the report, the outcome and a short note (name
the commit, or the edited entry, when the outcome is a repair), and never edits
the report the session filed. The console page then shows the report as
resolved and drops it from the default open list.
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
from autotrade.environment.tools.skill_feedback import (
    SKILL_FEEDBACK_OUTCOMES,
    append_skill_feedback_resolution,
    skill_feedback_path,
)

# Each log: where it lives and how a resolution is appended to it; the append
# refuses an outcome outside the log's own vocabulary.
LOGS = {
    "issue": (issue_reports_path, append_issue_resolution),
    "skill": (skill_feedback_path, append_skill_feedback_resolution),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", choices=list(LOGS), help="Which feedback log to answer.")
    parser.add_argument("--experiment", required=True, help="Experiment id.")
    parser.add_argument("--report", required=True, help="report_id from the log.")
    parser.add_argument(
        "--outcome",
        required=True,
        help=(
            f"issue: {', '.join(ISSUE_OUTCOMES)}; skill: {', '.join(SKILL_FEEDBACK_OUTCOMES)}. "
            "fixed and skill_updated name a repair; the other two close a report "
            "that needed none."
        ),
    )
    parser.add_argument(
        "--note", required=True, help="Short disposition, e.g. the fixing commit."
    )
    parser.add_argument("--experiments-root", type=Path, default=Path("experiments"))
    args = parser.parse_args()

    experiment_dir = args.experiments_root / args.experiment
    if not experiment_dir.is_dir():
        parser.error(f"unknown experiment: {experiment_dir}")
    log_path, append_resolution = LOGS[args.log]
    try:
        record = append_resolution(
            log_path(experiment_dir),
            report_id=args.report,
            outcome=args.outcome,
            note=args.note,
        )
    except (OSError, ValueError) as exc:
        # An unknown report id or outcome, a second resolution and an
        # unreadable log all land here: say which and write nothing.
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
