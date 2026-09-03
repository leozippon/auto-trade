#!/usr/bin/env python3
"""PIT event-layer entrypoint (docs/data-documentation.md §3.1).

Subcommands are used by the nightly PIT event job:
build-fundamental-events and audit-fundamental-events.
Experiment orchestration lives in scripts/experiments/run_interactive_experiment.py.
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

from autotrade.environment.data import (
    FUNDAMENTAL_EVENT_DATASETS,
    FundamentalEventsBuilder,
    FundamentalEventsConfig,
    audit_fundamental_events,
    month_aligned_replace_window,
)


def _refuse_manual_production_write(args: argparse.Namespace) -> None:
    """The cron runner is the ONLY production PIT write entrance (fence,
    per-job state, alerting). Only a genuine runner child -- proven by the
    inherited lock fd -- passes; non-production roots (tests, local lakes)
    and read-only audits pass."""
    from autotrade.data_sources.tushare.io import inherited_updater_lock_fd

    if inherited_updater_lock_fd(Path(__file__).resolve().parents[2]) is not None:
        return  # genuine cron-runner child (inherited the held lock fd)
    output_root = getattr(args, "output_root", None)
    if output_root is None:
        return  # audit subcommand: read-only
    repo_root = Path(__file__).resolve().parents[2]
    try:
        if Path(output_root).resolve() != (repo_root / "data" / "pit" / "fundamental_events").resolve():
            return
    except OSError:
        return
    raise SystemExit(
        "direct manual writes to the production PIT root are not supported; run\n"
        "  scripts/data/tushare_cron_update.py --job cn_nightly_pit_event_build "
        "--end-date <YYYYMMDD>   # fence + state + alerts"
    )


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    _refuse_manual_production_write(args)
    try:
        result = args.handler(args)
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"status": "error", "message": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 1
    print(json.dumps({"status": "ok", **result}, ensure_ascii=False, sort_keys=True, default=str))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PIT event-layer commands.")
    sub = parser.add_subparsers(dest="command", required=True)

    fundamental = sub.add_parser("build-fundamental-events", help="build PIT-ready fundamental event partitions")
    fundamental.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    fundamental.add_argument("--output-root", type=Path, default=Path("data/pit/fundamental_events"))
    fundamental.add_argument("--start-date", required=True, help="YYYYMMDD or ISO date.")
    fundamental.add_argument("--end-date", required=True, help="YYYYMMDD or ISO date.")
    fundamental.add_argument(
        "--dataset", action="append", choices=FUNDAMENTAL_EVENT_DATASETS, help="Dataset to include; repeatable."
    )
    fundamental.set_defaults(handler=run_build_fundamental_events)

    fundamental_audit = sub.add_parser("audit-fundamental-events", help="audit PIT-ready fundamental event partitions")
    fundamental_audit.add_argument("--events-root", type=Path, default=Path("data/pit/fundamental_events"))
    fundamental_audit.add_argument("--start-date", required=True, help="YYYYMMDD or ISO date.")
    fundamental_audit.add_argument("--end-date", required=True, help="YYYYMMDD or ISO date.")
    fundamental_audit.add_argument(
        "--dataset", action="append", choices=FUNDAMENTAL_EVENT_DATASETS, help="Dataset to include; repeatable."
    )
    fundamental_audit.add_argument("--output", type=Path, default=Path("results/data_quality/fundamental_events_status.json"))
    fundamental_audit.add_argument(
        "--require-partitions", action="store_true", help="Fail the audit when no PIT event rows exist in the window."
    )
    fundamental_audit.set_defaults(handler=run_audit_fundamental_events)
    return parser


def run_build_fundamental_events(args: argparse.Namespace) -> dict[str, object]:
    builder = FundamentalEventsBuilder(args.raw_dir)
    start_date, replace_months = month_aligned_replace_window(args.start_date, args.end_date)
    events = builder.build(
        FundamentalEventsConfig(
            start_date=start_date,
            end_date=args.end_date,
            datasets=tuple(args.dataset or FUNDAMENTAL_EVENT_DATASETS),
        )
    )
    if events.empty:
        # An all-empty build over a real window means the raw fundamentals are
        # unreadable or gone; replacing every partition with markers would
        # destroy the whole derived layer while reporting success.
        raise RuntimeError(
            f"raw fundamentals produced no PIT events for {start_date}..{args.end_date}; "
            "refusing to replace the window"
        )
    written = builder.write_partitioned(
        events,
        args.output_root,
        replace_months=replace_months,
        replace_datasets=tuple(args.dataset or FUNDAMENTAL_EVENT_DATASETS),
    )
    return {
        "rows": int(len(events)),
        "partitions": len(written),
        "output_dir": str(args.output_root),
        "first_partition": str(written[0]) if written else None,
        "last_partition": str(written[-1]) if written else None,
    }


def run_audit_fundamental_events(args: argparse.Namespace) -> dict[str, object]:
    report = audit_fundamental_events(
        args.events_root,
        FundamentalEventsConfig(
            start_date=args.start_date,
            end_date=args.end_date,
            datasets=tuple(args.dataset or FUNDAMENTAL_EVENT_DATASETS),
        ),
        output=args.output,
        require_partitions=getattr(args, "require_partitions", False),
    )
    counts = report["finding_counts"]
    if report["status"] == "error":
        raise ValueError(f"fundamental event audit failed: errors={counts['error']} output={args.output}")
    return {
        "audit_status": report["status"],
        "errors": counts["error"],
        "warnings": counts["warning"],
        "rows": report["metadata"]["rows"],
        "output": str(args.output),
    }


if __name__ == "__main__":
    raise SystemExit(main())
