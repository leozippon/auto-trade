#!/usr/bin/env python3
"""ADM-Cube Paper book: create it once, then decide one session per run.

``init`` pins one experiment's Paper candidate and its research environment
into a new book. ``run`` settles every session whose data has landed and makes
the pre-open decision for the target session (default: today, Asia/Shanghai),
then writes the order sheet to ``<orders-dir>/<date>_orders.md`` and
``latest_orders.md`` and prints it. A failed run writes the failure to the same
two files and exits non-zero; nothing is skipped silently.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from _bootstrap import add_repo_src

add_repo_src(__file__)

from autotrade.environment.llm import build_model_gateway
from autotrade.environment.strategy import CN_TZ
from autotrade.paper.book import create_book, load_book
from autotrade.paper.engine import DailyPaperEngine, PaperWriterBusy
from autotrade.paper.orders import render_failure, render_orders, write_orders
from autotrade.paper.pit import BookPITData
from autotrade.pipelines.folds import load_sse_trading_days

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATE_ROOT = Path("data/trading/paper")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init", help="Create a Paper book from one experiment's Paper candidate.")
    init.add_argument("--experiment", required=True, help="Experiment id under experiments/.")
    init.add_argument("--artifact", required=True, help="Frozen artifact id: the graduate, or its adjustment.")
    init.add_argument("--initial-cash", type=float, help="Book capital in CNY; default: the experiment's initial_cash.")
    init.add_argument("--note", default="", help="One-line status shown at the top of every order sheet.")
    init.add_argument("--state-root", type=Path, default=DEFAULT_STATE_ROOT)
    run = commands.add_parser("run", help="Settle landed sessions and decide one session.")
    run.add_argument("--trade-date", help="YYYYMMDD session; default: today in Asia/Shanghai.")
    run.add_argument("--state-root", type=Path, default=DEFAULT_STATE_ROOT)
    run.add_argument("--orders-dir", type=Path, default=Path("logs/paper"))
    return parser


def _resources(label: str) -> None:
    """GPU and host memory readings around a run, into the run's log."""

    print(f"--- resources ({label}) {datetime.now(CN_TZ).isoformat(timespec='seconds')} ---", file=sys.stderr)
    for command in (
        ["nvidia-smi", "--query-gpu=index,memory.used,memory.total", "--format=csv,noheader"],
        ["free", "-h"],
    ):
        try:
            output = subprocess.run(command, capture_output=True, text=True, timeout=30, check=False).stdout
        except (OSError, subprocess.TimeoutExpired) as exc:
            output = f"{command[0]} unavailable: {exc}\n"
        print(output.rstrip(), file=sys.stderr)


def init(args: argparse.Namespace) -> int:
    book = create_book(
        args.state_root,
        experiment_dir=REPO_ROOT / "experiments" / args.experiment,
        artifact_id=args.artifact,
        repo_root=REPO_ROOT,
        initial_cash=args.initial_cash,
        note=args.note,
    )
    print(
        f"created Paper book at {book.root}: {book.experiment_id}/{book.artifact_id} ({book.candidate_source}), "
        f"initial cash {book.profile.initial_cash:,.2f}, profile {book.profile.profile_id}, "
        f"commission {book.profile.commission_bps} bps, slippage {book.profile.slippage_bps} bps, "
        f"fit timeout {book.sandbox.limits.fit_timeout_seconds:g}s, image {book.sandbox.image}"
    )
    return 0


def run(args: argparse.Namespace) -> int:
    book = load_book(args.state_root)
    trade_date = args.trade_date or datetime.now(CN_TZ).strftime("%Y%m%d")
    if trade_date not in set(load_sse_trading_days(book.raw_dir)):
        print(f"{trade_date} is not an SSE session; nothing to decide")
        return 0

    def data_factory(start: str, target: str) -> BookPITData:
        return BookPITData(
            state_root=book.root,
            raw_dir=book.raw_dir,
            fundamental_events_root=book.fundamental_events_root,
            fundamental_events_status=book.fundamental_events_status,
            snapshot_config=book.snapshot_config,
            start=start,
            trade_date=target,
            nl_llm=build_model_gateway(**book.nl_gateway) if book.nl_gateway else None,
            nl_config=book.nl_config,
            nl_failure_policy=book.nl_failure_policy,
            max_intraday_row_group_rows=book.max_intraday_row_group_rows,
        )

    engine = DailyPaperEngine(
        strategy_path=book.strategy_path,
        strategy_revision=book.artifact_id,
        state_root=book.root,
        data_factory=data_factory,
        models_dir=book.models_dir,
        schedule=book.schedule,
        profile=book.profile,
        sandbox=book.sandbox,
    )
    _resources("before")
    try:
        engine.run_day(trade_date)
    except PaperWriterBusy:
        raise  # the running writer owns the sheet
    except Exception as exc:
        text = render_failure(book, trade_date, exc)
        path = write_orders(args.orders_dir, trade_date, text)
        print(text, file=sys.stderr)
        print(f"failure written to {path}", file=sys.stderr)
        _resources("after")
        raise
    text = render_orders(book, trade_date)
    path = write_orders(args.orders_dir, trade_date, text)
    print(text)
    print(f"orders written to {path}", file=sys.stderr)
    _resources("after")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return init(args) if args.command == "init" else run(args)


if __name__ == "__main__":
    raise SystemExit(main())
