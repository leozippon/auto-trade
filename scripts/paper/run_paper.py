#!/usr/bin/env python3
"""ADM-Cube Paper books: create each once, then decide one session per run.

Books live side by side under the state root, one directory per book id.
``init`` pins one experiment's Paper candidate and its research environment
into a new book (id: the experiment id, or ``--book``). ``run`` goes through
every book (or ``--book`` alone) one at a time: it settles each session whose
data has landed and makes the pre-open decision for the target session
(default: today, Asia/Shanghai), then writes the book's order sheet to
``<orders-dir>/<book>/<date>_orders.md`` and ``latest_orders.md`` and prints it.
A book that fails writes its failure to the same two files and the run goes on
to the next book; the run exits non-zero if any book failed.
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
from autotrade.paper.books import (
    list_books,
    run_books,
    validate_book_id,
)
from autotrade.paper.engine import DailyPaperEngine, PaperWriterBusy
from autotrade.paper.orders import (
    render_failure,
    render_orders,
    write_orders,
)
from autotrade.paper.pit import BookPITData
from autotrade.pipelines.calendar import load_sse_trading_days

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATE_ROOT = Path("data/trading/paper")
DEFAULT_ORDERS_DIR = Path("logs/paper")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init", help="Create a Paper book from one experiment's Paper candidate.")
    init.add_argument("--experiment", required=True, help="Experiment id under experiments/.")
    init.add_argument("--artifact", required=True, help="Frozen artifact id: the graduate, or its adjustment.")
    init.add_argument("--initial-cash", type=float, help="Book capital in CNY; default: the experiment's initial_cash.")
    init.add_argument("--note", default="", help="One-line status shown at the top of every order sheet.")
    init.add_argument("--book", help="Book id; default: the experiment id.")
    init.add_argument("--state-root", type=Path, default=DEFAULT_STATE_ROOT)
    run = commands.add_parser("run", help="Settle landed sessions and decide one session, book by book.")
    run.add_argument("--trade-date", help="YYYYMMDD session; default: today in Asia/Shanghai.")
    run.add_argument("--book", help="Run this book only; default: every book.")
    run.add_argument("--state-root", type=Path, default=DEFAULT_STATE_ROOT)
    run.add_argument("--orders-dir", type=Path, default=DEFAULT_ORDERS_DIR)
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
    list_books(args.state_root)  # refuses a root still in the single-book layout
    book = create_book(
        args.state_root / validate_book_id(args.book or args.experiment),
        experiment_dir=REPO_ROOT / "experiments" / args.experiment,
        artifact_id=args.artifact,
        repo_root=REPO_ROOT,
        initial_cash=args.initial_cash,
        note=args.note,
    )
    print(
        f"created Paper book {book.root.name} at {book.root}: {book.experiment_id}/{book.artifact_id} "
        f"({book.candidate_source}), initial cash {book.profile.initial_cash:,.2f}, profile {book.profile.profile_id}, "
        f"commission {book.profile.commission_bps} bps, slippage {book.profile.slippage_bps} bps, "
        f"fit timeout {book.sandbox.limits.fit_timeout_seconds:g}s, image {book.sandbox.image}"
    )
    return 0


def run_book(book_id: str, root: Path, trade_date: str, orders_dir: Path) -> None:
    book = load_book(root)
    if trade_date not in set(load_sse_trading_days(book.raw_dir)):
        print(f"[{book_id}] {trade_date} is not an SSE session; nothing to decide")
        return

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
    sheets = orders_dir / book_id
    try:
        engine.run_day(trade_date)
    except PaperWriterBusy:
        raise  # the running writer owns the sheet
    except Exception as exc:
        text = render_failure(book, trade_date, exc)
        print(text, file=sys.stderr)
        print(f"[{book_id}] failure written to {write_orders(sheets, trade_date, text)}", file=sys.stderr)
        raise
    text = render_orders(book, trade_date)
    print(text)
    print(f"[{book_id}] orders written to {write_orders(sheets, trade_date, text)}", file=sys.stderr)


def run(args: argparse.Namespace) -> int:
    trade_date = args.trade_date or datetime.now(CN_TZ).strftime("%Y%m%d")
    book_ids = [validate_book_id(args.book)] if args.book else list_books(args.state_root)
    if not book_ids:
        print(f"no Paper books under {args.state_root}; create one with `run_paper.py init`", file=sys.stderr)
        return 1
    _resources("before")
    failures = run_books(
        args.state_root, book_ids, lambda book_id, root: run_book(book_id, root, trade_date, args.orders_dir)
    )
    _resources("after")
    for book_id, exc in failures.items():
        print(f"[{book_id}] failed: {type(exc).__name__}: {exc}", file=sys.stderr)
    print(f"{trade_date}: {len(book_ids) - len(failures)}/{len(book_ids)} books ok", file=sys.stderr)
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return {"init": init, "run": run}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
