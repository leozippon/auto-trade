#!/usr/bin/env python3
"""ADM-Cube Paper books: create each once, then decide one session per run.

Books live side by side under the state root, one directory per book id.
``init`` pins one experiment's Paper candidate, its research environment and
the artifact's out-of-sample curve into a new book (id: the experiment id, or
``--book``); ``source-history`` backfills that curve into a book created before
init copied it. ``run`` goes through
every book (or ``--book`` alone) one at a time, the books that follow real
fills -- the ones the owner trades by hand -- first: it settles each session
whose data has landed and makes the pre-open decision for the target session
(default: today, Asia/Shanghai), then writes the book's order sheet to
``<orders-dir>/<book>/<date>_orders.md`` and ``latest_orders.md`` and prints it.
A book that fails writes its failure to the same two files and the run goes on
to the next book; the run exits non-zero if any book failed. Once a book's
sheet is written the run reads its Paper verdict (``paper/verdict.py``); a
reading that fails fails the book and leaves the sheet as it is. A killed book
decides nothing: its sheet lists the holdings to exit by hand, and it is
skipped, not failed. ``verdict`` reads every book's verdict (or ``--book``'s)
and prints it as JSON.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path

SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from _bootstrap import add_repo_src

add_repo_src(__file__)

from autotrade.environment.executor import StrategyExecutor
from autotrade.environment.llm import build_model_gateway
from autotrade.environment.replay.engine import StrategyDataView
from autotrade.environment.sandbox import SandboxConfig
from autotrade.environment.strategy import CN_TZ
from autotrade.paper.book import (
    SOURCE_HISTORY_NAME,
    VERDICT_LOG_NAME,
    book_status,
    copy_source_history,
    create_book,
    load_book,
    write_source_history,
)
from autotrade.paper.books import (
    PAPER_STATE_DIR,
    list_books,
    run_books,
    run_order,
    validate_book_id,
)
from autotrade.paper.engine import DailyPaperEngine, PaperWriterBusy, docker_executor
from autotrade.paper.orders import (
    render_failure,
    render_killed,
    render_orders,
    write_orders,
)
from autotrade.paper.pit import BookPITData
from autotrade.paper.storage import read_jsonl
from autotrade.paper.verdict import record_verdict
from autotrade.pipelines.calendar import load_sse_trading_days
from autotrade.pipelines.hitl_state import select_gpus

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATE_ROOT = PAPER_STATE_DIR
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
    history = commands.add_parser(
        "source-history",
        help="Copy an out-of-sample replay curve into a book created before init copied one.",
    )
    history.add_argument("--book", required=True, help="Book id under the state root.")
    picked = history.add_mutually_exclusive_group(required=True)
    picked.add_argument(
        "--experiment",
        help="Source experiment directory, wherever it now lives; its forward record names the replay.",
    )
    picked.add_argument(
        "--result",
        help="Replay result directory, for a source experiment whose ledger predates the forward record.",
    )
    history.add_argument("--heldout-start", help="YYYYMMDD the Held-out slice of --result opens on, if any.")
    history.add_argument("--state-root", type=Path, default=DEFAULT_STATE_ROOT)
    run = commands.add_parser("run", help="Settle landed sessions and decide one session, book by book.")
    run.add_argument("--trade-date", help="YYYYMMDD session; default: today in Asia/Shanghai.")
    run.add_argument("--book", help="Run this book only; default: every book.")
    run.add_argument("--state-root", type=Path, default=DEFAULT_STATE_ROOT)
    run.add_argument("--orders-dir", type=Path, default=DEFAULT_ORDERS_DIR)
    verdict = commands.add_parser("verdict", help="Read and record each book's Paper verdict; print it as JSON.")
    verdict.add_argument("--book", help="This book only; default: every book.")
    verdict.add_argument("--state-root", type=Path, default=DEFAULT_STATE_ROOT)
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
        track="graduated",
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


def source_history(args: argparse.Namespace) -> int:
    """Backfill one book created before init copied the curve.

    A current experiment names its own replay through the forward record; an
    arm retired under an older ledger has no such record, so the operator names
    the replay result directory instead.
    """

    root = args.state_root / validate_book_id(args.book)
    book = load_book(root)
    if (root / SOURCE_HISTORY_NAME).exists():
        raise FileExistsError(f"{book.root.name} already has {SOURCE_HISTORY_NAME}; a book copies it once")
    if args.experiment:
        experiment = Path(args.experiment).resolve(strict=True)
        if experiment.name != book.experiment_id:
            raise ValueError(
                f"{experiment} is not the source of {book.root.name}: the book names {book.experiment_id}"
            )
        if book.candidate_source != "graduated":
            raise ValueError(
                f"{book.root.name} is {book.candidate_source}: its history is the replay its incubation "
                "recorded, not the forward record's; name that replay with --result"
            )
        written = copy_source_history(root, experiment)
    else:
        written = write_source_history(
            root,
            experiment_id=book.experiment_id,
            result_file=Path(args.result).resolve(strict=True),
            heldout_start=args.heldout_start,
        )
    print(f"wrote {written}")
    return 0


def _executor_on_free_gpus(
    strategy_path: Path,
    sandbox: SandboxConfig,
    view: StrategyDataView,
    state_dir: Path | None,
    models_dir: Path | None,
) -> StrategyExecutor:
    """A book's strategy container, on cards taken as it starts.

    A book holds no card between its runs, so a run that has a decision to
    make takes the ones its book asks for from the one selection
    (``hitl_state.select_gpus``): whole cards nobody uses and no running
    research arm claims. With too few free the book cannot decide, and its
    sheet says why.
    """

    limits = sandbox.limits
    if limits.gpu_count > 0:
        devices = select_gpus(
            REPO_ROOT / "experiments", limits.gpu_count, require_name=limits.gpu_name_filter
        )
        sandbox = replace(sandbox, limits=replace(limits, gpu_devices=tuple(devices)))
    return docker_executor(strategy_path, sandbox, view, state_dir, models_dir)


def run_book(book_id: str, root: Path, trade_date: str, orders_dir: Path) -> None:
    book = load_book(root)
    if trade_date not in set(load_sse_trading_days(book.raw_dir)):
        print(f"[{book_id}] {trade_date} is not an SSE session; nothing to decide")
        return
    sheets = orders_dir / book_id
    if book_status(root) == "killed":
        transitions, _skipped = read_jsonl(root / VERDICT_LOG_NAME)
        text = render_killed(book, trade_date, transitions[0])
        print(text)
        print(f"[{book_id}] killed; nothing decided, exit sheet written to {write_orders(sheets, trade_date, text)}", file=sys.stderr)
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
        executor_factory=_executor_on_free_gpus,
    )
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
    # After the sheet, so the reading never delays it; a failure fails the book.
    reading = record_verdict(book)
    print(
        f"[{book_id}] verdict: {reading['status']} after {reading['days']} Paper sessions, "
        f"next checkpoint at {reading['next_checkpoint']}",
        file=sys.stderr,
    )


def verdict(args: argparse.Namespace) -> int:
    book_ids = [validate_book_id(args.book)] if args.book else list_books(args.state_root)
    readings: dict[str, object] = {}
    failures = run_books(
        args.state_root,
        book_ids,
        lambda book_id, root: readings.__setitem__(book_id, record_verdict(load_book(root))),
    )
    for book_id, exc in failures.items():
        readings[book_id] = {"error": f"{type(exc).__name__}: {exc}"}
    print(json.dumps(readings, ensure_ascii=False, indent=2, sort_keys=True))
    return 1 if failures else 0


def run(args: argparse.Namespace) -> int:
    trade_date = args.trade_date or datetime.now(CN_TZ).strftime("%Y%m%d")
    book_ids = [validate_book_id(args.book)] if args.book else run_order(args.state_root)
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
    return {"init": init, "run": run, "source-history": source_history, "verdict": verdict}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
