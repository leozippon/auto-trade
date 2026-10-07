#!/usr/bin/env python3
"""Incubate one node of a finished arm: freeze it, replay it over forward and
Held-out as a recorded reading, and append the arm's one ``incubation`` record.

An arm whose research ended without a graduation (verdict ``no_deliverable``
or ``discarded``) may send one node to Paper, deferring to data nobody has
seen the one question its freeze gate could not settle: whether the node is
the lucky best of many tries. The checks run in order before anything
changes, each a refusal with a non-zero exit: the arm's verdict and no earlier
incubation of the arm, no live worker, the node (and each
``--seed-replicate``) is a Step of the arm's research session, the entry
screen passes on the arm's own pinned rules (``pipelines/incubation.py``), and
fewer than ``INCUBATING_BOOK_CAP`` other incubating Paper books are open
(killed books do not count). GPUs are then taken for an arm that needs them,
the node's bytes are frozen and replayed exactly as a frozen artifact is, on
the boards its book may buy on (``incubation.incubation_boards``), and the
record states the entry reading, the frozen artifact, the boards, the forward
reading and the forward screen on it. When the screen passed, the arm's
incubating Paper book is opened (``books.open_incubating_book``). The arm's
verdict and its Paper candidate are untouched.

The printed JSON says whether the book was ``opened``, ``existed`` or was
``blocked`` by the forward screen, with the forward and Held-out headline
numbers. ``--dry-run`` stops after the entry screen and prints the gate
reading with the deferred and blocking reasons and the boards a real run
would replay on, writing nothing. Run again after the record exists, the
command changes nothing in the arm and only opens a book that is missing.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[1]
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from _bootstrap import add_repo_src

add_repo_src(__file__)

from autotrade.environment.artifacts import FilesystemArtifactStore
from autotrade.environment.identity import AgentRefStore
from autotrade.environment.sandbox import SandboxSpec
from autotrade.environment.sandbox_images import prepare_experiment_sandbox_image
from autotrade.paper.book import require_incubating_place
from autotrade.paper.books import (
    PAPER_STATE_DIR,
    open_incubating_book,
    validate_book_id,
)
from autotrade.pipelines.experiment import incubation_entry
from autotrade.pipelines.hitl_state import assert_no_live_writer, read_json, select_gpus
from autotrade.pipelines.incubation import incubation_boards
from autotrade.pipelines.ledger import (
    ExperimentLedger,
    incubation_record,
    require_incubable,
)
from autotrade.pipelines.worker import build_experiment_pipeline, resolve_worker_options

REPO_ROOT = Path(__file__).resolve().parents[2]


def headline(block: object) -> dict[str, object] | None:
    """One slice's headline numbers, or None for a reading without slices
    (a strategy error). The account, index and zero-skill panel returns are
    cumulative over the slice; the regressed (neutralised) and unregressed
    (plain) active excess against the panel, with their bounds, are per-year
    rates, so a Held-out quarter shows a rate four times its cumulative gap.
    ``seed_mean_plain_excess_per_year`` is what the screen reads when the
    node has seed replicates."""

    if not isinstance(block, dict):
        return None
    raw = block.get("raw_readings") or {}
    seed_mean = block.get("seed_mean")
    return {
        "account_return": raw.get("strategy_return"),
        "index_return": raw.get("benchmark_return"),
        "panel_return": raw.get("panel_return"),
        "neutralized_excess_per_year": block.get("neutralized_excess"),
        "neutralized_excess_per_year_lower_bound": block.get("lower_bound"),
        "plain_excess_per_year": block.get("plain_excess"),
        "plain_excess_per_year_lower_bound": block.get("plain_excess_lower_bound"),
        "seed_mean_plain_excess_per_year": seed_mean.get("plain_excess") if isinstance(seed_mean, dict) else None,
        "information_ratio": block.get("information_ratio"),
        "max_drawdown": block.get("max_drawdown"),
        "active_max_drawdown": block.get("active_max_drawdown"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True, help="Experiment id.")
    parser.add_argument("--step", required=True, help="Step id of the node to incubate.")
    parser.add_argument(
        "--seed-replicate",
        action="append",
        default=[],
        help="Step id of one of the node's seed replicates; repeat for each.",
    )
    parser.add_argument("--by", required=True, help="Who incubates the node.")
    parser.add_argument("--reason", required=True, help="Why, in a sentence or two.")
    parser.add_argument("--experiments-root", type=Path, default=REPO_ROOT / "experiments")
    parser.add_argument("--paper-state-root", type=Path, default=REPO_ROOT / PAPER_STATE_DIR)
    parser.add_argument(
        "--dry-run", action="store_true", help="Stop after the entry screen and print it; change nothing."
    )
    args = parser.parse_args()

    experiment_dir = args.experiments_root / args.experiment
    if not experiment_dir.is_dir():
        parser.error(f"unknown experiment: {experiment_dir}")
    ledger = ExperimentLedger(experiment_dir / "ledgers" / "experiment_ledger.jsonl")
    try:
        book_id = validate_book_id(args.experiment)
        records = ledger.read()
        record = None if args.dry_run else incubation_record(records)
        if record is None:
            require_incubable(records)
            assert_no_live_writer(experiment_dir)
            params_path = experiment_dir / "hitl" / "params.json"
            params = read_json(params_path)
            if not params:
                raise ValueError(f"missing experiment params: {params_path}")
            options = resolve_worker_options(params, experiment_dir=experiment_dir, repo_root=REPO_ROOT)
            reading = incubation_entry(records, args.step, args.seed_replicate, config=options.rolling)
            screen = reading["screen"]
            if args.dry_run:
                print(
                    json.dumps(
                        {
                            "dry_run": True,
                            "experiment_id": args.experiment,
                            "step_id": args.step,
                            "seed_replicates": args.seed_replicate,
                            **screen,
                            "permitted_boards": incubation_boards(options.rolling.broker_profile),
                            "gate": reading["entry"],
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                        default=str,
                    )
                )
                return 0 if screen["passed"] else 2
            if not screen["passed"]:
                raise ValueError(f"the entry screen blocks {args.step}: {', '.join(screen['blocking'])}")
            require_incubating_place(args.paper_state_root, book_id)
            # No console claims cards for this run: take them as _cli does.
            spec = options.agent_sandbox
            if spec is not None and spec.gpu == "auto":
                devices = select_gpus(args.experiments_root, spec.gpu_count, require_name=spec.gpu_name_filter)
                options = replace(options, agent_sandbox=replace(spec, gpu=tuple(devices)))
            if options.execution_mode == "sandbox" or options.developer_mode == "llm":
                options = replace(
                    options,
                    agent_sandbox=prepare_experiment_sandbox_image(
                        options.agent_sandbox or SandboxSpec(gpu=None),
                        experiment_id=options.experiment_id,
                        experiment_dir=options.experiment_dir,
                    ),
                )
            pipeline = build_experiment_pipeline(
                options,
                ledger=ledger,
                store=FilesystemArtifactStore(experiment_dir / "artifacts" / "strategy"),
                ref_store=AgentRefStore(experiment_dir),
            ).pipeline
            record = pipeline.incubate(
                args.step, args.seed_replicate, incubated_by=args.by, reason=args.reason
            )
        elif record["source_step_id"] != args.step:
            raise ValueError(
                f"the arm already incubated {record['source_step_id']}; one incubation per arm"
            )
        # Reached after the append and on a re-run, so a book that failed to
        # open is opened then.
        book = open_incubating_book(args.paper_state_root, REPO_ROOT, experiment_dir)
    except (KeyError, OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    reading = record["forward_reading"]
    slices = reading.get("slices") or {}
    print(
        json.dumps(
            {
                "dry_run": False,
                "run_id": record["run_id"],
                "source_step_id": record["source_step_id"],
                "deferred": record["entry"]["deferred"],
                "frozen": record["frozen"],
                "permitted_boards": record["permitted_boards"],
                "verdict_reasons": reading["verdict"]["reasons"],
                "screen": record["screen"],
                "forward": headline(slices.get("forward")),
                "heldout": headline(slices.get("heldout")),
                "book": book,
                "book_id": book_id,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
