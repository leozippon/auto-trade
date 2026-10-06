#!/usr/bin/env python3
"""Incubate one node of a finished arm: freeze it, replay it over forward and
Held-out as a recorded reading, and append the arm's one ``incubation`` record.

An arm whose research ended without a graduation (verdict ``no_deliverable``
or ``discarded``) may send one node to Paper, deferring to data nobody has
seen the one question its freeze gate could not settle: whether the node is
the lucky best of many tries. The checks run in order before anything
changes, each a refusal with a non-zero exit: the arm's verdict and no earlier
incubation of the arm, no live worker, the node (and each
``--seed-replicate``) is a Step of the arm's research session, and the entry
screen passes on the arm's own pinned rules (``pipelines/incubation.py``).
GPUs are then taken for an arm that needs them, the node's bytes are frozen
and replayed exactly as a frozen artifact is, and the record states the entry
reading, the frozen artifact, the forward reading and the forward screen on
it. The arm's verdict and its Paper candidate are untouched.

``--dry-run`` stops after the entry screen and prints the gate reading with
the deferred and blocking reasons, writing nothing. Run again after the
record exists, the command changes nothing in the arm.
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
from autotrade.pipelines.experiment import incubation_entry
from autotrade.pipelines.hitl_state import assert_no_live_writer, read_json, select_gpus
from autotrade.pipelines.ledger import (
    ExperimentLedger,
    incubation_record,
    require_incubable,
)
from autotrade.pipelines.worker import build_experiment_pipeline, resolve_worker_options

REPO_ROOT = Path(__file__).resolve().parents[2]


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
    parser.add_argument(
        "--dry-run", action="store_true", help="Stop after the entry screen and print it; change nothing."
    )
    args = parser.parse_args()

    experiment_dir = args.experiments_root / args.experiment
    if not experiment_dir.is_dir():
        parser.error(f"unknown experiment: {experiment_dir}")
    ledger = ExperimentLedger(experiment_dir / "ledgers" / "experiment_ledger.jsonl")
    try:
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
            # ---- Integration step: refuse here when the cap on incubating
            # Paper books is reached (design check 6).
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
        # ---- Integration step: open the Paper book here. -----------------
        # When record["screen"]["passed"] and the book is missing: open it
        # from ledger.incubation_candidate(ledger.read()) (track
        # ``incubating``). Reached both after the append above and on a re-run
        # whose record exists, so a book that failed to open is opened then.
    except (KeyError, OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "dry_run": False,
                "run_id": record["run_id"],
                "source_step_id": record["source_step_id"],
                "deferred": record["entry"]["deferred"],
                "frozen": record["frozen"],
                "verdict_reasons": record["forward_reading"]["verdict"]["reasons"],
                "screen": record["screen"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
