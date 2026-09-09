#!/usr/bin/env python3
"""Run one audit session: either meta-learning or one ordinary Fold.

This is intentionally narrower than ``run_experiment.py`` and than the
interactive worker. It is for manual process audits where running the full
Epoch/Fold/Held-out pipeline would hide the single session being inspected:
the Prompt, Trace, Sandbox and artifact handoff of exactly one session.

It builds the experiment through the same validated worker configuration the
console uses, then drives one session directly instead of the session loop.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[1]
_HERE = Path(__file__).resolve().parent
for _path in (_SCRIPTS, _HERE):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from _bootstrap import add_repo_src

add_repo_src(__file__)

from _cli import (
    add_acceptance_arguments,
    add_calendar_arguments,
    add_fold_exploration_directive_arguments,
    add_meta_directive_arguments,
    add_model_arguments,
    add_path_arguments,
    add_schedule_arguments,
    add_snapshot_window_arguments,
    build_worker_options,
    resolve_fold_exploration_directive,
    resolve_meta_learning_directive,
    resolve_period_args,
)

from autotrade.environment.artifacts import FilesystemArtifactStore
from autotrade.environment.identity import AgentRefStore
from autotrade.environment.sandbox import DEFAULT_IMAGE
from autotrade.pipelines import ExperimentLedger, build_fold_schedule
from autotrade.pipelines.skills import OPERATING_MEMORY_MODES
from autotrade.pipelines.worker import (
    build_experiment_pipeline,
    parent_from_step_node,
)


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("meta-learning", "fold"), required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--epoch-id", default="epoch_001")
    parser.add_argument("--fold-index", type=int, default=0, help="0-based Fold index to run; default first Fold.")
    add_path_arguments(parser, repo_root)
    add_calendar_arguments(parser)
    add_schedule_arguments(parser)
    add_snapshot_window_arguments(parser)
    parser.add_argument("--max-fold-minutes", type=int, default=20)
    add_model_arguments(parser)
    parser.add_argument("--local-dev", action="store_true", help="Use the trusted executor; audit default is real Docker.")
    parser.add_argument("--sandbox-image", help="Optional Docker image override for this audit session.")
    parser.add_argument("--no-thinking", action="store_true")
    add_meta_directive_arguments(parser)
    add_fold_exploration_directive_arguments(parser)
    parser.add_argument(
        "--workspace-reference",
        help=(
            "Repo-relative reference pack mounted read-only into the session, "
            "the way the console mounts an experiment's refs; none by default."
        ),
    )
    parser.add_argument(
        "--operating-memory",
        choices=OPERATING_MEMORY_MODES,
        help="Operating-memory mode for this session; the console default when omitted.",
    )
    parser.add_argument(
        "--prior-file",
        type=Path,
        help="Optional PRIOR.md seed for Meta or read-only PRIOR text for an ordinary Fold.",
    )
    parser.add_argument(
        "--fold-directive-file",
        type=Path,
        help="Optional UTF-8 researcher directive injected into this ordinary Fold's system prompt.",
    )
    parser.add_argument(
        "--parent-artifact-id",
        help="Optional frozen artifact ID used as this session's parent; validated through the artifact store.",
    )
    parser.add_argument(
        "--parent-artifact-root",
        type=Path,
        help="Artifact store holding --parent-artifact-id; defaults to this experiment's own store.",
    )
    parser.add_argument(
        "--parent-step-node",
        help="Optional validated Step-tree node id used as this session's parent instead of a frozen artifact.",
    )
    parser.add_argument(
        "--skip-image-check",
        action="store_true",
        help="Skip Docker image existence preflight. Useful only with --local-dev or custom Docker handling.",
    )
    add_acceptance_arguments(parser)
    args = parser.parse_args()
    resolve_period_args(parser, args)
    if args.parent_artifact_id and args.parent_step_node:
        parser.error("pass only one of --parent-artifact-id or --parent-step-node")
    if args.parent_artifact_root and not args.parent_artifact_id:
        parser.error("--parent-artifact-root requires --parent-artifact-id")
    if args.mode == "meta-learning" and args.fold_directive_file:
        parser.error("--fold-directive-file is only meaningful with --mode fold")
    image = args.sandbox_image or DEFAULT_IMAGE
    if not args.local_dev and not args.skip_image_check:
        _require_docker_image(image)

    meta_learning_directive = resolve_meta_learning_directive(parser, args)
    fold_exploration_directive = resolve_fold_exploration_directive(parser, args)
    prior_prompt = args.prior_file.read_text(encoding="utf-8") if args.prior_file else ""
    fold_directive = args.fold_directive_file.read_text(encoding="utf-8") if args.fold_directive_file else ""

    # One audited session, never a multi-epoch schedule: an explicit override of
    # the shared builder's defaults rather than a second hand-written config.
    overrides: dict[str, object] = {"epochs": 1}
    if args.sandbox_image:
        overrides["agent_sandbox_image"] = args.sandbox_image
    # What the console mounts into a session and _cli's shared parameter set
    # does not carry; the worker's loader validates both.
    if args.workspace_reference:
        overrides["workspace_reference"] = args.workspace_reference
    if args.operating_memory:
        overrides["operating_memory"] = args.operating_memory
    options = build_worker_options(
        args,
        repo_root=repo_root,
        meta_learning_directive=meta_learning_directive,
        fold_exploration_directive=fold_exploration_directive,
        overrides=overrides,
    )

    # The console's own assembly, so a session audited here is configured
    # exactly like the same session run by the worker. Only the driving loop
    # differs, and with it the two things a single audited session has no place
    # for: the worker's per-experiment derived image (this entrypoint takes
    # ``--sandbox-image`` instead) and its smoke-test command runner.
    pipeline, trading_days, _meta_enabled, _developer_label = (
        build_experiment_pipeline(
            options,
            ledger=ExperimentLedger(options.rolling.ledger_path),
            store=FilesystemArtifactStore(
                options.experiment_dir / "artifacts" / "strategy"
            ),
            ref_store=AgentRefStore(options.experiment_dir),
        )
    )
    folds = build_fold_schedule(
        options.rolling.development_first_period,
        options.rolling.development_last_period,
        trading_days,
        window_months=options.rolling.window_months,
        period=options.rolling.fold_period,
        min_region_trade_days=options.rolling.min_region_trade_days,
        test_stage=options.rolling.test_stage,
        validation_periods=options.rolling.validation_periods,
    )
    if not 0 <= args.fold_index < len(folds):
        raise SystemExit(f"--fold-index {args.fold_index} out of range for {len(folds)} folds")
    fold = folds[args.fold_index]
    parent = _parent_artifact(args, options)

    if args.mode == "meta-learning":
        prior, meta_parent = pipeline.run_meta_session(
            args.epoch_id,
            args.fold_index,
            fold,
            parent=parent,
            previous_prior=prior_prompt,
        )
        result: dict[str, object] = {
            "status": "ok",
            "mode": args.mode,
            "experiment_id": args.experiment_id,
            "epoch_id": args.epoch_id,
            "visible_fold": fold.to_record(),
            "prior_chars": len(prior),
            # The session may regularize the parent; report which artifact the
            # next Fold would start from so an audit run shows the real handoff.
            "next_parent_artifact_id": meta_parent.artifact_id if meta_parent else None,
            "experiment_dir": str(options.experiment_dir),
        }
    else:
        outcome = pipeline.run_fold(
            args.epoch_id,
            fold,
            parent=parent,
            prior=prior_prompt,
            session_context={"directive": fold_directive} if fold_directive else None,
        )
        result = {
            "status": "ok",
            "mode": args.mode,
            "experiment_id": args.experiment_id,
            "epoch_id": args.epoch_id,
            "fold": fold.to_record(),
            "run_id": outcome.run_id,
            "fold_status": outcome.fold_status,
            "frozen_strategy_artifact_id": (
                outcome.frozen.artifact_id if outcome.frozen is not None else None
            ),
            "validation_total_return": _metric(outcome.validation_summary, "total_return"),
            # None for a development Fold without a Test stage.
            "test_total_return": _metric(outcome.test_summary, "total_return"),
            "experiment_dir": str(options.experiment_dir),
        }
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


def _parent_artifact(args: argparse.Namespace, options):
    """Resolve the optional session parent through a validated identity check.

    ``FilesystemArtifactStore.frozen`` re-reads the artifact's own manifest,
    confirms its recorded identity and refuses a tree that is writable or
    contains symlinks, so an audit session can never start from a frozen
    artifact that was edited after it was frozen.
    """
    if args.parent_step_node:
        return parent_from_step_node(
            options.experiment_dir,
            args.parent_step_node,
            f"{args.epoch_id}/audit",
        )
    if not args.parent_artifact_id:
        return None
    root = (
        args.parent_artifact_root.resolve()
        if args.parent_artifact_root
        else options.experiment_dir / "artifacts" / "strategy"
    )
    return FilesystemArtifactStore(root).frozen(args.parent_artifact_id)


def _metric(summary: dict[str, object] | None, key: str) -> object:
    if not isinstance(summary, dict):
        return None
    return summary.get(key)


def _require_docker_image(image: str) -> None:
    result = subprocess.run(
        ["docker", "image", "inspect", image],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if result.returncode != 0:
        raise SystemExit(
            f"missing Docker image {image!r}. Build it first, for example: "
            "docker build -t autotrade-sandbox:latest -f ops/docker/sandbox.Dockerfile ."
        )


if __name__ == "__main__":
    raise SystemExit(main())
