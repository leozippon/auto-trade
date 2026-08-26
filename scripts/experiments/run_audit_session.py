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
    QUARTER_DEFAULT_FIRST_TEST_PERIOD,
    QUARTER_DEFAULT_LAST_TEST_PERIOD,
    add_acceptance_arguments,
    add_fold_exploration_directive_arguments,
    add_meta_directive_arguments,
    add_model_arguments,
    add_path_arguments,
    add_schedule_arguments,
    add_snapshot_window_arguments,
    build_worker_options,
    require_generic_period_args,
    resolve_fold_exploration_directive,
    resolve_meta_learning_directive,
)

from autotrade.environment.artifacts import FilesystemArtifactStore
from autotrade.environment.sandbox import DEFAULT_IMAGE
from autotrade.pipelines import (
    ExperimentLedger,
    LocalDailyEvaluationBackend,
    LocalDailySnapshotProvider,
    PITDailyEvaluationBackend,
    ResearchPITSnapshotProvider,
    RollingExperimentPipeline,
    build_fold_schedule,
)
from autotrade.pipelines.local_backend import LLMFoldDeveloper, LLMMetaLearner
from autotrade.pipelines.worker import _parent_from_step_node

QUARTER_DEFAULT_HELDOUT_PERIOD = "2026Q1"


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("meta-learning", "fold"), required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--epoch-id", default="epoch_001")
    parser.add_argument("--fold-index", type=int, default=0, help="0-based Fold index to run; default first Fold.")
    add_path_arguments(parser, repo_root)
    parser.add_argument("--fold-period", choices=("week", "month", "quarter", "year"), default="quarter")
    parser.add_argument("--first-test-period", help="default 2022Q1 for quarter folds; required otherwise")
    parser.add_argument("--last-test-period", help="default 2025Q4 for quarter folds; required otherwise")
    parser.add_argument("--heldout-first-period", help="default 2026Q1 for quarter folds; required otherwise")
    parser.add_argument("--heldout-last-period", help="default 2026Q1 for quarter folds; required otherwise")
    add_schedule_arguments(parser, verbose_help=False)
    add_snapshot_window_arguments(parser, verbose_help=False)
    parser.add_argument("--max-fold-minutes", type=int, default=20)
    add_model_arguments(parser, verbose_help=False)
    parser.add_argument("--local-dev", action="store_true", help="Use the trusted executor; audit default is real Docker.")
    parser.add_argument("--sandbox-image", help="Optional Docker image override for this audit session.")
    parser.add_argument("--no-thinking", action="store_true")
    add_meta_directive_arguments(parser, verbose_help=False)
    add_fold_exploration_directive_arguments(parser, verbose_help=False)
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
    add_acceptance_arguments(parser, verbose_help=False)
    args = parser.parse_args()
    require_generic_period_args(parser, args)
    if args.fold_period == "quarter":
        args.first_test_period = args.first_test_period or QUARTER_DEFAULT_FIRST_TEST_PERIOD
        args.last_test_period = args.last_test_period or QUARTER_DEFAULT_LAST_TEST_PERIOD
        args.heldout_first_period = args.heldout_first_period or QUARTER_DEFAULT_HELDOUT_PERIOD
        args.heldout_last_period = args.heldout_last_period or QUARTER_DEFAULT_HELDOUT_PERIOD
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
    options = build_worker_options(
        args,
        repo_root=repo_root,
        meta_learning_directive=meta_learning_directive,
        fold_exploration_directive=fold_exploration_directive,
        overrides=overrides,
    )

    pipeline, trading_days = _build_pipeline(options)
    folds = build_fold_schedule(
        options.rolling.first_test_period,
        options.rolling.last_test_period,
        trading_days,
        window_months=options.rolling.window_months,
        period=options.rolling.fold_period,
        min_region_trade_days=options.rolling.min_region_trade_days,
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
            "frozen_strategy_artifact_id": outcome.frozen.artifact_id,
            "validation_total_return": _metric(outcome.validation_summary, "total_return"),
            "test_total_return": _metric(outcome.test_summary, "total_return"),
            "experiment_dir": str(options.experiment_dir),
        }
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


def _build_pipeline(options) -> tuple[RollingExperimentPipeline, list[str]]:
    """Assemble the same pipeline the interactive worker builds, for one session.

    The audit entrypoint deliberately reuses the worker's validated options so a
    session run here is configured identically to the same session run by the
    console; only the driving loop differs.
    """
    ledger = ExperimentLedger(options.rolling.ledger_path)
    store = FilesystemArtifactStore(options.experiment_dir / "artifacts" / "strategy")
    if options.llm is None or options.agent_sandbox is None:
        raise SystemExit("audit sessions require the LLM developer configuration")
    fold_gateway = options.llm.build_gateway("main")
    meta_gateway = options.llm.build_gateway("meta")
    nl_gateway = options.llm.build_gateway("nl")
    compact_gateway = options.llm.build_gateway("compact") if options.llm.compact_enabled else None

    if options.data_backend == "pit":
        if options.raw_dir is None or options.fundamental_events_root is None or options.fundamental_events_status is None:
            raise SystemExit("data_backend=pit is missing validated raw/PIT paths")
        snapshots = ResearchPITSnapshotProvider(
            experiment_dir=options.experiment_dir,
            raw_dir=options.raw_dir,
            fundamental_events_root=options.fundamental_events_root,
            fundamental_events_status=options.fundamental_events_status,
            config=options.snapshot_config,
            cache_root=options.pit_cache_root,
        )
        evaluator = PITDailyEvaluationBackend(
            options.experiment_dir / "artifacts" / "results",
            execution_mode=options.execution_mode,
            nl_llm=nl_gateway,
            nl_config=options.nl_config,
            max_intraday_row_group_rows=options.max_intraday_row_group_rows,
        )
        trading_days = snapshots.trading_days
    else:
        if options.daily_path is None:
            raise SystemExit("data_backend=daily requires daily_path")
        snapshots = LocalDailySnapshotProvider(options.daily_path)
        evaluator = LocalDailyEvaluationBackend(
            options.daily_path,
            options.experiment_dir / "artifacts" / "results",
            execution_mode=options.execution_mode,
        )
        trading_days = evaluator.trading_days

    runtime_root = options.work_root / options.experiment_id
    developer = LLMFoldDeveloper(
        llm=fold_gateway,
        explore_llm=nl_gateway,
        compact_llm=compact_gateway,
        context_compaction=options.llm.compaction,
        baseline_strategy=options.baseline_strategy,
        artifact_store=store,
        evaluator=evaluator,
        schedule=options.rolling.schedule,
        broker_profile=options.rolling.broker_profile,
        ledger=ledger,
        experiment_dir=options.experiment_dir,
        runtime_root=runtime_root,
        sandbox_spec=options.agent_sandbox,
        max_response_tokens=options.llm.max_response_tokens,
        step_tree_enabled=options.rolling.step_tree_enabled,
        fold_exploration_directive=options.rolling.fold_exploration_directive,
    )
    meta_learner = LLMMetaLearner(
        llm=meta_gateway,
        compact_llm=compact_gateway,
        context_compaction=options.llm.compaction,
        baseline_strategy=options.baseline_strategy,
        artifact_store=store,
        experiment_dir=options.experiment_dir,
        runtime_root=runtime_root,
        max_llm_calls=options.rolling.max_llm_calls,
        deadline_seconds=options.rolling.max_fold_minutes * 60,
        max_response_tokens=options.llm.max_response_tokens,
        meta_learning_directive=options.rolling.meta_learning_directive,
        fold_exploration_directive=options.rolling.fold_exploration_directive,
    )
    pipeline = RollingExperimentPipeline(
        options.rolling,
        snapshots=snapshots,
        artifacts=store,
        evaluator=evaluator,
        developer=developer,
        meta_learner=meta_learner,
        ledger=ledger,
    )
    return pipeline, trading_days


def _parent_artifact(args: argparse.Namespace, options):
    """Resolve the optional session parent through a validated identity check.

    ``FilesystemArtifactStore.frozen`` re-reads the artifact's own manifest,
    confirms its recorded identity and refuses a tree that is writable or
    contains symlinks, so an audit session can never start from a frozen
    artifact that was edited after it was frozen.
    """
    if args.parent_step_node:
        return _parent_from_step_node(
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
