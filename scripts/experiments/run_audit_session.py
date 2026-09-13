#!/usr/bin/env python3
"""Run one research session of an experiment for a process audit.

This is intentionally narrower than the interactive worker. It is for manual
process audits where running the whole arm would hide the single session being
inspected: the Prompt, Trace, Sandbox and artifact handoff of exactly one
session.

It builds the experiment through the same validated worker configuration the
console uses, then drives one session directly instead of the session loop.
The session reads its start node and PRIOR from the experiment's own ledger,
so ``--session-index`` must be the next session the ledger is waiting for.
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
    add_model_arguments,
    add_path_arguments,
    add_schedule_arguments,
    add_snapshot_window_arguments,
    build_worker_options,
    resolve_fold_exploration_directive,
)

from autotrade.environment.artifacts import FilesystemArtifactStore
from autotrade.environment.identity import AgentRefStore
from autotrade.environment.sandbox import DEFAULT_IMAGE
from autotrade.pipelines import ExperimentLedger
from autotrade.pipelines.skills import OPERATING_MEMORY_MODES
from autotrade.pipelines.worker import build_experiment_pipeline


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument(
        "--session-index",
        type=int,
        default=1,
        help="1-based research session to run; the next one the ledger expects.",
    )
    add_path_arguments(parser, repo_root)
    add_calendar_arguments(parser)
    add_schedule_arguments(parser)
    add_snapshot_window_arguments(parser)
    parser.add_argument("--max-fold-minutes", type=int, default=20)
    add_model_arguments(parser)
    parser.add_argument("--local-dev", action="store_true", help="Use the trusted executor; audit default is real Docker.")
    parser.add_argument("--sandbox-image", help="Optional Docker image override for this audit session.")
    parser.add_argument("--no-thinking", action="store_true")
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
        "--directive-file",
        type=Path,
        help="Optional UTF-8 researcher directive injected into this session's system prompt.",
    )
    parser.add_argument(
        "--skip-image-check",
        action="store_true",
        help="Skip Docker image existence preflight. Useful only with --local-dev or custom Docker handling.",
    )
    add_acceptance_arguments(parser)
    args = parser.parse_args()
    image = args.sandbox_image or DEFAULT_IMAGE
    if not args.local_dev and not args.skip_image_check:
        _require_docker_image(image)

    fold_exploration_directive = resolve_fold_exploration_directive(parser, args)
    directive = args.directive_file.read_text(encoding="utf-8") if args.directive_file else ""

    overrides: dict[str, object] = {}
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
        fold_exploration_directive=fold_exploration_directive,
        overrides=overrides,
    )

    # The console's own assembly, so a session audited here is configured
    # exactly like the same session run by the worker. Only the driving loop
    # differs, and with it the two things a single audited session has no place
    # for: the worker's per-experiment image (this entrypoint takes
    # ``--sandbox-image`` instead) and its smoke-test command runner.
    pipeline, _trading_days, _developer_label = build_experiment_pipeline(
        options,
        ledger=ExperimentLedger(options.rolling.ledger_path),
        store=FilesystemArtifactStore(options.experiment_dir / "artifacts" / "strategy"),
        ref_store=AgentRefStore(options.experiment_dir),
    )
    record = pipeline.run_research_session(
        args.session_index,
        session_context={"directive": directive} if directive else None,
    )
    frozen = record.get("frozen")
    result = {
        "status": "ok",
        "experiment_id": args.experiment_id,
        "session_id": record["session_id"],
        "run_id": record["run_id"],
        "outcome": record["outcome"],
        "validations": len(record.get("steps") or ()),
        "frozen_strategy_artifact_id": (
            frozen.get("artifact_id") if isinstance(frozen, dict) else None
        ),
        "experiment_dir": str(options.experiment_dir),
    }
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


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
