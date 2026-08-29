#!/usr/bin/env python3
"""Interactive (HITL) experiment worker entrypoint (docs/pipeline-design.md).

Runs or resumes one experiment's gated Fold/Held-out loop from the parameters in
``experiments/<id>/hitl/params.json``, honouring ``control.json`` (pause / step
approvals / per-session directives / stop) and reporting position and heartbeats
to ``status.json``. Normally spawned detached by the web console
(``scripts/webui/run_webui.py``); can also be launched manually for a headless
HITL run driven purely through the control file.
"""
from __future__ import annotations

import argparse
import os
import signal
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[1]
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from _bootstrap import add_repo_src

REPO_ROOT = add_repo_src(__file__)

from autotrade.environment.runtime import utc_now_iso, write_json_atomic
from autotrade.pipelines.hitl_state import StatusReporter
from autotrade.pipelines.worker import load_worker_options, run_local_interactive_worker


def _terminate(signum, frame):  # noqa: ANN001 - signal handler signature
    raise SystemExit(128 + signum)


def _restore_child_reaping() -> None:
    """Undo the console parent's SIGCHLD=SIG_IGN inheritance for this worker."""
    signal.signal(signal.SIGCHLD, signal.SIG_DFL)


def _exec_self() -> None:
    """Replace this worker with a fresh interpreter running the same command.

    A deferred restart swaps the code without losing the session that was
    running. ``execv`` keeps the pid and its start time (so the console's
    liveness check and ``status.json`` stay valid), the process group created
    by ``start_new_session`` (so terminate/restart still signal the right
    group), the working directory, and the inherited append handle on the
    worker log. Only the program image changes.
    """
    sys.stdout.flush()
    sys.stderr.flush()
    os.execv(sys.executable, [sys.executable, *sys.argv])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--poll-seconds",
        type=float,
        default=2.0,
        help="control.json polling interval while paused or waiting for approval",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    _restore_child_reaping()
    # A graceful SIGTERM unwinds through the pipeline's finally blocks (docker
    # stop) and lets the worker record its terminal state before exiting.
    signal.signal(signal.SIGTERM, _terminate)
    args = build_parser().parse_args(argv)
    experiment_dir = args.experiment_dir.resolve(strict=True)
    status_path = experiment_dir / "hitl" / "status.json"
    bootstrap = StatusReporter(status_path)
    bootstrap.start()
    bootstrap.set(state="preparing", phase="strict_params_and_pit_data")
    try:
        options = load_worker_options(experiment_dir, repo_root=args.repo_root)
        bootstrap.stop()
        result = run_local_interactive_worker(options, poll_seconds=args.poll_seconds)
        if result.get("status") == "restart":
            print(
                f"===== worker restart at session boundary {utc_now_iso()} =====",
                flush=True,
            )
            # Never returns; a failure to exec falls through to the handler
            # below and is recorded as a failed run rather than a silent exit.
            _exec_self()
    except Exception as exc:  # noqa: BLE001 - CLI must persist every terminal failure
        bootstrap.stop()
        write_json_atomic(
            status_path,
            {
                "schema_version": 1,
                "state": "failed",
                "pid": os.getpid(),
                "failed_at": utc_now_iso(),
                "error": f"{type(exc).__name__}: {exc}",
            },
        )
        print(f"interactive experiment failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
