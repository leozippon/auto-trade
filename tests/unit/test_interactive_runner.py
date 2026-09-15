"""Orchestration contract of the HITL runner itself (docs/pipeline-design.md).

`tests/unit/test_research_arm_worker.py` drives the worker end to end, so the
runner's own control-plane branches -- durable stop/pause at a session
boundary, retries, resume, and which planned session is due -- are exercised
here directly against a recording executor so each branch is observed,
including the ones that must NOT fire.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from autotrade.pipelines import interactive
from autotrade.pipelines.hitl_state import (
    ControlState,
    PlannedSession,
    planned_sessions,
    read_control,
    read_status,
    write_control,
)
from autotrade.pipelines.interactive import ExperimentStopped, InteractiveExperimentRunner
from autotrade.pipelines.ledger import ExperimentLedger, FrozenArtifactMutated


# The plan of record of every arm: the research session, then the forward replay.
PLAN = planned_sessions()


class RecordingExecutor:
    """Appends the session's canonical ledger record, like the real pipeline.

    ``outcomes`` maps a session key to the research record's extra fields
    (``frozen`` or ``arm_end``); without them the research record ends nothing,
    which the real pipeline never writes, so most tests name one.
    """

    def __init__(self, ledger: ExperimentLedger, *, record: bool = True, outcomes=None) -> None:
        self.ledger = ledger
        self.record = record
        self.outcomes = dict(outcomes or {})
        self.calls: list[tuple[str, dict[str, object]]] = []
        self.on_call = None

    def __call__(self, session: PlannedSession, context: dict[str, object]):
        self.calls.append((session.session_key, dict(context)))
        if self.on_call is not None:
            self.on_call(session, context)
        if self.record:
            self.ledger.append(
                {
                    "record_type": "research_session" if session.kind == "research" else "forward",
                    "experiment_id": "exp",
                    "epoch_id": session.kind,
                    "fold_id": session.session_key,
                    "run_id": f"run_{len(self.calls)}",
                    "session_key": session.session_key,
                    **(
                        {"verdict": {"status": "discarded", "reasons": ["x"]}}
                        if session.kind == "forward"
                        else {}
                    ),
                    **self.outcomes.get(session.session_key, {}),
                }
            )
        return None

    @property
    def keys(self) -> list[str]:
        return [key for key, _context in self.calls]


class RunnerTestCase(unittest.TestCase):
    """Shared temp experiment: control/status files plus a real ledger."""

    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.hitl = self.root / "hitl"
        self.hitl.mkdir(parents=True)
        self.control = self.hitl / "control.json"
        self.status = self.hitl / "status.json"
        self.ledger = ExperimentLedger(self.root / "ledger.jsonl")
        write_control(self.control, ControlState(mode="auto"))

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def runner(self, sessions, executor, **kwargs) -> InteractiveExperimentRunner:
        options = {
            "experiment_id": "exp",
            "sessions": sessions,
            "execute_session": executor,
            "ledger": self.ledger,
            "control_path": self.control,
            "status_path": self.status,
            "poll_seconds": 0.01,
        }
        options.update(kwargs)
        return InteractiveExperimentRunner(**options)

    def set_control(self, **values: object) -> None:
        state = read_control(self.control)
        for key, value in values.items():
            setattr(state, key, value)
        write_control(self.control, state)


FROZEN = {"frozen": {"artifact_id": "strategy_research_x", "output_path": "unused"}}
NO_EDGE = {"arm_end": {"status": "no_deliverable", "reason": "no_edge: nothing"}}


class InteractiveRunnerTest(RunnerTestCase):
    def test_the_research_session_runs_once_and_the_forward_replay_follows_a_freeze(self) -> None:
        executor = RecordingExecutor(self.ledger, outcomes={"research": FROZEN})
        result = self.runner(PLAN, executor).run()
        self.assertEqual(result, {"status": "complete", "sessions_run": 2})
        self.assertEqual(executor.keys, ["research", "forward"])
        status = read_status(self.status)
        self.assertEqual(status["completed_sessions"], 2)
        self.assertEqual(status["total_sessions"], 2)

    def test_research_that_ends_without_a_freeze_runs_no_forward_replay(self) -> None:
        executor = RecordingExecutor(self.ledger, outcomes={"research": NO_EDGE})
        result = self.runner(PLAN, executor).run()
        self.assertEqual(result["status"], "complete")
        self.assertEqual(executor.keys, ["research"])

    def test_resume_after_a_freeze_runs_only_the_forward_replay(self) -> None:
        self.runner(PLAN, RecordingExecutor(self.ledger, outcomes={"research": FROZEN}, record=True)).run()
        rows = self.ledger.read()
        # Drop the forward record: the worker stopped between freeze and replay.
        self.ledger.path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows if row["record_type"] != "forward"),
            encoding="utf-8",
        )
        again = RecordingExecutor(self.ledger)
        self.runner(PLAN, again).run()
        self.assertEqual(again.keys, ["forward"])

    def test_a_session_is_retried_after_a_transient_failure(self) -> None:
        class Flaky(RecordingExecutor):
            def __call__(self, session, context):
                if len(self.calls) < 2:
                    self.calls.append((session.session_key, dict(context)))
                    raise RuntimeError(f"boom{len(self.calls)}")
                return super().__call__(session, context)

        executor = Flaky(self.ledger, outcomes={"research": NO_EDGE})
        result = self.runner(PLAN, executor, session_max_attempts=3).run()
        self.assertEqual(result["status"], "complete")
        self.assertEqual(len(executor.calls), 3)
        self.assertEqual(read_status(self.status)["state"], "running_session")

    def test_a_successful_retry_clears_the_recorded_attempt_error(self) -> None:
        seen: list[object] = []
        status_path = self.status

        class Flaky(RecordingExecutor):
            def __call__(self, session, context):
                seen.append(read_status(status_path).get("error"))
                if len(self.calls) < 1:
                    self.calls.append((session.session_key, dict(context)))
                    raise RuntimeError("boom")
                return super().__call__(session, context)

        result = self.runner(
            PLAN, Flaky(self.ledger, outcomes={"research": FROZEN}), session_max_attempts=3
        ).run()
        self.assertEqual(result["status"], "complete")
        # The failure stays visible while its retry runs; the forward replay
        # starts without the research attempt's stale text.
        self.assertEqual(seen, [None, "RuntimeError: boom (attempt 1/3)", None])
        self.assertIsNone(read_status(self.status)["error"])

    def test_each_attempt_refreshes_session_timing(self) -> None:
        started: list[str] = []
        walls: list[float] = []
        stages: list[object] = []
        run_ids: list[object] = []
        status_path = self.status

        class Flaky(RecordingExecutor):
            def __call__(self, session, context):
                payload = read_status(status_path)
                started.append(str(payload.get("session_started_at") or ""))
                stages.append(payload.get("environment_stage"))
                run_ids.append(payload.get("run_id"))
                timing_fn = context["session_timing"]
                timing = timing_fn() if callable(timing_fn) else {}
                wall = timing.get("run_wall_seconds") if isinstance(timing, dict) else 0.0
                walls.append(float(wall or 0.0))
                if len(self.calls) < 1:
                    self.calls.append((session.session_key, dict(context)))
                    raise RuntimeError("boom")
                return super().__call__(session, context)

        executor = Flaky(self.ledger, outcomes={"research": NO_EDGE})
        self.runner(PLAN, executor, session_max_attempts=3).run()
        self.assertEqual(len(started), 2)
        self.assertTrue(started[0])
        self.assertTrue(started[1])
        self.assertNotEqual(started[0], started[1])
        self.assertEqual(stages, ["preparing_session", "preparing_session"])
        self.assertEqual(run_ids, [None, None])
        self.assertGreaterEqual(walls[1], 0.0)

    def test_frozen_artifact_mutation_is_not_retried(self) -> None:
        class Mutating(RecordingExecutor):
            def __call__(self, session, context):
                self.calls.append((session.session_key, dict(context)))
                self.ledger.append(
                    {
                        "record_type": "forward",
                        "experiment_id": "exp",
                        "epoch_id": "forward",
                        "fold_id": "forward",
                        "run_id": f"run_{len(self.calls)}",
                        "session_key": session.session_key,
                        "state_changed_during_test": True,
                    }
                )
                raise FrozenArtifactMutated(
                    "strategy or model artifacts changed during the forward replay"
                )

        executor = Mutating(self.ledger, record=False)
        with self.assertRaises(FrozenArtifactMutated):
            self.runner(PLAN[:1], executor, session_max_attempts=3).run()
        self.assertEqual(len(executor.calls), 1)
        self.assertEqual(len(self.ledger.read("forward")), 1)
        self.assertEqual(read_status(self.status)["state"], "failed")

    def test_resume_refuses_a_flagged_integrity_record_without_new_rows(self) -> None:
        self.ledger.append(
            {
                "record_type": "forward",
                "experiment_id": "exp",
                "epoch_id": "forward",
                "fold_id": "forward",
                "run_id": "run_flagged",
                "session_key": "forward",
                "state_changed_during_test": True,
            }
        )
        before = [row.get("run_id") for row in self.ledger.read()]
        executor = RecordingExecutor(self.ledger)
        with self.assertRaises(FrozenArtifactMutated):
            self.runner(PLAN, executor).run()
        self.assertEqual(executor.keys, [])
        self.assertEqual([row.get("run_id") for row in self.ledger.read()], before)
        self.assertEqual(read_status(self.status)["state"], "failed")

    def test_a_session_fails_the_experiment_after_the_attempt_budget(self) -> None:
        class AlwaysFail(RecordingExecutor):
            def __call__(self, session, context):
                self.calls.append((session.session_key, dict(context)))
                raise RuntimeError("still broken")

        executor = AlwaysFail(self.ledger)
        with self.assertRaisesRegex(RuntimeError, "still broken"):
            self.runner(PLAN, executor, session_max_attempts=3).run()
        self.assertEqual(len(executor.calls), 3)
        self.assertEqual(read_status(self.status)["state"], "failed")
        # The final failure keeps its text, without an attempt suffix.
        self.assertEqual(read_status(self.status)["error"], "RuntimeError: still broken")

    def test_a_positive_poll_interval_is_required(self) -> None:
        for bad in (0, -1.0):
            with self.subTest(poll_seconds=bad), self.assertRaisesRegex(ValueError, "poll_seconds"):
                self.runner(PLAN, RecordingExecutor(self.ledger), poll_seconds=bad)

    def test_stop_requested_during_a_session_halts_at_the_next_boundary(self) -> None:
        executor = RecordingExecutor(self.ledger, outcomes={"research": FROZEN})
        executor.on_call = lambda _session, _context: self.set_control(request="stop")
        result = self.runner(PLAN, executor).run()
        # The session in flight finishes; the forward replay never starts.
        self.assertEqual(result, {"status": "stop", "sessions_run": 1})
        self.assertEqual(executor.keys, ["research"])
        self.assertEqual(read_status(self.status)["state"], "stopped")

    def test_pause_requested_during_a_session_halts_and_reports_paused(self) -> None:
        executor = RecordingExecutor(self.ledger, outcomes={"research": FROZEN})
        executor.on_call = lambda _session, _context: self.set_control(request="pause")
        result = self.runner(PLAN, executor).run()
        self.assertEqual(result["status"], "pause")
        self.assertEqual(result["sessions_run"], 1)
        self.assertEqual(read_status(self.status)["state"], "paused")

    def test_a_stop_pending_at_the_gate_raises_before_any_session_runs(self) -> None:
        self.set_control(request="stop")
        executor = RecordingExecutor(self.ledger)
        with self.assertRaisesRegex(ExperimentStopped, "stop requested"):
            self.runner(PLAN, executor).run()
        self.assertEqual(executor.keys, [])
        status = read_status(self.status)
        self.assertEqual(status["state"], "failed")
        self.assertIn("ExperimentStopped", status["error"])

    def test_session_directives_and_overrides_reach_the_session_then_are_consumed(self) -> None:
        executor = RecordingExecutor(self.ledger, outcomes={"research": NO_EDGE})
        self.set_control(
            directives={"research": "try momentum"},
            resource_overrides={"research": {"max_replay_years": 2}},
        )
        self.runner(PLAN, executor).run()
        _key, context = executor.calls[0]
        self.assertEqual(context["directive"], "try momentum")
        self.assertEqual(context["resource_override"], {"max_replay_years": 2})
        self.assertEqual(context["session_key"], "research")
        for hook in ("progress_hook", "session_timing"):
            self.assertTrue(callable(context[hook]), hook)
        control = read_control(self.control)
        self.assertEqual((control.directives, control.resource_overrides), ({}, {}))

    def test_a_session_gpu_count_reaches_only_its_own_session_and_is_consumed(self) -> None:
        """`set_gpu_count` is a one-shot allocation, like an approval.

        The console writes it against one session key; the runner must hand it
        to that session alone and clear it afterwards, or the forward replay
        would silently inherit an allocation nobody asked for.
        """
        executor = RecordingExecutor(self.ledger, outcomes={"research": FROZEN})
        self.set_control(gpu_counts={"research": 3})
        self.runner(PLAN, executor).run()
        self.assertEqual(
            {key: context["sandbox_gpu_count"] for key, context in executor.calls},
            {"research": 3, "forward": None},
        )
        self.assertEqual(read_control(self.control).gpu_counts, {})

    def test_a_session_that_records_nothing_durable_fails_fast(self) -> None:
        executor = RecordingExecutor(self.ledger, record=False)
        with self.assertRaisesRegex(RuntimeError, "without a durable success record"):
            self.runner(PLAN, executor).run()
        self.assertEqual(read_status(self.status)["state"], "failed")

    def test_resume_skips_the_session_already_recorded_in_the_ledger(self) -> None:
        first = RecordingExecutor(self.ledger, outcomes={"research": FROZEN})
        first.on_call = lambda session, _context: (
            self.set_control(request="stop") if session.session_key == "research" else None
        )
        self.runner(PLAN, first).run()
        self.assertEqual(first.keys, ["research"])
        self.set_control(request=None)
        second = RecordingExecutor(self.ledger)
        result = self.runner(PLAN, second).run()
        self.assertEqual(second.keys, ["forward"])
        self.assertEqual(result["sessions_run"], 1)
        self.assertEqual(read_status(self.status)["completed_sessions"], 2)

    def test_no_session_gate_holds_a_worker_the_researcher_started(self) -> None:
        """The session gate only checks stop/seal/restart; nothing blocks.

        There is no per-session approval any more: a worker the researcher
        started runs its planned sessions, and holding it is `pause`/`stop`.
        """
        with patch.object(
            interactive.time, "sleep", side_effect=AssertionError("gate slept")
        ):
            self.runner(PLAN, RecordingExecutor(self.ledger, outcomes={"research": NO_EDGE})).run()

class WorkerEntrypointTest(unittest.TestCase):
    def test_the_worker_restores_child_reaping_the_console_disabled(self) -> None:
        """The console sets SIGCHLD=SIG_IGN; a worker inheriting it would get
        -1 from every subprocess.run(), silently breaking docker exit codes."""
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
        from experiments import run_interactive_experiment

        previous = signal.getsignal(signal.SIGCHLD)
        try:
            signal.signal(signal.SIGCHLD, signal.SIG_IGN)
            # With reaping ignored the exit status is lost (-1 on Linux).
            ignored = subprocess.run([sys.executable, "-c", "raise SystemExit(7)"], check=False)
            self.assertNotEqual(ignored.returncode, 7)
            run_interactive_experiment._restore_child_reaping()
            restored = subprocess.run([sys.executable, "-c", "raise SystemExit(7)"], check=False)
            self.assertEqual(restored.returncode, 7)
        finally:
            signal.signal(signal.SIGCHLD, previous)

    def test_the_entrypoint_stamps_a_terminal_state_when_sigterm_unwinds_it(
        self,
    ) -> None:
        """A graceful terminate must not leave a live state behind a dead pid.

        The console stamps a terminal state itself only when it has to escalate
        to SIGKILL. A worker that honours SIGTERM inside the grace window
        unwinds through the handler's `SystemExit`, which is not an `Exception`:
        while this entrypoint caught only `Exception`, `status.json` kept
        `running_session` and its dead pid for good, and the console showed a
        finished arm as live.
        """
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
        from experiments import run_interactive_experiment

        previous = signal.getsignal(signal.SIGTERM)
        with TemporaryDirectory() as tmp:
            experiment_dir = Path(tmp) / "exp"
            (experiment_dir / "hitl").mkdir(parents=True)
            status_path = experiment_dir / "hitl/status.json"

            def live_session(_options, **_kwargs):
                # What the runner leaves on disk while a session is running.
                status_path.write_text(
                    json.dumps(
                        {
                            "schema_version": 1,
                            "state": "running_session",
                            "pid": os.getpid(),
                            "session_key": "research",
                            "completed_sessions": 17,
                        }
                    ),
                    encoding="utf-8",
                )
                os.kill(os.getpid(), signal.SIGTERM)
                time.sleep(5)  # the handler unwinds this sleep
                raise AssertionError("SIGTERM was never delivered")

            try:
                with patch.object(
                    run_interactive_experiment, "load_worker_options"
                ), patch.object(
                    run_interactive_experiment,
                    "run_local_interactive_worker",
                    live_session,
                ):
                    code = run_interactive_experiment.main(
                        ["--experiment-dir", str(experiment_dir)]
                    )
            finally:
                signal.signal(signal.SIGTERM, previous)
            self.assertEqual(code, 143)
            status = json.loads(status_path.read_text(encoding="utf-8"))
            self.assertEqual(status["state"], "terminated")
            self.assertIsNone(status["error"])
            self.assertTrue(status["terminated_at"])
            # Where the run stopped survives, as on the escalated path.
            self.assertEqual(status["session_key"], "research")
            self.assertEqual(status["completed_sessions"], 17)

    def test_the_entrypoint_persists_a_terminal_failure_status(self) -> None:
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
        from experiments import run_interactive_experiment

        with TemporaryDirectory() as tmp:
            experiment_dir = Path(tmp) / "exp"
            (experiment_dir / "hitl").mkdir(parents=True)
            code = run_interactive_experiment.main(["--experiment-dir", str(experiment_dir)])
            self.assertEqual(code, 1)
            status = json.loads(
                (experiment_dir / "hitl/status.json").read_text(encoding="utf-8")
            )
            self.assertEqual(status["state"], "failed")
            self.assertTrue(status["error"])
            self.assertEqual(status["pid"], __import__("os").getpid())


if __name__ == "__main__":
    unittest.main()
