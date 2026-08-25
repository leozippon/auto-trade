"""Orchestration contract of the HITL runner itself (docs/pipeline-design.md).

`tests/unit/test_interactive_worker_local.py` drives the worker end to end, so
the runner's own control-plane branches -- durable stop/pause at a session
boundary, `skip_to_heldout`, `parent_override` delivery, the advisory post-fold
hook, resume, and the re-run token -- were only ever exercised incidentally.
These tests drive `InteractiveExperimentRunner` directly against a recording
executor so each branch is observed, including the ones that must NOT fire.
"""

from __future__ import annotations

import json
import signal
import subprocess
import sys
import unittest
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from autotrade.pipelines import interactive
from autotrade.pipelines.folds import FoldSpec
from autotrade.pipelines.hitl_state import (
    ControlState,
    DevelopmentSession,
    read_control,
    read_status,
    write_control,
)
from autotrade.pipelines.interactive import ExperimentStopped, InteractiveExperimentRunner
from autotrade.pipelines.ledger import ExperimentLedger


def fold_spec(fold_id: str) -> FoldSpec:
    return FoldSpec(
        fold_id=fold_id,
        input_window_start="2020-01-01",
        input_window_end="2021-12-31",
        validation_start="2022-01-01",
        validation_end="2022-03-31",
        test_start="2022-04-01",
        test_end="2022-06-30",
        valid_decision_time=datetime(2022, 4, 1, tzinfo=UTC),
        test_decision_time=datetime(2022, 7, 1, tzinfo=UTC),
    )


def sessions_for(*keys: str) -> tuple[DevelopmentSession, ...]:
    """`meta:<n>` builds a meta session, anything else a fold session."""
    built: list[DevelopmentSession] = []
    for index, key in enumerate(keys):
        if key.startswith("meta:"):
            built.append(
                DevelopmentSession(f"epoch_001/{key}", "meta", "epoch_001", fold_spec("fold_a"), index)
            )
        else:
            built.append(
                DevelopmentSession(f"epoch_001/{key}", "fold", "epoch_001", fold_spec(key), index)
            )
    return tuple(built)


class RecordingExecutor:
    """Appends a canonical ledger record, like the real session executor."""

    def __init__(self, ledger: ExperimentLedger, *, record: bool = True) -> None:
        self.ledger = ledger
        self.record = record
        self.calls: list[tuple[str, dict[str, object]]] = []
        self.on_call = None

    def __call__(self, session: DevelopmentSession, context: dict[str, object]):
        self.calls.append((session.session_key, dict(context)))
        if self.on_call is not None:
            self.on_call(session, context)
        if self.record:
            self.ledger.append(
                {
                    "record_type": "meta_learning" if session.kind == "meta" else "fold",
                    "experiment_id": "exp",
                    "epoch_id": session.epoch_id,
                    "fold_id": session.fold.fold_id,
                    "run_id": f"run_{len(self.calls)}",
                    "session_key": session.session_key,
                    "rerun_id": str(context.get("rerun_id") or ""),
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


class InteractiveRunnerTest(RunnerTestCase):
    def test_auto_mode_runs_every_session_in_order_and_reports_complete(self) -> None:
        executor = RecordingExecutor(self.ledger)
        result = self.runner(sessions_for("fold_a", "meta:1", "fold_b"), executor).run()
        self.assertEqual(result, {"status": "complete", "sessions_run": 3, "reran_sessions": []})
        self.assertEqual(
            executor.keys, ["epoch_001/fold_a", "epoch_001/meta:1", "epoch_001/fold_b"]
        )
        status = read_status(self.status)
        self.assertEqual(status["state"], "development_complete")
        self.assertEqual(status["completed_sessions"], 3)
        # total_sessions reserves the trailing held-out slot the worker runs.
        self.assertEqual(status["total_sessions"], 4)

    def test_a_session_is_retried_after_a_transient_failure(self) -> None:
        class Flaky(RecordingExecutor):
            def __call__(self, session, context):
                if len(self.calls) < 2:
                    self.calls.append((session.session_key, dict(context)))
                    raise RuntimeError(f"boom{len(self.calls)}")
                return super().__call__(session, context)

        executor = Flaky(self.ledger)
        result = self.runner(
            sessions_for("fold_a"), executor, session_max_attempts=3
        ).run()
        self.assertEqual(result["status"], "complete")
        self.assertEqual(len(executor.calls), 3)
        self.assertEqual(read_status(self.status)["state"], "development_complete")

    def test_a_session_fails_the_experiment_after_the_attempt_budget(self) -> None:
        class AlwaysFail(RecordingExecutor):
            def __call__(self, session, context):
                self.calls.append((session.session_key, dict(context)))
                raise RuntimeError("still broken")

        executor = AlwaysFail(self.ledger)
        with self.assertRaisesRegex(RuntimeError, "still broken"):
            self.runner(
                sessions_for("fold_a"), executor, session_max_attempts=3
            ).run()
        self.assertEqual(len(executor.calls), 3)
        self.assertEqual(read_status(self.status)["state"], "failed")

    def test_a_positive_poll_interval_is_required(self) -> None:
        for bad in (0, -1.0):
            with self.subTest(poll_seconds=bad), self.assertRaisesRegex(ValueError, "poll_seconds"):
                self.runner(sessions_for("fold_a"), RecordingExecutor(self.ledger), poll_seconds=bad)

    def test_stop_requested_during_a_session_halts_at_the_next_boundary(self) -> None:
        executor = RecordingExecutor(self.ledger)
        executor.on_call = lambda _session, _context: self.set_control(request="stop")
        result = self.runner(sessions_for("fold_a", "fold_b"), executor).run()
        # The session in flight finishes; the next one never starts.
        self.assertEqual(result, {"status": "stop", "sessions_run": 1, "reran_sessions": []})
        self.assertEqual(executor.keys, ["epoch_001/fold_a"])
        self.assertEqual(read_status(self.status)["state"], "stopped")

    def test_pause_requested_during_a_session_halts_and_reports_paused(self) -> None:
        executor = RecordingExecutor(self.ledger)
        executor.on_call = lambda _session, _context: self.set_control(request="pause")
        result = self.runner(sessions_for("fold_a", "fold_b"), executor).run()
        self.assertEqual(result["status"], "pause")
        self.assertEqual(result["sessions_run"], 1)
        self.assertEqual(read_status(self.status)["state"], "paused")

    def test_a_stop_pending_at_the_gate_raises_before_any_session_runs(self) -> None:
        self.set_control(request="stop")
        executor = RecordingExecutor(self.ledger)
        with self.assertRaisesRegex(ExperimentStopped, "stop requested"):
            self.runner(sessions_for("fold_a"), executor).run()
        self.assertEqual(executor.keys, [])
        status = read_status(self.status)
        self.assertEqual(status["state"], "failed")
        self.assertIn("ExperimentStopped", status["error"])

    def test_a_revealed_experiment_is_sealed_against_further_development(self) -> None:
        self.set_control(test_revealed=True)
        executor = RecordingExecutor(self.ledger)
        with self.assertRaisesRegex(ExperimentStopped, "sealed"):
            self.runner(sessions_for("fold_a"), executor).run()
        self.assertEqual(executor.keys, [])

    def test_skip_to_heldout_stops_after_the_current_fold_only(self) -> None:
        executor = RecordingExecutor(self.ledger)
        self.set_control(skip_to_heldout=True)
        result = self.runner(sessions_for("fold_a", "fold_b", "fold_c"), executor).run()
        self.assertEqual(executor.keys, ["epoch_001/fold_a"])
        # Development is complete, not stopped: held-out still runs afterwards.
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["sessions_run"], 1)
        self.assertEqual(read_status(self.status)["state"], "development_complete")

    def test_skip_to_heldout_does_not_break_out_of_a_meta_session(self) -> None:
        """The skip must land on a frozen fold artifact, never mid-epoch on a
        Meta session whose fold has not run yet."""
        executor = RecordingExecutor(self.ledger)
        self.set_control(skip_to_heldout=True)
        self.runner(sessions_for("meta:1", "fold_a", "fold_b"), executor).run()
        self.assertEqual(executor.keys, ["epoch_001/meta:1", "epoch_001/fold_a"])

    def test_parent_override_is_delivered_to_its_own_session_and_no_other(self) -> None:
        executor = RecordingExecutor(self.ledger)
        self.set_control(parent_overrides={"epoch_001/fold_b": "node_42"})
        self.runner(sessions_for("fold_a", "fold_b", "fold_c"), executor).run()
        overrides = {key: context["parent_override"] for key, context in executor.calls}
        self.assertEqual(
            overrides,
            {"epoch_001/fold_a": "", "epoch_001/fold_b": "node_42", "epoch_001/fold_c": ""},
        )
        # A parent override is a standing choice, not a one-shot approval: the
        # completed-session sweep must not silently discard it.
        self.assertEqual(read_control(self.control).parent_overrides, {"epoch_001/fold_b": "node_42"})

    def test_per_session_directives_and_overrides_reach_the_session_then_are_consumed(self) -> None:
        executor = RecordingExecutor(self.ledger)
        self.set_control(
            directives={"epoch_001/fold_a": "try momentum"},
            prompt_overrides={"epoch_001/fold_a": "custom prompt"},
            resource_overrides={"epoch_001/fold_a": {"max_steps": 2}},
        )
        self.runner(sessions_for("fold_a"), executor).run()
        _key, context = executor.calls[0]
        self.assertEqual(context["directive"], "try momentum")
        self.assertEqual(context["prompt_override"], "custom prompt")
        self.assertEqual(context["resource_override"], {"max_steps": 2})
        self.assertEqual(context["session_key"], "epoch_001/fold_a")
        for hook in ("step_gate_hook", "user_question_hook", "progress_hook", "session_timing"):
            self.assertTrue(callable(context[hook]), hook)
        control = read_control(self.control)
        self.assertEqual((control.directives, control.prompt_overrides, control.resource_overrides),
                         ({}, {}, {}))

    def test_a_per_session_gpu_count_reaches_only_its_own_session_and_is_consumed(self) -> None:
        """`set_gpu_count` is a one-shot allocation, like an approval.

        The console writes it against one session key; the runner must hand it
        to that session alone and clear it afterwards, or the next fold would
        silently inherit an allocation nobody asked for.
        """
        executor = RecordingExecutor(self.ledger)
        self.set_control(gpu_counts={"epoch_001/fold_a": 3})
        self.runner(sessions_for("fold_a", "fold_b"), executor).run()
        self.assertEqual(
            {key: context["sandbox_gpu_count"] for key, context in executor.calls},
            {"epoch_001/fold_a": 3, "epoch_001/fold_b": None},
        )
        self.assertEqual(read_control(self.control).gpu_counts, {})

    def test_a_session_that_records_nothing_durable_fails_fast(self) -> None:
        executor = RecordingExecutor(self.ledger, record=False)
        with self.assertRaisesRegex(RuntimeError, "without a durable success record"):
            self.runner(sessions_for("fold_a"), executor).run()
        self.assertEqual(read_status(self.status)["state"], "failed")

    def test_a_record_returned_by_the_executor_is_appended_with_its_session_key(self) -> None:
        ledger = self.ledger

        def execute(session, _context):
            return {
                "record_type": "fold",
                "experiment_id": "exp",
                "epoch_id": session.epoch_id,
                "fold_id": session.fold.fold_id,
                "run_id": "run_1",
            }

        self.runner(sessions_for("fold_a"), execute).run()
        rows = ledger.read("fold")
        self.assertEqual([row["session_key"] for row in rows], ["epoch_001/fold_a"])

    def test_resume_skips_sessions_already_recorded_in_the_ledger(self) -> None:
        first = RecordingExecutor(self.ledger)
        self.runner(sessions_for("fold_a", "fold_b"), first).run()
        second = RecordingExecutor(self.ledger)
        result = self.runner(sessions_for("fold_a", "fold_b", "fold_c"), second).run()
        self.assertEqual(second.keys, ["epoch_001/fold_c"])
        self.assertEqual(result["sessions_run"], 1)
        self.assertEqual(read_status(self.status)["completed_sessions"], 3)

    def test_a_pending_rerun_token_reruns_a_completed_fold_and_is_recorded(self) -> None:
        self.runner(sessions_for("fold_a", "fold_b"), RecordingExecutor(self.ledger)).run()
        self.set_control(rerun_sessions={"epoch_001/fold_a": "rerun-7"})
        executor = RecordingExecutor(self.ledger)
        result = self.runner(sessions_for("fold_a", "fold_b"), executor).run()
        self.assertEqual(executor.keys, ["epoch_001/fold_a"])
        self.assertEqual(result["reran_sessions"], ["epoch_001/fold_a"])
        self.assertEqual(executor.calls[0][1]["rerun_id"], "rerun-7")
        rows = [row for row in self.ledger.read("fold") if row["session_key"] == "epoch_001/fold_a"]
        self.assertEqual([row.get("rerun_id") for row in rows], ["", "rerun-7"])

    def test_a_rerun_whose_record_omits_the_token_refuses_to_advance(self) -> None:
        self.runner(sessions_for("fold_a"), RecordingExecutor(self.ledger)).run()
        self.set_control(rerun_sessions={"epoch_001/fold_a": "rerun-7"})

        class Forgetful(RecordingExecutor):
            def __call__(self, session, context):
                return super().__call__(session, {**context, "rerun_id": ""})

        with self.assertRaisesRegex(RuntimeError, "did not record its rerun id"):
            self.runner(sessions_for("fold_a"), Forgetful(self.ledger)).run()

    def test_a_stale_rerun_token_already_absorbed_does_not_rerun(self) -> None:
        self.set_control(rerun_sessions={"epoch_001/fold_a": "rerun-7"})
        self.runner(sessions_for("fold_a"), RecordingExecutor(self.ledger)).run()
        again = RecordingExecutor(self.ledger)
        result = self.runner(sessions_for("fold_a"), again).run()
        self.assertEqual(again.keys, [])
        self.assertEqual(result["reran_sessions"], [])

    def test_a_rerun_token_never_reruns_a_meta_session(self) -> None:
        self.runner(sessions_for("meta:1"), RecordingExecutor(self.ledger)).run()
        self.set_control(rerun_sessions={"epoch_001/meta:1": "rerun-7"})
        again = RecordingExecutor(self.ledger)
        self.assertEqual(self.runner(sessions_for("meta:1"), again).run()["sessions_run"], 0)
        self.assertEqual(again.keys, [])

    def test_manual_mode_waits_at_the_gate_until_the_session_is_approved(self) -> None:
        executor = RecordingExecutor(self.ledger)
        self.set_control(mode="manual")
        waits: list[dict[str, object]] = []

        def approve_on_first_poll(_seconds: float) -> None:
            # The gate polls control.json between sleeps; capture what the
            # console would have seen, then approve.
            waits.append(read_status(self.status))
            self.set_control(approved_sessions=("epoch_001/fold_a",))

        with patch.object(interactive.time, "sleep", approve_on_first_poll):
            self.runner(sessions_for("fold_a"), executor).run()

        self.assertEqual(len(waits), 1, "the gate did not block on an unapproved session")
        self.assertEqual(waits[0]["state"], "waiting_user")
        self.assertEqual(waits[0]["session_key"], "epoch_001/fold_a")
        self.assertEqual(waits[0]["session_kind"], "fold")
        self.assertIsNotNone(waits[0]["wait_started_at"])
        self.assertEqual(executor.keys, ["epoch_001/fold_a"])
        # The one-shot approval is consumed when the session completes.
        self.assertEqual(read_control(self.control).approved_sessions, ())

    def test_manual_mode_does_not_gate_a_session_approved_up_front(self) -> None:
        self.set_control(mode="manual", approved_sessions=("epoch_001/fold_a",))
        with patch.object(interactive.time, "sleep", side_effect=AssertionError("gate slept")):
            self.runner(sessions_for("fold_a"), RecordingExecutor(self.ledger)).run()

    def test_a_stop_arriving_while_the_gate_waits_ends_the_run(self) -> None:
        self.set_control(mode="manual")
        executor = RecordingExecutor(self.ledger)
        with patch.object(
            interactive.time, "sleep", lambda _s: self.set_control(request="stop")
        ), self.assertRaisesRegex(ExperimentStopped, "stop requested"):
            self.runner(sessions_for("fold_a"), executor).run()
        self.assertEqual(executor.keys, [])

class PostFoldHookTest(RunnerTestCase):
    def test_the_hook_receives_the_folds_own_ledger_record(self) -> None:
        seen: list[dict[str, object]] = []
        executor = RecordingExecutor(self.ledger)
        self.runner(
            sessions_for("fold_a", "fold_b"), executor, post_fold_hook=seen.append
        ).run()
        self.assertEqual(
            [row["session_key"] for row in seen], ["epoch_001/fold_a", "epoch_001/fold_b"]
        )
        self.assertEqual([row["run_id"] for row in seen], ["run_1", "run_2"])

    def test_the_hook_never_runs_for_a_meta_session(self) -> None:
        seen: list[dict[str, object]] = []
        self.runner(
            sessions_for("meta:1"), RecordingExecutor(self.ledger), post_fold_hook=seen.append
        ).run()
        self.assertEqual(seen, [])

    def test_a_failing_hook_is_advisory_and_never_invalidates_the_fold(self) -> None:
        def explode(_record: dict[str, object]) -> None:
            raise RuntimeError("analysis model unavailable")

        result = self.runner(
            sessions_for("fold_a"), RecordingExecutor(self.ledger), post_fold_hook=explode
        ).run()
        self.assertEqual(result["status"], "complete")
        status = read_status(self.status)
        self.assertEqual(status["analysis_error"], "RuntimeError: analysis model unavailable")
        self.assertEqual(len(self.ledger.read("fold")), 1)

    def test_a_recovered_hook_clears_the_previous_analysis_error(self) -> None:
        calls: list[str] = []

        def flaky(record: dict[str, object]) -> None:
            calls.append(str(record["session_key"]))
            if len(calls) == 1:
                raise RuntimeError("transient")

        self.runner(
            sessions_for("fold_a", "fold_b"), RecordingExecutor(self.ledger), post_fold_hook=flaky
        ).run()
        self.assertEqual(len(calls), 2)
        self.assertIsNone(read_status(self.status)["analysis_error"])


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
