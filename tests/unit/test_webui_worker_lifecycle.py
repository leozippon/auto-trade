"""Console operations that touch REAL worker processes: terminate, restart, create.

`tests/unit/test_webui_backend.py` covers the rest of the console by
synthesizing state on disk and patching spawn out. These three cannot be: two
of them signal a live process and the third decides whether one is started at
all. Every test here drives the real HTTP route against a genuine detached
child or a genuine worker entrypoint, and asserts what happened to that
process and to the files on disk.

Two subtleties have their own tests. `_terminate` waits out a SIGTERM grace,
during which the exiting worker still has to take `control_lock` to run
`consume_session_controls`; dispatching the action inside that lock would
deadlock the worker against the console and silently turn every graceful stop
into a SIGKILL. And `create_experiment`'s pre-flight has to reject BEFORE
anything is written -- a create that fails after the mkdir leaves a `failed`
experiment for the researcher to diagnose instead of an error they can act on,
which is the exact failure mode it was restored to remove.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

import autotrade
from autotrade.environment.identity import AgentRefStore
from autotrade.environment.runtime import write_json_atomic
from autotrade.pipelines.hitl_state import (
    WEB_CREATE_DEFAULTS,
    ControlState,
    proc_start_ticks,
    read_control,
    write_control,
)
from autotrade.pipelines.ledger import ExperimentLedger
from autotrade.webui import manager as manager_module
from autotrade.webui.manager import ManagerError
from autotrade.webui.server import create_app

SRC_ROOT = str(Path(autotrade.__file__).resolve().parents[1])

# A worker that exits on SIGTERM, as the real one does.
_COOPERATIVE = """
import signal, sys, time
signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
open({ready!r}, "w").close()
time.sleep(300)
"""

# A worker that exits on SIGTERM only after consuming its own session's
# controls -- the real runner's last act before returning from a session.
_CONSUMES_CONTROLS = """
import signal, sys, time
sys.path.insert(0, {src!r})
from autotrade.pipelines.hitl_state import consume_session_controls

def _handle(signum, frame):
    consume_session_controls({control!r}, {session!r})
    sys.exit(0)

signal.signal(signal.SIGTERM, _handle)
open({ready!r}, "w").close()
time.sleep(300)
"""

# A worker stuck in blocking work (an LLM retry, a docker build) that never
# gets to its handler.
_STUBBORN = """
import signal, sys, time
signal.signal(signal.SIGTERM, signal.SIG_IGN)
open({ready!r}, "w").close()
time.sleep(300)
"""

# A worker that honours a session-boundary restart the way the real one does:
# it consumes the flag between sessions and calls the entrypoint's own
# ``_exec_self``. Written to a file (not ``python -c``) because that is what
# the console spawns and what ``sys.argv`` has to reproduce across the exec.
_DEFERRED_RESTART = """
import importlib.util, os, sys, time
from pathlib import Path

sys.path.insert(0, "__SRC__")
from autotrade.environment.runtime import write_json_atomic
from autotrade.pipelines.hitl_state import (
    consume_restart_request,
    proc_start_ticks,
    read_control,
)

_spec = importlib.util.spec_from_file_location("worker_entry", "__ENTRY__")
_entry = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_entry)

pid = os.getpid()
with open("__MARKER__", "a", encoding="utf-8") as handle:
    handle.write(" ".join([str(pid), str(proc_start_ticks(pid)), *sys.argv[1:]]) + chr(10))
write_json_atomic(
    Path("__STATUS__"),
    dict(
        schema_version=1,
        state="running_session",
        pid=pid,
        pid_start_ticks=proc_start_ticks(pid),
        session_key="epoch_001/fold_2022Q2",
    ),
)
open("__READY__", "w").close()
deadline = time.monotonic() + 60.0
while time.monotonic() < deadline:
    if read_control("__CONTROL__").restart_pending and consume_restart_request("__CONTROL__"):
        _entry._exec_self()
    time.sleep(0.05)
"""


class WorkerLifecycleTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.repo_root = Path(self._tmp.name)
        self.experiments_root = self.repo_root / "experiments"
        self.directory = self.experiments_root / "exp_ctl"
        AgentRefStore(self.directory)
        self.hitl = self.directory / "hitl"
        self.hitl.mkdir(parents=True)
        self.control_path = self.hitl / "control.json"
        self.status_path = self.hitl / "status.json"
        write_json_atomic(self.hitl / "params.json", {"experiment_id": "exp_ctl"})
        write_control(self.control_path, ControlState(mode="manual"))
        write_json_atomic(
            self.status_path, {"schema_version": 1, "pid": 999_999_999, "state": "stopped"}
        )
        write_json_atomic(
            self.hitl / "schedule.json",
            {
                "schema_version": 1,
                "epochs": 1,
                "sessions": [
                    {"key": "epoch_001/fold_2022Q1", "kind": "fold",
                     "epoch_id": "epoch_001", "fold_id": "fold_2022Q1"},
                    {"key": "epoch_001/fold_2022Q2", "kind": "fold",
                     "epoch_id": "epoch_001", "fold_id": "fold_2022Q2"},
                    {"key": "heldout", "kind": "heldout", "epoch_id": "epoch_001",
                     "periods": [{"label": "2023Q1"}]},
                ],
            },
        )
        # fold_2022Q1 is settled (it has a durable record); fold_2022Q2 is not.
        ExperimentLedger(self.directory / "ledgers/experiment_ledger.jsonl").append(
            {
                "record_type": "fold", "experiment_id": "exp_ctl", "epoch_id": "epoch_001",
                "fold_id": "fold_2022Q1", "run_id": "run_001",
                "session_key": "epoch_001/fold_2022Q1", "fold_status": "frozen",
                "validation_result": {"total_return": 0.1},
            }
        )
        self.client = TestClient(create_app(self.repo_root, self.experiments_root))

    # ---- helpers ---------------------------------------------------------
    def _spawn(self, source: str) -> subprocess.Popen:
        ready = self.repo_root / f"ready_{os.urandom(4).hex()}"
        process = subprocess.Popen(
            [sys.executable, "-c", textwrap.dedent(source).format(
                ready=str(ready), src=SRC_ROOT,
                control=str(self.control_path), session="epoch_001/fold_2022Q2",
            )],
            start_new_session=True, stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        self.addCleanup(self._reap, process)
        deadline = time.monotonic() + 20.0
        while time.monotonic() < deadline and not ready.exists():
            time.sleep(0.02)
        self.assertTrue(ready.exists(), "the child never reached its signal handler install")
        return process

    def _spawn_deferred_worker(self) -> tuple[subprocess.Popen, Path]:
        """A live worker that re-execs itself when the flag is set."""
        entry = (
            Path(SRC_ROOT).parent / "scripts/experiments/run_interactive_experiment.py"
        )
        self.assertTrue(entry.is_file(), "the real worker entrypoint is missing")
        marker = self.repo_root / "restarts.txt"
        ready = self.repo_root / f"ready_{os.urandom(4).hex()}"
        script = self.repo_root / "fake_worker.py"
        script.write_text(
            textwrap.dedent(_DEFERRED_RESTART)
            .replace("__SRC__", SRC_ROOT)
            .replace("__ENTRY__", str(entry))
            .replace("__MARKER__", str(marker))
            .replace("__STATUS__", str(self.status_path))
            .replace("__READY__", str(ready))
            .replace("__CONTROL__", str(self.control_path)),
            encoding="utf-8",
        )
        process = subprocess.Popen(
            [sys.executable, str(script), "--experiment-dir", str(self.directory)],
            start_new_session=True, stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        self.addCleanup(self._reap, process)
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline and not ready.exists():
            time.sleep(0.02)
        self.assertTrue(ready.exists(), "the fake worker never published its status")
        return process, marker

    def _reap(self, process: subprocess.Popen) -> None:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                process.kill()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:  # pragma: no cover - defensive
            pass

    def _publish(self, process: subprocess.Popen, *, session_key: str) -> None:
        write_json_atomic(
            self.status_path,
            {
                "schema_version": 1, "pid": process.pid,
                "pid_start_ticks": proc_start_ticks(process.pid),
                "state": "running_session", "session_key": session_key,
            },
        )

    def _post(self, **payload):
        return self.client.post("/api/experiments/exp_ctl/control", json=payload)

    # ---- terminate -------------------------------------------------------
    def test_terminate_signals_the_worker_group_and_reports_a_graceful_exit(self) -> None:
        process = self._spawn(_COOPERATIVE)
        self._publish(process, session_key="epoch_001/fold_2022Q2")
        started = time.monotonic()
        response = self._post(action="terminate")
        elapsed = time.monotonic() - started
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["terminated_pid"], process.pid)
        self.assertIs(body["escalated"], False)
        self.assertIsInstance(body["reclaimed_containers"], list)
        self.assertLess(elapsed, 9.0, "a cooperative worker must not wait out the grace")
        process.wait(timeout=10)
        self.assertEqual(process.returncode, 0, "the worker ran its own SIGTERM handler")

    def test_terminate_does_not_hold_control_lock_across_the_grace(self) -> None:
        """The regression that would make every graceful stop a kill.

        The child takes `control_lock` from inside its SIGTERM handler, which
        is exactly what `InteractiveExperimentRunner` does on the way out. If
        the console dispatched `terminate` while holding that lock, the child
        would block for the full 10 s grace and be SIGKILLed.
        """
        write_control(
            self.control_path,
            ControlState(
                mode="manual",
                approved_sessions=("epoch_001/fold_2022Q2",),
                directives={"epoch_001/fold_2022Q2": "keep the turnover down"},
            ),
        )
        process = self._spawn(_CONSUMES_CONTROLS)
        self._publish(process, session_key="epoch_001/fold_2022Q2")
        started = time.monotonic()
        body = self._post(action="terminate").json()
        elapsed = time.monotonic() - started
        process.wait(timeout=10)
        self.assertEqual(process.returncode, 0, "the worker was killed instead of exiting")
        self.assertIs(body["escalated"], False)
        self.assertLess(elapsed, 9.0)
        # Proof the worker really acquired the lock while terminate was waiting.
        self.assertEqual(read_control(self.control_path).directives, {})

    def test_terminate_returns_an_unsettled_session_to_its_approval_gate(self) -> None:
        write_control(
            self.control_path,
            ControlState(mode="auto", approved_sessions=("epoch_001/fold_2022Q2",)),
        )
        process = self._spawn(_COOPERATIVE)
        self._publish(process, session_key="epoch_001/fold_2022Q2")
        body = self._post(action="terminate").json()
        expected_public_session = (
            "epoch_001/"
            + AgentRefStore(self.directory).get_or_create("fold", "fold_2022Q2")
        )
        self.assertEqual(body["approval_revoked_session"], expected_public_session)
        self.assertNotIn("fold_2022Q2", str(body))
        control = read_control(self.control_path)
        self.assertEqual(control.approved_sessions, ())
        # Without the auto -> manual flip the revocation is meaningless: auto
        # mode does not gate, so the resumed worker would run straight past it.
        self.assertEqual(control.mode, "manual")

    def test_terminate_leaves_a_settled_folds_approval_alone(self) -> None:
        write_control(
            self.control_path,
            ControlState(mode="auto", approved_sessions=("epoch_001/fold_2022Q1",)),
        )
        process = self._spawn(_COOPERATIVE)
        self._publish(process, session_key="epoch_001/fold_2022Q1")
        body = self._post(action="terminate").json()
        self.assertNotIn("approval_revoked_session", body)
        control = read_control(self.control_path)
        self.assertEqual(control.approved_sessions, ("epoch_001/fold_2022Q1",))
        self.assertEqual(control.mode, "auto")

    def test_terminate_refuses_when_no_worker_is_alive(self) -> None:
        refused = self._post(action="terminate")
        self.assertEqual(refused.status_code, 400)
        self.assertEqual(refused.json()["detail"], "no live worker to terminate")

    def test_terminate_escalates_to_sigkill_and_stamps_a_terminal_state(self) -> None:
        """A worker that ignores SIGTERM is killed after the 10 s grace.

        Real wall time on purpose: the grace is the guarantee a researcher
        relies on when a fold is wedged in a blocking call, and a mocked clock
        would not prove the process ever dies.
        """
        process = self._spawn(_STUBBORN)
        self._publish(process, session_key="epoch_001/fold_2022Q2")
        started = time.monotonic()
        body = self._post(action="terminate").json()
        elapsed = time.monotonic() - started
        self.assertIs(body["escalated"], True)
        self.assertEqual(body["terminated_pid"], process.pid)
        process.wait(timeout=10)
        self.assertEqual(process.returncode, -signal.SIGKILL)
        self.assertGreaterEqual(elapsed, 10.0, "the grace was cut short")
        self.assertLess(elapsed, 20.0)
        # SIGKILL leaves nobody to write a terminal state, so the console does.
        status = json.loads(self.status_path.read_text(encoding="utf-8"))
        self.assertEqual(status["state"], "terminated")
        self.assertIsNone(status["error"])
        self.assertTrue(status["terminated_at"])

    # ---- restart ---------------------------------------------------------
    def _install_worker_script(self) -> None:
        script = self.repo_root / "scripts/experiments/run_interactive_experiment.py"
        script.parent.mkdir(parents=True, exist_ok=True)
        script.write_text("import time\ntime.sleep(120)\n", encoding="utf-8")

    def test_restart_terminates_the_old_worker_and_spawns_a_new_one(self) -> None:
        self._install_worker_script()
        write_control(
            self.control_path,
            ControlState(mode="manual", request="stop", restart_pending=True),
        )
        process = self._spawn(_COOPERATIVE)
        self._publish(process, session_key="epoch_001/fold_2022Q2")
        response = self._post(action="restart")
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.addCleanup(self._kill_pid, int(body["spawned_pid"]))
        self.assertIs(body["restarted"], True)
        self.assertEqual(body["at"], "immediate")
        self.assertIs(body["escalated"], False)
        self.assertIs(body["spawned"], True)
        process.wait(timeout=10)
        self.assertEqual(process.returncode, 0)
        self.assertNotEqual(body["spawned_pid"], process.pid)
        status = json.loads(self.status_path.read_text(encoding="utf-8"))
        self.assertEqual(status["pid"], body["spawned_pid"])
        self.assertEqual(status["state"], "launching")
        # The restored `start_worker` clearing: a leftover stop request would
        # make the freshly spawned worker halt at its first gate, and a
        # leftover deferred restart belongs to the worker that was replaced.
        control = read_control(self.control_path)
        self.assertIsNone(control.request)
        self.assertFalse(control.restart_pending)

    def test_restart_does_not_hold_control_lock_while_the_old_worker_exits(self) -> None:
        """`restart` is exposed to the same deadlock as `terminate`, twice over.

        It waits for the old worker (which takes `control_lock` on its way
        out) and then calls `start_worker`, which takes the same lock to clear
        a stale stop request. Dispatching it inside the lock would hang both.
        """
        self._install_worker_script()
        write_control(
            self.control_path,
            ControlState(
                mode="manual",
                request="stop",
                directives={"epoch_001/fold_2022Q2": "keep the turnover down"},
            ),
        )
        process = self._spawn(_CONSUMES_CONTROLS)
        self._publish(process, session_key="epoch_001/fold_2022Q2")
        started = time.monotonic()
        response = self._post(action="restart")
        elapsed = time.monotonic() - started
        self.assertEqual(response.status_code, 200, response.text)
        self.addCleanup(self._kill_pid, int(response.json()["spawned_pid"]))
        process.wait(timeout=10)
        self.assertEqual(process.returncode, 0)
        self.assertLess(elapsed, 25.0, "restart waited out its budget instead of the worker")
        control = read_control(self.control_path)
        self.assertEqual(control.directives, {}, "the exiting worker never got the lock")
        self.assertIsNone(control.request)

    def test_start_worker_clears_a_stale_stop_request_but_keeps_mode_and_approvals(self) -> None:
        self._install_worker_script()
        write_control(
            self.control_path,
            ControlState(
                mode="step", request="stop", approved_sessions=("epoch_001/fold_2022Q2",)
            ),
        )
        spawned = manager_module.ExperimentManager(
            self.repo_root, self.experiments_root
        ).start_worker("exp_ctl")
        self.addCleanup(self._kill_pid, int(spawned["spawned_pid"]))
        control = read_control(self.control_path)
        self.assertIsNone(control.request)
        self.assertEqual(control.mode, "step")
        self.assertEqual(control.approved_sessions, ("epoch_001/fold_2022Q2",))

    def test_deferred_restart_swaps_code_in_place_without_killing_the_session(
        self,
    ) -> None:
        """The whole point: a code swap that costs no in-progress Fold.

        The request only records a flag — the live worker is never signalled —
        and the worker re-executes itself at its next session boundary. The
        exec keeps the pid, its start time and its argv, which is what the
        console's liveness check, its signalling and ``status.json`` rely on.
        """
        process, marker = self._spawn_deferred_worker()
        started = time.monotonic()
        response = self._post(action="restart", at="session_boundary")
        elapsed = time.monotonic() - started
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["at"], "session_boundary")
        self.assertIs(body["restarted"], False)
        self.assertIs(body["restart_pending"], True)
        self.assertLess(elapsed, 5.0, "a deferred restart must not wait on the worker")
        self.assertTrue(read_control(self.control_path).restart_pending)

        deadline = time.monotonic() + 60.0
        lines: list[str] = []
        while time.monotonic() < deadline:
            lines = marker.read_text(encoding="utf-8").split("\n")
            lines = [line for line in lines if line]
            if len(lines) >= 2:
                break
            time.sleep(0.05)
        self.assertEqual(len(lines), 2, "the worker never re-executed itself")
        before, after = (line.split(" ", 2) for line in lines)
        self.assertEqual(after[0], str(process.pid), "the pid changed across the exec")
        self.assertEqual(after[1], before[1], "the process start time changed")
        self.assertEqual(
            after[2], f"--experiment-dir {self.directory}", "argv was not reproduced"
        )
        self.assertIsNone(process.poll(), "the worker process was replaced, not killed")
        # One-shot: the new image must not restart again at the next boundary.
        self.assertFalse(read_control(self.control_path).restart_pending)
        status = json.loads(self.status_path.read_text(encoding="utf-8"))
        self.assertEqual(status["pid"], process.pid)
        self.assertEqual(
            self.client.get("/api/experiments/exp_ctl").json()["worker_alive"],
            True,
            "the console lost the worker across the exec",
        )

    def test_deferred_restart_refuses_without_a_live_worker(self) -> None:
        refused = self._post(action="restart", at="session_boundary")
        self.assertEqual(refused.status_code, 400)
        self.assertEqual(
            refused.json()["detail"], "session_boundary restart requires a live worker"
        )
        self.assertFalse(read_control(self.control_path).restart_pending)

    def test_restart_rejects_an_unknown_boundary_and_at_on_other_actions(self) -> None:
        rejected = self._post(action="restart", at="tomorrow")
        self.assertEqual(rejected.status_code, 400)
        self.assertIn("immediate or session_boundary", rejected.json()["detail"])
        misplaced = self._post(action="pause", at="session_boundary")
        self.assertEqual(misplaced.status_code, 400)
        self.assertEqual(misplaced.json()["detail"], "at is only accepted by restart")

    def test_restart_sigkills_a_worker_that_outlives_its_grace_and_resumes(self) -> None:
        """A wedged worker is forced, not left to the researcher.

        The work a worker ignores SIGTERM for is a model call that routinely
        runs minutes, so refusing turned nearly every restart into a manual
        force-terminate followed by a resume. The grace is patched down: the
        branch under test is the escalation, not the length of the wait.
        """
        self._install_worker_script()
        process = self._spawn(_STUBBORN)
        self._publish(process, session_key="epoch_001/fold_2022Q2")
        with patch.object(manager_module, "_RESTART_GRACE_SECONDS", 1.0):
            response = self._post(action="restart")
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.addCleanup(self._kill_pid, int(body["spawned_pid"]))
        self.assertIs(body["restarted"], True)
        self.assertIs(body["escalated"], True)
        process.wait(timeout=10)
        self.assertEqual(process.returncode, -signal.SIGKILL)
        self.assertNotEqual(body["spawned_pid"], process.pid)
        # The replacement worker owns the status file, so the console never
        # reports the killed pid as live.
        status = json.loads(self.status_path.read_text(encoding="utf-8"))
        self.assertEqual(status["pid"], body["spawned_pid"])
        self.assertEqual(status["state"], "launching")

    def _kill_pid(self, pid: int) -> None:
        try:
            os.killpg(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            try:
                os.kill(pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                return
        try:
            os.waitpid(pid, 0)
        except ChildProcessError:
            pass


if __name__ == "__main__":
    unittest.main()


#: (params override, the validator message the browser must be shown). Every
#: entry is a value that used to reach `params.json` and kill the worker a
#: second later; the pre-flight turns each into an actionable HTTP 400.
_REJECTED_CREATES = (
    ({"epochs": -1}, "epochs must be a positive integer"),
    ({"max_steps_per_fold": 0}, "max_steps_per_fold must be a positive integer"),
    ({"initial_cash": 0}, "initial_cash must be a positive finite number"),
    ({"max_drawdown": 1.5}, "max_drawdown must be between 0.0 and 1.0"),
    ({"compact_max_calls": -1}, "compact_max_calls must be a non-negative integer"),
    ({"window_months": 0}, "window_months must be a positive integer"),
    # Range, not availability: a GPU-less host must still see this message.
    ({"gpu_count": 9}, "gpu_count must be between 0 and 4"),
    ({"inference_time": "25:00"}, "inference_time must use 24-hour HH:MM"),
    ({"strategy_period": "fortnight"}, "period must be one of"),
    ({"reasoning_effort": "turbo"}, "reasoning_effort must be one of"),
    ({"events_datasets": ["not_a_dataset"]}, "unknown events_datasets"),
    ({"screen_min_price": -1.0}, "screen_min_price must be a non-negative finite number"),
    ({"analysis_max_tokens": 0}, "analysis_max_tokens must be a positive integer"),
    ({"meta_memory_max_epochs": -1}, "meta_memory_max_epochs must be a non-negative integer"),
)


class CreatePreflightTest(unittest.TestCase):
    """`create_experiment` validates the whole request before it writes anything.

    The property under test is not the status code -- it is that a refused
    create leaves NO trace: no experiment directory, no `status.json`, and no
    spawned worker. Spawn is deliberately NOT patched out; the worker
    entrypoint is a real script that records the fact it ran, because patching
    it would hide the regression this guards against.
    """

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.repo_root = Path(self._tmp.name)
        self.experiments_root = self.repo_root / "experiments"
        self.experiments_root.mkdir(parents=True)
        self.marker = self.repo_root / "worker_ran"
        script = self.repo_root / "scripts/experiments/run_interactive_experiment.py"
        script.parent.mkdir(parents=True, exist_ok=True)
        script.write_text(
            f"import pathlib, time\n"
            f"pathlib.Path({str(self.marker)!r}).write_text('ran')\n"
            f"time.sleep(120)\n",
            encoding="utf-8",
        )
        self.client = TestClient(create_app(self.repo_root, self.experiments_root))

    def _create(self, **overrides):
        payload = {
            "experiment_id": "preflight_demo",
            "fold_period": "quarter",
            "development_first_period": "2026Q1",
            "development_last_period": "2026Q1",
            "heldout_first_period": "2026Q2",
            "heldout_last_period": "2026Q2",
        }
        payload.update(overrides)
        return self.client.post("/api/experiments", json=payload)

    def _assert_nothing_was_created(self) -> None:
        self.assertEqual(list(self.experiments_root.iterdir()), [])
        self.assertEqual(list(self.experiments_root.rglob("status.json")), [])

    def _await_marker(self, timeout: float = 10.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline and not self.marker.exists():
            time.sleep(0.02)
        return self.marker.exists()

    def test_a_bad_parameter_is_refused_with_its_own_message_and_writes_nothing(self) -> None:
        for overrides, message in _REJECTED_CREATES:
            with self.subTest(**overrides):
                response = self._create(**overrides)
                self.assertEqual(response.status_code, 400, response.text)
                # The validator's own words, not a generic "invalid request":
                # a refactor that swallowed the reason would still 400.
                self.assertIn(message, response.json()["detail"])
                self._assert_nothing_was_created()

    def test_a_refused_create_never_spawns_a_worker(self) -> None:
        """Spawn is not patched: the entrypoint records that it ran."""
        response = self._create(epochs=-1)
        self.assertEqual(response.status_code, 400)
        self._assert_nothing_was_created()
        # A worker that DID start records itself in ~0.04 s on this host under
        # load; two seconds is ample margin for the negative assertion.
        time.sleep(2.0)
        self.assertFalse(self.marker.exists(), "a refused create started a worker")

    def test_a_good_create_still_works_end_to_end(self) -> None:
        response = self._create()
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertIs(body["spawned"], True)
        self.addCleanup(self._kill_pid, int(body["spawned_pid"]))
        directory = self.experiments_root / "preflight_demo"
        self.assertNotIn("experiment_dir", body)
        for name in ("params.json", "control.json", "status.json"):
            self.assertTrue((directory / "hitl" / name).is_file(), name)
        params = json.loads((directory / "hitl/params.json").read_text(encoding="utf-8"))
        self.assertEqual(params["_creation_surface"], "webui")
        self.assertTrue(params["_created_at"])
        self.assertTrue(self._await_marker(), "the worker entrypoint never ran")

    def test_the_params_a_create_persists_are_accepted_by_the_worker_itself(self) -> None:
        """The console must never write something its own worker would reject.

        The pre-flight runs `resolve_worker_options(..., preflight=True)`; the
        spawned worker runs the same function with `preflight=False`. This
        asserts the persisted file survives the stricter pass -- every
        parameter check, only the deployment-state steps skipped.
        """
        from autotrade.pipelines.worker import _ALLOWED_PARAMS, resolve_worker_options

        response = self._create()
        self.assertEqual(response.status_code, 200, response.text)
        self.addCleanup(self._kill_pid, int(response.json()["spawned_pid"]))
        directory = self.experiments_root / "preflight_demo"
        params = json.loads((directory / "hitl/params.json").read_text(encoding="utf-8"))
        self.assertTrue(
            set(params) - {key for key in params if key.startswith("_")} <= set(_ALLOWED_PARAMS)
        )
        # preflight=False would additionally stat the data roots and pin a
        # research release, neither of which exists under a tmp repo root; the
        # parameter checks are what must pass, and they are shared verbatim.
        resolve_worker_options(
            params, experiment_dir=directory, repo_root=self.repo_root, preflight=True
        )

    def test_the_console_and_the_worker_validate_through_one_shared_body(self) -> None:
        """Not two validators that happen to agree today -- one body both call.

        Proved by moving the shared validator and watching both paths follow:
        a change in `worker._positive_int` has to surface in the console's
        create-time 400 AND in the worker's own load of a params.json it
        already accepted. Two copies would diverge here.

        Uses the real repository root because the worker's pass
        (`preflight=False`) pins a research release and reads the trading
        calendar -- the two steps the pre-flight skips.
        """
        from autotrade.pipelines import worker as worker_module

        repository = Path(autotrade.__file__).resolve().parents[2]
        manager = manager_module.ExperimentManager(repository, self.experiments_root)
        self.assertTrue(manager.worker_script.is_file())
        payload = {
            "experiment_id": "shared_body",
            "fold_period": "quarter",
            "development_first_period": "2026Q1",
            "development_last_period": "2026Q1",
            "heldout_first_period": "2026Q2",
            "heldout_last_period": "2026Q2",
        }
        with patch.object(manager, "start_worker", lambda experiment_id: {"spawned": False}):
            created = manager.create_experiment(dict(payload))
        self.assertEqual(created, {"experiment_id": "shared_body", "spawned": False})
        directory = self.experiments_root / "shared_body"
        # Baseline: what the console persisted survives the worker's stricter pass.
        worker_module.load_worker_options(directory, repo_root=repository)

        original = worker_module._positive_int

        def stricter(value: object, name: str) -> int:
            if name == "epochs" and value == WEB_CREATE_DEFAULTS["epochs"]:
                raise ValueError("epochs is temporarily unavailable")
            return original(value, name)

        with patch.object(worker_module, "_positive_int", stricter):
            with self.assertRaisesRegex(ValueError, "epochs is temporarily unavailable"):
                worker_module.load_worker_options(directory, repo_root=repository)
            with self.assertRaisesRegex(ManagerError, "epochs is temporarily unavailable"):
                with patch.object(manager, "start_worker", lambda experiment_id: {"spawned": False}):
                    manager.create_experiment({**payload, "experiment_id": "shared_body_2"})
        self.assertFalse((self.experiments_root / "shared_body_2").exists())

    def test_an_unsatisfiable_gpu_request_is_refused_before_anything_is_written(self) -> None:
        """Create is coupled to GPU availability, exactly as closed source is.

        `_preflight` runs closed's `select_gpus(spec.gpu_count,
        require_name=spec.gpu_name_filter)` check, so an experiment can no
        longer be created against a host that cannot serve its default
        allocation. Both directions are asserted here so the coupling is
        pinned without the test depending on this host's devices.
        """
        from autotrade.environment.gpu import GpuUnavailableError

        with patch(
            "autotrade.environment.gpu.select_gpus",
            side_effect=GpuUnavailableError("requested 1 GPU(s), available matching GPUs: none"),
        ):
            refused = self._create()
        self.assertEqual(refused.status_code, 400, refused.text)
        self.assertIn("当前 GPU 无法满足实验默认分配", refused.json()["detail"])
        self._assert_nothing_was_created()
        time.sleep(2.0)
        self.assertFalse(self.marker.exists())
        with patch("autotrade.environment.gpu.select_gpus", return_value=[0]) as selector:
            allowed = self._create()
        self.assertEqual(allowed.status_code, 200, allowed.text)
        self.addCleanup(self._kill_pid, int(allowed.json()["spawned_pid"]))
        # Read off the resolved spec, not hardcoded: gpu.py is the single
        # configuration source for the device filter.
        selector.assert_called_once_with(1, require_name="L20")

    def _kill_pid(self, pid: int) -> None:
        try:
            os.killpg(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            try:
                os.kill(pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                return
        try:
            os.waitpid(pid, 0)
        except ChildProcessError:
            pass
