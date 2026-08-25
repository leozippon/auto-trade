"""Persistent local experiment lifecycle for the HITL console."""

from __future__ import annotations

import json
import os
import re
import shutil
import signal
import stat
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Callable, Mapping
from pathlib import Path

from autotrade.environment.broker import BrokerProfile
from autotrade.environment.runtime import chmod_tree, utc_now_iso, write_json_atomic
from autotrade.environment.strategy import StrategySchedule
from autotrade.pipelines import DailyStrategyPipeline, StrategyExperimentConfig
from autotrade.pipelines.agent_inbox import (
    INBOX_NAME,
    InboxError,
    enqueue_inbox_message,
)
from autotrade.pipelines.hitl_state import (
    LIVE_RUN_STATES,
    WEB_CLOSED_PARAMS,
    WEB_CREATE_DEFAULTS,
    WEB_INTERNAL_PARAMS,
    WEB_REQUIRED_PARAMS,
    ControlState,
    control_lock,
    proc_start_ticks,
    read_control,
    read_json,
    status_pid_alive,
    write_control,
)
from autotrade.pipelines.ledger import ExperimentLedger, latest_fold_records
from autotrade.pipelines.meta_schedule import meta_record_session_key

from .registry import experiment_state, heldout_complete, test_results_revealed

MAX_RUNNING_EXPERIMENTS = 5
_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,99}$")
_TERMINAL_RESUMABLE_STATES = (
    "stopped",
    "failed",
    "interrupted",
    "terminated",
    "created",
)
_ACTIONS = {
    "pause",
    "resume",
    "stop",
    "set_mode",
    "approve",
    "set_directive",
    "set_prompt_override",
    "set_gpu_count",
    "set_step_gate",
    "approve_step",
    "reply_question",
    "skip_to_heldout",
    "cancel_skip_to_heldout",
    "set_parent_override",
    "rollback_fold",
    "rerun_fold",
    "reveal_test_results",
    "restart",
    "terminate",
    "inject_message",
}
# Every control operation that could restart or steer learning after the
# Test/Held-out numbers are on screen. `resume` and `restart` belong here:
# putting the worker back on a sealed experiment continues development against
# results the researcher has already seen. Lifecycle-only controls
# (pause/stop/terminate/set_mode), the per-session GPU allocation and the
# reveal itself stay available.
_SEALED_BLOCKED_ACTIONS = frozenset(
    {
        "approve",
        "resume",
        "restart",
        "set_directive",
        "set_prompt_override",
        "set_step_gate",
        "approve_step",
        "reply_question",
        "set_parent_override",
        "skip_to_heldout",
        "cancel_skip_to_heldout",
        "rollback_fold",
        "rerun_fold",
        "inject_message",
    }
)


def _remove_sandbox_tree(path: Path) -> bool:
    """Remove a per-experiment sandbox dir, escalating through docker when
    plain rmtree leaves residue: under rootless docker the container agent's
    files map to a host subuid, so directories it created cannot be removed
    by the host user directly. A root-in-userns container maps those subuids
    and can delete them. Returns True when the tree is gone."""
    import shutil

    shutil.rmtree(path, ignore_errors=True)
    if not path.exists():
        return True
    try:
        from autotrade.environment.sandbox import DEFAULT_IMAGE

        subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "--user",
                "0",
                "--network=none",
                "-v",
                f"{path}:/purge",
                DEFAULT_IMAGE,
                "sh",
                "-c",
                "rm -rf /purge/* /purge/.[!.]* /purge/..?*",
            ],
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        pass
    shutil.rmtree(path, ignore_errors=True)
    return not path.exists()


def _remove_readonly_tree(path: Path) -> None:
    """Remove a host-owned tree containing frozen 0555 directories.

    On Unix, unlinking a file depends on its parent directory's write bit, not
    the file's own mode. Frozen PIT/revision files can also be hard-linked into
    shared caches, so changing file modes here would mutate every link. Make
    only directories inside ``path`` writable before ``shutil.rmtree`` performs
    its symlink-safe traversal.
    """

    def make_directories_writable(directory: Path) -> None:
        info = directory.lstat()
        if not stat.S_ISDIR(info.st_mode):
            return
        os.chmod(
            directory,
            stat.S_IMODE(info.st_mode) | stat.S_IRWXU,
            follow_symlinks=False,
        )
        with os.scandir(directory) as entries:
            children = [
                Path(entry.path)
                for entry in entries
                if entry.is_dir(follow_symlinks=False)
            ]
        for child in children:
            make_directories_writable(child)

    make_directories_writable(path)
    shutil.rmtree(path)
    if path.exists():
        raise OSError(f"experiment tree still exists after removal: {path}")


def _derived_sandbox_tree(repo_root: Path, experiment_id: str) -> Path | None:
    """Return the safe lexical per-experiment sandbox directory, if present."""

    try:
        repository = repo_root.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ManagerDeleteError(
            f"cannot validate repository root: {type(exc).__name__}: {exc}"
        ) from exc
    runtime_root = repository / ".runtime"
    sandbox_root = runtime_root / "sandboxes"
    for label, path in (("runtime root", runtime_root), ("sandbox root", sandbox_root)):
        try:
            info = path.lstat()
        except FileNotFoundError:
            return None
        except OSError as exc:
            raise ManagerDeleteError(
                f"cannot inspect {label} {path}: {type(exc).__name__}: {exc}"
            ) from exc
        if stat.S_ISLNK(info.st_mode):
            raise ManagerDeleteError(f"refusing to use symbolic-link {label}: {path}")
        if not stat.S_ISDIR(info.st_mode):
            raise ManagerDeleteError(f"{label} is not a directory: {path}")
    try:
        resolved_sandbox_root = sandbox_root.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ManagerDeleteError(
            f"cannot validate sandbox root: {type(exc).__name__}: {exc}"
        ) from exc
    if resolved_sandbox_root == repository or not resolved_sandbox_root.is_relative_to(
        repository
    ):
        raise ManagerDeleteError(
            f"sandbox root is outside the repository: {sandbox_root}"
        )
    expected = sandbox_root / experiment_id
    try:
        info = expected.lstat()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise ManagerDeleteError(
            f"cannot inspect sandbox path {expected}: {type(exc).__name__}: {exc}"
        ) from exc
    if stat.S_ISLNK(info.st_mode):
        raise ManagerDeleteError(
            f"refusing to delete sandbox symbolic link: {expected}"
        )
    if not stat.S_ISDIR(info.st_mode):
        raise ManagerDeleteError(f"sandbox path is not a directory: {expected}")
    try:
        resolved_parent = expected.parent.resolve(strict=True)
        resolved_expected = expected.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ManagerDeleteError(
            f"cannot validate sandbox path {expected}: {type(exc).__name__}: {exc}"
        ) from exc
    if (
        resolved_parent != resolved_sandbox_root
        or resolved_expected.parent != resolved_sandbox_root
        or resolved_expected.name != experiment_id
    ):
        raise ManagerDeleteError(f"refusing to delete unsafe sandbox path: {expected}")
    return expected


def _reclaim_sandbox_containers(experiment_id: str) -> list[str]:
    """Force-remove sandbox containers labelled for this experiment.

    A SIGKILLed worker skips its finally-block docker.stop(); the labels are
    set at container start (DockerSandbox). Best-effort: no docker on PATH or
    an empty listing simply reclaims nothing."""
    try:
        listing = subprocess.run(
            ["docker", "ps", "-aq", "--filter", f"label=adm.experiment={experiment_id}"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        containers = [
            line.strip() for line in listing.stdout.splitlines() if line.strip()
        ]
        if containers:
            subprocess.run(
                ["docker", "rm", "-f", *containers],
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
        return containers
    except (OSError, subprocess.SubprocessError):
        return []


def _signal_worker_group(pid: int, sig: signal.Signals) -> None:
    """Signal the worker's dedicated process group, falling back to its PID."""

    try:
        os.killpg(pid, sig)
    except (ProcessLookupError, PermissionError):
        os.kill(pid, sig)


def _require_session_reapproval(control: ControlState, session_key: str) -> bool:
    """Make a session wait at the existing editable approval gate."""

    changed = session_key in control.approved_sessions
    control.approved_sessions = tuple(
        key for key in control.approved_sessions if key != session_key
    )
    if control.mode == "auto":
        control.mode = "manual"
        changed = True
    return changed


class ManagerError(RuntimeError):
    pass


class ManagerDeleteError(ManagerError):
    """A terminal experiment could not be fully removed from local storage."""


class ExperimentManager:
    def __init__(
        self,
        repo_root: Path,
        experiments_root: Path | None = None,
        *,
        analysis_pending: Callable[[str], bool] | None = None,
    ) -> None:
        self.repo_root = Path(repo_root).resolve()
        self.experiments_root = Path(
            experiments_root or self.repo_root / "experiments"
        ).resolve()
        self.worker_script = (
            self.repo_root / "scripts/experiments/run_interactive_experiment.py"
        )
        # AnalysisService worker threads write into <experiment>/hitl/analysis/;
        # the server wires in the service's pending view so delete can refuse
        # while such a write may still be in flight. Standalone managers (tests,
        # scripts) have no background analyses to guard against.
        self._analysis_pending = analysis_pending
        self._mutate = threading.RLock()

    def run_experiment(self, params: Mapping[str, object]) -> dict[str, object]:
        """Legacy one-shot endpoint retained for simple daily experiments."""
        if "period" in params:
            raise ManagerError(
                "unknown experiment parameter: period; use strategy_period"
            )
        if str(params.get("data_backend") or "daily") != "daily":
            raise ManagerError("data_backend=pit requires a persistent experiment_id")
        if str(params.get("developer_mode") or "baseline") != "baseline":
            raise ManagerError("developer_mode=llm requires a persistent experiment_id")
        strategy_path = self._local_file(
            params.get("strategy_path"), label="strategy_path"
        )
        daily_path = self._local_file(params.get("daily_path"), label="daily_path")
        try:
            schedule = StrategySchedule(
                period=str(params.get("strategy_period") or "day"),  # type: ignore[arg-type]
                inference_time=str(params.get("inference_time") or "08:30"),
            )
            profile = BrokerProfile(
                initial_cash=float(params.get("initial_cash", 1_000_000))
            )
            config = StrategyExperimentConfig(
                strategy_path=strategy_path,
                schedule=schedule,
                broker_profile=profile,
                execution_mode=str(params.get("execution_mode") or "sandbox"),  # type: ignore[arg-type]
            )
            result = DailyStrategyPipeline(config).run(daily_path)
        except (TypeError, ValueError, OSError, RuntimeError) as exc:
            raise ManagerError(str(exc)) from exc
        return {"schedule": schedule.to_record(), "result": result.to_record()}

    def create_experiment(self, params: dict[str, object]) -> dict[str, object]:
        with self._mutate:
            console_managed = sorted(set(params) & WEB_CLOSED_PARAMS)
            if console_managed:
                raise ManagerError(
                    "console-managed parameters are not accepted: "
                    + ", ".join(console_managed)
                )
            unknown = sorted(set(params) - set(WEB_CREATE_DEFAULTS))
            if unknown:
                raise ManagerError(
                    "unknown experiment parameters: " + ", ".join(unknown)
                )
            merged = {**WEB_CREATE_DEFAULTS, **params}
            missing = sorted(
                key for key in WEB_REQUIRED_PARAMS if merged.get(key) in (None, "")
            )
            if missing:
                raise ManagerError(
                    "missing required experiment parameters: " + ", ".join(missing)
                )
            experiment_id = str(params.get("experiment_id") or "").strip()
            if not _ID.fullmatch(experiment_id):
                raise ManagerError(
                    "experiment_id must match [A-Za-z0-9][A-Za-z0-9_-]{0,99} (letters, digits, _ and -)"
                )
            directory = self.experiments_root / experiment_id
            if directory.exists():
                raise ManagerError(f"experiment {experiment_id!r} already exists")
            self._require_running_slot()
            merged.update(
                {
                    **WEB_INTERNAL_PARAMS,
                    "experiment_id": experiment_id,
                    "experiments_root": str(self.experiments_root),
                    "work_root": str(self.repo_root / ".runtime/sandboxes"),
                    "_creation_surface": "webui",
                }
            )
            self._preflight(merged, directory)
            hitl = directory / "hitl"
            hitl.mkdir(parents=True)
            inherit_from = str(merged.get("inherit_from") or "").strip()
            if inherit_from:
                try:
                    merged["_inherited_artifact"] = self._import_inherited_artifact(
                        directory, inherit_from
                    )
                except Exception:
                    shutil.rmtree(
                        directory, ignore_errors=True
                    )  # leave no half-created experiment
                    raise
            merged["_created_at"] = utc_now_iso()
            write_json_atomic(hitl / "params.json", merged)
            write_control(
                hitl / "control.json",
                ControlState(mode=str(merged["initial_control_mode"])),
            )
            write_json_atomic(
                hitl / "status.json", {"schema_version": 1, "state": "created"}
            )
            spawn = (
                self.start_worker(experiment_id)
                if self.worker_script.is_file()
                else {"spawned": False}
            )
            return {
                "experiment_id": experiment_id,
                "experiment_dir": str(directory),
                **spawn,
            }

    def _preflight(self, merged: dict[str, object], directory: Path) -> None:
        """Reject a bad create in the browser, not minutes later on disk.

        Runs the worker's own parameter validation and the GPU availability
        check BEFORE anything is written, so an invalid create is an HTTP 400
        the researcher can act on instead of a `failed` experiment they have to
        diagnose from status.json. `resolve_worker_options` is the same
        function the spawned worker calls, so the two cannot drift.
        """
        from autotrade.pipelines.worker import resolve_worker_options

        try:
            options = resolve_worker_options(
                merged,
                experiment_dir=directory,
                repo_root=self.repo_root,
                preflight=True,
            )
        except (TypeError, ValueError) as exc:
            raise ManagerError(str(exc)) from exc
        sandbox = options.agent_sandbox
        if sandbox is None or sandbox.gpu is None:
            return  # a CPU-only session allocates no device
        from autotrade.environment.gpu import GpuUnavailableError, select_gpus

        try:
            select_gpus(sandbox.gpu_count, require_name=sandbox.gpu_name_filter)
        except GpuUnavailableError as exc:
            raise ManagerError(f"当前 GPU 无法满足实验默认分配：{exc}") from exc

    def _import_inherited_artifact(
        self, experiment_dir: Path, source_id: str
    ) -> dict[str, object]:
        """Copy the source experiment's LATEST frozen fold output (+models) into
        the new experiment as a read-only snapshot, so the new experiment is
        self-contained even if the source is later deleted."""
        from autotrade.environment.artifacts import copy_artifact, copy_model_artifacts

        source_dir = self._experiment_dir(source_id)
        records = ExperimentLedger(
            source_dir / "ledgers/experiment_ledger.jsonl"
        ).read()
        folds = list(latest_fold_records(records).values())
        folds.sort(
            key=lambda row: (
                str(row.get("epoch_id")),
                str(row.get("test_period") or row.get("fold_id")),
            )
        )
        if not folds:
            raise ManagerError(
                f"源实验 {source_id!r} 没有已完成的 Fold，无法继承其 Agent Output"
            )
        record = folds[-1]
        source = Path(str(record.get("frozen_strategy_artifact_path") or ""))
        if not source.is_dir():
            raise ManagerError(f"源实验 {source_id!r} 的冻结产物目录缺失：{source}")
        artifact_id = f"strategy_inherited_{source_id}"
        dest_root = experiment_dir / "artifacts/strategy/_inherited"
        dest = dest_root / artifact_id
        dest_root.mkdir(parents=True, exist_ok=True)
        # copy_artifact validates the tree it copies (suffix allowlist, no
        # hidden/runtime cache files) and locks the copy read-only, so the
        # inherited parent cannot drift after creation.
        copy_artifact(source, dest)
        model_source = record.get("frozen_model_artifact_path")
        model_dest: Path | None = None
        if model_source and Path(str(model_source)).is_dir():
            model_dest = dest_root / f"{artifact_id}.models"
            copy_model_artifacts(Path(str(model_source)), model_dest)
            chmod_tree(model_dest, file_mode=0o444, dir_mode=0o555)
        chmod_tree(dest, file_mode=0o444, dir_mode=0o555)
        return {
            "artifact_id": artifact_id,
            "path": str(dest),
            "model_path": str(model_dest) if model_dest else None,
            "revision_id": str(record.get("frozen_strategy_artifact_id") or ""),
            "source_experiment_id": source_id,
            "source_epoch_id": record.get("epoch_id"),
            "source_fold_id": record.get("fold_id"),
            "source_artifact_id": record.get("frozen_strategy_artifact_id"),
        }

    def running_experiments(self) -> list[str]:
        if not self.experiments_root.is_dir():
            return []
        running: list[str] = []
        for directory in self.experiments_root.iterdir():
            if not directory.is_dir() or directory.name.startswith("."):
                continue
            status = _read_json(directory / "hitl/status.json")
            if status_pid_alive(status) and status.get("state") not in {
                "completed",
                "failed",
                "stopped",
            }:
                running.append(directory.name)
        return running

    def unreadable_experiments(self) -> list[dict[str, object]]:
        """Experiments whose hitl/status.json cannot be read (corrupt JSON or
        a foreign schema_version). They are excluded from every running roster
        by construction; /api/health surfaces them so a broken control plane
        degrades the reported status instead of hiding behind a green check."""
        broken: list[dict[str, object]] = []
        if not self.experiments_root.is_dir():
            return broken
        for entry in sorted(self.experiments_root.iterdir()):
            if not entry.is_dir() or entry.name.startswith("."):
                continue
            state = experiment_state(entry)
            if state.get("state") == "unreadable":
                broken.append(
                    {
                        "experiment_id": entry.name,
                        "error": str(state.get("error") or ""),
                    }
                )
        return broken

    def start_worker(self, experiment_id: str) -> dict[str, object]:
        with self._mutate:
            directory = self._experiment_dir(experiment_id)
            status_path = directory / "hitl/status.json"
            status = _read_json(status_path)
            if _worker_live(status):
                raise ManagerError(
                    f"experiment {experiment_id!r} already has a live worker"
                )
            self._require_running_slot()
            if not self.worker_script.is_file():
                raise ManagerError("interactive worker entrypoint is unavailable")
            # A stop request left behind by a previous run would immediately
            # re-stop the resumed worker; clear it (mode and approvals are
            # preserved).
            control_path = directory / "hitl/control.json"
            with control_lock(control_path):
                control = read_control(control_path)
                if control.request == "stop":
                    control.request = None
                    write_control(control_path, control)
            process = subprocess.Popen(
                [
                    sys.executable,
                    str(self.worker_script),
                    "--experiment-dir",
                    str(directory),
                ],
                cwd=str(self.repo_root),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            write_json_atomic(
                status_path,
                {
                    "schema_version": 1,
                    "state": "launching",
                    "pid": process.pid,
                    "pid_start_ticks": proc_start_ticks(process.pid),
                    "launched_at": utc_now_iso(),
                },
            )
            return {"spawned": True, "spawned_pid": process.pid}

    def _require_running_slot(self) -> None:
        running = self.running_experiments()
        if len(running) >= MAX_RUNNING_EXPERIMENTS:
            raise ManagerError(
                f"parallel experiment cap reached ({MAX_RUNNING_EXPERIMENTS}); "
                f"running: {', '.join(sorted(running))}"
            )

    def control(
        self,
        experiment_id: str,
        action: str,
        *,
        session_key: str | None = None,
        step_index: object = None,
        directive: str | None = None,
        mode: str | None = None,
        text: object = None,
        interrupt: object = False,
    ) -> dict[str, object]:
        if action not in _ACTIONS:
            raise ManagerError(f"unknown control action: {action!r}")
        with self._mutate:
            directory = self._experiment_dir(experiment_id)
            path = directory / "hitl/control.json"
            # Effective seal: manual reveal OR held-out completed
            # (auto-reveal). Reading only the control flag left every
            # auto-revealed experiment unsealed.
            if action in _SEALED_BLOCKED_ACTIONS and test_results_revealed(directory):
                raise ManagerError(
                    "测试结果已揭示，实验已封存：不能再进行影响后续学习的控制操作"
                )
            # The two worker-signalling actions wait out a SIGTERM grace, so
            # they must not hold control_lock: an exiting worker still consumes
            # its own session controls under that lock, and blocking it there
            # would turn a graceful shutdown into a forced kill.
            if action == "terminate":
                return self._terminate(experiment_id, directory)
            if action == "restart":
                return self._restart(experiment_id, directory)
            if action == "inject_message":
                return self._inject_message(
                    directory,
                    session_key=session_key,
                    text=text,
                    interrupt=interrupt,
                )
            with control_lock(path):
                control = read_control(path)
                self._apply_control_action(
                    directory,
                    control,
                    action=action,
                    session_key=session_key,
                    step_index=step_index,
                    directive=directive,
                    mode=mode,
                )
                write_control(path, control)
                response = {"control": control.to_record()}
            if action in {"resume", "rollback_fold", "rerun_fold"}:
                state = experiment_state(directory)
                if (
                    not state.get("worker_alive")
                    and state.get("state") in _TERMINAL_RESUMABLE_STATES
                    and self.worker_script.is_file()
                ):
                    return {**response, **self.start_worker(experiment_id)}
            return response

    def _inject_message(
        self,
        directory: Path,
        *,
        session_key: str | None,
        text: object,
        interrupt: object,
    ) -> dict[str, object]:
        if not isinstance(session_key, str) or not session_key.strip():
            raise ManagerError("inject_message requires session_key")
        session_key = session_key.strip()
        if interrupt is None:
            interrupt_flag = False
        elif isinstance(interrupt, bool):
            interrupt_flag = interrupt
        else:
            raise ManagerError("inject_message interrupt must be a boolean")
        state = experiment_state(directory)
        if not state.get("worker_alive"):
            raise ManagerError("inject_message requires a live worker")
        status = state.get("status")
        status_map = status if isinstance(status, Mapping) else {}
        run_state = str(status_map.get("state") or "")
        current = str(status_map.get("session_key") or "")
        if run_state not in LIVE_RUN_STATES or not current:
            raise ManagerError(
                "cannot inject_message into a finished or failed session"
            )
        if current != session_key:
            raise ManagerError(
                "inject_message session_key must match the current Agent session"
            )
        if isinstance(text, str):
            _reject_calendar_text(text)
        try:
            return enqueue_inbox_message(
                directory / "hitl" / INBOX_NAME,
                session_key=session_key,
                text=text,
                interrupt=interrupt_flag,
            )
        except InboxError as exc:
            raise ManagerError(str(exc)) from exc

    def _apply_control_action(
        self,
        directory: Path,
        control: ControlState,
        *,
        action: str,
        session_key: str | None,
        step_index: object,
        directive: str | None,
        mode: str | None,
    ) -> None:
        if action == "pause":
            control.request = "pause"
        elif action == "resume":
            control.request = None
        elif action == "stop":
            control.request = "stop"
        elif action == "reveal_test_results":
            control.test_revealed = True
        elif action == "set_mode":
            if mode not in {"auto", "manual", "step"}:
                raise ManagerError("set_mode requires mode auto|manual|step")
            control.mode = mode
        elif action == "approve":
            if not session_key:
                raise ManagerError("approve requires session_key")
            self._require_planned_session(directory, session_key)
            control.approved_sessions = tuple(
                dict.fromkeys([*control.approved_sessions, session_key])
            )
            if directive:
                _reject_calendar_text(directive)
                control.directives[session_key] = directive
        elif action == "approve_step":
            if not session_key:
                raise ManagerError("approve_step requires session_key")
            if directive is not None and not isinstance(directive, str):
                raise ManagerError("approve_step directive must be a string")
            self._require_planned_session(directory, session_key)
            status = read_json(directory / "hitl/status.json")
            if (
                status.get("state") != "waiting_step_user"
                or status.get("session_key") != session_key
                or type(step_index) is not int
                or status.get("step_index") != step_index
            ):
                raise ManagerError("approve_step must match the current waiting Step")
            current = int(control.step_go.get(session_key, 0))
            if current >= step_index:
                raise ManagerError("the current Step was already approved")
            control.step_go[session_key] = step_index
            directive_key = f"{session_key}#{step_index}"
            if directive in {None, ""}:
                control.step_directives.pop(directive_key, None)
            else:
                _reject_calendar_text(str(directive))
                control.step_directives[directive_key] = str(directive)
        elif action == "reply_question":
            if not session_key:
                raise ManagerError("reply_question requires session_key")
            if directive is not None and not isinstance(directive, str):
                raise ManagerError("reply_question directive must be a string")
            status = read_json(directory / "hitl/status.json")
            if (
                status.get("state") != "waiting_user_reply"
                or status.get("question_key") != session_key
            ):
                raise ManagerError("reply_question must match the current question key")
            self._require_planned_session(
                directory, str(status.get("session_key") or "")
            )
            if session_key in control.user_replies:
                raise ManagerError("the current question was already answered")
            reply = str(directive or "")
            _reject_calendar_text(reply)
            control.user_replies[session_key] = reply
        elif action == "set_gpu_count":
            if not session_key:
                raise ManagerError("set_gpu_count requires session_key")
            self._require_planned_session(directory, session_key)
            raw = str(directive or "").strip()
            if raw:
                try:
                    count = int(raw)
                except ValueError as exc:
                    raise ManagerError("GPU 数量必须是整数") from exc
                if not 0 <= count <= 4:
                    raise ManagerError("GPU 数量须在 0..4 之间")
                control.gpu_counts[session_key] = count
            else:
                control.gpu_counts.pop(session_key, None)
        elif action in {
            "set_directive",
            "set_prompt_override",
            "set_step_gate",
        }:
            if not session_key:
                raise ManagerError(f"{action} requires session_key")
            self._require_planned_session(directory, session_key.split("#", 1)[0])
            if action == "set_directive":
                target = control.directives
            elif action == "set_prompt_override":
                target = control.prompt_overrides
            elif action == "set_step_gate":
                if directive in {None, ""}:
                    control.step_gate.pop(session_key, None)
                else:
                    control.step_gate[session_key] = str(directive).lower() not in {
                        "0",
                        "false",
                        "off",
                    }
                target = None
            if target is not None:
                if directive in {None, ""}:
                    target.pop(session_key, None)
                else:
                    _reject_calendar_text(str(directive))
                    target[session_key] = str(directive)
        elif action == "skip_to_heldout":
            if not latest_fold_records(
                ExperimentLedger(directory / "ledgers/experiment_ledger.jsonl").read()
            ):
                raise ManagerError("尚无已完成的 Fold，无法提前进入 Held-out")
            control.skip_to_heldout = True
            control.request = None
        elif action == "cancel_skip_to_heldout":
            control.skip_to_heldout = False
        elif action == "set_parent_override":
            if not session_key:
                raise ManagerError("set_parent_override requires session_key")
            node_id = str(directive or "").strip()
            if node_id:
                self._validate_parent_override(directory, session_key, node_id)
                control.parent_overrides[session_key] = node_id
            else:
                control.parent_overrides.pop(session_key, None)
        elif action == "rollback_fold":
            if not session_key:
                raise ManagerError("rollback_fold requires session_key")
            state = experiment_state(directory)
            if state.get("worker_alive") or state.get("state") == "launching":
                raise ManagerError("先停止运行中的 worker（停止/强制终止）再回滚")
            self._rollback_to_fold(directory, session_key, control)
            control.request = None
            control.skip_to_heldout = False
        elif action == "rerun_fold":
            if not session_key:
                raise ManagerError("rerun_fold requires session_key")
            self._validate_rerun_target(directory, session_key)
            state = experiment_state(directory)
            if state.get("worker_alive") or state.get("state") == "launching":
                raise ManagerError(
                    "先停止运行中的 worker（停止/强制终止）再重跑该 Fold"
                )
            control.rerun_sessions[session_key] = uuid.uuid4().hex[:12]
            # The re-run must be re-approved (prompt edits land first) and its
            # step gating starts afresh: stale step_go would auto-release the
            # first N step holds, stale per-step directives would replay.
            # Dropping the approval alone only gates in manual/step mode —
            # _gate returns immediately under mode="auto" — so the helper also
            # falls back to manual, exactly as the terminate path does.
            _require_session_reapproval(control, session_key)
            control.step_go.pop(session_key, None)
            for mapping in (control.step_directives, control.user_replies):
                for key in list(mapping):
                    if key.split("#", 1)[0] == session_key:
                        mapping.pop(key, None)
            control.request = None
        else:
            raise ManagerError(f"unknown control action: {action!r}")

    def _rollback_to_fold(
        self, directory: Path, session_key: str, control: ControlState
    ) -> None:
        """Make ``session_key`` the experiment's frontier again.

        Drops every ledger record AFTER the target fold (later folds, later
        meta-learning sessions, and ALL held-out records — they reflect the
        discarded frontier), archives the dropped records' frozen artifact
        dirs (so resume neither trips the orphan check nor collides in
        _freeze), and backs up the original ledger next to it. The target
        fold's own records — including earlier re-runs — are kept verbatim.
        """
        sessions = self._planned_sessions(directory)
        fold_keys = [key for key, kind in sessions if kind == "fold"]
        if session_key not in fold_keys:
            raise ManagerError(f"{session_key!r} is not a fold session")
        target_epoch, _, target_fold = session_key.partition("/")
        records = ExperimentLedger(directory / "ledgers/experiment_ledger.jsonl").read()
        if (target_epoch, target_fold) not in latest_fold_records(records):
            raise ManagerError("目标 Fold 还没有账本记录，无法回滚到它")
        target_position = next(
            index for index, (key, _kind) in enumerate(sessions) if key == session_key
        )
        dropped_planned = sessions[target_position + 1 :]
        dropped_fold_keys = {key for key, kind in dropped_planned if kind == "fold"}
        dropped_meta_keys = {key for key, kind in dropped_planned if kind == "meta"}

        def _dropped(record: dict[str, object]) -> bool:
            kind = record.get("record_type")
            if kind == "fold":
                return (
                    f"{record.get('epoch_id')}/{record.get('fold_id')}"
                    in dropped_fold_keys
                )
            if kind == "meta_learning":
                return str(record.get("session_key") or "") in dropped_meta_keys
            return kind == "heldout"

        ledger_path = directory / "ledgers/experiment_ledger.jsonl"
        raw_lines = (
            ledger_path.read_text(encoding="utf-8").splitlines()
            if ledger_path.exists()
            else []
        )
        kept_lines: list[str] = []
        dropped_records: list[dict[str, object]] = []
        for line in raw_lines:
            text = line.strip()
            if not text:
                continue
            try:
                record = json.loads(text)
            except json.JSONDecodeError:
                kept_lines.append(
                    line
                )  # never silently discard unparseable audit lines
                continue
            if isinstance(record, dict) and _dropped(record):
                dropped_records.append(record)
            else:
                kept_lines.append(line)
        if not dropped_records:
            raise ManagerError(
                "该 Fold 之后没有任何账本记录（Fold/元学习/Held-out），无需回滚"
            )

        stamp = (
            utc_now_iso().replace("-", "").replace(":", "")[:15]
            + f"_{uuid.uuid4().hex[:8]}"
        )
        archive_root = directory / "artifacts/strategy/_archive" / f"rollback_{stamp}"
        backup = ledger_path.with_name(f"experiment_ledger.rollback_{stamp}.jsonl")
        shutil.copy2(ledger_path, backup)
        artifact_root = (directory / "artifacts").resolve()
        candidates: list[Path] = []
        for record in dropped_records:
            for field_name in (
                "frozen_strategy_artifact_path",
                "frozen_model_artifact_path",
            ):
                raw = record.get(field_name)
                if raw:
                    candidates.append(Path(str(raw)))
        for path in candidates:
            if not path.is_dir():
                continue
            resolved = path.resolve()
            if resolved == artifact_root or not resolved.is_relative_to(artifact_root):
                raise ManagerError(
                    f"账本中的冻结产物路径越出当前实验目录，拒绝回滚：{path}"
                )
        archived: list[str] = []
        for path in candidates:
            if not path.is_dir():
                continue
            archive_root.mkdir(parents=True, exist_ok=True)
            dest = archive_root / path.name
            suffix = 1
            while dest.exists():
                dest = archive_root / f"{path.name}.{suffix}"
                suffix += 1
            chmod_tree(path, file_mode=0o600, dir_mode=0o700)
            shutil.move(str(path), str(dest))
            archived.append(str(path))

        self._prune_step_tree(directory, dropped_fold_keys, archive_root)

        tmp = ledger_path.with_name(f".{ledger_path.name}.rollback.tmp")
        tmp.write_text("".join(f"{line}\n" for line in kept_lines), encoding="utf-8")
        os.replace(tmp, ledger_path)

        from autotrade.pipelines.prior import restore_current_from_records

        kept_records: list[dict[str, object]] = []
        for line in kept_lines:
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                kept_records.append(payload)
        try:
            restore_current_from_records(directory, kept_records)
        except (FileNotFoundError, ValueError) as exc:
            raise ManagerError(str(exc)) from exc

        dropped_session_keys = dropped_fold_keys | dropped_meta_keys | {"heldout"}
        control.approved_sessions = tuple(
            key for key in control.approved_sessions if key not in dropped_session_keys
        )
        # Session-scoped inputs of dropped sessions are stale by definition:
        # directives/prompt overrides describe runs that no longer exist, and
        # leftover step_go would auto-release the re-run's early step gates.
        for mapping in (
            control.directives,
            control.prompt_overrides,
            control.step_gate,
            control.step_go,
            control.rerun_sessions,
            control.parent_overrides,
            control.resource_overrides,
            control.gpu_counts,
        ):
            for key in list(mapping):
                if key in dropped_session_keys:
                    mapping.pop(key, None)
        for mapping in (control.step_directives, control.user_replies):
            for key in list(mapping):
                if key.split("#", 1)[0] in dropped_session_keys:
                    mapping.pop(key, None)

    def _prune_step_tree(
        self, directory: Path, dropped_fold_keys: set[str], archive_root: Path
    ) -> int:
        """Step-tree symmetry for fold rollback.

        Nodes recorded by the dropped fold sessions carry validation metrics and
        full strategy snapshots from periods that are FUTURE relative to the new
        frontier; the next fold's sandbox receives the experiment tree verbatim,
        so leaving them in place would hand the re-run Agent future-validated
        strategies. Dropped nodes (plus descendants) move into the rollback
        archive next to the frozen artifacts; tree.json is backed up there too."""
        from autotrade.environment.identity import agent_visible_ref
        from autotrade.environment.step_tree import TREE_FILE, StepTree

        steps_root = directory / "steps"
        if not (steps_root / TREE_FILE).exists():
            return 0
        dropped_pairs = set()
        for key in dropped_fold_keys:
            epoch_id, _, fold_id = key.partition("/")
            dropped_pairs.add((epoch_id, agent_visible_ref(fold_id, prefix="fold_ref")))
        tree = StepTree(steps_root)
        dropped_ids = {
            str(node["node_id"])
            for node in tree.nodes()
            if (str(node.get("epoch_id")), str(node.get("fold_id"))) in dropped_pairs
        }
        if not dropped_ids:
            return 0
        changed = True
        while changed:  # descendants of a dropped node are dropped too
            changed = False
            for node in tree.nodes():
                if (
                    node["node_id"] not in dropped_ids
                    and node.get("parent_node_id") in dropped_ids
                ):
                    dropped_ids.add(str(node["node_id"]))
                    changed = True
        archive_steps = archive_root / "steps"
        archive_steps.mkdir(parents=True, exist_ok=True)
        shutil.copy2(steps_root / TREE_FILE, archive_steps / TREE_FILE)
        for node_id in sorted(dropped_ids):
            node_dir = steps_root / node_id
            if node_dir.is_dir():
                chmod_tree(node_dir, file_mode=0o600, dir_mode=0o700)
                shutil.move(str(node_dir), str(archive_steps / node_id))
        tree.data["nodes"] = [
            node for node in tree.nodes() if str(node["node_id"]) not in dropped_ids
        ]
        if tree.data.get("current_node_id") in dropped_ids:
            tree.data["current_node_id"] = None
        tree.save()
        return len(dropped_ids)

    def _validate_rerun_target(self, directory: Path, session_key: str) -> None:
        """Only the LATEST recorded fold may be re-run: earlier folds already
        fed their frozen artifacts into successors, so re-running them would
        break the parent chain the later records were built on."""
        sessions = self._planned_sessions(directory)
        fold_keys = [key for key, kind in sessions if kind == "fold"]
        if session_key not in fold_keys:
            raise ManagerError(f"{session_key!r} is not a fold session")
        records = ExperimentLedger(directory / "ledgers/experiment_ledger.jsonl").read()
        recorded = latest_fold_records(records)
        recorded_keys = [
            key for key in fold_keys if tuple(key.split("/", 1)) in recorded
        ]
        if not recorded_keys:
            raise ManagerError("该实验还没有已完成的 Fold 可重跑")
        if session_key != recorded_keys[-1]:
            raise ManagerError(
                f"只能重跑最新完成的 Fold（{recorded_keys[-1]}）——更早的 Fold 已被后续继承"
            )
        target_position = next(
            index for index, (key, _kind) in enumerate(sessions) if key == session_key
        )
        recorded_meta_keys = {
            str(record.get("session_key") or "")
            for record in records
            if record.get("record_type") == "meta_learning"
        }
        later_meta = next(
            (
                key
                for key, kind in sessions[target_position + 1 :]
                if kind == "meta" and key in recorded_meta_keys
            ),
            None,
        )
        if later_meta is not None:
            raise ManagerError(
                f"后续元学习会话 {later_meta} 已继承该 Fold；请先回滚到目标 Fold 再重跑"
            )

    def _validate_parent_override(
        self, directory: Path, session_key: str, node_id: str
    ) -> None:
        """The override target must be a fold session and the node a restorable snapshot.

        Past-only: a node recorded by a LATER fold session embodies strategies
        validated on periods after the target session's window; allowing it as
        the parent would leak future-fitted strategies backwards. The node's own
        session (rerun-from-node) and earlier sessions are allowed. Which fold
        may consume it is enforced where it matters: an already-run fold only
        picks the override up through rerun_fold (itself restricted to the
        latest fold), an unrun fold at its next start."""
        from autotrade.environment.step_tree import StepTree
        from autotrade.pipelines.hitl_state import assert_node_not_from_later_fold

        from .steps import node_export_dir

        fold_keys = [
            key for key, kind in self._planned_sessions(directory) if kind == "fold"
        ]
        if session_key not in fold_keys:
            raise ManagerError(f"{session_key!r} is not a fold session")
        try:
            node_export_dir(directory, node_id)
        except ValueError as exc:
            raise ManagerError(str(exc)) from exc
        node = StepTree(directory / "steps").get_node(node_id)
        try:
            assert_node_not_from_later_fold(node, session_key, fold_keys)
        except ValueError as exc:
            raise ManagerError(str(exc)) from exc

    @staticmethod
    def _planned_sessions(directory: Path) -> list[tuple[str, str]]:
        schedule = read_json(directory / "hitl/schedule.json")
        raw = schedule.get("sessions")
        if not isinstance(raw, list):
            raise ManagerError("experiment session plan is missing")
        sessions: list[tuple[str, str]] = []
        for item in raw:
            if not isinstance(item, dict):
                raise ManagerError("experiment session plan is invalid")
            key = str(item.get("session_key") or item.get("key") or "")
            kind = str(item.get("kind") or "")
            if kind == "meta_learning":
                kind = "meta"
            if not key or kind not in {"fold", "meta", "heldout"}:
                raise ManagerError(
                    "experiment session plan contains an invalid session"
                )
            sessions.append((key, kind))
        if len({key for key, _kind in sessions}) != len(sessions):
            raise ManagerError("experiment session plan contains duplicate keys")
        return sessions

    def _require_planned_session(self, directory: Path, session_key: str) -> None:
        if session_key == "heldout":
            return
        if session_key not in {key for key, _kind in self._planned_sessions(directory)}:
            raise ManagerError(f"unknown session: {session_key}")

    def _session_is_settled(
        self, directory: Path, session_key: str, control: ControlState
    ) -> bool:
        """Whether the scheduled session has a durable successful ledger record."""
        try:
            kind = dict(self._planned_sessions(directory)).get(session_key)
        except ManagerError:
            # An unreadable plan cannot prove the session finished; treat it as
            # unsettled so the interrupted session returns to its gate.
            return False
        if kind is None:
            return False
        records = ExperimentLedger(directory / "ledgers/experiment_ledger.jsonl").read()
        if kind == "meta":
            return any(
                record.get("record_type") == "meta_learning"
                and meta_record_session_key(record) == session_key
                for record in records
            )
        if kind == "fold":
            epoch_id, _, fold_id = session_key.partition("/")
            matching = [
                record
                for record in records
                if record.get("record_type") == "fold"
                and str(record.get("epoch_id")) == epoch_id
                and str(record.get("fold_id")) == fold_id
            ]
            if not matching:
                return False
            rerun_id = control.rerun_sessions.get(session_key)
            return (
                rerun_id is None or str(matching[-1].get("rerun_id") or "") == rerun_id
            )
        # Single source for "all planned held-out periods are recorded";
        # registry.heldout_complete also correctly treats an (impossible)
        # period-less held-out session as not settled.
        return heldout_complete(directory, records)

    def _revoke_unsettled_session_approval(
        self, directory: Path, session_key: str
    ) -> str | None:
        """Return an interrupted session to its editable approval gate."""
        if not session_key:
            return None
        path = directory / "hitl/control.json"
        with control_lock(path):
            control = read_control(path)
            if self._session_is_settled(directory, session_key, control):
                return None
            if not _require_session_reapproval(control, session_key):
                return None
            write_control(path, control)
        return session_key

    def _terminate(self, experiment_id: str, directory: Path) -> dict[str, object]:
        """Graceful first, then guaranteed: the worker's SIGTERM handler unwinds
        through finally blocks, but blocking work (LLM retries, derived-image
        docker build) can ignore it for a long time. After a short grace,
        SIGKILL the whole process group (the worker runs with
        start_new_session=True)."""
        status_path = directory / "hitl/status.json"
        status = _read_json(status_path)
        if not status_pid_alive(status):
            raise ManagerError("no live worker to terminate")
        pid = int(status["pid"])
        session_key = str(status.get("session_key") or "")
        try:
            _signal_worker_group(pid, signal.SIGTERM)
        except ProcessLookupError as exc:  # exited between check and signal
            raise ManagerError("worker 已退出") from exc
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            if not status_pid_alive(_read_json(status_path)):
                result: dict[str, object] = {
                    "terminated_pid": pid,
                    "escalated": False,
                    "reclaimed_containers": _reclaim_sandbox_containers(experiment_id),
                }
                revoked = self._revoke_unsettled_session_approval(
                    directory, session_key
                )
                if revoked is not None:
                    result["approval_revoked_session"] = revoked
                return result
            time.sleep(0.5)
        _signal_worker_group(pid, signal.SIGKILL)
        reclaimed = _reclaim_sandbox_containers(experiment_id)
        # SIGKILL leaves no worker to stamp a terminal state; without this the
        # page shows a stale running state until pid-liveness kicks in and the
        # user cannot tell whether termination worked.
        status = _read_json(status_path)
        status.update(
            {"state": "terminated", "error": None, "terminated_at": utc_now_iso()}
        )
        write_json_atomic(status_path, status)
        result = {
            "terminated_pid": pid,
            "escalated": True,
            "reclaimed_containers": reclaimed,
        }
        revoked = self._revoke_unsettled_session_approval(directory, session_key)
        if revoked is not None:
            result["approval_revoked_session"] = revoked
        return result

    def _restart(self, experiment_id: str, directory: Path) -> dict[str, object]:
        """Terminate-and-restart in one step: SIGTERM the live worker, wait for
        the pid to die (bounded), then resume via the ledger."""
        status_path = directory / "hitl/status.json"
        status = _read_json(status_path)
        if status_pid_alive(status):
            _signal_worker_group(int(status["pid"]), signal.SIGTERM)
            deadline = time.monotonic() + 30.0
            while time.monotonic() < deadline and status_pid_alive(
                _read_json(status_path)
            ):
                time.sleep(0.5)
            if status_pid_alive(_read_json(status_path)):
                raise ManagerError(
                    "worker 未在 30s 内退出；请稍后重试或强制终止后手动恢复"
                )
        _reclaim_sandbox_containers(experiment_id)
        return {"restarted": True, **self.start_worker(experiment_id)}

    def delete_experiment(self, experiment_id: str) -> dict[str, object]:
        with self._mutate:
            directory = self._experiment_dir(experiment_id)
            state = experiment_state(directory)
            if state.get("worker_alive") or state.get("state") == "launching":
                raise ManagerError(
                    f"experiment {experiment_id!r} has a live worker; stop or terminate it before deleting"
                )
            # AnalysisService background threads write into hitl/analysis/ after
            # the HTTP request returns; rmtree under a live writer would race it.
            if self._analysis_pending is not None and self._analysis_pending(
                experiment_id
            ):
                raise ManagerError(
                    f"experiment {experiment_id!r} has a strategy analysis in progress; "
                    "wait for it to finish before deleting"
                )
            removed_work_root: str | None = None
            # Deletion does not depend on params.json being readable: a failed
            # worker can leave it corrupt or inaccessible. The lenient reader
            # falls back to the deployment-derived sandbox path and, crucially,
            # never adopts an unreadable file's explicit work_root.
            params = _read_json(directory / "hitl/params.json")
            work_root = params.get("work_root")
            # The per-experiment sandbox dir is derived from the experiment id, so
            # it is removed even when params.json is unreadable; an explicit
            # work_root is honored only when it IS that dir (never a shared root).
            expected = _derived_sandbox_tree(self.repo_root, experiment_id)
            if expected is not None:
                try:
                    work_path = (
                        Path(str(work_root)).resolve() if work_root else expected
                    )
                except (OSError, RuntimeError) as exc:
                    raise ManagerDeleteError(
                        f"cannot validate configured sandbox path: {type(exc).__name__}: {exc}"
                    ) from exc
                if work_path == expected:
                    if not _remove_sandbox_tree(expected):
                        raise ManagerDeleteError(
                            f"sandbox 目录未能完全删除：{expected}"
                        )
                    removed_work_root = str(expected)
            try:
                _remove_readonly_tree(directory)
            except OSError as exc:
                raise ManagerDeleteError(
                    f"experiment {experiment_id!r} was not fully deleted: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            _reclaim_sandbox_containers(experiment_id)
            return {"deleted": experiment_id, "removed_work_root": removed_work_root}

    def _experiment_dir(self, experiment_id: str) -> Path:
        if not _ID.fullmatch(experiment_id):
            raise ManagerError("invalid experiment ID")
        candidate = self.experiments_root / experiment_id
        if candidate.is_symlink():
            raise ManagerError("invalid experiment ID")
        directory = candidate.resolve()
        if (
            not directory.is_relative_to(self.experiments_root)
            or not directory.is_dir()
        ):
            raise ManagerError(f"unknown experiment: {experiment_id}")
        return directory

    def _local_file(self, raw: object, *, label: str) -> Path:
        text = str(raw or "").strip()
        if not text:
            raise ManagerError(f"{label} is required")
        candidate = Path(text)
        path = (
            (self.repo_root / candidate).resolve()
            if not candidate.is_absolute()
            else candidate.resolve()
        )
        if not path.is_relative_to(self.repo_root):
            raise ManagerError(f"{label} must be inside the local repository")
        if not path.is_file():
            raise ManagerError(
                f"{label} does not exist: {path.relative_to(self.repo_root)}"
            )
        return path


def _reject_calendar_text(text: str) -> None:
    if not text.strip():
        return
    from autotrade.agent.runner import calendar_policy_violation

    reason = calendar_policy_violation(text)
    if reason:
        raise ManagerError(f"指令含有不可迁移的日历日期：{reason}")


def _read_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _worker_live(status: Mapping[str, object]) -> bool:
    return str(status.get("state") or "") not in {
        "completed",
        "failed",
        "stopped",
        "terminated",
        "interrupted",
    } and status_pid_alive(status)
