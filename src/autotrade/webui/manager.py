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
from collections.abc import Mapping
from pathlib import Path

from autotrade.environment.identity import (
    LEGACY_EXPERIMENT_MESSAGE,
    AgentRefStore,
    LegacyExperimentError,
)
from autotrade.environment.llm.model_profiles import (
    LOCAL_QWEN_MODEL,
    canonicalize_model_name,
)
from autotrade.environment.runtime import utc_now_iso, write_json_atomic
from autotrade.environment.sandbox import EXPERIMENT_LABEL, SandboxSpec
from autotrade.environment.sandbox_images import reclaim_experiment_sandbox_images
from autotrade.environment.tools.workspace import WORKSPACE_MIN_FREE_BYTES
from autotrade.pipelines.agent_inbox import (
    INBOX_NAME,
    InboxError,
    enqueue_inbox_message,
)
from autotrade.pipelines.config import acceptance_for
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
    read_gpu_claim,
    status_pid_alive,
    write_control,
    write_gpu_claim,
)
from autotrade.pipelines.ledger import research_over
from autotrade.pipelines.lineage import extract_lineage, lineage_arm_ids, write_lineage
from autotrade.pipelines.skills import create_operating_memory_snapshot

from .public_identity import PublicIdentity
from .registry import experiment_state, read_ledger_records, worker_log_ref

# Running-arm limits, one rule for create and resume: a start is refused when
# MAX_RUNNING_EXPERIMENTS arms already run, or when the arm is a local one
# (uses_local_model) and MAX_RUNNING_LOCAL_EXPERIMENTS local arms already run.
# Runner memory does not bind (a runner holds 1-3 GiB since evaluation slots
# stream instead of decoding whole, and an arm's replays run in containers
# capped at 16 GiB). The local limit protects the shared local model service:
# its aggregate generation throughput levels off at about 8 concurrent
# requests (~150 tok/s), while four parent conversations with their sub-agent
# fan-out (at most 4 concurrent each) run 3-5 requests at a time (p50/p90) and
# use about half of that. An arm whose main session and sub-agents are hosted
# leaves that service almost idle, so only the total limit, which protects the
# host's page cache and IO, applies to it. Measured rationale:
# docs/deployment-documentation.md. GPUs are not a count: a GPU arm starts only
# on whole cards nobody else uses or holds, and keeps them while its worker
# lives (gpu_slots).
MAX_RUNNING_EXPERIMENTS = 8
MAX_RUNNING_LOCAL_EXPERIMENTS = 4
# What one new arm writes of its own before its first validation: the PIT views
# and as-of stash its seed does not carry, measured at 0.5-2.5 GiB of unique
# bytes per arm. A create is refused unless the experiments filesystem keeps the
# Agent tools' free-space floor after that, so an arm never builds its views
# only to have its shell and batch_validate calls refused below the floor.
ARM_DISK_MARGIN_BYTES = 3 * 1024**3
# SIGTERM graces before the worker's process group is SIGKILLed. Terminate is
# an explicit stop, so it stays short; restart has to outwait the in-flight
# work a worker cannot interrupt (a model call runs minutes) before forcing it.
_TERMINATE_GRACE_SECONDS = 10.0
_RESTART_GRACE_SECONDS = 30.0
_SIGKILL_EXIT_SECONDS = 5.0
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
    "set_directive",
    "set_gpu_count",
    "restart",
    "terminate",
    "inject_message",
}


def _remove_sandbox_tree(path: Path) -> bool:
    from autotrade.environment.sandbox import remove_sandbox_tree

    return remove_sandbox_tree(path)


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

    A SIGKILLed worker skips its finally-block docker.stop(); the label is
    set at container start on the session container and on every strategy
    container (``experiment_container_labels``). Best-effort: no docker on
    PATH or an empty listing simply reclaims nothing."""
    try:
        listing = subprocess.run(
            ["docker", "ps", "-aq", "--filter", f"label={EXPERIMENT_LABEL}={experiment_id}"],
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


def _await_worker_exit(status_path: Path, timeout: float) -> bool:
    """Poll the recorded worker pid until it is gone, or `timeout` elapses."""

    deadline = time.monotonic() + timeout
    while True:
        if not status_pid_alive(_read_json(status_path)):
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.5)


def _signal_worker_group(pid: int, sig: signal.Signals) -> None:
    """Signal the worker's dedicated process group, falling back to its PID."""

    try:
        os.killpg(pid, sig)
    except (ProcessLookupError, PermissionError):
        os.kill(pid, sig)


def uses_local_model(params: Mapping[str, object]) -> bool:
    """Whether an arm counts against MAX_RUNNING_LOCAL_EXPERIMENTS: its main
    session or its sub-agents run on the local model.

    Read the way the registry reads an arm's effective configuration, its
    params over the create defaults. A role left empty counts as local: the
    worker runs an empty main role on the local model and an empty sub-agent
    role on the main model, so this can only over-count the local service's
    load.
    """

    effective = {**WEB_CREATE_DEFAULTS, **params}
    return any(
        canonicalize_model_name(str(effective.get(role) or LOCAL_QWEN_MODEL))
        == LOCAL_QWEN_MODEL
        for role in ("model", "subagent_model")
    )


def gpu_request(params: Mapping[str, object]) -> int:
    """The GPUs an arm's worker attaches, read like :func:`uses_local_model`:
    its params over the create defaults, so an unreadable params.json counts
    as the default request."""

    return int({**WEB_CREATE_DEFAULTS, **params}.get("gpu_count") or 0)


class ManagerError(RuntimeError):
    pass


class ManagerDeleteError(ManagerError):
    """A terminal experiment could not be fully removed from local storage."""


def _modern_ref_store(directory: Path) -> AgentRefStore:
    try:
        return AgentRefStore(directory)
    except LegacyExperimentError as exc:
        raise ManagerError(LEGACY_EXPERIMENT_MESSAGE) from exc
    except ValueError as exc:
        raise ManagerError("experiment identity state is unreadable") from exc


class ExperimentManager:
    def __init__(
        self,
        repo_root: Path,
        experiments_root: Path | None = None,
    ) -> None:
        self.repo_root = Path(repo_root).resolve()
        self.experiments_root = Path(
            experiments_root or self.repo_root / "experiments"
        ).resolve()
        self.worker_script = (
            self.repo_root / "scripts/experiments/run_interactive_experiment.py"
        )
        # One mutation lock per experiment, not one for the console. Mutating
        # calls that touch the SAME experiment's worker, control state or
        # directory still serialize, but terminate and restart wait out a
        # SIGTERM grace (10 s / 35 s) and delete can spend minutes escalating a
        # stuck rmtree through docker; under a console-wide lock every one of
        # those waits blocked every other experiment's create, start, pause,
        # resume and set_directive.
        self._experiment_locks: dict[str, threading.RLock] = {}
        self._locks_guard = threading.Lock()
        # The one genuinely console-wide invariant left: the running-arm
        # limits. A holder keeps it from the running-slot count through the
        # spawn that consumes the slot, so two callers can never claim the same
        # one: create_experiment holds it across its pre-flight and the
        # operating-memory snapshot as well, and restart holds it across the
        # terminate that frees the slot it is about to retake. Creates and starts therefore wait for each other, and for a
        # restart's SIGTERM grace; control actions on other experiments never
        # take it. Reentrant because both paths end in the start_worker that
        # takes it again on the same thread.
        self._slots = threading.RLock()

    def _experiment_lock(self, experiment_id: str) -> threading.RLock:
        """The mutation lock for one experiment, created on first use.

        Keyed by the validated id rather than by the resolved directory, so
        the lock is taken before the directory is read: a delete racing a
        control call on the same experiment still resolves in one order.
        """
        if not _ID.fullmatch(experiment_id):
            raise ManagerError("invalid experiment ID")
        with self._locks_guard:
            lock = self._experiment_locks.get(experiment_id)
            if lock is None:
                lock = self._experiment_locks[experiment_id] = threading.RLock()
            return lock

    def create_experiment(self, params: dict[str, object]) -> dict[str, object]:
        # Request-shape validation is a pure function of params and needs no
        # lock; only the id has to be valid before one can be taken.
        console_managed = sorted(set(params) & WEB_CLOSED_PARAMS)
        if console_managed:
            raise ManagerError(
                "console-managed parameters are not accepted: "
                + ", ".join(console_managed)
            )
        unknown = sorted(set(params) - set(WEB_CREATE_DEFAULTS))
        if unknown:
            raise ManagerError("unknown experiment parameters: " + ", ".join(unknown))
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
        with self._experiment_lock(experiment_id), self._slots:
            directory = self.experiments_root / experiment_id
            if directory.exists():
                raise ManagerError(f"experiment {experiment_id!r} already exists")
            self._require_running_slot(local=uses_local_model(merged))
            self._require_free_space()
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
            _modern_ref_store(directory)
            hitl = directory / "hitl"
            hitl.mkdir(parents=True)
            # Operating memory is fixed for the life of the experiment: resolve
            # the library and the graduated tier once, here, and copy the result
            # in read-only. Every session then
            # mounts that snapshot, so a library change reaches the next
            # experiment instead of the middle of this one.
            try:
                create_operating_memory_snapshot(
                    directory,
                    mode=str(merged.get("operating_memory") or ""),
                    repo_root=self.repo_root,
                    experiments_root=self.experiments_root,
                )
                # A lineage is read from the earlier arms here, once, into a
                # file beside this arm's ledger; nothing reads them again.
                if merged.get("lineage_arms"):
                    write_lineage(
                        directory,
                        extract_lineage(
                            self.experiments_root,
                            lineage_arm_ids(merged["lineage_arms"], experiment_id),
                            research_start=str(merged["research_start"]),
                            research_end=str(merged["research_end"]),
                        ),
                    )
            except Exception:
                self._discard_half_created(directory)
                raise
            # The acceptance rules are resolved once, here, and stamped in
            # like every other default this table merges: an arm's gates then
            # live in its own params.json, and a later change to the defaults
            # or to the beta band that travels with a cap cannot move an arm
            # that has already been created.
            merged.update(acceptance_for(merged).to_record())
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
            return {"experiment_id": experiment_id, **spawn}

    def _discard_half_created(self, directory: Path) -> None:
        """Leave no half-created experiment, and never mask why it failed.

        The tree can already hold read-only copies (the operating-memory
        snapshot), which a plain ``rmtree`` leaves behind.
        """

        try:
            _remove_readonly_tree(directory)
        except OSError:
            shutil.rmtree(directory, ignore_errors=True)

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
        self._claimable_gpus(sandbox.gpu_count)

    def gpu_slots(self) -> dict[str, object]:
        """The cards a GPU arm starting now would take, and what running arms hold.

        A running GPU arm holds the devices claimed for its worker at start
        (``hitl/gpu_claim.json``) for as long as that worker lives, whether
        or not they show memory in use: between batches, while its model
        thinks, they show none. A card is free when it matches the sandbox
        device filter, no process holds memory on it (``gpu.idle_gpus``) and
        no running arm claims it; ``gpus_free`` lists the free cards in the
        order a start takes them. ``gpu_claims`` maps every running GPU arm to
        its devices, ``None`` for one without a claim (no ``gpu_claim.json``:
        started before claims existed, or by hand). While any such arm runs no
        card is free: its legs pick the card with the most free memory at
        every start, which is exactly a claimed card its owner is not using
        yet, and two of them beside the owner's three overfill it. The rule
        needs no removal: once no unclaimed arm runs it never applies. When
        the devices cannot be read no card is free and ``gpu_error`` says why.
        """
        from autotrade.environment.gpu import GpuUnavailableError, idle_gpus

        claims: dict[str, list[int] | None] = {}
        for name in sorted(self.running_experiments()):
            hitl = self.experiments_root / name / "hitl"
            if gpu_request(_read_json(hitl / "params.json")):
                claim = read_gpu_claim(hitl)
                claims[name] = None if claim is None else list(claim)
        if None in claims.values():
            return {"gpus_free": [], "gpu_claims": claims}
        claimed = {device for devices in claims.values() for device in devices or ()}
        try:
            idle = idle_gpus(require_name=SandboxSpec().gpu_name_filter)
        except GpuUnavailableError as exc:
            return {"gpus_free": [], "gpu_claims": claims, "gpu_error": str(exc)}
        free = [device for device in idle if device not in claimed]
        return {"gpus_free": free, "gpu_claims": claims}

    def _claimable_gpus(self, count: int) -> list[int]:
        """The ``count`` cards a GPU arm starting now takes, or a refusal
        naming the cards in use and the running arms that hold the rest."""

        slots = self.gpu_slots()
        free = list(slots["gpus_free"])  # type: ignore[call-overload]
        if len(free) >= count:
            return free[:count]
        claims = slots["gpu_claims"]
        unclaimed = [name for name, devices in claims.items() if devices is None]  # type: ignore[attr-defined]
        holders = "; ".join(
            f"{name} {devices}"
            for name, devices in claims.items()  # type: ignore[attr-defined]
            if devices is not None
        )
        reason = (
            "no card is claimable while running GPU arms without a claim pick cards "
            f"themselves: {', '.join(unclaimed)}"
            if unclaimed
            else "a card is free only when nothing holds memory on it and no running "
            "arm claims it"
        )
        raise ManagerError(
            f"当前 GPU 无法满足实验默认分配：requested {count} GPU(s), {len(free)} free "
            f"({', '.join(map(str, free)) or 'none'}); {reason}; "
            f"claimed by running arms: {holders or 'none'}"
            + (f"; GPUs unreadable: {slots['gpu_error']}" if "gpu_error" in slots else "")
        )

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

    def _running_roster(self) -> tuple[list[str], list[str]]:
        """Every running arm, and those of them that count against the local
        limit. An arm whose params.json cannot be read counts as local
        (:func:`uses_local_model` over the defaults)."""

        running = self.running_experiments()
        return running, [
            name
            for name in running
            if uses_local_model(
                _read_json(self.experiments_root / name / "hitl/params.json")
            )
        ]

    def running_slots(self) -> dict[str, object]:
        """Both limits and the running roster, as the console reports them."""

        running, running_local = self._running_roster()
        return {
            "max_running_experiments": MAX_RUNNING_EXPERIMENTS,
            "max_running_local_experiments": MAX_RUNNING_LOCAL_EXPERIMENTS,
            "running": running,
            "running_local": running_local,
        }

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
        # Slot lock as well as the experiment's own: the spawn below is what
        # consumes a running slot, and it only becomes visible to
        # running_experiments() once the status write at the end lands.
        with self._experiment_lock(experiment_id), self._slots:
            directory = self._experiment_dir(experiment_id)
            _modern_ref_store(directory)
            status_path = directory / "hitl/status.json"
            status = _read_json(status_path)
            if _worker_live(status):
                raise ManagerError(
                    f"experiment {experiment_id!r} already has a live worker"
                )
            params = _read_json(directory / "hitl/params.json")
            self._require_running_slot(local=uses_local_model(params))
            if not self.worker_script.is_file():
                raise ManagerError("interactive worker entrypoint is unavailable")
            # Claimed before anything else is touched, so a refusal leaves the
            # arm as it was, and written before the spawn, which reads it.
            gpus = gpu_request(params)
            if gpus:
                write_gpu_claim(directory / "hitl", self._claimable_gpus(gpus))
            # A stop request left behind by a previous run would immediately
            # re-stop the resumed worker; clear it (the session directives are
            # preserved).
            control_path = directory / "hitl/control.json"
            with control_lock(control_path):
                control = read_control(control_path)
                if control.request == "stop" or control.restart_pending:
                    control.request = None
                    # A deferred restart belongs to the worker that was
                    # running when it was asked for; this one is already new.
                    control.restart_pending = False
                    write_control(control_path, control)
            # A worker that dies on an unhandled traceback used to write it to
            # /dev/null, leaving the run unexplainable. Append both streams to a
            # per-experiment file under the ignored logs/ tree instead; the
            # parent closes its handle immediately, the detached child keeps it.
            log_ref = worker_log_ref(experiment_id)
            log_path = self.repo_root / log_ref
            log_path.parent.mkdir(parents=True, exist_ok=True)
            # The worker's temporary files (every replay's strategy state dir,
            # the frozen-tree copy around an official evaluation, library
            # defaults) live on the repository volume, not the host's shared
            # /tmp tmpfs: other tenants fill that one, and a replay then failed
            # with ENOSPC creating its state dir. The directory is this
            # experiment's alone and no worker of it is live, so whatever a
            # killed worker left behind is removed here.
            temp_dir = self.repo_root / ".runtime/tmp" / experiment_id
            if temp_dir.exists():
                _remove_readonly_tree(temp_dir)
            temp_dir.mkdir(parents=True)
            with log_path.open("a", encoding="utf-8") as log:
                log.write(f"\n===== worker start {utc_now_iso()} =====\n")
                log.flush()
                process = subprocess.Popen(
                    [
                        sys.executable,
                        str(self.worker_script),
                        "--experiment-dir",
                        str(directory),
                    ],
                    cwd=str(self.repo_root),
                    env={**os.environ, "TMPDIR": str(temp_dir)},
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
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
                    # Repo-relative only: status.json is projected to the API and
                    # the console, and the public boundary never carries a host
                    # path. It is a plain location, not an opaque ref.
                    "worker_log": log_ref,
                },
            )
            return {
                "spawned": True,
                "spawned_pid": process.pid,
                "worker_log": log_ref,
            }

    def _require_running_slot(self, *, local: bool) -> None:
        """Refuse a start past either running-arm limit, naming the limit and
        the running arms that count against it."""

        running, running_local = self._running_roster()
        if len(running) >= MAX_RUNNING_EXPERIMENTS:
            raise ManagerError(
                f"running-arm limit reached: {MAX_RUNNING_EXPERIMENTS} arms in total; "
                f"running: {', '.join(sorted(running))}"
            )
        if local and len(running_local) >= MAX_RUNNING_LOCAL_EXPERIMENTS:
            raise ManagerError(
                "local-model limit reached: "
                f"{MAX_RUNNING_LOCAL_EXPERIMENTS} running arms whose main session or "
                f"sub-agents run on {LOCAL_QWEN_MODEL}; running on it: "
                f"{', '.join(sorted(running_local))}"
            )

    def _require_free_space(self) -> None:
        # The nearest existing directory is on the filesystem the experiment
        # directory will be created on.
        root = next(
            path
            for path in (self.experiments_root, *self.experiments_root.parents)
            if path.exists()
        )
        free = shutil.disk_usage(root).free
        needed = WORKSPACE_MIN_FREE_BYTES + ARM_DISK_MARGIN_BYTES
        if free < needed:
            raise ManagerError(
                f"not enough free disk space to create an arm: {free / 1024**3:.1f} GiB "
                f"free on the experiments filesystem, {needed / 1024**3:g} GiB needed "
                f"(the {WORKSPACE_MIN_FREE_BYTES / 1024**3:g} GiB floor below which "
                "shell and batch_validate are refused, plus "
                f"{ARM_DISK_MARGIN_BYTES / 1024**3:g} GiB for the arm's own PIT views)"
            )

    def control(
        self,
        experiment_id: str,
        action: str,
        *,
        session_key: str | None = None,
        directive: str | None = None,
        text: object = None,
        interrupt: object = False,
        at: str | None = None,
    ) -> dict[str, object]:
        if action not in _ACTIONS:
            raise ManagerError(f"unknown control action: {action!r}")
        if at is not None and action != "restart":
            raise ManagerError("at is only accepted by restart")
        with self._experiment_lock(experiment_id):
            directory = self._experiment_dir(experiment_id)
            _modern_ref_store(directory)
            try:
                identity = PublicIdentity(directory)
            except (OSError, ValueError) as exc:
                raise ManagerError("experiment identity state is unreadable") from exc
            try:
                raw_session_key = (
                    identity.raw_session_key(session_key)
                    if isinstance(session_key, str) and session_key
                    else None
                )
            except (KeyError, ValueError) as exc:
                raise ManagerError(str(exc)) from exc
            path = directory / "hitl/control.json"
            # The two worker-signalling actions wait out a SIGTERM grace, so
            # they must not hold control_lock: an exiting worker still consumes
            # its own session controls under that lock, and blocking it there
            # would turn a graceful shutdown into a forced kill.
            if action == "terminate":
                return self._terminate(experiment_id, directory)
            if action == "restart":
                return self._restart(experiment_id, directory, at=at)
            if action == "inject_message":
                receipt = self._inject_message(
                    directory,
                    identity,
                    session_key=raw_session_key,
                    text=text,
                    interrupt=interrupt,
                )
                queued_session = receipt.get("session_key")
                if isinstance(queued_session, str) and queued_session:
                    receipt["session_key"] = identity.public_session_key(queued_session)
                return receipt
            with control_lock(path):
                control = read_control(path)
                self._apply_control_action(
                    directory,
                    identity,
                    control,
                    action=action,
                    session_key=raw_session_key,
                    directive=directive,
                )
                write_control(path, control)
                response: dict[str, object] = {
                    "control": identity.public_control(control.to_record())
                }
            if action == "resume":
                state = experiment_state(directory)
                resumable = state.get("state") in _TERMINAL_RESUMABLE_STATES
                if (
                    not state.get("worker_alive")
                    and resumable
                    and self.worker_script.is_file()
                ):
                    return {**response, **self.start_worker(experiment_id)}
            return response

    def _inject_message(
        self,
        directory: Path,
        identity: PublicIdentity,
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
        if current != session_key or identity.session(current)["kind"] != "research":
            raise ManagerError(
                "inject_message session_key must match the current Agent session"
            )
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
        identity: PublicIdentity,
        control: ControlState,
        *,
        action: str,
        session_key: str | None,
        directive: str | None,
    ) -> None:
        if action == "pause":
            control.request = "pause"
        elif action == "resume":
            control.request = None
        elif action == "stop":
            control.request = "stop"
        elif action == "set_gpu_count":
            self._require_pending_research_session(directory, identity, action, session_key)
            raw = str(directive or "").strip()
            if raw:
                try:
                    count = int(raw)
                except ValueError as exc:
                    raise ManagerError("GPU 数量必须是整数") from exc
                if not 0 <= count <= 4:
                    raise ManagerError("GPU 数量须在 0..4 之间")
                control.gpu_counts[session_key] = count  # type: ignore[index]
            else:
                control.gpu_counts.pop(session_key, None)  # type: ignore[arg-type]
        elif action == "set_directive":
            self._require_pending_research_session(directory, identity, action, session_key)
            if directive in {None, ""}:
                control.directives.pop(session_key, None)  # type: ignore[arg-type]
            else:
                control.directives[session_key] = str(directive)  # type: ignore[index]
        else:
            raise ManagerError(f"unknown control action: {action!r}")

    @staticmethod
    def _require_pending_research_session(
        directory: Path,
        identity: PublicIdentity,
        action: str,
        session_key: str | None,
    ) -> None:
        """A per-session setting targets a planned research session, and only
        while research lasts: after the freeze no Agent session remains to read
        it."""
        if not session_key:
            raise ManagerError(f"{action} requires session_key")
        if identity.session(session_key)["kind"] != "research":
            raise ManagerError(f"{action} applies to research sessions only")
        if research_over(read_ledger_records(directory)):
            raise ManagerError("research is over; no research session remains")

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
        try:
            _signal_worker_group(pid, signal.SIGTERM)
        except ProcessLookupError as exc:  # exited between check and signal
            raise ManagerError("worker 已退出") from exc
        escalated = not _await_worker_exit(status_path, _TERMINATE_GRACE_SECONDS)
        if escalated:
            _signal_worker_group(pid, signal.SIGKILL)
        result: dict[str, object] = {
            "terminated_pid": pid,
            "escalated": escalated,
            "reclaimed_containers": _reclaim_sandbox_containers(experiment_id),
        }
        if escalated:
            # SIGKILL leaves no worker to stamp a terminal state; without this
            # the page shows a stale running state until pid-liveness kicks in
            # and the user cannot tell whether termination worked.
            status = _read_json(status_path)
            status.update(
                {"state": "terminated", "error": None, "terminated_at": utc_now_iso()}
            )
            write_json_atomic(status_path, status)
        return result

    def _restart(
        self, experiment_id: str, directory: Path, *, at: str | None = None
    ) -> dict[str, object]:
        """Terminate-and-restart in one step: SIGTERM the live worker, SIGKILL
        it if it outlives the grace, then resume via the ledger.

        ``at="session_boundary"`` defers instead: the request is recorded in
        the control state and the live worker re-executes itself once the
        session it is running has been recorded, so a code swap costs no session.

        Escalating rather than refusing: a worker that ignores SIGTERM is
        almost always inside a model call, which routinely outlasts any grace
        worth blocking a request on, so refusing only handed the researcher the
        same force-terminate-then-resume by hand. Forcing it is no less safe
        than that manual path, which has always ended in the same SIGKILL:
        ledger records are appended under an exclusive lock and fsynced,
        status and control writes are atomic, and the restarted worker
        re-derives its position from the ledger, rerunning the interrupted
        session whole."""
        if at is not None and at not in {"immediate", "session_boundary"}:
            raise ManagerError("restart at must be immediate or session_boundary")
        status_path = directory / "hitl/status.json"
        status = _read_json(status_path)
        if at == "session_boundary":
            # Only a live worker can consume the flag; without one the
            # researcher would watch a request that never applies.
            if not status_pid_alive(status):
                raise ManagerError("session_boundary restart requires a live worker")
            control_path = directory / "hitl/control.json"
            with control_lock(control_path):
                control = read_control(control_path)
                control.restart_pending = True
                write_control(control_path, control)
            return {
                "restarted": False,
                "at": "session_boundary",
                "restart_pending": True,
            }
        escalated = False
        # The slot lock spans the terminate AND the respawn, because the
        # terminate below frees this experiment's own running slot: taking it
        # only in start_worker let a concurrent create claim that slot while
        # the old worker was exiting, and the restart then failed on a
        # running-arm limit, leaving the experiment stopped. The price is
        # that a create or start begun during a restart waits out its grace;
        # control actions on other experiments still take no slot lock. Lock
        # order stays experiment -> slots: control() already holds this
        # experiment's lock.
        with self._slots:
            if status_pid_alive(status):
                pid = int(status["pid"])
                _signal_worker_group(pid, signal.SIGTERM)
                escalated = not _await_worker_exit(status_path, _RESTART_GRACE_SECONDS)
                if escalated:
                    try:
                        _signal_worker_group(pid, signal.SIGKILL)
                    except ProcessLookupError:  # exited just after the last poll
                        pass
                    if not _await_worker_exit(status_path, _SIGKILL_EXIT_SECONDS):
                        raise ManagerError(
                            f"worker pid {pid} 未响应 SIGKILL；请先排查该进程再重启"
                        )
            _reclaim_sandbox_containers(experiment_id)
            return {
                "restarted": True,
                "at": "immediate",
                "escalated": escalated,
                **self.start_worker(experiment_id),
            }

    def delete_experiment(self, experiment_id: str) -> dict[str, object]:
        with self._experiment_lock(experiment_id):
            directory = self._experiment_dir(experiment_id)
            _modern_ref_store(directory)
            state = experiment_state(directory)
            if state.get("worker_alive") or state.get("state") == "launching":
                raise ManagerError(
                    f"experiment {experiment_id!r} has a live worker; stop or terminate it before deleting"
                )
            removed_work_root: str | None = None
            # Deletion does not depend on params.json being readable: a failed
            # worker can leave it corrupt or inaccessible. The lenient reader
            # falls back to the deployment-derived sandbox path and, crucially,
            # never adopts an unreadable file's explicit work_root.
            params = _read_json(directory / "hitl/params.json")
            work_root = params.get("work_root")
            # work_root is the shared sandbox root: the worker runs this
            # experiment under work_root/<experiment_id> (pipelines.worker), and
            # relative values resolve against the repository there. Apply the
            # same derivation here, then remove that directory only when it is
            # the validated per-experiment dir under <repo>/.runtime/sandboxes —
            # never the shared root itself and never a sibling experiment. A
            # missing or unreadable params.json falls back to that same path.
            expected = _derived_sandbox_tree(self.repo_root, experiment_id)
            if expected is not None:
                try:
                    if work_root:
                        configured = Path(str(work_root))
                        if not configured.is_absolute():
                            configured = self.repo_root / configured
                        work_path = (configured / experiment_id).resolve()
                    else:
                        work_path = expected
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
            # The experiment's own image tags, before the state file naming
            # them goes with the directory.
            removed_image_refs = reclaim_experiment_sandbox_images(directory)
            try:
                _remove_readonly_tree(directory)
            except OSError as exc:
                raise ManagerDeleteError(
                    f"experiment {experiment_id!r} was not fully deleted: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            _reclaim_sandbox_containers(experiment_id)
            return {
                "deleted": experiment_id,
                "removed_work_root": removed_work_root,
                "removed_image_refs": removed_image_refs,
            }

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
