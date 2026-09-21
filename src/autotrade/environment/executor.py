"""Trusted and Docker-isolated executors for the daily strategy contract."""

from __future__ import annotations

import json
import os
import selectors
import shutil
import subprocess
import threading
import time
import uuid
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, Self, runtime_checkable

from .container_stats import ContainerResourceMonitor
from .contract_fingerprint import (
    SandboxImageContractMismatch,
    assert_image_contract_current,
)
from .gpu import device_request, gpu_memory_contention, select_gpus_with_free_memory
from .runtime import chmod_tree
from .sandbox import DockerSandbox, SandboxConfig, SandboxLimits, container_thread_env
from .strategy import BarTable, FitSchedule, StrategyContext, StrategyFunction
from .strategy_loader import load_strategy_module, validate_strategy_package

if TYPE_CHECKING:
    from .tools.base import CommandResult

_HOST_TIMEOUT_BUFFER_SECONDS = 15.0
_PROCESS_STOP_TIMEOUT_SECONDS = 2.0
# Readiness handshake sequence. Request sequences are non-negative, so a worker
# that echoes this one is answering the probe and nothing else. The worker
# rejects the unknown message type as a protocol error and echoes the
# sequence with it; that
# reply is the proof that the container is up, the interpreter is running and
# the strategy module finished importing. The one reply the worker emits
# *before* reading anything is its import failure, which carries no sequence —
# that is how a dead worker is told apart from a ready one.
_READY_SEQUENCE = -1
# The worker's two failure replies. Both may arrive without a sequence (an
# import failure before the first read, a line the worker could not parse at
# all), so neither is held to the sequence check that every other message is.
_WORKER_ERROR_TYPES = frozenset({"error", "protocol_error"})
# What ``docker run`` reports when the container's main process was SIGKILLed.
# With swap pinned to the memory cap (``SandboxLimits.memory``) that is what
# the container's own memory boundary looks like from here.
_SIGKILL_EXIT_CODE = 137
# Container-side path of the read-only strategy package (the directory that
# holds main.py), and of the read-only data roots and the per-replay state
# directory, which is read-only for generate_orders and read-write for fit.
CONTAINER_STRATEGY_DIR = "/strategy"
CONTAINER_SNAPSHOT_DIR = "/strategy-data/snapshot"
CONTAINER_ASOF_DIR = "/strategy-data/asof"
CONTAINER_MODELS_DIR = "/strategy-data/models"
CONTAINER_STATE_DIR = "/strategy-data/state"


class StrategyExecutionError(RuntimeError):
    """A strategy executor cannot return a truthful order payload."""


class StrategyRaised(StrategyExecutionError):
    """The strategy's own code raised inside a ``fit`` or ``generate_orders`` call.

    That is a measurement of the strategy -- it cannot run on this data -- and
    the only executor failure that is: a timeout, a broken pipe, a worker that
    never started or broke protocol stays a plain ``StrategyExecutionError``,
    because it measured the environment, not the strategy.
    """


class GpuMemoryContention(StrategyExecutionError):
    """A strategy call ran out of GPU memory because the device was taken.

    The strategy had less than the promised free-memory floor to work with
    (``environment.gpu.gpu_memory_contention``), so the failure measured the
    shared card, not the strategy: an environment failure like a timeout.
    """


def raised_by_strategy(exc: BaseException) -> bool:
    """Whether ``exc``, or anything it was raised from, is ``StrategyRaised``.

    Replay layers wrap an executor failure (``BacktestError`` and the
    evaluator above it), so the classification is read off the whole chain.
    """

    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        if isinstance(current, StrategyRaised):
            return True
        seen.add(id(current))
        current = current.__cause__ or current.__context__
    return False


# Where a failed replay carries its container telemetry. A replay that raises
# produces no result to put it in, and a fit that died on its clock or on a
# shared card is exactly the run whose peak memory and headroom are worth
# reading, so the measurement rides on the exception to the row that reports it.
_STRATEGY_RESOURCES = "strategy_resources"


def strategy_resource_usage(executor: object) -> dict[str, object]:
    """Container telemetry of one replay's executor; empty when it has none.

    A trusted in-process executor and the test doubles run no container and
    report nothing, which is not a failure: the block simply stays out.
    """

    usage = getattr(executor, "resource_usage", None)
    return dict(usage()) if callable(usage) else {}


def attach_strategy_resources(exc: BaseException, executor: object) -> None:
    usage = strategy_resource_usage(executor)
    if usage:
        setattr(exc, _STRATEGY_RESOURCES, usage)


def strategy_resources_of(exc: BaseException) -> dict[str, object]:
    """Telemetry carried by ``exc`` or by anything it was raised from.

    Replay layers wrap an executor failure, so the block is read off the whole
    chain for the same reason ``raised_by_strategy`` is.
    """

    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        usage = getattr(current, _STRATEGY_RESOURCES, None)
        if isinstance(usage, Mapping):
            return dict(usage)
        seen.add(id(current))
        current = current.__cause__ or current.__context__
    return {}


@runtime_checkable
class StrategyExecutor(Protocol):
    def execute(self, context: StrategyContext) -> object: ...

    def close(self) -> None: ...


@runtime_checkable
class FittableStrategyExecutor(StrategyExecutor, Protocol):
    """An executor that also runs the strategy's optional ``fit(context)``.

    ``fit_schedule`` is ``None`` when ``main.py`` declares no ``fit``.
    ``context_state_dir`` and ``context_models_dir`` are the path strings the
    strategy sees on its context, empty when the directory is absent.
    """

    fit_schedule: FitSchedule | None
    context_state_dir: str
    context_models_dir: str

    def fit(self, context: StrategyContext) -> None: ...


def _lock_state_dir(state_dir: Path | None, *, writable: bool) -> None:
    """Host-side read-only gate over an in-process strategy's state directory."""

    if state_dir is None:
        return
    if writable:
        chmod_tree(state_dir, file_mode=0o644, dir_mode=0o755)
    else:
        chmod_tree(state_dir, file_mode=0o444, dir_mode=0o555)


class TrustedStrategyExecutor:
    """Run an explicitly trusted, reviewed strategy in the host process.

    This is intentionally not an isolation boundary and is never selected by
    an implicit fallback from Docker execution. The state directory is still
    chmod-locked between ``fit`` calls so a ``generate_orders`` write fails.
    """

    def __init__(
        self,
        strategy: StrategyFunction,
        *,
        fit: Callable[[StrategyContext], object] | None = None,
        fit_schedule: FitSchedule | None = None,
        state_dir: str | Path | None = None,
        models_dir: str | Path | None = None,
    ) -> None:
        if not callable(strategy):
            raise TypeError("trusted strategy must be callable")
        if (fit is None) != (fit_schedule is None):
            raise TypeError("fit and fit_schedule must be given together")
        self._strategy = strategy
        self._fit = fit
        self.fit_schedule = fit_schedule
        self.state_dir = _existing_dir(state_dir, "state_dir")
        self.models_dir = _existing_dir(models_dir, "models_dir")
        if fit is not None and self.state_dir is None:
            raise StrategyExecutionError(
                "strategy defines fit(context) but the executor has no state_dir"
            )
        self.context_state_dir = str(self.state_dir) if self.state_dir is not None else ""
        self.context_models_dir = str(self.models_dir) if self.models_dir is not None else ""
        _lock_state_dir(self.state_dir, writable=False)

    @classmethod
    def from_path(
        cls,
        strategy_path: str | Path,
        *,
        state_dir: str | Path | None = None,
        models_dir: str | Path | None = None,
    ) -> TrustedStrategyExecutor:
        loaded = load_strategy_module(strategy_path)
        return cls(
            loaded.generate_orders,
            fit=loaded.fit,
            fit_schedule=loaded.fit_schedule,
            state_dir=state_dir,
            models_dir=models_dir,
        )

    def fit(self, context: StrategyContext) -> None:
        if self._fit is None:
            raise StrategyExecutionError("strategy defines no fit(context)")
        _lock_state_dir(self.state_dir, writable=True)
        try:
            self._fit(context)
        finally:
            _lock_state_dir(self.state_dir, writable=False)

    def execute(self, context: StrategyContext) -> object:
        return self._strategy(context)

    def close(self) -> None:
        return None


class DockerStrategyExecutor:
    """Reuse one locked-down Docker worker for every inference in an experiment.

    The strategy package (the directory holding ``main.py``) is bind-mounted
    read-only as a whole, so ``main.py`` can import its sibling modules.
    A strategy that declares ``fit`` gets a second, equally locked-down worker
    whose only difference is a read-write bind of the state directory; the
    inference worker binds the same directory read-only, so the kernel — not
    the strategy — decides that ``generate_orders`` cannot write state.

    Both workers attach the experiment's GPU request (``SandboxLimits``): the
    inference container selects the devices once at start and the fit worker
    inherits exactly those, so one evaluation holds one allocation and a
    replay whose experiment asked for no GPU stays CPU-only.

    ``agent_contract`` says whether an Agent is reading the contract text this
    checkout states: on for a research replay, off for a Paper book replaying a
    strategy frozen long ago. It only widens the image check from both halves
    of the contract to the runtime modules the container really enforces
    (``contract_fingerprint``); the runtime half is never optional.
    """

    def __init__(
        self,
        strategy_path: str | Path,
        config: SandboxConfig | None = None,
        *,
        snapshot_dir: str | Path | None = None,
        asof_dir: str | Path | None = None,
        models_dir: str | Path | None = None,
        state_dir: str | Path | None = None,
        state_writable: bool = False,
        agent_contract: bool = True,
    ) -> None:
        self.strategy_path = Path(strategy_path).resolve()
        if not self.strategy_path.is_file():
            raise StrategyExecutionError(f"strategy file does not exist: {self.strategy_path}")
        self.fit_schedule = validate_strategy_package(self.strategy_path)
        self.config = config or SandboxConfig()
        self.snapshot_dir = _existing_dir(snapshot_dir, "snapshot_dir")
        self.asof_dir = _existing_dir(asof_dir, "asof_dir")
        self.models_dir = _existing_dir(models_dir, "models_dir")
        self.state_dir = _existing_dir(state_dir, "state_dir")
        if self.fit_schedule is not None and self.state_dir is None:
            raise StrategyExecutionError(
                "strategy defines fit(context) but the executor has no state_dir"
            )
        self.context_state_dir = CONTAINER_STATE_DIR if self.state_dir is not None else ""
        self.context_models_dir = CONTAINER_MODELS_DIR if self.models_dir is not None else ""
        self._state_writable = state_writable
        self._agent_contract = agent_contract
        self._fit_worker: DockerStrategyExecutor | None = None
        # Resolved before the container exists and rendered into its run
        # arguments below. An unsatisfiable request fails right here, so a
        # replay never silently trains on CPU instead of the GPU it asked for.
        self.gpu_indices, self.gpu_free_at_start_mib = _select_strategy_gpus(
            self.config.limits
        )
        self.container_name = f"autotrade-strategy-{uuid.uuid4().hex}"
        # Host-side telemetry of this container: what it actually cost, beside
        # the limits it was started with. Bound once the container is up.
        self._monitor = ContainerResourceMonitor(
            self.container_name,
            docker_executable=self.config.docker_executable,
            sample_gpu=bool(self.gpu_indices),
        )
        # Wall clock of each completed ``fit`` roundtrip, measured on the same
        # clock ``fit_timeout_seconds`` bounds.
        self._fit_seconds: list[float] = []
        self._process: subprocess.Popen[bytes] | None = None
        self._stdout_buffer = bytearray()
        self._stderr_tail: deque[bytes] = deque()
        self._stderr_size = 0
        self._stderr_thread: threading.Thread | None = None
        self._closed = False
        self._reset_transport_state()
        self._start()

    def docker_command(self) -> list[str]:
        """Render the complete container boundary for inspection and testing."""

        limits = self.config.limits
        strategy_mount = (
            f"type=bind,src={self.strategy_path.parent},dst={CONTAINER_STRATEGY_DIR},readonly"
        )
        command = [
            self.config.docker_executable,
            "run",
            "--pull",
            "never",
            "--rm",
            "-i",
            "--name",
            self.container_name,
            "--network",
            "none",
            "--user",
            "61000:61000",
            "--read-only",
            "--tmpfs",
            f"/tmp:rw,noexec,nosuid,nodev,size={limits.tmpfs_size}",
            # Shared memory for multi-process training inside fit(context):
            # without it Docker's 64 MB default makes a torch DataLoader with
            # workers or a joblib memmapped backend fail on a container that
            # otherwise has 8 CPUs to use.
            "--shm-size",
            limits.shm_size,
            "--cpus",
            f"{limits.cpus:g}",
            # Swap ceiling equal to the memory cap: see ``SandboxLimits.memory``
            # for why the two are the same number.
            "--memory",
            limits.memory,
            "--memory-swap",
            limits.memory,
            "--pids-limit",
            str(limits.pids),
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
        ]
        for key, value in sorted(self.config.labels.items()):
            command.extend(["--label", f"{key}={value}"])
        if self.gpu_indices:
            command.extend(["--gpus", device_request(self.gpu_indices)])
        for key, value in sorted(container_thread_env(limits.cpus).items()):
            command.extend(["--env", f"{key}={value}"])
        command.extend(["--mount", strategy_mount])
        for source, target in (
            (self.snapshot_dir, CONTAINER_SNAPSHOT_DIR),
            (self.asof_dir, CONTAINER_ASOF_DIR),
            (self.models_dir, CONTAINER_MODELS_DIR),
        ):
            if source is not None:
                command.extend(["--mount", f"type=bind,src={source},dst={target},readonly"])
        if self.state_dir is not None:
            mode = "" if self._state_writable else ",readonly"
            command.extend(
                ["--mount", f"type=bind,src={self.state_dir},dst={CONTAINER_STATE_DIR}{mode}"]
            )
        command.extend([
            "--workdir",
            CONTAINER_STRATEGY_DIR,
            self.config.image,
            "python",
            "-m",
            "autotrade.environment.strategy_worker",
            f"{CONTAINER_STRATEGY_DIR}/{self.strategy_path.name}",
        ])
        return command

    def fit(self, context: StrategyContext) -> None:
        if self.fit_schedule is None:
            raise StrategyExecutionError("strategy defines no fit(context)")
        if self._closed:
            raise StrategyExecutionError("Docker strategy executor is closed")
        if self._fit_worker is None:
            self._fit_worker = DockerStrategyExecutor(
                self.strategy_path,
                # Same boundary, plus this evaluation's already-selected
                # devices: the fit worker shares the inference container's
                # GPUs instead of selecting a second set of its own.
                replace(
                    self.config,
                    limits=replace(
                        self.config.limits, gpu_devices=tuple(self.gpu_indices)
                    ),
                ),
                snapshot_dir=self.snapshot_dir,
                asof_dir=self.asof_dir,
                models_dir=self.models_dir,
                state_dir=self.state_dir,
                state_writable=True,
                agent_contract=self._agent_contract,
            )
        try:
            self._fit_worker._roundtrip(
                context, kind="fit", timeout_seconds=self.config.limits.fit_timeout_seconds
            )
        except StrategyExecutionError:
            self._abort()
            raise

    def execute(self, context: StrategyContext) -> object:
        return self._roundtrip(
            context, kind="execute", timeout_seconds=self.config.limits.timeout_seconds
        )

    def _roundtrip(
        self, context: StrategyContext, *, kind: str, timeout_seconds: float
    ) -> object:
        if self._closed or self._process is None:
            raise StrategyExecutionError("Docker strategy executor is closed")
        label = "strategy fit" if kind == "fit" else "strategy inference"
        # The message a deadline miss reports, read by _write/_read_line.
        self._active_limit = f"{label} exceeded {timeout_seconds:g}s"
        expected = "fitted" if kind == "fit" else "orders"
        consumed = 0
        # Host NL waits extend the deadline, so they are not the strategy's
        # clock and must not land in the fit timing either.
        nl_seconds = 0.0
        try:
            # Materializing and validating the request is the host's own work on
            # host-owned PIT data; only what happens from the hand-over onwards
            # is the strategy's, so its clock starts here and not before.
            request, last_available_at = self._prepare_execute(context)
            request["type"] = kind
            sequence = request["sequence"]
            started = time.monotonic()
            deadline = started + timeout_seconds
            self._send(request, deadline)
            while True:
                message, size = self._read_message(deadline)
                consumed += size
                if consumed > self.config.limits.max_output_chars:
                    raise StrategyExecutionError("strategy protocol output exceeded max_output_chars")
                message_type = message.get("type")
                response_sequence = message.get("sequence")
                if (message_type not in _WORKER_ERROR_TYPES or "sequence" in message) and (
                    isinstance(response_sequence, bool)
                    or not isinstance(response_sequence, int)
                    or response_sequence != sequence
                ):
                    raise StrategyExecutionError("worker response sequence does not match request")
                if message_type == "nl_request":
                    request = message.get("request")
                    if not isinstance(request, Mapping):
                        raise StrategyExecutionError("worker sent an invalid NL request")
                    # Host NL is a trusted service with its own quotas. Its wait
                    # is not untrusted strategy compute and must not burn the
                    # strategy's own inference or fit cap.
                    nl_started = time.monotonic()
                    try:
                        try:
                            response = context.nl(**dict(request))
                        except Exception as exc:  # noqa: BLE001 - host errors cross the protocol
                            payload: dict[str, object] = {
                                "type": "nl_response",
                                "sequence": sequence,
                                "error": str(exc),
                            }
                        else:
                            payload = {
                                "type": "nl_response",
                                "sequence": sequence,
                                "result": dict(response),
                            }
                    finally:
                        waited = time.monotonic() - nl_started
                        deadline += waited
                        nl_seconds += waited
                    self._send(payload, deadline)
                    continue
                if message_type == expected:
                    if kind == "fit":
                        self._fit_seconds.append(
                            round(time.monotonic() - started - nl_seconds, 1)
                        )
                    self._transport_sequence = sequence
                    self._transport_inference_at = context.inference_at
                    self._transport_bars = context.bars
                    self._transport_table = context._bars_table
                    self._transport_last_available_at = last_available_at
                    return message.get("orders")
                if message_type == "protocol_error":
                    # The worker could not speak to the message this host sent
                    # it. That measures the environment exactly as a timeout
                    # does, so the replay year is refunded rather than charged
                    # to a candidate whose code never ran.
                    detail = str(message.get("error") or "no detail")
                    raise StrategyExecutionError(
                        f"Docker {label} failed: strategy worker rejected the host "
                        f"protocol: {detail}"
                    )
                if message_type == "error":
                    # The worker's reply to a call whose strategy code raised.
                    text = str(message.get("error") or "strategy worker failed")
                    contention = gpu_memory_contention(text) if self.gpu_indices else None
                    if contention is not None:
                        raise GpuMemoryContention(f"{contention}; strategy error: {text}")
                    raise StrategyRaised(text)
                raise StrategyExecutionError(f"unexpected strategy worker message: {message_type!r}")
        except (TimeoutError, BrokenPipeError, OSError, ValueError, json.JSONDecodeError) as exc:
            self._abort()
            detail = self._stderr_text()
            suffix = f"; worker stderr: {detail}" if detail else ""
            raise StrategyExecutionError(f"Docker {label} failed: {exc}{suffix}") from exc
        except StrategyExecutionError:
            self._abort()
            raise

    def _prepare_execute(
        self,
        context: StrategyContext,
    ) -> tuple[dict[str, object], datetime | None]:
        reset = self._transport_sequence < 0
        sequence = 0 if reset else self._transport_sequence + 1
        base_count = 0 if reset else len(self._transport_bars)
        table = context._bars_table
        total_count = len(context.bars)
        if len(table) < total_count:
            raise StrategyExecutionError("strategy bar PIT metadata is inconsistent")
        if not reset:
            previous_inference = self._transport_inference_at
            if previous_inference is None or context.inference_at <= previous_inference:
                raise StrategyExecutionError("strategy inference_at must increase monotonically")
            if total_count < base_count:
                raise StrategyExecutionError("strategy bars are not append-only")
            if not self._transport_table.prefix_matches(table, base_count):
                raise StrategyExecutionError("strategy bars changed before base_count")

        last_available_at = None if reset else self._transport_last_available_at
        for index in range(base_count, total_count):
            available_at = table.available_at(index)
            if available_at > context.inference_at:
                raise StrategyExecutionError("strategy context contains data not visible at inference time")
            if last_available_at is not None and available_at < last_available_at:
                raise StrategyExecutionError("strategy bar available_at must be monotonic")
            last_available_at = available_at

        return (
            {
                # The caller sets "type": this request serves both fit and
                # generate_orders and only the caller knows which.
                "sequence": sequence,
                "reset": reset,
                "base_count": base_count,
                "total_count": total_count,
                "context": self._context_record(context),
                # Only the delta is materialized; the shipped prefix stays columnar.
                "bars": [table.record(index) for index in range(base_count, total_count)],
            },
            last_available_at,
        )

    def _context_record(self, context: StrategyContext) -> dict[str, object]:
        record = {
            "inference_at": context.inference_at.isoformat(),
            "account": context.account.to_record(),
            "snapshot_dir": context.snapshot_dir,
            "asof_dir": context.asof_dir,
            "asof_version": context.asof_version,
            "state_dir": context.state_dir,
            "models_dir": context.models_dir,
        }
        for name, target in (
            ("snapshot_dir", CONTAINER_SNAPSHOT_DIR),
            ("asof_dir", CONTAINER_ASOF_DIR),
            ("models_dir", CONTAINER_MODELS_DIR),
            ("state_dir", CONTAINER_STATE_DIR),
        ):
            if not getattr(context, name):
                continue
            if getattr(self, name) is None:
                raise StrategyExecutionError(f"strategy context has {name} without a mount")
            record[name] = target
        return record

    def resource_usage(self) -> dict[str, object]:
        """What this evaluation's strategy containers actually cost.

        Read once at container end (``ContainerResourceMonitor``) and reported
        beside the limits that were in force, so a replay that died on a clock
        or a shared card says how close it was to which ceiling instead of
        leaving the session to reconstruct it from a different container. A fit
        strategy runs two containers; the peaks are the higher of the two,
        because they are the same evaluation and the same host.

        Only measured values appear: a figure the host could not read is left
        out rather than reported as zero.
        """

        limits = self.config.limits
        record: dict[str, object] = dict(self._monitor.record())
        worker = self._fit_worker
        if worker is not None:
            for name, value in worker._monitor.record().items():
                current = record.get(name)
                record[name] = (
                    max(int(current), value) if isinstance(current, int) else value
                )
        fit_seconds = list(self._fit_seconds) + (
            list(worker._fit_seconds) if worker is not None else []
        )
        if fit_seconds:
            record["fit_seconds"] = fit_seconds
        if self.fit_schedule is not None:
            record["fit_timeout_seconds"] = limits.fit_timeout_seconds
        record["decision_timeout_seconds"] = limits.timeout_seconds
        if self.gpu_free_at_start_mib is not None:
            record["gpu_free_at_start_bytes"] = (
                self.gpu_free_at_start_mib * 1024 * 1024
            )
        return record

    def close(self) -> None:
        self._monitor.stop()
        if self._fit_worker is not None:
            self._fit_worker.close()
        if self._closed and self._process is None:
            self._reset_transport_state()
            return
        self._closed = True
        process = self._process
        if process is None:
            self._reset_transport_state()
            return
        try:
            if process.poll() is None and process.stdin is not None:
                try:
                    self._write(
                        process.stdin,
                        {"type": "close"},
                        time.monotonic() + _PROCESS_STOP_TIMEOUT_SECONDS,
                    )
                except (BrokenPipeError, OSError):
                    pass
                try:
                    process.stdin.close()
                except OSError:
                    pass
            self._reap_process(process, force=False)
        finally:
            self._finalize_process(process)
            self._reset_transport_state()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def _start(self) -> None:
        _require_local_image(self.config, agent_contract=self._agent_contract)
        try:
            self._process = subprocess.Popen(
                self.docker_command(),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
            )
        except OSError as exc:
            raise StrategyExecutionError(f"cannot start Docker strategy worker: {exc}") from exc
        assert self._process.stderr is not None
        self._stderr_thread = threading.Thread(target=self._drain_stderr, daemon=True)
        self._stderr_thread.start()
        self._await_ready()
        # Only now is the container certainly up, so this is where its cgroup
        # can be resolved; a container that never became ready has no usage to
        # report anyway.
        self._monitor.start()

    def _await_ready(self) -> None:
        """Block until the container's worker loop can answer, or fail explicitly.

        ``docker run`` returns as soon as the process is spawned: container
        scheduling, interpreter start and the strategy module import all finish
        lazily, and without this handshake their wall clock lands inside the
        first ``generate_orders`` call and is charged to the strategy's
        per-decision timeout. That is an environment cost, so it gets its own
        generous environment-side budget and its own error.
        """

        limit = self.config.limits.startup_timeout_seconds
        self._active_limit = f"strategy worker startup exceeded {limit:g}s"
        deadline = time.monotonic() + limit
        try:
            self._send({"type": "ready", "sequence": _READY_SEQUENCE}, deadline)
            message, _ = self._read_message(deadline)
        except (
            TimeoutError,
            BrokenPipeError,
            OSError,
            ValueError,
            json.JSONDecodeError,
            StrategyExecutionError,
        ) as exc:
            # Abort first: it joins the stderr drain, so a container that died
            # on startup (missing image, daemon error) still has its own words
            # in the message instead of a race with the reader thread.
            self._abort()
            detail = self._stderr_text()
            suffix = f"; worker stderr: {detail}" if detail else ""
            raise StrategyExecutionError(
                "Docker strategy worker did not become ready (host side: container "
                f"start, host contention or a broken worker, not strategy compute): "
                f"{exc}{suffix}"
            ) from exc
        finally:
            self._active_limit = None
        if message.get("sequence") != _READY_SEQUENCE:
            # The only message that precedes the worker's first read.
            error = str(message.get("error") or message)
            self._abort()
            raise StrategyExecutionError(f"Docker strategy worker failed to start: {error}")

    def _send(self, message: Mapping[str, object], deadline: float) -> None:
        process = self._process
        if process is None or process.stdin is None:
            raise BrokenPipeError("strategy worker stdin is unavailable")
        self._write(process.stdin, message, deadline)

    def _timeout_message(self) -> str:
        return self._active_limit or (
            f"strategy inference exceeded {self.config.limits.timeout_seconds:g}s"
        )

    def _write(self, stream, message: Mapping[str, object], deadline: float) -> None:
        encoded = json.dumps(message, ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n"
        fd = stream.fileno()
        os.set_blocking(fd, False)
        view = memoryview(encoded)
        with selectors.DefaultSelector() as selector:
            selector.register(fd, selectors.EVENT_WRITE)
            while view:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not selector.select(remaining):
                    raise TimeoutError(self._timeout_message())
                try:
                    written = os.write(fd, view)
                except (BlockingIOError, InterruptedError):
                    continue
                if written <= 0:
                    raise BrokenPipeError("strategy worker stdin closed during request")
                view = view[written:]

    def _read_message(self, deadline: float) -> tuple[dict[str, object], int]:
        line = self._read_line(deadline)
        try:
            message = json.loads(line)
        except json.JSONDecodeError as exc:
            raise StrategyExecutionError("worker emitted invalid JSON") from exc
        if not isinstance(message, dict):
            raise StrategyExecutionError("worker message must be a JSON object")
        return message, len(line)

    def _read_line(self, deadline: float) -> bytes:
        process = self._process
        if process is None or process.stdout is None:
            raise BrokenPipeError("strategy worker stdout is unavailable")
        fd = process.stdout.fileno()
        with selectors.DefaultSelector() as selector:
            selector.register(fd, selectors.EVENT_READ)
            while True:
                newline = self._stdout_buffer.find(b"\n")
                if newline >= 0:
                    line = bytes(self._stdout_buffer[:newline])
                    del self._stdout_buffer[: newline + 1]
                    return line
                if len(self._stdout_buffer) > self.config.limits.max_output_chars:
                    raise StrategyExecutionError("strategy protocol line exceeded max_output_chars")
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not selector.select(remaining):
                    raise TimeoutError(self._timeout_message())
                chunk = os.read(fd, min(4096, self.config.limits.max_output_chars + 1))
                if not chunk:
                    code = process.poll()
                    if code == _SIGKILL_EXIT_CODE:
                        # Name the boundary instead of a bare signal number: in
                        # a container with no network and no capabilities, whose
                        # host side only kills it on a deadline it has already
                        # reported, the memory cap is what kills a worker this
                        # way. The cgroup goes with the container (``--rm``), so
                        # this run also has no peak to read -- say so, rather
                        # than leave the session comparing its fit against the
                        # surviving container's figure.
                        raise BrokenPipeError(
                            f"strategy worker was killed before a response (exit {code}, "
                            f"SIGKILL): the container's {self.config.limits.memory} memory "
                            "cap is what kills a worker this way, and its cgroup is gone "
                            "with the container, so this run reports no peak memory"
                        )
                    raise BrokenPipeError(f"strategy worker exited before a response (code={code})")
                self._stdout_buffer.extend(chunk)

    def _drain_stderr(self) -> None:
        process = self._process
        if process is None or process.stderr is None:
            return
        limit = self.config.limits.max_output_chars
        while True:
            chunk = process.stderr.read(4096)
            if not chunk:
                return
            self._stderr_tail.append(chunk)
            self._stderr_size += len(chunk)
            while self._stderr_size > limit and self._stderr_tail:
                self._stderr_size -= len(self._stderr_tail.popleft())

    def _stderr_text(self) -> str:
        return b"".join(self._stderr_tail).decode("utf-8", errors="replace").strip()

    def _abort(self) -> None:
        # Before the container is removed: its cgroup is the only place the
        # peak of a run that just died on a clock or a card still exists.
        self._monitor.stop()
        self._closed = True
        if self._fit_worker is not None and self._fit_worker is not self:
            self._fit_worker._abort()
        process = self._process
        if process is None:
            self._reset_transport_state()
            return
        try:
            self._reap_process(process, force=True)
        finally:
            self._finalize_process(process)
            self._reset_transport_state()

    def _reset_transport_state(self) -> None:
        self._active_limit: str | None = None
        self._transport_sequence = -1
        self._transport_inference_at = None
        self._transport_bars = ()
        self._transport_table = BarTable()
        self._transport_last_available_at = None

    def _reap_process(self, process: subprocess.Popen[bytes], *, force: bool) -> None:
        if force:
            self._remove_container()
            self._terminate_and_wait(process)
            return
        try:
            process.wait(timeout=_PROCESS_STOP_TIMEOUT_SECONDS)
        except (OSError, subprocess.TimeoutExpired):
            self._remove_container()
            self._terminate_and_wait(process)

    @staticmethod
    def _terminate_and_wait(process: subprocess.Popen[bytes]) -> None:
        if process.poll() is None:
            try:
                process.terminate()
            except OSError:
                pass
        try:
            process.wait(timeout=_PROCESS_STOP_TIMEOUT_SECONDS)
            return
        except (OSError, subprocess.TimeoutExpired):
            pass
        if process.poll() is None:
            try:
                process.kill()
            except OSError:
                pass
        try:
            process.wait(timeout=_PROCESS_STOP_TIMEOUT_SECONDS)
        except (OSError, subprocess.TimeoutExpired):
            pass

    def _finalize_process(self, process: subprocess.Popen[bytes]) -> None:
        thread = self._stderr_thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=_PROCESS_STOP_TIMEOUT_SECONDS)
        close_process_pipes(process)
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=_PROCESS_STOP_TIMEOUT_SECONDS)
        self._stderr_thread = None
        if self._process is process:
            self._process = None

    def _remove_container(self) -> None:
        try:
            subprocess.run(
                [self.config.docker_executable, "rm", "--force", self.container_name],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=10,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            pass


class PersistentCommandRunner:
    """CommandRunner adapter over one already-started persistent Sandbox."""

    def __init__(self, sandbox: DockerSandbox) -> None:
        self.sandbox = sandbox

    def run(
        self,
        argv: Sequence[str],
        *,
        cwd: str,
        timeout_seconds: float,
        max_output_chars: int,
        input_text: str | None = None,
    ) -> CommandResult:
        from .tools.base import CommandResult

        if not argv or any(not str(item) for item in argv):
            raise ValueError("argv must contain non-empty strings")
        completed = self.sandbox.exec_limited(
            tuple(map(str, argv)),
            cwd=cwd,
            timeout_seconds=timeout_seconds,
            max_output_chars=max_output_chars,
            input_text=input_text,
        )
        return CommandResult(
            completed.exit_code,
            completed.stdout,
            completed.stderr,
            timed_out=completed.exit_code == 124,
            stdout_truncated=completed.stdout_truncated,
            stderr_truncated=completed.stderr_truncated,
        )


@dataclass(frozen=True)
class ExecResult:
    exit_code: int
    stdout: str
    stderr: str
    stdout_truncated: bool = False
    stderr_truncated: bool = False


def docker_available(docker_executable: str = "docker") -> bool:
    executable = shutil.which(docker_executable)
    if executable is None:
        return False
    try:
        completed = subprocess.run([executable, "info"], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10, check=False)
    except (OSError, subprocess.SubprocessError):
        return False
    return completed.returncode == 0


def _with_container_timeout(argv: Sequence[str], timeout_seconds: float) -> list[str]:
    return [
        "timeout",
        "--signal=TERM",
        "--kill-after=5",
        f"{float(timeout_seconds):g}",
        *map(str, argv),
    ]


def _run_limited_capture(
    argv: Sequence[str],
    *,
    cwd: Path | str | None = None,
    env: Mapping[str, str] | None = None,
    timeout_seconds: float,
    max_output_chars: int,
    input_text: str | None = None,
) -> ExecResult:
    """Run a command while bounding retained stdout and stderr."""

    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    if isinstance(max_output_chars, bool) or max_output_chars <= 0:
        raise ValueError("max_output_chars must be a positive integer")
    stdin = subprocess.PIPE if input_text is not None else subprocess.DEVNULL
    process = subprocess.Popen(
        list(map(str, argv)),
        cwd=str(cwd) if cwd is not None else None,
        env=dict(env) if env is not None else None,
        stdin=stdin,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert process.stdout is not None
    assert process.stderr is not None
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ, "stdout")
    selector.register(process.stderr, selectors.EVENT_READ, "stderr")
    pending_input = memoryview((input_text or "").encode("utf-8"))
    input_offset = 0
    if process.stdin is not None:
        os.set_blocking(process.stdin.fileno(), False)
        selector.register(process.stdin, selectors.EVENT_WRITE, "stdin")
    stdout = bytearray()
    stderr = bytearray()
    stdout_truncated = False
    stderr_truncated = False
    timed_out = False
    deadline = time.monotonic() + timeout_seconds

    try:
        while selector.get_map():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                timed_out = True
                process.kill()
                break
            events = selector.select(timeout=min(0.1, remaining))
            if not events and process.poll() is not None:
                events = selector.select(timeout=0)
            for key, _mask in events:
                if key.data == "stdin":
                    try:
                        if input_offset < len(pending_input):
                            input_offset += os.write(
                                key.fileobj.fileno(), pending_input[input_offset:]
                            )
                        if input_offset >= len(pending_input):
                            selector.unregister(key.fileobj)
                            key.fileobj.close()
                    except (BrokenPipeError, OSError):
                        selector.unregister(key.fileobj)
                        key.fileobj.close()
                    continue
                chunk = os.read(key.fileobj.fileno(), 8192)
                if not chunk:
                    selector.unregister(key.fileobj)
                    continue
                target = stdout if key.data == "stdout" else stderr
                available = max(0, max_output_chars - len(target))
                if available:
                    target.extend(chunk[:available])
                if len(chunk) > available:
                    if key.data == "stdout":
                        stdout_truncated = True
                    else:
                        stderr_truncated = True
    finally:
        for key in list(selector.get_map().values()):
            try:
                selector.unregister(key.fileobj)
            except KeyError:
                pass
        selector.close()

    if timed_out:
        try:
            process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        timeout_message = f"\ntimeout after {timeout_seconds:g}s".encode()
        if len(stderr) + len(timeout_message) <= max_output_chars:
            stderr.extend(timeout_message)
        else:
            stderr_truncated = True
        return_code = 124
    else:
        return_code = process.wait()
    close_process_pipes(process)
    return ExecResult(
        return_code,
        stdout.decode("utf-8", errors="replace"),
        stderr.decode("utf-8", errors="replace"),
        stdout_truncated,
        stderr_truncated,
    )


def close_process_pipes(process: subprocess.Popen[bytes]) -> None:
    for pipe in (process.stdin, process.stdout, process.stderr):
        if pipe is not None and not pipe.closed:
            try:
                pipe.close()
            except OSError:
                pass


def _limited_text(value: str | bytes | None, maximum: int) -> str:
    if isinstance(value, bytes):
        text = value.decode("utf-8", errors="replace")
    else:
        text = value or ""
    return text[:maximum]


def _existing_dir(value: str | Path | None, name: str) -> Path | None:
    if value is None or str(value) == "":
        return None
    path = Path(value).resolve()
    if not path.is_dir():
        raise StrategyExecutionError(f"{name} does not exist: {path}")
    return path


def _select_strategy_gpus(limits: SandboxLimits) -> tuple[list[int], int | None]:
    """Device indexes this strategy container attaches, and their free memory.

    Already-selected devices (``gpu_devices``, how the fit worker inherits the
    inference container's allocation) are used as they are. Otherwise the Agent
    session container's policy applies unchanged: the requested number of
    devices matching ``gpu_name_filter`` with the most free video memory at
    container start. There is no CPU fallback — ``fit(context)`` training on a
    device the experiment did not get would be a different computation reported
    as the same result, so an unsatisfiable request raises
    ``GpuUnavailableError`` instead.

    The second element is the free video memory the admission probe measured
    across the selected devices, which is the headroom the strategy started
    with on a shared card. It is ``None`` when no probe ran: no device was
    asked for, or this is the fit worker reusing the allocation the inference
    container already made.
    """

    if limits.gpu_count <= 0:
        return [], None
    if limits.gpu_devices:
        return list(limits.gpu_devices), None
    selected = select_gpus_with_free_memory(
        limits.gpu_count, require_name=limits.gpu_name_filter
    )
    return [index for index, _free in selected], sum(free for _index, free in selected)


def _require_local_image(config: SandboxConfig, *, agent_contract: bool) -> str:
    """Resolve Docker and reject an absent or stale image without any pull attempt.

    The image carries its own copy of the strategy loader, so an image baking
    other bytes would enforce superseded rules against a strategy written to
    the rules this checkout states. That divergence is invisible from inside
    the container, so it is checked here — once per strategy worker start,
    before the worker exists. ``agent_contract`` additionally requires the
    image to have been built from the README the Agent reads; see
    ``contract_fingerprint``.
    """

    executable = shutil.which(config.docker_executable)
    if executable is None:
        raise StrategyExecutionError(f"Docker executable is unavailable: {config.docker_executable}")
    try:
        inspected = subprocess.run(
            [executable, "image", "inspect", config.image],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise StrategyExecutionError(f"Docker is unavailable: {exc}") from exc
    if inspected.returncode != 0:
        detail = _limited_text(inspected.stderr, 4000).strip()
        raise StrategyExecutionError(
            f"sandbox image is unavailable locally: {config.image}"
            + (f" ({detail})" if detail else "")
        )
    try:
        assert_image_contract_current(
            config.image, docker_executable=executable, agent_contract=agent_contract
        )
    except SandboxImageContractMismatch as exc:
        raise StrategyExecutionError(str(exc)) from exc
    except (OSError, subprocess.SubprocessError) as exc:
        raise StrategyExecutionError(f"Docker is unavailable: {exc}") from exc
    return executable


__all__ = [
    "CONTAINER_ASOF_DIR",
    "CONTAINER_MODELS_DIR",
    "CONTAINER_SNAPSHOT_DIR",
    "CONTAINER_STATE_DIR",
    "CONTAINER_STRATEGY_DIR",
    "DockerStrategyExecutor",
    "ExecResult",
    "FittableStrategyExecutor",
    "GpuMemoryContention",
    "PersistentCommandRunner",
    "StrategyExecutionError",
    "StrategyExecutor",
    "StrategyRaised",
    "TrustedStrategyExecutor",
    "attach_strategy_resources",
    "docker_available",
    "raised_by_strategy",
    "strategy_resource_usage",
    "strategy_resources_of",
]
