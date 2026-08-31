"""Trusted and Docker-isolated executors for the daily strategy contract."""

from __future__ import annotations

import json
import math
import os
import selectors
import shutil
import subprocess
import threading
import time
import uuid
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, Self, runtime_checkable

from .runtime import SandboxPaths, chmod_tree
from .sandbox import DockerSandbox, SandboxConfig
from .strategy import BarTable, FitSchedule, StrategyContext, StrategyFunction
from .strategy_loader import load_strategy_module, validate_strategy_package

if TYPE_CHECKING:
    from .tools.base import CommandResult

_HOST_TIMEOUT_BUFFER_SECONDS = 15.0
_MAX_STRATEGY_THREADS = 16
_PROCESS_STOP_TIMEOUT_SECONDS = 2.0
# Readiness handshake sequence. Request sequences are non-negative, so a worker
# that echoes this one is answering the probe and nothing else. The worker
# rejects the unknown message type and echoes the sequence with its error; that
# reply is the proof that the container is up, the interpreter is running and
# the strategy module finished importing. The one reply the worker emits
# *before* reading anything is its import failure, which carries no sequence —
# that is how a dead worker is told apart from a ready one.
_READY_SEQUENCE = -1
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
        self._fit_worker: DockerStrategyExecutor | None = None
        self.container_name = f"autotrade-strategy-{uuid.uuid4().hex}"
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
            "--cpus",
            f"{limits.cpus:g}",
            "--memory",
            limits.memory,
            "--pids-limit",
            str(limits.pids),
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
        ]
        thread_limit = str(max(1, min(_MAX_STRATEGY_THREADS, math.ceil(limits.cpus))))
        strategy_env = {
            "MKL_NUM_THREADS": thread_limit,
            "NUMEXPR_NUM_THREADS": thread_limit,
            "OMP_NUM_THREADS": thread_limit,
            "OPENBLAS_NUM_THREADS": thread_limit,
        }
        for key, value in sorted(strategy_env.items()):
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
                self.config,
                snapshot_dir=self.snapshot_dir,
                asof_dir=self.asof_dir,
                models_dir=self.models_dir,
                state_dir=self.state_dir,
                state_writable=True,
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
        try:
            # Materializing and validating the request is the host's own work on
            # host-owned PIT data; only what happens from the hand-over onwards
            # is the strategy's, so its clock starts here and not before.
            request, last_available_at = self._prepare_execute(context)
            request["type"] = kind
            sequence = request["sequence"]
            deadline = time.monotonic() + timeout_seconds
            self._send(request, deadline)
            while True:
                message, size = self._read_message(deadline)
                consumed += size
                if consumed > self.config.limits.max_output_chars:
                    raise StrategyExecutionError("strategy protocol output exceeded max_output_chars")
                message_type = message.get("type")
                response_sequence = message.get("sequence")
                if (message_type != "error" or "sequence" in message) and (
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
                        deadline += time.monotonic() - nl_started
                    self._send(payload, deadline)
                    continue
                if message_type == expected:
                    self._transport_sequence = sequence
                    self._transport_inference_at = context.inference_at
                    self._transport_bars = context.bars
                    self._transport_table = context._bars_table
                    self._transport_last_available_at = last_available_at
                    return message.get("orders")
                if message_type == "error":
                    raise StrategyExecutionError(str(message.get("error") or "strategy worker failed"))
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
                "type": "execute",
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

    def close(self) -> None:
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
        _require_local_image(self.config)
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


class ExecutorError(RuntimeError):
    pass


class LocalExecutor:
    """Host executor for explicit local development and unit tests."""

    name = "local"

    def __init__(self, paths: SandboxPaths, *, python: str | None = None) -> None:
        import sys
        self.paths = paths
        self.python = python or sys.executable

    def map_path(self, host_path: Path | str) -> str:
        return str(host_path)

    def runtime_path(self, host_path: Path | str) -> str:
        return str(host_path)

    def _base_env(self, env: dict[str, str] | None) -> dict[str, str]:
        base = {
            "PATH": (
                f"{self.paths.workspace}/.local/bin:"
                f"{self.paths.workspace}/.npm-global/bin:/usr/local/bin:/usr/bin:/bin"
            ),
            "HOME": str(self.paths.workspace),
            "PYTHONUSERBASE": str(self.paths.workspace / ".local"),
            "PIP_USER": "1",
            "npm_config_prefix": str(self.paths.workspace / ".npm-global"),
            "PYTHONDONTWRITEBYTECODE": "1",
        }
        if "PYTHONPATH" in os.environ:
            base["PYTHONPATH"] = os.environ["PYTHONPATH"]
        base.update(env or {})
        return base

    def run(self, argv: list[str], *, env: dict[str, str] | None = None, cwd: Path | None = None, timeout_seconds: float = 120.0, user: str = "agent", max_output_chars: int | None = None) -> ExecResult:
        del user
        if max_output_chars is not None:
            if isinstance(max_output_chars, bool) or max_output_chars <= 0:
                raise ValueError("max_output_chars must be a positive integer")
            return _run_limited_capture(
                argv,
                cwd=cwd or self.paths.agent,
                env=self._base_env(env),
                timeout_seconds=timeout_seconds,
                max_output_chars=max_output_chars,
            )
        try:
            completed = subprocess.run(argv, cwd=str(cwd or self.paths.agent), env=self._base_env(env), capture_output=True, text=True, errors="replace", timeout=timeout_seconds, check=False)
        except subprocess.TimeoutExpired as exc:
            return ExecResult(124, _limited_text(exc.stdout, 1_000_000), f"timeout after {timeout_seconds}s")
        return ExecResult(completed.returncode, completed.stdout, completed.stderr)

    def popen(self, argv: list[str], *, env: dict[str, str] | None = None, cwd: Path | None = None, user: str = "agent") -> subprocess.Popen[str]:
        del user
        return subprocess.Popen(argv, cwd=str(cwd or self.paths.agent), env=self._base_env(env), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, errors="replace")

    def kill_marker(self, marker: str, *, user: str = "agent") -> None:
        del marker, user

    def cleanup_user_processes(self, *, user: str = "agent") -> None:
        del user


class DockerExecutor:
    """Pipeline executor inside an existing persistent Sandbox container."""

    name = "docker"

    def __init__(self, container: str, host_paths: SandboxPaths, *, docker_executable: str = "docker", python: str = "python3", **_kwargs: object) -> None:
        self.container = container
        self.host_paths = host_paths
        self.docker_executable = docker_executable
        self.python = python

    def map_path(self, host_path: Path | str) -> str:
        path = Path(host_path).resolve()
        for base in (self.host_paths.current_snapshot, self.host_paths.snapshot):
            try:
                return str(Path("/mnt/snapshot") / path.relative_to(base.resolve()))
            except ValueError:
                pass
        try:
            return str(Path("/mnt") / path.relative_to(self.host_paths.root))
        except ValueError as exc:
            raise ExecutorError(f"path is outside sandbox root: {path}") from exc

    def runtime_path(self, host_path: Path | str) -> str:
        return str(Path("/opt/autotrade_runtime") / Path(host_path).name)

    @staticmethod
    def _merged_env(env: Mapping[str, str] | None) -> dict[str, str]:
        return {
            "PATH": "/mnt/agent/workspace/.local/bin:/mnt/agent/workspace/.npm-global/bin:/usr/local/bin:/usr/bin:/bin",
            "HOME": "/mnt/agent/workspace",
            "PYTHONUSERBASE": "/mnt/agent/workspace/.local",
            "PIP_USER": "1",
            "npm_config_prefix": "/mnt/agent/workspace/.npm-global",
            "PYTHONDONTWRITEBYTECODE": "1",
            **dict(env or {}),
        }

    def _command(self, argv: Sequence[str], *, env: Mapping[str, str] | None, cwd: Path | None, user: str) -> list[str]:
        command = [self.docker_executable, "exec", "-i", "--user", user]
        for key, value in sorted(self._merged_env(env).items()):
            command.extend(["--env", f"{key}={value}"])
        command.extend(["--workdir", self.map_path(cwd) if cwd else "/mnt/agent", self.container, *argv])
        return command

    def run(self, argv: list[str], *, env: dict[str, str] | None = None, cwd: Path | None = None, timeout_seconds: float = 120.0, user: str = "61000:61000", max_output_chars: int | None = None) -> ExecResult:
        command = self._command(
            _with_container_timeout(argv, timeout_seconds), env=env, cwd=cwd, user=user
        )
        host_timeout = timeout_seconds + _HOST_TIMEOUT_BUFFER_SECONDS
        if max_output_chars is not None:
            if isinstance(max_output_chars, bool) or max_output_chars <= 0:
                raise ValueError("max_output_chars must be a positive integer")
            return _run_limited_capture(
                command,
                timeout_seconds=host_timeout,
                max_output_chars=max_output_chars,
            )
        try:
            completed = subprocess.run(command, capture_output=True, text=True, errors="replace", timeout=host_timeout, check=False)
        except subprocess.TimeoutExpired as exc:
            return ExecResult(124, _limited_text(exc.stdout, 1_000_000), f"timeout after {timeout_seconds}s")
        return ExecResult(completed.returncode, completed.stdout, completed.stderr)

    def popen(self, argv: list[str], *, env: dict[str, str] | None = None, cwd: Path | None = None, user: str = "61000:61000") -> subprocess.Popen[str]:
        return subprocess.Popen(self._command(argv, env=env, cwd=cwd, user=user), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, errors="replace")

    def kill_marker(self, marker: str, *, user: str = "61000") -> None:
        subprocess.run([self.docker_executable, "exec", "--user", "0", self.container, "pkill", "-f", marker], capture_output=True, timeout=15, check=False)
        self.cleanup_user_processes(user=user)

    def cleanup_user_processes(self, *, user: str = "61000") -> None:
        subprocess.run([self.docker_executable, "exec", "--user", "0", self.container, "pkill", "-KILL", "-u", user], capture_output=True, timeout=15, check=False)


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


def _require_local_image(config: SandboxConfig) -> str:
    """Resolve Docker and reject absent local images without any pull attempt."""

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
    return executable


__all__ = [
    "CONTAINER_ASOF_DIR",
    "CONTAINER_MODELS_DIR",
    "CONTAINER_SNAPSHOT_DIR",
    "CONTAINER_STATE_DIR",
    "CONTAINER_STRATEGY_DIR",
    "DockerExecutor",
    "DockerStrategyExecutor",
    "ExecResult",
    "ExecutorError",
    "FittableStrategyExecutor",
    "LocalExecutor",
    "PersistentCommandRunner",
    "StrategyExecutionError",
    "StrategyExecutor",
    "TrustedStrategyExecutor",
    "docker_available",
]
