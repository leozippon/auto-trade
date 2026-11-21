"""One bounded command: its runner and its result.

``run_limited_capture`` runs a command under a deadline and a cap on the
output it keeps, and builds the ``CommandResult`` of every sandbox command:
the session container runs each command through it
(``sandbox.DockerSandbox.exec_limited``, which ``PersistentCommandRunner``
adapts to the tool layer), and the tool layer (``tools.base``, which
re-exports the result) hands the result to the Agent. It imports neither at
run time, so both can import it.
"""

from __future__ import annotations

import os
import selectors
import subprocess
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .sandbox import DockerSandbox

# How much longer than a command's in-container ``timeout``
# (:func:`with_container_timeout`) the host waits before it kills the
# ``docker exec`` itself.
HOST_TIMEOUT_BUFFER_SECONDS = 15.0


@dataclass(frozen=True)
class CommandResult:
    exit_code: int
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    # The runner discarded output beyond its capture cap: the stream held
    # here is a prefix of what the command produced, not the whole of it.
    stdout_truncated: bool = False
    stderr_truncated: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.exit_code, bool) or not isinstance(self.exit_code, int):
            raise TypeError("exit_code must be an integer")

    def to_record(self) -> dict[str, object]:
        return {
            "exit_code": self.exit_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "timed_out": self.timed_out,
        }


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
        if not argv or any(not str(item) for item in argv):
            raise ValueError("argv must contain non-empty strings")
        return self.sandbox.exec_limited(
            tuple(map(str, argv)),
            cwd=cwd,
            timeout_seconds=timeout_seconds,
            max_output_chars=max_output_chars,
            input_text=input_text,
        )


def with_container_timeout(argv: Sequence[str], timeout_seconds: float) -> list[str]:
    return [
        "timeout",
        "--signal=TERM",
        "--kill-after=5",
        f"{float(timeout_seconds):g}",
        *map(str, argv),
    ]


def run_limited_capture(
    argv: Sequence[str],
    *,
    cwd: Path | str | None = None,
    env: Mapping[str, str] | None = None,
    timeout_seconds: float,
    max_output_chars: int,
    stdin: int = subprocess.DEVNULL,
) -> CommandResult:
    """Run a command while bounding retained stdout and stderr.

    ``stdin`` is handed to the command as is; an attached Docker command gets
    one that stays open while it runs (``sandbox.attached_stdin``). Exit code
    124 is a timeout, the host's own here or the container's ``timeout``
    (``with_container_timeout``).
    """

    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    if isinstance(max_output_chars, bool) or max_output_chars <= 0:
        raise ValueError("max_output_chars must be a positive integer")
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
    return CommandResult(
        return_code,
        stdout.decode("utf-8", errors="replace"),
        stderr.decode("utf-8", errors="replace"),
        timed_out=return_code == 124,
        stdout_truncated=stdout_truncated,
        stderr_truncated=stderr_truncated,
    )


def close_process_pipes(process: subprocess.Popen[bytes]) -> None:
    for pipe in (process.stdin, process.stdout, process.stderr):
        if pipe is not None and not pipe.closed:
            try:
                pipe.close()
            except OSError:
                pass


__all__ = [
    "HOST_TIMEOUT_BUFFER_SECONDS",
    "CommandResult",
    "PersistentCommandRunner",
    "close_process_pipes",
    "run_limited_capture",
    "with_container_timeout",
]
