"""Container-side JSON-lines worker for one Agent-authored strategy."""

from __future__ import annotations

import contextlib
import json
import os
import sys
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path

from .strategy import (
    AccountSnapshot,
    BarTable,
    StrategyContext,
    StrategyContractError,
    _parse_datetime,
)
from .strategy_loader import load_strategy_module


class WorkerProtocolError(RuntimeError):
    pass


# The container's own cgroup, as the container sees it. The host cannot always
# read it from outside (a Docker daemon in another PID namespace exposes none),
# but the container always can, so every answer carries this reading.
CGROUP_ROOT = Path("/sys/fs/cgroup")
# Its memory high-water mark and ceiling: cgroup v2 keeps them at the root of
# the container's cgroup, cgroup v1 under the memory controller. The first
# layout whose high-water file reads is the one this host runs.
_CGROUP_MEMORY_FILES = (
    ("memory.peak", "memory.max"),
    ("memory/memory.max_usage_in_bytes", "memory/memory.limit_in_bytes"),
)


def container_resources() -> dict[str, int]:
    """This container's peaks so far, under the host's ``resources`` keys.

    Only what was read appears: a cgroup that exposes neither layout, a
    container without a memory limit and a strategy that never brought CUDA up
    leave their keys out rather than report a zero nobody measured.
    """

    readings: dict[str, int] = {}
    for peak_name, limit_name in _CGROUP_MEMORY_FILES:
        peak = _cgroup_int(peak_name)
        if peak is None:
            continue
        readings["peak_memory_bytes"] = peak
        limit = _cgroup_int(limit_name)
        if limit is not None:
            readings["memory_limit_bytes"] = limit
        break
    gpu = _torch_reserved_peak()
    if gpu is not None:
        readings["peak_gpu_memory_bytes"] = gpu
    return readings


def _cgroup_int(name: str) -> int | None:
    try:
        return int((CGROUP_ROOT / name).read_text(encoding="utf-8"))
    except (OSError, ValueError):  # absent, or v2's "max" for no limit
        return None


def _torch_reserved_peak() -> int | None:
    """High-water mark of the strategy's torch allocator, summed over devices.

    The driver lists a container's GPU processes by host pid, which nothing
    inside the container can map to its own, so the allocator's own counter is
    the one figure here that is certainly this container's. It is a floor: the
    CUDA context and memory other libraries or processes hold are not in it.
    Read only once the strategy has brought CUDA up itself; the worker never
    imports torch or touches a device.
    """

    torch = sys.modules.get("torch")
    if torch is None:
        return None
    try:
        if not torch.cuda.is_initialized():
            return None
        return sum(
            torch.cuda.max_memory_reserved(device)
            for device in range(torch.cuda.device_count())
        )
    except Exception:  # noqa: BLE001 - a broken torch state must not cost the call its answer
        return None


class Protocol:
    def __init__(self) -> None:
        self._input = sys.stdin
        # Keep the JSON protocol on a private duplicate of the original stdout
        # pipe, then redirect fd 1 itself to stderr before strategy code loads.
        # This covers Python prints, os.write(1, ...), and native extensions;
        # redirect_stdout alone cannot protect a line-oriented RPC channel.
        protocol_fd = os.dup(sys.stdout.fileno())
        os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
        self._output = os.fdopen(protocol_fd, "w", encoding="utf-8", buffering=1)

    def read(self) -> dict[str, object] | None:
        line = self._input.readline()
        if not line:
            return None
        value = json.loads(line)
        if not isinstance(value, dict):
            raise WorkerProtocolError("protocol message must be a JSON object")
        return value

    def write(self, value: Mapping[str, object]) -> None:
        line = json.dumps(value, ensure_ascii=False, allow_nan=False)
        self._output.write(line + "\n")
        self._output.flush()


_EXECUTE_KEYS = {
    "type",
    "sequence",
    "reset",
    "base_count",
    "total_count",
    "context",
    "bars",
}
_CONTEXT_KEYS = {
    "inference_at",
    "account",
    "snapshot_dir",
    "asof_dir",
    "asof_version",
    "state_dir",
    "models_dir",
    "pretrained_dir",
}
# ``execute`` runs generate_orders and answers ``orders``; ``fit`` runs the
# optional fit entrypoint over the same context shape and answers ``fitted``.
_CALL_KINDS = {"execute": "orders", "fit": "fitted"}


class BarStream:
    """Worker-owned append-only bars, independent of strategy module state."""

    def __init__(self) -> None:
        # Columnar and append-only: each delta becomes one chunk, so the
        # worker's memory grows by the delta's typed columns, not by a dict
        # per bar, and a rejected delta never touches the accepted table.
        self._table = BarTable()
        self._sequence = -1
        self._inference_at: datetime | None = None

    def context(self, message: Mapping[str, object]) -> StrategyContext:
        if set(message) != _EXECUTE_KEYS:
            raise WorkerProtocolError("execute message has missing or unsupported fields")
        if message.get("type") not in _CALL_KINDS:
            raise WorkerProtocolError("expected execute or fit message")
        sequence = _integer(message.get("sequence"), "sequence")
        base_count = _integer(message.get("base_count"), "base_count")
        total_count = _integer(message.get("total_count"), "total_count")
        reset = message.get("reset")
        if not isinstance(reset, bool):
            raise WorkerProtocolError("reset must be a boolean")
        if reset:
            if sequence != 0 or base_count != 0:
                raise WorkerProtocolError("reset execute must start at sequence=0 and base_count=0")
            table = BarTable()
            previous_inference = None
        else:
            if self._sequence < 0:
                raise WorkerProtocolError("first execute message must reset the bar stream")
            if sequence != self._sequence + 1:
                raise WorkerProtocolError("execute sequence is not contiguous")
            if base_count != len(self._table):
                raise WorkerProtocolError("base_count does not match worker bar state")
            table = self._table
            previous_inference = self._inference_at

        raw_context = message.get("context")
        if not isinstance(raw_context, dict) or set(raw_context) != _CONTEXT_KEYS:
            raise WorkerProtocolError("execute context has missing or unsupported fields")
        inference_at = _parse_datetime(raw_context.get("inference_at"), "inference_at")
        if previous_inference is not None and inference_at <= previous_inference:
            raise WorkerProtocolError("inference_at must increase monotonically")
        raw_delta = message.get("bars")
        if not isinstance(raw_delta, list):
            raise WorkerProtocolError("execute bars must be a JSON array")
        if total_count != base_count + len(raw_delta):
            raise WorkerProtocolError("total_count does not equal base_count plus delta")

        try:
            table = table.extended(
                raw_delta, inference_at=inference_at, decoded_json=True, monotonic=True
            )
        except StrategyContractError as exc:
            raise WorkerProtocolError(str(exc)) from exc

        try:
            context = StrategyContext(
                inference_at=inference_at,
                bars=table.rows(len(table)),
                account=AccountSnapshot.from_record(raw_context.get("account")),
                snapshot_dir=raw_context.get("snapshot_dir"),  # type: ignore[arg-type]
                asof_dir=raw_context.get("asof_dir"),  # type: ignore[arg-type]
                asof_version=raw_context.get("asof_version"),  # type: ignore[arg-type]
                state_dir=raw_context.get("state_dir"),  # type: ignore[arg-type]
                models_dir=raw_context.get("models_dir"),  # type: ignore[arg-type]
                pretrained_dir=raw_context.get("pretrained_dir"),  # type: ignore[arg-type]
            )
        except StrategyContractError as exc:
            raise WorkerProtocolError(str(exc)) from exc

        self._table = table
        self._sequence = sequence
        self._inference_at = inference_at
        return context


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise WorkerProtocolError(f"{name} must be a non-negative integer")
    return value


def run(strategy_path: str | Path) -> int:
    protocol = Protocol()
    stream = BarStream()
    try:
        with contextlib.redirect_stdout(sys.stderr):
            strategy = load_strategy_module(strategy_path)
    except Exception as exc:  # noqa: BLE001 - report import failures through the protocol
        # The only message written before the first request is read, and the
        # only one that carries no ``sequence``: that is how the host's
        # readiness handshake tells a dead worker from a running one.
        protocol.write({"type": "error", "error": f"strategy import failed: {exc}"})
        return 1
    while True:
        message: dict[str, object] | None = None
        try:
            message = protocol.read()
            if message is None or message.get("type") == "close":
                return 0
            kind = message.get("type")
            if kind not in _CALL_KINDS:
                # Reaching this line already proves the module import finished,
                # so the host probes readiness with an unknown message type and
                # reads the echoed sequence out of the error below.
                raise WorkerProtocolError("expected execute or fit message")
            if kind == "fit" and strategy.fit is None:
                raise WorkerProtocolError("strategy defines no fit(context)")
            context = stream.context(message)
            sequence = message["sequence"]
            with contextlib.redirect_stdout(sys.stderr):
                if kind == "fit":
                    strategy.fit(context)  # type: ignore[misc]
                    response: dict[str, object] = {}
                else:
                    response = {"orders": strategy.generate_orders(context)}
            protocol.write(
                {
                    "type": _CALL_KINDS[kind],
                    "sequence": sequence,
                    **response,
                    "resources": container_resources(),
                }
            )
        except Exception as exc:  # noqa: BLE001 - isolate each untrusted strategy call
            # Two failures share this wire and they are not the same
            # measurement. An exception out of strategy code is a reading of
            # the strategy; a message this worker cannot speak to was built by
            # the host, so it reads the environment. They travel as distinct
            # types because the host cannot tell them apart from the text, and
            # charging the second one to the candidate costs it a replay year
            # for a failure it did not cause.
            error: dict[str, object] = {
                "type": "protocol_error" if isinstance(exc, WorkerProtocolError) else "error",
                "error": str(exc),
                "resources": container_resources(),
            }
            if isinstance(message, Mapping):
                sequence = message.get("sequence")
                if isinstance(sequence, int) and not isinstance(sequence, bool):
                    error["sequence"] = sequence
            protocol.write(error)


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print("usage: python -m autotrade.environment.strategy_worker STRATEGY.py", file=sys.stderr)
        return 2
    return run(args[0])


if __name__ == "__main__":
    raise SystemExit(main())
