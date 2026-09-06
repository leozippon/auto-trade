"""GPU selection for sandbox containers.

Default policy: allocate the requested number of GPUs with the most free video
memory at container start, optionally restricted to a device-name substring
(``SandboxSpec.gpu_name_filter`` is the single configuration source). A one-GPU
sandbox is the normal case, but the same selector supports wider ML experiments
without changing Docker plumbing.
"""

from __future__ import annotations

import subprocess
from collections.abc import Sequence


def device_request(indices: Sequence[int]) -> str:
    """The ``--gpus`` value that pins exactly ``indices``.

    Docker splits this value on commas, so a bare ``device=0,1`` is read as one
    device id plus a device *count* and the request is refused ("cannot set
    both Count and DeviceIDs on device request"). The documented form quotes
    the whole value; the quotes belong to the argument itself, not to a shell,
    so they stay in the argv element. Single-device requests use the same form.
    """

    return f'"device={",".join(str(int(index)) for index in indices)}"'


def _int_or_none(value: str) -> int | None:
    try:
        return int(value)
    except ValueError:  # "[N/A]" on some drivers
        return None


class GpuUnavailableError(RuntimeError):
    pass


def list_gpus() -> list[dict[str, object]]:
    """[{index, name, memory_free_mib, memory_total_mib, utilization_pct, temperature_c}] from nvidia-smi."""
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.free,memory.total,utilization.gpu,temperature.gpu", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise GpuUnavailableError(f"nvidia-smi not available: {exc}") from exc
    if completed.returncode != 0:
        raise GpuUnavailableError(f"nvidia-smi failed: {completed.stderr.strip()[:200]}")
    gpus = []
    for line in completed.stdout.strip().splitlines():
        index, name, free, total, util, temp = (part.strip() for part in line.split(",", 5))
        gpus.append(
            {"index": int(index), "name": name, "memory_free_mib": int(free), "memory_total_mib": int(total),
             "utilization_pct": _int_or_none(util), "temperature_c": _int_or_none(temp)}
        )
    if not gpus:
        raise GpuUnavailableError("nvidia-smi reported no GPUs")
    return gpus


def select_gpus(count: int = 1, *, require_name: str | None = None) -> list[int]:
    """GPU indexes sorted by descending free memory.

    ``require_name`` restricts selection to devices whose name contains the
    substring (case-insensitive); ``None`` allows any visible NVIDIA GPU.
    """
    if count <= 0:
        raise ValueError(f"count must be positive: {count}")
    gpus = list_gpus()
    if require_name:
        gpus = [gpu for gpu in gpus if require_name.lower() in str(gpu["name"]).lower()]
    if len(gpus) < count:
        names = ", ".join(str(gpu["name"]) for gpu in gpus) or "none"
        raise GpuUnavailableError(f"requested {count} GPU(s), available matching GPUs: {names}")
    selected = sorted(gpus, key=lambda gpu: int(gpu["memory_free_mib"]), reverse=True)[:count]
    return [int(gpu["index"]) for gpu in selected]
