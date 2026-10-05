"""GPUs for sandbox containers: what the host has, and what a strategy may count on.

The Environment attaches the devices a container is given and never chooses
one. Choosing is one selection, ``pipelines.hitl_state.select_gpus``: whole
devices nobody holds memory on (``idle_gpus``) that no running arm claims.
The console takes an arm's cards there when it starts its worker, and every
container of that worker attaches exactly those; whatever else needs a card
-- a Paper book's morning run, an operator's replay -- takes its cards there
for one run and pins them the same way. A container that asks for a device
and was given none refuses to start.

The cards are shared with services outside this project and are never
reserved, so ``MIN_FREE_GPU_MEMORY_MIB`` is the memory a GPU strategy may
count on: a CUDA out-of-memory error whose own report shows the strategy had
less than that to work with is contention (``gpu_memory_contention``), not a
measurement of the strategy.
"""

from __future__ import annotations

import re
import subprocess
from collections.abc import Sequence

# The video memory a GPU strategy may count on while it runs. Sized for one
# quarterly sequence-model refit on a 46 GiB L20 (a measured 9 GiB peak) with
# headroom for a second concurrent candidate on the same card.
MIN_FREE_GPU_MEMORY_MIB = 12 * 1024
# Memory a device may show in use and still be idle. Any process that opened a
# CUDA context on it holds more (an idle strategy container's context is about
# 390 MiB on an L20, and an unused L20 reports 0), so a card above this carries
# somebody's work, however much of it is free.
IDLE_GPU_MEMORY_MIB = 256


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
    """[{index, name, memory_used_mib, memory_free_mib, memory_total_mib, utilization_pct, temperature_c}] from nvidia-smi."""
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.used,memory.free,memory.total,utilization.gpu,temperature.gpu", "--format=csv,noheader,nounits"],
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
        index, name, used, free, total, util, temp = (part.strip() for part in line.split(",", 6))
        gpus.append(
            {"index": int(index), "name": name, "memory_used_mib": int(used), "memory_free_mib": int(free),
             "memory_total_mib": int(total), "utilization_pct": _int_or_none(util), "temperature_c": _int_or_none(temp)}
        )
    if not gpus:
        raise GpuUnavailableError("nvidia-smi reported no GPUs")
    return gpus


def idle_gpus(*, require_name: str | None = None) -> list[int]:
    """Indexes of the matching devices nobody holds memory on, ascending.

    A card in use by anyone -- the local model service, an operator's replay
    -- is not idle however much of it is free. The converse is the caller's to
    know: a running arm whose card shows nothing between its batches still
    holds it (``pipelines.hitl_state.gpu_claims``).
    """
    return sorted(
        int(gpu["index"])
        for gpu in list_gpus()
        if (not require_name or require_name.lower() in str(gpu["name"]).lower())
        and int(gpu["memory_used_mib"]) < IDLE_GPU_MEMORY_MIB
    )


_SIZE_UNITS_MIB = {"bytes": 1 / 2**20, "B": 1 / 2**20, "KiB": 1 / 2**10, "MiB": 1.0, "GiB": 2**10, "TiB": 2**20}
_SIZE = r"([0-9]+(?:\.[0-9]+)?) (bytes|B|KiB|MiB|GiB|TiB)"
_CUDA_OOM_FREE = re.compile(r"CUDA out of memory\..*? of which " + _SIZE + r" is free", re.DOTALL)
_CUDA_OOM_OWN = re.compile(
    r"Of the allocated memory " + _SIZE + r" is allocated by PyTorch, and " + _SIZE
    + r" is reserved by PyTorch but unallocated"
)


def _mib(value: str, unit: str) -> float:
    return float(value) * _SIZE_UNITS_MIB[unit]


def gpu_memory_contention(error_text: str) -> str | None:
    """Why a CUDA out-of-memory error was contention, or ``None`` if it was not.

    Reads PyTorch's own out-of-memory report, taken by the allocator at the
    failing call: the memory the strategy had to work with is the device's free
    memory plus what this process's PyTorch allocator already holds. Below
    ``MIN_FREE_GPU_MEMORY_MIB`` the card was taken by someone else and the
    failure measures the environment; at or above it the strategy exhausted the
    memory it was promised, which is its own error. A report that does not carry
    both figures (another library, or a changed message format) is not judged
    and stays the strategy's error. Memory held by the same evaluation's other
    strategy container counts as someone else's, because the report cannot tell
    the two apart.
    """
    free = _CUDA_OOM_FREE.search(error_text)
    own = _CUDA_OOM_OWN.search(error_text)
    if free is None or own is None:
        return None
    available = _mib(*free.groups()) + _mib(*own.groups()[:2]) + _mib(*own.groups()[2:])
    if available >= MIN_FREE_GPU_MEMORY_MIB:
        return None
    return (
        f"CUDA out of memory with {available:.0f} MiB available to the strategy "
        f"(device free plus its own PyTorch pool), below the {MIN_FREE_GPU_MEMORY_MIB} MiB "
        "floor a GPU strategy is promised: the device was taken by another process"
    )
