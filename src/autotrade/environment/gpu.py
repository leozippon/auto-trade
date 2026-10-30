"""GPUs for sandbox containers: what the host has, and what a strategy may count on.

The Environment attaches the devices a container is given and never chooses
one. Choosing is one selection, ``pipelines.hitl_state.select_gpus``: devices
no running arm claims that are either empty (``idle_gpus``) or, for a caller
that names an admission margin, keep that much free (``shareable_gpus``).
The console takes an arm's cards there when it starts its worker, and every
container of that worker attaches exactly those; whatever else needs a card
-- a Paper book's morning run, an operator's replay -- takes its cards there
for one run and pins them the same way.

A card somebody else holds memory on is not idle, and only a caller that
names a margin -- the console, by default, with the free memory a GPU strategy
is promised (``host_min_free_gpu_memory_mib``) -- or that opts into sharing
(the Paper's morning run, with at least ``MIN_FREE_GPU_MEMORY_MIB`` free) takes
it. A container that asks for a device and was given none refuses to start.
Every card nvidia-smi reports is a candidate: no list of cards keeps the
project off another tenant's card, the margin does, and a running arm's claim
keeps everybody off the cards it holds.

The cards are shared -- by a batch's legs, and with services outside this
project -- and nothing reserves memory on them. A CUDA out-of-memory error
whose own report shows the strategy had less than ``MIN_FREE_GPU_MEMORY_MIB``
to work with is classed as contention (``gpu_memory_contention``), an
environment failure, rather than a measurement of the strategy.
"""

from __future__ import annotations

import os
import re
import subprocess
from collections.abc import Sequence

# The device model this host's sandboxes run on: a case-insensitive substring
# of the nvidia-smi name that every selection on the host filters by (the
# console's claims, a round's smoke, command-line runs). It is a property of
# the host, not of an experiment, so it is read from the environment the
# console and its workers share; unset, it is the L20 workstation's.
GPU_NAME_ENV = "AUTOTRADE_GPU_NAME"
DEFAULT_GPU_NAME = "L20"
# The free memory the console requires on a card it places a GPU arm on
# (``AUTOTRADE_GPU_MIN_FREE_GIB``, a positive whole number of GiB; unset means
# ``MIN_FREE_GPU_MEMORY_MIB``). It is an admission margin, not a cap: on a host
# whose cards all carry somebody else's work, an arm still starts on one that
# keeps this much free, and nothing reserves it. Read where the console makes
# its selections, because a console only runs the settings it started with.
MIN_FREE_GPU_GIB_ENV = "AUTOTRADE_GPU_MIN_FREE_GIB"


def host_gpu_name() -> str:
    """The device-name filter this host's GPU selections use."""

    return os.environ.get(GPU_NAME_ENV, "").strip() or DEFAULT_GPU_NAME


def host_min_free_gpu_memory_mib() -> int:
    """The free memory a card must keep for the console to place an arm on it."""

    raw = os.environ.get(MIN_FREE_GPU_GIB_ENV, "").strip()
    if not raw:
        return MIN_FREE_GPU_MEMORY_MIB
    try:
        gib = int(raw)
    except ValueError:
        gib = 0
    if gib <= 0:
        raise ValueError(
            f"{MIN_FREE_GPU_GIB_ENV} must be a positive whole number of GiB, got {raw!r}"
        )
    return gib * 1024


# The refund threshold for a CUDA out-of-memory failure: a strategy call that
# had less than this to work with is classed as an environment failure
# (``gpu_memory_contention``), so a batch leg's replay-years are refunded.
# Nothing reserves it and no strategy is capped at it: it is what one quarterly
# sequence-model refit on a 46 GiB L20 was measured to need (a 9 GiB peak), and
# it is also the console's default admission margin for a card it places an arm
# on (``host_min_free_gpu_memory_mib``) and what the Paper's morning run needs
# free on a card it shares (``shareable_gpus``).
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
    """[{index, name, memory_used_mib, memory_free_mib, memory_total_mib, utilization_pct, temperature_c}]
    from nvidia-smi, for every card it reports."""
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

    A card in use by anyone -- another tenant, the local model service, an
    operator's replay -- is not idle however much of it is free. The converse is the caller's to
    know: a running arm whose card shows nothing between its batches still
    holds it (``pipelines.hitl_state.gpu_claims``).
    """
    return sorted(
        int(gpu["index"])
        for gpu in list_gpus()
        if (not require_name or require_name.lower() in str(gpu["name"]).lower())
        and int(gpu["memory_used_mib"]) < IDLE_GPU_MEMORY_MIB
    )


def shareable_gpus(
    *, require_name: str | None = None, min_free_memory_mib: int | None = None
) -> list[int]:
    """Indexes of the matching devices that keep ``min_free_memory_mib`` free
    (``MIN_FREE_GPU_MEMORY_MIB`` when it is not given), the most free first:
    the cards a caller may share with somebody else's work and still have that
    much to work with. Whether a caller may share at all is the selection's to
    decide, as is whether a running arm claims the card."""
    floor = MIN_FREE_GPU_MEMORY_MIB if min_free_memory_mib is None else min_free_memory_mib
    rows = [
        gpu
        for gpu in list_gpus()
        if (not require_name or require_name.lower() in str(gpu["name"]).lower())
        and int(gpu["memory_free_mib"]) >= floor
    ]
    rows.sort(key=lambda gpu: (-int(gpu["memory_free_mib"]), int(gpu["index"])))
    return [int(gpu["index"]) for gpu in rows]


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
    ``MIN_FREE_GPU_MEMORY_MIB`` the rest of the card was held by other processes
    and the failure is classed as the environment's; at or above it the strategy
    ran out of the memory it had, which is its own error. A report that does not carry
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
        f"CUDA out of memory with {available:.0f} MiB available to this call "
        f"(device free plus its own PyTorch pool), under {MIN_FREE_GPU_MEMORY_MIB} MiB: "
        "other processes on the card, the batch's other legs included, held the rest, so "
        "the failure is classed as the environment's, not the strategy's (a batch_validate "
        "leg's replay-years are refunded)"
    )
