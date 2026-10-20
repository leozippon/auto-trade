"""GPUs for sandbox containers: what the host has, and what a strategy may count on.

The Environment attaches the devices a container is given and never chooses
one. Choosing is one selection, ``pipelines.hitl_state.select_gpus``: whole
devices nobody holds memory on (``idle_gpus``) that no running arm claims.
The console takes an arm's cards there when it starts its worker, and every
container of that worker attaches exactly those; whatever else needs a card
-- a Paper book's morning run, an operator's replay -- takes its cards there
for one run and pins them the same way. Only the Paper run may opt into a
card it shares when none is idle (``shareable_gpus``). A container that asks
for a device and was given none refuses to start. Cards the host keeps for
other users are never read at all (``host_gpu_devices``).

The cards are shared with services outside this project and are never
reserved, so ``MIN_FREE_GPU_MEMORY_MIB`` is the memory a GPU strategy may
count on: a CUDA out-of-memory error whose own report shows the strategy had
less than that to work with is contention (``gpu_memory_contention``), not a
measurement of the strategy.
"""

from __future__ import annotations

import os
import re
import subprocess
from collections.abc import Sequence
from pathlib import Path

from .llm.openai_compatible import load_env_value

# The device model this host's sandboxes run on: a case-insensitive substring
# of the nvidia-smi name that every selection on the host filters by (the
# console's claims, a round's smoke, command-line runs). It is a property of
# the host, not of an experiment, so it is read from the environment the
# console and its workers share; unset, it is the L20 workstation's.
GPU_NAME_ENV = "AUTOTRADE_GPU_NAME"
DEFAULT_GPU_NAME = "L20"


def host_gpu_name() -> str:
    """The device-name filter this host's GPU selections use."""

    return os.environ.get(GPU_NAME_ENV, "").strip() or DEFAULT_GPU_NAME

# The devices this project may use on a host it shares with other users: a
# comma-separated list of nvidia-smi indices. It is applied where the devices
# are read (``list_gpus``), so no selection, claim or view ever sees another
# user's card, idle or not. Unlike the model name it is also read from the
# repository's .env, so the console, its workers and every command-line run on
# the host apply it however they were launched; unset or empty, every device
# is the project's.
GPU_DEVICES_ENV = "AUTOTRADE_GPU_DEVICES"
HOST_ENV_FILE = Path(__file__).resolve().parents[3] / ".env"

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


def host_gpu_devices() -> frozenset[int] | None:
    """The device indices this project may use on this host, ``None`` for all.

    A value that is not a list of indices refuses rather than allowing every
    card."""

    value = load_env_value(GPU_DEVICES_ENV, HOST_ENV_FILE)
    if not value:
        return None
    parts = [part.strip() for part in value.split(",")]
    if not all(re.fullmatch(r"[0-9]+", part) for part in parts):
        raise GpuUnavailableError(
            f"{GPU_DEVICES_ENV}={value!r} is not a comma-separated list of nvidia-smi indices"
        )
    return frozenset(int(part) for part in parts)


def list_gpus() -> list[dict[str, object]]:
    """[{index, name, memory_used_mib, memory_free_mib, memory_total_mib, utilization_pct, temperature_c}]
    from nvidia-smi, for the devices this project may use (``host_gpu_devices``)."""
    allowed = host_gpu_devices()
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
    return [gpu for gpu in gpus if allowed is None or gpu["index"] in allowed]


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


def shareable_gpus(*, require_name: str | None = None) -> list[int]:
    """Indexes of the matching devices that keep ``MIN_FREE_GPU_MEMORY_MIB``
    free, the most free first: the cards a GPU strategy may share with
    somebody else's work and still have the memory it is promised. Whether a
    caller may share at all is the selection's to decide, as is whether a
    running arm claims the card."""
    rows = [
        gpu
        for gpu in list_gpus()
        if (not require_name or require_name.lower() in str(gpu["name"]).lower())
        and int(gpu["memory_free_mib"]) >= MIN_FREE_GPU_MEMORY_MIB
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
