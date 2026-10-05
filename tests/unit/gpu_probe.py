"""Stub the host's GPU roster.

The console admits and claims a GPU arm's devices against the live host
(``ExperimentManager.gpu_slots`` reads ``autotrade.environment.gpu.list_gpus``
through ``idle_gpus``), so a create request that passes parameter validation
depends on what is free on this machine. Tests that create an experiment for
some other reason wrap the request in ``stubbed_gpu_probe``, which replaces
only the ``nvidia-smi`` reading: the selection and the claims stay real. The
selectors themselves are covered in ``test_sandbox_runtime``, and the admission
and claim of the create route in ``test_webui_worker_lifecycle``.
"""

from __future__ import annotations

from collections.abc import Sequence
from unittest.mock import patch


def gpu_roster(idle: Sequence[int] = (0,), busy: Sequence[int] = ()) -> list[dict[str, object]]:
    """``list_gpus`` rows for L20 cards: ``idle`` ones nobody uses, ``busy``
    ones another process holds 10 GiB on."""

    return sorted(
        (
            {
                "index": index,
                "name": "NVIDIA L20",
                "memory_used_mib": used,
                "memory_free_mib": 45_460 - used,
                "memory_total_mib": 46_068,
                "utilization_pct": 0,
                "temperature_c": 40,
            }
            for indexes, used in ((idle, 0), (busy, 10_240))
            for index in indexes
        ),
        key=lambda row: int(row["index"]),
    )


def stubbed_gpu_probe(devices: Sequence[int] = (0,), *, busy: Sequence[int] = ()):
    """A context manager (or ``start()``-able patcher) reporting ``devices``
    idle and ``busy`` in use by somebody else."""

    return patch(
        "autotrade.environment.gpu.list_gpus", return_value=gpu_roster(devices, busy)
    )


__all__ = ["gpu_roster", "stubbed_gpu_probe"]
