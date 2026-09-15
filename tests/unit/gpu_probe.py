"""Stub the console's create-time GPU admission.

``ExperimentManager._preflight`` admits an experiment's GPU request against
the live host through ``select_gpus`` (imported lazily from
``autotrade.environment.gpu``), so a create request that passes parameter
validation depends on what is free on this machine. Tests that create an
experiment for some other reason wrap the request in ``stubbed_gpu_probe``;
the probe itself is covered in ``test_sandbox_runtime``, and the refusal path
of the create route in ``test_webui_worker_lifecycle`` and
``test_pipeline_config`` with an explicit ``GpuUnavailableError``.
"""

from __future__ import annotations

from collections.abc import Sequence
from unittest.mock import patch


def stubbed_gpu_probe(devices: Sequence[int] = (0,)):
    """A context manager (or ``start()``-able patcher) admitting ``devices``."""

    return patch("autotrade.environment.gpu.select_gpus", return_value=list(devices))


__all__ = ["stubbed_gpu_probe"]
