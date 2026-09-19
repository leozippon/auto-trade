"""The research worker entry applies its process-level settings before the libraries load."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "experiments" / "run_interactive_experiment.py"
SETTINGS = ("ARROW_DEFAULT_MEMORY_POOL", "MALLOC_MMAP_THRESHOLD_", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS")


def test_worker_entry_selects_the_system_arrow_pool_and_caps_blas_threads() -> None:
    probe = (
        "import os, runpy, sys\n"
        f"runpy.run_path({str(SCRIPT)!r}, run_name='worker_entry')\n"
        "import pyarrow as pa\n"
        "print(pa.default_memory_pool().backend_name, "
        + ", ".join(f"os.environ[{name!r}]" for name in SETTINGS)
        + ")\n"
    )
    env = {name: value for name, value in os.environ.items() if name not in SETTINGS}
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True, env=env
    )
    assert result.stdout.split() == ["system", "system", "131072", "4", "4"]
