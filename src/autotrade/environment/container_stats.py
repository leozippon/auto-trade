"""Host-side resource telemetry of one strategy container.

A strategy container runs with ``--rm``, so nothing about it survives its exit:
``docker stats`` reports the current moment and ``docker inspect`` has nothing
left to read once the replay ends. Its cgroup does carry the high-water mark
(``memory.peak`` on cgroup v2) for as long as the container exists, so the
monitor resolves that directory once while the container is up and reads the
peak once at its end -- one ``docker inspect`` per container start and two file
reads per container end, whatever the replay costs.

Video memory has no cgroup counterpart and the worker protocol reports none, so
a GPU strategy's peak is sampled from the host instead: ``nvidia-smi`` lists the
video memory each compute process holds and the container's own cgroup lists
which host pids are its. A sampled peak can miss a spike between two samples, so
it is a floor on what the container used, never an upper bound.

Telemetry never decides a result, so an unreadable cgroup or an absent
``nvidia-smi`` leaves the field out rather than failing the replay. It is
reported as measured or not at all -- never as a zero that was never observed.
"""

from __future__ import annotations

import subprocess
import threading
from pathlib import Path

HOST_CGROUP_ROOT = Path("/sys/fs/cgroup")
# How often a GPU container's own compute processes are polled. One
# ``nvidia-smi`` call per sample per container: at three concurrent candidates
# with a fit worker each this is well under one call a second, and it resolves
# a fit long enough to be worth measuring.
GPU_SAMPLE_INTERVAL_SECONDS = 15.0
_DOCKER_TIMEOUT_SECONDS = 15.0


class ContainerResourceMonitor:
    """Peak memory (and, when asked, peak video memory) of one container.

    ``start`` binds the monitor to the running container and ``stop`` takes the
    final reading; ``record`` then returns what was measured, and only what was
    measured. Both are idempotent, so an executor that aborts and then closes
    reads the peak once, before the container is removed.
    """

    def __init__(
        self,
        container_name: str,
        *,
        docker_executable: str = "docker",
        sample_gpu: bool = False,
    ) -> None:
        self.container_name = container_name
        self.docker_executable = docker_executable
        self.sample_gpu = sample_gpu
        self._cgroup: Path | None = None
        self._peak_memory: int | None = None
        self._memory_limit: int | None = None
        self._peak_gpu_memory: int | None = None
        self._stopped = threading.Event()
        self._sampler: threading.Thread | None = None

    def start(self) -> None:
        self._cgroup = self._resolve_cgroup()
        if self._cgroup is None or not self.sample_gpu:
            return
        self._sampler = threading.Thread(target=self._sample_gpu_loop, daemon=True)
        self._sampler.start()

    def stop(self) -> None:
        if self._stopped.is_set():
            return
        # Order matters: the peak must be read before the caller removes the
        # container, and the sampler must not race that last read.
        self._stopped.set()
        self._peak_memory = self._read_cgroup_int("memory.peak")
        self._memory_limit = self._read_cgroup_int("memory.max")
        sampler = self._sampler
        if sampler is not None and sampler is not threading.current_thread():
            sampler.join(timeout=_DOCKER_TIMEOUT_SECONDS)
        self._sampler = None

    def record(self) -> dict[str, int]:
        return {
            name: value
            for name, value in (
                ("peak_memory_bytes", self._peak_memory),
                ("memory_limit_bytes", self._memory_limit),
                ("peak_gpu_memory_bytes", self._peak_gpu_memory),
            )
            if value is not None
        }

    # ---- cgroup ----

    def _resolve_cgroup(self) -> Path | None:
        """The container's own cgroup directory on this host.

        Taken from the container process's ``/proc/<pid>/cgroup`` rather than
        from a guessed layout, so it holds under either cgroup driver and under
        rootless Docker, where the scope sits below the user slice.
        """

        try:
            inspected = subprocess.run(
                [
                    self.docker_executable,
                    "inspect",
                    "--format",
                    "{{.State.Pid}}",
                    self.container_name,
                ],
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=_DOCKER_TIMEOUT_SECONDS,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        if inspected.returncode != 0:
            return None
        try:
            pid = int(inspected.stdout.strip())
            relative = (
                Path(f"/proc/{pid}/cgroup")
                .read_text(encoding="utf-8")
                .strip()
                .rsplit(":", 1)[-1]
            )
        except (OSError, ValueError, IndexError):
            return None
        if pid <= 0 or not relative.startswith("/"):
            return None
        directory = HOST_CGROUP_ROOT / relative.lstrip("/")
        return directory if (directory / "memory.peak").is_file() else None

    def _read_cgroup_int(self, name: str) -> int | None:
        if self._cgroup is None:
            return None
        try:
            text = (self._cgroup / name).read_text(encoding="utf-8").strip()
        except OSError:
            return None
        try:
            # ``memory.max`` reads "max" when the container has no limit.
            return int(text)
        except ValueError:
            return None

    def _container_pids(self) -> set[int]:
        if self._cgroup is None:
            return set()
        try:
            text = (self._cgroup / "cgroup.procs").read_text(encoding="utf-8")
        except OSError:
            return set()
        pids: set[int] = set()
        for line in text.split():
            try:
                pids.add(int(line))
            except ValueError:
                continue
        return pids

    # ---- GPU ----

    def _sample_gpu_loop(self) -> None:
        while True:
            self._sample_gpu_once()
            if self._stopped.wait(GPU_SAMPLE_INTERVAL_SECONDS):
                return

    def _sample_gpu_once(self) -> None:
        pids = self._container_pids()
        if not pids:
            return
        usage = compute_app_memory_bytes()
        mine = [usage[pid] for pid in pids if pid in usage]
        if not mine:
            # Nothing of this container is on a device right now. That is not
            # a measurement of zero: the peak stays where it was.
            return
        total = sum(mine)
        self._peak_gpu_memory = max(self._peak_gpu_memory or 0, total)


def compute_app_memory_bytes() -> dict[int, int]:
    """Video memory each CUDA compute process currently holds, by host pid.

    Empty when ``nvidia-smi`` is unavailable or reports nothing, which reads the
    same as "no process was observed" -- the caller records no peak either way.
    """

    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=_DOCKER_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return {}
    if completed.returncode != 0:
        return {}
    usage: dict[int, int] = {}
    for line in completed.stdout.strip().splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 2:
            continue
        try:
            pid, used_mib = int(parts[0]), int(parts[1])
        except ValueError:  # "[N/A]" on some drivers
            continue
        usage[pid] = usage.get(pid, 0) + used_mib * 1024 * 1024
    return usage


__all__ = [
    "GPU_SAMPLE_INTERVAL_SECONDS",
    "ContainerResourceMonitor",
    "compute_app_memory_bytes",
]
