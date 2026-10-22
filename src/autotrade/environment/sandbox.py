"""Configuration for the isolated daily-strategy container."""

from __future__ import annotations

import fcntl
import json
import math
import os
import re
import shutil
import subprocess
import uuid
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path

from .gpu import GpuUnavailableError, device_request, host_gpu_name
from .pretrained import pretrained_pin, require_pretrained_store
from .runtime import (
    ARTIFACT_TOP_LEVEL,
    RUNTIME_CACHE_DIR_NAMES,
    RUNTIME_CACHE_SUFFIXES,
    SandboxPaths,
    chmod_tree,
    new_id,
    utc_now_iso,
    write_json_atomic,
)

DEFAULT_IMAGE = "autotrade-sandbox:latest"


@contextmanager
def attached_stdin(data: bytes = b"") -> Iterator[int]:
    """The stdin of one attached ``docker run -i`` / ``docker exec -i``.

    The Docker client half-closes its attach stream once its stdin ends, and a
    daemon reached through a socket proxy that cannot half-close then drops
    everything the process writes afterwards: the command prints nothing and
    exits 0. So the stream must not end while the command runs. The yielded
    descriptor is a pipe holding ``data`` that stays open until the block
    exits, after the command returned; a payload that needs EOF gets it from
    an in-band byte count instead (``_EXEC_PAYLOAD``). ``data`` is buffered in
    the pipe up front, so no writer outlives the call; more than the pipe can
    be grown to hold fails here, explicitly.
    """

    read_fd, write_fd = os.pipe()
    try:
        if data:
            fcntl.fcntl(write_fd, fcntl.F_SETPIPE_SZ, len(data))
            view = memoryview(data)
            while view:
                view = view[os.write(write_fd, view):]
        yield read_fd
    finally:
        os.close(read_fd)
        os.close(write_fd)


def _unlock_owned_directories(path: Path) -> None:
    """Make host-owned directories removable without changing file modes.

    Snapshot directories are mode 0555. Unlinking their entries needs a
    writable parent. File modes stay untouched: a payload may be a hard link
    into a shared pit view, and chmod would change that shared inode.
    """

    for directory, _dirnames, _filenames in os.walk(path, topdown=False):
        try:
            os.chmod(directory, 0o700)
        except OSError:
            continue


def remove_sandbox_tree(path: Path) -> bool:
    """Remove one experiment's sandbox directory.

    Host-owned directories are unlocked and removed first. Files the container
    created belong to its subuid, so what remains is deleted from a
    root-in-userns container that can see those ids. Returns True when the
    tree is gone.
    """

    target = Path(path)

    def drop() -> None:
        if not target.exists():
            return
        _unlock_owned_directories(target)
        shutil.rmtree(target, ignore_errors=True)

    drop()
    if not target.exists():
        return True
    try:
        subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "--user",
                "0",
                "--network=none",
                "-v",
                f"{target}:/purge",
                DEFAULT_IMAGE,
                "sh",
                "-c",
                "rm -rf /purge/* /purge/.[!.]* /purge/..?*",
            ],
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        pass
    drop()
    return not target.exists()


# Trusted research tooling bind-mounted read-only into the Agent session: the
# signal screen is a self-contained script (numpy/pandas/pyarrow, all in the
# image), so shipping it is a mount, not an image rebuild. The mount path is
# the single source the experiment facts and docs quote.
SCREENING_TOOL_SOURCE = Path(__file__).resolve().parent / "screening" / "screen.py"
SCREENING_TOOL_MOUNT = "/mnt/tools/screen.py"
# Where the Agent session sees the pretrained-model store (``environment.pretrained``)
# when its arm pins one; a strategy sees the same store as ``context.pretrained_dir``.
PRETRAINED_SESSION_MOUNT = "/mnt/pretrained"

RUNTIME_ENV_SCHEMA_VERSION = 2
_MEMORY_LIMIT = re.compile(r"^[1-9][0-9]*(?:[kKmMgG])?$")
_IMAGE_TAG = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/+-]{0,200}$")


# Numeric-thread ceiling for BOTH containers. Without an explicit cap
# NumPy/OpenBLAS size their pools by the number of cores the HOST exposes
# (192 here), fill the container's PID slot and leave PyArrow/jemalloc unable
# to create the threads they need. The CPU quota is the real bound; this is
# the ceiling above which more threads only cost PIDs.
MAX_CONTAINER_THREADS = 16


def container_thread_env(cpus: float) -> dict[str, str]:
    """Numeric-library thread caps for a container with ``cpus`` of quota.

    One source for the Agent session container and the strategy container:
    the Agent runs the same pandas/pyarrow/duckdb stack under ``shell`` that a
    strategy runs under ``fit``, and the mounted refs promise both are capped.
    """

    limit = str(max(1, min(MAX_CONTAINER_THREADS, math.ceil(cpus))))
    return {
        "MKL_NUM_THREADS": limit,
        "NUMEXPR_NUM_THREADS": limit,
        "OMP_NUM_THREADS": limit,
        "OPENBLAS_NUM_THREADS": limit,
    }


@dataclass(frozen=True)
class SandboxLimits:
    """Per-container resources and per-inference protocol limits."""

    # The same CPU and memory boundary the Agent session container runs under
    # (``SandboxSpec``). One size for both means a session that profiles its
    # ``fit`` under ``shell`` measures against the very ceiling the formal
    # replay enforces, instead of extrapolating from its own container to a
    # larger one; the pair is published as the run facts ``budgets.strategy_cpus``
    # and ``budgets.strategy_memory_bytes``.
    # Both containers also pass ``memory`` as ``--memory-swap``, which is what
    # makes the cap a boundary rather than a slope: Docker's default grants a
    # container as much swap again, and the container then pages instead of
    # failing. With swap pinned there is no headroom above the cap at all, so
    # the cap has to be sized for what a ``fit`` actually needs rather than for
    # what today's arms happen to use. Measured over the replays on record:
    # rule-based arms peak at 3.8 GiB (48% of the former 8 GiB cap), but the
    # two fit-bearing ML arms (``alpha158_lgbm``, ``gru_ranker``) peaked
    # 6.70-7.60 GiB, i.e. 85-95% of it -- a model only slightly larger would be
    # an OOM kill rather than a slow replay, so the cap went to 16 GiB. An
    # in-process vLLM engine then took it to 32 GiB: a retrieval arm's
    # full-span leg peaked at 14.1 GiB of the 16 (cgroup reading, page cache
    # included) and its in-context path adds about 2.5 GiB. The cap is a
    # ceiling, not a reservation: with one session plus three concurrent
    # validation containers per experiment at the cap, four experiments would
    # hold 4 x 4 x 32 = 512 GiB, already past the 503 GiB host, which relies
    # on containers not filling at once. CPU stays at 8: strategy containers
    # use ~1.1 of the 8 cores and host CPU PSI is 0.
    cpus: float = 8.0
    memory: str = "32g"
    # A fork-bomb guard, not a budget, counting threads as well as processes:
    # one worker with torch, LightGBM and XGBoost thread pools at the
    # CPU-derived thread cap plus PyArrow's own pool needs well over the 64
    # the single-threaded NumPy era allowed, and a vLLM engine run tensor
    # parallel across two cards peaked at 280 (its engine and two GPU worker
    # processes), past the 256 this was before. The session container runs
    # the same engines under ``vllm serve`` and has the same guard.
    pids: int = 1024
    # Wall clock for one ``generate_orders(context)`` call (published as
    # ``strategy_inference_timeout_seconds``) and for one ``fit(context)``
    # call; the pipeline knob ``strategy_fit_timeout_seconds`` sets the latter.
    # Both start only once the worker holds the decision inputs, so neither
    # bills container scheduling or host-side request materialization to the
    # strategy. Both are runaway guards rather than budgets: the inference
    # cap is sized like the fit cap, at roughly 6x the heaviest measured call
    # on record (a rebalance day of a cross-sectional LightGBM arm, median
    # 60.7 s over 49 such days), so a genuine runaway still dies while host
    # contention does not decide a verdict.
    timeout_seconds: float = 360.0
    fit_timeout_seconds: float = 3600.0
    # Environment-side bound on bringing one strategy container up to its
    # request loop: container scheduling, interpreter start and the strategy
    # module import. Generous on purpose — a loaded host must not turn into a
    # strategy timeout — but finite, so a wedged Docker daemon or container
    # still fails explicitly instead of blocking the replay forever.
    startup_timeout_seconds: float = 900.0
    max_output_chars: int = 1_000_000
    # Scratch inside the read-only container: ``/tmp`` and ``/dev/shm``. Both
    # were sized for the single-threaded NumPy era (64 MB, and Docker's 64 MB
    # default for shared memory), which rules out multi-process training inside
    # ``fit`` altogether -- a torch ``DataLoader(num_workers>0)`` and joblib's
    # memmapped parallel backend both fail on a 64 MB ``/dev/shm``, and any
    # spill or intermediate parquet fails on a 64 MB ``/tmp``. They are charged
    # to the container's own ``memory`` cap, and only what it actually writes.
    tmpfs_size: str = "2g"
    shm_size: str = "2g"
    # The experiment's own GPU request, carried over from its ``SandboxSpec``
    # (``pipelines.worker._strategy_sandbox_from_spec``). ``fit(context)`` runs
    # in THIS container rather than in the Agent session, so an experiment that
    # asked for GPUs must reach it; 0 attaches no device and keeps the formal
    # replay CPU-only, which is what ``gpu_count=0`` experiments get.
    gpu_count: int = 0
    gpu_name_filter: str | None = None
    # The devices every container of this evaluation attaches: the arm's
    # claimed cards, or the ones another caller took for this run
    # (``pipelines.hitl_state.select_gpus``), or one of them for a leg of a
    # concurrent batch (``SandboxConfig.on_devices``). The Environment chooses
    # none, so a request for devices that names none here refuses to start.
    gpu_devices: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if isinstance(self.cpus, bool) or not math.isfinite(self.cpus) or self.cpus <= 0:
            raise ValueError("sandbox cpus must be a positive finite number")
        if not _MEMORY_LIMIT.fullmatch(self.memory):
            raise ValueError("sandbox memory must be a positive Docker memory limit")
        if isinstance(self.pids, bool) or not isinstance(self.pids, int) or self.pids <= 0:
            raise ValueError("sandbox pids must be a positive integer")
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError("sandbox timeout_seconds must be positive")
        if not math.isfinite(self.fit_timeout_seconds) or self.fit_timeout_seconds <= 0:
            raise ValueError("sandbox fit_timeout_seconds must be positive")
        if (
            not math.isfinite(self.startup_timeout_seconds)
            or self.startup_timeout_seconds <= 0
        ):
            raise ValueError("sandbox startup_timeout_seconds must be positive")
        if (
            isinstance(self.max_output_chars, bool)
            or not isinstance(self.max_output_chars, int)
            or self.max_output_chars <= 0
        ):
            raise ValueError("sandbox max_output_chars must be a positive integer")
        if not _MEMORY_LIMIT.fullmatch(self.tmpfs_size):
            raise ValueError("sandbox tmpfs_size must be a positive Docker memory limit")
        if not _MEMORY_LIMIT.fullmatch(self.shm_size):
            raise ValueError("sandbox shm_size must be a positive Docker memory limit")
        if (
            isinstance(self.gpu_count, bool)
            or not isinstance(self.gpu_count, int)
            or self.gpu_count < 0
        ):
            raise ValueError("sandbox gpu_count must be a non-negative integer")
        if not isinstance(self.gpu_devices, tuple) or any(
            isinstance(index, bool) or not isinstance(index, int) or index < 0
            for index in self.gpu_devices
        ):
            raise ValueError("sandbox gpu_devices must be a tuple of device indexes")
        if self.gpu_devices and len(self.gpu_devices) != self.gpu_count:
            raise ValueError("sandbox gpu_devices must pin exactly gpu_count devices")

    @property
    def memory_bytes(self) -> int:
        """``memory`` as the byte count the container's cgroup will carry.

        The same number a replay reports as ``memory_limit_bytes``, so the
        session compares its measured peak against the ceiling in the same
        unit instead of parsing a Docker suffix itself.
        """

        return _memory_limit_bytes(self.memory)


_MEMORY_SUFFIX_BYTES = {"k": 1024, "m": 1024**2, "g": 1024**3}


def _memory_limit_bytes(value: str) -> int:
    suffix = value[-1].lower()
    if suffix in _MEMORY_SUFFIX_BYTES:
        return int(value[:-1]) * _MEMORY_SUFFIX_BYTES[suffix]
    return int(value)


# The label every container an experiment starts carries, so the console can
# list and reclaim them by experiment: the session container and the strategy
# container of every replay alike.
EXPERIMENT_LABEL = "adm.experiment"


def experiment_container_labels(
    experiment_id: str, *, run_id: str | None = None
) -> dict[str, str]:
    """Docker labels that tie a container to its experiment (and run)."""

    labels = {EXPERIMENT_LABEL: experiment_id}
    if run_id:
        labels["adm.run"] = run_id
    return labels


@dataclass(frozen=True)
class SandboxConfig:
    """Fail-closed Docker strategy execution configuration."""

    image: str = DEFAULT_IMAGE
    limits: SandboxLimits = field(default_factory=SandboxLimits)
    docker_executable: str = "docker"
    # Rendered as ``--label`` on the strategy container; the experiment's
    # labels from ``experiment_container_labels``.
    labels: Mapping[str, str] = field(default_factory=dict)
    # The pretrained store the arm pinned (``environment.pretrained``): set,
    # every strategy container mounts it read-only after comparing its
    # manifest; empty, none mounts anything.
    pretrained_manifest_sha256: str = ""

    def __post_init__(self) -> None:
        _validate_explicit_image_tag(self.image)
        if not self.docker_executable.strip():
            raise ValueError("docker_executable must be non-empty")
        pretrained_pin(self.pretrained_manifest_sha256)

    def on_devices(self, devices: tuple[int, ...] | None) -> SandboxConfig:
        """This boundary with its containers attaching exactly ``devices``.

        A leg of a concurrent batch replays on one card of the arm's claim
        (``pipelines.session_tools.leg_cards``); None keeps every device the
        limits pin, which is what a replay running alone attaches.
        """

        if devices is None:
            return self
        return replace(
            self,
            limits=replace(self.limits, gpu_count=len(devices), gpu_devices=tuple(devices)),
        )


@dataclass(frozen=True)
class SandboxSpec:
    """Resource and capability boundary for one persistent Agent session."""

    image: str = DEFAULT_IMAGE
    build_generation_id: str = ""
    user: str = "61000:61000"
    network: str = "none"
    # One boundary with the strategy container: both must stay equal to
    # ``SandboxLimits.cpus``/``memory``, whose rationale carries the sizing.
    cpus: float = 8.0
    memory: str = "32g"
    # ``SandboxLimits.pids``, whose rationale carries the sizing.
    pids_limit: int = 1024
    tmpfs_size: str = "1g"
    # /dev/shm, sized like the strategy container's (``SandboxLimits.shm_size``)
    # for the same users: a model server the session runs across its cards
    # (``vllm serve`` with tensor parallel) and multi-process data loading in
    # a ``fit`` the session profiles. Docker's default is 64 MB.
    shm_size: str = "2g"
    # "auto" asks for gpu_count devices and names none yet: whoever starts the
    # session pins them first (an arm's worker the cards the console claimed,
    # ``pipelines.worker.load_worker_options``), and a session still on "auto"
    # refuses to start. An integer or list pins devices; None runs CPU-only.
    gpu: str | int | Sequence[int] | None = "auto"
    gpu_count: int = 1
    gpu_name_filter: str | None = field(default_factory=host_gpu_name)
    docker_executable: str = "docker"
    # The arm's pretrained-store pin; the session mounts the store at
    # ``PRETRAINED_SESSION_MOUNT`` only when it is set (see ``SandboxConfig``).
    pretrained_manifest_sha256: str = ""

    def __post_init__(self) -> None:
        _validate_explicit_image_tag(self.image)
        pretrained_pin(self.pretrained_manifest_sha256)
        if self.build_generation_id:
            try:
                generation = uuid.UUID(self.build_generation_id)
            except ValueError as exc:
                raise ValueError("sandbox build_generation_id must be UUID4") from exc
            if generation.version != 4 or str(generation) != self.build_generation_id:
                raise ValueError("sandbox build_generation_id must be canonical UUID4")
        if not self.user.strip() or not self.docker_executable.strip():
            raise ValueError("sandbox user and Docker executable must be non-empty")
        if self.network != "none":
            raise ValueError("ordinary Agent sandboxes require network='none'")
        if isinstance(self.cpus, bool) or not math.isfinite(self.cpus) or self.cpus <= 0:
            raise ValueError("sandbox cpus must be positive")
        if not all(
            _MEMORY_LIMIT.fullmatch(value)
            for value in (self.memory, self.tmpfs_size, self.shm_size)
        ):
            raise ValueError("invalid Docker memory limit")
        if (
            isinstance(self.pids_limit, bool)
            or not isinstance(self.pids_limit, int)
            or self.pids_limit <= 0
        ):
            raise ValueError("pids_limit must be a positive integer")
        if (
            isinstance(self.gpu_count, bool)
            or not isinstance(self.gpu_count, int)
            or self.gpu_count < 0
        ):
            raise ValueError("gpu_count cannot be negative")
        if self.gpu is not None and self.gpu_count <= 0:
            raise ValueError("gpu_count must be a positive integer when GPUs are requested")

    def to_record(self) -> dict[str, object]:
        return {
            "image_ref": self.image,
            "build_generation_id": self.build_generation_id or None,
            "user": self.user,
            "network": self.network,
            "cpus": self.cpus,
            "memory": self.memory,
            "pids_limit": self.pids_limit,
            "tmpfs_size": self.tmpfs_size,
            "shm_size": self.shm_size,
            "gpu": self.gpu,
            "gpu_count": self.gpu_count,
            "gpu_name_filter": self.gpu_name_filter,
            "pretrained_manifest_sha256": self.pretrained_manifest_sha256 or None,
        }


class LocalSandbox:
    """Host layout mounted by a persistent Agent container."""

    def __init__(self, root: str | Path) -> None:
        self.paths = SandboxPaths(Path(root))

    def prepare_layout(self) -> SandboxPaths:
        for path in (
            self.paths.snapshot_views, self.paths.current_snapshot,
            self.paths.parent_output, self.paths.parent_model_artifacts,
            self.paths.steps, self.paths.workspace,
            self.paths.runtime,
        ):
            path.mkdir(parents=True, exist_ok=True)
        self.paths.agent.chmod(0o555)
        self.paths.artifacts.chmod(0o755)
        self.paths.runtime.chmod(0o700)
        self.paths.workspace.chmod(0o777)
        self.write_runtime_env(mode="local")
        return self.paths

    def write_runtime_env(
        self,
        *,
        mode: str,
        sandbox_spec: SandboxSpec | None = None,
        image_probe: dict[str, object] | None = None,
    ) -> Path:
        if mode not in {"local", "docker"}:
            raise ValueError(f"unsupported runtime mode: {mode}")
        record = {
            "schema_version": RUNTIME_ENV_SCHEMA_VERSION,
            "generated_at": utc_now_iso(),
            "mode": mode,
            "network": sandbox_spec.network if sandbox_spec else ("host" if mode == "local" else "none"),
            "sandbox_spec": sandbox_spec.to_record() if sandbox_spec else None,
            "image": dict(image_probe or {}),
            "policy": {
                "ordinary_agent_network": "disabled",
                "external_source_control": "disabled",
                "install_packages_during_session": False,
            },
        }
        write_json_atomic(self.paths.runtime_env, record)
        self.paths.runtime_env.chmod(0o444)
        return self.paths.runtime_env

    def bind_snapshot_view(self, view_dir: Path) -> None:
        _replace_dir_contents(Path(view_dir), self.paths.current_snapshot)

    def collect_artifacts(self, dest_dir: Path) -> Path:
        """Collect runtime outputs into the host experiment run directory.

        Runtime separates trusted `/mnt/artifacts` from agent-writable
        `/mnt/agent`; the collected experiment directory keeps the historical
        flat layout for reports and ledgers. The formal working copies live
        inside the workspace (``workspace/output``, ``workspace/models``) and
        are collected with it; the formal record of a strategy is its revision
        in the artifact store, not this copy.
        """
        dest_dir.parent.mkdir(parents=True, exist_ok=True)
        if dest_dir.exists():
            raise FileExistsError(f"artifact collection target already exists: {dest_dir}")
        dest_dir.mkdir()
        for name in ARTIFACT_TOP_LEVEL:
            source = self.paths.artifacts / name
            if source.exists():
                _copy_path(source, dest_dir / name)
        if self.paths.host_run_manifest.exists():
            _copy_path(self.paths.host_run_manifest, dest_dir / "host_run_manifest.json")
        # ``workspace`` is collected best-effort — a single unreadable/special
        # file there is skipped and logged instead of aborting the collection.
        workspace_source = self.paths.agent / _AGENT_WORKSPACE
        if workspace_source.exists():
            try:
                _copy_path(workspace_source, dest_dir / _AGENT_WORKSPACE)
            except (OSError, shutil.Error) as exc:
                _record_collect_skip(dest_dir, _AGENT_WORKSPACE, exc)
        chmod_tree(dest_dir, file_mode=0o644, dir_mode=0o755)
        return dest_dir


def probe_image_runtime(image: str, *, docker_executable: str = "docker", timeout_seconds: float = 120.0) -> dict[str, object]:
    _validate_explicit_image_tag(image)
    script = "import json,platform; print(json.dumps({'python_version':platform.python_version()}))"
    with attached_stdin() as stdin:
        completed = subprocess.run(
            [docker_executable, "run", "--pull", "never", "--rm", "-i", "--network", "none", "--entrypoint", "python", image, "-c", script],
            stdin=stdin, capture_output=True, text=True, timeout=timeout_seconds, check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(f"sandbox image probe failed: {completed.stderr.strip()}")
    value = json.loads(completed.stdout)
    if not isinstance(value, dict):
        raise RuntimeError("sandbox image probe returned an invalid record")
    return value


# The workspace is shared through a bind mount, but the two sides are
# different users: `shell` runs as the container's `agent` (a subuid the host
# does not own), the typed writers (`tools.files`) run as the host project
# user. The host already keeps its own creations reachable from the container
# (0o666/0o777, `tools.files._SANDBOX_FILE_MODE`); a zero umask closes the
# reverse direction, so a file the Agent creates in `shell` stays editable by
# `write_file`/`edit_file` instead of arriving 0o644 and refusing them.
# The exec's own stdin stays open until it returns (``attached_stdin``), so the
# payload reads its input through `head -c "$0"`, the input's byte count: it
# gets exactly those bytes and then EOF, at once when there are none. The
# payload is handed to `sh` as positional parameters and is never re-parsed,
# and as the last command of the pipeline it owns the stdout, the stderr and
# the exit status (`timeout`'s 124 included).
_EXEC_PAYLOAD = ("sh", "-c", 'umask 0000; head -c "$0" | exec "$@"')


class DockerSandbox:
    """One persistent, network-disabled container for an Agent session."""

    def __init__(self, local: LocalSandbox, spec: SandboxSpec, labels: dict[str, str] | None = None) -> None:
        self.local = local
        self.spec = spec
        self.labels = {"adm.role": "agent-session", **dict(labels or {})}
        self.container = new_id("admsbx")
        self.session_id = new_id("sandbox_session")
        self.image_runtime: dict[str, object] = {}
        self.gpu_indices: list[int] = []
        # The verified store directory, resolved at start when the arm pins one.
        self.pretrained_dir: Path | None = None
        self._started = False

    def docker_command(self) -> list[str]:
        paths = self.local.paths
        command = [
            self.spec.docker_executable, "run", "--pull", "never", "--detach", "--init",
            "--name", self.container, "--network", "none", "--user", self.spec.user,
            # exec for Triton's kernel cache, as in the strategy container
            # (``executor.DockerStrategyExecutor.docker_command``).
            "--read-only", "--tmpfs", f"/tmp:rw,exec,nosuid,nodev,size={self.spec.tmpfs_size}",
            "--shm-size", self.spec.shm_size,
            # Swap ceiling equal to the memory cap: see ``SandboxLimits.memory``
            # for why the two are the same number.
            "--cpus", f"{self.spec.cpus:g}", "--memory", self.spec.memory,
            "--memory-swap", self.spec.memory,
            "--pids-limit", str(self.spec.pids_limit), "--ulimit", "core=0:0",
            "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
        ]
        for key, value in sorted(self.labels.items()):
            command.extend(["--label", f"{key}={value}"])
        if self.gpu_indices:
            command.extend(["--gpus", device_request(self.gpu_indices)])
        for key, value in (
            ("XDG_CACHE_HOME", "/tmp/cache"), ("PIP_CACHE_DIR", "/tmp/cache/pip"),
            ("HF_HOME", "/tmp/cache/hf"), ("MPLCONFIGDIR", "/tmp/cache/mpl"),
            # py_compile and compileall write bytecode whatever
            # PYTHONDONTWRITEBYTECODE says, and a __pycache__ left under
            # output/ or a candidate fails modification_check; the prefix
            # sends every .pyc to the tmpfs instead.
            ("PYTHONPYCACHEPREFIX", "/tmp/cache/pycache"),
            *sorted(container_thread_env(self.spec.cpus).items()),
        ):
            command.extend(["--env", f"{key}={value}"])
        command.extend([
            "--mount", f"type=bind,src={paths.current_snapshot},dst=/mnt/snapshot,readonly",
            "--mount", f"type=bind,src={paths.artifacts},dst=/mnt/artifacts,readonly",
            "--mount", f"type=bind,src={SCREENING_TOOL_SOURCE},dst={SCREENING_TOOL_MOUNT},readonly",
        ])
        if self.pretrained_dir is not None:
            command.extend([
                "--mount",
                f"type=bind,src={self.pretrained_dir},dst={PRETRAINED_SESSION_MOUNT},readonly",
            ])
        command.extend([
            "--mount", f"type=bind,src={paths.agent},dst=/mnt/agent",
            "--workdir", "/mnt/agent/workspace", self.spec.image, "sleep", "infinity",
        ])
        return command

    def start(self) -> str:
        if self._started:
            return self.container
        if self.spec.gpu is not None:
            if self.spec.gpu == "auto":
                raise GpuUnavailableError(
                    f"this session asks for {self.spec.gpu_count} GPU(s) and was given no "
                    "device: an arm's worker attaches the cards the console claimed for it, "
                    "and anything else takes its cards from pipelines.hitl_state.select_gpus"
                )
            if isinstance(self.spec.gpu, int):
                self.gpu_indices = [self.spec.gpu]
            elif isinstance(self.spec.gpu, str):
                self.gpu_indices = [int(item) for item in self.spec.gpu.split(",") if item.strip()]
            else:
                self.gpu_indices = [int(item) for item in self.spec.gpu]
        if self.spec.pretrained_manifest_sha256:
            self.pretrained_dir = require_pretrained_store(self.spec.pretrained_manifest_sha256)
        self.image_runtime = probe_image_runtime(
            self.spec.image,
            docker_executable=self.spec.docker_executable,
        )
        completed = subprocess.run(
            self.docker_command(), capture_output=True, text=True, timeout=120, check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(f"failed to start persistent sandbox: {completed.stderr.strip()}")
        self._started = True
        self.local.write_runtime_env(
            mode="docker",
            sandbox_spec=self.spec,
            image_probe={
                "image_ref": self.spec.image,
                "build_generation_id": self.spec.build_generation_id or None,
                "session_id": self.session_id,
                "runtime": self.image_runtime,
            },
        )
        return self.container

    def exec(
        self,
        argv: Sequence[str],
        *,
        cwd: str = ".",
        timeout_seconds: float = 30.0,
        input_text: str | None = None,
    ) -> subprocess.CompletedProcess[str]:
        data = (input_text or "").encode("utf-8")
        command = self._exec_command(argv, cwd=cwd, input_bytes=len(data))
        with attached_stdin(data) as stdin:
            return subprocess.run(
                command,
                stdin=stdin, capture_output=True, text=True, errors="replace",
                timeout=timeout_seconds, check=False,
            )

    def exec_limited(
        self,
        argv: Sequence[str],
        *,
        cwd: str = ".",
        timeout_seconds: float = 30.0,
        max_output_chars: int,
        input_text: str | None = None,
    ):
        from .executor import (
            _HOST_TIMEOUT_BUFFER_SECONDS,
            _run_limited_capture,
            _with_container_timeout,
        )

        data = (input_text or "").encode("utf-8")
        command = self._exec_command(
            _with_container_timeout(argv, timeout_seconds), cwd=cwd, input_bytes=len(data)
        )
        with attached_stdin(data) as stdin:
            return _run_limited_capture(
                command,
                timeout_seconds=timeout_seconds + _HOST_TIMEOUT_BUFFER_SECONDS,
                max_output_chars=max_output_chars,
                stdin=stdin,
            )

    def _exec_command(
        self, argv: Sequence[str], *, cwd: str, input_bytes: int
    ) -> list[str]:
        if not self._started:
            raise RuntimeError("persistent sandbox is not started")
        relative = Path(cwd)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("sandbox cwd must stay inside the workspace")
        work = Path("/mnt/agent/workspace") / relative
        return [
            self.spec.docker_executable,
            "exec",
            "--user",
            self.spec.user,
            "--workdir",
            str(work),
            "-i",
            self.container,
            *_EXEC_PAYLOAD,
            str(input_bytes),
            *map(str, argv),
        ]

    def stop(self) -> None:
        if not self._started:
            return
        completed = subprocess.run(
            [self.spec.docker_executable, "rm", "--force", self.container],
            capture_output=True, text=True, timeout=60, check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(f"failed to destroy persistent sandbox: {completed.stderr.strip()}")
        self._started = False


def link_copytree(source: str | Path, dest: str | Path) -> Path:
    source, dest = Path(source), Path(dest)
    if dest.exists():
        chmod_tree(dest, file_mode=0o644, dir_mode=0o755)
        shutil.rmtree(dest)
    shutil.copytree(source, dest, copy_function=_link_or_copy)
    return dest


def _link_or_copy(source: str, dest: str) -> None:
    try:
        os.link(source, dest)
    except OSError:
        shutil.copy2(source, dest)


# Transient caches/tooling dirs are scratch, not experiment artifacts. They are
# also often written by the container user with restrictive perms (e.g. pip's
# 0600 cache), which the host collector cannot read; archiving them is both
# wrong and a copy failure. Excluded from artifact collection.
_COLLECT_IGNORE = shutil.ignore_patterns(
    ".cache",
    ".nv",
    "core.[0-9]*",  # PID-suffixed core dumps (RLIMIT_CORE=0 prevents these; belt-and-suspenders)
    *RUNTIME_CACHE_DIR_NAMES,  # __pycache__ (shared with artifacts._is_runtime_cache)
    *(f"*{_suffix}" for _suffix in RUNTIME_CACHE_SUFFIXES),  # *.pyc, *.pyo
    ".git",
    ".hg",
    ".svn",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".ipynb_checkpoints",
    "node_modules",
    ".venv",
    ".conda",
    ".npm",
)


# The single agent-writable tree under /mnt/agent.
_AGENT_WORKSPACE = "workspace"


def _copy_path(source: Path, dest: Path) -> None:
    if source.is_dir():
        shutil.copytree(source, dest, symlinks=True, ignore=_COLLECT_IGNORE)
    else:
        shutil.copy2(source, dest)


def _record_collect_skip(dest_dir: Path, name: str, exc: Exception) -> None:
    """Record a best-effort collection skip so a partially-collected workspace is
    visible in the run directory rather than silently dropped."""
    try:
        (dest_dir / f"{name}.collect_error.txt").write_text(
            f"{type(exc).__name__}: {exc}\n", encoding="utf-8"
        )
    except OSError:
        pass


def _replace_dir_contents(source: Path, dest: Path) -> None:
    if not source.is_dir():
        raise FileNotFoundError(source)
    dest.mkdir(parents=True, exist_ok=True)
    for child in list(dest.iterdir()):
        shutil.rmtree(child) if child.is_dir() and not child.is_symlink() else child.unlink()
    for child in source.iterdir():
        target = dest / child.name
        shutil.copytree(child, target, copy_function=_link_or_copy) if child.is_dir() else _link_or_copy(str(child), str(target))
    chmod_tree(dest, file_mode=0o444, dir_mode=0o555)


def _validate_explicit_image_tag(image: str) -> None:
    value = image.strip()
    final_segment = value.rsplit("/", maxsplit=1)[-1]
    if not _IMAGE_TAG.fullmatch(value) or "@" in value or ":" not in final_segment:
        raise ValueError("sandbox image must be an explicit local tag")


__all__ = [
    "DEFAULT_IMAGE", "EXPERIMENT_LABEL", "MAX_CONTAINER_THREADS", "PRETRAINED_SESSION_MOUNT",
    "SCREENING_TOOL_MOUNT",
    "SCREENING_TOOL_SOURCE", "DockerSandbox", "LocalSandbox", "SandboxConfig", "SandboxLimits",
    "SandboxSpec", "attached_stdin", "container_thread_env", "experiment_container_labels", "link_copytree",
    "probe_image_runtime",
]
