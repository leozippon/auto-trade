"""Configuration for the isolated daily-strategy container."""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import stat
import subprocess
import uuid
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from .artifacts import (
    copy_artifact,
    copy_model_artifacts,
    init_from_template,
    make_formal_artifacts_readonly,
    restore_formal_artifacts_writable,
)
from .runtime import (
    AGENT_TOP_LEVEL,
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
DEFAULT_HOST_FRACTION = 0.10

RUNTIME_ENV_SCHEMA_VERSION = 2
_MEMORY_LIMIT = re.compile(r"^[1-9][0-9]*(?:[kKmMgG])?$")
_IMAGE_TAG = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/+-]{0,200}$")


@dataclass(frozen=True)
class SandboxLimits:
    """Per-container resources and per-inference protocol limits."""

    cpus: float = 8.0
    memory: str = "16g"
    pids: int = 64
    timeout_seconds: float = 30.0
    max_output_chars: int = 1_000_000
    tmpfs_size: str = "64m"

    def __post_init__(self) -> None:
        if isinstance(self.cpus, bool) or not math.isfinite(self.cpus) or self.cpus <= 0:
            raise ValueError("sandbox cpus must be a positive finite number")
        if not _MEMORY_LIMIT.fullmatch(self.memory):
            raise ValueError("sandbox memory must be a positive Docker memory limit")
        if isinstance(self.pids, bool) or not isinstance(self.pids, int) or self.pids <= 0:
            raise ValueError("sandbox pids must be a positive integer")
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError("sandbox timeout_seconds must be positive")
        if (
            isinstance(self.max_output_chars, bool)
            or not isinstance(self.max_output_chars, int)
            or self.max_output_chars <= 0
        ):
            raise ValueError("sandbox max_output_chars must be a positive integer")
        if not _MEMORY_LIMIT.fullmatch(self.tmpfs_size):
            raise ValueError("sandbox tmpfs_size must be a positive Docker memory limit")


@dataclass(frozen=True)
class SandboxConfig:
    """Fail-closed Docker strategy execution configuration."""

    image: str = DEFAULT_IMAGE
    limits: SandboxLimits = field(default_factory=SandboxLimits)
    docker_executable: str = "docker"

    def __post_init__(self) -> None:
        _validate_explicit_image_tag(self.image)
        if not self.docker_executable.strip():
            raise ValueError("docker_executable must be non-empty")


@dataclass(frozen=True)
class SandboxSpec:
    """Resource and capability boundary for one persistent Agent session."""

    image: str = DEFAULT_IMAGE
    build_generation_id: str = ""
    user: str = "61000:61000"
    network: str = "none"
    cpus: float = 4.0
    memory: str = "8g"
    pids_limit: int = 512
    tmpfs_size: str = "1g"
    # "auto" allocates gpu_count matching GPUs with the most free memory at
    # container start; an integer or list pins devices; None runs CPU-only.
    gpu: str | int | Sequence[int] | None = "auto"
    gpu_count: int = 1
    gpu_name_filter: str | None = "L20"
    docker_executable: str = "docker"

    def __post_init__(self) -> None:
        _validate_explicit_image_tag(self.image)
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
        if not _MEMORY_LIMIT.fullmatch(self.memory) or not _MEMORY_LIMIT.fullmatch(self.tmpfs_size):
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

    @classmethod
    def from_host_fraction(cls, fraction: float = DEFAULT_HOST_FRACTION, **overrides: object) -> SandboxSpec:
        if not 0 < fraction <= 1:
            raise ValueError("fraction must be in (0, 1]")
        cpus = max(1.0, round((os.cpu_count() or 4) * fraction, 1))
        with Path("/proc/meminfo").open(encoding="ascii") as stream:
            total_kib = int(next(line for line in stream if line.startswith("MemTotal:")).split()[1])
        memory = f"{max(1, int(total_kib / 1024 / 1024 * fraction))}g"
        return cls(cpus=cpus, memory=memory, **overrides)  # pyright: ignore[reportArgumentType]

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
            "gpu": self.gpu,
            "gpu_count": self.gpu_count,
            "gpu_name_filter": self.gpu_name_filter,
        }


class LocalSandbox:
    """Host layout mounted by a persistent Agent container."""

    def __init__(self, root: str | Path) -> None:
        self.paths = SandboxPaths(Path(root))

    def prepare_layout(self) -> SandboxPaths:
        for path in (
            self.paths.train, self.paths.valid, self.paths.test, self.paths.snapshot_views,
            self.paths.current_snapshot, self.paths.parent_output, self.paths.parent_model_artifacts,
            self.paths.results, self.paths.steps, self.paths.workspace,
            self.paths.agent_output, self.paths.model_artifacts, self.paths.runtime,
        ):
            path.mkdir(parents=True, exist_ok=True)
        self.paths.agent.chmod(0o555)
        self.paths.artifacts.chmod(0o755)
        self.paths.runtime.chmod(0o700)
        self.paths.test.chmod(0o700)
        for path in (self.paths.workspace, self.paths.agent_output, self.paths.model_artifacts):
            path.chmod(0o777)
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

    def install_strategy_artifact(
        self,
        source_root: Path | None,
        template_dir: Path,
        *,
        source_model_root: Path | None = None,
    ) -> bool:
        if source_root is None:
            init_from_template(template_dir, self.paths.parent_output)
            init_from_template(template_dir, self.paths.agent_output)
            is_initial = True
        else:
            copy_artifact(source_root, self.paths.parent_output)
            copy_artifact(source_root, self.paths.agent_output)
            is_initial = False
        copy_model_artifacts(source_model_root, self.paths.parent_model_artifacts)
        copy_model_artifacts(source_model_root, self.paths.model_artifacts)
        chmod_tree(self.paths.parent_output, file_mode=0o444, dir_mode=0o555)
        chmod_tree(self.paths.parent_model_artifacts, file_mode=0o444, dir_mode=0o555)
        self.unlock_agent_output()
        return is_initial

    def install_replay_slot(self, slot: str, source_dir: Path) -> Path:
        if slot not in {"train", "valid", "test"}:
            raise ValueError(f"unknown replay slot: {slot}")
        target = getattr(self.paths, slot)
        link_copytree(source_dir, target)
        target.chmod(0o700 if slot == "test" else 0o755)
        return target

    def bind_snapshot_view(self, view_dir: Path) -> None:
        _replace_dir_contents(Path(view_dir), self.paths.current_snapshot)

    def bind_formal_snapshot_view(self, view_dir: Path) -> None:
        source = Path(view_dir).resolve()
        allowed = (self.paths.snapshot_views.resolve(), self.paths.current_snapshot.resolve())
        if not source.is_dir() or not any(source == root or source.is_relative_to(root) for root in allowed):
            raise ValueError("formal snapshot view is outside sandbox snapshot roots")
        self._bind_snapshot_selector(self.paths.formal_snapshot, source)

    @staticmethod
    def _bind_snapshot_selector(link: Path, source: Path) -> None:
        if link.is_symlink() or link.exists():
            if link.is_dir() and not link.is_symlink():
                raise ValueError(f"snapshot selector must be a symlink, found directory: {link}")
            link.unlink()
        link.symlink_to(source.resolve(), target_is_directory=True)

    def lock_agent_output(self) -> None:
        make_formal_artifacts_readonly(self.paths)

    def unlock_agent_output(self) -> None:
        restore_formal_artifacts_writable(self.paths)

    def collect_artifacts(self, dest_dir: Path) -> Path:
        """Collect runtime outputs into the host experiment run directory.

        Runtime separates trusted `/mnt/artifacts` from agent-writable
        `/mnt/agent`; the collected experiment directory keeps the historical
        flat layout for reports and ledgers.
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
        # Collect official output/ and models/ FIRST so an uncollectable file
        # later in the adversarial workspace tree cannot displace them.
        # Prefer workspace/<name> when that directory has any file (the pipeline
        # working copy); otherwise fall back to the /mnt/agent sibling.
        # ``workspace`` is collected LAST and best-effort — a single
        # unreadable/special file there is skipped and logged instead of
        # aborting the whole collection.
        for name in AGENT_TOP_LEVEL:
            if name == _AGENT_WORKSPACE:
                continue
            workspace_copy = self.paths.workspace / name
            sibling = self.paths.agent / name
            source = (
                workspace_copy
                if workspace_copy.is_dir()
                and any(path.is_file() for path in workspace_copy.rglob("*"))
                else sibling
            )
            if source.exists():
                _copy_path(source, dest_dir / name)
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
    completed = subprocess.run(
        [docker_executable, "run", "--pull", "never", "--rm", "--network", "none", "--entrypoint", "python", image, "-c", script],
        capture_output=True, text=True, timeout=timeout_seconds, check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"sandbox image probe failed: {completed.stderr.strip()}")
    value = json.loads(completed.stdout)
    if not isinstance(value, dict):
        raise RuntimeError("sandbox image probe returned an invalid record")  # noqa: TRY004
    return value


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
        self._started = False

    def docker_command(self) -> list[str]:
        paths = self.local.paths
        command = [
            self.spec.docker_executable, "run", "--pull", "never", "--detach", "--init",
            "--name", self.container, "--network", "none", "--user", self.spec.user,
            "--read-only", "--tmpfs", f"/tmp:rw,nosuid,nodev,size={self.spec.tmpfs_size}",
            "--cpus", f"{self.spec.cpus:g}", "--memory", self.spec.memory,
            "--pids-limit", str(self.spec.pids_limit), "--ulimit", "core=0:0",
            "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
        ]
        for key, value in sorted(self.labels.items()):
            command.extend(["--label", f"{key}={value}"])
        if self.gpu_indices:
            command.extend(["--gpus", f"device={','.join(map(str, self.gpu_indices))}"])
        for key, value in (
            ("XDG_CACHE_HOME", "/tmp/cache"), ("PIP_CACHE_DIR", "/tmp/cache/pip"),
            ("HF_HOME", "/tmp/cache/hf"), ("MPLCONFIGDIR", "/tmp/cache/mpl"),
        ):
            command.extend(["--env", f"{key}={value}"])
        command.extend([
            "--mount", f"type=bind,src={paths.train},dst=/mnt/snapshots/train,readonly",
            "--mount", f"type=bind,src={paths.valid},dst=/mnt/snapshots/valid,readonly",
            "--mount", f"type=bind,src={paths.current_snapshot},dst=/mnt/snapshot,readonly",
            "--mount", f"type=bind,src={paths.artifacts},dst=/mnt/artifacts,readonly",
            "--mount", f"type=bind,src={paths.agent},dst=/mnt/agent",
            "--workdir", "/mnt/agent/workspace", self.spec.image, "sleep", "infinity",
        ])
        return command

    def start(self) -> str:
        if self._started:
            return self.container
        if self.spec.gpu is not None:
            from .gpu import select_gpus
            if self.spec.gpu == "auto":
                self.gpu_indices = select_gpus(self.spec.gpu_count, require_name=self.spec.gpu_name_filter)
            elif isinstance(self.spec.gpu, int):
                self.gpu_indices = [self.spec.gpu]
            elif isinstance(self.spec.gpu, str):
                self.gpu_indices = [int(item) for item in self.spec.gpu.split(",") if item.strip()]
            else:
                self.gpu_indices = [int(item) for item in self.spec.gpu]
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
        command = self._exec_command(argv, cwd=cwd)
        return subprocess.run(
            command,
            input=input_text, capture_output=True, text=True, errors="replace",
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

        command = self._exec_command(
            _with_container_timeout(argv, timeout_seconds), cwd=cwd
        )
        return _run_limited_capture(
            command,
            timeout_seconds=timeout_seconds + _HOST_TIMEOUT_BUFFER_SECONDS,
            max_output_chars=max_output_chars,
            input_text=input_text,
        )

    def _exec_command(self, argv: Sequence[str], *, cwd: str) -> list[str]:
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
            *map(str, argv),
        ]

    def allocation_record(self) -> dict[str, object]:
        return {
            "container": self.container,
            "session_id": self.session_id,
            "image_runtime": dict(self.image_runtime),
            "allocated_gpu_indices": list(self.gpu_indices),
            **self.spec.to_record(),
        }

    @contextmanager
    def formal_guard(self) -> Iterator[None]:
        if not self._started:
            raise RuntimeError("persistent sandbox is not started")
        paused = subprocess.run(
            [self.spec.docker_executable, "pause", self.container],
            capture_output=True, text=True, timeout=30, check=False,
        )
        if paused.returncode != 0:
            raise RuntimeError(f"failed to pause sandbox: {paused.stderr.strip()}")
        try:
            yield
        finally:
            resumed = subprocess.run(
                [self.spec.docker_executable, "unpause", self.container],
                capture_output=True, text=True, timeout=30, check=False,
            )
            if resumed.returncode != 0:
                self.stop()
                raise RuntimeError("sandbox could not be safely unpaused and was destroyed")

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
    "TODO.json",
)


# The single agent-writable top-level tree; everything else under /mnt/agent
# (output/, models/) is a controlled, chmod-locked artifact.
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


@contextmanager
def hide_snapshot_slots_from_agent(paths: SandboxPaths) -> Iterator[None]:
    previous: list[tuple[Path, int]] = []
    for path in (paths.train, paths.valid, paths.test, paths.artifacts):
        if path.exists():
            previous.append((path, stat.S_IMODE(path.stat().st_mode)))
            path.chmod(0o700)
    try:
        yield
    finally:
        for path, mode in previous:
            path.chmod(mode)


def _validate_explicit_image_tag(image: str) -> None:
    value = image.strip()
    final_segment = value.rsplit("/", maxsplit=1)[-1]
    if not _IMAGE_TAG.fullmatch(value) or "@" in value or ":" not in final_segment:
        raise ValueError("sandbox image must be an explicit local tag")


__all__ = [
    "DEFAULT_HOST_FRACTION", "DEFAULT_IMAGE", "DockerSandbox",
    "LocalSandbox", "SandboxConfig", "SandboxLimits", "SandboxSpec", "hide_snapshot_slots_from_agent",
    "link_copytree", "probe_image_runtime",
]
