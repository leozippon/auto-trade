"""Derived sandbox image lifecycle (meta-learning ``sandbox_environment.json``).

Meta-learning may request stable new dependencies for later ordinary Folds by
writing ``workspace/sandbox_environment.json``. This module owns the whole
image-extension domain: request validation, Dockerfile rendering (with a
build-time import smoke test), the ``docker build``, UUID4 build generations,
and best-effort GC of stale derived images. The pipeline only wires config
knobs in and records the result.
"""

from __future__ import annotations

import fcntl
import json
import os
import re
import shlex
import subprocess
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import cast

from .runtime import utc_now_iso, write_json_atomic
from .sandbox import SandboxSpec, probe_image_runtime

SANDBOX_ENVIRONMENT_REQUEST_NAME = "sandbox_environment.json"
SANDBOX_ENVIRONMENT_EXAMPLE_NAME = "sandbox_environment.example.json"
SANDBOX_IMAGE_STATE_NAME = "sandbox_image.json"
_SANDBOX_ENVIRONMENT_EXAMPLE = {
    "python_packages": [],
    "apt_packages": [],
    "npm_packages": [],
    "reason": (
        "Copy this example to sandbox_environment.json only when later ordinary Folds "
        "need stable new dependencies."
    ),
    "notes": (
        "Do not include shell commands, URLs, tokens, cache paths, local files, "
        "or temporary exploration artifacts."
    ),
}

_PYTHON_PACKAGE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-\[\],<>=!~:+]*$")
_SYSTEM_PACKAGE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.+-]*$")
_NPM_PACKAGE_RE = re.compile(
    r"^(?:@[A-Za-z0-9][A-Za-z0-9_.-]*/)?[A-Za-z0-9][A-Za-z0-9_.-]*(?:@[A-Za-z0-9][A-Za-z0-9_.+~^-]*)?$"
)
_DOCKER_IMAGE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/+-]{0,200}$")


def write_sandbox_environment_example(workspace: Path) -> Path:
    path = Path(workspace) / SANDBOX_ENVIRONMENT_EXAMPLE_NAME
    path.write_text(
        json.dumps(_SANDBOX_ENVIRONMENT_EXAMPLE, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def _image_state_path(experiment_dir: Path) -> Path:
    return Path(experiment_dir) / "hitl" / SANDBOX_IMAGE_STATE_NAME


@contextmanager
def _image_state_lock(experiment_dir: Path) -> Iterator[None]:
    lock_path = _image_state_path(Path(experiment_dir)).with_name(
        f".{SANDBOX_IMAGE_STATE_NAME}.lock"
    )
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _load_image_state(path: Path, *, experiment_id: str) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid persisted sandbox image state: {path}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(  # noqa: TRY004
            f"invalid persisted sandbox image state: {path}"
        )
    image_ref = payload.get("image_ref")
    generation_id = payload.get("build_generation_id")
    if payload.get("experiment_id") != experiment_id:
        raise RuntimeError("persisted sandbox image belongs to another experiment")
    if not isinstance(image_ref, str) or not isinstance(generation_id, str):
        raise RuntimeError(  # noqa: TRY004
            "persisted sandbox image lacks required identity fields"
        )
    try:
        generation = uuid.UUID(generation_id)
    except ValueError as exc:
        raise RuntimeError("persisted sandbox build generation is not a UUID") from exc
    if generation.version != 4 or str(generation) != generation_id:
        raise RuntimeError("persisted sandbox build generation is not canonical UUID4")
    # SandboxSpec validates the explicit local-tag syntax. Unknown legacy
    # fields are deliberately ignored and never participate in resume.
    SandboxSpec(image=image_ref, build_generation_id=generation_id)
    owned_value = payload.get("owned_image_refs")
    if owned_value is None:
        owned_image_refs = [image_ref]
    elif (
        not isinstance(owned_value, list)
        or not owned_value
        or not all(isinstance(item, str) and item for item in owned_value)
    ):
        raise RuntimeError("persisted sandbox image ownership list is invalid")
    else:
        owned_image_refs = list(dict.fromkeys(owned_value))
    for owned_ref in owned_image_refs:
        SandboxSpec(image=owned_ref)
    if image_ref not in owned_image_refs:
        raise RuntimeError("active sandbox image is absent from its ownership list")
    return {
        "image_ref": image_ref,
        "build_generation_id": generation_id,
        "owned_image_refs": owned_image_refs,
    }


def _new_image_ref(
    experiment_id: str,
    *,
    purpose: str,
    docker_executable: str,
) -> tuple[str, str]:
    for _ in range(8):
        generation_id = str(uuid.uuid4())
        image_ref = (
            f"autotrade-sandbox:{_docker_tag_component(experiment_id)}-"
            f"{_docker_tag_component(purpose)}-{generation_id}"
        )
        existing = subprocess.run(
            [
                docker_executable,
                "image",
                "inspect",
                "--format",
                "{{json .RepoTags}}",
                image_ref,
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=30,
            check=False,
        )
        if existing.returncode != 0:
            return generation_id, image_ref
    raise RuntimeError("could not allocate an unused sandbox image UUID4 tag")


def _write_image_state(
    path: Path,
    *,
    experiment_id: str,
    image_ref: str,
    build_generation_id: str,
    base_image_ref: str,
    kind: str,
    runtime: dict[str, object],
    owned_image_refs: list[str],
) -> None:
    write_json_atomic(
        path,
        {
            "schema_version": 1,
            "experiment_id": experiment_id,
            "image_ref": image_ref,
            "build_generation_id": build_generation_id,
            "base_image_ref": base_image_ref,
            "kind": kind,
            "runtime": dict(runtime),
            "owned_image_refs": list(dict.fromkeys(owned_image_refs)),
            "created_at": utc_now_iso(),
        },
    )


def _remove_image_ref(image_ref: str, *, docker_executable: str) -> None:
    subprocess.run(
        [docker_executable, "image", "rm", image_ref],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=30,
        check=False,
    )


def prepare_experiment_sandbox_image(
    base_spec: SandboxSpec,
    *,
    experiment_id: str,
    experiment_dir: Path,
) -> SandboxSpec:
    """Persist one immutable UUID4-tagged base image for an experiment.

    The first preparation clones the configured local tag and smoke-tests it
    offline. Resumes read only the persisted unique tag; legacy identity fields
    are ignored and Docker image IDs or registry digests are never consulted.
    A host administrator deleting or retargeting that tag is an explicit
    controlled-host limitation; the application adds no content check.
    """
    experiment_dir = Path(experiment_dir)
    state_path = _image_state_path(experiment_dir)
    with _image_state_lock(experiment_dir):
        if state_path.exists():
            state = _load_image_state(state_path, experiment_id=experiment_id)
            return replace(
                base_spec,
                image=str(state["image_ref"]),
                build_generation_id=str(state["build_generation_id"]),
            )
        generation_id, image_ref = _new_image_ref(
            experiment_id,
            purpose="base",
            docker_executable=base_spec.docker_executable,
        )
        tagged = subprocess.run(
            [base_spec.docker_executable, "tag", base_spec.image, image_ref],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        if tagged.returncode != 0:
            raise RuntimeError(
                f"failed to clone configured sandbox image {base_spec.image!r} "
                f"to experiment image {image_ref!r}"
            )
        try:
            runtime = probe_image_runtime(
                image_ref, docker_executable=base_spec.docker_executable
            )
        except Exception:
            _remove_image_ref(image_ref, docker_executable=base_spec.docker_executable)
            raise
        _write_image_state(
            state_path,
            experiment_id=experiment_id,
            image_ref=image_ref,
            build_generation_id=generation_id,
            base_image_ref=base_spec.image,
            kind="base_clone",
            runtime=runtime,
            owned_image_refs=[image_ref],
        )
        return replace(
            base_spec,
            image=image_ref,
            build_generation_id=generation_id,
        )


def maybe_rebuild_sandbox_image(
    request_path: Path,
    *,
    base_spec: SandboxSpec,
    experiment_id: str,
    epoch_id: str,
    experiment_dir: Path,
    manifest,
    use_docker: bool,
    rebuild_enabled: bool,
    timeout_seconds: int,
    image_keep: int = 3,
) -> tuple[dict[str, object] | None, SandboxSpec]:
    with _image_state_lock(Path(experiment_dir)):
        state_path = _image_state_path(Path(experiment_dir))
        if state_path.exists():
            state = _load_image_state(state_path, experiment_id=experiment_id)
            base_spec = replace(
                base_spec,
                image=str(state["image_ref"]),
                build_generation_id=str(state["build_generation_id"]),
            )
        return _maybe_rebuild_sandbox_image(
            request_path,
            base_spec=base_spec,
            experiment_id=experiment_id,
            epoch_id=epoch_id,
            experiment_dir=experiment_dir,
            manifest=manifest,
            use_docker=use_docker,
            rebuild_enabled=rebuild_enabled,
            timeout_seconds=timeout_seconds,
            image_keep=image_keep,
        )


def _maybe_rebuild_sandbox_image(
    request_path: Path,
    *,
    base_spec: SandboxSpec,
    experiment_id: str,
    epoch_id: str,
    experiment_dir: Path,
    manifest,
    use_docker: bool,
    rebuild_enabled: bool,
    timeout_seconds: int,
    image_keep: int = 3,
) -> tuple[dict[str, object] | None, SandboxSpec]:
    """Build a derived image from a meta-learning environment request.

    Returns ``(result_record, active_spec)``: the spec switches to the new
    image tag only on a successful build. Every outcome (rejected, skipped,
    timeout, failed, ok) is recorded into the run ``manifest`` before a hard
    failure is raised, so the audit trail survives the fail-fast."""
    request_path = Path(request_path)
    if not request_path.exists():
        return None, base_spec
    request_ref = f"/mnt/agent/workspace/{request_path.name}"
    try:
        request = _load_sandbox_environment_request(request_path)
    except ValueError as exc:
        result = {"status": "rejected", "reason": str(exc), "request_ref": request_ref}
        manifest.update(sandbox_image_update=result)
        raise RuntimeError(f"meta-learning sandbox environment request rejected: {exc}") from exc
    if not _environment_request_has_packages(request):
        result = {"status": "skipped_empty", "request_ref": request_ref}
        manifest.update(sandbox_image_update=result)
        return result, base_spec
    if not use_docker:
        result = {"status": "skipped_local_dev", "request_ref": request_ref}
        manifest.update(sandbox_image_update=result)
        return result, base_spec
    if not rebuild_enabled:
        result = {"status": "disabled", "request_ref": request_ref}
        manifest.update(sandbox_image_update=result)
        return result, base_spec

    state_path = _image_state_path(Path(experiment_dir))
    existing_state = (
        _load_image_state(state_path, experiment_id=experiment_id)
        if state_path.exists()
        else None
    )
    owned_image_refs = (
        list(cast(list[str], existing_state["owned_image_refs"]))
        if existing_state is not None
        else []
    )

    # Every request gets a UUID4 generation and an unpublished unique tag.
    # The active experiment state changes only after build and offline smoke
    # both succeed, so failed attempts cannot displace the resumable image.
    build_generation_id, image_ref = _new_image_ref(
        experiment_id,
        purpose=_docker_tag_component(epoch_id),
        docker_executable=base_spec.docker_executable,
    )
    build_dir = (
        Path(experiment_dir)
        / "sandbox_images"
        / _docker_tag_component(epoch_id)
        / build_generation_id
    )
    build_dir.mkdir(parents=True, exist_ok=False)
    dockerfile = build_dir / "Dockerfile"
    try:
        dockerfile_text = _render_sandbox_extension_dockerfile(base_spec.image, request)
    except ValueError as exc:
        result = {
            "status": "rejected",
            "reason": str(exc),
            "request_ref": request_ref,
            "base_image_ref": base_spec.image,
            "build_generation_id": build_generation_id,
        }
        manifest.update(sandbox_image_update=result)
        raise RuntimeError(f"meta-learning sandbox image rebuild rejected: {exc}") from exc
    dockerfile.write_text(dockerfile_text, encoding="utf-8")
    request_copy = build_dir / SANDBOX_ENVIRONMENT_REQUEST_NAME
    request_copy.write_text(
        json.dumps(request, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    command = [base_spec.docker_executable, "build", "--network=host"]
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY"):
        if os.environ.get(name):
            # Docker's predefined proxy build args are omitted from image
            # history. Passing the name without a value forwards this process's
            # environment without copying credentials into the command/manifest.
            command.extend(["--build-arg", name])
    command.extend(["--tag", image_ref, "--file", str(dockerfile), str(build_dir)])
    started_at = utc_now_iso()
    try:
        completed = subprocess.run(
            command, capture_output=True, text=True, timeout=timeout_seconds, check=False,
        )
    except subprocess.TimeoutExpired as exc:
        result = {
            "status": "timeout",
            "request_ref": request_ref,
            "host_request_ref": str(request_copy),
            "dockerfile_ref": str(dockerfile),
            "base_image_ref": base_spec.image,
            "image_ref": image_ref,
            "build_generation_id": build_generation_id,
            "started_at": started_at,
            "finished_at": utc_now_iso(),
            "timeout_seconds": timeout_seconds,
        }
        manifest.update(sandbox_image_update=result)
        raise RuntimeError(f"meta-learning sandbox image rebuild timed out: {image_ref}") from exc
    result: dict[str, object] = {
        "status": "ok" if completed.returncode == 0 else "failed",
        "request_ref": request_ref,
        "host_request_ref": str(request_copy),
        "dockerfile_ref": str(dockerfile),
        "base_image_ref": base_spec.image,
        "image_ref": image_ref,
        "build_generation_id": build_generation_id,
        "started_at": started_at,
        "finished_at": utc_now_iso(),
        "returncode": completed.returncode,
    }
    if completed.returncode != 0:
        manifest.update(sandbox_image_update=result)
        raise RuntimeError(f"meta-learning sandbox image rebuild failed: {image_ref}")
    try:
        runtime = probe_image_runtime(
            image_ref, docker_executable=base_spec.docker_executable
        )
    except Exception as exc:
        result["status"] = "smoke_failed"
        result["reason"] = f"{type(exc).__name__}: sandbox runtime smoke failed"
        result["finished_at"] = utc_now_iso()
        _remove_image_ref(image_ref, docker_executable=base_spec.docker_executable)
        manifest.update(sandbox_image_update=result)
        raise RuntimeError(
            f"meta-learning sandbox image smoke failed: {image_ref}"
        ) from exc
    result["runtime"] = runtime
    result["finished_at"] = utc_now_iso()
    owned_image_refs.append(image_ref)
    _write_image_state(
        state_path,
        experiment_id=experiment_id,
        image_ref=image_ref,
        build_generation_id=build_generation_id,
        base_image_ref=base_spec.image,
        kind="derived_build",
        runtime=runtime,
        owned_image_refs=owned_image_refs,
    )
    active_spec = replace(
        base_spec,
        image=image_ref,
        build_generation_id=build_generation_id,
    )
    pruned, retained = _gc_owned_sandbox_images(
        owned_image_refs,
        keep=image_keep,
        keep_image=image_ref,
        docker_executable=base_spec.docker_executable,
    )
    if pruned:
        _write_image_state(
            state_path,
            experiment_id=experiment_id,
            image_ref=image_ref,
            build_generation_id=build_generation_id,
            base_image_ref=base_spec.image,
            kind="derived_build",
            runtime=runtime,
            owned_image_refs=retained,
        )
    result["pruned_image_refs"] = pruned
    manifest.update(sandbox_image_update=result)
    return result, active_spec


def _gc_owned_sandbox_images(
    owned_image_refs: list[str],
    *,
    keep: int,
    keep_image: str,
    docker_executable: str = "docker",
) -> tuple[list[str], list[str]]:
    """Best-effort GC over this experiment's exact persisted tags only."""

    owned = list(dict.fromkeys(owned_image_refs))
    if keep_image not in owned:
        owned.append(keep_image)
    if keep <= 0:
        return [], owned
    retained_set = set(owned[-keep:])
    retained_set.add(keep_image)
    stale = [image_ref for image_ref in owned if image_ref not in retained_set]
    pruned: list[str] = []
    for image_ref in stale:
        try:
            removed = subprocess.run(
                [docker_executable, "image", "rm", image_ref],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            continue
        if removed.returncode == 0:
            pruned.append(image_ref)
    return pruned, [image_ref for image_ref in owned if image_ref not in pruned]


def _load_sandbox_environment_request(path: Path) -> dict[str, object]:
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"sandbox_environment.json is not valid JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError("sandbox_environment.json must be a JSON object")  # noqa: TRY004
    allowed = {"python_packages", "apt_packages", "npm_packages", "reason", "notes"}
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"sandbox_environment.json contains unsupported fields: {unknown}")
    request: dict[str, object] = {
        "python_packages": _validated_package_list(
            raw.get("python_packages"), field="python_packages", pattern=_PYTHON_PACKAGE_RE, max_items=40
        ),
        "apt_packages": _validated_package_list(
            raw.get("apt_packages"), field="apt_packages", pattern=_SYSTEM_PACKAGE_RE, max_items=30
        ),
        "npm_packages": _validated_package_list(
            raw.get("npm_packages"), field="npm_packages", pattern=_NPM_PACKAGE_RE, max_items=30
        ),
    }
    for key in ("reason", "notes"):
        value = raw.get(key)
        if value is not None:
            if not isinstance(value, str):
                raise ValueError(f"{key} must be a string")
            request[key] = value[:2000]
    return request


def _validated_package_list(
    value: object,
    *,
    field: str,
    pattern: re.Pattern[str],
    max_items: int,
) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a list")  # noqa: TRY004
    packages: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"{field} entries must be non-empty strings")
        package = item.strip()
        lowered = package.lower()
        if (
            package.startswith("-") or not pattern.fullmatch(package) or "://" in package
            or lowered.startswith(("git+", "hg+", "svn+", "bzr+"))
        ):
            raise ValueError(f"unsupported {field} entry: {package!r}")
        if package not in packages:
            packages.append(package)
    if len(packages) > max_items:
        raise ValueError(f"{field} has {len(packages)} entries > {max_items}")
    return packages


def _environment_request_has_packages(request: dict[str, object]) -> bool:
    return any(request.get(key) for key in ("python_packages", "apt_packages", "npm_packages"))


def _docker_tag_component(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip(".-")
    return (cleaned or "experiment")[:48].lower()


def _render_sandbox_extension_dockerfile(base_image: str, request: dict[str, object]) -> str:
    if not _DOCKER_IMAGE_RE.fullmatch(base_image) or "@" in base_image:
        raise ValueError(f"unsupported base sandbox image: {base_image!r}")
    lines = [
        "# Generated by AutoTrade Pipeline from meta-learning sandbox_environment.json.",
        f"FROM {base_image}",
        "ARG PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple",
        "ARG NPM_CONFIG_REGISTRY=https://registry.npmmirror.com",
        "ARG DEBIAN_MIRROR=https://mirrors.tuna.tsinghua.edu.cn/debian",
        "ARG DEBIAN_SECURITY_MIRROR=https://mirrors.tuna.tsinghua.edu.cn/debian-security",
        "USER root",
    ]
    raw_apt = request.get("apt_packages", [])
    apt_packages = (
        [shlex.quote(str(item)) for item in raw_apt]
        if isinstance(raw_apt, list)
        else []
    )
    if apt_packages:
        lines.append(
            'RUN sed -i -e "s|http://deb.debian.org/debian-security|'
            '${DEBIAN_SECURITY_MIRROR}|g" -e "s|http://deb.debian.org/debian|'
            '${DEBIAN_MIRROR}|g" /etc/apt/sources.list.d/debian.sources '
            "&& apt-get update && apt-get install -y --no-install-recommends "
            + " ".join(apt_packages)
            + " && rm -rf /var/lib/apt/lists/*"
        )
    raw_python = request.get("python_packages", [])
    python_specs = (
        [str(item) for item in raw_python] if isinstance(raw_python, list) else []
    )
    python_packages = [shlex.quote(item) for item in python_specs]
    if python_packages:
        lines.append(
            'RUN python -m pip install --no-cache-dir -i "${PIP_INDEX_URL}" '
            + " ".join(python_packages)
        )
        # Verification layer: a build that installs a package but cannot import it
        # is a silent transfer failure for later Folds. Fail the build here so
        # "image built" implies "importable", not just "installable".
        imports = _python_import_names(python_specs)
        if imports:
            statement = "; ".join(f"import {name}" for name in imports)
            lines.append(f"RUN python -c {shlex.quote(statement)}")
    raw_npm = request.get("npm_packages", [])
    npm_packages = (
        [shlex.quote(str(item)) for item in raw_npm]
        if isinstance(raw_npm, list)
        else []
    )
    if npm_packages:
        lines.append(
            'RUN npm install -g --no-fund --no-audit --registry "${NPM_CONFIG_REGISTRY}" '
            + " ".join(npm_packages)
        )
    lines.extend(["USER 61000:61000", "WORKDIR /mnt/agent", ""])
    return "\n".join(lines)


# PyPI distribution name -> import module name for the cases where they diverge.
_IMPORT_NAME_ALIASES = {
    "scikit-learn": "sklearn",
    "opencv-python": "cv2",
    "opencv-contrib-python": "cv2",
    "umap-learn": "umap",
    "pillow": "PIL",
    "pyyaml": "yaml",
    "beautifulsoup4": "bs4",
    "python-dateutil": "dateutil",
    "msgpack-python": "msgpack",
    "faiss-cpu": "faiss",
    "faiss-gpu": "faiss",
}


def _python_import_names(specs: list[str]) -> list[str]:
    """Top-level import names for declared python_packages, for a build-time smoke
    test. Only emit a name we are confident about: a known alias, or a simple
    distribution name with no '-'/'.' (where dist == import). For a hyphenated/dotted
    name that is not aliased the import module is unguessable (e.g. umap-learn->umap,
    opencv-contrib-python->cv2), so we SKIP its smoke import rather than reject a
    validly-installed package; the build still verifies pip install succeeded."""
    names: list[str] = []
    for spec in specs:
        dist = re.split(r"[<>=!~;\[\s]", str(spec).strip(), maxsplit=1)[0].strip()
        if not dist:
            continue
        lower = dist.lower()
        if lower in _IMPORT_NAME_ALIASES:
            module = _IMPORT_NAME_ALIASES[lower]
        elif "-" in lower or "." in lower:
            continue  # ambiguous import name — rely on pip install success
        else:
            module = lower
        if module and module.isidentifier() and module not in names:
            names.append(module)
    return names


__all__ = [
    "SANDBOX_ENVIRONMENT_EXAMPLE_NAME",
    "SANDBOX_ENVIRONMENT_REQUEST_NAME",
    "SANDBOX_IMAGE_STATE_NAME",
    "maybe_rebuild_sandbox_image",
    "prepare_experiment_sandbox_image",
    "write_sandbox_environment_example",
]
