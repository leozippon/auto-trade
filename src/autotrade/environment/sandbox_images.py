"""Derived sandbox image lifecycle (meta-learning ``sandbox_environment.json``).

Meta-learning may request stable new dependencies for later ordinary Folds by
writing ``workspace/sandbox_environment.json``. This module owns the whole
image-extension domain: request validation, Dockerfile rendering (with a
build-time import smoke test), the ``docker build``, image identity, and
best-effort GC of stale derived images. The pipeline only wires config knobs in
and records the result.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import uuid
from dataclasses import replace
from pathlib import Path

from .runtime import utc_now_iso
from .sandbox import SandboxSpec, inspect_local_image_tags

SANDBOX_ENVIRONMENT_REQUEST_NAME = "sandbox_environment.json"
SANDBOX_ENVIRONMENT_EXAMPLE_NAME = "sandbox_environment.example.json"
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

    # One directory per build: without a content-addressed tag every request
    # produces a fresh build, so the Dockerfile and the request copy of each
    # build have to stay side by side for the audit trail.
    build_id = uuid.uuid4().hex[:10]
    image_tag = f"autotrade-sandbox:{_docker_tag_component(experiment_id)}-{epoch_id}-{build_id}"
    build_dir = Path(experiment_dir) / "sandbox_images" / epoch_id / build_id
    build_dir.mkdir(parents=True, exist_ok=False)
    dockerfile = build_dir / "Dockerfile"
    try:
        dockerfile_text = _render_sandbox_extension_dockerfile(base_spec.image, request)
    except ValueError as exc:
        result = {
            "status": "rejected",
            "reason": str(exc),
            "request_ref": request_ref,
            "base_image": base_spec.image,
            "build_id": build_id,
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
    command.extend(["--tag", image_tag, "--file", str(dockerfile), str(build_dir)])
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
            "base_image": base_spec.image,
            "image": image_tag,
            "build_id": build_id,
            "started_at": started_at,
            "finished_at": utc_now_iso(),
            "timeout_seconds": timeout_seconds,
            "stdout_tail": str(exc.stdout or "")[-4000:],
            "stderr_tail": str(exc.stderr or "")[-4000:],
        }
        manifest.update(sandbox_image_update=result)
        raise RuntimeError(f"meta-learning sandbox image rebuild timed out: {image_tag}") from exc
    result: dict[str, object] = {
        "status": "ok" if completed.returncode == 0 else "failed",
        "request_ref": request_ref,
        "host_request_ref": str(request_copy),
        "dockerfile_ref": str(dockerfile),
        "base_image": base_spec.image,
        "image": image_tag,
        "build_id": build_id,
        "started_at": started_at,
        "finished_at": utc_now_iso(),
        "returncode": completed.returncode,
        "stdout_tail": str(completed.stdout)[-4000:],
        "stderr_tail": str(completed.stderr)[-4000:],
    }
    active_spec = base_spec
    if completed.returncode == 0:
        active_spec = replace(base_spec, image=image_tag)
        result["image_tags"] = inspect_local_image_tags(
            image_tag, docker_executable=base_spec.docker_executable
        )
        result["pruned_images"] = _gc_derived_sandbox_images(
            experiment_id,
            keep=image_keep,
            keep_image=image_tag,
            docker_executable=base_spec.docker_executable,
        )
    manifest.update(sandbox_image_update=result)
    if completed.returncode != 0:
        raise RuntimeError(f"meta-learning sandbox image rebuild failed: {image_tag}")
    return result, active_spec


def _gc_derived_sandbox_images(
    experiment_id: str,
    *,
    keep: int,
    keep_image: str,
    docker_executable: str = "docker",
) -> list[str]:
    """Best-effort prune of stale derived images for this experiment, keeping the
    most recent ``keep`` (and always the active one). Docker image GC must never
    fail a build, so all errors are swallowed."""
    if keep <= 0:
        return []
    prefix = f"autotrade-sandbox:{_docker_tag_component(experiment_id)}-"
    try:
        listed = subprocess.run(
            [docker_executable, "images", "--format", "{{.Repository}}:{{.Tag}}\t{{.CreatedAt}}",
             "autotrade-sandbox"],
            capture_output=True, text=True, timeout=30, check=False,
        )
        if listed.returncode != 0:
            return []
        rows: list[tuple[str, str]] = []
        for line in listed.stdout.splitlines():
            if not line.startswith(prefix):
                continue
            tag, _, created = line.partition("\t")
            rows.append((tag, created))
        # Sort newest first by Docker's CreatedAt (lexicographic on the
        # "YYYY-MM-DD HH:MM:SS …" prefix is chronological) rather than trusting
        # `docker images` default order; keep the newest, drop the older tail,
        # never removing the just-built active image.
        rows.sort(key=lambda row: row[1], reverse=True)
        stale = [tag for tag, _ in rows[keep:] if tag != keep_image]
        pruned: list[str] = []
        for tag in stale:
            removed = subprocess.run(
                [docker_executable, "image", "rm", tag],
                capture_output=True, text=True, timeout=30, check=False,
            )
            if removed.returncode == 0:
                pruned.append(tag)
        return pruned
    except (OSError, subprocess.SubprocessError):
        return []


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
    "maybe_rebuild_sandbox_image",
    "write_sandbox_environment_example",
]
