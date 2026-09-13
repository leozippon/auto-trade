"""The experiment's own sandbox image tag.

Every experiment clones the configured local sandbox image once into a UUID4
tag and persists it in ``hitl/sandbox_image.json``, so a resumed experiment and
its Paper book keep running the image they started on. The pipeline only wires
config knobs in and records the result.
"""

from __future__ import annotations

import fcntl
import json
import re
import subprocess
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

from .runtime import utc_now_iso, write_json_atomic
from .sandbox import SandboxSpec, probe_image_runtime

SANDBOX_IMAGE_STATE_NAME = "sandbox_image.json"


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


def _docker_tag_component(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip(".-")
    return (cleaned or "experiment")[:48].lower()


__all__ = [
    "SANDBOX_IMAGE_STATE_NAME",
    "prepare_experiment_sandbox_image",
]
