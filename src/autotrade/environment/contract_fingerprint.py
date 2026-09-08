"""Fingerprint of the Agent-facing strategy contract baked into a sandbox image.

The sandbox image carries its own copy of the trusted strategy runtime, so the
loader that accepts or rejects an Agent-authored strategy lives *inside* the
image while the rule the Agent is told about lives in the repository. A source
change that is never rebuilt therefore splits the contract in two: the mounted
``output/README.md`` promises one rule and the container enforces another.

The build hashes the contract sources and writes the digest into the image; the
host recomputes it from the same files before every strategy container starts
and refuses a stale image. Both sides call :func:`compute_contract_fingerprint`,
so the two digests can only agree on identical bytes.

Stdlib only: this module is copied into the sandbox image and imported there
during the build, where nothing else of ``autotrade`` is available.
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

# Repository-relative sources that define the contract: the scheduled strategy
# protocol, the static validator that enforces it, the sandbox-side worker that
# imports the strategy, and the template README that states the same rules to
# the Agent. Ordered for a stable digest.
CONTRACT_SOURCE_PATHS: tuple[str, ...] = (
    "configs/agent_output_template/README.md",
    "src/autotrade/environment/strategy.py",
    "src/autotrade/environment/strategy_loader.py",
    "src/autotrade/environment/strategy_worker.py",
)
# Written by ops/docker/sandbox.Dockerfile; derived images inherit the file.
IMAGE_FINGERPRINT_PATH = "/opt/autotrade/SOURCE_FINGERPRINT"
REBUILD_COMMAND = (
    "docker build -t autotrade-sandbox:latest -f ops/docker/sandbox.Dockerfile ."
)


class SandboxImageContractMismatch(RuntimeError):
    """A sandbox image cannot be shown to carry this checkout's strategy contract.

    Raised for a differing digest, an image built before the guard existed, and
    an unreadable contract source — the cases are told apart by the message,
    never by silently continuing.
    """


def compute_contract_fingerprint(root: str | Path) -> str:
    """Digest the contract sources below ``root`` (a repository-shaped tree).

    ``root`` is the repository root on the host and the staging tree the image
    build copies the same files into, so one function serves both sides. A
    missing or unreadable source is a build/installation defect, not a reason
    to fall back to a partial digest.
    """

    base = Path(root)
    digest = hashlib.sha256()
    for relative in CONTRACT_SOURCE_PATHS:
        path = base / relative
        try:
            payload = path.read_bytes()
        except OSError as exc:
            raise SandboxImageContractMismatch(
                f"strategy contract source is unreadable: {path}"
            ) from exc
        digest.update(f"{relative}\0{len(payload)}\0".encode())
        digest.update(payload)
    return digest.hexdigest()


def host_contract_fingerprint() -> str:
    """Digest the contract sources of the checkout this module belongs to."""

    return compute_contract_fingerprint(Path(__file__).resolve().parents[3])


def read_image_contract_fingerprint(
    image: str, *, docker_executable: str = "docker", timeout_seconds: float = 60.0
) -> str:
    """Read the digest the image build baked in.

    An image built before the guard has no such file. That is reported as its
    own failure rather than as a differing digest, so the operator is not sent
    hunting for a source change that never happened.
    """

    completed = subprocess.run(
        [
            docker_executable, "run", "--pull", "never", "--rm", "--network", "none",
            "--entrypoint", "cat", image, IMAGE_FINGERPRINT_PATH,
        ],
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        check=False,
    )
    if completed.returncode != 0:
        raise SandboxImageContractMismatch(
            f"sandbox image {image} carries no Agent strategy contract fingerprint at "
            f"{IMAGE_FINGERPRINT_PATH}, so what it would enforce is unknown "
            f"({completed.stderr.strip() or 'no detail'}). Rebuild the base image with "
            f"`{REBUILD_COMMAND}`, then re-point every pinned experiment tag at it "
            "(see docs/environment-design.md)."
        )
    return completed.stdout.strip()


def check_contract_fingerprint(image: str, image_fingerprint: str) -> None:
    """Raise unless ``image_fingerprint`` matches this checkout's contract."""

    host = host_contract_fingerprint()
    if image_fingerprint == host:
        return
    raise SandboxImageContractMismatch(
        f"sandbox image {image} carries a different Agent strategy contract than this "
        f"checkout (image {image_fingerprint or '<empty>'}, host {host}); the container "
        f"would enforce superseded rules on {', '.join(CONTRACT_SOURCE_PATHS)}. Rebuild "
        f"the base image with `{REBUILD_COMMAND}`, then re-point every pinned experiment "
        "tag at it (see docs/environment-design.md)."
    )


def assert_image_contract_current(image: str, *, docker_executable: str = "docker") -> None:
    """Fail fast when ``image``'s baked contract has drifted from the host's."""

    check_contract_fingerprint(
        image, read_image_contract_fingerprint(image, docker_executable=docker_executable)
    )


__all__ = [
    "CONTRACT_SOURCE_PATHS",
    "IMAGE_FINGERPRINT_PATH",
    "REBUILD_COMMAND",
    "SandboxImageContractMismatch",
    "assert_image_contract_current",
    "check_contract_fingerprint",
    "compute_contract_fingerprint",
    "host_contract_fingerprint",
    "read_image_contract_fingerprint",
]
