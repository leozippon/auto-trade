"""Fingerprints of the strategy contract a sandbox image carries.

The sandbox image bakes its own copy of the trusted strategy runtime, so the
loader that accepts or rejects an Agent-authored strategy lives *inside* the
image while the rule the Agent is told about lives in the repository. Those are
two contracts, and they drift apart for different reasons:

- The **runtime contract** — ``strategy.py``, ``strategy_loader.py`` and
  ``strategy_worker.py`` — is what the container actually enforces. An image
  baking other bytes would accept or reject a strategy by superseded rules, and
  nothing inside the container reveals that. Every strategy container start
  checks it, research and Paper alike.
- The **Agent-facing contract** — ``configs/agent_output_template/README.md`` —
  is the text that states those same rules to the Agent. It matters only where
  an Agent reads it, that is in a research session; a Paper book replays a
  strategy frozen long before, with no README anywhere in the loop.

Splitting them is what lets a README-only rebuild of the base image leave a
Paper book running on a pinned image that bakes exactly this checkout's
runtime, while any drift of the runtime modules is still refused everywhere.

The two digests come from one hashing function and one file list. The runtime
digest is computed from the modules the image baked — the very bytes the worker
imports — so it needs no build-time record and can be read from any image ever
built. The full contract has no such in-image evidence, because the build
stages the README only to hash it; the build therefore writes that digest into
the image and the research check compares it.

Stdlib only: the host runs this module *inside* the image (``python -``) to
digest the baked modules with the same function, and the image build imports it
as ``autotrade.environment.contract_fingerprint``; in neither place is anything
else of ``autotrade`` available.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

# The strategy runtime the image bakes and the container imports: the scheduled
# strategy protocol, the static validator that enforces it, and the sandbox-side
# worker. Ordered for a stable digest.
RUNTIME_CONTRACT_MODULES: tuple[str, ...] = (
    "strategy.py",
    "strategy_loader.py",
    "strategy_worker.py",
)
HOST_RUNTIME_MODULE_DIR = "src/autotrade/environment"
# Where the Dockerfile copies those modules; a derived image inherits them.
IMAGE_RUNTIME_MODULE_DIR = "/opt/autotrade/autotrade/environment"
# The Agent-facing half: the template README that states the same rules to the
# Agent. The image keeps no copy, only the full digest below.
AGENT_CONTRACT_PATHS: tuple[str, ...] = ("configs/agent_output_template/README.md",)
# Repository-relative sources of the full contract, in the order the build
# hashes them.
CONTRACT_SOURCE_PATHS: tuple[str, ...] = AGENT_CONTRACT_PATHS + tuple(
    f"{HOST_RUNTIME_MODULE_DIR}/{name}" for name in RUNTIME_CONTRACT_MODULES
)
# Written by ops/docker/sandbox.Dockerfile; derived images inherit the file.
IMAGE_FINGERPRINT_PATH = "/opt/autotrade/SOURCE_FINGERPRINT"
REBUILD_COMMAND = (
    "docker build -t autotrade-sandbox:latest -f ops/docker/sandbox.Dockerfile ."
)


class SandboxImageContractMismatch(RuntimeError):
    """A sandbox image cannot be shown to carry this checkout's strategy contract.

    Raised for a differing digest, an image built before the guard existed, an
    image that cannot be read at all, and an unreadable contract source — the
    cases are told apart by the message, never by silently continuing.
    """


def _digest(entries: Iterable[tuple[str, Path]]) -> str:
    """Digest ``(name, file)`` pairs: names and bytes, length-framed.

    A missing or unreadable source is a build/installation defect, not a reason
    to fall back to a partial digest.
    """

    digest = hashlib.sha256()
    for name, path in entries:
        try:
            payload = path.read_bytes()
        except OSError as exc:
            raise SandboxImageContractMismatch(
                f"strategy contract source is unreadable: {path}"
            ) from exc
        digest.update(f"{name}\0{len(payload)}\0".encode())
        digest.update(payload)
    return digest.hexdigest()


def compute_contract_fingerprint(root: str | Path) -> str:
    """Digest the full contract below ``root`` (a repository-shaped tree).

    ``root`` is the repository root on the host and the staging tree the image
    build copies the same files into, so one function serves both sides.
    """

    base = Path(root)
    return _digest((relative, base / relative) for relative in CONTRACT_SOURCE_PATHS)


def compute_runtime_fingerprint(module_dir: str | Path) -> str:
    """Digest the three strategy runtime modules sitting in ``module_dir``.

    ``module_dir`` is ``src/autotrade/environment`` on the host and the baked
    package directory inside the image, which is why the module names — not
    repository paths — are what enters the digest.
    """

    base = Path(module_dir)
    return _digest((name, base / name) for name in RUNTIME_CONTRACT_MODULES)


def host_contract_fingerprint() -> str:
    """Digest the full contract of the checkout this module belongs to."""

    return compute_contract_fingerprint(_repository_root())


def host_runtime_fingerprint() -> str:
    """Digest the strategy runtime modules of this checkout."""

    return compute_runtime_fingerprint(_repository_root() / HOST_RUNTIME_MODULE_DIR)


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class ImageContract:
    """What a sandbox image would enforce.

    ``runtime`` is digested from the modules the image baked. ``contract`` is
    the full-contract digest its build recorded, empty for an image built
    before the guard existed.
    """

    runtime: str
    contract: str


def read_image_contract(
    image: str, *, docker_executable: str = "docker", timeout_seconds: float = 120.0
) -> ImageContract:
    """Read both fingerprints out of ``image`` in one offline container run.

    This module is piped in as the script, so the digest of the baked modules
    is computed by the same function the host compares against and no image
    needs to be rebuilt to be readable.
    """

    completed = subprocess.run(
        [
            docker_executable, "run", "--pull", "never", "--rm", "--network", "none",
            "-i", "--entrypoint", "python", image, "-",
        ],
        input=Path(__file__).read_bytes(),
        capture_output=True,
        timeout=timeout_seconds,
        check=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise SandboxImageContractMismatch(
            f"sandbox image {image} cannot be read for the strategy contract it bakes, "
            f"so what it would enforce is unknown ({detail or 'no detail'}). Rebuild the "
            f"base image with `{REBUILD_COMMAND}`, then re-point every pinned experiment "
            "tag at it (see docs/environment-design.md)."
        )
    try:
        payload = json.loads(completed.stdout)
        return ImageContract(
            runtime=str(payload["runtime"]), contract=str(payload["contract"])
        )
    except (ValueError, KeyError, TypeError) as exc:
        raise SandboxImageContractMismatch(
            f"sandbox image {image} returned no readable strategy contract report"
        ) from exc


def check_runtime_contract(image: str, image_runtime: str) -> None:
    """Raise unless the image bakes this checkout's strategy runtime modules."""

    host = host_runtime_fingerprint()
    if image_runtime == host:
        return
    raise SandboxImageContractMismatch(
        f"sandbox image {image} bakes a different strategy runtime than this checkout "
        f"(image {image_runtime or '<empty>'}, host {host}); the container would enforce "
        f"superseded rules from {', '.join(RUNTIME_CONTRACT_MODULES)}. Rebuild the base "
        f"image with `{REBUILD_COMMAND}`, then re-point every pinned experiment tag at it "
        "(see docs/environment-design.md)."
    )


def check_agent_contract(image: str, image_contract: str) -> None:
    """Raise unless the image was built from the contract text the Agent reads.

    Called only for a session in which an Agent reads that text. The recorded
    digest covers the runtime modules as well, which ``check_runtime_contract``
    has already compared byte for byte; what this adds is the README.
    """

    if not image_contract:
        raise SandboxImageContractMismatch(
            f"sandbox image {image} carries no strategy contract fingerprint at "
            f"{IMAGE_FINGERPRINT_PATH}, so the rules it was built from are unknown. "
            f"Rebuild the base image with `{REBUILD_COMMAND}`, then re-point every "
            "pinned experiment tag at it (see docs/environment-design.md)."
        )
    host = host_contract_fingerprint()
    if image_contract == host:
        return
    raise SandboxImageContractMismatch(
        f"sandbox image {image} was built from a different Agent strategy contract than "
        f"this checkout (image {image_contract}, host {host}); the Agent would be told "
        f"rules from {', '.join(AGENT_CONTRACT_PATHS)} that the image was not built "
        f"from. Rebuild the base image with `{REBUILD_COMMAND}`, then re-point every "
        "pinned experiment tag at it (see docs/environment-design.md)."
    )


def assert_image_contract_current(
    image: str, *, docker_executable: str = "docker", agent_contract: bool = True
) -> None:
    """Fail fast when ``image``'s contract has drifted from this checkout's.

    The runtime half is always checked. ``agent_contract`` adds the README half
    and is on for every caller but Paper, where no Agent reads it.
    """

    found = read_image_contract(image, docker_executable=docker_executable)
    check_runtime_contract(image, found.runtime)
    if agent_contract:
        check_agent_contract(image, found.contract)


def _image_contract_report() -> str:
    """The in-image side of :func:`read_image_contract`, as JSON."""

    try:
        contract = Path(IMAGE_FINGERPRINT_PATH).read_text(encoding="utf-8").strip()
    except OSError:
        contract = ""
    return json.dumps(
        {
            "runtime": compute_runtime_fingerprint(IMAGE_RUNTIME_MODULE_DIR),
            "contract": contract,
        }
    )


__all__ = [
    "AGENT_CONTRACT_PATHS",
    "CONTRACT_SOURCE_PATHS",
    "HOST_RUNTIME_MODULE_DIR",
    "IMAGE_FINGERPRINT_PATH",
    "IMAGE_RUNTIME_MODULE_DIR",
    "REBUILD_COMMAND",
    "RUNTIME_CONTRACT_MODULES",
    "ImageContract",
    "SandboxImageContractMismatch",
    "assert_image_contract_current",
    "check_agent_contract",
    "check_runtime_contract",
    "compute_contract_fingerprint",
    "compute_runtime_fingerprint",
    "host_contract_fingerprint",
    "host_runtime_fingerprint",
    "read_image_contract",
]


if __name__ == "__main__":
    sys.stdout.write(_image_contract_report())
