"""The strategy contract a sandbox image carries, in its two halves.

What the container enforces is the strategy runtime it bakes; what the Agent is
told is the template README the build only hashed. An image whose runtime
differs from the one the starting process imported would silently judge a
strategy by superseded rules and is refused at every container start. An image
built from a different README only misleads an Agent, so it is refused where a
research session mounts its copy, and nowhere else. These tests cover both
digests, both refusals, and the build step that puts the recorded digest into
the image.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from autotrade.environment import contract_fingerprint
from autotrade.environment.contract_fingerprint import (
    AGENT_CONTRACT_PATHS,
    CONTRACT_SOURCE_PATHS,
    HOST_RUNTIME_MODULE_DIR,
    IMAGE_FINGERPRINT_PATH,
    IMAGE_RUNTIME_MODULE_DIR,
    REBUILD_COMMAND,
    RUNTIME_CONTRACT_MODULES,
    ImageContract,
    SandboxImageContractMismatch,
    assert_image_contract_current,
    check_agent_contract,
    check_runtime_contract,
    compute_contract_fingerprint,
    compute_runtime_fingerprint,
    host_contract_fingerprint,
    host_runtime_fingerprint,
    read_image_contract,
)
from autotrade.environment.executor import (
    IMPORTED_RUNTIME_FINGERPRINT,
    StrategyExecutionError,
    _require_local_image,
)
from autotrade.environment.sandbox import SandboxConfig

from .fixtures_sandbox import docker_available

REPO =Path(__file__).resolve().parents[2]
DOCKERFILE = REPO / "ops/docker/sandbox.Dockerfile"
IMAGE = "autotrade-sandbox:exp-base-1234"
BASE_IMAGE = "autotrade-sandbox:latest"


def _contract_tree(root: Path) -> Path:
    """A repository-shaped copy of the contract sources, as the image stages them."""

    for relative in CONTRACT_SOURCE_PATHS:
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / relative, destination)
    return root


def _image_contract(tree: Path) -> ImageContract:
    """What an image built from ``tree`` reports about itself."""

    return ImageContract(
        runtime=compute_runtime_fingerprint(tree / HOST_RUNTIME_MODULE_DIR),
        contract=compute_contract_fingerprint(tree),
    )


def _image_reports(monkeypatch: pytest.MonkeyPatch, found: ImageContract) -> None:
    """Answer the one Docker call with ``found``, leaving every check real."""

    monkeypatch.setattr(
        contract_fingerprint, "read_image_contract", lambda image, **_kwargs: found
    )


def _container_start(tmp_path: Path) -> SandboxConfig:
    """A strategy container's config whose Docker finds every image locally;
    what the image bakes is answered by ``_image_reports``."""

    docker = tmp_path / "docker"
    docker.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    docker.chmod(0o755)
    return SandboxConfig(image=IMAGE, docker_executable=str(docker))


def test_both_fingerprints_are_stable_and_layout_independent(tmp_path: Path) -> None:
    staged = _contract_tree(tmp_path / "contract")
    # The image stages and bakes the same files under different absolute roots;
    # only the bytes and the names may enter either digest.
    assert compute_contract_fingerprint(staged) == host_contract_fingerprint()
    assert compute_runtime_fingerprint(staged / HOST_RUNTIME_MODULE_DIR) == (
        host_runtime_fingerprint()
    )
    # A session's copy of an unchanged README digests like the template.
    assert host_contract_fingerprint(staged / AGENT_CONTRACT_PATHS[0]) == (
        host_contract_fingerprint()
    )
    # Nothing has moved under this process since it imported the runtime.
    assert IMPORTED_RUNTIME_FINGERPRINT == host_runtime_fingerprint()


@pytest.mark.parametrize("relative", CONTRACT_SOURCE_PATHS)
def test_full_fingerprint_changes_with_any_contract_source(
    tmp_path: Path, relative: str
) -> None:
    staged = _contract_tree(tmp_path / "contract")
    before = compute_contract_fingerprint(staged)
    edited = staged / relative
    edited.write_bytes(edited.read_bytes() + b"\n# contract change\n")
    assert compute_contract_fingerprint(staged) != before


def test_runtime_fingerprint_follows_the_modules_and_not_the_contract_text(
    tmp_path: Path,
) -> None:
    staged = _contract_tree(tmp_path / "contract")
    runtime_dir = staged / HOST_RUNTIME_MODULE_DIR
    before = compute_runtime_fingerprint(runtime_dir)
    readme = staged / AGENT_CONTRACT_PATHS[0]
    readme.write_bytes(readme.read_bytes() + b"\nEach decision gets 360 s.\n")
    assert compute_runtime_fingerprint(runtime_dir) == before
    for module in RUNTIME_CONTRACT_MODULES:
        edited = runtime_dir / module
        edited.write_bytes(edited.read_bytes() + b"\n# loader change\n")
        assert compute_runtime_fingerprint(runtime_dir) != before


def test_missing_contract_source_fails_instead_of_digesting_a_subset(
    tmp_path: Path,
) -> None:
    staged = _contract_tree(tmp_path / "contract")
    (staged / HOST_RUNTIME_MODULE_DIR / RUNTIME_CONTRACT_MODULES[0]).unlink()
    with pytest.raises(SandboxImageContractMismatch, match="unreadable"):
        compute_runtime_fingerprint(staged / HOST_RUNTIME_MODULE_DIR)
    with pytest.raises(SandboxImageContractMismatch, match="unreadable"):
        compute_contract_fingerprint(staged)


def test_drift_is_refused_with_both_fingerprints_and_the_rebuild_command() -> None:
    stale = "0" * 64
    for check, host in (
        (check_runtime_contract, host_runtime_fingerprint()),
        (check_agent_contract, host_contract_fingerprint()),
    ):
        check(IMAGE, host, host)
        with pytest.raises(SandboxImageContractMismatch) as raised:
            check(IMAGE, stale, host)
        message = str(raised.value)
        assert IMAGE in message
        assert stale in message
        assert host in message
        assert REBUILD_COMMAND in message


@pytest.mark.parametrize("module", RUNTIME_CONTRACT_MODULES)
def test_runtime_drift_stops_every_container_and_every_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, module: str
) -> None:
    staged = _contract_tree(tmp_path / "image")
    edited = staged / HOST_RUNTIME_MODULE_DIR / module
    edited.write_bytes(edited.read_bytes() + b"\n# superseded loader rule\n")
    _image_reports(monkeypatch, _image_contract(staged))
    with pytest.raises(StrategyExecutionError, match=module):
        _require_local_image(_container_start(tmp_path))
    with pytest.raises(SandboxImageContractMismatch, match=module):
        assert_image_contract_current(
            IMAGE,
            runtime=IMPORTED_RUNTIME_FINGERPRINT,
            readme=staged / AGENT_CONTRACT_PATHS[0],
        )


def test_a_readme_edit_is_refused_where_a_session_mounts_it_and_nowhere_else(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The arm's image was built before the template README changed in this
    checkout. A session seeded afterwards mounts the new text and is refused
    before its Agent reads it. A strategy container -- a running arm's next
    validation, its forward replay, a Paper book -- never reads the README and
    starts, and a resumed session keeps the copy it was seeded with."""

    built = _contract_tree(tmp_path / "built")
    _image_reports(monkeypatch, _image_contract(built))
    checkout = _contract_tree(tmp_path / "checkout")
    readme = checkout / AGENT_CONTRACT_PATHS[0]
    readme.write_bytes(readme.read_bytes() + b"\nEach decision gets 360 s.\n")
    monkeypatch.setattr(contract_fingerprint, "_repository_root", lambda: checkout)

    config = _container_start(tmp_path)
    assert _require_local_image(config) == config.docker_executable
    with pytest.raises(SandboxImageContractMismatch, match=AGENT_CONTRACT_PATHS[0]):
        assert_image_contract_current(
            IMAGE, runtime=IMPORTED_RUNTIME_FINGERPRINT, readme=readme
        )
    assert_image_contract_current(
        IMAGE,
        runtime=IMPORTED_RUNTIME_FINGERPRINT,
        readme=built / AGENT_CONTRACT_PATHS[0],
    )


def test_a_container_is_compared_with_the_runtime_its_process_imported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A runtime module edited under a running worker changes neither the
    validator that worker runs nor what its next container must bake: the
    pinned image still matches, and an image rebuilt from the edit does not."""

    built = _contract_tree(tmp_path / "built")
    checkout = _contract_tree(tmp_path / "checkout")
    edited = checkout / HOST_RUNTIME_MODULE_DIR / RUNTIME_CONTRACT_MODULES[1]
    edited.write_bytes(edited.read_bytes() + b"\n# a later loader rule\n")
    monkeypatch.setattr(contract_fingerprint, "_repository_root", lambda: checkout)
    config = _container_start(tmp_path)
    _image_reports(monkeypatch, _image_contract(built))
    _require_local_image(config)
    _image_reports(monkeypatch, _image_contract(checkout))
    with pytest.raises(StrategyExecutionError, match=RUNTIME_CONTRACT_MODULES[1]):
        _require_local_image(config)


def test_image_predating_the_recorded_digest_runs_containers_and_stops_a_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _image_reports(monkeypatch, ImageContract(IMPORTED_RUNTIME_FINGERPRINT, ""))
    _require_local_image(_container_start(tmp_path))
    with pytest.raises(SandboxImageContractMismatch) as raised:
        assert_image_contract_current(
            IMAGE,
            runtime=IMPORTED_RUNTIME_FINGERPRINT,
            readme=REPO / AGENT_CONTRACT_PATHS[0],
        )
    assert IMAGE_FINGERPRINT_PATH in str(raised.value)


@pytest.mark.skipif(not docker_available(), reason="Docker is unavailable")
def test_unreadable_image_is_refused_rather_than_reported_as_drift() -> None:
    with pytest.raises(SandboxImageContractMismatch, match="cannot be read"):
        assert_image_contract_current(
            "autotrade-sandbox:absent-0000", runtime=IMPORTED_RUNTIME_FINGERPRINT
        )


@pytest.mark.skipif(not docker_available(), reason="Docker is unavailable")
def test_real_base_image_reports_the_runtime_it_bakes() -> None:
    """The in-image half, against the image the host actually builds from."""

    if subprocess.run(
        ["docker", "image", "inspect", BASE_IMAGE], capture_output=True, check=False
    ).returncode:
        pytest.skip(f"{BASE_IMAGE} is not built locally")
    found = read_image_contract(BASE_IMAGE)
    # AGENTS.md requires a rebuild whenever the runtime modules change, so the
    # local base image must bake exactly this checkout's modules.
    assert found.runtime == host_runtime_fingerprint()
    assert len(found.contract) == 64


def test_dockerfile_bakes_the_runtime_and_records_the_full_contract() -> None:
    """The build side: the modules stay in the image, the full digest is recorded."""

    text = DOCKERFILE.read_text(encoding="utf-8")
    assert (
        "COPY src/autotrade/environment/contract_fingerprint.py "
        "/opt/autotrade/autotrade/environment/contract_fingerprint.py" in text
    )
    baked: set[str] = set()
    staged: set[str] = set()
    for line in text.splitlines():
        if not line.startswith("COPY "):
            continue
        *sources, destination = line.split()[1:]
        for source in sources:
            target = (
                destination + Path(source).name
                if destination.endswith("/")
                else destination
            )
            if target.startswith("/opt/autotrade/contract/"):
                staged.add(target)
            elif target.startswith(f"{IMAGE_RUNTIME_MODULE_DIR}/"):
                baked.add(target)
    # The digest the host recomputes from the image comes from these copies.
    assert {f"{IMAGE_RUNTIME_MODULE_DIR}/{name}" for name in RUNTIME_CONTRACT_MODULES} <= (
        baked
    )
    assert staged == {f"/opt/autotrade/contract/{item}" for item in CONTRACT_SOURCE_PATHS}
    assert "compute_contract_fingerprint" in text
    assert "IMAGE_FINGERPRINT_PATH" in text
    assert IMAGE_FINGERPRINT_PATH == "/opt/autotrade/SOURCE_FINGERPRINT"
    # The staged copies exist only to be hashed; the image keeps the digest.
    assert "rm -rf /opt/autotrade/contract" in text
