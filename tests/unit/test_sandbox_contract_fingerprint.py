"""The sandbox image's baked strategy contract must match the host's sources.

The image carries its own copy of the strategy loader, so an image built before
a contract change enforces superseded rules while the session's mounted README
promises the new ones. The fingerprint is what makes that divergence loud, so
these tests cover the digest itself, the refusal, and the build step that puts
the digest into the image.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from autotrade.environment.contract_fingerprint import (
    CONTRACT_SOURCE_PATHS,
    IMAGE_FINGERPRINT_PATH,
    REBUILD_COMMAND,
    SandboxImageContractMismatch,
    check_contract_fingerprint,
    compute_contract_fingerprint,
    host_contract_fingerprint,
)

REPO = Path(__file__).resolve().parents[2]
DOCKERFILE = REPO / "ops/docker/sandbox.Dockerfile"


def _contract_tree(root: Path) -> Path:
    """A repository-shaped copy of the contract sources, as the image stages them."""

    for relative in CONTRACT_SOURCE_PATHS:
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / relative, destination)
    return root


def test_fingerprint_is_stable_and_layout_independent(tmp_path: Path) -> None:
    staged = _contract_tree(tmp_path / "contract")
    assert compute_contract_fingerprint(staged) == compute_contract_fingerprint(staged)
    # The image stages the same files under a different absolute root; only the
    # bytes and the repository-relative names may enter the digest.
    assert compute_contract_fingerprint(staged) == host_contract_fingerprint()


@pytest.mark.parametrize("relative", CONTRACT_SOURCE_PATHS)
def test_fingerprint_changes_with_any_contract_source(tmp_path: Path, relative: str) -> None:
    staged = _contract_tree(tmp_path / "contract")
    before = compute_contract_fingerprint(staged)
    edited = staged / relative
    edited.write_bytes(edited.read_bytes() + b"\n# contract change\n")
    assert compute_contract_fingerprint(staged) != before


def test_missing_contract_source_fails_instead_of_digesting_a_subset(tmp_path: Path) -> None:
    staged = _contract_tree(tmp_path / "contract")
    (staged / CONTRACT_SOURCE_PATHS[0]).unlink()
    with pytest.raises(SandboxImageContractMismatch, match="unreadable"):
        compute_contract_fingerprint(staged)


def test_matching_image_fingerprint_is_accepted() -> None:
    check_contract_fingerprint("autotrade-sandbox:latest", host_contract_fingerprint())


def test_diverged_image_is_refused_with_both_fingerprints_and_the_rebuild_command() -> None:
    host = host_contract_fingerprint()
    stale = "0" * 64
    with pytest.raises(SandboxImageContractMismatch) as raised:
        check_contract_fingerprint("autotrade-sandbox:exp-base-1234", stale)
    message = str(raised.value)
    assert "autotrade-sandbox:exp-base-1234" in message
    assert stale in message
    assert host in message
    assert REBUILD_COMMAND in message


def test_empty_image_fingerprint_is_refused() -> None:
    with pytest.raises(SandboxImageContractMismatch, match="<empty>"):
        check_contract_fingerprint("autotrade-sandbox:latest", "")


def test_dockerfile_stages_and_hashes_exactly_the_contract_sources() -> None:
    """The build side of the digest: same file list, same function, digest kept."""

    text = DOCKERFILE.read_text(encoding="utf-8")
    assert (
        "COPY src/autotrade/environment/contract_fingerprint.py "
        "/opt/autotrade/autotrade/environment/contract_fingerprint.py" in text
    )
    staged: set[str] = set()
    for line in text.splitlines():
        if not line.startswith("COPY "):
            continue
        *sources, destination = line.split()[1:]
        if not destination.startswith("/opt/autotrade/contract/"):
            continue
        for source in sources:
            staged.add(
                destination + Path(source).name
                if destination.endswith("/")
                else destination
            )
    assert staged == {f"/opt/autotrade/contract/{item}" for item in CONTRACT_SOURCE_PATHS}
    assert "compute_contract_fingerprint" in text
    assert "IMAGE_FINGERPRINT_PATH" in text
    assert IMAGE_FINGERPRINT_PATH == "/opt/autotrade/SOURCE_FINGERPRINT"
    # The staged copies exist only to be hashed; the image keeps the digest.
    assert "rm -rf /opt/autotrade/contract" in text
