"""A paper Pack's untracked payload, checked against its manifest and copied into refs/.

What must hold. A Pack with ``PAYLOAD.sha256`` reaches a session only when
``data/pack_payloads/<pack>/`` holds exactly the listed files with the listed
digests; each refusal names the Pack and the path. The listed files land in
``refs/`` byte for byte beside the tracked copy, as copies the session cannot
alter the store through. A Pack without a manifest is untouched by any of it.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from autotrade.pipelines.pack_payload import (
    PAYLOAD_MANIFEST,
    PackPayloadError,
    format_manifest,
    pack_payload,
    payload_digests,
)
from autotrade.pipelines.research_session import (
    _REFERENCE_PDF_MAX_BYTES,
    install_workspace_reference,
)
from autotrade.pipelines.worker import _optional_workspace_reference

PACK = "configs/workspace_refs/paper_test"
STORE = "data/pack_payloads/paper_test"
BIG_PDF = b"%PDF-1.4\n" + bytes(range(256)) * (_REFERENCE_PDF_MAX_BYTES // 256 + 8)
PAYLOAD = {
    "papers/demo/paper.pdf": BIG_PDF,
    "papers/demo/paper.txt": b"Abstract\n  We forecast returns.\n",
    "papers/demo/repo/.gitignore": b"*.pt\n",
    "papers/demo/repo/.github/ci.yml": b"on: push\n",
    "papers/demo/repo/model.py": b"def forecast(x):\n    return x\n",
    "wheels/tinypkg-0.1-py3-none-any.whl": b"PK\x03\x04 not really a wheel",
}


def _paper_pack(repo: Path) -> Path:
    """A tracked Pack with notes, a tracked PDF too large for refs/, and a payload its manifest lists."""

    pack = repo / PACK
    (pack / "references").mkdir(parents=True)
    (pack / "papers").mkdir()
    (pack / "README.md").write_text("Reproduce the demo paper.\n", encoding="utf-8")
    (pack / "references" / "notes.md").write_text("notes\n", encoding="utf-8")
    (pack / "papers" / "index.md").write_text("demo: papers/demo/\n", encoding="utf-8")
    (pack / "references" / "scan.pdf").write_bytes(BIG_PDF)
    store = repo / STORE
    for relative, data in PAYLOAD.items():
        (store / relative).parent.mkdir(parents=True, exist_ok=True)
        (store / relative).write_bytes(data)
    (pack / PAYLOAD_MANIFEST).write_text(format_manifest(payload_digests(store, STORE)), encoding="utf-8")
    return pack


def test_the_payload_lands_in_refs_byte_for_byte_beside_the_tracked_copy(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _paper_pack(repo)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    install_workspace_reference(workspace, PACK, repo_root=repo)
    refs = workspace / "refs"
    landed = {path.relative_to(refs).as_posix(): path.read_bytes() for path in refs.rglob("*") if path.is_file()}
    # The tracked PDF over the cap stays out; the payload's PDF and hidden files arrive.
    assert landed.keys() == {"README.md", "references/notes.md", "papers/index.md", PAYLOAD_MANIFEST, *PAYLOAD}
    assert {relative: landed[relative] for relative in PAYLOAD} == PAYLOAD
    for path in (refs, *refs.rglob("*")):
        assert stat.S_IMODE(path.stat().st_mode) == (0o777 if path.is_dir() else 0o666)
    # A copy, not a link: rewriting it in the session leaves the store as listed.
    copied = refs / "papers/demo/repo/model.py"
    assert not os.path.samefile(copied, repo / STORE / "papers/demo/repo/model.py")
    copied.write_bytes(b"edited\n")
    assert pack_payload(repo / PACK, repo)


def _corrupt(repo: Path, case: str) -> str:
    """Break the fixture one way; returns the text the refusal must carry."""

    pack, store = repo / PACK, repo / STORE
    manifest = pack / PAYLOAD_MANIFEST
    if case == "missing":
        (store / "papers/demo/paper.txt").unlink()
        return f"{STORE}/papers/demo/paper.txt is missing"
    if case == "store absent":
        for path in sorted(store.rglob("*"), reverse=True):
            path.rmdir() if path.is_dir() else path.unlink()
        store.rmdir()
        return f"{STORE}/papers/demo/paper.pdf is missing"
    if case == "digest":
        (store / "papers/demo/repo/model.py").write_bytes(b"def forecast(x):\n    return -x\n")
        return f"{STORE}/papers/demo/repo/model.py has sha256"
    if case == "extra":
        (store / "papers/demo/repo/__pycache__").mkdir()
        (store / "papers/demo/repo/__pycache__/model.cpython-311.pyc").write_bytes(b"\0")
        return f"{STORE}/papers/demo/repo/__pycache__/model.cpython-311.pyc is not listed"
    if case == "symlink":
        (store / "papers/demo/link.txt").symlink_to(store / "papers/demo/paper.txt")
        return f"{STORE}/papers/demo/link.txt is a symlink"
    if case in ("absolute", "dotdot", "malformed", "unsorted"):
        line = {
            "absolute": f"{'0' * 64}  /etc/passwd",
            "dotdot": f"{'0' * 64}  papers/../../escape.txt",
            "malformed": f"{'0' * 63}  papers/demo/short.txt",
            "unsorted": f"{'0' * 64}  aaa.txt",
        }[case]
        manifest.write_text(manifest.read_text(encoding="utf-8") + line + "\n", encoding="utf-8")
        return {
            "absolute": "lists '/etc/passwd', which is not a relative path",
            "dotdot": "lists 'papers/../../escape.txt', which is not a relative path",
            "malformed": f"{PAYLOAD_MANIFEST} line {len(PAYLOAD) + 1} is not",
            "unsorted": "is not sorted bytewise without repeats at aaa.txt",
        }[case]
    if case == "collision":
        (pack / "papers/demo").mkdir()
        (pack / "papers/demo/paper.txt").write_text("a tracked text\n", encoding="utf-8")
        return "payload path papers/demo/paper.txt collides with a tracked file of the Pack"
    if case == "collision with a tracked file as a directory":
        (pack / "wheels").write_text("not a directory\n", encoding="utf-8")
        return "payload path wheels/tinypkg-0.1-py3-none-any.whl collides"
    raise AssertionError(case)


CASES = ("missing", "store absent", "digest", "extra", "symlink", "absolute", "dotdot", "malformed", "unsorted",
         "collision", "collision with a tracked file as a directory")


@pytest.mark.parametrize("case", CASES)
def test_the_preflight_and_the_session_start_refuse_a_payload_not_as_listed(tmp_path: Path, case: str) -> None:
    repo = (tmp_path / "repo").resolve()
    _paper_pack(repo)
    assert _optional_workspace_reference(PACK, repo) == PACK
    reason = _corrupt(repo, case)
    # The console and the round launcher refuse a ValueError as a parameter.
    with pytest.raises(ValueError) as refused:
        _optional_workspace_reference(PACK, repo)
    assert isinstance(refused.value, PackPayloadError)
    assert str(refused.value).startswith("Pack paper_test: ") and reason in str(refused.value)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with pytest.raises(PackPayloadError, match="Pack paper_test: "):
        install_workspace_reference(workspace, PACK, repo_root=repo)
    assert not (workspace / "refs").exists()


def test_a_pack_without_a_manifest_ignores_any_store_and_installs_as_before(tmp_path: Path) -> None:
    repo = (tmp_path / "repo").resolve()
    pack = _paper_pack(repo)
    (pack / PAYLOAD_MANIFEST).unlink()
    (repo / STORE / "papers/demo/link.txt").symlink_to("/etc/passwd")
    assert pack_payload(pack, repo) == []
    assert _optional_workspace_reference(PACK, repo) == PACK
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    install_workspace_reference(workspace, PACK, repo_root=repo)
    assert sorted(path.relative_to(workspace / "refs").as_posix() for path in (workspace / "refs").rglob("*")) == [
        "README.md", "papers", "papers/index.md", "references", "references/notes.md",
    ]
