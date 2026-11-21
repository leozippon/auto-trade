"""A research Pack's payload: untracked files its manifest lists, installed beside the Pack.

A Pack (``configs/workspace_refs/<pack>/``, an experiment's ``workspace_reference``)
is tracked and small. A paper Pack also carries what git should not -- the
paper, its text, the authors' repository at one commit, wheels or weights. It
lists them in ``PAYLOAD.sha256`` (sha256sum format, ``<sha256>  <path under
refs/>``, sorted bytewise) and keeps them untracked under
``data/pack_payloads/<pack>/`` at the same paths.

``pack_payload`` is the one check. The worker's pre-flight (the console's
create, a round's dry run, a run on the ZGCA instance), the session start that
copies the files into ``refs/`` and the ZGCA deploy all call it, so no
experiment starts on a payload that is missing or differs from its manifest.
"""

from __future__ import annotations

import hashlib
import os
import re
import stat
from pathlib import Path, PurePosixPath

PAYLOAD_MANIFEST = "PAYLOAD.sha256"
# Repository-relative; one directory per Pack, named as the Pack's directory.
PAYLOAD_ROOT = PurePosixPath("data/pack_payloads")

_MANIFEST_LINE = re.compile(r"([0-9a-f]{64})  (.+)")


class PackPayloadError(ValueError):
    """A Pack whose payload is not as its manifest lists it; the message names the Pack and the path."""


def payload_dir(pack: Path, repo_root: Path) -> Path:
    return repo_root / PAYLOAD_ROOT / pack.name


def _valid_relative(relative: str) -> bool:
    """A path a manifest line can carry and refs/ can hold: relative, no
    empty, ``.`` or ``..`` part, printable, and no backslash (sha256sum
    escapes names with one)."""

    return (
        relative.isprintable()
        and "\\" not in relative
        and not relative.startswith("/")
        and all(part not in ("", ".", "..") for part in relative.split("/"))
    )


def read_manifest(pack: Path) -> dict[str, str]:
    """``{path under refs/: sha256}`` from the Pack's manifest, {} when it has none."""

    manifest = pack / PAYLOAD_MANIFEST
    if not manifest.is_file():
        return {}
    listed: dict[str, str] = {}
    previous = b""
    for number, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
        match = _MANIFEST_LINE.fullmatch(line)
        if match is None:
            raise PackPayloadError(f"Pack {pack.name}: {PAYLOAD_MANIFEST} line {number} is not '<sha256>  <path>'")
        digest, relative = match.groups()
        if not _valid_relative(relative):
            raise PackPayloadError(
                f"Pack {pack.name}: {PAYLOAD_MANIFEST} lists {relative!r}, which is not a relative path inside refs/"
            )
        if relative.encode() <= previous:
            raise PackPayloadError(f"Pack {pack.name}: {PAYLOAD_MANIFEST} is not sorted bytewise without repeats at {relative}")
        previous = relative.encode()
        listed[relative] = digest
    return listed


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def payload_digests(root: Path, label: str) -> dict[str, str]:
    """``{relative path: sha256}`` of every regular file below ``root``.

    Refuses, prefixed with ``label``, a symlink or special file anywhere below
    it and a name no manifest line can carry.
    """

    digests: dict[str, str] = {}
    for directory, dirnames, filenames in os.walk(root):
        for name in (*dirnames, *filenames):
            path = Path(directory, name)
            relative = path.relative_to(root).as_posix()
            mode = path.lstat().st_mode
            if stat.S_ISLNK(mode):
                raise PackPayloadError(f"{label}/{relative} is a symlink")
            if not _valid_relative(relative):
                raise PackPayloadError(f"{label}/{relative!r} has a name a manifest line cannot carry")
            if stat.S_ISREG(mode):
                digests[relative] = _sha256(path)
            elif not stat.S_ISDIR(mode):
                raise PackPayloadError(f"{label}/{relative} is not a regular file")
    return digests


def format_manifest(digests: dict[str, str]) -> str:
    return "".join(f"{digests[relative]}  {relative}\n" for relative in sorted(digests, key=str.encode))


def pack_payload(pack: Path, repo_root: Path | None) -> list[tuple[str, Path]]:
    """The payload to install, as ``(path under refs/, file in the store)``; [] for a Pack without a manifest.

    Refuses, naming the Pack and the path: a manifest line that is not
    ``<sha256>  <relative path>`` in bytewise order; a listed path where the
    Pack has a tracked file or directory (both land in refs/); a symlink or
    special file in the store; a listed file missing there, a file there the
    manifest does not list, and a digest that differs.
    """

    listed = read_manifest(pack)
    if not listed:
        return []
    if repo_root is None:
        raise PackPayloadError(f"Pack {pack.name}: its payload is found from the repository root, which was not given")
    for relative in listed:
        target = pack / relative
        blocked = os.path.lexists(target) or any(
            os.path.lexists(pack / parent) and (os.path.islink(pack / parent) or not (pack / parent).is_dir())
            for parent in PurePosixPath(relative).parents
            if parent != PurePosixPath(".")
        )
        if blocked:
            raise PackPayloadError(f"Pack {pack.name}: payload path {relative} collides with a tracked file of the Pack")
    store = payload_dir(pack, repo_root)
    shown = f"{PAYLOAD_ROOT}/{pack.name}"
    present = payload_digests(store, f"Pack {pack.name}: {shown}") if store.is_dir() else {}
    for relative, digest in listed.items():
        if relative not in present:
            raise PackPayloadError(
                f"Pack {pack.name}: {shown}/{relative} is missing"
            )
        if present[relative] != digest:
            raise PackPayloadError(
                f"Pack {pack.name}: {shown}/{relative} has sha256 {present[relative]}, not the listed {digest}"
            )
    extra = sorted(present.keys() - listed.keys(), key=str.encode)
    if extra:
        raise PackPayloadError(
            f"Pack {pack.name}: {shown}/{extra[0]} is not listed in {PAYLOAD_MANIFEST}"
            + (f" ({len(extra) - 1} more)" if len(extra) > 1 else "")
        )
    return [(relative, store / relative) for relative in listed]
