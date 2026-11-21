"""A research Pack's payload: untracked files its manifest lists, installed beside the Pack.

A Pack (``configs/workspace_refs/<pack>/``, an experiment's ``workspace_reference``)
is tracked and small. A paper Pack also carries what git should not -- the
paper, its text, the authors' repository at one commit, wheels or weights. It
lists them in ``PAYLOAD.sha256`` (sha256sum format, ``<sha256>  <path under
refs/>``, sorted bytewise) and keeps them untracked under
``data/pack_payloads/<pack>/`` at the same paths; ``sources.json`` beside the
manifest records where each paper came from and how
``scripts/experiments/fetch_pack_payload.py`` fetches it again (``read_sources``).

``pack_payload`` is the one check. The worker's pre-flight (the console's
create, a round's dry run, a run on the ZGCA instance), the session start that
copies the files into ``refs/`` and the ZGCA deploy all call it, so no
experiment starts on a payload that is missing or differs from its manifest.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from datetime import date
from pathlib import Path, PurePosixPath

PAYLOAD_MANIFEST = "PAYLOAD.sha256"
PAYLOAD_SOURCES = "sources.json"
# Repository-relative; one directory per Pack, named as the Pack's directory.
PAYLOAD_ROOT = PurePosixPath("data/pack_payloads")

_MANIFEST_LINE = re.compile(r"([0-9a-f]{64})  (.+)")
# What each sources.json entry must and may carry; read_sources says what each means.
SOURCE_FIELDS = ("slug", "title", "venue", "paper_url", "repo", "commit", "licence", "data_end", "strip")
OPTIONAL_SOURCE_FIELDS = ("arxiv", "wheels", "weights", "dataset")
_SLUG = re.compile(r"[a-z0-9][a-z0-9._-]*")
_URL = re.compile(r"https?://\S+")
_GITHUB_REPO = re.compile(r"https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+")
_COMMIT = re.compile(r"[0-9a-f]{40}")
# A versioned arXiv id (2401.01234v2, hep-th/9901001v1): its source never changes.
_ARXIV = re.compile(r"\S+v[0-9]+")
_PINNED_REQUIREMENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*==[A-Za-z0-9][A-Za-z0-9.+!_-]*")


class PackPayloadError(ValueError):
    """A Pack whose payload or sources are not as its files state; the message names the Pack and the path."""


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



def _text(value: object, pattern: re.Pattern[str] | None = None) -> bool:
    """A non-empty string, matching ``pattern`` in full when one is given."""

    return isinstance(value, str) and bool(value.strip()) and (pattern is None or bool(pattern.fullmatch(value)))


def read_sources(pack: Path) -> list[dict[str, object]]:
    """The papers the Pack's ``sources.json`` lists, each entry checked.

    Every entry carries ``slug`` (its directory under ``papers/``), ``title``,
    ``venue`` (e.g. "ICLR 2024"), ``paper_url`` (the PDF itself), ``repo`` (a
    GitHub repository URL) with ``commit`` (40 hex) or both null, ``licence``
    (an SPDX id or "none stated"), ``data_end`` (the ISO date of the last day
    of the paper's evaluation data when that data is financial, else null) and
    ``strip`` (globs dropped from the repository). It may carry ``arxiv`` (a
    versioned id, whose source becomes ``tex/``), ``wheels`` (pinned
    ``name==version`` strings), ``weights`` (``{"url", "revision"}`` objects,
    each URL naming its revision) and ``dataset`` (URLs).
    """

    def refuse(message: str) -> PackPayloadError:
        return PackPayloadError(f"Pack {pack.name}: {PAYLOAD_SOURCES} {message}")

    try:
        entries = json.loads((pack / PAYLOAD_SOURCES).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise refuse(f"is unreadable: {exc}") from exc
    if not isinstance(entries, list) or not entries:
        raise refuse("is not a list of papers")
    slugs: set[str] = set()
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise refuse(f"entry {index} is not an object")
        missing = [name for name in SOURCE_FIELDS if name not in entry]
        unknown = sorted(entry.keys() - {*SOURCE_FIELDS, *OPTIONAL_SOURCE_FIELDS})
        if missing or unknown:
            raise refuse(f"entry {index} lacks {missing} or carries unknown {unknown}")
        slug = entry["slug"]
        if not _text(slug, _SLUG) or slug in slugs:
            raise refuse(f"entry {index} has slug {slug!r}: lowercase letters, digits, '.', '_' or '-', unique")
        slugs.add(slug)
        for name in ("title", "venue", "licence"):
            if not _text(entry[name]):
                raise refuse(f"{slug}: {name} is not a non-empty string")
        if not _text(entry["paper_url"], _URL):
            raise refuse(f"{slug}: paper_url is not an http(s) URL")
        if (entry["repo"], entry["commit"]) != (None, None) and not (
            _text(entry["repo"], _GITHUB_REPO) and _text(entry["commit"], _COMMIT)
        ):
            raise refuse(f"{slug}: repo must be https://github.com/<owner>/<name> with a 40-hex commit, or both null")
        data_end = entry["data_end"]
        if data_end is not None:
            try:
                valid = isinstance(data_end, str) and date.fromisoformat(data_end).isoformat() == data_end
            except ValueError:
                valid = False
            if not valid:
                raise refuse(f"{slug}: data_end {data_end!r} is not an ISO date (YYYY-MM-DD) or null")
        if not isinstance(entry["strip"], list) or not all(_text(glob) for glob in entry["strip"]):
            raise refuse(f"{slug}: strip is not a list of globs")
        if "arxiv" in entry and not _text(entry["arxiv"], _ARXIV):
            raise refuse(f"{slug}: arxiv is not a versioned id such as 2401.01234v2")
        wheels = entry.get("wheels", [])
        if not isinstance(wheels, list) or not all(_text(wheel, _PINNED_REQUIREMENT) for wheel in wheels):
            raise refuse(f"{slug}: wheels is not a list of name==version requirements")
        weights = entry.get("weights", [])
        if not isinstance(weights, list) or not all(
            isinstance(weight, dict) and weight.keys() == {"url", "revision"}
            and _text(weight["url"], _URL) and _text(weight["revision"]) and weight["revision"] in weight["url"]
            for weight in weights
        ):
            raise refuse(f"{slug}: weights is not a list of {{url, revision}}, each URL naming its revision")
        dataset = entry.get("dataset", [])
        if not isinstance(dataset, list) or not all(_text(url, _URL) for url in dataset):
            raise refuse(f"{slug}: dataset is not a list of http(s) URLs")
    return entries
