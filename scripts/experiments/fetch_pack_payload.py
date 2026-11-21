#!/usr/bin/env python3
"""Fetch a paper Pack's payload into data/pack_payloads/<pack>/.

<pack> is the Pack's directory (configs/workspace_refs/<name>). Its
sources.json says what to fetch (``pack_payload.read_sources``); for each
paper the build writes, into a fresh staging directory under
.runtime/pack_payloads/:

  papers/<slug>/paper.pdf   the PDF at ``paper_url``
  papers/<slug>/paper.txt   ``pdftotext -layout`` of it, made here: the sandbox
                            has no PDF reader and the Agent's file tools read text
  papers/<slug>/tex/        the arXiv source, when ``arxiv`` names a version
  papers/<slug>/repo/       the GitHub archive of ``repo`` at ``commit``, without
                            its top directory, its links, the ``strip`` globs,
                            the checkpoint formats in DEFAULT_STRIP and any file
                            over FILE_MAX_BYTES
  wheels/                   ``pip download --only-binary=:all: --no-deps`` of the
                            pinned ``wheels``, for this host's Python and platform
                            (the sandbox image's: CPython 3.11, glibc 2.41)
  weights/<slug>/, data/<slug>/   the ``weights`` and ``dataset`` downloads

A strip glob is matched (fnmatch, ``*`` crossing ``/``) against each path
under the repository and every directory above it, so ``data`` drops a whole
top-level directory. Archives are unpacked with tarfile's ``data`` filter,
which refuses a member leaving the directory, and nothing downloaded is ever
executed. Only a file named in sources.json (the PDF, wheels, weights,
dataset) may exceed FILE_MAX_BYTES; the whole payload may not exceed
PACK_MAX_BYTES. A GitHub URL is tried through the proxy prefix
ZGCA_GITHUB_PROXY names first, as ops/zgca/zgca.py fetches its releases.
A failed download is tried again after each of RETRY_DELAYS, and so is a PDF
that does not end with its end-of-file marker (a cut-off transfer).

By default the build must reproduce the Pack's PAYLOAD.sha256 file by file;
it then replaces data/pack_payloads/<pack>/ by renames (a reader in between
finds it missing, never half written). ``--write`` creates or deliberately
changes the payload: it writes PAYLOAD.sha256 from the build instead. Either
way the sizes are printed and the result is verified as the experiment
pre-flight verifies it (``pack_payload``).
"""

from __future__ import annotations

import argparse
import fnmatch
import gzip
import http.client
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.parse
import urllib.request
from collections.abc import Callable, Sequence
from pathlib import Path, PurePosixPath

_SCRIPTS = Path(__file__).resolve().parents[1]
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from _bootstrap import add_repo_src

REPO_ROOT = add_repo_src(__file__)

from autotrade.pipelines.pack_payload import (
    PAYLOAD_MANIFEST,
    PackPayloadError,
    format_manifest,
    pack_payload,
    payload_digests,
    payload_dir,
    read_manifest,
    read_sources,
)

MIB = 2**20
PACK_MAX_BYTES = 512 * MIB
FILE_MAX_BYTES = 64 * MIB
# Checkpoint and weight formats a repository may carry; weights a Pack needs
# are named in sources.json instead.
DEFAULT_STRIP = ("*.pt", "*.pth", "*.ckpt", "*.safetensors", "*.bin", "*.h5", "*.hdf5", "*.onnx", "*.tflite", "*.pb")
# The variable ops/zgca/zgca.py reads (RELEASE_PROXY_ENV): a URL prefix tried before a GitHub URL.
GITHUB_PROXY_ENV = "ZGCA_GITHUB_PROXY"
# Seconds waited before the second and the third try of a download; downloads
# from this host fail intermittently.
RETRY_DELAYS = (10.0, 60.0)
# Where a whole PDF carries its end-of-file marker: in its last 1024 bytes.
PDF_EOF_MARKER, PDF_TAIL_BYTES = b"%%EOF", 1024

Download = Callable[[str, Path], None]


class FetchError(RuntimeError):
    """A payload that could not be built as sources.json and the caps say; the message says why."""


class DownloadFailed(FetchError):
    """A download that did not arrive whole; it is tried again (``_download``)."""


def download_url(url: str, dest: Path) -> None:
    """``url`` into the file ``dest``, through the GitHub proxy first for a GitHub URL; at most PACK_MAX_BYTES.

    Raises DownloadFailed when no source answered with the whole file."""

    proxy = os.environ.get(GITHUB_PROXY_ENV, "")
    sources = ([proxy + url] if proxy and url.startswith("https://github.com/") else []) + [url]
    failures = []
    for source in sources:
        request = urllib.request.Request(source, headers={"User-Agent": "fetch-pack-payload"})
        try:
            with urllib.request.urlopen(request, timeout=120) as response, dest.open("wb") as out:
                copied = 0
                while chunk := response.read(MIB):
                    copied += len(chunk)
                    if copied > PACK_MAX_BYTES:
                        raise FetchError(f"{url} is larger than the {PACK_MAX_BYTES // MIB} MiB a Pack may hold")
                    out.write(chunk)
            return
        except (OSError, http.client.HTTPException) as exc:  # URLError, a cut-off body (IncompleteRead)
            failures.append(f"{source}: {exc}")
    raise DownloadFailed("no source answered: " + "; ".join(failures))


def _download(download: Download, url: str, dest: Path, *, pdf: bool = False) -> None:
    """``download(url, dest)``, tried again after each of RETRY_DELAYS while it raises DownloadFailed.

    With ``pdf`` the file must start as a PDF (else FetchError: the URL serves
    something else) and end with PDF_EOF_MARKER (else the transfer was cut
    off, which is tried again like a failed download)."""

    for delay in (*RETRY_DELAYS, None):
        try:
            download(url, dest)
            if pdf:
                with dest.open("rb") as handle:
                    if handle.read(5) != b"%PDF-":
                        raise FetchError(f"{url} did not answer a PDF")
                    handle.seek(max(0, dest.stat().st_size - PDF_TAIL_BYTES))
                    if PDF_EOF_MARKER not in handle.read():
                        raise DownloadFailed(f"{url}: the PDF ends without {PDF_EOF_MARKER.decode()} (cut off)")
            return
        except DownloadFailed as exc:
            if delay is None:
                raise
            print(f"{exc}; trying again in {delay:g} s", flush=True)
            time.sleep(delay)


def _stripped(relative: str, globs: Sequence[str]) -> bool:
    path = PurePosixPath(relative)
    return any(
        fnmatch.fnmatchcase(str(candidate), glob)
        for candidate in (path, *path.parents)
        if str(candidate) != "."
        for glob in globs
    )


def unpack(archive: Path, dest: Path, globs: Sequence[str], dropped: dict[str, list[str]]) -> None:
    """Unpack the tar ``archive`` into ``dest``, without a single top directory all members share.

    Links, the ``globs``, DEFAULT_STRIP and files over FILE_MAX_BYTES stay
    out, each recorded under its reason in ``dropped``; a member the ``data``
    filter refuses (one leaving ``dest``, a device) refuses the archive.
    """

    with tarfile.open(archive, "r:*") as tar:
        members = tar.getmembers()
        tops = {member.name.split("/", 1)[0] for member in members}
        top = f"{tops.pop()}/" if len(tops) == 1 and all("/" in m.name or m.isdir() for m in members) else ""
        kept = []
        for member in members:
            relative = member.name.removeprefix(top)
            if member.isdir() or not relative:
                continue
            if member.issym() or member.islnk():
                dropped["link"].append(relative)
            elif _stripped(relative, globs):
                dropped["strip glob"].append(relative)
            elif _stripped(relative, DEFAULT_STRIP):
                dropped["checkpoint format"].append(relative)
            elif member.size > FILE_MAX_BYTES:
                dropped["over the file cap"].append(f"{relative} ({member.size / MIB:.1f} MiB)")
            else:
                kept.append(member.replace(name=relative, deep=False))
        try:
            tar.extractall(dest, members=kept, filter="data")
        except tarfile.FilterError as exc:
            raise FetchError(f"{archive.name} refused: {exc}") from exc


def _basename(url: str) -> str:
    name = PurePosixPath(urllib.parse.urlsplit(url).path).name
    if not name or name in (".", ".."):
        raise FetchError(f"{url} names no file")
    return name


def build(pack: Path, payload: Path, work: Path, download: Download) -> None:
    """Fetch every paper of ``pack`` into the empty directory ``payload``; ``work`` holds the archives."""

    dropped: dict[str, list[str]] = {"link": [], "strip glob": [], "checkpoint format": [], "over the file cap": []}
    requirements: list[str] = []
    for entry in read_sources(pack):
        slug = str(entry["slug"])
        paper = payload / "papers" / slug
        paper.mkdir(parents=True)
        print(f"{slug}: {entry['paper_url']}", flush=True)
        _download(download, str(entry["paper_url"]), paper / "paper.pdf", pdf=True)
        subprocess.run(["pdftotext", "-layout", str(paper / "paper.pdf"), str(paper / "paper.txt")], check=True)
        if "arxiv" in entry:
            source = work / f"{slug}.arxiv"
            _download(download, f"https://arxiv.org/e-print/{entry['arxiv']}", source)
            if tarfile.is_tarfile(source):
                unpack(source, paper / "tex", [], dropped)
            elif source.read_bytes()[:2] == b"\x1f\x8b":  # one gzipped TeX file
                (paper / "tex").mkdir()
                (paper / "tex" / "main.tex").write_bytes(gzip.decompress(source.read_bytes()))
            else:
                print(f"{slug}: arXiv {entry['arxiv']} has no TeX source; tex/ left out", flush=True)
        if entry["repo"] is not None:
            archive = work / f"{slug}.tar.gz"
            _download(download, f"{entry['repo']}/archive/{entry['commit']}.tar.gz", archive)
            unpack(archive, paper / "repo", [str(glob) for glob in entry["strip"]], dropped)
        for kind, urls in (("weights", [weight["url"] for weight in entry.get("weights", [])]), ("data", entry.get("dataset", []))):
            for url in urls:
                target = payload / kind / slug / _basename(url)
                if target.exists():
                    raise FetchError(f"{slug}: two {kind} downloads are named {target.name}")
                target.parent.mkdir(parents=True, exist_ok=True)
                _download(download, url, target)
        requirements.extend(str(wheel) for wheel in entry.get("wheels", []))
    if requirements:
        subprocess.run(
            [sys.executable, "-m", "pip", "download", "--only-binary=:all:", "--no-deps",
             "--dest", str(payload / "wheels"), *sorted(set(requirements))],
            check=True,
        )
    for reason, paths in dropped.items():
        if paths:
            print(f"left out ({reason}): {len(paths)} files" + (f": {', '.join(paths[:10])}" if reason != "strip glob" else ""))


def _sizes(payload: Path) -> int:
    """Print the bytes of each top part of the payload; returns the total."""

    parts: dict[str, int] = {}
    for path in payload.rglob("*"):
        if path.is_file():
            relative = path.relative_to(payload).parts
            key = "/".join(relative[:3] if relative[0] == "papers" else relative[:2])
            parts[key] = parts.get(key, 0) + path.stat().st_size
    for key in sorted(parts):
        print(f"{parts[key] / MIB:10.2f} MiB  {key}")
    total = sum(parts.values())
    print(f"{total / MIB:10.2f} MiB  total (cap {PACK_MAX_BYTES // MIB} MiB)")
    return total


def fetch(pack: Path, repo_root: Path, *, write: bool, download: Download = download_url) -> None:
    """Build ``pack``'s payload, hold it to the manifest (or write the manifest) and put it in place."""

    listed = {} if write else read_manifest(pack)
    if not write and not listed:
        raise FetchError(f"Pack {pack.name} has no {PAYLOAD_MANIFEST}: --write creates it")
    staging_root = repo_root / ".runtime" / "pack_payloads"
    staging_root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f"{pack.name}-", dir=staging_root))
    try:
        payload, work = staging / "payload", staging / "work"
        payload.mkdir()
        work.mkdir()
        build(pack, payload, work, download)
        if _sizes(payload) > PACK_MAX_BYTES:
            raise FetchError(f"Pack {pack.name}: the payload exceeds {PACK_MAX_BYTES // MIB} MiB; strip more of it")
        built = payload_digests(payload, f"Pack {pack.name}: the build")
        if write:
            (pack / PAYLOAD_MANIFEST).write_text(format_manifest(built), encoding="utf-8")
            print(f"wrote {pack / PAYLOAD_MANIFEST}: {len(built)} files")
        else:
            _require_listed(pack, listed, built)
        target = payload_dir(pack, repo_root)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            target.rename(staging / "replaced")
        payload.rename(target)
        pack_payload(pack, repo_root)
        print(f"{target.relative_to(repo_root)}: in place and verified")
    finally:
        shutil.rmtree(staging)


def _require_listed(pack: Path, listed: dict[str, str], built: dict[str, str]) -> None:
    missing = sorted(listed.keys() - built.keys())
    extra = sorted(built.keys() - listed.keys())
    differ = sorted(path for path in listed.keys() & built.keys() if listed[path] != built[path])
    if missing or extra or differ:
        said = "; ".join(f"{label}: {', '.join(paths[:5])}" + (" ..." if len(paths) > 5 else "")
                         for label, paths in (("not built", missing), ("not listed", extra), ("differs", differ)) if paths)
        hint = ""
        if not missing and not extra and all(PurePosixPath(path).name == "paper.txt" for path in differ):
            hint = (" (only pdftotext output differs: a poppler version other than the one that wrote the manifest "
                    "may be the cause; compare `pdftotext -v`, and --write to adopt this text)")
        raise FetchError(f"Pack {pack.name}: the build does not reproduce {PAYLOAD_MANIFEST}: {said}{hint}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pack", help="the Pack's directory, e.g. configs/workspace_refs/<name>")
    parser.add_argument("--write", action="store_true", help="write PAYLOAD.sha256 from this build instead of holding the build to it")
    args = parser.parse_args(argv)
    pack = (REPO_ROOT / args.pack).resolve()
    if not pack.is_dir() or REPO_ROOT not in pack.parents:
        parser.error(f"{args.pack} is not a directory inside the repository")
    try:
        fetch(pack, REPO_ROOT, write=args.write)
    except (FetchError, PackPayloadError, subprocess.CalledProcessError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
