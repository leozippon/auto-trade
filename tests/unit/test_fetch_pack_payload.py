"""scripts/experiments/fetch_pack_payload.py on local fixtures: no network.

What must hold. The build takes exactly what sources.json names, leaves out
links, strip globs, checkpoint formats and oversized repository files, refuses
an archive member that would leave its directory and a payload over the Pack
cap, and puts nothing in place unless the result reproduces the manifest (or
--write writes it); what it puts in place passes the experiment pre-flight.
Only the downloader is replaced, by one that serves local files.
"""

from __future__ import annotations

import io
import json
import re
import shutil
import tarfile
from pathlib import Path

import pytest

from autotrade.pipelines.pack_payload import (
    PAYLOAD_MANIFEST,
    PAYLOAD_SOURCES,
    PackPayloadError,
    pack_payload,
    payload_digests,
    read_sources,
)
from scripts.experiments import fetch_pack_payload as fetch

PACK = "configs/workspace_refs/paper_test"
STORE = "data/pack_payloads/paper_test"
COMMIT = "0123456789abcdef0123456789abcdef01234567"
SOURCE = {
    "slug": "demo",
    "title": "A demo paper",
    "venue": "ICLR 2024",
    "paper_url": "https://example.invalid/demo.pdf",
    "repo": "https://github.com/owner/demo",
    "commit": COMMIT,
    "licence": "MIT",
    "data_end": "2020-12-31",
    "strip": ["data", "*.ipynb"],
    "dataset": ["https://example.invalid/prices.csv"],
}
needs_pdftotext = pytest.mark.skipif(shutil.which("pdftotext") is None, reason="the fetch runs pdftotext")


def _pdf(text: str) -> bytes:
    """A one-page PDF whose only text is ``text``, with a correct cross-reference table."""

    stream = f"BT /F1 12 Tf 10 50 Td ({text}) Tj ET".encode()
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 300 100] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length %d >>\nstream\n%s\nendstream" % (len(stream), stream),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, 1):
        offsets.append(len(out))
        out += b"%d 0 obj\n%s\nendobj\n" % (number, body)
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    out += b"".join(b"%010d 00000 n \n" % offset for offset in offsets)
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objects) + 1, xref)
    return bytes(out)


def _tar(path: Path, files: dict[str, bytes], links: dict[str, str] | None = None) -> None:
    """A gzipped tar as GitHub serves an archive: every member below one top directory."""

    with tarfile.open(path, "w:gz") as tar:
        top = tarfile.TarInfo(f"demo-{COMMIT}")
        top.type = tarfile.DIRTYPE
        tar.addfile(top)
        for name, data in files.items():
            info = tarfile.TarInfo(f"demo-{COMMIT}/{name}")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
        for name, target in (links or {}).items():
            info = tarfile.TarInfo(f"demo-{COMMIT}/{name}")
            info.type = tarfile.SYMTYPE
            info.linkname = target
            tar.addfile(info)


REPOSITORY = {
    "README.md": b"# demo\n",
    ".gitignore": b"*.pt\n",
    "src/model.py": b"def forecast(x):\n    return x\n",
    "ckpt/model.pt": b"weights",
    "data/raw.csv": b"date,close\n",
    "notebooks/explore.ipynb": b"{}",
    "results/big.npy": b"\0" * 4096,
}


@pytest.fixture
def layout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, dict[str, Path]]:
    """A repository with the Pack, and the files the fake downloader serves by URL."""

    monkeypatch.setattr(fetch, "FILE_MAX_BYTES", 1024)
    repo = tmp_path / "repo"
    pack = repo / PACK
    pack.mkdir(parents=True)
    (pack / "README.md").write_text("Reproduce papers/demo/.\n", encoding="utf-8")
    (pack / PAYLOAD_SOURCES).write_text(json.dumps([SOURCE]), encoding="utf-8")
    served = tmp_path / "served"
    served.mkdir()
    (served / "demo.pdf").write_bytes(_pdf("Momentum in demo markets"))
    (served / "prices.csv").write_bytes(b"date,close\n" * 200)  # over FILE_MAX_BYTES, but named in sources.json
    _tar(served / "demo.tar.gz", REPOSITORY, links={"link.py": "src/model.py"})
    urls = {
        SOURCE["paper_url"]: served / "demo.pdf",
        SOURCE["dataset"][0]: served / "prices.csv",
        f"{SOURCE['repo']}/archive/{COMMIT}.tar.gz": served / "demo.tar.gz",
    }
    return repo, urls


def _serve(urls: dict[str, Path]):
    def download(url: str, dest: Path) -> None:
        shutil.copyfile(urls[url], dest)  # an unexpected URL is a KeyError

    return download


@needs_pdftotext
def test_a_written_manifest_is_reproduced_and_what_lands_passes_the_preflight(layout: tuple[Path, dict[str, Path]]) -> None:
    repo, urls = layout
    pack = repo / PACK
    with pytest.raises(fetch.FetchError, match=r"has no PAYLOAD\.sha256: --write creates it"):
        fetch.fetch(pack, repo, write=False, download=_serve(urls))
    assert not (repo / STORE).exists()
    fetch.fetch(pack, repo, write=True, download=_serve(urls))
    landed = {relative for relative, _ in pack_payload(pack, repo)}
    assert landed == {
        "papers/demo/paper.pdf", "papers/demo/paper.txt", "data/demo/prices.csv",
        "papers/demo/repo/README.md", "papers/demo/repo/.gitignore", "papers/demo/repo/src/model.py",
    }
    assert "Momentum in demo markets" in (repo / STORE / "papers/demo/paper.txt").read_text(encoding="utf-8")
    assert (repo / STORE / "papers/demo/repo/src/model.py").read_bytes() == REPOSITORY["src/model.py"]
    # The default mode rebuilds, holds the build to the manifest and puts it in place again.
    fetch.fetch(pack, repo, write=False, download=_serve(urls))
    before = payload_digests(repo / STORE, STORE)
    assert pack_payload(pack, repo)
    # A build that differs only in the extracted text points at poppler; any other difference does not.
    manifest = (pack / PAYLOAD_MANIFEST).read_text(encoding="utf-8")
    for path, poppler in (("papers/demo/paper.txt", True), ("papers/demo/repo/src/model.py", False)):
        (pack / PAYLOAD_MANIFEST).write_text(
            manifest.replace(f"{before[path]}  {path}", f"{'0' * 64}  {path}"), encoding="utf-8"
        )
        with pytest.raises(fetch.FetchError, match=re.escape(f"does not reproduce PAYLOAD.sha256: differs: {path}")) as refused:
            fetch.fetch(pack, repo, write=False, download=_serve(urls))
        assert ("poppler" in str(refused.value)) is poppler
        assert payload_digests(repo / STORE, STORE) == before
    assert not any((repo / ".runtime" / "pack_payloads").iterdir())


def test_an_archive_member_leaving_its_directory_refuses_the_build(layout: tuple[Path, dict[str, Path]]) -> None:
    repo, urls = layout
    _tar(urls[f"{SOURCE['repo']}/archive/{COMMIT}.tar.gz"], {"README.md": b"# demo\n", "../../escape.txt": b"out\n"})
    with pytest.raises(fetch.FetchError, match=r"demo\.tar\.gz refused: .*escape\.txt"):
        fetch.fetch(repo / PACK, repo, write=True, download=_serve(urls))
    assert not (repo / STORE).exists() and not (repo / PACK / PAYLOAD_MANIFEST).exists()
    assert not list(repo.parent.rglob("escape.txt"))
    assert not any((repo / ".runtime" / "pack_payloads").iterdir())


def test_a_payload_over_the_pack_cap_is_refused(layout: tuple[Path, dict[str, Path]], monkeypatch: pytest.MonkeyPatch) -> None:
    repo, urls = layout
    monkeypatch.setattr(fetch, "PACK_MAX_BYTES", 1024)
    with pytest.raises(fetch.FetchError, match="exceeds 0 MiB"):
        fetch.fetch(repo / PACK, repo, write=True, download=_serve(urls))
    assert not (repo / STORE).exists() and not (repo / PACK / PAYLOAD_MANIFEST).exists()


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        ({"commit": "abc123"}, "40-hex commit"),
        ({"repo": "https://gitlab.com/owner/demo"}, "https://github.com/<owner>/<name>"),
        ({"licence": ""}, "licence is not a non-empty string"),
        ({"data_end": "2020/12/31"}, "data_end '2020/12/31' is not an ISO date"),
        ({"license": "MIT"}, "unknown ['license']"),
        ({"arxiv": "2401.01234"}, "arxiv is not a versioned id"),
        ({"wheels": ["numpy>=2"]}, "name==version"),
        ({"weights": [{"url": "https://example.invalid/w.pt", "revision": "abc"}]}, "each URL naming its revision"),
    ],
)
def test_sources_refuse_an_entry_missing_provenance(tmp_path: Path, change: dict[str, object], reason: str) -> None:
    pack = tmp_path / "paper_test"
    pack.mkdir()
    (pack / PAYLOAD_SOURCES).write_text(json.dumps([SOURCE]), encoding="utf-8")
    assert read_sources(pack) == [SOURCE]
    (pack / PAYLOAD_SOURCES).write_text(json.dumps([{**SOURCE, **change}]), encoding="utf-8")
    with pytest.raises(PackPayloadError, match=r"^Pack paper_test: sources\.json ") as refused:
        read_sources(pack)
    assert reason in str(refused.value)
