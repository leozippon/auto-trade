"""Every file the sandbox Dockerfile fetches by URL is checked against a recorded SHA-256.

A curl in a build step must write a file, and the step's next command must be
``echo "<digest>  <file>" | sha256sum -c -`` before anything reads that file.
The digest is written in the Dockerfile, or is a build arg declared there with
a digest as its default, or with an empty one for a download only a build that
names its URL makes: that build must name the digest too, and the checks run
below show that an empty digest fails like a wrong one.
"""

from __future__ import annotations

import hashlib
import os
import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DOCKERFILE = REPO / "ops/docker/sandbox.Dockerfile"
DIGEST = "[0-9a-f]{64}"
# curl in command position: a step's start, or after &&, ;, then or a pipe.
INVOKED = r"(?:^|&&|;|\bthen\b|\|)\s*curl\s"
FETCH = re.compile(INVOKED + r"[^&;|]*?-o\s+(\S+)")
CHECK = re.compile(r'\s*&&\s*(echo "(\$\{(\w+)\}|' + DIGEST + r')  (\S+)" \| sha256sum -c -)\s*&&')


def _instructions() -> list[list[str]]:
    """[keyword, rest] of each instruction, continuation lines joined and comment lines dropped."""

    lines = [line for line in DOCKERFILE.read_text(encoding="utf-8").splitlines() if not line.lstrip().startswith("#")]
    joined = "\n".join(lines).replace("\\\n", " ")
    return [line.strip().split(None, 1) for line in joined.splitlines() if line.strip()]


def _checks() -> list[dict[str, str]]:
    """Each download's check: its command, the file, the digest as written, the build arg it names and what that arg records."""

    defaults: dict[str, str] = {}
    checks = []
    for keyword, body in _instructions():
        if keyword == "ARG":
            name, _, default = body.partition("=")
            defaults[name] = default
        assert not (keyword == "ADD" and "://" in body), f"ADD fetches a URL unchecked: {body}"
        if keyword != "RUN":
            continue
        assert "wget" not in body, body
        fetches = list(FETCH.finditer(body))
        assert len(fetches) == len(re.findall(INVOKED, body)), f"a curl that writes no file: {body}"
        for fetch in fetches:
            check = CHECK.match(body, fetch.end())
            assert check is not None, f"{fetch.group(1)} is not checked against a SHA-256 right after its download"
            command, digest, arg, path = check.groups()
            assert path == fetch.group(1), command
            if arg:
                assert arg in defaults, f"{arg} is not a build arg declared before {command}"
                assert re.fullmatch(f"({DIGEST})?", defaults[arg]), f"{arg} records no digest: {defaults[arg]!r}"
            checks.append({"command": command, "path": path, "digest": digest, "arg": arg or "",
                           "recorded": defaults[arg] if arg else digest})
    return checks


def test_every_url_download_is_checked_against_a_recorded_sha256() -> None:
    checks = _checks()
    assert [check["path"] for check in checks] == ["/tmp/duckdb_cli.zip", "/tmp/pyright.tgz", "/tmp/cuda-compat.deb"]
    # The two the default build downloads record their digest; the CUDA
    # compatibility package's comes with its URL (ops/zgca/remote.sh).
    assert [bool(check["recorded"]) for check in checks] == [True, True, False]


def test_a_check_passes_only_the_bytes_its_digest_names(tmp_path: Path) -> None:
    """Each check as the build's shell runs it, on a stand-in for the download:
    it fails against the recorded digest and against none, and passes only the
    digest of the bytes actually there."""

    stand_in = tmp_path / "download"
    stand_in.write_bytes(b"not the release\n")
    own = hashlib.sha256(stand_in.read_bytes()).hexdigest()
    for check in _checks():
        command = check["command"].replace(check["path"], str(stand_in))
        for digest, passes in ((check["recorded"], False), ("", False), (own, True)):
            env = {"PATH": os.environ["PATH"]}
            if check["arg"]:
                env[check["arg"]] = digest
                script = command
            else:
                script = command.replace(check["digest"], digest)
            completed = subprocess.run(["sh", "-c", script], env=env, capture_output=True, text=True, check=False)
            assert (completed.returncode == 0) is passes, (check["path"], digest, completed.stdout, completed.stderr)
