"""Sandbox Pyright pin and config PIT roots. (The prompts carry no pyright
how-to; that wording is pinned by the PROMPTS.md snapshot.)"""

from __future__ import annotations

import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DOCKERFILE = REPO / "ops/docker/sandbox.Dockerfile"
PYRIGHTCONFIG = REPO / "ops/docker/pyrightconfig.json"


def test_dockerfile_pins_pyright_with_same_layer_version_check() -> None:
    text = DOCKERFILE.read_text(encoding="utf-8")
    assert "COPY ops/docker/pyrightconfig.json /opt/autotrade/pyrightconfig.json" in text
    # The pinned release tarball, checked (tests/unit/test_sandbox_downloads.py)
    # and installed from the file, then the binary's own version check.
    match = re.search(
        r'"\$\{NPM_CONFIG_REGISTRY\}/pyright/-/pyright-1\.1\.411\.tgz"\s*\\\s*\n\s*'
        r"-o /tmp/pyright\.tgz\s*\\\s*\n.*\n\s*"
        r'&& npm install -g --prefix /usr/local --no-fund --no-audit --registry '
        r'"\$\{NPM_CONFIG_REGISTRY\}" /tmp/pyright\.tgz\s*\\\s*\n\s*'
        r"&& rm /tmp/pyright\.tgz\s*\\\s*\n\s*"
        r"&& /usr/local/bin/pyright --version",
        text,
    )
    assert match is not None
    install_block = match.group(0)
    assert "pip " not in install_block
    assert "pi-lens" not in text
    assert "ARG PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple" in text
    assert "ARG NPM_CONFIG_REGISTRY=https://registry.npmmirror.com" in text
    assert "ARG DEBIAN_MIRROR=https://mirrors.tuna.tsinghua.edu.cn/debian" in text
    assert "ARG DUCKDB_CLI_URL=" in text
    assert '"${DUCKDB_CLI_URL}"' in text
    assert "ENV HTTP_PROXY" not in text
    assert "ENV HTTPS_PROXY" not in text


def test_pyrightconfig_is_basic_and_excludes_pit_roots() -> None:
    config = json.loads(PYRIGHTCONFIG.read_text(encoding="utf-8"))
    assert config["pythonVersion"] == "3.11"
    assert config["typeCheckingMode"] == "basic"
    assert "include" not in config
    # The trusted runtime, plus the strategy package root so pyright resolves
    # main.py's sibling-module imports the same way the loader does.
    assert config["extraPaths"] == ["/opt/autotrade", "/mnt/agent/workspace/output"]
    assert config["reportMissingImports"] == "warning"
    exclude = config["exclude"]
    assert "**/snapshots" in exclude
    assert "**/artifacts" in exclude
    assert "**/node_modules" in exclude
    assert "**/cache" in exclude
    rendered = json.dumps(config)
    assert "/mnt/snapshot" not in rendered
    assert "/mnt/snapshots" not in rendered
    assert "/mnt/artifacts" not in rendered
    assert "stubPath" not in config


