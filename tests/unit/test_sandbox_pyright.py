"""Sandbox Pyright pin, config PIT roots, and Fold/sub-agent prompt contract."""

from __future__ import annotations

import json
import re
from pathlib import Path

from autotrade.agent.subagent import META_SUBAGENT_SYSTEM_PROMPT, subagent_system_prompt
from autotrade.agent.prompts import build_system_prompt

REPO = Path(__file__).resolve().parents[2]
DOCKERFILE = REPO / "ops/docker/sandbox.Dockerfile"
PYRIGHTCONFIG = REPO / "ops/docker/pyrightconfig.json"
# The documented how-to (the working copy is workspace/output, so the
# workspace target already covers it); it belongs in docs, never in a prompt.
COMMAND = "pyright --project /opt/autotrade/pyrightconfig.json /mnt/agent/workspace"


def test_dockerfile_pins_pyright_with_same_layer_version_check() -> None:
    text = DOCKERFILE.read_text(encoding="utf-8")
    assert "COPY ops/docker/pyrightconfig.json /opt/autotrade/pyrightconfig.json" in text
    match = re.search(
        r'npm install -g --prefix /usr/local --no-fund --no-audit --registry '
        r'"\$\{NPM_CONFIG_REGISTRY\}" pyright@1\.1\.411\s*\\\s*\n\s*'
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
    lowered = text.lower()
    for project_hash_contract in (
        "pyright_sha256",
        "sha256sum",
        "@sha256:",
        "repodigest",
        "checksum",
    ):
        assert project_hash_contract not in lowered


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


def test_agent_prompts_leave_pyright_how_to_out_of_system_text() -> None:
    fold = build_system_prompt(mode="fold", experiment_facts={})
    meta = build_system_prompt(mode="meta", experiment_facts={})
    # The writing child is the one that would be tempted to type-check.
    child = subagent_system_prompt("fold", "developer")
    assert COMMAND not in fold
    assert COMMAND not in child
    assert COMMAND not in meta
    assert COMMAND not in META_SUBAGENT_SYSTEM_PROMPT
    assert "pyright" not in fold
    assert "pyright" not in child
    assert "pyright" not in meta
    assert "pi-lens" not in fold
    assert "pi-lens" not in child
    assert "pi-lens" not in meta
