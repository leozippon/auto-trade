"""Sandbox Pyright pin, config PIT roots, and Fold/Explore prompt contract."""

from __future__ import annotations

import json
import re
from pathlib import Path

from autotrade.agent.explore import EXPLORE_SYSTEM_PROMPT, META_EXPLORE_SYSTEM_PROMPT
from autotrade.agent.prompts import build_system_prompt

REPO = Path(__file__).resolve().parents[2]
DOCKERFILE = REPO / "ops/docker/sandbox.Dockerfile"
PYRIGHTCONFIG = REPO / "ops/docker/pyrightconfig.json"
COMMAND = (
    "pyright --project /opt/autotrade/pyrightconfig.json "
    "/mnt/agent/workspace /mnt/agent/output"
)


def test_dockerfile_pins_pyright_with_same_layer_version_check() -> None:
    text = DOCKERFILE.read_text(encoding="utf-8")
    assert "COPY ops/docker/pyrightconfig.json /opt/autotrade/pyrightconfig.json" in text
    match = re.search(
        r'npm install -g --no-fund --no-audit --registry "\$\{NPM_CONFIG_REGISTRY\}" '
        r"pyright@1\.1\.411\s*\\\s*\n\s*&& pyright --version",
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
    assert config["extraPaths"] == ["/opt/autotrade"]
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


def test_fold_and_explore_prompts_name_foreground_pyright_meta_does_not() -> None:
    fold = build_system_prompt(mode="fold", experiment_facts={})
    meta = build_system_prompt(mode="meta", experiment_facts={})
    assert COMMAND in fold
    assert COMMAND in EXPLORE_SYSTEM_PROMPT
    assert "debug 顾问" in fold
    assert "不得后台" in fold
    assert COMMAND not in meta
    assert "pyright" not in meta
    assert COMMAND not in META_EXPLORE_SYSTEM_PROMPT
    assert "pi-lens" not in fold
    assert "pi-lens" not in EXPLORE_SYSTEM_PROMPT
    assert "pi-lens" not in meta
