"""The pretrained-model store: pinned at arm creation, compared at every container start.

Every test runs against the fixture store ``tests/conftest.py`` points
``environment.pretrained.PRETRAINED_STORE`` at; none touches the host weights.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from autotrade.environment import pretrained
from autotrade.environment.executor import (
    CONTAINER_PRETRAINED_DIR,
    DockerStrategyExecutor,
    StrategyExecutionError,
)
from autotrade.environment.pretrained import (
    PretrainedStoreError,
    manifest_digest,
    pretrained_pin,
    require_pretrained_store,
)
from autotrade.environment.sandbox import (
    PRETRAINED_SESSION_MOUNT,
    DockerSandbox,
    LocalSandbox,
    SandboxConfig,
    SandboxSpec,
)
from autotrade.environment.strategy import CN_TZ, AccountSnapshot, StrategyContext
from autotrade.environment.strategy_loader import (
    StrategyLoadError,
    validate_strategy_source,
)
from autotrade.paper.book import create_book, load_book
from autotrade.paper.engine import DailyPaperEngine
from autotrade.pipelines.config import DEFAULT_RESEARCH_GEOMETRY
from autotrade.pipelines.worker import (
    _strategy_sandbox_from_spec,
    resolve_worker_options,
)
from autotrade.webui.server import create_app
from tests.unit.gpu_probe import stubbed_gpu_probe
from tests.unit.paper_book_fixture import paper_root
from tests.unit.webui_research_arm import build_arm

OTHER_PIN = "0" * 64


def _strategy(directory: Path) -> Path:
    path = directory / "main.py"
    path.write_text("def generate_orders(context):\n    return []\n", encoding="utf-8")
    return path


# ----- the store and its pin -----


def test_the_pin_is_the_manifest_digest_and_only_that_store_passes(tmp_path: Path) -> None:
    pin = manifest_digest()
    assert require_pretrained_store(pin) == pretrained.PRETRAINED_STORE.resolve()
    with pytest.raises(PretrainedStoreError, match="not the one this arm or book pinned"):
        require_pretrained_store(OTHER_PIN)
    # A store whose manifest lists one more file is another store.
    other = tmp_path / "other"
    other.mkdir()
    manifest = (pretrained.PRETRAINED_STORE / "MANIFEST.sha256").read_text(encoding="utf-8")
    (other / "MANIFEST.sha256").write_text(manifest + f"{OTHER_PIN}  Extra/x\n", encoding="utf-8")
    with pytest.raises(PretrainedStoreError, match="not the one"):
        require_pretrained_store(pin, other)
    with pytest.raises(PretrainedStoreError, match=r"no readable MANIFEST\.sha256"):
        require_pretrained_store(pin, tmp_path / "missing")


def test_a_pin_is_empty_or_a_sha256_hex_digest() -> None:
    assert pretrained_pin(None) == pretrained_pin("") == ""
    assert pretrained_pin(OTHER_PIN) == OTHER_PIN
    for bad in ("abc", OTHER_PIN.upper().replace("0", "A"), 7, OTHER_PIN + "0"):
        with pytest.raises(ValueError, match="pretrained_manifest_sha256"):
            pretrained_pin(bad)
    with pytest.raises(ValueError, match="pretrained_manifest_sha256"):
        SandboxConfig(pretrained_manifest_sha256="abc")


# ----- strategy containers -----


def test_a_pinned_strategy_container_mounts_the_store_read_only(tmp_path: Path) -> None:
    strategy = _strategy(tmp_path)
    with patch.object(DockerStrategyExecutor, "_start"):
        pinned = DockerStrategyExecutor(
            strategy, SandboxConfig(pretrained_manifest_sha256=manifest_digest())
        )
        unpinned = DockerStrategyExecutor(strategy)
    store = pretrained.PRETRAINED_STORE.resolve()
    mount = f"type=bind,src={store},dst={CONTAINER_PRETRAINED_DIR},readonly"
    assert mount in pinned.docker_command()
    assert pinned.context_pretrained_dir == CONTAINER_PRETRAINED_DIR == "/strategy-data/pretrained"
    # An arm or book from before the store: exactly the boundary it had.
    assert not any(CONTAINER_PRETRAINED_DIR in item for item in unpinned.docker_command())
    assert unpinned.context_pretrained_dir == ""
    context = StrategyContext(
        inference_at=datetime(2026, 1, 2, 8, 30, tzinfo=CN_TZ),
        bars=(),
        account=AccountSnapshot(cash=1.0, positions={}),
        pretrained_dir=pinned.context_pretrained_dir,
    )
    assert pinned._context_record(context)["pretrained_dir"] == CONTAINER_PRETRAINED_DIR
    with pytest.raises(StrategyExecutionError, match="pretrained_dir without a mount"):
        unpinned._context_record(context)
    for executor in (pinned, unpinned):
        executor.close()


def test_no_strategy_container_starts_on_a_missing_or_different_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    strategy = _strategy(tmp_path)
    pin = manifest_digest()
    for store, pinned, reason in (
        (pretrained.PRETRAINED_STORE, OTHER_PIN, "not the one"),
        (tmp_path / "gone", pin, "no readable MANIFEST.sha256"),
    ):
        monkeypatch.setattr(pretrained, "PRETRAINED_STORE", store)
        with (
            patch.object(DockerStrategyExecutor, "_start") as start,
            pytest.raises(StrategyExecutionError, match=reason),
        ):
            DockerStrategyExecutor(strategy, SandboxConfig(pretrained_manifest_sha256=pinned))
        start.assert_not_called()


def test_the_validator_roots_model_directories_like_every_other_file() -> None:
    reads = (
        "def generate_orders(context):\n"
        "    from transformers import AutoModelForCausalLM\n"
        "    root = context.pretrained_dir + '/Qwen3.5-0.8B'\n"
        "    AutoModelForCausalLM.from_pretrained(f'{context.pretrained_dir}/Qwen3-0.6B', dtype=None)\n"
        "    AutoModelForCausalLM.from_pretrained(root).save_pretrained(context.state_dir + '/adapter')\n"
        "    return []\n"
    )
    validate_strategy_source(reads)
    for source, reason in (
        # The session's path, or the container's spelled out, is not the context.
        ("AutoModelForCausalLM.from_pretrained('/mnt/pretrained/Qwen3-0.6B')", "absolute path literal"),
        ("AutoModelForCausalLM.from_pretrained('/strategy-data/pretrained/Qwen3-0.6B')", "context.pretrained_dir"),
        ("model.save_pretrained(context.pretrained_dir + '/x')", "mounted read-only"),
        ("torch.save(model, context.pretrained_dir + '/x.pt')", "mounted read-only"),
    ):
        with pytest.raises(StrategyLoadError, match=reason):
            validate_strategy_source(
                "import torch\nfrom transformers import AutoModelForCausalLM\n"
                f"def generate_orders(context):\n    model = None\n    {source}\n    return []\n"
            )


# ----- the session container -----


def test_a_pinned_session_mounts_the_store_and_refuses_another(tmp_path: Path) -> None:
    local = LocalSandbox(tmp_path / "session")
    local.prepare_layout()
    completed = subprocess.CompletedProcess(args=[], returncode=0, stdout="container\n", stderr="")
    started = DockerSandbox(local, SandboxSpec(gpu=None, pretrained_manifest_sha256=manifest_digest()))
    with (
        patch("autotrade.environment.sandbox.probe_image_runtime", return_value={}),
        patch("autotrade.environment.sandbox.subprocess.run", return_value=completed) as run,
    ):
        started.start()
    command = run.call_args.args[0]
    store = pretrained.PRETRAINED_STORE.resolve()
    assert f"type=bind,src={store},dst={PRETRAINED_SESSION_MOUNT},readonly" in command
    assert PRETRAINED_SESSION_MOUNT == "/mnt/pretrained"
    record = json.loads(local.paths.runtime_env.read_text(encoding="utf-8"))
    assert record["sandbox_spec"]["pretrained_manifest_sha256"] == manifest_digest()
    assert not any(PRETRAINED_SESSION_MOUNT in item for item in DockerSandbox(local, SandboxSpec(gpu=None)).docker_command())
    refused = DockerSandbox(local, SandboxSpec(gpu=None, pretrained_manifest_sha256=OTHER_PIN))
    with (
        patch("autotrade.environment.sandbox.probe_image_runtime") as probe,
        patch("autotrade.environment.sandbox.subprocess.run") as run,
        pytest.raises(PretrainedStoreError, match="not the one"),
    ):
        refused.start()
    probe.assert_not_called()
    run.assert_not_called()


# ----- creation, worker options and Paper books -----


def _create(client: TestClient, experiment_id: str, **params: object):
    with stubbed_gpu_probe():
        return client.post(
            "/api/experiments",
            json={"params": {"experiment_id": experiment_id, **DEFAULT_RESEARCH_GEOMETRY.to_record(), **params}},
        )


def test_the_console_pins_the_store_and_a_request_cannot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = TestClient(create_app(tmp_path))
    assert _create(client, "pinned").status_code == 200
    params = json.loads((tmp_path / "experiments/pinned/hitl/params.json").read_text(encoding="utf-8"))
    assert params["pretrained_manifest_sha256"] == manifest_digest()
    refused = _create(client, "chosen", pretrained_manifest_sha256=OTHER_PIN)
    assert refused.status_code == 400
    assert "pretrained_manifest_sha256" in refused.text
    monkeypatch.setattr(pretrained, "PRETRAINED_STORE", tmp_path / "gone")
    missing = _create(client, "storeless")
    assert missing.status_code == 400
    assert "no readable MANIFEST.sha256" in missing.text
    assert not (tmp_path / "experiments/chosen").exists()
    assert not (tmp_path / "experiments/storeless").exists()


def test_the_pin_reaches_the_session_and_every_strategy_container(tmp_path: Path) -> None:
    arm = build_arm(tmp_path / "experiments", "exp", "created")
    params = json.loads((arm / "hitl/params.json").read_text(encoding="utf-8"))

    def spec(**extra: object) -> SandboxSpec:
        options = resolve_worker_options(
            {**params, **extra}, experiment_dir=arm, repo_root=tmp_path, preflight=True
        )
        assert options.agent_sandbox is not None
        return options.agent_sandbox

    pinned = spec(pretrained_manifest_sha256=OTHER_PIN)
    assert pinned.pretrained_manifest_sha256 == OTHER_PIN
    strategy = _strategy_sandbox_from_spec(pinned, fit_timeout_seconds=60)
    assert strategy.pretrained_manifest_sha256 == OTHER_PIN
    assert spec().pretrained_manifest_sha256 == ""
    with pytest.raises(ValueError, match="pretrained_manifest_sha256"):
        spec(pretrained_manifest_sha256="latest")


def test_a_book_keeps_its_arms_pin_and_an_older_book_mounts_nothing(tmp_path: Path) -> None:
    arm = build_arm(tmp_path / "experiments", "exp", "graduated")
    path = arm / "hitl/params.json"
    params = json.loads(path.read_text(encoding="utf-8"))
    path.write_text(json.dumps({**params, "pretrained_manifest_sha256": manifest_digest()}), encoding="utf-8")
    book = create_book(
        paper_root(tmp_path) / "exp",
        experiment_dir=arm,
        artifact_id="strategy_research_abc",
        repo_root=tmp_path,
        track="graduated",
    )
    assert book.sandbox.pretrained_manifest_sha256 == manifest_digest()
    record_path = book.root / "book.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    assert record["sandbox"]["pretrained_manifest_sha256"] == manifest_digest()
    del record["sandbox"]["pretrained_manifest_sha256"]
    record_path.write_text(json.dumps(record), encoding="utf-8")
    assert load_book(book.root).sandbox.pretrained_manifest_sha256 == ""


def test_a_paper_run_of_a_pinned_book_refuses_a_different_store(tmp_path: Path) -> None:
    def engine(pin: str) -> DailyPaperEngine:
        return DailyPaperEngine(
            strategy_path=_strategy(tmp_path),
            strategy_revision="strategy_research_abc",
            state_root=tmp_path / "book",
            data_factory=lambda start, target: pytest.fail("no data before the store check"),
            sandbox=SandboxConfig(pretrained_manifest_sha256=pin),
        )

    engine(manifest_digest())
    engine("")
    with pytest.raises(PretrainedStoreError, match="not the one"):
        engine(OTHER_PIN)
