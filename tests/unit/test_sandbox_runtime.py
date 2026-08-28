from __future__ import annotations

import errno
import json
import os
import stat
import subprocess
import sys
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from autotrade.environment.artifacts import (
    ArtifactError,
    FilesystemArtifactStore,
    copy_artifact,
    copy_model_artifacts,
    restore_working_artifacts_writable,
)
from autotrade.environment.executor import (
    DockerStrategyExecutor,
    ExecResult,
    LocalExecutor,
    PersistentCommandRunner,
    StrategyExecutionError,
    docker_available,
)
from autotrade.environment.replay import DailyMarketData
from autotrade.environment.sandbox import (
    DockerSandbox,
    LocalSandbox,
    SandboxConfig,
    SandboxLimits,
    SandboxSpec,
)
from autotrade.environment.sandbox_images import (
    _gc_owned_sandbox_images,
    maybe_rebuild_sandbox_image,
    prepare_experiment_sandbox_image,
)
from autotrade.environment.strategy import (
    CN_TZ,
    AccountSnapshot,
    StrategyContext,
    StrategySchedule,
    validate_order_payload,
)
from autotrade.environment.strategy_loader import StrategyLoadError
from autotrade.environment.strategy_worker import BarStream, WorkerProtocolError
from autotrade.environment.tools.base import CommandResult
from autotrade.pipelines import DailyStrategyPipeline, StrategyExperimentConfig
from autotrade.pipelines.worker import _activate_experiment_sandbox


def _strategy(tmp_path: Path, source: str = "def generate_orders(context):\n    return []\n") -> Path:
    path = tmp_path / "main.py"
    path.write_text(source, encoding="utf-8")
    return path


def _context() -> StrategyContext:
    return StrategyContext(
        inference_at=datetime(2026, 1, 2, 8, 30, tzinfo=CN_TZ),
        bars=(),
        account=AccountSnapshot(cash=10_000, positions={}),
    )


def _wire_request(
    context: StrategyContext,
    *,
    sequence: int = 0,
    reset: bool = True,
    base_count: int = 0,
    bars: list[dict[str, object]] | None = None,
    total_count: int | None = None,
) -> dict[str, object]:
    record = context.to_record()
    full_bars = record.pop("bars")
    assert isinstance(full_bars, list)
    delta = full_bars if bars is None else bars
    return {
        "type": "execute",
        "sequence": sequence,
        "reset": reset,
        "base_count": base_count,
        "total_count": base_count + len(delta) if total_count is None else total_count,
        "context": record,
        "bars": delta,
    }


def _executor_for_process(
    process: subprocess.Popen[bytes],
    *,
    limits: SandboxLimits | None = None,
    drain_stderr: bool = False,
) -> DockerStrategyExecutor:
    executor = object.__new__(DockerStrategyExecutor)
    executor.config = SandboxConfig(limits=limits or SandboxLimits())
    executor.container_name = "test-container"
    executor._process = process
    executor._stdout_buffer = bytearray()
    executor._stderr_tail = deque()
    executor._stderr_size = 0
    executor._stderr_thread = None
    executor._closed = False
    executor.snapshot_dir = None
    executor.asof_dir = None
    executor._reset_transport_state()
    if drain_stderr:
        thread = threading.Thread(target=executor._drain_stderr, daemon=True)
        executor._stderr_thread = thread
        thread.start()
    return executor


def test_persistent_sandbox_command_has_bounded_explicit_identity_mounts(tmp_path: Path):
    local = LocalSandbox(tmp_path / "session")
    local.prepare_layout()
    sandbox = DockerSandbox(local, SandboxSpec(gpu=None), labels={"adm.experiment": "exp1"})
    command = sandbox.docker_command()
    assert command[:5] == ["docker", "run", "--pull", "never", "--detach"]
    assert "--read-only" in command
    assert command[command.index("--network") + 1] == "none"
    assert command[command.index("--cap-drop") + 1] == "ALL"
    assert command[command.index("--security-opt") + 1] == "no-new-privileges"
    rendered = " ".join(command)
    assert "/var/run/docker.sock" not in rendered
    assert ".git" not in rendered
    assert "adm.experiment=exp1" in command


def test_local_sandbox_runtime_contract_has_no_git_and_no_image_record(tmp_path: Path):
    local = LocalSandbox(tmp_path / "session")
    paths = local.prepare_layout()
    payload = json.loads(paths.runtime_env.read_text(encoding="utf-8"))
    assert payload["policy"]["external_source_control"] == "disabled"
    assert payload["image"] == {}
    assert not list(paths.artifacts.glob("*.jsonl"))
    assert {path.name for path in paths.artifacts.iterdir()} == {
        "parent_models",
        "parent_output",
        "runtime_env.json",
        "steps",
    }


def test_local_executor_bounds_streamed_output_without_communicate_buffer(tmp_path: Path):
    paths = LocalSandbox(tmp_path / "session").prepare_layout()
    result = LocalExecutor(paths).run(
        [sys.executable, "-c", "import sys; print('x' * 10000); print('y' * 10000, file=sys.stderr)"],
        max_output_chars=64,
    )
    assert result.exit_code == 0
    assert len(result.stdout) == 64 and result.stdout_truncated
    assert len(result.stderr) == 64 and result.stderr_truncated


def test_persistent_sandbox_start_records_tag_session_and_runtime_probe(tmp_path: Path):
    local = LocalSandbox(tmp_path / "session")
    paths = local.prepare_layout()
    generation_id = "123e4567-e89b-42d3-a456-426614174000"
    sandbox = DockerSandbox(
        local,
        SandboxSpec(
            image="autotrade-sandbox:exp-base-123e4567-e89b-42d3-a456-426614174000",
            build_generation_id=generation_id,
            gpu=None,
        ),
    )
    completed = subprocess.CompletedProcess(args=[], returncode=0, stdout="container\n", stderr="")
    with (
        patch(
            "autotrade.environment.sandbox.probe_image_runtime",
            return_value={"python_version": "3.11.13"},
        ),
        patch("autotrade.environment.sandbox.subprocess.run", return_value=completed),
    ):
        sandbox.start()
    payload = json.loads(paths.runtime_env.read_text(encoding="utf-8"))
    assert payload["image"] == {
        "image_ref": sandbox.spec.image,
        "build_generation_id": generation_id,
        "session_id": sandbox.session_id,
        "runtime": {"python_version": "3.11.13"},
    }
    allocation = sandbox.allocation_record()
    assert allocation["session_id"] == sandbox.session_id
    assert allocation["image_ref"] == sandbox.spec.image
    assert allocation["build_generation_id"] == generation_id
    assert "image_id" not in allocation
    assert "image_repo_digests" not in allocation


def test_persistent_command_timeout_keeps_sandbox_for_followup_work():
    sandbox = Mock()
    sandbox.exec_limited.return_value = ExecResult(124, "partial", "timed out")
    result = PersistentCommandRunner(sandbox).run(
        ["python", "-c", "pass"],
        cwd=".",
        timeout_seconds=1,
        max_output_chars=100,
    )
    assert result == CommandResult(124, "partial", "timed out", timed_out=True)
    sandbox.stop.assert_not_called()


@pytest.mark.skipif(not docker_available(), reason="Docker is unavailable")
def test_real_persistent_timeout_kills_command_only_and_preserves_session(tmp_path: Path):
    image = "autotrade-sandbox:latest"
    inspected = subprocess.run(
        ["docker", "image", "inspect", image],
        capture_output=True,
        check=False,
    )
    if inspected.returncode != 0:
        pytest.skip(f"local sandbox image is unavailable: {image}")
    local = LocalSandbox(tmp_path / "session")
    paths = local.prepare_layout()
    sandbox = DockerSandbox(
        local,
        SandboxSpec(
            image=image,
            gpu=None,
            cpus=1,
            memory="1g",
            pids_limit=64,
            tmpfs_size="64m",
        ),
    )
    sandbox.start()
    runner = PersistentCommandRunner(sandbox)
    try:
        marker = "adm_persistent_timeout_child"
        timed_out = runner.run(
            [
                "sh",
                "-c",
                (
                    "printf retained > timeout-proof.txt; "
                    f"python -c 'import time; time.sleep(60)' {marker} & "
                    "echo $! > timeout-child.pid; wait"
                ),
            ],
            cwd=".",
            timeout_seconds=0.2,
            max_output_chars=4_000,
        )
        assert timed_out.exit_code == 124
        assert timed_out.timed_out is True
        assert (paths.workspace / "timeout-proof.txt").read_text(encoding="utf-8") == "retained"

        child_pid = (paths.workspace / "timeout-child.pid").read_text(encoding="ascii").strip()
        no_residual = runner.run(
            [
                "sh",
                "-c",
                (
                    f"if [ -e /proc/{child_pid}/cmdline ] && "
                    f"tr '\\0' ' ' < /proc/{child_pid}/cmdline | grep -Fq {marker}; "
                    "then exit 9; fi"
                ),
            ],
            cwd=".",
            timeout_seconds=5,
            max_output_chars=4_000,
        )
        assert no_residual.exit_code == 0
        assert no_residual.timed_out is False

        followup = runner.run(
            ["sh", "-c", "cat timeout-proof.txt; printf '\\nready\\n'"],
            cwd=".",
            timeout_seconds=5,
            max_output_chars=4_000,
        )
        assert followup.exit_code == 0
        assert followup.stdout == "retained\nready\n"
        with sandbox.formal_guard():
            assert (paths.workspace / "timeout-proof.txt").is_file()
        assert sandbox.exec(["true"], timeout_seconds=5).returncode == 0
    finally:
        sandbox.stop()


def test_experiment_image_prepare_is_atomic_and_resume_uses_the_same_uuid4_tag(
    tmp_path: Path,
):
    experiment_dir = tmp_path / "experiment"
    completed = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")

    def docker(command, **_kwargs):
        if command[1:3] == ["image", "inspect"]:
            return subprocess.CompletedProcess(command, 1, "", "missing")
        return completed

    results: list[SandboxSpec] = []
    errors: list[BaseException] = []

    def prepare() -> None:
        try:
            results.append(
                prepare_experiment_sandbox_image(
                    SandboxSpec(image="autotrade-sandbox:latest", gpu=None),
                    experiment_id="experiment_001",
                    experiment_dir=experiment_dir,
                )
            )
        except BaseException as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    with (
        patch(
            "autotrade.environment.sandbox_images.subprocess.run",
            side_effect=docker,
        ) as run,
        patch(
            "autotrade.environment.sandbox_images.probe_image_runtime",
            return_value={"python_version": "3.11.13"},
        ) as probe,
    ):
        threads = [threading.Thread(target=prepare) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5)
        assert all(not thread.is_alive() for thread in threads)
        resumed = prepare_experiment_sandbox_image(
            SandboxSpec(image="operator-retargeted:ignored", gpu=None),
            experiment_id="experiment_001",
            experiment_dir=experiment_dir,
        )

    assert errors == []
    assert len(results) == 2
    assert results[0].image == results[1].image == resumed.image
    assert results[0].build_generation_id == results[1].build_generation_id
    assert results[0].image != "autotrade-sandbox:latest"
    assert len([call for call in run.call_args_list if call.args[0][1] == "tag"]) == 1
    probe.assert_called_once()
    state = json.loads(
        (experiment_dir / "hitl/sandbox_image.json").read_text(encoding="utf-8")
    )
    assert state["image_ref"] == resumed.image
    assert state["build_generation_id"] == resumed.build_generation_id
    assert state["owned_image_refs"] == [resumed.image]
    assert state["kind"] == "base_clone"
    assert "operator-retargeted:ignored" not in json.dumps(state)
    assert "image_id" not in state and "image_repo_digests" not in state


def test_experiment_image_smoke_failure_does_not_publish_state(tmp_path: Path):
    experiment_dir = tmp_path / "experiment"

    def docker(command, **_kwargs):
        return subprocess.CompletedProcess(
            command,
            1 if command[1:3] == ["image", "inspect"] else 0,
            "",
            "",
        )

    with (
        patch(
            "autotrade.environment.sandbox_images.subprocess.run",
            side_effect=docker,
        ) as run,
        patch(
            "autotrade.environment.sandbox_images.probe_image_runtime",
            side_effect=RuntimeError("offline smoke failed"),
        ),
        pytest.raises(RuntimeError, match="offline smoke failed"),
    ):
        prepare_experiment_sandbox_image(
            SandboxSpec(gpu=None),
            experiment_id="experiment_001",
            experiment_dir=experiment_dir,
        )
    assert not (experiment_dir / "hitl/sandbox_image.json").exists()
    assert any(call.args[0][1:3] == ["image", "rm"] for call in run.call_args_list)


def test_derived_sandbox_image_records_tag_and_build_uuid(tmp_path: Path):
    request_path = tmp_path / "sandbox_environment.json"
    request_path.write_text('{"python_packages":["numpy==2.4.2"]}\n', encoding="utf-8")
    completed = subprocess.CompletedProcess(args=[], returncode=0, stdout="built\n", stderr="")
    manifest = Mock()
    with (
        patch.dict(
            os.environ,
            {
                "HTTP_PROXY": "http://127.0.0.1:54202",
                "HTTPS_PROXY": "http://127.0.0.1:54202",
                "NO_PROXY": "127.0.0.1,localhost",
            },
            clear=True,
        ),
        patch(
            "autotrade.environment.sandbox_images.subprocess.run",
            side_effect=lambda command, **_: (
                subprocess.CompletedProcess(command, 1, "", "missing")
                if command[1:3] == ["image", "inspect"]
                else completed
            ),
        ) as run,
        patch(
            "autotrade.environment.sandbox_images.probe_image_runtime",
            return_value={"python_version": "3.11.13"},
        ),
    ):
        result, spec = maybe_rebuild_sandbox_image(
            request_path,
            base_spec=SandboxSpec(image="autotrade-sandbox:latest"),
            experiment_id="experiment_001",
            epoch_id="epoch_001",
            experiment_dir=tmp_path / "experiment",
            manifest=manifest,
            use_docker=True,
            rebuild_enabled=True,
            timeout_seconds=120,
        )
    assert result is not None
    build_commands = [
        call.args[0]
        for call in run.call_args_list
        if len(call.args[0]) >= 2 and call.args[0][1] == "build"
    ]
    assert len(build_commands) == 1
    command = build_commands[0]
    assert command[:3] == ["docker", "build", "--network=host"]
    assert command.count("--build-arg") == 3
    assert "HTTP_PROXY" in command
    assert "HTTPS_PROXY" in command
    assert "NO_PROXY" in command
    assert all("http://" not in item for item in command)
    dockerfile = Path(str(result["dockerfile_ref"])).read_text(encoding="utf-8")
    assert "PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple" in dockerfile
    assert "NPM_CONFIG_REGISTRY=https://registry.npmmirror.com" in dockerfile
    assert "DEBIAN_MIRROR=https://mirrors.tuna.tsinghua.edu.cn/debian" in dockerfile
    assert str(result["build_generation_id"]) in str(result["image_ref"])
    assert result["runtime"] == {"python_version": "3.11.13"}
    assert spec.image == result["image_ref"]
    assert spec.build_generation_id == result["build_generation_id"]
    persisted = json.loads(
        (tmp_path / "experiment/hitl/sandbox_image.json").read_text(encoding="utf-8")
    )
    assert persisted["image_ref"] == result["image_ref"]
    assert persisted["owned_image_refs"] == [result["image_ref"]]
    assert "image_id" not in persisted
    assert "image_repo_digests" not in persisted
    manifest.update.assert_called_once_with(sandbox_image_update=result)


def test_derived_image_build_failure_keeps_active_state_unpublished(tmp_path: Path):
    request_path = tmp_path / "sandbox_environment.json"
    request_path.write_text('{"python_packages":["numpy==2.4.2"]}\n', encoding="utf-8")
    manifest = Mock()
    state_path = tmp_path / "experiment/hitl/sandbox_image.json"
    state_path.parent.mkdir(parents=True)
    previous_state = {
        "schema_version": 1,
        "experiment_id": "experiment_001",
        "image_ref": "autotrade-sandbox:experiment_001-base-existing",
        "build_generation_id": "123e4567-e89b-42d3-a456-426614174000",
        "base_image_ref": "autotrade-sandbox:latest",
        "kind": "base_clone",
    }
    state_path.write_text(json.dumps(previous_state), encoding="utf-8")

    def docker(command, **_kwargs):
        if command[1:3] == ["image", "inspect"]:
            return subprocess.CompletedProcess(command, 1, "", "missing")
        if command[1] == "build":
            return subprocess.CompletedProcess(command, 17, "", "private failure")
        return subprocess.CompletedProcess(command, 0, "", "")

    with (
        patch.dict(
            os.environ,
            {"HTTPS_PROXY": "http://operator:secret@127.0.0.1:54202"},
            clear=True,
        ),
        patch(
            "autotrade.environment.sandbox_images.subprocess.run", side_effect=docker
        ) as run,
        patch(
            "autotrade.environment.sandbox_images.probe_image_runtime"
        ) as probe,
        pytest.raises(RuntimeError, match="rebuild failed"),
    ):
        maybe_rebuild_sandbox_image(
            request_path,
            base_spec=SandboxSpec(
                image=str(previous_state["image_ref"]),
                build_generation_id=str(previous_state["build_generation_id"]),
                gpu=None,
            ),
            experiment_id="experiment_001",
            epoch_id="epoch_001",
            experiment_dir=tmp_path / "experiment",
            manifest=manifest,
            use_docker=True,
            rebuild_enabled=True,
            timeout_seconds=120,
        )
    probe.assert_not_called()
    assert json.loads(state_path.read_text(encoding="utf-8")) == previous_state
    result = manifest.update.call_args.kwargs["sandbox_image_update"]
    assert result["status"] == "failed"
    assert "secret" not in json.dumps(result)
    build = next(call.args[0] for call in run.call_args_list if call.args[0][1] == "build")
    assert "HTTPS_PROXY" in build
    assert all("secret" not in part for part in build)


def test_derived_image_smoke_failure_removes_tag_without_publishing(tmp_path: Path):
    request_path = tmp_path / "sandbox_environment.json"
    request_path.write_text('{"python_packages":["numpy==2.4.2"]}\n', encoding="utf-8")
    manifest = Mock()

    def docker(command, **_kwargs):
        return subprocess.CompletedProcess(
            command,
            1 if command[1:3] == ["image", "inspect"] else 0,
            "",
            "",
        )

    with (
        patch(
            "autotrade.environment.sandbox_images.subprocess.run", side_effect=docker
        ) as run,
        patch(
            "autotrade.environment.sandbox_images.probe_image_runtime",
            side_effect=RuntimeError("probe broke"),
        ),
        pytest.raises(RuntimeError, match="smoke failed"),
    ):
        maybe_rebuild_sandbox_image(
            request_path,
            base_spec=SandboxSpec(gpu=None),
            experiment_id="experiment_001",
            epoch_id="epoch_001",
            experiment_dir=tmp_path / "experiment",
            manifest=manifest,
            use_docker=True,
            rebuild_enabled=True,
            timeout_seconds=120,
        )
    assert not (tmp_path / "experiment/hitl/sandbox_image.json").exists()
    assert any(call.args[0][1:3] == ["image", "rm"] for call in run.call_args_list)
    result = manifest.update.call_args.kwargs["sandbox_image_update"]
    assert result["status"] == "smoke_failed"
    assert result["reason"] == "RuntimeError: sandbox runtime smoke failed"


def test_concurrent_derived_builds_use_distinct_immutable_tags(tmp_path: Path):
    request_path = tmp_path / "sandbox_environment.json"
    request_path.write_text('{"python_packages":["numpy==2.4.2"]}\n', encoding="utf-8")
    experiment_dir = tmp_path / "experiment"
    results: list[dict[str, object]] = []
    errors: list[BaseException] = []

    def docker(command, **_kwargs):
        if command[1:3] == ["image", "inspect"]:
            return subprocess.CompletedProcess(command, 1, "", "missing")
        return subprocess.CompletedProcess(command, 0, "", "")

    def build(epoch_id: str) -> None:
        try:
            result, _spec = maybe_rebuild_sandbox_image(
                request_path,
                base_spec=SandboxSpec(gpu=None),
                experiment_id="experiment_001",
                epoch_id=epoch_id,
                experiment_dir=experiment_dir,
                manifest=Mock(),
                use_docker=True,
                rebuild_enabled=True,
                timeout_seconds=120,
            )
            assert result is not None
            results.append(result)
        except BaseException as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    with (
        patch(
            "autotrade.environment.sandbox_images.subprocess.run", side_effect=docker
        ) as run,
        patch(
            "autotrade.environment.sandbox_images.probe_image_runtime",
            return_value={"python_version": "3.11.13"},
        ),
    ):
        threads = [
            threading.Thread(target=build, args=(f"epoch_{index}",))
            for index in range(2)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5)
        assert all(not thread.is_alive() for thread in threads)

    assert errors == []
    assert len(results) == 2
    refs = {str(result["image_ref"]) for result in results}
    assert len(refs) == 2
    build_commands = [
        call.args[0] for call in run.call_args_list if call.args[0][1] == "build"
    ]
    assert len(build_commands) == 2
    built_refs = [command[command.index("--tag") + 1] for command in build_commands]
    assert len(set(built_refs)) == 2
    second_result = next(
        result for result in results if result["image_ref"] == built_refs[1]
    )
    second_dockerfile = Path(str(second_result["dockerfile_ref"])).read_text(
        encoding="utf-8"
    )
    assert f"\nFROM {built_refs[0]}\n" in second_dockerfile
    state = json.loads(
        (experiment_dir / "hitl/sandbox_image.json").read_text(encoding="utf-8")
    )
    assert state["image_ref"] == built_refs[1]
    assert state["owned_image_refs"] == built_refs


def test_image_gc_removes_only_exact_persisted_ownership_refs() -> None:
    owned = [
        "autotrade-sandbox:expa-base-11111111-1111-4111-8111-111111111111",
        "autotrade-sandbox:expa-epoch-22222222-2222-4222-8222-222222222222",
    ]
    other_experiment = (
        "autotrade-sandbox:expa-base-33333333-3333-4333-8333-333333333333"
    )
    completed = subprocess.CompletedProcess([], 0, "", "")
    with patch(
        "autotrade.environment.sandbox_images.subprocess.run", return_value=completed
    ) as run:
        pruned, retained = _gc_owned_sandbox_images(
            owned,
            keep=1,
            keep_image=owned[-1],
        )
    assert pruned == [owned[0]]
    assert retained == [owned[-1]]
    removed_refs = [call.args[0][-1] for call in run.call_args_list]
    assert removed_refs == [owned[0]]
    assert other_experiment not in removed_refs


def test_active_sandbox_update_reaches_agent_and_formal_evaluator() -> None:
    class Developer:
        sandbox_spec: SandboxSpec | None = None

        def set_sandbox_spec(self, spec: SandboxSpec) -> None:
            self.sandbox_spec = spec

    class Evaluator:
        sandbox = SandboxConfig(
            image="autotrade-sandbox:experiment-base",
            docker_executable="docker-old",
        )

    developer = Developer()
    evaluator = Evaluator()
    active = SandboxSpec(
        image="autotrade-sandbox:experiment-derived",
        build_generation_id="123e4567-e89b-42d3-a456-426614174000",
        docker_executable="docker-new",
        gpu=None,
    )
    _activate_experiment_sandbox(
        active,
        developer=developer,  # type: ignore[arg-type]
        evaluator=evaluator,  # type: ignore[arg-type]
    )
    assert developer.sandbox_spec is active
    assert evaluator.sandbox.image == active.image
    assert evaluator.sandbox.docker_executable == active.docker_executable


def test_filesystem_artifact_store_freezes_explicit_revision_identity(tmp_path: Path):
    output = tmp_path / "output"
    output.mkdir()
    (output / "main.py").write_text("def generate_orders(context):\n    return []\n", encoding="utf-8")
    store = FilesystemArtifactStore(tmp_path / "store")
    revision = store.create_revision(output, revision_id="revision_001")
    frozen = store.freeze_revision(
        revision.revision_id,
        artifact_id="artifact_001",
        experiment_id="experiment_001",
        epoch_id="epoch_001",
        fold_id="fold_001",
        run_id="run_001",
        step_id="step_001",
    )
    assert frozen.source_step_id == "step_001"
    metadata = json.loads((tmp_path / "store/frozen/artifact_001/revision.json").read_text(encoding="utf-8"))
    assert metadata["revision_id"] == "revision_001"
    assert metadata["artifact_id"] == "artifact_001"


def test_frozen_artifacts_copy_to_writable_fold_workspace_without_mutating_source(tmp_path: Path):
    source = tmp_path / "source"
    models = tmp_path / "source-models"
    source.mkdir()
    models.mkdir()
    strategy_text = "def generate_orders(context):\n    return []\n"
    (source / "main.py").write_text(strategy_text, encoding="utf-8")
    (source / "README.md").write_text("contract\n", encoding="utf-8")
    (models / "weights.bin").write_bytes(b"parent-model")
    store = FilesystemArtifactStore(tmp_path / "store")
    revision = store.create_revision(source, models_path=models, revision_id="revision_parent")
    frozen = store.freeze_revision(
        revision.revision_id,
        artifact_id="artifact_parent",
        experiment_id="experiment_001",
        epoch_id="epoch_001",
        fold_id="fold_001",
        run_id="run_001",
        step_id="step_001",
    )

    work = tmp_path / "workspace"
    output = work / "output"
    work_models = work / "models"
    copy_artifact(frozen.path, output)
    copy_model_artifacts(frozen.model_path, work_models)
    restore_working_artifacts_writable(output, work_models)

    assert stat.S_IMODE(output.stat().st_mode) == 0o777
    assert stat.S_IMODE((output / "main.py").stat().st_mode) == 0o666
    assert stat.S_IMODE((output / "README.md").stat().st_mode) == 0o444
    assert stat.S_IMODE(work_models.stat().st_mode) == 0o777
    assert stat.S_IMODE((work_models / "weights.bin").stat().st_mode) == 0o666
    assert stat.S_IMODE(Path(frozen.path).stat().st_mode) == 0o555
    assert stat.S_IMODE((Path(frozen.path) / "main.py").stat().st_mode) == 0o444
    assert (Path(frozen.path) / "main.py").read_text(encoding="utf-8") == strategy_text
    assert (Path(frozen.model_path) / "weights.bin").read_bytes() == b"parent-model"


def test_working_artifact_permission_restore_fails_closed(tmp_path: Path):
    output = tmp_path / "output"
    models = tmp_path / "models"
    output.mkdir()
    models.mkdir()
    (output / "main.py").write_text("def generate_orders(context):\n    return []\n", encoding="utf-8")
    with (
        patch("autotrade.environment.artifacts.chmod_tree"),
        pytest.raises(ArtifactError, match="unsafe permissions"),
    ):
        restore_working_artifacts_writable(output, models)


@pytest.mark.skipif(not docker_available(), reason="Docker is unavailable")
def test_real_inherited_fold_workspace_is_writable_only_inside_agent_boundary(tmp_path: Path):
    image = "autotrade-sandbox:latest"
    inspected = subprocess.run(
        ["docker", "image", "inspect", image],
        capture_output=True,
        check=False,
    )
    if inspected.returncode != 0:
        pytest.skip(f"local sandbox image is unavailable: {image}")

    source = tmp_path / "source"
    source_models = tmp_path / "source-models"
    source.mkdir()
    source_models.mkdir()
    original_strategy = "def generate_orders(context):\n    return []\n"
    (source / "main.py").write_text(original_strategy, encoding="utf-8")
    (source / "README.md").write_text("read only\n", encoding="utf-8")
    (source_models / "weights.bin").write_bytes(b"parent")
    store = FilesystemArtifactStore(tmp_path / "store")
    revision = store.create_revision(source, models_path=source_models, revision_id="revision_parent")
    frozen = store.freeze_revision(
        revision.revision_id,
        artifact_id="artifact_parent",
        experiment_id="experiment_001",
        epoch_id="epoch_001",
        fold_id="fold_001",
        run_id="run_001",
        step_id="step_001",
    )

    local = LocalSandbox(tmp_path / "session")
    paths = local.prepare_layout()
    output = paths.workspace / "output"
    models = paths.workspace / "models"
    copy_artifact(frozen.path, output)
    copy_model_artifacts(frozen.model_path, models)
    restore_working_artifacts_writable(output, models)
    sandbox = DockerSandbox(
        local,
        SandboxSpec(
            image=image,
            gpu=None,
            cpus=1,
            memory="1g",
            pids_limit=64,
            tmpfs_size="64m",
        ),
    )
    sandbox.start()
    try:
        result = sandbox.exec(
            [
                "python",
                "-c",
                (
                    "import os; from pathlib import Path; "
                    "assert os.geteuid() == 61000; "
                    "Path('output/main.py').write_text('def generate_orders(context):\\n    return [1]\\n'); "
                    "Path('models/weights.bin').write_bytes(b'child'); "
                    "denied = 0; "
                    "\nfor path in (Path('output/README.md'), Path('../escape.txt'), Path('/mnt/artifacts/runtime_env.json')):\n"
                    "    try: path.write_text('forbidden')\n"
                    "    except OSError: denied += 1\n"
                    "assert denied == 3"
                ),
            ],
            timeout_seconds=15,
        )
        assert result.returncode == 0, result.stderr
    finally:
        sandbox.stop()

    assert "return [1]" in (output / "main.py").read_text(encoding="utf-8")
    assert (models / "weights.bin").read_bytes() == b"child"
    assert (Path(frozen.path) / "main.py").read_text(encoding="utf-8") == original_strategy
    assert (Path(frozen.path) / "README.md").read_text(encoding="utf-8") == "read only\n"
    assert (Path(frozen.model_path) / "weights.bin").read_bytes() == b"parent"


def test_docker_command_has_fail_closed_boundary(tmp_path: Path):
    (tmp_path / ".env").write_text("SECRET=do-not-mount\n", encoding="utf-8")
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "private.csv").write_text("private\n", encoding="utf-8")
    strategy = _strategy(tmp_path)
    with patch.object(DockerStrategyExecutor, "_start"):
        executor = DockerStrategyExecutor(strategy)
    command = executor.docker_command()
    assert command[:5] == ["docker", "run", "--pull", "never", "--rm"]
    for pair in (
        ["--network", "none"],
        ["--user", "61000:61000"],
        ["--cap-drop", "ALL"],
        ["--security-opt", "no-new-privileges"],
    ):
        offset = command.index(pair[0])
        assert command[offset : offset + 2] == pair
    assert "--read-only" in command
    assert "--tmpfs" in command
    assert "--cpus" in command
    assert "--memory" in command
    assert "--pids-limit" in command
    assert command[command.index("--cpus") + 1] == "8"
    assert command[command.index("--memory") + 1] == "16g"
    assert command[command.index("--pids-limit") + 1] == "64"
    env_pairs = [
        command[index + 1]
        for index, value in enumerate(command)
        if value == "--env"
    ]
    assert env_pairs == [
        "MKL_NUM_THREADS=8",
        "NUMEXPR_NUM_THREADS=8",
        "OMP_NUM_THREADS=8",
        "OPENBLAS_NUM_THREADS=8",
    ]
    mount = command[command.index("--mount") + 1]
    assert mount == f"type=bind,src={strategy},dst=/strategy/main.py,readonly"
    assert f"src={tmp_path}," not in mount
    assert str(tmp_path / ".env") not in command
    assert str(tmp_path / "data") not in command
    assert command[-1] == "/strategy/main.py"
    executor.close()


@pytest.mark.parametrize(("cpus", "expected"), [(0.25, "1"), (1.5, "2"), (32.0, "8")])
def test_strategy_thread_limit_tracks_fractional_cpus_and_stays_bounded(
    tmp_path: Path,
    cpus: float,
    expected: str,
):
    strategy = _strategy(tmp_path)
    config = SandboxConfig(limits=SandboxLimits(cpus=cpus))
    with patch.object(DockerStrategyExecutor, "_start"):
        executor = DockerStrategyExecutor(strategy, config)
    command = executor.docker_command()
    assert {
        command[index + 1]
        for index, value in enumerate(command)
        if value == "--env"
    } == {
        f"MKL_NUM_THREADS={expected}",
        f"NUMEXPR_NUM_THREADS={expected}",
        f"OMP_NUM_THREADS={expected}",
        f"OPENBLAS_NUM_THREADS={expected}",
    }
    executor.close()


def test_docker_missing_fails_without_trusted_fallback(tmp_path: Path):
    strategy = _strategy(tmp_path)
    with (
        patch("autotrade.environment.executor.shutil.which", return_value=None),
        pytest.raises(StrategyExecutionError, match="Docker executable is unavailable"),
    ):
        DockerStrategyExecutor(strategy)


def test_sandbox_strategy_contract_rejects_sibling_import(tmp_path: Path):
    (tmp_path / "helper.py").write_text("VALUE = 1\n", encoding="utf-8")
    strategy = _strategy(
        tmp_path,
        "import helper\ndef generate_orders(context):\n    return []\n",
    )
    with (
        patch.object(DockerStrategyExecutor, "_start"),
        pytest.raises(StrategyLoadError, match="unsupported module: helper"),
    ):
        DockerStrategyExecutor(strategy)


def test_strategy_image_inspect_failure_never_starts_container(tmp_path: Path):
    strategy = _strategy(tmp_path)
    inspected = subprocess.CompletedProcess([], 1, stderr=b"image missing")
    with (
        patch("autotrade.environment.executor.shutil.which", return_value="/usr/bin/docker"),
        patch("autotrade.environment.executor.subprocess.run", return_value=inspected) as run,
        patch("autotrade.environment.executor.subprocess.Popen") as popen,
        pytest.raises(StrategyExecutionError, match="unavailable locally"),
    ):
        DockerStrategyExecutor(strategy)
    run.assert_called_once()
    assert run.call_args.args[0] == [
        "/usr/bin/docker",
        "image",
        "inspect",
        "autotrade-sandbox:latest",
    ]
    popen.assert_not_called()


def test_persistent_sandbox_run_args_forbid_pull_network_and_privilege_escalation(tmp_path: Path):
    """The container hardening the deleted one-shot runner used to assert now
    belongs to the persistent sandbox, which is the live Fold execution path."""
    local = LocalSandbox(tmp_path / "session")
    local.prepare_layout()
    sandbox = DockerSandbox(local, SandboxSpec(gpu=None))
    completed = subprocess.CompletedProcess(args=[], returncode=0, stdout="container\n", stderr="")
    with (
        patch("autotrade.environment.sandbox.probe_image_runtime", return_value={}),
        patch("autotrade.environment.sandbox.subprocess.run", return_value=completed) as run,
    ):
        sandbox.start()
    runs = [call.args[0] for call in run.call_args_list]
    command = next(argv for argv in runs if argv[1] == "run")
    assert command[:3] == ["docker", "run", "--pull"]
    assert command[3] == "never"  # a local image only: never reach a registry
    assert command[command.index("--network") : command.index("--network") + 2] == ["--network", "none"]
    assert command[command.index("--security-opt") + 1] == "no-new-privileges"
    assert command[command.index("--cap-drop") + 1] == "ALL"
    assert "/var/run/docker.sock" not in " ".join(command)


def test_persistent_command_runner_rejects_an_empty_or_blank_argv():
    sandbox = Mock()
    runner = PersistentCommandRunner(sandbox)
    for argv in ([], [""], ["python", ""]):
        with pytest.raises(ValueError, match="non-empty"):
            runner.run(argv, cwd=".", timeout_seconds=1, max_output_chars=100)
    sandbox.exec_limited.assert_not_called()


def test_worker_nl_protocol_and_host_order_revalidation(tmp_path: Path):
    strategy = _strategy(
        tmp_path,
        """def generate_orders(context):
    print("strategy output goes to stderr")
    answer = context.nl(question="signal")
    return [{
        "symbol": answer["symbol"],
        "action": "buy",
        "quantity": 100,
        "execute_at": "2026-01-02T09:30:00+08:00",
    }]
""",
    )
    process = subprocess.Popen(
        [sys.executable, "-m", "autotrade.environment.strategy_worker", str(strategy)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert process.stdin is not None
    assert process.stdout is not None
    process.stdin.write(json.dumps(_wire_request(_context())) + "\n")
    process.stdin.flush()
    request = json.loads(process.stdout.readline())
    assert request == {
        "type": "nl_request",
        "sequence": 0,
        "request": {"question": "signal"},
    }
    process.stdin.write(
        json.dumps(
            {
                "type": "nl_response",
                "sequence": 0,
                "result": {"symbol": "000001.SZ"},
            }
        )
        + "\n"
    )
    process.stdin.flush()
    response = json.loads(process.stdout.readline())
    orders = validate_order_payload(response["orders"], inference_at=_context().inference_at)
    assert orders[0].symbol == "000001.SZ"
    process.stdin.write('{"type":"close"}\n')
    process.stdin.flush()
    assert process.wait(timeout=5) == 0
    assert "strategy output goes to stderr" in process.stderr.read()  # type: ignore[union-attr]


def test_worker_protocol_isolated_from_raw_fd_one_writes(tmp_path: Path):
    strategy = _strategy(
        tmp_path,
        '''def generate_orders(context):
    importer = __builtins__["__import__"]
    importer("os").write(1, b"native-noise-without-newline")
    return []
''',
    )
    process = subprocess.Popen(
        [sys.executable, "-m", "autotrade.environment.strategy_worker", str(strategy)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert process.stdin is not None
    assert process.stdout is not None
    process.stdin.write(json.dumps(_wire_request(_context())) + "\n")
    process.stdin.flush()
    assert json.loads(process.stdout.readline()) == {"type": "orders", "sequence": 0, "orders": []}
    process.stdin.write('{"type":"close"}\n')
    process.stdin.flush()
    assert process.wait(timeout=5) == 0
    assert "native-noise-without-newline" in process.stderr.read()  # type: ignore[union-attr]


def test_incremental_worker_receives_only_delta_and_rebuilds_full_history(tmp_path: Path):
    strategy = _strategy(
        tmp_path,
        '''def generate_orders(context):
    latest = context.latest("000001.SZ")
    return [{
        "symbol": "000001.SZ",
        "action": "buy",
        "quantity": 100,
        "execute_at": context.inference_at.isoformat(),
        "seen_count": len(context.bars),
        "history_count": len(context.history("000001.SZ")),
        "latest_close": latest["close"],
    }]
''',
    )
    process = subprocess.Popen(
        [sys.executable, "-m", "autotrade.environment.strategy_worker", str(strategy)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    executor = _executor_for_process(process, drain_stderr=True)
    market = DailyMarketData(
        pd.DataFrame(
            [
                {
                    "trade_date": "20260102",
                    "symbol": "000001.SZ",
                    "open": 10.0,
                    "close": 11.0,
                },
                {
                    "trade_date": "20260105",
                    "symbol": "000001.SZ",
                    "open": 12.0,
                    "close": 13.0,
                },
            ]
        )
    )
    sent: list[dict[str, object]] = []
    original_send = executor._send

    def capture(message, deadline):
        if message.get("type") == "execute":
            sent.append(message)
        original_send(message, deadline)

    try:
        with patch.object(executor, "_send", side_effect=capture):
            first_at = datetime(2026, 1, 2, 18, 0, tzinfo=CN_TZ)
            first = executor.execute(
                StrategyContext(
                    inference_at=first_at,
                    bars=market.visible_at(first_at),
                    account=AccountSnapshot(cash=10_000, positions={}),
                )
            )
            second_at = datetime(2026, 1, 5, 18, 0, tzinfo=CN_TZ)
            second = executor.execute(
                StrategyContext(
                    inference_at=second_at,
                    bars=market.visible_at(second_at),
                    account=AccountSnapshot(cash=10_000, positions={}),
                )
            )
        assert first[0]["seen_count"] == first[0]["history_count"] == 1
        assert second[0]["seen_count"] == second[0]["history_count"] == 2
        assert second[0]["latest_close"] == 13.0
        assert [(item["reset"], item["base_count"], item["total_count"]) for item in sent] == [
            (True, 0, 1),
            (False, 1, 2),
        ]
        assert [len(item["bars"]) for item in sent] == [1, 1]
        assert "20260102" not in json.dumps(sent[1])
        assert len(json.dumps(sent[1])) <= len(json.dumps(sent[0])) + 32
    finally:
        executor.close()
    assert executor._transport_sequence == -1
    assert executor._transport_bars == ()


def test_worker_bar_stream_rejects_inconsistent_updates_without_mutation():
    class ProtocolStub:
        @staticmethod
        def nl_query(_request, **_kwargs):
            return {}

    first_at = datetime(2026, 1, 2, 18, 0, tzinfo=CN_TZ)
    first_context = StrategyContext(
        inference_at=first_at,
        bars=(
            {
                "trade_date": "20260102",
                "symbol": "000001.SZ",
                "close": 11.0,
                "available_at": "2026-01-02T17:30:00+08:00",
            },
        ),
        account=AccountSnapshot(cash=10_000, positions={}),
    )
    later_context = StrategyContext(
        inference_at=datetime(2026, 1, 5, 18, 0, tzinfo=CN_TZ),
        bars=tuple(dict(bar) for bar in first_context.bars),
        account=first_context.account,
    )
    stream = BarStream()
    protocol = ProtocolStub()
    assert len(stream.context(_wire_request(first_context), protocol).bars) == 1  # type: ignore[arg-type]
    valid = _wire_request(
        later_context,
        sequence=1,
        reset=False,
        base_count=1,
        bars=[],
        total_count=1,
    )
    invalid = []
    for update in (
        {"sequence": 2},
        {"base_count": 0},
        {"total_count": 2},
        {"context": {**valid["context"], "inference_at": first_at.isoformat()}},
        {
            "bars": [
                {
                    "trade_date": "20260101",
                    "symbol": "000001.SZ",
                    "close": 10.0,
                    "available_at": "2026-01-02T17:00:00+08:00",
                }
            ],
            "total_count": 2,
        },
        {
            "bars": [
                {
                    "trade_date": "20260106",
                    "symbol": "000001.SZ",
                    "close": 14.0,
                    "available_at": "2026-01-06T17:30:00+08:00",
                }
            ],
            "total_count": 2,
        },
    ):
        invalid.append({**valid, **update})
    for request in invalid:
        with pytest.raises(WorkerProtocolError):
            stream.context(request, protocol)  # type: ignore[arg-type]

    restored = stream.context(valid, protocol)  # type: ignore[arg-type]
    assert len(restored.bars) == 1
    replacement = stream.context(_wire_request(later_context), protocol)  # type: ignore[arg-type]
    assert len(replacement.bars) == 1
    assert replacement.latest("000001.SZ")["close"] == 11.0


@pytest.mark.parametrize(
    ("original", "revised"),
    [(1, True), (True, 1), (1, 1.0), (0.0, -0.0)],
)
def test_executor_rejects_strict_json_type_or_value_drift_in_cached_prefix(
    original: object,
    revised: object,
):
    executor = object.__new__(DockerStrategyExecutor)
    executor.snapshot_dir = None
    executor.asof_dir = None
    executor._reset_transport_state()

    def context(day: int, value: object) -> StrategyContext:
        return StrategyContext(
            inference_at=datetime(2026, 1, day, 18, 0, tzinfo=CN_TZ),
            bars=(
                {
                    "trade_date": "20260102",
                    "symbol": "000001.SZ",
                    "signal": value,
                    "available_at": "2026-01-02T17:30:00+08:00",
                },
            ),
            account=AccountSnapshot(cash=10_000, positions={}),
        )

    initial = context(2, original)
    request, last_available_at = executor._prepare_execute(initial)
    executor._transport_sequence = request["sequence"]
    executor._transport_inference_at = initial.inference_at
    executor._transport_bars = initial.bars
    executor._transport_bar_identity = initial._bar_identity
    executor._transport_last_available_at = last_available_at
    original_identity = executor._transport_bar_identity

    with pytest.raises(StrategyExecutionError, match="changed before base_count"):
        executor._prepare_execute(context(5, revised))
    assert executor._transport_sequence == 0
    assert executor._transport_inference_at == initial.inference_at
    assert executor._transport_bars is initial.bars
    assert executor._transport_bar_identity is original_identity


def test_host_nl_wait_does_not_consume_the_inference_cap(tmp_path: Path):
    strategy = _strategy(
        tmp_path,
        """def generate_orders(context):
    context.nl(query="signal", mode="search", limit=1)
    return []
""",
    )
    process = subprocess.Popen(
        [sys.executable, "-m", "autotrade.environment.strategy_worker", str(strategy)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    executor = _executor_for_process(process, limits=SandboxLimits(timeout_seconds=0.2))

    def slow_nl(request, *, inference_at):
        time.sleep(0.5)
        return {"status": "ok", "query": request.get("query"), "evidence": []}

    context = StrategyContext(
        inference_at=datetime(2026, 1, 2, 8, 30, tzinfo=CN_TZ),
        bars=(),
        account=AccountSnapshot(cash=10_000, positions={}),
        _nl_query=slow_nl,
    )
    started = time.monotonic()
    try:
        assert executor.execute(context) == []
    finally:
        executor.close()
    assert time.monotonic() - started >= 0.5
    assert process.poll() == 0


def test_inference_timeout_aborts_and_closes_worker():
    limits = SandboxLimits(timeout_seconds=0.05)
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import sys,time; print('worker-tail', file=sys.stderr, flush=True); time.sleep(60)",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    executor = _executor_for_process(process, limits=limits, drain_stderr=True)
    pipes = (process.stdin, process.stdout, process.stderr)
    with (
        patch.object(executor, "_remove_container") as remove,
        pytest.raises(StrategyExecutionError, match="exceeded"),
    ):
        executor.execute(_context())
    assert process.poll() is not None
    assert executor._closed is True
    assert executor._process is None
    assert executor._stderr_thread is None
    assert all(pipe is not None and pipe.closed for pipe in pipes)
    remove.assert_called_once()
    executor.close()
    executor.close()
    remove.assert_called_once()


def test_request_write_timeout_aborts_worker_that_does_not_read_stdin():
    limits = SandboxLimits(timeout_seconds=0.05)
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    executor = _executor_for_process(process, limits=limits)
    started = time.monotonic()
    with (
        patch.object(executor, "_context_record", return_value={"payload": "x" * 2_000_000}),
        patch.object(executor, "_remove_container") as remove,
        pytest.raises(StrategyExecutionError, match="exceeded"),
    ):
        executor.execute(_context())
    elapsed = time.monotonic() - started
    assert elapsed < 1
    assert process.poll() is not None
    assert executor._closed is True
    assert executor._process is None
    remove.assert_called_once()


def test_request_write_retries_eintr_and_completes_partial_writes():
    executor = object.__new__(DockerStrategyExecutor)
    executor.config = SandboxConfig()
    read_fd, write_fd = os.pipe()
    chunks: list[bytes] = []
    calls = 0

    def partial_write(_fd, data):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise InterruptedError(errno.EINTR, "interrupted")
        size = min(3, len(data))
        chunks.append(bytes(data[:size]))
        return size

    try:
        with os.fdopen(write_fd, "wb", buffering=0) as stream, patch(
            "autotrade.environment.executor.os.write",
            side_effect=partial_write,
        ):
            executor._write(stream, {"value": "x"}, time.monotonic() + 1)
    finally:
        os.close(read_fd)
    assert calls > 2
    assert b"".join(chunks) == b'{"value": "x"}\n'


def test_request_write_propagates_epipe():
    executor = object.__new__(DockerStrategyExecutor)
    executor.config = SandboxConfig()
    read_fd, write_fd = os.pipe()
    try:
        with (
            os.fdopen(write_fd, "wb", buffering=0) as stream,
            patch(
                "autotrade.environment.executor.os.write",
                side_effect=BrokenPipeError(errno.EPIPE, "broken pipe"),
            ),
            pytest.raises(BrokenPipeError),
        ):
            executor._write(stream, {"value": "x"}, time.monotonic() + 1)
    finally:
        os.close(read_fd)


def test_worker_exit_before_response_is_reaped_and_cleared():
    process = subprocess.Popen(
        [sys.executable, "-c", "import sys; sys.stdin.readline(); raise SystemExit(7)"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    executor = _executor_for_process(process, drain_stderr=True)
    pipes = (process.stdin, process.stdout, process.stderr)
    with (
        patch.object(executor, "_remove_container") as remove,
        pytest.raises(StrategyExecutionError, match="worker exited before a response"),
    ):
        executor.execute(_context())
    assert process.returncode == 7
    assert executor._closed is True
    assert executor._process is None
    assert executor._stderr_thread is None
    assert all(pipe is not None and pipe.closed for pipe in pipes)
    remove.assert_called_once()


def test_close_reaps_worker_closes_pipes_and_is_idempotent():
    process = subprocess.Popen(
        [sys.executable, "-c", "import sys; sys.stdin.readline()"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    executor = _executor_for_process(process, drain_stderr=True)
    pipes = (process.stdin, process.stdout, process.stderr)
    with patch.object(executor, "_remove_container") as remove:
        executor.close()
        executor.close()
    assert process.returncode == 0
    assert executor._closed is True
    assert executor._process is None
    assert executor._stderr_thread is None
    assert all(pipe is not None and pipe.closed for pipe in pipes)
    remove.assert_not_called()


def test_pipeline_factory_reuses_replay_and_always_closes(tmp_path: Path):
    strategy = _strategy(tmp_path)
    daily = pd.DataFrame(
        [{"trade_date": "20260102", "symbol": "000001.SZ", "open": 10.0, "close": 11.0}]
    )

    class FakeExecutor:
        def __init__(self) -> None:
            self.calls = 0
            self.closed = False

        def execute(self, _context):
            self.calls += 1
            return []

        def close(self):
            self.closed = True

    executor = FakeExecutor()
    config = StrategyExperimentConfig(strategy_path=strategy, schedule=StrategySchedule())
    result = DailyStrategyPipeline(config, executor_factory=lambda _config: executor).run(daily)
    assert result.executions == ()
    assert executor.calls == 1
    assert executor.closed is True


_GPU_ROSTER = [
    {"index": 0, "name": "NVIDIA L20", "memory_free_mib": 8_000, "memory_total_mib": 46_000,
     "utilization_pct": 80, "temperature_c": 60},
    {"index": 1, "name": "NVIDIA L20", "memory_free_mib": 45_000, "memory_total_mib": 46_000,
     "utilization_pct": 0, "temperature_c": 30},
    {"index": 2, "name": "NVIDIA A100-SXM4-80GB", "memory_free_mib": 80_000,
     "memory_total_mib": 81_000, "utilization_pct": 0, "temperature_c": 30},
    {"index": 5, "name": "NVIDIA L20", "memory_free_mib": 40_000, "memory_total_mib": 46_000,
     "utilization_pct": 5, "temperature_c": 35},
]


def test_the_default_sandbox_spec_allocates_the_freest_matching_gpus():
    """`gpu="auto"` + `gpu_name_filter="L20"` is the shipped default.

    With `gpu=None` the whole selector is dead code and the per-session GPU
    count the console offers means nothing, so the defaults are part of the
    contract, not incidental.
    """
    spec = SandboxSpec()
    assert (spec.gpu, spec.gpu_count, spec.gpu_name_filter) == ("auto", 1, "L20")
    record = spec.to_record()
    assert record["image_ref"] == "autotrade-sandbox:latest"
    assert record["build_generation_id"] is None
    assert "image" not in record
    assert record["gpu"] == "auto" and record["gpu_count"] == 1
    assert record["gpu_name_filter"] == "L20"


def test_select_gpus_ranks_matching_devices_by_free_memory():
    from autotrade.environment.gpu import GpuUnavailableError, select_gpus

    with patch("autotrade.environment.gpu.list_gpus", return_value=_GPU_ROSTER):
        # Freest first, and the non-L20 device is never offered even though it
        # has the most free memory of all.
        assert select_gpus(1, require_name="L20") == [1]
        assert select_gpus(2, require_name="L20") == [1, 5]
        assert select_gpus(3, require_name="L20") == [1, 5, 0]
        assert select_gpus(1) == [2]
        with pytest.raises(GpuUnavailableError, match="requested 4 GPU"):
            select_gpus(4, require_name="L20")
        with pytest.raises(GpuUnavailableError, match="available matching GPUs: none"):
            select_gpus(1, require_name="H100")


def test_persistent_sandbox_start_pins_the_selected_gpus_on_the_container(tmp_path: Path):
    local = LocalSandbox(tmp_path / "session")
    local.prepare_layout()
    sandbox = DockerSandbox(local, SandboxSpec(gpu_count=2))
    completed = subprocess.CompletedProcess(args=[], returncode=0, stdout="container\n", stderr="")
    with (
        patch("autotrade.environment.gpu.list_gpus", return_value=_GPU_ROSTER),
        patch("autotrade.environment.sandbox.probe_image_runtime", return_value={}),
        patch("autotrade.environment.sandbox.subprocess.run", return_value=completed) as run,
    ):
        sandbox.start()
    assert sandbox.gpu_indices == [1, 5]
    command = run.call_args_list[0][0][0]
    assert command[command.index("--gpus") + 1] == "device=1,5"
    assert sandbox.allocation_record()["allocated_gpu_indices"] == [1, 5]


def test_a_cpu_only_sandbox_never_consults_the_gpu_selector(tmp_path: Path):
    local = LocalSandbox(tmp_path / "session")
    local.prepare_layout()
    sandbox = DockerSandbox(local, SandboxSpec(gpu=None))
    completed = subprocess.CompletedProcess(args=[], returncode=0, stdout="container\n", stderr="")
    with (
        patch("autotrade.environment.gpu.list_gpus", side_effect=AssertionError("selector ran")),
        patch("autotrade.environment.sandbox.probe_image_runtime", return_value={}),
        patch("autotrade.environment.sandbox.subprocess.run", return_value=completed) as run,
    ):
        sandbox.start()
    assert sandbox.gpu_indices == []
    assert "--gpus" not in run.call_args_list[0][0][0]


def test_layout_has_no_decoy_output_siblings_and_collects_the_workspace(tmp_path: Path):
    """The formal working copies live inside the workspace; the layout must not
    create empty ``agent/output`` / ``agent/models`` siblings that documents,
    tools or a model could mistake for the real ones."""
    local = LocalSandbox(tmp_path / "session")
    paths = local.prepare_layout()
    assert not (paths.agent / "output").exists()
    assert not (paths.agent / "models").exists()
    assert not hasattr(paths, "agent_output") and not hasattr(paths, "writable_root_map")
    work_output = paths.workspace / "output"
    work_models = paths.workspace / "models"
    work_output.mkdir(parents=True, exist_ok=True)
    work_models.mkdir(parents=True, exist_ok=True)
    (work_output / "main.py").write_text("working copy\n", encoding="utf-8")
    (work_models / "weights.json").write_text('{"src": "workspace"}\n', encoding="utf-8")
    dest = local.collect_artifacts(tmp_path / "collected")
    assert (dest / "workspace" / "output" / "main.py").read_text(encoding="utf-8") == "working copy\n"
    assert (dest / "workspace" / "models" / "weights.json").read_text(encoding="utf-8") == (
        '{"src": "workspace"}\n'
    )
    assert not (dest / "output").exists() and not (dest / "models").exists()
