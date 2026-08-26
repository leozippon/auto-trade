from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/llm/run_vllm.py"
SPEC = importlib.util.spec_from_file_location("run_local_vllm", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
run_vllm = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(run_vllm)


def test_headless_entry_rejects_endpoint_override_before_persistence(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location(
        "experiment_cli_endpoint_guard",
        repo_root / "scripts/experiments/_cli.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    args = argparse.Namespace(
        experiments_root=tmp_path / "experiments",
        experiment_id="blocked_endpoint",
    )

    with pytest.raises(
        ValueError,
        match="provider endpoint parameters cannot be persisted: llm_base_url",
    ):
        module.build_worker_options(
            args,
            repo_root=tmp_path,
            overrides={"llm_base_url": "https://untrusted.example.test/v1"},
        )
    assert not (tmp_path / "experiments/blocked_endpoint").exists()


def test_launcher_defaults_to_bf16_two_card_profile() -> None:
    args = run_vllm.build_parser().parse_args([])
    assert args.model_path == Path("/Data/public/Qwen3.8-27B")
    assert args.tensor_parallel_size == 2
    assert args.max_model_len == 262_144


def test_launcher_keeps_key_out_of_argv_and_binds_loopback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_dir = tmp_path / "Qwen3.8-27B-FP8"
    model_dir.mkdir()
    env_file = tmp_path / ".env"
    env_file.write_text(
        "export VLLM_API_KEY='local-test-secret'\n"
        "export HF_TOKEN='hf-test-token'\n"
        "export HF_ENDPOINT='https://hf.example.test'\n",
        encoding="utf-8",
    )
    resource_checks: list[list[str]] = []
    launched: list[tuple[str, list[str], str]] = []
    environment_bin = tmp_path / "vllm-env/bin"
    interpreter_bin = tmp_path / "interpreter/bin"
    environment_bin.mkdir(parents=True)
    interpreter_bin.mkdir(parents=True)
    interpreter = environment_bin / "python"
    interpreter.symlink_to(interpreter_bin / "python")
    monkeypatch.delenv("VLLM_API_KEY", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HF_ENDPOINT", raising=False)
    monkeypatch.setenv("PATH", "/existing/first:/existing/second")
    monkeypatch.setattr(run_vllm.sys, "executable", str(interpreter))
    monkeypatch.setattr(
        run_vllm.subprocess,
        "run",
        lambda argv, check: resource_checks.append(list(argv)),
    )
    monkeypatch.setattr(
        run_vllm.os,
        "execv",
        lambda executable, argv: launched.append(
            (executable, list(argv), os.environ["PATH"])
        ),
    )

    assert (
        run_vllm.main(
            [
                "--model-path",
                str(model_dir),
                "--env-file",
                str(env_file),
                "--tensor-parallel-size",
                "1",
            ]
        )
        == 0
    )

    assert resource_checks == [["nvidia-smi"], ["free", "-h"]]
    executable, argv, inherited_path = launched[0]
    assert executable == sys.executable
    assert argv[:3] == [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
    ]
    assert argv[argv.index("--host") + 1] == "127.0.0.1"
    assert argv[argv.index("--port") + 1] == "8010"
    assert argv[argv.index("--served-model-name") + 1] == run_vllm.LOCAL_QWEN_MODEL
    assert argv[argv.index("--model") + 1] == str(model_dir)
    assert argv[argv.index("--tensor-parallel-size") + 1] == "1"
    assert argv[argv.index("--max-model-len") + 1] == "262144"
    assert argv[argv.index("--kv-cache-dtype") + 1] == "fp8"
    assert argv[argv.index("--tool-call-parser") + 1] == "qwen3_coder"
    assert argv[argv.index("--reasoning-parser") + 1] == "qwen3"
    assert "--api-key" not in argv
    assert "local-test-secret" not in argv
    assert inherited_path.split(os.pathsep) == [
        str(environment_bin.resolve()),
        "/existing/first",
        "/existing/second",
    ]
    assert os.environ["VLLM_API_KEY"] == "local-test-secret"
    assert os.environ["HF_TOKEN"] == "hf-test-token"
    assert os.environ["HF_ENDPOINT"] == "https://hf.example.test"


def test_launcher_fails_before_resource_checks_without_api_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_dir = tmp_path / "Qwen3.8-27B-FP8"
    model_dir.mkdir()
    env_file = tmp_path / ".env"
    env_file.write_text("VLLM_API_KEY=\n", encoding="utf-8")
    monkeypatch.delenv("VLLM_API_KEY", raising=False)
    monkeypatch.setattr(
        run_vllm.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("resource check must not run"),
    )

    with pytest.raises(ValueError, match="VLLM_API_KEY is required"):
        run_vllm.main(["--model-path", str(model_dir), "--env-file", str(env_file)])
