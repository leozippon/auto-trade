from __future__ import annotations

import json
import multiprocessing
import os
import stat
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from autotrade.environment.identity import (
    LEGACY_EXPERIMENT_MESSAGE,
    AgentRefStore,
    AgentRefStoreError,
    LegacyExperimentError,
)
from autotrade.environment.runtime import SandboxPaths, write_json_atomic
from autotrade.environment.step_tree import StepTree
from autotrade.pipelines.hitl_state import ControlState, write_control
from autotrade.pipelines.worker import InteractiveWorkerOptions, run_local_interactive_worker
from autotrade.webui.manager import ExperimentManager, ManagerError
from autotrade.webui.prompt_preview import build_prompt_preview
from autotrade.webui.steps import step_tree_view


def _create_ref(args: tuple[str, str, str]) -> str:
    experiment, namespace, source = args
    return AgentRefStore(experiment).get_or_create(namespace, source)


def _payload(experiment: Path, entries: list[dict[str, str]]) -> dict[str, object]:
    return {
        "schema_version": 1,
        "experiment": experiment.name,
        "entries": entries,
    }


def _replace_store(experiment: Path, payload: object) -> None:
    path = experiment / ".host/agent-refs.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    path.chmod(0o600)


def test_store_is_restart_stable_experiment_scoped_and_namespaced(tmp_path: Path) -> None:
    first_dir = tmp_path / "exp_a"
    second_dir = tmp_path / "exp_b"
    first = AgentRefStore(first_dir)
    fold_ref = first.get_or_create("fold", "fold_2026Q1")

    assert AgentRefStore(first_dir).get_or_create("fold", "fold_2026Q1") == fold_ref
    assert AgentRefStore(second_dir).get_or_create("fold", "fold_2026Q1") != fold_ref
    assert first.get_or_create("trace", "fold_2026Q1") != fold_ref
    assert "2026Q1" not in fold_ref
    parsed = uuid.UUID(fold_ref.removeprefix("fold_ref_"))
    assert parsed.version == 4
    assert first.resolve("fold", fold_ref) == "fold_2026Q1"
    with pytest.raises(AgentRefStoreError):
        first.resolve("trace", fold_ref)


def test_get_or_create_is_multiprocess_singleton(tmp_path: Path) -> None:
    experiment = tmp_path / "parallel"
    AgentRefStore(experiment)
    args = [(str(experiment), "fold", "fold_2026Q1")] * 16
    context = multiprocessing.get_context("spawn")
    with context.Pool(8) as pool:
        refs = pool.map(_create_ref, args)

    assert len(set(refs)) == 1
    payload = json.loads((experiment / ".host/agent-refs.json").read_text(encoding="utf-8"))
    assert len(payload["entries"]) == 1


def test_store_permissions_are_private(tmp_path: Path) -> None:
    experiment = tmp_path / "permissions"
    store = AgentRefStore(experiment)
    store.get_or_create("run", "run_raw")

    assert stat.S_IMODE(store.host_dir.stat().st_mode) == 0o700
    assert stat.S_IMODE(store.path.stat().st_mode) == 0o600
    assert stat.S_IMODE(store.lock_path.stat().st_mode) == 0o600


@pytest.mark.parametrize(
    "payload",
    [
        "not-json",
        {"schema_version": 1, "experiment": "wrong", "entries": []},
        _payload(
            Path("exp"),
            [
                {
                    "namespace": "fold",
                    "source": "fold_a",
                    "ref": "fold_ref_00000000-0000-4000-8000-000000000001",
                },
                {
                    "namespace": "fold",
                    "source": "fold_a",
                    "ref": "fold_ref_00000000-0000-4000-8000-000000000002",
                },
            ],
        ),
        _payload(
            Path("exp"),
            [
                {
                    "namespace": "fold",
                    "source": "fold_a",
                    "ref": "fold_ref_00000000-0000-5000-8000-000000000001",
                }
            ],
        ),
    ],
)
def test_store_rejects_corrupt_duplicate_mismatch_and_non_uuid4(
    tmp_path: Path, payload: object
) -> None:
    experiment = tmp_path / "exp"
    store = AgentRefStore(experiment)
    if payload == "not-json":
        store.path.write_text(str(payload), encoding="utf-8")
        store.path.chmod(0o600)
    else:
        _replace_store(experiment, payload)
    with pytest.raises(AgentRefStoreError):
        AgentRefStore(experiment)


def test_write_failure_never_returns_or_persists_new_ref(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = AgentRefStore(tmp_path / "write-failure")
    original = os.replace

    def fail_replace(source: str | bytes | os.PathLike[str] | os.PathLike[bytes], dest: str | bytes | os.PathLike[str] | os.PathLike[bytes]) -> None:
        if Path(os.fsdecode(dest)) == store.path:
            raise OSError("injected replace failure")
        original(source, dest)

    monkeypatch.setattr(os, "replace", fail_replace)
    with pytest.raises(OSError, match="injected"):
        store.get_or_create("fold", "fold_2026Q1")
    monkeypatch.setattr(os, "replace", original)

    payload = json.loads(store.path.read_text(encoding="utf-8"))
    assert payload["entries"] == []
    assert not list(store.host_dir.glob(".agent-refs.*.tmp"))


@pytest.mark.parametrize(
    "legacy_path",
    [
        "ledgers/experiment_ledger.jsonl",
        "steps/tree.json",
        "artifacts/traces/run_old.jsonl",
    ],
)
def test_legacy_identity_artifacts_are_read_only(
    tmp_path: Path, legacy_path: str
) -> None:
    experiment = tmp_path / "legacy"
    path = experiment / legacy_path
    path.parent.mkdir(parents=True)
    path.write_text("{}\n", encoding="utf-8")

    with pytest.raises(LegacyExperimentError, match=LEGACY_EXPERIMENT_MESSAGE):
        AgentRefStore(experiment)
    with pytest.raises(LegacyExperimentError, match=LEGACY_EXPERIMENT_MESSAGE):
        run_local_interactive_worker(
            cast(InteractiveWorkerOptions, SimpleNamespace(experiment_dir=experiment))
        )
    assert not (experiment / ".host").exists()


def test_legacy_web_audit_remains_readable_but_mutations_and_preview_fail(
    tmp_path: Path,
) -> None:
    experiments = tmp_path / "experiments"
    experiment = experiments / "legacy"
    tree = StepTree(experiment / "steps")
    tree.record_failed_attempt(
        epoch_id="epoch_001",
        fold_id="fold_ref_deadbeef00",
        run_id="run_old",
        result_name="valid_001",
        error="old failure",
    )
    hitl = experiment / "hitl"
    hitl.mkdir(parents=True)
    write_control(hitl / "control.json", ControlState(mode="manual"))
    write_json_atomic(hitl / "status.json", {"schema_version": 1, "state": "stopped"})
    write_json_atomic(hitl / "params.json", {"strategy_period": "day"})
    write_json_atomic(
        hitl / "schedule.json",
        {
            "sessions": [
                {
                    "session_key": "epoch_001/fold_2024Q1",
                    "kind": "fold",
                    "epoch_id": "epoch_001",
                    "fold_id": "fold_2024Q1",
                }
            ]
        },
    )

    audit = step_tree_view(experiment)
    assert audit["nodes"][0]["fold_ref"] == "fold_ref_deadbeef00"
    assert audit["nodes"][0]["fold_id"] is None
    with pytest.raises(LegacyExperimentError, match=LEGACY_EXPERIMENT_MESSAGE):
        build_prompt_preview(experiment, "epoch_001/fold_2024Q1", "")
    with pytest.raises(ManagerError, match=LEGACY_EXPERIMENT_MESSAGE):
        ExperimentManager(tmp_path, experiments).control(
            "legacy",
            "set_directive",
            session_key="epoch_001/fold_2024Q1",
            directive="keep it simple",
        )


def test_host_mapping_is_not_a_sandbox_path(tmp_path: Path) -> None:
    experiment = tmp_path / "exp"
    store = AgentRefStore(experiment)
    paths = SandboxPaths(tmp_path / "sandbox")

    assert store.host_dir not in paths.writable_root_map.values()
    assert all(".host" not in str(path) for path in paths.writable_root_map.values())
