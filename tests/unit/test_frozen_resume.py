"""Resume parent replay and post-fold prune of unreferenced frozen trees."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from autotrade.environment.artifacts import FilesystemArtifactStore
from autotrade.pipelines.experiment import _keep_frozen_artifact_ids
from autotrade.pipelines.ledger import (
    ExperimentLedger,
    FrozenArtifactMutated,
    assert_no_frozen_artifact_mutation,
    latest_fold_records,
    latest_heldout_records,
)
from autotrade.pipelines.worker import _latest_artifact

MAIN = "def generate_orders(context):\n    return []\n"


def _freeze(store: FilesystemArtifactStore, artifact_id: str, *, experiment_id: str = "exp"):
    source = store.root / f"_src_{artifact_id}"
    source.mkdir()
    (source / "main.py").write_text(MAIN, encoding="utf-8")
    revision = store.create_revision(source, revision_id=f"revision_{artifact_id}")
    return store.freeze_revision(
        revision.revision_id,
        artifact_id=artifact_id,
        experiment_id=experiment_id,
        epoch_id="epoch_001",
        fold_id=artifact_id,
        run_id=f"run_{artifact_id}",
        step_id="step_001",
    )


def _append_fold(
    ledger: ExperimentLedger,
    *,
    fold_id: str,
    artifact_id: str | None,
    path: str | None,
    run_id: str,
    status: str = "frozen",
    state_changed: bool = False,
) -> None:
    record: dict[str, object] = {
        "record_type": "fold",
        "experiment_id": "exp",
        "epoch_id": "epoch_001",
        "fold_id": fold_id,
        "run_id": run_id,
        "session_key": f"epoch_001/{fold_id}",
        "fold_status": status,
        "frozen_strategy_artifact_id": artifact_id,
        "frozen_strategy_artifact_path": path,
    }
    if state_changed:
        record["state_changed_during_test"] = True
    ledger.append(record)


def _append_heldout(
    ledger: ExperimentLedger,
    *,
    period: str,
    artifact_id: str,
    run_id: str,
    state_changed: bool = False,
) -> None:
    record: dict[str, object] = {
        "record_type": "heldout",
        "experiment_id": "exp",
        "epoch_id": "epoch_001",
        "fold_id": f"heldout_{period}",
        "run_id": run_id,
        "session_key": "heldout",
        "period": period,
        "strategy_artifact_id": artifact_id,
    }
    if state_changed:
        record["state_changed_during_test"] = True
    ledger.append(record)


def _append_meta(
    ledger: ExperimentLedger,
    *,
    session_key: str,
    status: str,
    artifact_id: str | None = None,
    path: str | None = None,
    run_id: str = "run_meta",
    fold_id: str = "epoch_001",
) -> None:
    ledger.append(
        {
            "record_type": "meta_learning",
            "experiment_id": "exp",
            "epoch_id": "epoch_001",
            "fold_id": fold_id,
            "run_id": run_id,
            "session_key": session_key,
            "status": status,
            "frozen_strategy_artifact_id": artifact_id,
            "frozen_strategy_artifact_path": path,
        }
    )


def test_freeze_revision_is_a_valid_fold_parent_without_validation_flag(tmp_path: Path) -> None:
    store = FilesystemArtifactStore(tmp_path / "store")
    frozen = _freeze(store, "strategy_fold_001")
    assert frozen.requires_validation is False
    reloaded = store.frozen(frozen.artifact_id, expected_path=frozen.path, experiment_id="exp")
    assert reloaded.requires_validation is False


def test_prune_keeps_prior_fold_and_meta_and_drops_superseded(tmp_path: Path) -> None:
    store = FilesystemArtifactStore(tmp_path / "store")
    ledger = ExperimentLedger(tmp_path / "ledger.jsonl")
    fold_n1 = _freeze(store, "strategy_fold_n1")
    superseded = _freeze(store, "strategy_fold_n_old")
    fold_n = _freeze(store, "strategy_fold_n")
    meta = _freeze(store, "strategy_meta")
    leftover = store.create_revision(
        store.root / "_src_strategy_fold_n1", revision_id="revision_leftover"
    )

    _append_fold(
        ledger,
        fold_id="fold_001",
        artifact_id=fold_n1.artifact_id,
        path=str(fold_n1.path),
        run_id="run_fold_n1",
    )
    _append_fold(
        ledger,
        fold_id="fold_002",
        artifact_id=superseded.artifact_id,
        path=str(superseded.path),
        run_id="run_fold_n_old",
    )
    _append_meta(
        ledger,
        session_key="epoch_001/meta_learning_after_fold_001",
        status="meta_regularized",
        artifact_id=meta.artifact_id,
        path=str(meta.path),
    )
    _append_fold(
        ledger,
        fold_id="fold_002",
        artifact_id=fold_n.artifact_id,
        path=str(fold_n.path),
        run_id="run_fold_n",
    )

    store.prune_transient(
        keep_frozen_ids=_keep_frozen_artifact_ids(
            ledger.read(), extra_id=fold_n.artifact_id
        )
    )

    frozen = {path.name for path in store.frozen_root.iterdir()}
    assert frozen == {
        fold_n1.artifact_id,
        fold_n.artifact_id,
        meta.artifact_id,
    }
    assert not any(store.revisions_root.iterdir())
    assert leftover.revision_id not in frozen


def test_latest_artifact_walks_back_from_baseline_missing(tmp_path: Path) -> None:
    store = FilesystemArtifactStore(tmp_path / "store")
    ledger = ExperimentLedger(tmp_path / "ledger.jsonl")
    previous = _freeze(store, "strategy_fold_001")
    _append_fold(
        ledger,
        fold_id="fold_001",
        artifact_id=previous.artifact_id,
        path=str(previous.path),
        run_id="run_fold_001",
    )
    _append_fold(
        ledger,
        fold_id="fold_002",
        artifact_id=None,
        path=None,
        run_id="run_fold_002",
        status="baseline_missing",
    )

    parent = _latest_artifact(ledger, store)

    assert parent is not None
    assert parent.artifact_id == previous.artifact_id
    assert parent.path == Path(previous.path)
    assert parent.model_path == (
        Path(previous.model_path) if previous.model_path is not None else None
    )
    assert parent.source_run_id == previous.source_run_id
    assert parent.source_fold_id == previous.source_fold_id
    assert parent.source_step_id == previous.source_step_id
    assert parent.revision_id == previous.revision_id
    assert parent.requires_validation is False


def test_latest_artifact_uses_meta_regularized_parent(tmp_path: Path) -> None:
    store = FilesystemArtifactStore(tmp_path / "store")
    ledger = ExperimentLedger(tmp_path / "ledger.jsonl")
    previous = _freeze(store, "strategy_fold_001")
    meta = _freeze(store, "strategy_meta")
    _append_fold(
        ledger,
        fold_id="fold_001",
        artifact_id=previous.artifact_id,
        path=str(previous.path),
        run_id="run_fold_001",
    )
    _append_meta(
        ledger,
        session_key="epoch_001/meta_learning_after_fold_001",
        status="prior_only_kept_parent",
        run_id="run_meta_prior",
        fold_id="epoch_001_after_fold_001",
    )
    _append_meta(
        ledger,
        session_key="epoch_001/meta_learning_after_fold_001",
        status="meta_regularized",
        artifact_id=meta.artifact_id,
        path=str(meta.path),
        run_id="run_meta",
        fold_id="epoch_001_after_fold_001",
    )

    parent = _latest_artifact(ledger, store)

    assert parent is not None
    assert parent.artifact_id == meta.artifact_id
    assert parent.path == Path(meta.path)
    assert parent.model_path == (
        Path(meta.model_path) if meta.model_path is not None else None
    )
    assert parent.source_run_id == meta.source_run_id
    assert parent.source_fold_id == meta.source_fold_id
    assert parent.source_step_id == meta.source_step_id
    assert parent.revision_id == meta.revision_id
    assert parent.requires_validation is True


def test_latest_artifact_refuses_a_flagged_fold(tmp_path: Path) -> None:
    store = FilesystemArtifactStore(tmp_path / "store")
    ledger = ExperimentLedger(tmp_path / "ledger.jsonl")
    previous = _freeze(store, "strategy_fold_001")
    flagged = _freeze(store, "strategy_fold_002")
    _append_fold(
        ledger,
        fold_id="fold_001",
        artifact_id=previous.artifact_id,
        path=str(previous.path),
        run_id="run_fold_001",
    )
    _append_fold(
        ledger,
        fold_id="fold_002",
        artifact_id=flagged.artifact_id,
        path=str(flagged.path),
        run_id="run_fold_002",
        state_changed=True,
    )
    latest = latest_fold_records(ledger.read())
    assert set(latest) == {("epoch_001", "fold_001")}
    assert latest[("epoch_001", "fold_001")]["run_id"] == "run_fold_001"
    with pytest.raises(FrozenArtifactMutated):
        _latest_artifact(ledger, store)


def test_latest_heldout_records_skip_flagged_and_parent_selection_refuses(
    tmp_path: Path,
) -> None:
    store = FilesystemArtifactStore(tmp_path / "store")
    ledger = ExperimentLedger(tmp_path / "ledger.jsonl")
    frozen = _freeze(store, "strategy_fold_001")
    _append_fold(
        ledger,
        fold_id="fold_001",
        artifact_id=frozen.artifact_id,
        path=str(frozen.path),
        run_id="run_fold_001",
    )
    _append_heldout(
        ledger,
        period="2026Q2",
        artifact_id=frozen.artifact_id,
        run_id="run_heldout_ok",
    )
    _append_heldout(
        ledger,
        period="2026Q2",
        artifact_id=frozen.artifact_id,
        run_id="run_heldout_flagged",
        state_changed=True,
    )
    latest = latest_heldout_records(ledger.read())
    assert len(latest) == 1
    assert latest[0]["run_id"] == "run_heldout_ok"
    assert "state_changed_during_test" not in latest[0]
    with pytest.raises(FrozenArtifactMutated):
        _latest_artifact(ledger, store)


def _console_experiment(
    tmp_path: Path, *, periods: tuple[str, ...]
) -> tuple[Path, ExperimentLedger]:
    experiment = tmp_path / "exp"
    hitl = experiment / "hitl"
    hitl.mkdir(parents=True)
    (hitl / "schedule.json").write_text(
        json.dumps(
            {
                "sessions": [
                    {
                        "session_key": "heldout",
                        "kind": "heldout",
                        "periods": [{"label": period} for period in periods],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    return experiment, ExperimentLedger(
        experiment / "ledgers" / "experiment_ledger.jsonl"
    )


def test_flagged_heldout_never_completes_or_auto_reveals(tmp_path: Path) -> None:
    from autotrade.webui.registry import heldout_complete, test_results_revealed

    experiment, ledger = _console_experiment(tmp_path, periods=("2026Q2",))
    _append_heldout(
        ledger,
        period="2026Q2",
        artifact_id="strategy_fold_001",
        run_id="run_heldout_flagged",
        state_changed=True,
    )
    records = ledger.read()
    assert latest_heldout_records(records) == []
    assert heldout_complete(experiment, records) is False
    assert test_results_revealed(experiment, records) is False


def test_heldout_success_then_flagged_does_not_wash_into_reveal(
    tmp_path: Path,
) -> None:
    from autotrade.webui.registry import heldout_complete, test_results_revealed

    experiment, ledger = _console_experiment(tmp_path, periods=("2026Q2",))
    _append_heldout(
        ledger,
        period="2026Q2",
        artifact_id="strategy_fold_001",
        run_id="run_heldout_ok",
    )
    assert heldout_complete(experiment, ledger.read()) is True
    _append_heldout(
        ledger,
        period="2026Q2",
        artifact_id="strategy_fold_001",
        run_id="run_heldout_flagged",
        state_changed=True,
    )
    records = ledger.read()
    latest = latest_heldout_records(records)
    assert len(latest) == 1
    assert latest[0]["run_id"] == "run_heldout_ok"
    assert heldout_complete(experiment, records) is False
    assert test_results_revealed(experiment, records) is False


def test_later_heldout_success_does_not_wash_a_flagged_row(tmp_path: Path) -> None:
    from autotrade.webui.registry import heldout_complete, test_results_revealed

    experiment, ledger = _console_experiment(tmp_path, periods=("2026Q2",))
    _append_heldout(
        ledger,
        period="2026Q2",
        artifact_id="strategy_fold_001",
        run_id="run_heldout_flagged",
        state_changed=True,
    )
    _append_heldout(
        ledger,
        period="2026Q2",
        artifact_id="strategy_fold_001",
        run_id="run_heldout_ok",
    )
    records = ledger.read()
    latest = latest_heldout_records(records)
    assert len(latest) == 1
    assert latest[0]["run_id"] == "run_heldout_ok"
    assert heldout_complete(experiment, records) is False
    assert test_results_revealed(experiment, records) is False


def test_prune_keeps_a_flagged_freeze_on_the_real_store(tmp_path: Path) -> None:
    store = FilesystemArtifactStore(tmp_path / "store")
    ledger = ExperimentLedger(tmp_path / "ledger.jsonl")
    flagged = _freeze(store, "strategy_fold_flagged")
    leftover = _freeze(store, "strategy_fold_leftover")
    _append_fold(
        ledger,
        fold_id="fold_001",
        artifact_id=flagged.artifact_id,
        path=str(flagged.path),
        run_id="run_flagged",
        state_changed=True,
    )
    store.prune_transient(keep_frozen_ids=_keep_frozen_artifact_ids(ledger.read()))
    frozen = {path.name for path in store.frozen_root.iterdir()}
    assert flagged.artifact_id in frozen
    assert leftover.artifact_id not in frozen


def test_flagged_fold_is_not_counted_complete(tmp_path: Path) -> None:
    from autotrade.webui.registry import _durable_session_progress

    experiment = tmp_path / "exp"
    hitl = experiment / "hitl"
    hitl.mkdir(parents=True)
    sessions = [
        {
            "session_key": "epoch_001/fold_001",
            "kind": "fold",
            "epoch_id": "epoch_001",
            "fold_id": "fold_001",
        },
        {
            "session_key": "heldout",
            "kind": "heldout",
            "periods": [{"label": "2026Q2"}],
        },
    ]
    (hitl / "schedule.json").write_text(
        json.dumps({"sessions": sessions}), encoding="utf-8"
    )
    ledger = ExperimentLedger(experiment / "ledgers" / "experiment_ledger.jsonl")
    _append_fold(
        ledger,
        fold_id="fold_001",
        artifact_id="strategy_fold_001",
        path="missing",
        run_id="run_flagged",
        state_changed=True,
    )
    completed, total = _durable_session_progress(sessions, ledger.read())
    assert completed == 0
    assert total == 2


def test_first_fold_flagged_prune_rollback_and_retry(tmp_path: Path) -> None:
    import pandas as pd

    from autotrade.environment.runtime import chmod_tree, write_json_atomic
    from autotrade.pipelines import (
        EvaluationResult,
        FoldSessionResult,
        RollingExperimentConfig,
        RollingExperimentPipeline,
        StepResult,
    )
    from autotrade.pipelines.folds import build_fold_schedule
    from autotrade.pipelines.hitl_state import ControlState, read_control, write_control
    from autotrade.webui.manager import ExperimentManager

    experiment = tmp_path / "experiments" / "experiment_a"
    store = FilesystemArtifactStore(experiment / "artifacts" / "strategy")
    source = tmp_path / "src"
    source.mkdir()
    (source / "main.py").write_text(MAIN, encoding="utf-8")

    class Snapshots:
        def prepare(self, *, fold, phase, start, end, decision_time):
            from autotrade.pipelines.config import SnapshotBundle

            return SnapshotBundle(f"{phase}_{start}_{end}", "decision", "replay")

    class Mutating:
        def evaluate(self, request):
            output = request.revision.output_path
            chmod_tree(output, file_mode=0o666, dir_mode=0o777)
            main = output / "main.py"
            main.write_text(
                main.read_text(encoding="utf-8") + "# mutated\n", encoding="utf-8"
            )
            return EvaluationResult(
                {"total_return": 0.02, "max_drawdown": -0.03}, "result/frozen_test"
            )

    class Stable:
        def evaluate(self, request):
            return EvaluationResult(
                {"total_return": 0.02, "max_drawdown": -0.03}, f"result/{request.mode}"
            )

    def developer(request):
        revision = store.create_revision(source)
        return FoldSessionResult(
            "conversation_1",
            (
                StepResult(
                    "step_1",
                    revision.revision_id,
                    EvaluationResult(
                        {"total_return": 0.05, "max_drawdown": -0.02}, "result/valid"
                    ),
                    True,
                ),
            ),
            "step_1",
        )

    config = RollingExperimentConfig(
        "experiment_a",
        tmp_path / "experiments",
        "2025Q4",
        "2026Q1",
        "2026Q2",
        "2026Q2",
        fold_period="quarter",
        test_stage=True,
        epochs=1,
    )
    ledger = ExperimentLedger(config.ledger_path)
    pipeline = RollingExperimentPipeline(
        config,
        snapshots=Snapshots(),
        artifacts=store,
        evaluator=Mutating(),
        developer=developer,
        ledger=ledger,
    )
    days = [
        stamp.strftime("%Y%m%d")
        for stamp in pd.bdate_range("2025-09-29", "2026-06-30")
    ]
    fold = build_fold_schedule(
        "2025Q4", "2026Q1", days, window_months=24, test_stage=True
    )[0]
    with pytest.raises(FrozenArtifactMutated, match="changed during frozen test"):
        pipeline.run_fold("epoch_001", fold, parent=None)
    records = ledger.read("fold")
    assert len(records) == 1
    artifact_id = str(records[0]["frozen_strategy_artifact_id"])
    frozen_dir = store.frozen_root / artifact_id
    assert frozen_dir.is_dir()
    main = frozen_dir / "output" / "main.py"
    assert main.read_text(encoding="utf-8") == MAIN
    assert not (main.stat().st_mode & 0o222)

    hitl = experiment / "hitl"
    hitl.mkdir(parents=True, exist_ok=True)
    write_json_atomic(hitl / "params.json", {"experiment_id": "experiment_a"})
    write_control(hitl / "control.json", ControlState(mode="manual"))
    write_json_atomic(
        hitl / "status.json",
        {"schema_version": 1, "pid": 999_999_999, "state": "stopped"},
    )
    write_json_atomic(
        hitl / "schedule.json",
        {
            "schema_version": 1,
            "sessions": [
                {
                    "key": f"epoch_001/{fold.fold_id}",
                    "kind": "fold",
                    "epoch_id": "epoch_001",
                    "fold_id": fold.fold_id,
                },
                {
                    "key": "heldout",
                    "kind": "heldout",
                    "epoch_id": "epoch_001",
                    "periods": [{"label": "2026Q2"}],
                },
            ],
        },
    )
    manager = ExperimentManager(tmp_path, tmp_path / "experiments")
    manager._rollback_to_fold(
        experiment, f"epoch_001/{fold.fold_id}", read_control(hitl / "control.json")
    )
    remaining = ledger.read()
    assert remaining == []
    assert_no_frozen_artifact_mutation(remaining)
    assert not frozen_dir.is_dir()
    assert list((experiment / "artifacts/strategy/_archive").glob("rollback_*"))

    retry = RollingExperimentPipeline(
        config,
        snapshots=Snapshots(),
        artifacts=store,
        evaluator=Stable(),
        developer=developer,
        ledger=ledger,
    )
    outcome = retry.run_fold("epoch_001", fold, parent=None)
    assert outcome.fold_status == "frozen"
    assert_no_frozen_artifact_mutation(ledger.read())
    assert latest_fold_records(ledger.read())
