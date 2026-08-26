"""Resume parent replay and post-fold prune of unreferenced frozen trees."""

from __future__ import annotations

from pathlib import Path

from autotrade.environment.artifacts import FilesystemArtifactStore
from autotrade.pipelines.experiment import _keep_frozen_artifact_ids
from autotrade.pipelines.ledger import ExperimentLedger
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
) -> None:
    ledger.append(
        {
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
    )


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
