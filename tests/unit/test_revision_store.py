"""The strategy revision store as the arm's artifact history.

Every revision a Validation recorded is kept, so the conditions that must hold
are retention, cheap storage (one copy of each distinct file), a lineage that
survives on disk, and readability of the revisions the older layout left
behind.
"""

from __future__ import annotations

import stat
from pathlib import Path

import pytest

from autotrade.environment.artifacts import (
    REVISION_MANIFEST_FILE,
    FilesystemArtifactStore,
    artifact_fingerprint,
    copy_artifact,
    copy_model_artifacts,
)
from autotrade.environment.runtime import chmod_tree

STRATEGY = "def generate_orders(context):\n    return []\n"
HELPER = "\n".join(f"VALUE_{index} = {index}" for index in range(40)) + "\n"


def _working_artifact(root: Path, *, strategy: str = STRATEGY) -> tuple[Path, Path]:
    output = root / "output"
    models = root / "models"
    (output / "lib").mkdir(parents=True)
    models.mkdir()
    (output / "main.py").write_text(strategy, encoding="utf-8")
    (output / "README.md").write_text("contract\n", encoding="utf-8")
    (output / "lib" / "features.py").write_text(HELPER, encoding="utf-8")
    (models / "weights.npy").write_bytes(b"\x93NUMPY parameters")
    return output, models


def test_two_revisions_sharing_a_file_store_its_bytes_once(tmp_path: Path):
    """Retention is only affordable if a candidate costs the files it changed.

    Two candidates of one round differ in a single module; every other file of
    the second revision must be the first revision's bytes, not a second copy.
    """

    output, models = _working_artifact(tmp_path / "work")
    store = FilesystemArtifactStore(tmp_path / "store")
    first = store.create_revision(output, models_path=models, revision_id="revision_a")
    (output / "lib" / "features.py").write_text(HELPER + "VALUE_40 = 40\n", encoding="utf-8")
    second = store.create_revision(
        output, models_path=models, revision_id="revision_b", parent_revision_id="revision_a"
    )

    shared = ("output/main.py", "output/README.md", "models/weights.npy")
    for relpath in shared:
        left = (first.output_path.parent / relpath).stat()
        right = (second.output_path.parent / relpath).stat()
        assert left.st_ino == right.st_ino, relpath
    changed = "output/lib/features.py"
    assert (first.output_path.parent / changed).stat().st_ino != (
        second.output_path.parent / changed
    ).stat().st_ino

    # One object per distinct file: four files, one of them in two versions.
    objects = sorted(path.name for path in (tmp_path / "store" / "objects").iterdir())
    assert len(objects) == 5
    assert all(len(name) == 64 for name in objects)

    # Sharing must not weaken immutability: every revision path stays read-only.
    for root in (first.output_path.parent, second.output_path.parent):
        for path in (root, *root.rglob("*")):
            assert path.stat().st_mode & 0o222 == 0, path
            assert not path.is_symlink()


def test_a_revision_manifest_records_its_parent_and_every_file_digest(tmp_path: Path):
    output, models = _working_artifact(tmp_path / "work")
    store = FilesystemArtifactStore(tmp_path / "store")
    parent = store.create_revision(output, models_path=models, revision_id="revision_parent")
    child = store.create_revision(
        output,
        models_path=models,
        revision_id="revision_child",
        parent_revision_id=parent.revision_id,
    )

    manifest = store.revision_manifest(child.revision_id)
    assert manifest["parent_revision_id"] == "revision_parent"
    assert manifest["fingerprint"] == child.fingerprint == artifact_fingerprint(output, models)
    assert manifest["created_at"]
    assert manifest["layout"] == "objects"
    assert [entry["path"] for entry in manifest["files"]] == [
        "output/README.md",
        "output/lib/features.py",
        "output/main.py",
        "models/weights.npy",
    ]
    # Each digest addresses the object the tree is linked to.
    for entry in manifest["files"]:
        object_path = tmp_path / "store" / "objects" / str(entry["sha256"])
        assert object_path.stat().st_ino == (
            child.output_path.parent / str(entry["path"])
        ).stat().st_ino
        assert entry["size"] == object_path.stat().st_size

    assert store.revision_manifest(parent.revision_id)["parent_revision_id"] is None
    assert store.revision_ids() == ["revision_child", "revision_parent"]


def test_recorded_revisions_survive_the_end_of_session_prune(tmp_path: Path):
    """The arm's history is what the session leaves behind, so recording the
    session must not take the validated artifacts with it. Only frozen
    artifacts no ledger record still references are pruned."""

    output, models = _working_artifact(tmp_path / "work")
    store = FilesystemArtifactStore(tmp_path / "store")
    store.create_revision(output, models_path=models, revision_id="revision_a")
    (output / "main.py").write_text(STRATEGY + "# tuned\n", encoding="utf-8")
    store.create_revision(
        output, models_path=models, revision_id="revision_b", parent_revision_id="revision_a"
    )
    for artifact_id in ("artifact_keep", "artifact_drop"):
        store.freeze_revision(
            "revision_b",
            artifact_id=artifact_id,
            experiment_id="experiment_001",
            epoch_id="research",
            fold_id="research",
            run_id="run_001",
            step_id="step_001",
        )

    store.prune_superseded_frozen(keep_frozen_ids=("artifact_keep",))

    assert store.revision_ids() == ["revision_a", "revision_b"]
    assert store.revision("revision_a").output_path.joinpath("main.py").read_text(
        encoding="utf-8"
    ) == STRATEGY
    assert store.revision("revision_b").output_path.joinpath("main.py").read_text(
        encoding="utf-8"
    ) == STRATEGY + "# tuned\n"
    assert store.frozen("artifact_keep", experiment_id="experiment_001").artifact_id == "artifact_keep"
    assert not (tmp_path / "store" / "frozen" / "artifact_drop").exists()


def test_freezing_a_shared_revision_copies_the_bytes_it_locks(tmp_path: Path):
    """The frozen artifact outlives the store's sharing: it must be its own
    copy, or unlocking it for a restore would unlock every revision that shares
    those bytes."""

    output, models = _working_artifact(tmp_path / "work")
    store = FilesystemArtifactStore(tmp_path / "store")
    revision = store.create_revision(output, models_path=models, revision_id="revision_a")
    frozen = store.freeze_revision(
        revision.revision_id,
        artifact_id="artifact_001",
        experiment_id="experiment_001",
        epoch_id="research",
        fold_id="research",
        run_id="run_001",
        step_id="step_001",
    )

    assert (Path(frozen.path) / "main.py").read_text(encoding="utf-8") == STRATEGY
    assert (Path(frozen.model_path) / "weights.npy").read_bytes() == b"\x93NUMPY parameters"
    assert artifact_fingerprint(frozen.path, frozen.model_path) == revision.fingerprint
    assert (Path(frozen.path) / "main.py").stat().st_ino != (
        revision.output_path / "main.py"
    ).stat().st_ino


def test_discarding_a_revision_leaves_the_bytes_it_shared_read_only(tmp_path: Path):
    """A commit that fails its fingerprint check discards its revision. That
    revision's files are the other revisions' files, so the discard must unlink
    them without making the shared bytes writable."""

    output, models = _working_artifact(tmp_path / "work")
    store = FilesystemArtifactStore(tmp_path / "store")
    kept = store.create_revision(output, models_path=models, revision_id="revision_keep")
    store.create_revision(
        output, models_path=models, revision_id="revision_drop", parent_revision_id="revision_keep"
    )

    store.discard_revision("revision_drop")

    assert not (tmp_path / "store" / "revisions" / "revision_drop").exists()
    for path in (kept.output_path, *kept.output_path.rglob("*")):
        assert path.stat().st_mode & 0o222 == 0, path
    assert (kept.output_path / "main.py").read_text(encoding="utf-8") == STRATEGY
    for path in (tmp_path / "store" / "objects").iterdir():
        assert stat.S_IMODE(path.stat().st_mode) & 0o222 == 0, path


def test_a_revision_written_before_the_object_store_stays_readable(tmp_path: Path):
    """The arms that were running when the store changed left plain copied
    revision directories behind. They keep their bytes and their digests; what
    they cannot have is a lineage that was never recorded."""

    output, models = _working_artifact(tmp_path / "work")
    store = FilesystemArtifactStore(tmp_path / "store")
    legacy = tmp_path / "store" / "revisions" / "revision_legacy"
    copy_artifact(output, legacy / "output")
    copy_model_artifacts(models, legacy / "models")
    chmod_tree(legacy, file_mode=0o444, dir_mode=0o555)
    (output / "main.py").write_text(STRATEGY + "# newer\n", encoding="utf-8")
    store.create_revision(output, models_path=models, revision_id="revision_modern")

    assert not (legacy / REVISION_MANIFEST_FILE).exists()
    manifest = store.revision_manifest("revision_legacy")
    assert manifest["layout"] == "legacy"
    assert manifest["parent_revision_id"] is None and manifest["created_at"] is None
    assert [entry["path"] for entry in manifest["files"]] == [
        "output/README.md",
        "output/lib/features.py",
        "output/main.py",
        "models/weights.npy",
    ]
    assert store.revision_ids() == ["revision_legacy", "revision_modern"]
    assert store.revision("revision_legacy").output_path.joinpath("main.py").read_text(
        encoding="utf-8"
    ) == STRATEGY


def test_an_unknown_revision_is_named_rather_than_guessed(tmp_path: Path):
    store = FilesystemArtifactStore(tmp_path / "store")
    with pytest.raises(KeyError, match="unknown artifact revision"):
        store.revision_manifest("revision_missing")
