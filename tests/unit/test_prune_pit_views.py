"""The PIT-view prune: which arms it may touch, what it counts, what it deletes.

Real trees in a temporary repository: arms written by ``build_arm`` exactly as
the pipeline leaves them at each stage, and a ``pit_views/`` that shares one
file with a seed by hard link, as a seeded arm does.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from scripts.experiments import prune_pit_views as prune_tool
from tests.unit.webui_research_arm import build_arm

PAYLOAD = b"x" * 65536


def _views(directory: Path, seed: Path) -> dict[str, Path]:
    """A pit_views tree: a seed view linked in, a read-only view of the arm's
    own whose file the slot store shares inside the tree, and a stash shard."""

    views = directory / "pit_views"
    shared = views / "decision" / "20250630T235959+0800" / "daily.parquet"
    shared.parent.mkdir(parents=True)
    os.link(seed, shared)
    own = views / "replay" / "20250701_20260630_x" / "daily.parquet"
    own.parent.mkdir(parents=True)
    own.write_bytes(PAYLOAD)
    phase = views / "replay" / "heldout" / "20250701_20260630_x"
    phase.mkdir(parents=True)
    os.link(own, phase / "daily.parquet")
    shard = views / "asof_stash" / "decision" / "part-000001.parquet"
    shard.parent.mkdir(parents=True)
    shard.write_bytes(PAYLOAD)
    for frozen in (shared.parent, own.parent, phase):
        frozen.chmod(0o555)
    return {"views": views, "shared": shared, "own": own, "shard": shard}


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    seed = tmp_path / "data" / "seed" / "daily.parquet"
    seed.parent.mkdir(parents=True)
    seed.write_bytes(PAYLOAD)
    experiments = tmp_path / "experiments"
    for experiment_id, stage, alive in (
        ("arm_graduated", "graduated", False),
        ("arm_no_edge", "no_deliverable", False),
        # Frozen, forward replay not yet recorded: the worker will still read
        # its F and H slots.
        ("arm_sealed", "sealed", False),
        ("arm_live", "research", True),
        ("arm_broken", "broken", False),
    ):
        _views(build_arm(experiments, experiment_id, stage, alive=alive), seed)
    sandbox = tmp_path / ".runtime" / "sandboxes" / "arm_no_edge" / "research"
    sandbox.mkdir(parents=True)
    (sandbox / "notes.txt").write_text("scratch", encoding="utf-8")
    return tmp_path


def _blocks(*paths: Path) -> int:
    return sum(path.lstat().st_blocks * 512 for path in paths)


def test_only_inodes_wholly_inside_the_tree_count(repo: Path) -> None:
    """The seed's view frees nothing; the arm's own view counts once however
    many of its links the tree holds; every directory counts."""

    tree = _views(build_arm(repo / "experiments", "arm_count", "graduated"), repo / "data/seed/daily.parquet")
    directories = [Path(root) for root, _dirs, _files in os.walk(tree["views"])]
    expected = _blocks(*directories, tree["own"], tree["shard"])
    assert tree["shared"].stat().st_nlink > 1  # the seed holds a link outside
    assert prune_tool.freed_bytes(tree["views"]) == expected


def test_a_dry_run_selects_finished_arms_and_changes_nothing(
    repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    before = sorted(str(path) for path in repo.rglob("*"))
    assert prune_tool.prune(repo, apply=False) == 0
    out = capsys.readouterr().out
    lines = {line.split(":", 1)[0]: line for line in out.splitlines()}
    assert "would free" in lines["arm_graduated"]
    assert "sandbox" in lines["arm_no_edge"]
    assert "without a verdict" in lines["arm_sealed"]
    assert "live worker" in lines["arm_live"]
    assert "'failed' is not terminal" in lines["arm_broken"]
    assert "total: 2 finished arms" in lines["total"]
    assert "3 kept, 0 refused" in lines["total"]
    assert "dry run" in lines["total"]
    assert sorted(str(path) for path in repo.rglob("*")) == before


def test_apply_deletes_the_caches_and_nothing_durable(
    repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    experiments = repo / "experiments"
    durable = {
        path: path.read_bytes()
        for arm in ("arm_graduated", "arm_no_edge")
        for name in ("ledgers", "hitl", "artifacts", "steps", ".host")
        for path in (experiments / arm / name).rglob("*")
        if path.is_file()
    }
    assert durable
    assert prune_tool.prune(repo, apply=True) == 0
    assert "freed" in capsys.readouterr().out
    for arm in ("arm_graduated", "arm_no_edge"):
        assert not (experiments / arm / "pit_views").exists()
    assert not (repo / ".runtime" / "sandboxes" / "arm_no_edge").exists()
    assert (repo / ".runtime" / "sandboxes").is_dir()
    assert {path: path.read_bytes() for path in durable} == durable
    for arm in ("arm_sealed", "arm_live", "arm_broken"):
        assert (experiments / arm / "pit_views" / "asof_stash").is_dir()
    seed = repo / "data" / "seed" / "daily.parquet"
    assert seed.read_bytes() == PAYLOAD
    assert seed.stat().st_nlink == 4  # the three kept arms' links remain
    # Idempotent: a second pass finds nothing left to free.
    assert prune_tool.prune(repo, apply=True) == 0
    assert "total: 0 finished arms" in capsys.readouterr().out


def test_an_arm_whose_state_cannot_be_read_is_refused(
    repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Unprovable is unsafe: a corrupt ledger or status keeps the views and
    the run exits non-zero, while the provable arms are still pruned."""

    experiments = repo / "experiments"
    ledger = experiments / "arm_graduated" / "ledgers" / "experiment_ledger.jsonl"
    ledger.write_text(ledger.read_text(encoding="utf-8") + "{not json\n", encoding="utf-8")
    (experiments / "arm_no_edge" / "hitl" / "status.json").write_text(
        json.dumps({"schema_version": 99, "state": "completed"}), encoding="utf-8"
    )
    assert prune_tool.prune(repo, apply=True) == 1
    err = capsys.readouterr().err
    assert "arm_graduated: refused" in err
    assert "arm_no_edge: refused" in err
    assert (experiments / "arm_graduated" / "pit_views").is_dir()
    assert (experiments / "arm_no_edge" / "pit_views").is_dir()
    assert (repo / ".runtime" / "sandboxes" / "arm_no_edge").is_dir()


def test_a_symlinked_view_tree_is_refused(repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Deleting through a link could reach a tree the arm does not own."""

    arm = repo / "experiments" / "arm_graduated"
    elsewhere = repo / "elsewhere"
    (arm / "pit_views").rename(elsewhere)
    (arm / "pit_views").symlink_to(elsewhere, target_is_directory=True)
    assert prune_tool.prune(repo, apply=True) == 1
    assert "arm_graduated: refused" in capsys.readouterr().err
    assert (elsewhere / "asof_stash").is_dir()
