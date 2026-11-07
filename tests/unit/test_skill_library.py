"""Skill library: library, snapshot, mount, index and guard.

The curated library reaches a session read-only. It is repository content a
researcher wrote or copied in by hand, so it must satisfy the same format
contract ``write_skill`` enforces, and the session that reads it may not
rewrite it. An experiment freezes the library once, at creation.
"""

from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest

from autotrade.environment.tools.base import ToolError
from autotrade.environment.tools.workspace import SafeWorkspace
from autotrade.pipelines.skills import (
    CURATED_MEMORY_SOURCE,
    DEFAULT_SKILL_LIBRARY,
    SKILL_LIBRARY_DIRNAME,
    SKILL_LIBRARY_PATH,
    DeleteSkillTool,
    WriteSkillTool,
    build_skills_index,
    create_skill_library_snapshot,
    ensure_skill_library_snapshot,
    install_skill_library,
    install_workspace_skills,
    read_skill_library_snapshot,
    resolve_skill_library,
    skill_library_entries,
    skill_library_snapshot_path,
    snapshot_memory_sources,
    validate_skills_tree,
)
from autotrade.pipelines.worker import load_worker_options

from .test_interactive_worker_local import _experiment

REPO_ROOT = Path(__file__).resolve().parents[2]
LIBRARY = REPO_ROOT / SKILL_LIBRARY_PATH


def _workspace(
    root: Path,
    *,
    mode: str,
    repo_root: object = None,
    experiment_id: str = "current",
) -> tuple[Path, tuple[object, ...]]:
    """One session workspace with memory mounted the way a session gets it.

    Two steps, in the order the pipeline performs them: the experiment
    freezes its snapshot once, and the session mounts that snapshot.
    """

    experiment_dir = root / experiment_id
    experiment_dir.mkdir(parents=True, exist_ok=True)
    ensure_skill_library_snapshot(experiment_dir, mode=mode, repo_root=repo_root)
    workspace = root / "workspace"
    workspace.mkdir()
    (workspace / "inputs").mkdir()
    mounted = install_skill_library(workspace, experiment_dir)
    install_workspace_skills(
        None, workspace, index_path=workspace / "inputs" / "skills_index.json"
    )
    return workspace, mounted


def test_the_curated_library_is_a_valid_skill_tree_of_operational_entries() -> None:
    stats = validate_skills_tree(LIBRARY, require_writable=False)
    entries = skill_library_entries(LIBRARY)
    assert stats.count == len(entries) >= 1
    for entry in build_skills_index(LIBRARY)["skills"]:
        assert entry["title"], entry["name"]
        assert entry["summary"], entry["name"]
        assert entry["bytes"] > 0, entry["name"]
    # A small set of abstract entries, each carrying knowledge no tool
    # description or prompt already states; renaming one is a deliberate
    # library change.
    assert set(entries) == {
        "offline-readings",
        "refit-snapshot-rescore",
        "residual-label-benchmark",
        "validation-readings",
    }


def test_a_new_experiment_mounts_the_curated_library(tmp_path: Path) -> None:
    """Dropping a directory into the library is the whole registration, and
    the index is what the Agent reads."""

    workspace, _ = _workspace(tmp_path, mode="curated", repo_root=REPO_ROOT)
    index = json.loads(
        (workspace / "inputs" / "skills_index.json").read_text(encoding="utf-8")
    )
    mounted = {entry["name"]: entry for entry in index["skill_library"]}
    assert set(mounted) == set(skill_library_entries(LIBRARY))
    entry = mounted["offline-readings"]
    assert entry["origin"] == "curated"
    assert entry["path"] == f"memory/{CURATED_MEMORY_SOURCE}/offline-readings/SKILL.md"
    assert (workspace / entry["path"]).read_bytes() == (
        LIBRARY / "offline-readings" / "SKILL.md"
    ).read_bytes()


def test_the_mode_parameter_defaults_to_the_curated_library() -> None:
    assert resolve_skill_library(None) == DEFAULT_SKILL_LIBRARY == "curated"
    assert resolve_skill_library("") == DEFAULT_SKILL_LIBRARY
    assert resolve_skill_library(" curated ") == "curated"
    assert resolve_skill_library("none") == "none"
    for bad in ("graduated", "curated+graduated", "all", "curated+failed", 3):
        with pytest.raises(ValueError, match="skill_library must be one of"):
            resolve_skill_library(bad)


def test_the_curated_library_mounts_read_only(tmp_path: Path) -> None:
    workspace, mounted = _workspace(tmp_path, mode="curated", repo_root=REPO_ROOT)
    memory = workspace / SKILL_LIBRARY_DIRNAME
    assert [source.source for source in mounted] == [CURATED_MEMORY_SOURCE]
    assert [path.name for path in memory.iterdir()] == [CURATED_MEMORY_SOURCE]
    entry = memory / CURATED_MEMORY_SOURCE / "offline-readings" / "SKILL.md"
    assert entry.read_bytes() == (LIBRARY / "offline-readings" / "SKILL.md").read_bytes()
    assert not stat.S_IMODE(entry.stat().st_mode) & 0o222
    assert not stat.S_IMODE(entry.parent.stat().st_mode) & 0o222
    with pytest.raises(PermissionError):
        entry.write_text("rewritten\n", encoding="utf-8")
    # The writable session tree is untouched by the mount.
    assert list((workspace / "skills").iterdir()) == []
    assert stat.S_IMODE((workspace / "skills").stat().st_mode) & 0o222


def test_a_snapshot_entry_that_is_not_the_curated_librarys_is_refused(tmp_path: Path) -> None:
    experiment = tmp_path / "current"
    experiment.mkdir()
    create_skill_library_snapshot(experiment, mode="curated", repo_root=_repo_with_library(tmp_path))
    record = read_skill_library_snapshot(experiment)
    assert record is not None
    assert [(source.source, source.entries) for source in snapshot_memory_sources(record, experiment)] == [
        (CURATED_MEMORY_SOURCE, ("pit-read-budget",))
    ]
    for origin, source in (("graduated", "adopted"), ("session", CURATED_MEMORY_SOURCE), ("curated", "adopted")):
        unknown = {**record, "entries": [*record["entries"], {"origin": origin, "source": source, "name": "x"}]}
        with pytest.raises(ValueError, match="snapshot entry is invalid"):
            snapshot_memory_sources(unknown, experiment)


def test_mode_none_mounts_nothing(tmp_path: Path) -> None:
    workspace, mounted = _workspace(tmp_path, mode="none", repo_root=REPO_ROOT)
    assert mounted == ()
    assert not (workspace / SKILL_LIBRARY_DIRNAME).exists()
    index = json.loads(
        (workspace / "inputs" / "skills_index.json").read_text(encoding="utf-8")
    )
    assert index["skill_library"] == []


def test_snapshotting_and_mounting_each_refuse_an_unusable_request(
    tmp_path: Path,
) -> None:
    experiment = tmp_path / "current"
    experiment.mkdir()
    with pytest.raises(ValueError, match="skill_library must be one of"):
        create_skill_library_snapshot(experiment, mode="graduated", repo_root=REPO_ROOT)
    with pytest.raises(ValueError, match="skill_library must be one of"):
        create_skill_library_snapshot(experiment, mode="curated+graduated", repo_root=REPO_ROOT)
    with pytest.raises(ValueError, match="repository root"):
        create_skill_library_snapshot(experiment, mode="curated")
    # A refused snapshot writes nothing at all.
    assert not skill_library_snapshot_path(experiment).exists()
    assert not (experiment / "artifacts").exists()

    create_skill_library_snapshot(experiment, mode="curated", repo_root=REPO_ROOT)
    with pytest.raises(FileExistsError, match="snapshot already exists"):
        create_skill_library_snapshot(experiment, mode="curated", repo_root=REPO_ROOT)

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    install_skill_library(workspace, experiment)
    with pytest.raises(FileExistsError, match="memory directory"):
        install_skill_library(workspace, experiment)
    # An experiment that never snapshotted cannot be mounted by guessing.
    other = tmp_path / "other"
    other.mkdir()
    with pytest.raises(FileNotFoundError, match="snapshot"):
        install_skill_library(other, tmp_path / "never-created")


def test_the_index_lists_every_source_tagged_by_origin(tmp_path: Path) -> None:
    workspace, _ = _workspace(tmp_path, mode="curated", repo_root=REPO_ROOT)
    write = WriteSkillTool(SafeWorkspace(workspace))
    assert write.invoke(
        {
            "name": "fold-notes",
            "path": "SKILL.md",
            "content": "# Fold notes\n\nWhat this session learned.\n",
        }
    ).ok
    index = build_skills_index(workspace / "skills")
    assert [entry["name"] for entry in index["skills"]] == ["fold-notes"]
    assert [entry["origin"] for entry in index["skills"]] == ["session"]
    memory = {entry["name"]: entry for entry in index["skill_library"]}
    assert memory["offline-readings"]["origin"] == "curated"
    assert memory["offline-readings"]["source"] == CURATED_MEMORY_SOURCE
    assert (
        memory["offline-readings"]["path"]
        == f"memory/{CURATED_MEMORY_SOURCE}/offline-readings/SKILL.md"
    )
    # count/files/bytes stay the writable tree's ledger fields.
    assert index["count"] == 1
    assert index["files"] == 1
    # Only the curated library mounts beside the skills.
    stray = tmp_path / "stray"
    (stray / "skills").mkdir(parents=True)
    (stray / SKILL_LIBRARY_DIRNAME / "adopted").mkdir(parents=True)
    with pytest.raises(ValueError, match="unknown skill library source: adopted"):
        build_skills_index(stray / "skills")


def test_a_session_can_neither_rewrite_nor_delete_mounted_memory(
    tmp_path: Path,
) -> None:
    workspace, _ = _workspace(tmp_path, mode="curated", repo_root=REPO_ROOT)
    memory = workspace / SKILL_LIBRARY_DIRNAME
    before = {
        path: path.read_bytes() for path in memory.rglob("SKILL.md")
    }
    safe = SafeWorkspace(workspace)
    for name in ("offline-readings", "validation-readings"):
        for tool, arguments in (
            (WriteSkillTool(safe), {"name": name, "path": "SKILL.md", "content": "# gone\n"}),
            (
                WriteSkillTool(safe),
                {"name": name, "path": "references/note.md", "content": "extra\n"},
            ),
            (DeleteSkillTool(safe), {"name": name}),
        ):
            with pytest.raises(ToolError, match="read-only"):
                tool.invoke(arguments)
        assert not (workspace / "skills" / name).exists()
    assert {path: path.read_bytes() for path in memory.rglob("SKILL.md")} == before
    # An unrelated name is still writable, so the guard is the memory, not the tool.
    assert WriteSkillTool(safe).invoke(
        {"name": "own-skill", "path": "SKILL.md", "content": "# Own\n\nMine.\n"}
    ).ok


def test_the_run_config_carries_the_mode(tmp_path: Path) -> None:
    repo, experiment = _experiment(tmp_path)
    params_path = experiment / "hitl" / "params.json"
    params = json.loads(params_path.read_text(encoding="utf-8"))

    assert (
        load_worker_options(experiment, repo_root=repo).experiment.skill_library
        == DEFAULT_SKILL_LIBRARY
    )

    params["skill_library"] = "none"
    params_path.write_text(json.dumps(params), encoding="utf-8")
    assert (
        load_worker_options(experiment, repo_root=repo).experiment.skill_library
        == "none"
    )

    params["skill_library"] = "everything"
    params_path.write_text(json.dumps(params), encoding="utf-8")
    with pytest.raises(ValueError, match="skill_library must be one of"):
        load_worker_options(experiment, repo_root=repo)


def _repo_with_library(root: Path) -> Path:
    """A checkout with one curated entry, so the curated tier still mounts."""

    repo = root / "repo"
    item = repo / SKILL_LIBRARY_PATH / "pit-read-budget"
    item.mkdir(parents=True)
    (item / "SKILL.md").write_text("# PIT 读取预算\n\n先读摘要。\n", encoding="utf-8")
    return repo


# ---- the snapshot is the experiment's own copy ----------------------------
#
# The curated library keeps moving; an experiment does not. Resolving it once
# at creation is what makes an experiment's own sessions comparable to each
# other.


def test_two_sessions_mount_the_same_entries_after_the_library_changes(
    tmp_path: Path,
) -> None:
    repo = _repo_with_library(tmp_path)
    experiment = tmp_path / "current"
    experiment.mkdir()
    record = create_skill_library_snapshot(experiment, mode="curated", repo_root=repo)
    assert record["created_from"] == "creation"
    assert [entry["name"] for entry in record["entries"]] == ["pit-read-budget"]
    assert record["snapshot_id"]

    first = tmp_path / "first"
    first.mkdir()
    (first / "workspace").mkdir()
    install_skill_library(first / "workspace", experiment)

    # The library moves under the experiment: a new entry, and a rewritten one.
    library = repo / SKILL_LIBRARY_PATH
    added = library / "added-later"
    added.mkdir()
    (added / "SKILL.md").write_text("# 后加的\n\n正文\n", encoding="utf-8")
    (library / "pit-read-budget" / "SKILL.md").write_text(
        "# 改过的正文\n\n完全不同。\n", encoding="utf-8"
    )

    second = tmp_path / "second"
    second.mkdir()
    (second / "workspace").mkdir()
    mounted = install_skill_library(second / "workspace", experiment)
    assert [source.entries for source in mounted] == [("pit-read-budget",)]
    body = (
        second
        / "workspace"
        / SKILL_LIBRARY_DIRNAME
        / CURATED_MEMORY_SOURCE
        / "pit-read-budget"
        / "SKILL.md"
    ).read_text(encoding="utf-8")
    assert body.startswith("# PIT 读取预算")
    assert "added-later" not in {
        path.name
        for path in (
            second / "workspace" / SKILL_LIBRARY_DIRNAME / CURATED_MEMORY_SOURCE
        ).iterdir()
    }
    # And the snapshot record still describes what was frozen, not the library.
    assert read_skill_library_snapshot(experiment) == record


def test_an_experiment_without_a_snapshot_gets_one_on_its_first_session(
    tmp_path: Path,
) -> None:
    """Experiments created before snapshotting must keep running: the first
    session that needs memory freezes it, once, and says where it came from."""

    repo = _repo_with_library(tmp_path)
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    assert read_skill_library_snapshot(legacy) is None

    record = ensure_skill_library_snapshot(legacy, mode="curated", repo_root=repo)
    assert record["created_from"] == "first_session"
    assert skill_library_snapshot_path(legacy).is_file()
    # The second session reuses it rather than resolving again.
    (repo / SKILL_LIBRARY_PATH / "added-later").mkdir()
    (repo / SKILL_LIBRARY_PATH / "added-later" / "SKILL.md").write_text(
        "# 后加的\n\n正文\n", encoding="utf-8"
    )
    again = ensure_skill_library_snapshot(legacy, mode="curated", repo_root=repo)
    assert again == record


def test_the_snapshot_copy_is_read_only_and_valid_for_the_mount(
    tmp_path: Path,
) -> None:
    repo = _repo_with_library(tmp_path)
    experiment = tmp_path / "current"
    experiment.mkdir()
    create_skill_library_snapshot(experiment, mode="curated", repo_root=repo)
    root = experiment / "artifacts" / SKILL_LIBRARY_DIRNAME
    assert [path.name for path in root.iterdir()] == [CURATED_MEMORY_SOURCE]
    validate_skills_tree(root / CURATED_MEMORY_SOURCE, require_writable=False)
    frozen = root / CURATED_MEMORY_SOURCE / "pit-read-budget" / "SKILL.md"
    assert not stat.S_IMODE(frozen.stat().st_mode) & 0o222
    with pytest.raises(PermissionError):
        frozen.write_text("rewritten\n", encoding="utf-8")


def test_a_snapshot_with_an_unknown_schema_is_refused(tmp_path: Path) -> None:
    experiment = tmp_path / "current"
    experiment.mkdir()
    create_skill_library_snapshot(experiment, mode="none")
    path = skill_library_snapshot_path(experiment)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["mode"] == "none" and payload["entries"] == []
    payload["schema_version"] = 99
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="schema_version"):
        read_skill_library_snapshot(experiment)
