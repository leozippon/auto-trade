"""Cross-experiment operating memory: library, verdict gate, mount, index, guard.

Two tiers reach a session read-only. The curated tier is repository content a
researcher promoted by hand, so it must satisfy the same format contract
``write_skill`` enforces. The graduated tier is what other experiments'
research sessions wrote, admitted only after the forward verdict graduated
that experiment. Neither tier may be rewritten by the session that reads it.
"""

from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest

from autotrade.environment.tools.base import ToolError
from autotrade.environment.tools.workspace import SafeWorkspace
from autotrade.pipelines.ledger import ExperimentLedger
from autotrade.pipelines.skills import (
    CURATED_MEMORY_SOURCE,
    DEFAULT_OPERATING_MEMORY,
    OPERATING_MEMORY_DIRNAME,
    OPERATING_MEMORY_LIBRARY,
    DeleteSkillTool,
    ExperimentSkillsStore,
    WriteSkillTool,
    build_skills_index,
    create_operating_memory_snapshot,
    ensure_operating_memory_snapshot,
    graduated_memory_sources,
    install_operating_memory,
    install_workspace_skills,
    operating_memory_entries,
    operating_memory_snapshot_path,
    read_operating_memory_snapshot,
    resolve_operating_memory,
    validate_skills_tree,
)
from autotrade.pipelines.worker import load_worker_options

from .test_interactive_worker_local import _experiment

REPO_ROOT = Path(__file__).resolve().parents[2]
LIBRARY = REPO_ROOT / OPERATING_MEMORY_LIBRARY
GRADUATED_SKILL = "same-window-parent-control"


def _experiment_with_skill(
    root: Path,
    name: str,
    *,
    graduated: bool | None = True,
    mutated: bool = False,
    skills: bool = True,
    extra_skill: str = "",
) -> Path:
    """One finished experiment on disk: its skills generation and its ledger."""

    directory = root / name
    record: dict[str, object] = {
        "record_type": "research_session",
        "experiment_id": name,
        "epoch_id": "research",
        "fold_id": "s1",
        "run_id": f"run_{name}",
    }
    if skills:
        source = directory / "artifacts" / f"run_{name}" / "workspace" / "skills"
        item = source / GRADUATED_SKILL
        item.mkdir(parents=True)
        (item / "SKILL.md").write_text(
            "# 同窗父本对照\n\n候选与父本在同一窗口各跑一次再比较。\n",
            encoding="utf-8",
        )
        if extra_skill:
            extra = source / extra_skill
            extra.mkdir()
            (extra / "SKILL.md").write_text(
                "# 另一条经验\n\n与上一条无关。\n", encoding="utf-8"
            )
        publication = ExperimentSkillsStore(directory).publish(
            source, generation_id="gen-1"
        )
        record |= {
            "skills_ref": publication.skills_ref,
            "skills_generation_id": publication.generation_id,
            **publication.stats.ledger_fields(),
        }
    ledger = ExperimentLedger(directory / "ledgers" / "experiment_ledger.jsonl")
    ledger.append(record)
    if graduated is not None:
        forward: dict[str, object] = {
            "record_type": "forward",
            "experiment_id": name,
            "epoch_id": "forward",
            "fold_id": "forward",
            "run_id": f"run_{name}_forward",
            "verdict": (
                {"status": "graduated", "reasons": []}
                if graduated
                else {"status": "discarded", "reasons": ["forward_lower_bound_not_positive"]}
            ),
        }
        if mutated:
            forward["state_changed_during_test"] = True
        ledger.append(forward)
    return directory


def _workspace(
    root: Path,
    *,
    mode: str,
    repo_root: object = None,
    experiments_root: object = None,
    experiment_id: str = "current",
) -> tuple[Path, tuple[object, ...]]:
    """One session workspace with memory mounted the way a session gets it.

    Two steps now, in the order the pipeline performs them: the experiment
    freezes its snapshot once, and the session mounts that snapshot.
    """

    experiment_dir = (
        Path(experiments_root) / experiment_id
        if experiments_root is not None
        else root / experiment_id
    )
    experiment_dir.mkdir(parents=True, exist_ok=True)
    ensure_operating_memory_snapshot(
        experiment_dir,
        mode=mode,
        repo_root=repo_root,
        experiments_root=experiments_root,
    )
    workspace = root / "workspace"
    workspace.mkdir()
    (workspace / "inputs").mkdir()
    mounted = install_operating_memory(workspace, experiment_dir)
    install_workspace_skills(
        None, workspace, index_path=workspace / "inputs" / "skills_index.json"
    )
    return workspace, mounted


def test_the_curated_library_is_a_valid_skill_tree_of_operational_entries() -> None:
    stats = validate_skills_tree(LIBRARY, require_writable=False)
    entries = operating_memory_entries(LIBRARY)
    assert stats.count == len(entries) >= 1
    for entry in build_skills_index(LIBRARY)["skills"]:
        assert entry["title"], entry["name"]
        assert entry["summary"], entry["name"]
        assert entry["bytes"] > 0, entry["name"]
    # A small set of abstract entries, each carrying knowledge no tool
    # description or prompt already states; renaming one is a deliberate
    # library change.
    assert set(entries) == {
        "artifact-hygiene",
        "grid-plateau-selection",
        "verify-own-measurements",
    }


def test_a_new_experiment_mounts_the_curated_library(tmp_path: Path) -> None:
    """Dropping a directory into the library is the whole registration, and
    the index is what the Agent reads."""

    workspace, _ = _workspace(tmp_path, mode="curated", repo_root=REPO_ROOT)
    index = json.loads(
        (workspace / "inputs" / "skills_index.json").read_text(encoding="utf-8")
    )
    mounted = {entry["name"]: entry for entry in index["operating_memory"]}
    assert set(mounted) == set(operating_memory_entries(LIBRARY))
    hygiene = mounted["artifact-hygiene"]
    assert hygiene["origin"] == "curated"
    assert hygiene["path"] == f"memory/{CURATED_MEMORY_SOURCE}/artifact-hygiene/SKILL.md"
    assert (workspace / hygiene["path"]).read_bytes() == (
        LIBRARY / "artifact-hygiene" / "SKILL.md"
    ).read_bytes()


def test_the_mode_parameter_defaults_to_both_tiers() -> None:
    assert resolve_operating_memory(None) == DEFAULT_OPERATING_MEMORY
    assert resolve_operating_memory("") == DEFAULT_OPERATING_MEMORY
    assert resolve_operating_memory(" curated ") == "curated"
    assert resolve_operating_memory("none") == "none"
    for bad in ("graduated", "all", "curated+failed", 3):
        with pytest.raises(ValueError, match="operating_memory must be one of"):
            resolve_operating_memory(bad)


def test_only_graduated_experiments_contribute_their_skills(tmp_path: Path) -> None:
    experiments = tmp_path / "experiments"
    experiments.mkdir()
    _experiment_with_skill(experiments, "adopted")
    _experiment_with_skill(experiments, "not_adopted", graduated=False)
    _experiment_with_skill(experiments, "still_running", graduated=None)
    _experiment_with_skill(experiments, "dirty_frozen_tree", mutated=True)
    _experiment_with_skill(experiments, "adopted_without_skills", skills=False)
    _experiment_with_skill(experiments, "current")
    (experiments / "not_an_experiment").mkdir()

    sources = graduated_memory_sources(experiments, exclude="current")
    assert [source.source for source in sources] == ["adopted"]
    assert sources[0].origin == "graduated"
    assert sources[0].entries == (GRADUATED_SKILL,)


def test_a_fold_era_graduated_row_is_not_a_verdict(tmp_path: Path) -> None:
    """Only a forward record's verdict graduates an experiment."""

    experiments = tmp_path / "experiments"
    experiments.mkdir()
    directory = _experiment_with_skill(experiments, "fold_era", graduated=None)
    ExperimentLedger(directory / "ledgers" / "experiment_ledger.jsonl").append(
        {
            "record_type": "heldout",
            "experiment_id": "fold_era",
            "epoch_id": "epoch_001",
            "fold_id": "heldout_1",
            "run_id": "run_fold_era_heldout",
            "verdict": {"status": "graduated", "reasons": []},
        }
    )
    assert graduated_memory_sources(experiments) == ()


def test_both_tiers_mount_read_only_with_their_provenance(tmp_path: Path) -> None:
    experiments = tmp_path / "experiments"
    experiments.mkdir()
    _experiment_with_skill(experiments, "adopted")
    workspace, mounted = _workspace(
        tmp_path,
        mode="curated+graduated",
        repo_root=REPO_ROOT,
        experiments_root=experiments,
        experiment_id="current",
    )
    memory = workspace / OPERATING_MEMORY_DIRNAME
    assert [source.source for source in mounted] == [CURATED_MEMORY_SOURCE, "adopted"]
    assert sorted(path.name for path in memory.iterdir()) == [
        "adopted",
        CURATED_MEMORY_SOURCE,
    ]
    curated_entry = memory / CURATED_MEMORY_SOURCE / "artifact-hygiene" / "SKILL.md"
    graduated_entry = memory / "adopted" / GRADUATED_SKILL / "SKILL.md"
    assert curated_entry.read_bytes() == (
        LIBRARY / "artifact-hygiene" / "SKILL.md"
    ).read_bytes()
    for path in (curated_entry, graduated_entry):
        assert not stat.S_IMODE(path.stat().st_mode) & 0o222
        assert not stat.S_IMODE(path.parent.stat().st_mode) & 0o222
        with pytest.raises(PermissionError):
            path.write_text("rewritten\n", encoding="utf-8")
    # The writable session tree is untouched by the mount.
    assert list((workspace / "skills").iterdir()) == []
    assert stat.S_IMODE((workspace / "skills").stat().st_mode) & 0o222


def test_curated_mode_leaves_other_experiments_out(tmp_path: Path) -> None:
    experiments = tmp_path / "experiments"
    experiments.mkdir()
    _experiment_with_skill(experiments, "adopted")
    workspace, mounted = _workspace(
        tmp_path,
        mode="curated",
        repo_root=REPO_ROOT,
        experiments_root=experiments,
        experiment_id="current",
    )
    assert [source.source for source in mounted] == [CURATED_MEMORY_SOURCE]
    assert [
        path.name for path in (workspace / OPERATING_MEMORY_DIRNAME).iterdir()
    ] == [CURATED_MEMORY_SOURCE]


def test_mode_none_mounts_nothing(tmp_path: Path) -> None:
    workspace, mounted = _workspace(tmp_path, mode="none", repo_root=REPO_ROOT)
    assert mounted == ()
    assert not (workspace / OPERATING_MEMORY_DIRNAME).exists()
    index = json.loads(
        (workspace / "inputs" / "skills_index.json").read_text(encoding="utf-8")
    )
    assert index["operating_memory"] == []


def test_snapshotting_and_mounting_each_refuse_an_unusable_request(
    tmp_path: Path,
) -> None:
    experiment = tmp_path / "current"
    experiment.mkdir()
    with pytest.raises(ValueError, match="operating_memory must be one of"):
        create_operating_memory_snapshot(experiment, mode="graduated", repo_root=REPO_ROOT)
    with pytest.raises(ValueError, match="repository root"):
        create_operating_memory_snapshot(experiment, mode="curated")
    # A refused snapshot writes nothing at all.
    assert not operating_memory_snapshot_path(experiment).exists()
    assert not (experiment / "artifacts").exists()

    create_operating_memory_snapshot(experiment, mode="curated", repo_root=REPO_ROOT)
    with pytest.raises(FileExistsError, match="snapshot already exists"):
        create_operating_memory_snapshot(experiment, mode="curated", repo_root=REPO_ROOT)

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    install_operating_memory(workspace, experiment)
    with pytest.raises(FileExistsError, match="memory directory"):
        install_operating_memory(workspace, experiment)
    # An experiment that never snapshotted cannot be mounted by guessing.
    other = tmp_path / "other"
    other.mkdir()
    with pytest.raises(FileNotFoundError, match="snapshot"):
        install_operating_memory(other, tmp_path / "never-created")


def test_an_experiment_may_not_take_the_reserved_curated_name(tmp_path: Path) -> None:
    experiments = tmp_path / "experiments"
    experiments.mkdir()
    _experiment_with_skill(experiments, CURATED_MEMORY_SOURCE)
    with pytest.raises(ValueError, match="reserved"):
        graduated_memory_sources(experiments)


def test_the_index_lists_every_source_tagged_by_origin(tmp_path: Path) -> None:
    experiments = tmp_path / "experiments"
    experiments.mkdir()
    _experiment_with_skill(experiments, "adopted")
    workspace, _ = _workspace(
        tmp_path,
        mode="curated+graduated",
        repo_root=REPO_ROOT,
        experiments_root=experiments,
        experiment_id="current",
    )
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
    memory = {entry["name"]: entry for entry in index["operating_memory"]}
    assert memory["artifact-hygiene"]["origin"] == "curated"
    assert memory["artifact-hygiene"]["source"] == CURATED_MEMORY_SOURCE
    assert (
        memory["artifact-hygiene"]["path"]
        == f"memory/{CURATED_MEMORY_SOURCE}/artifact-hygiene/SKILL.md"
    )
    assert memory[GRADUATED_SKILL]["origin"] == "graduated"
    assert memory[GRADUATED_SKILL]["source"] == "adopted"
    assert memory[GRADUATED_SKILL]["title"]
    # count/files/bytes stay the writable tree's ledger fields.
    assert index["count"] == 1
    assert index["files"] == 1


def test_a_session_can_neither_rewrite_nor_delete_mounted_memory(
    tmp_path: Path,
) -> None:
    experiments = tmp_path / "experiments"
    experiments.mkdir()
    _experiment_with_skill(experiments, "adopted")
    workspace, _ = _workspace(
        tmp_path,
        mode="curated+graduated",
        repo_root=REPO_ROOT,
        experiments_root=experiments,
        experiment_id="current",
    )
    memory = workspace / OPERATING_MEMORY_DIRNAME
    before = {
        path: path.read_bytes() for path in memory.rglob("SKILL.md")
    }
    safe = SafeWorkspace(workspace)
    for name in ("artifact-hygiene", GRADUATED_SKILL):
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
        load_worker_options(experiment, repo_root=repo).rolling.operating_memory
        == DEFAULT_OPERATING_MEMORY
    )

    params["operating_memory"] = "none"
    params_path.write_text(json.dumps(params), encoding="utf-8")
    assert (
        load_worker_options(experiment, repo_root=repo).rolling.operating_memory
        == "none"
    )

    params["operating_memory"] = "everything"
    params_path.write_text(json.dumps(params), encoding="utf-8")
    with pytest.raises(ValueError, match="operating_memory must be one of"):
        load_worker_options(experiment, repo_root=repo)


def _repo_with_library(root: Path) -> Path:
    """A checkout with one curated entry, so the curated tier still mounts."""

    repo = root / "repo"
    item = repo / OPERATING_MEMORY_LIBRARY / "pit-read-budget"
    item.mkdir(parents=True)
    (item / "SKILL.md").write_text("# PIT 读取预算\n\n先读摘要。\n", encoding="utf-8")
    return repo


# ---- the snapshot is the experiment's own copy ----------------------------
#
# The curated library and the graduated tier keep moving; an experiment does
# not. Resolving them once at creation is what makes an experiment's own Folds
# comparable to each other.


def test_two_sessions_mount_the_same_entries_after_the_library_changes(
    tmp_path: Path,
) -> None:
    experiments = tmp_path / "experiments"
    experiments.mkdir()
    repo = _repo_with_library(tmp_path)
    experiment = experiments / "current"
    experiment.mkdir()
    record = create_operating_memory_snapshot(
        experiment, mode="curated", repo_root=repo, experiments_root=experiments
    )
    assert record["created_from"] == "creation"
    assert [entry["name"] for entry in record["entries"]] == ["pit-read-budget"]
    assert record["curated_digest"] and record["snapshot_id"]

    first = tmp_path / "first"
    first.mkdir()
    (first / "workspace").mkdir()
    install_operating_memory(first / "workspace", experiment)

    # The library moves under the experiment: a new entry, and a rewritten one.
    library = repo / OPERATING_MEMORY_LIBRARY
    added = library / "added-later"
    added.mkdir()
    (added / "SKILL.md").write_text("# 后加的\n\n正文\n", encoding="utf-8")
    (library / "pit-read-budget" / "SKILL.md").write_text(
        "# 改过的正文\n\n完全不同。\n", encoding="utf-8"
    )

    second = tmp_path / "second"
    second.mkdir()
    (second / "workspace").mkdir()
    mounted = install_operating_memory(second / "workspace", experiment)
    assert [source.entries for source in mounted] == [("pit-read-budget",)]
    body = (
        second
        / "workspace"
        / OPERATING_MEMORY_DIRNAME
        / CURATED_MEMORY_SOURCE
        / "pit-read-budget"
        / "SKILL.md"
    ).read_text(encoding="utf-8")
    assert body.startswith("# PIT 读取预算")
    assert "added-later" not in {
        path.name
        for path in (
            second / "workspace" / OPERATING_MEMORY_DIRNAME / CURATED_MEMORY_SOURCE
        ).iterdir()
    }
    # And the snapshot record still describes what was frozen, not the library.
    assert read_operating_memory_snapshot(experiment) == record


def test_an_experiment_without_a_snapshot_gets_one_on_its_first_session(
    tmp_path: Path,
) -> None:
    """Experiments created before snapshotting must keep running: the first
    session that needs memory freezes it, once, and says where it came from."""

    experiments = tmp_path / "experiments"
    experiments.mkdir()
    repo = _repo_with_library(tmp_path)
    legacy = experiments / "legacy"
    legacy.mkdir()
    assert read_operating_memory_snapshot(legacy) is None

    record = ensure_operating_memory_snapshot(
        legacy, mode="curated", repo_root=repo, experiments_root=experiments
    )
    assert record["created_from"] == "first_session"
    assert operating_memory_snapshot_path(legacy).is_file()
    # The second session reuses it rather than resolving again.
    (repo / OPERATING_MEMORY_LIBRARY / "added-later").mkdir()
    (repo / OPERATING_MEMORY_LIBRARY / "added-later" / "SKILL.md").write_text(
        "# 后加的\n\n正文\n", encoding="utf-8"
    )
    again = ensure_operating_memory_snapshot(
        legacy, mode="curated", repo_root=repo, experiments_root=experiments
    )
    assert again == record


def test_the_snapshot_copy_is_read_only_and_valid_for_the_mount(
    tmp_path: Path,
) -> None:
    experiments = tmp_path / "experiments"
    experiments.mkdir()
    _experiment_with_skill(experiments, "adopted")
    repo = _repo_with_library(tmp_path)
    experiment = experiments / "current"
    experiment.mkdir()
    create_operating_memory_snapshot(
        experiment,
        mode="curated+graduated",
        repo_root=repo,
        experiments_root=experiments,
    )
    root = experiment / "artifacts" / OPERATING_MEMORY_DIRNAME
    assert sorted(path.name for path in root.iterdir()) == [
        "adopted",
        CURATED_MEMORY_SOURCE,
    ]
    for source in root.iterdir():
        validate_skills_tree(source, require_writable=False)
    frozen = root / "adopted" / GRADUATED_SKILL / "SKILL.md"
    assert not stat.S_IMODE(frozen.stat().st_mode) & 0o222
    with pytest.raises(PermissionError):
        frozen.write_text("rewritten\n", encoding="utf-8")


def test_a_snapshot_with_an_unknown_schema_is_refused(tmp_path: Path) -> None:
    experiment = tmp_path / "current"
    experiment.mkdir()
    create_operating_memory_snapshot(experiment, mode="none")
    path = operating_memory_snapshot_path(experiment)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["mode"] == "none" and payload["entries"] == []
    payload["schema_version"] = 99
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="schema_version"):
        read_operating_memory_snapshot(experiment)
