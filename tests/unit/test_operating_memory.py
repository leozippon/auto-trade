"""Cross-experiment operating memory: library, verdict gate, mount, index, guard.

Two tiers reach a session read-only. The curated tier is repository content a
researcher promoted by hand, so it must satisfy the same format contract
``write_skill`` enforces. The graduated tier is what other experiments' Meta
sessions wrote, admitted only after the held-out verdict adopted that
experiment. Neither tier may be rewritten by the session that reads it.
"""

from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest

from autotrade.environment.artifacts import FilesystemArtifactStore
from autotrade.environment.llm import ScriptedLLM, ToolCall
from autotrade.environment.tools.base import ToolError
from autotrade.environment.tools.workspace import SafeWorkspace
from autotrade.pipelines.ledger import ExperimentLedger
from autotrade.pipelines.local_backend import LLMMetaLearner
from autotrade.pipelines.skills import (
    CURATED_MEMORY_SOURCE,
    DEFAULT_OPERATING_MEMORY,
    OPERATING_MEMORY_DIRNAME,
    OPERATING_MEMORY_LIBRARY,
    DeleteSkillTool,
    ExperimentSkillsStore,
    WriteSkillTool,
    build_skills_index,
    graduated_memory_sources,
    install_operating_memory,
    install_workspace_skills,
    operating_memory_entries,
    resolve_operating_memory,
    validate_skills_tree,
)
from autotrade.pipelines.worker import load_worker_options

from .test_interactive_worker_local import _agent_then, _experiment

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
) -> Path:
    """One finished experiment on disk: its skills generation and its ledger."""

    directory = root / name
    record: dict[str, object] = {
        "record_type": "fold",
        "experiment_id": name,
        "epoch_id": "epoch_001",
        "fold_id": "fold_a",
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
        heldout: dict[str, object] = {
            "record_type": "heldout",
            "experiment_id": name,
            "epoch_id": "epoch_001",
            "fold_id": "heldout_1",
            "run_id": f"run_{name}_heldout",
            # The per-period block the pipeline stamps on every held-out row.
            "verdict": (
                {"status": "graduated", "reasons": []}
                if graduated
                else {"status": "discarded", "reasons": ["excess_return <= 0"]}
            ),
        }
        if mutated:
            heldout["state_changed_during_test"] = True
        ledger.append(heldout)
    return directory


def _workspace(root: Path, **kwargs: object) -> tuple[Path, tuple[object, ...]]:
    """One session workspace with memory mounted the way a session gets it."""

    workspace = root / "workspace"
    workspace.mkdir()
    (workspace / "inputs").mkdir()
    mounted = install_operating_memory(workspace, **kwargs)  # type: ignore[arg-type]
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
    # Distilled from run traces so a later session does not rediscover them;
    # renaming one is a deliberate library change.
    assert {
        "pit-read-budget",
        "output-dir-hygiene",
        "shell-argv-usage",
        "redesign-after-second-failure",
        "workspace-path-rules",
        "subagent-delegation-pattern",
    } <= set(entries)


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


def test_a_stray_graduated_key_is_not_a_verdict(tmp_path: Path) -> None:
    """The verdict is ``verdict.status``; no writer emits a top-level flag."""

    experiments = tmp_path / "experiments"
    experiments.mkdir()
    directory = _experiment_with_skill(experiments, "legacy_flag", graduated=None)
    ExperimentLedger(directory / "ledgers" / "experiment_ledger.jsonl").append(
        {
            "record_type": "heldout",
            "experiment_id": "legacy_flag",
            "epoch_id": "epoch_001",
            "fold_id": "heldout_1",
            "run_id": "run_legacy_flag_heldout",
            "graduated": True,
        }
    )
    assert graduated_memory_sources(experiments) == ()


def test_every_heldout_period_must_graduate(tmp_path: Path) -> None:
    """One discarded period keeps the whole experiment out, as the ledger says."""

    experiments = tmp_path / "experiments"
    experiments.mkdir()
    directory = _experiment_with_skill(experiments, "split_verdict")
    ExperimentLedger(directory / "ledgers" / "experiment_ledger.jsonl").append(
        {
            "record_type": "heldout",
            "experiment_id": "split_verdict",
            "epoch_id": "epoch_001",
            "fold_id": "heldout_2",
            "run_id": "run_split_verdict_heldout_2",
            "verdict": {"status": "discarded", "reasons": ["sharpe <= 0"]},
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
    curated_entry = memory / CURATED_MEMORY_SOURCE / "pit-read-budget" / "SKILL.md"
    graduated_entry = memory / "adopted" / GRADUATED_SKILL / "SKILL.md"
    assert curated_entry.read_bytes() == (
        LIBRARY / "pit-read-budget" / "SKILL.md"
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


def test_mounting_refuses_an_unusable_request(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with pytest.raises(ValueError, match="operating_memory must be one of"):
        install_operating_memory(workspace, mode="graduated", repo_root=REPO_ROOT)
    with pytest.raises(ValueError, match="repository root"):
        install_operating_memory(workspace, mode="curated")
    assert not (workspace / OPERATING_MEMORY_DIRNAME).exists()
    install_operating_memory(workspace, mode="curated", repo_root=REPO_ROOT)
    with pytest.raises(FileExistsError, match="memory directory"):
        install_operating_memory(workspace, mode="curated", repo_root=REPO_ROOT)


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
    assert memory["pit-read-budget"]["origin"] == "curated"
    assert memory["pit-read-budget"]["source"] == CURATED_MEMORY_SOURCE
    assert (
        memory["pit-read-budget"]["path"]
        == f"memory/{CURATED_MEMORY_SOURCE}/pit-read-budget/SKILL.md"
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
    for name in ("pit-read-budget", GRADUATED_SKILL):
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


def test_a_meta_session_mounts_memory_and_records_it_in_the_run_manifest(
    tmp_path: Path,
) -> None:
    baseline = tmp_path / "baseline" / "main.py"
    baseline.parent.mkdir()
    baseline.write_text(
        "def generate_orders(context):\n    return []\n", encoding="utf-8"
    )
    experiments = tmp_path / "experiments"
    experiments.mkdir()
    _experiment_with_skill(experiments, "adopted")
    learner = LLMMetaLearner(
        llm=ScriptedLLM(
            [*_agent_then(ToolCall("finish_meta", "finish_meta", {}), roles=())]
        ),
        baseline_strategy=baseline,
        artifact_store=FilesystemArtifactStore(tmp_path / "artifacts"),
        experiment_dir=experiments / "current",
        runtime_root=tmp_path / "runtime",
        max_llm_calls=2,
        deadline_seconds=30.0,
        operating_memory="curated+graduated",
        repo_root=REPO_ROOT,
        use_docker=False,
        rebuild_enabled=False,
    )
    learner(
        {
            "run_id": "run_memory",
            "experiment_id": "current",
            "epoch_id": "epoch_002",
            "meta_learning_id": "epoch_002",
            "previous_prior": "keep the current transferable direction",
        }
    )
    collected = tmp_path / "run_memory" / "workspace"
    assert (
        collected / OPERATING_MEMORY_DIRNAME / "adopted" / GRADUATED_SKILL / "SKILL.md"
    ).is_file()
    assert (
        collected
        / OPERATING_MEMORY_DIRNAME
        / CURATED_MEMORY_SOURCE
        / "pit-read-budget"
        / "SKILL.md"
    ).is_file()
    index = json.loads(
        (collected / "inputs" / "skills_index.json").read_text(encoding="utf-8")
    )
    assert {entry["source"] for entry in index["operating_memory"]} == {
        CURATED_MEMORY_SOURCE,
        "adopted",
    }
    assert index["skills"] == []
    manifest = json.loads(
        (tmp_path / "run_memory" / "run_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["operating_memory"]["mode"] == "curated+graduated"
    assert [source["source"] for source in manifest["operating_memory"]["sources"]] == [
        CURATED_MEMORY_SOURCE,
        "adopted",
    ]
    assert manifest["operating_memory"]["sources"][1] == {
        "source": "adopted",
        "origin": "graduated",
        "entries": [GRADUATED_SKILL],
    }
    # Mounted memory is not this experiment's knowledge, so it is never part of
    # what the session publishes as its own skills generation.
    assert not (collected / "skills" / GRADUATED_SKILL).exists()


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
