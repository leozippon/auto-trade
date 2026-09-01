"""Shared experiment skills are bounded, ledger-reachable, and never formal output."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from autotrade.environment.tools import SafeWorkspace, ToolRegistry, WriteFileTool
from autotrade.pipelines.local_backend import _assert_skills_absent_from_formal
from autotrade.pipelines.skills import (
    DeleteSkillTool,
    ExperimentSkillsStore,
    SkillsSnapshot,
    WriteSkillTool,
    build_skills_index,
    install_workspace_skills,
    latest_skills_snapshot,
    resolve_collected_skills_source,
    skills_trees_equal,
    validate_skills_tree,
)


def _workspace(tmp_path: Path) -> Path:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    install_workspace_skills(None, workspace)
    return workspace


def _write_skill(root: Path, name: str = "schema-notes", body: str | None = None) -> None:
    item = root / name
    item.mkdir(parents=True)
    (item / "SKILL.md").write_text(
        body or "# Schema Notes\n\nRead schema before selecting columns.\n",
        encoding="utf-8",
    )


def test_skill_tools_write_index_and_delete_without_exposing_body(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    registry = ToolRegistry(
        [WriteSkillTool(SafeWorkspace(workspace)), DeleteSkillTool(SafeWorkspace(workspace))]
    )
    created = registry.invoke(
        "write_skill",
        {
            "name": "schema-notes",
            "path": "SKILL.md",
            "content": "# Schema Notes\n\nRead schema before selecting columns.\n",
        },
    )
    assert created.ok, created.error
    script = registry.invoke(
        "write_skill",
        {
            "name": "schema-notes",
            "path": "scripts/check.py",
            "content": "print('manual only')\n",
        },
    )
    assert script.ok, script.error

    index = build_skills_index(workspace / "skills")
    assert index["count"] == 1
    assert index["files"] == 2
    entry = index["skills"][0]
    assert entry["name"] == "schema-notes"
    assert entry["title"] == "Schema Notes"
    assert entry["summary"] == "Read schema before selecting columns."
    assert entry["path"] == "skills/schema-notes/SKILL.md"
    assert {item["path"] for item in entry["files"]} == {
        "skills/schema-notes/SKILL.md",
        "skills/schema-notes/scripts/check.py",
    }
    rendered = json.dumps(index, ensure_ascii=False)
    assert "manual only" not in rendered
    assert "sha" not in rendered.lower()

    deleted = registry.invoke("delete_skill", {"name": "schema-notes"})
    assert deleted.ok, deleted.error
    assert deleted.value["skills_count"] == 0
    assert not (workspace / "skills" / "schema-notes").exists()


def test_generic_writer_cannot_bypass_dedicated_skill_tools(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    result = ToolRegistry([WriteFileTool(SafeWorkspace(workspace))]).invoke(
        "write_file",
        {"path": "skills/bypass/SKILL.md", "content": "# bypass\n"},
    )
    assert not result.ok
    assert result.value["error_type"] == "readonly"
    assert "write_skill" in result.error


@pytest.mark.parametrize(
    ("name", "path"),
    [
        ("BadName", "SKILL.md"),
        ("bad/name", "SKILL.md"),
        ("held-out-sharpe-1-2", "SKILL.md"),
        ("valid-name", "references/test-sharpe-1-2.md"),
        ("valid-name", "../SKILL.md"),
        ("valid-name", "/tmp/SKILL.md"),
        ("valid-name", ".hidden.md"),
        ("valid-name", "notes.md"),
        ("valid-name", "scripts/code.exe"),
    ],
)
def test_write_skill_rejects_invalid_names_and_paths(
    tmp_path: Path, name: str, path: str
) -> None:
    workspace = _workspace(tmp_path)
    result = ToolRegistry([WriteSkillTool(SafeWorkspace(workspace))]).invoke(
        "write_skill", {"name": name, "path": path, "content": "# note\n"}
    )
    assert not result.ok
    assert result.value["error_type"] == "skill_policy"


def test_write_skill_states_and_enforces_the_front_matter_contract(
    tmp_path: Path,
) -> None:
    """The one recognized key is in the tool's own description and in the
    refusal, so a SKILL.md does not have to be written twice to discover it."""

    description = WriteSkillTool.spec.description
    assert "supersedes: <source>/<name>" in description
    assert "no front matter" in description
    workspace = _workspace(tmp_path)
    registry = ToolRegistry([WriteSkillTool(SafeWorkspace(workspace))])
    refused = registry.invoke(
        "write_skill",
        {
            "name": "schema-notes",
            "path": "SKILL.md",
            "content": "---\nname: schema-notes\ndescription: x\n---\n# Schema Notes\n",
        },
    )
    assert not refused.ok
    assert refused.value["error_type"] == "skill_policy"
    assert "supersedes" in refused.error
    unclosed = registry.invoke(
        "write_skill",
        {
            "name": "schema-notes",
            "path": "SKILL.md",
            "content": "---\nsupersedes: curated/pit-read-budget\n",
        },
    )
    assert not unclosed.ok
    assert "not closed" in unclosed.error
    plain = registry.invoke(
        "write_skill",
        {"name": "schema-notes", "path": "SKILL.md", "content": "# Schema Notes\n\n正文\n"},
    )
    assert plain.ok, plain.error


def test_skill_content_uses_prior_boundary_but_allows_dates_and_security_knowledge(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    registry = ToolRegistry([WriteSkillTool(SafeWorkspace(workspace))])
    allowed = registry.invoke(
        "write_skill",
        {
            "name": "market-conventions",
            "path": "SKILL.md",
            "content": (
                "# Market conventions\n\n"
                "2024 年复核过字段命名；000001.SZ 是证券代码示例，"
                "一般日期和证券知识可以保留。\n"
            ),
        },
    )
    assert allowed.ok, allowed.error
    heldout = registry.invoke(
        "write_skill",
        {
            "name": "market-conventions",
            "path": "SKILL.md",
            "content": "# leak\n\nHeld-out sharpe 1.2。\n",
        },
    )
    assert not heldout.ok
    assert "Held-out" in heldout.error
    selection = registry.invoke(
        "write_skill",
        {
            "name": "market-conventions",
            "path": "SKILL.md",
            "content": "# leak\n\n根据 Test 选择动量因子。\n",
        },
    )
    assert not selection.ok
    assert "choose" in selection.error
    pure_boundary = registry.invoke(
        "write_skill",
        {
            "name": "visibility-boundary",
            "path": "SKILL.md",
            "content": "# Boundary\n\n不得使用 Test/Held-out。\n",
        },
    )
    assert pure_boundary.ok, pure_boundary.error
    disguised = registry.invoke(
        "write_skill",
        {
            "name": "visibility-boundary",
            "path": "SKILL.md",
            "content": (
                "# Leak\n\n不要忽略 Held-out sharpe 1.2，"
                "根据 Test 选择动量。\n"
            ),
        },
    )
    assert not disguised.ok
    assert "Held-out" in disguised.error


def test_tree_validation_rejects_missing_skill_hidden_symlink_binary_and_limits(
    tmp_path: Path,
) -> None:
    root = tmp_path / "skills"
    root.mkdir()
    missing = root / "missing-skill"
    missing.mkdir()
    with pytest.raises(ValueError, match="missing SKILL.md"):
        validate_skills_tree(root)
    missing.rmdir()

    _write_skill(root)
    (root / "schema-notes" / ".hidden.txt").write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="hidden"):
        validate_skills_tree(root)
    (root / "schema-notes" / ".hidden.txt").unlink()

    outside = tmp_path / "outside.md"
    outside.write_text("outside", encoding="utf-8")
    (root / "schema-notes" / "references").mkdir()
    (root / "schema-notes" / "references" / "link.md").symlink_to(outside)
    with pytest.raises(ValueError, match="symlink"):
        validate_skills_tree(root)
    (root / "schema-notes" / "references" / "link.md").unlink()

    (root / "schema-notes" / "references" / "binary.txt").write_bytes(b"\xff")
    with pytest.raises(ValueError, match="UTF-8"):
        validate_skills_tree(root)
    (root / "schema-notes" / "references" / "binary.txt").unlink()

    (root / "schema-notes" / "references" / "large.txt").write_text(
        "x" * (64 * 1024 + 1), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="65536 bytes"):
        validate_skills_tree(root)


def test_store_publishes_immutable_generations_and_compares_file_bytes(
    tmp_path: Path,
) -> None:
    experiment = tmp_path / "experiment"
    source = tmp_path / "source" / "skills"
    source.mkdir(parents=True)
    _write_skill(source)
    store = ExperimentSkillsStore(experiment)

    first = store.publish(source, generation_id="gen_1")
    assert first.published is True
    assert first.skills_ref == "artifacts/skills/generations/gen_1/skills"
    assert first.stats.count == 1
    first_root = experiment / first.skills_ref
    assert first_root.is_dir()
    assert not stat_is_writable(first_root)
    assert not stat_is_writable(first_root / "schema-notes" / "SKILL.md")

    current = SkillsSnapshot(
        first.skills_ref, first.generation_id, first.stats, first_root
    )
    unchanged = store.publish(source, generation_id="gen_2", previous=current)
    assert unchanged.published is False
    assert unchanged.generation_id == "gen_1"
    assert not (store.root / "generations" / "gen_2").exists()

    (source / "schema-notes" / "SKILL.md").write_text(
        "# Schema Notes\n\nRead only confirmed columns.\n", encoding="utf-8"
    )
    assert not skills_trees_equal(source, first_root)
    second = store.publish(source, generation_id="gen_2", previous=current)
    assert second.published is True
    assert second.generation_id == "gen_2"


def stat_is_writable(path: Path) -> bool:
    return bool(path.stat().st_mode & 0o222)


def test_ledger_is_the_only_reachability_point_and_old_rows_mean_empty(
    tmp_path: Path,
) -> None:
    experiment = tmp_path / "experiment"
    source = tmp_path / "source" / "skills"
    source.mkdir(parents=True)
    _write_skill(source)
    published = ExperimentSkillsStore(experiment).publish(source, generation_id="gen_1")
    row = {
        "record_type": "fold",
        "skills_ref": published.skills_ref,
        "skills_generation_id": published.generation_id,
    }
    snapshot = latest_skills_snapshot([row], experiment_dir=experiment)
    assert snapshot.generation_id == "gen_1"
    assert snapshot.stats.count == 1
    assert latest_skills_snapshot([], experiment_dir=experiment) == SkillsSnapshot()
    # Rolling back to a legacy successful row clears reachability even though
    # the orphan generation remains available for audit.
    assert latest_skills_snapshot(
        [row, {"record_type": "meta_learning", "prior": "legacy"}],
        experiment_dir=experiment,
    ) == SkillsSnapshot()
    with pytest.raises(ValueError, match="non-empty skills metadata"):
        latest_skills_snapshot(
            [
                {
                    "record_type": "fold",
                    "skills_generation_id": "ghost",
                    "skills_count": 1,
                }
            ],
            experiment_dir=experiment,
        )
    with pytest.raises(ValueError, match="skills_count does not match"):
        latest_skills_snapshot(
            [
                {
                    **row,
                    "skills_count": 2,
                    "skills_files": 1,
                    "skills_bytes": published.stats.bytes,
                }
            ],
            experiment_dir=experiment,
        )
    published_file = experiment / published.skills_ref / "schema-notes" / "SKILL.md"
    os.chmod(published_file, 0o644)
    with pytest.raises(ValueError, match="published skills path is writable"):
        latest_skills_snapshot([row], experiment_dir=experiment)
    os.chmod(published_file, 0o444)
    with pytest.raises(ValueError, match="experiment-relative"):
        latest_skills_snapshot(
            [{"record_type": "fold", "skills_ref": str(snapshot.root)}],
            experiment_dir=experiment,
        )
    with pytest.raises(ValueError, match="does not match"):
        latest_skills_snapshot(
            [
                {
                    "record_type": "fold",
                    "skills_ref": published.skills_ref,
                    "skills_generation_id": "different-generation",
                }
            ],
            experiment_dir=experiment,
        )


def test_collected_source_must_be_this_runs_workspace_copy(tmp_path: Path) -> None:
    experiment = tmp_path / "experiment"
    expected = experiment / "artifacts" / "run_a" / "workspace" / "skills"
    expected.mkdir(parents=True)
    _write_skill(expected)
    assert resolve_collected_skills_source(experiment, "run_a", expected) == expected.resolve()

    wrong = experiment / "artifacts" / "run_b" / "workspace" / "skills"
    wrong.mkdir(parents=True)
    _write_skill(wrong)
    with pytest.raises(ValueError, match="this run"):
        resolve_collected_skills_source(experiment, "run_a", wrong)


def test_formal_output_rejects_embedded_skill_tree(tmp_path: Path) -> None:
    output = tmp_path / "output"
    models = tmp_path / "models"
    output.mkdir()
    models.mkdir()
    _assert_skills_absent_from_formal(output, models)
    copied = output / "skills" / "schema-notes"
    copied.mkdir(parents=True)
    (copied / "SKILL.md").write_text("# copied\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"formal output: output/skills") as error:
        _assert_skills_absent_from_formal(output, models)
    assert str(tmp_path) not in str(error.value)


def test_install_workspace_skills_copies_bytes_and_writes_index(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    _write_skill(source)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    stats = install_workspace_skills(source, workspace)
    assert stats.count == 1
    assert (workspace / "skills" / "schema-notes" / "SKILL.md").read_bytes() == (
        source / "schema-notes" / "SKILL.md"
    ).read_bytes()
    index = json.loads((workspace / "inputs" / "skills_index.json").read_text())
    assert index["count"] == 1
    assert not any("sha" in key.lower() for key in index)
    assert os.access(workspace / "skills", os.W_OK)
