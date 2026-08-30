"""Operating-memory console model: library, tier gate, mounts, curated CRUD.

The read view's invariants are what it refuses to show and what it refuses to
invent: a malformed library reports an error rather than an empty shelf, an
experiment that has not revealed its held-out results publishes no verdict here
either, admission is never recomputed beside ``skills.graduated_memory_sources``,
and nothing crosses the HTTP boundary carrying a host path or a raw run identity.

The curated writes add three of their own: an entry a session would refuse never
reaches the library, the library is never left half-written, and the writes are
accepted only from the local console — the public edge keeps the read-only page.

Every fixture is synthesized in a tempfile repo root exactly as the pipeline
writes it: a curated library under ``configs/operating_memory/``, experiment
ledgers with per-period held-out verdicts, published skills generations, and a
collected session's ``host_run_manifest.json``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from autotrade.environment.identity import AgentRefStore
from autotrade.pipelines.hitl_state import ControlState, write_control
from autotrade.pipelines.ledger import ExperimentLedger
from autotrade.pipelines.skills import (
    MAX_SKILL_CHARS,
    OPERATING_MEMORY_LIBRARY,
    ExperimentSkillsStore,
    build_skills_index,
    create_operating_memory_snapshot,
    operating_memory_snapshot_path,
    read_operating_memory_snapshot,
    validate_skills_tree,
)
from autotrade.webui import memory
from autotrade.webui.server import create_app

HOST_PATH = "/Data2/lzp/ADMCubeQuant/experiments/adopted/artifacts"
SESSION_KEY = "epoch_001/fold_2022Q1"
RUN_ID = "run_5b1d0a9c8e7f46329d1c4b7a2e6f8d03"


def _curated(repo_root: Path, name: str, body: str) -> Path:
    item = repo_root / OPERATING_MEMORY_LIBRARY / name
    item.mkdir(parents=True)
    (item / "SKILL.md").write_text(body, encoding="utf-8")
    return item


def _library(repo_root: Path) -> Path:
    _curated(
        repo_root,
        "pit-read-budget",
        "# PIT 读取预算\n\n先读摘要再读明细，避免整表扫描。\n",
    )
    _curated(
        repo_root,
        "output-dir-hygiene",
        "# 产物目录卫生\n\n只把正式代码写进 output/。\n",
    )
    return repo_root / OPERATING_MEMORY_LIBRARY


def _experiment(
    experiments_root: Path,
    name: str,
    *,
    verdict: str | None = "graduated",
    revealed: bool = True,
    skills: bool = True,
    reference: str = "",
) -> Path:
    """One finished experiment: skills generation, ledger, reveal state."""

    directory = experiments_root / name
    # Before any ledger or artifact exists: the store refuses to initialize
    # beside pre-random-ref artifacts, exactly as it does for a real experiment.
    AgentRefStore(directory)
    record: dict[str, object] = {
        "record_type": "fold",
        "experiment_id": name,
        "epoch_id": "epoch_001",
        "fold_id": "fold_2022Q1",
        "run_id": f"run_{name}",
    }
    if skills:
        source = directory / "artifacts" / f"run_{name}" / "workspace" / "skills"
        item = source / "same-window-parent-control"
        item.mkdir(parents=True)
        (item / "SKILL.md").write_text(
            "# 同窗父本对照\n\n候选与父本在同一窗口各跑一次再比较。\n", encoding="utf-8"
        )
        if reference:
            (item / "references").mkdir()
            (item / "references" / "notes.md").write_text(reference, encoding="utf-8")
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
    if verdict is not None:
        ledger.append(
            {
                "record_type": "heldout",
                "experiment_id": name,
                "epoch_id": "epoch_001",
                "fold_id": "heldout_1",
                "run_id": f"run_{name}_heldout",
                "verdict": {
                    "status": verdict,
                    "reasons": [] if verdict == "graduated" else ["sharpe <= 0"],
                },
            }
        )
    hitl = directory / "hitl"
    hitl.mkdir(parents=True, exist_ok=True)
    (hitl / "schedule.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "sessions": [
                    {
                        "kind": "fold",
                        "session_key": SESSION_KEY,
                        "epoch_id": "epoch_001",
                        "fold_id": "fold_2022Q1",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    if revealed:
        write_control(hitl / "control.json", ControlState(test_revealed=True))
    return directory


def _mounted_run(directory: Path, *, run_dir: str = RUN_ID, **overrides: object) -> Path:
    manifest = directory / "artifacts" / run_dir / "host_run_manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {
        "run_id": run_dir,
        "kind": "fold",
        "session_key": SESSION_KEY,
        "operating_memory": {
            "mode": "curated+graduated",
            "sources": [
                {
                    "source": "curated",
                    "origin": "curated",
                    "entries": ["output-dir-hygiene", "pit-read-budget"],
                },
                {
                    "source": "adopted",
                    "origin": "graduated",
                    "entries": ["same-window-parent-control"],
                },
            ],
        },
    }
    payload.update(overrides)
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    return manifest


def _params(directory: Path, **values: object) -> None:
    hitl = directory / "hitl"
    hitl.mkdir(parents=True, exist_ok=True)
    (hitl / "params.json").write_text(json.dumps(values), encoding="utf-8")


# ---- curated library ------------------------------------------------------


def test_the_curated_view_is_the_index_the_session_reads(tmp_path: Path) -> None:
    """A different title here than in ``skills_index.json`` would describe an
    entry the Agent reads by another name, so both come from one index."""

    library = _library(tmp_path)
    payload = memory.curated_library(tmp_path)
    indexed = {entry["name"]: entry for entry in build_skills_index(library)["skills"]}
    assert [entry["name"] for entry in payload["entries"]] == sorted(indexed)
    assert payload["library"] == OPERATING_MEMORY_LIBRARY
    assert payload["source"] == "curated"
    assert "error" not in payload
    for entry in payload["entries"]:
        source = indexed[entry["name"]]
        assert entry["title"] == source["title"]
        assert entry["summary"] == source["summary"]
        assert entry["bytes"] == source["bytes"] > 0
        assert entry["files"] == 1


def test_a_checkout_without_a_library_is_empty_not_broken(tmp_path: Path) -> None:
    payload = memory.curated_library(tmp_path)
    assert payload["entries"] == []
    assert "error" not in payload


def test_a_malformed_library_reports_an_error_instead_of_an_empty_shelf(
    tmp_path: Path,
) -> None:
    """An entry the mount would refuse must not read as "nothing curated"."""

    _library(tmp_path)
    (tmp_path / OPERATING_MEMORY_LIBRARY / "no-skill-md").mkdir()
    payload = memory.curated_library(tmp_path)
    assert payload["entries"] == []
    assert payload["error"].endswith("curated memory library is unreadable")
    # The underlying ValueError names the offending item, not the host tree.
    assert str(tmp_path) not in payload["error"]


def test_one_entry_body_is_served_and_nothing_else_is(tmp_path: Path) -> None:
    _library(tmp_path)
    entry = memory.curated_entry(tmp_path, tmp_path / "experiments", "pit-read-budget")
    assert entry["name"] == "pit-read-budget"
    assert entry["title"] == "PIT 读取预算"
    assert entry["content"].startswith("# PIT 读取预算")
    with pytest.raises(KeyError):
        memory.curated_entry(tmp_path, tmp_path / "experiments", "not-curated")
    for bad in ("../configs", "PIT", "a/b", ""):
        with pytest.raises(ValueError):
            memory.curated_entry(tmp_path, tmp_path / "experiments", bad)


def test_a_curated_body_is_served_verbatim_so_an_edit_round_trips(
    tmp_path: Path,
) -> None:
    """The library is researcher-authored repository content that this console
    also edits: rewriting a body for display would silently save the rewrite
    over the real line the next time it is opened and saved."""

    _library(tmp_path)
    body = f"# 路径规则\n\n只写 /mnt/agent/workspace/output，不要写 {HOST_PATH}。\n"
    _curated(tmp_path, "workspace-path-rules", body)
    assert memory.curated_entry(tmp_path, tmp_path / "experiments", "workspace-path-rules")["content"] == body
    memory.update_curated_entry(tmp_path, "workspace-path-rules", body)
    assert (
        tmp_path / OPERATING_MEMORY_LIBRARY / "workspace-path-rules" / "SKILL.md"
    ).read_text(encoding="utf-8") == body


# ---- graduated tier -------------------------------------------------------


def test_the_tier_lists_every_experiment_and_admits_only_graduated_ones(
    tmp_path: Path,
) -> None:
    experiments = tmp_path / "experiments"
    experiments.mkdir()
    _experiment(experiments, "adopted")
    _experiment(experiments, "not_adopted", verdict="discarded")
    _experiment(experiments, "still_running", verdict=None)
    _experiment(experiments, "adopted_without_skills", skills=False)

    rows = {row["experiment_id"]: row for row in memory.graduated_tier(tmp_path, experiments)["experiments"]}
    assert set(rows) == {
        "adopted",
        "not_adopted",
        "still_running",
        "adopted_without_skills",
    }
    assert rows["adopted"]["verdict"] == "graduated"
    assert rows["adopted"]["admitted"] is True
    assert rows["adopted"]["entries"] == ["same-window-parent-control"]
    assert rows["not_adopted"]["verdict"] == "discarded"
    assert rows["not_adopted"]["admitted"] is False
    # No held-out record yet: no verdict to show, and nothing to admit.
    assert rows["still_running"]["verdict"] is None
    assert rows["still_running"]["admitted"] is False
    # Graduated, but it published no skills generation, so it contributes
    # nothing — the console must not promise knowledge that will not mount.
    assert rows["adopted_without_skills"]["verdict"] == "graduated"
    assert rows["adopted_without_skills"]["admitted"] is False
    assert rows["adopted_without_skills"]["entries"] == []


def test_an_unrevealed_experiment_publishes_no_verdict_here_either(
    tmp_path: Path,
) -> None:
    """The reveal gate is the console's, not one page's: a verdict shown before
    its own experiment revealed would hand back the sealed held-out judgment."""

    experiments = tmp_path / "experiments"
    experiments.mkdir()
    _experiment(experiments, "mid_heldout", revealed=False)
    row = memory.graduated_tier(tmp_path, experiments)["experiments"][0]
    assert row["revealed"] is False
    assert row["verdict"] is None
    assert row["admitted"] is False
    assert row["entries"] == []
    # The mount itself is unaffected: this is a display gate only.
    from autotrade.pipelines.skills import graduated_memory_sources

    assert [
        source.source
        for source in graduated_memory_sources(experiments, repo_root=tmp_path)
    ] == [
        "mid_heldout"
    ]


def test_an_unreadable_ledger_becomes_a_row_and_suspends_admission(
    tmp_path: Path,
) -> None:
    """One broken ledger is also what the mount hits: the tier cannot be
    resolved, so admission is unknown rather than a "not admitted" the read
    model cannot stand behind — and the broken experiment is still listed."""

    experiments = tmp_path / "experiments"
    experiments.mkdir()
    _experiment(experiments, "adopted")
    broken = experiments / "broken" / "ledgers"
    broken.mkdir(parents=True)
    (broken / "experiment_ledger.jsonl").write_text("{not json\n", encoding="utf-8")
    payload = memory.graduated_tier(tmp_path, experiments)
    rows = {row["experiment_id"]: row for row in payload["experiments"]}
    assert payload["error"].endswith("graduated memory cannot be resolved")
    assert rows["broken"]["error"].endswith("experiment state is unreadable")
    assert str(tmp_path) not in rows["broken"]["error"]
    assert rows["adopted"]["verdict"] == "graduated"
    assert rows["adopted"]["admitted"] is None
    assert rows["adopted"]["entries"] == []


def test_a_tier_that_cannot_be_resolved_is_reported_not_hidden(tmp_path: Path) -> None:
    """``curated`` is the reserved mount name; a graduated experiment holding it
    breaks the mount for every new session, so the page must say so."""

    experiments = tmp_path / "experiments"
    experiments.mkdir()
    _experiment(experiments, "curated")
    payload = memory.graduated_tier(tmp_path, experiments)
    assert payload["error"].endswith("graduated memory cannot be resolved")
    assert str(tmp_path) not in payload["error"]
    assert [row["experiment_id"] for row in payload["experiments"]] == ["curated"]
    assert payload["experiments"][0]["admitted"] is None


# ---- the snapshot one experiment froze -------------------------------------
#
# Operating memory is fixed per experiment: the tiers are resolved once, when
# the experiment is created, and every session mounts that same copy. So the
# block answers one question for the whole experiment, and a library change
# after the fact belongs to the next experiment.


def _snapshot(
    repo_root: Path,
    experiments: Path,
    experiment_id: str,
    mode: str = "curated+graduated",
) -> dict:
    directory = experiments / experiment_id
    directory.mkdir(parents=True, exist_ok=True)
    return create_operating_memory_snapshot(
        directory, mode=mode, repo_root=repo_root, experiments_root=experiments
    )


def test_the_block_lists_the_snapshot_this_experiment_froze(tmp_path: Path) -> None:
    _library(tmp_path)
    experiments = tmp_path / "experiments"
    experiments.mkdir()
    _experiment(experiments, "adopted")
    directory = _experiment(experiments, "current", verdict=None, skills=False)
    _params(directory, operating_memory="curated+graduated")
    record = _snapshot(tmp_path, experiments, "current")
    _mounted_run(directory)

    payload = memory.experiment_memory(experiments, "current")
    assert payload["mode"] == "curated+graduated"
    assert payload["sessions_seen"] == 1
    snapshot = payload["snapshot"]
    assert snapshot["snapshot_id"] == record["snapshot_id"]
    assert snapshot["created_from"] == "creation" and snapshot["created_at"]
    assert [source["source"] for source in snapshot["sources"]] == [
        "curated",
        "adopted",
    ]
    assert [source["origin"] for source in snapshot["sources"]] == [
        "curated",
        "graduated",
    ]
    assert snapshot["sources"][0]["entries"] == [
        "output-dir-hygiene",
        "pit-read-budget",
    ]
    # Nothing per session is projected any more, and no host identity leaks.
    assert "sessions" not in payload
    assert RUN_ID not in json.dumps(payload, ensure_ascii=False)


def test_creating_an_experiment_freezes_its_operating_memory(tmp_path: Path) -> None:
    """The snapshot is taken at creation, beside the inherited parent copy, and
    a failure there leaves no half-created experiment behind."""

    from unittest.mock import patch

    from autotrade.webui.manager import ExperimentManager

    _library(tmp_path)
    experiments = tmp_path / "experiments"
    experiments.mkdir()
    manager = ExperimentManager(tmp_path, experiments)
    params = {
        "experiment_id": "exp_frozen",
        "fold_period": "quarter",
        "development_first_period": "2024Q1",
        "development_last_period": "2024Q1",
        "heldout_first_period": "2024Q2",
        "heldout_last_period": "2024Q2",
    }
    with (
        patch.object(manager, "_preflight"),
        patch.object(manager, "start_worker", return_value={"spawned": False}),
    ):
        manager.create_experiment(dict(params))
    record = read_operating_memory_snapshot(experiments / "exp_frozen")
    assert record["created_from"] == "creation"
    assert [entry["name"] for entry in record["entries"]] == [
        "output-dir-hygiene",
        "pit-read-budget",
    ]
    payload = memory.experiment_memory(experiments, "exp_frozen")
    assert payload["snapshot"]["sources"][0]["entries"] == [
        "output-dir-hygiene",
        "pit-read-budget",
    ]

    with (
        patch.object(manager, "_preflight"),
        patch.object(manager, "start_worker", return_value={"spawned": False}),
        patch(
            "autotrade.webui.manager.create_operating_memory_snapshot",
            side_effect=OSError("no space"),
        ),
        pytest.raises(OSError),
    ):
        manager.create_experiment({**params, "experiment_id": "exp_broken"})
    assert not (experiments / "exp_broken").exists()


def test_a_later_library_change_does_not_move_a_frozen_experiment(
    tmp_path: Path,
) -> None:
    _library(tmp_path)
    experiments = tmp_path / "experiments"
    experiments.mkdir()
    directory = _experiment(experiments, "current", verdict=None, skills=False)
    _params(directory, operating_memory="curated")
    _snapshot(tmp_path, experiments, "current", mode="curated")
    _curated(tmp_path, "added-later", "# 后加的\n\n正文\n")

    listed = memory.experiment_memory(experiments, "current")["snapshot"]["sources"][0]
    assert listed["entries"] == ["output-dir-hygiene", "pit-read-budget"]
    # The library itself did move on, which is exactly the distinction.
    assert len(memory.curated_library(tmp_path)["entries"]) == 3


def test_an_experiment_without_a_snapshot_says_so(tmp_path: Path) -> None:
    """An experiment that predates snapshotting has none until its next session
    takes one; the block must say that rather than invent an empty mount."""

    _library(tmp_path)
    experiments = tmp_path / "experiments"
    experiments.mkdir()
    directory = _experiment(experiments, "legacy", verdict=None, skills=False)
    _params(directory, operating_memory="curated")
    payload = memory.experiment_memory(experiments, "legacy")
    assert payload["snapshot"] is None and payload["sessions_seen"] == 0
    assert "error" not in payload


def test_an_unreadable_snapshot_is_reported_not_hidden(tmp_path: Path) -> None:
    _library(tmp_path)
    experiments = tmp_path / "experiments"
    experiments.mkdir()
    directory = _experiment(experiments, "current", verdict=None, skills=False)
    _snapshot(tmp_path, experiments, "current", mode="curated")
    operating_memory_snapshot_path(directory).write_text("{truncated", encoding="utf-8")
    payload = memory.experiment_memory(experiments, "current")
    assert payload["snapshot"] is None
    assert payload["error"].endswith("operating memory snapshot is unreadable")
    assert str(tmp_path) not in payload["error"]


def test_an_experiment_without_params_falls_back_to_the_pipeline_default(
    tmp_path: Path,
) -> None:
    experiments = tmp_path / "experiments"
    experiments.mkdir()
    _experiment(experiments, "adopted")
    payload = memory.experiment_memory(experiments, "adopted")
    assert payload["mode"] == payload["default_mode"]
    assert payload["snapshot"] is None and payload["sessions_seen"] == 0


def test_the_snapshot_entry_route_serves_what_this_experiment_holds(
    tmp_path: Path,
) -> None:
    """Deliberately the frozen copy, not the library's current text: this is
    what the experiment's sessions actually read."""

    _library(tmp_path)
    experiments = tmp_path / "experiments"
    experiments.mkdir()
    _experiment(experiments, "adopted")
    directory = _experiment(experiments, "current", verdict=None, skills=False)
    _params(directory, operating_memory="curated+graduated")
    _snapshot(tmp_path, experiments, "current")
    client = TestClient(create_app(tmp_path, experiments))

    entry = client.get("/api/experiments/current/memory/curated/pit-read-budget")
    assert entry.status_code == 200
    body = entry.json()
    assert {
        "experiment_id",
        "source",
        "origin",
        "name",
        "title",
        "bytes",
        "content",
    } <= body.keys()
    assert body["origin"] == "curated"
    assert body["content"].startswith("# PIT 读取预算")

    graduated = client.get(
        "/api/experiments/current/memory/adopted/same-window-parent-control"
    )
    assert graduated.status_code == 200 and graduated.json()["origin"] == "graduated"

    # The library moves; the frozen copy does not.
    memory.update_curated_entry(tmp_path, "pit-read-budget", "# 改过的\n\n完全不同。\n")
    assert (
        client.get("/api/experiments/current/memory/curated/pit-read-budget")
        .json()["content"]
        .startswith("# PIT 读取预算")
    )
    assert (
        client.get("/api/memory/curated/pit-read-budget")
        .json()["content"]
        .startswith("# 改过的")
    )

    assert (
        client.get("/api/experiments/current/memory/curated/never-mounted").status_code
        == 404
    )
    assert (
        client.get("/api/experiments/current/memory/curated/Not_Kebab").status_code
        == 400
    )
    assert (
        client.get("/api/experiments/adopted/memory/curated/pit-read-budget").status_code
        == 404
    )
    assert (
        client.get("/api/experiments/unknown/memory/curated/pit-read-budget").status_code
        == 404
    )


# ---- routes ---------------------------------------------------------------


def test_the_console_routes_serve_the_keys_the_page_reads(tmp_path: Path) -> None:
    _library(tmp_path)
    experiments = tmp_path / "experiments"
    experiments.mkdir()
    directory = _experiment(experiments, "adopted")
    _params(directory, operating_memory="curated+graduated")
    _mounted_run(directory)
    client = TestClient(create_app(tmp_path, experiments))

    overview = client.get("/api/memory")
    assert overview.status_code == 200
    body = overview.json()
    assert {"default_mode", "curated", "graduated", "feedback"} <= body.keys()
    assert body["graduated"]["exclusions"].endswith(".json")
    assert {"source", "library", "entries"} <= body["curated"].keys()
    assert {"name", "title", "summary", "bytes", "files"} <= body["curated"][
        "entries"
    ][0].keys()
    assert {
        "experiment_id",
        "verdict",
        "revealed",
        "admitted",
        "entries",
        "excluded",
    } <= body["graduated"]["experiments"][0].keys()

    entry = client.get("/api/memory/curated/pit-read-budget")
    assert entry.status_code == 200
    assert entry.json()["content"].startswith("# PIT 读取预算")
    assert client.get("/api/memory/curated/not-curated").status_code == 404
    assert client.get("/api/memory/curated/Not_Kebab").status_code == 400

    mounted = client.get("/api/experiments/adopted/memory")
    assert mounted.status_code == 200
    assert {
        "experiment_id",
        "mode",
        "default_mode",
        "sessions_seen",
        "snapshot",
    } <= mounted.json().keys()
    assert client.get("/api/experiments/unknown/memory").status_code == 404


# ---- curated writes -------------------------------------------------------

# What the public edge stamps on every request it proxies; the console treats
# such a request like any other client.
PUBLIC_EDGE = {"X-Forwarded-For": "203.0.113.7", "X-Forwarded-Proto": "https"}


def _entry_names(payload: dict) -> list[str]:
    return [entry["name"] for entry in payload["curated"]["entries"]]


def _staging_leftovers(repo_root: Path) -> list[str]:
    """The swap stages beside the library; a failed write must leave nothing."""

    return sorted(
        item.name
        for item in (repo_root / OPERATING_MEMORY_LIBRARY).parent.iterdir()
        if item.name.startswith(".")
    )


def test_a_created_entry_is_readable_by_the_mount_and_listed_back(
    tmp_path: Path,
) -> None:
    """The write's whole point: what lands must be what a session would mount,
    and the caller must not have to guess what the library now holds."""

    _library(tmp_path)
    result = memory.create_curated_entry(
        tmp_path, "cash-buffer-rule", "# 现金缓冲\n\n留出一天的申赎缓冲。\n"
    )
    assert result["action"] == "created"
    assert "sessions started afterwards" in str(result["note"])
    assert _entry_names(result) == [
        "cash-buffer-rule",
        "output-dir-hygiene",
        "pit-read-budget",
    ]
    library = tmp_path / OPERATING_MEMORY_LIBRARY
    # The mount's own validator, and the index the Agent reads, both accept it.
    assert validate_skills_tree(library, require_writable=False).count == 3
    indexed = {entry["name"]: entry for entry in build_skills_index(library)["skills"]}
    assert indexed["cash-buffer-rule"]["title"] == "现金缓冲"
    assert memory.curated_entry(tmp_path, tmp_path / "experiments", "cash-buffer-rule")["content"].startswith(
        "# 现金缓冲"
    )


@pytest.mark.parametrize(
    "name",
    ["Not_Kebab", "../escape", "nested/name", "", "curated"],
)
def test_a_name_the_mount_layout_refuses_never_reaches_the_library(
    tmp_path: Path, name: str
) -> None:
    """``curated`` is the mount's own directory name; the rest are the shared
    skill-name rule, which is also what stops a path escape."""

    _library(tmp_path)
    with pytest.raises(ValueError):
        memory.create_curated_entry(tmp_path, name, "# x\n")
    assert sorted(
        item.name for item in (tmp_path / OPERATING_MEMORY_LIBRARY).iterdir()
    ) == ["output-dir-hygiene", "pit-read-budget"]
    assert _staging_leftovers(tmp_path) == []


def test_a_body_the_skill_format_refuses_leaves_the_library_untouched(
    tmp_path: Path,
) -> None:
    _library(tmp_path)
    before = memory.curated_library(tmp_path)["entries"]
    with pytest.raises(ValueError):
        memory.create_curated_entry(tmp_path, "too-long", "x" * (MAX_SKILL_CHARS + 1))
    assert not (tmp_path / OPERATING_MEMORY_LIBRARY / "too-long").exists()
    assert memory.curated_library(tmp_path)["entries"] == before
    assert _staging_leftovers(tmp_path) == []


def test_creating_over_an_existing_entry_is_refused_not_silently_merged(
    tmp_path: Path,
) -> None:
    _library(tmp_path)
    with pytest.raises(FileExistsError):
        memory.create_curated_entry(tmp_path, "pit-read-budget", "# 覆盖\n")
    assert memory.curated_entry(tmp_path, tmp_path / "experiments", "pit-read-budget")["content"].startswith(
        "# PIT 读取预算"
    )


def test_an_edit_replaces_the_body_and_keeps_the_entry_s_other_files(
    tmp_path: Path,
) -> None:
    _library(tmp_path)
    item = tmp_path / OPERATING_MEMORY_LIBRARY / "pit-read-budget"
    (item / "references").mkdir()
    (item / "references" / "notes.md").write_text("附注\n", encoding="utf-8")
    result = memory.update_curated_entry(tmp_path, "pit-read-budget", "# 新正文\n\n改过。\n")
    assert result["action"] == "updated"
    assert (item / "SKILL.md").read_text(encoding="utf-8") == "# 新正文\n\n改过。\n"
    assert (item / "references" / "notes.md").read_text(encoding="utf-8") == "附注\n"
    # Two entries, three files: the edit replaced one body, not the item.
    assert validate_skills_tree(
        tmp_path / OPERATING_MEMORY_LIBRARY, require_writable=False
    ).files == 3


def test_editing_an_absent_entry_is_a_missing_entry_not_a_create(
    tmp_path: Path,
) -> None:
    _library(tmp_path)
    with pytest.raises(KeyError):
        memory.update_curated_entry(tmp_path, "not-curated", "# x\n")
    assert not (tmp_path / OPERATING_MEMORY_LIBRARY / "not-curated").exists()


def test_a_delete_removes_the_whole_item_and_says_what_it_does_not_touch(
    tmp_path: Path,
) -> None:
    """Running sessions hold their own read-only copy, so the delete is allowed
    while they run; the response is where that is said."""

    _library(tmp_path)
    result = memory.delete_curated_entry(tmp_path, "pit-read-budget")
    assert result["action"] == "deleted"
    assert "running sessions keep" in str(result["note"])
    assert _entry_names(result) == ["output-dir-hygiene"]
    assert not (tmp_path / OPERATING_MEMORY_LIBRARY / "pit-read-budget").exists()
    assert _staging_leftovers(tmp_path) == []
    with pytest.raises(KeyError):
        memory.delete_curated_entry(tmp_path, "pit-read-budget")


def test_a_library_the_mount_refuses_blocks_new_entries_but_can_be_repaired(
    tmp_path: Path,
) -> None:
    """An entry cannot be added to a library a session would already refuse —
    and deleting the offending item is how the console repairs it."""

    _library(tmp_path)
    (tmp_path / OPERATING_MEMORY_LIBRARY / "no-skill-md").mkdir()
    with pytest.raises(ValueError):
        memory.create_curated_entry(tmp_path, "cash-buffer-rule", "# 现金缓冲\n")
    result = memory.delete_curated_entry(tmp_path, "no-skill-md")
    assert "error" not in result["curated"]
    assert memory.create_curated_entry(tmp_path, "cash-buffer-rule", "# 现金缓冲\n")[
        "action"
    ] == "created"


def test_a_name_a_running_experiment_still_maintains_is_refused(
    tmp_path: Path,
) -> None:
    """Mounted memory is read-only and a session may not shadow it under its own
    name, so this entry would stop that experiment's next session from
    maintaining its own skill."""

    _library(tmp_path)
    experiments = tmp_path / "experiments"
    experiments.mkdir()
    _experiment(experiments, "adopted")
    with pytest.raises(ValueError, match="adopted"):
        memory.create_curated_entry(
            tmp_path,
            "same-window-parent-control",
            "# 同窗父本对照\n",
            experiments_root=experiments,
            live_experiments=["adopted"],
        )
    # Nothing is running: the same name is then just a normal promotion target.
    assert memory.create_curated_entry(
        tmp_path,
        "same-window-parent-control",
        "# 同窗父本对照\n",
        experiments_root=experiments,
        live_experiments=[],
    )["action"] == "created"


def test_a_candidate_body_is_readable_only_while_the_tier_admits_it(
    tmp_path: Path,
) -> None:
    """The review that precedes a promotion runs through the promotion's own
    gate, so the page can never show a candidate it could not copy."""

    _library(tmp_path)
    experiments = tmp_path / "experiments"
    experiments.mkdir()
    _experiment(experiments, "adopted")
    _experiment(experiments, "mid_heldout", revealed=False)
    entry = memory.graduated_entry(tmp_path, experiments, "adopted", "same-window-parent-control")
    assert entry["experiment_id"] == "adopted"
    assert entry["title"] == "同窗父本对照"
    assert entry["content"].startswith("# 同窗父本对照")
    assert entry["files"] == 1 and entry["bytes"] > 0
    for experiment_id, skill in (
        ("adopted", "not-a-skill"),
        ("mid_heldout", "same-window-parent-control"),
        ("unknown", "same-window-parent-control"),
    ):
        with pytest.raises(KeyError):
            memory.graduated_entry(tmp_path, experiments, experiment_id, skill)
    with pytest.raises(ValueError):
        memory.graduated_entry(tmp_path, experiments, "adopted", "Not_Kebab")


def test_a_promotion_copies_the_admitted_skill_verbatim(tmp_path: Path) -> None:
    """The whole item is what the mount would have carried, ``references/``
    included, so that is what the curated copy starts from."""

    _library(tmp_path)
    experiments = tmp_path / "experiments"
    experiments.mkdir()
    _experiment(experiments, "adopted", reference="父本对照读法\n")
    result = memory.promote_curated_entry(
        tmp_path,
        experiments,
        name="parent-control-reading",
        experiment_id="adopted",
        skill="same-window-parent-control",
    )
    assert result["action"] == "promoted"
    item = tmp_path / OPERATING_MEMORY_LIBRARY / "parent-control-reading"
    assert item.joinpath("SKILL.md").read_text(encoding="utf-8").startswith("# 同窗父本对照")
    assert item.joinpath("references/notes.md").read_text(encoding="utf-8") == "父本对照读法\n"
    # Landed writable in the repository, not as the read-only generation copy.
    assert validate_skills_tree(tmp_path / OPERATING_MEMORY_LIBRARY, require_writable=True)


def test_a_promotion_can_only_copy_from_a_candidate_the_page_offers(
    tmp_path: Path,
) -> None:
    """One admission rule, and the console's reveal gate: an unrevealed or
    discarded experiment is not a promotion source, or this route would answer a
    held-out question the page refuses to answer."""

    _library(tmp_path)
    experiments = tmp_path / "experiments"
    experiments.mkdir()
    _experiment(experiments, "adopted")
    _experiment(experiments, "not_adopted", verdict="discarded")
    _experiment(experiments, "mid_heldout", revealed=False)
    for experiment_id, skill in (
        ("adopted", "no-such-skill"),
        ("not_adopted", "same-window-parent-control"),
        ("mid_heldout", "same-window-parent-control"),
        ("unknown", "same-window-parent-control"),
    ):
        with pytest.raises(KeyError):
            memory.promote_curated_entry(
                tmp_path,
                experiments,
                name="promoted-entry",
                experiment_id=experiment_id,
                skill=skill,
            )
    assert not (tmp_path / OPERATING_MEMORY_LIBRARY / "promoted-entry").exists()
    assert _staging_leftovers(tmp_path) == []


def test_withdrawing_a_graduated_skill_takes_it_out_of_the_tier_and_back(
    tmp_path: Path,
) -> None:
    """Graduated skills are another experiment's immutable artifacts, so the
    console never edits one: it records that future sessions stop mounting it,
    and the row keeps the withdrawal visible so it can be undone."""

    _library(tmp_path)
    experiments = tmp_path / "experiments"
    experiments.mkdir()
    _experiment(experiments, "adopted")

    result = memory.exclude_graduated_skill(
        tmp_path,
        experiments,
        experiment_id="adopted",
        skill="same-window-parent-control",
        reason="已被更好的做法取代",
    )
    assert result["action"] == "excluded"
    assert "sessions started afterwards" in str(result["note"])
    row = next(
        item
        for item in result["graduated"]["experiments"]
        if item["experiment_id"] == "adopted"
    )
    assert row["admitted"] is False and row["entries"] == []
    assert row["excluded"] == [
        {
            "skill": "same-window-parent-control",
            "reason": "已被更好的做法取代",
            "excluded_at": row["excluded"][0]["excluded_at"],
        }
    ]
    # A withdrawn skill is no longer a candidate the console can read or copy.
    with pytest.raises(KeyError):
        memory.graduated_entry(
            tmp_path, experiments, "adopted", "same-window-parent-control"
        )
    with pytest.raises(KeyError):
        memory.promote_curated_entry(
            tmp_path,
            experiments,
            name="parent-control-reading",
            experiment_id="adopted",
            skill="same-window-parent-control",
        )

    restored = memory.restore_graduated_skill(
        tmp_path, experiments, experiment_id="adopted", skill="same-window-parent-control"
    )
    assert restored["action"] == "restored"
    row = next(
        item
        for item in restored["graduated"]["experiments"]
        if item["experiment_id"] == "adopted"
    )
    assert row["admitted"] is True
    assert row["entries"] == ["same-window-parent-control"] and row["excluded"] == []


def test_only_an_admitted_skill_can_be_withdrawn_and_only_once(
    tmp_path: Path,
) -> None:
    _library(tmp_path)
    experiments = tmp_path / "experiments"
    experiments.mkdir()
    _experiment(experiments, "adopted")
    _experiment(experiments, "mid_heldout", revealed=False)
    for experiment_id, skill in (
        ("adopted", "not-a-skill"),
        ("mid_heldout", "same-window-parent-control"),
        ("unknown", "same-window-parent-control"),
    ):
        with pytest.raises(KeyError):
            memory.exclude_graduated_skill(
                tmp_path, experiments, experiment_id=experiment_id, skill=skill
            )
    memory.exclude_graduated_skill(
        tmp_path, experiments, experiment_id="adopted", skill="same-window-parent-control"
    )
    with pytest.raises(FileExistsError):
        memory.exclude_graduated_skill(
            tmp_path,
            experiments,
            experiment_id="adopted",
            skill="same-window-parent-control",
        )
    with pytest.raises(KeyError):
        memory.restore_graduated_skill(
            tmp_path, experiments, experiment_id="adopted", skill="never-excluded"
        )


# ---- write routes ---------------------------------------------------------


def test_the_curated_write_routes_carry_one_crud_cycle(tmp_path: Path) -> None:
    _library(tmp_path)
    experiments = tmp_path / "experiments"
    experiments.mkdir()
    _experiment(experiments, "adopted")
    client = TestClient(create_app(tmp_path, experiments))

    created = client.post(
        "/api/memory/curated",
        json={"name": "cash-buffer-rule", "content": "# 现金缓冲\n\n留出缓冲。\n"},
    )
    assert created.status_code == 200
    assert {"name", "action", "note", "curated"} <= created.json().keys()
    assert "cash-buffer-rule" in _entry_names(created.json())

    edited = client.put(
        "/api/memory/curated/cash-buffer-rule", json={"content": "# 现金缓冲\n\n改过。\n"}
    )
    assert edited.status_code == 200
    assert client.get("/api/memory/curated/cash-buffer-rule").json()["content"].endswith(
        "改过。\n"
    )

    candidate = client.get(
        "/api/memory/graduated/adopted/same-window-parent-control"
    )
    assert candidate.status_code == 200
    assert {"experiment_id", "name", "title", "summary", "bytes", "files", "content"} <= (
        candidate.json().keys()
    )
    assert client.get("/api/memory/graduated/adopted/absent-skill").status_code == 404

    promoted = client.post(
        "/api/memory/curated/parent-control-reading/promote",
        json={"experiment_id": "adopted", "skill": "same-window-parent-control"},
    )
    assert promoted.status_code == 200
    assert "parent-control-reading" in _entry_names(promoted.json())

    deleted = client.delete("/api/memory/curated/cash-buffer-rule")
    assert deleted.status_code == 200
    assert "cash-buffer-rule" not in _entry_names(deleted.json())


def test_the_exclusion_routes_carry_the_withdraw_and_restore_cycle(
    tmp_path: Path,
) -> None:
    _library(tmp_path)
    experiments = tmp_path / "experiments"
    experiments.mkdir()
    _experiment(experiments, "adopted")
    client = TestClient(create_app(tmp_path, experiments))
    skill = "same-window-parent-control"

    excluded = client.post(
        f"/api/memory/graduated/adopted/{skill}/exclude", json={"reason": "过期"}
    )
    assert excluded.status_code == 200
    assert {"experiment_id", "skill", "action", "note", "graduated"} <= (
        excluded.json().keys()
    )
    row = next(
        item
        for item in excluded.json()["graduated"]["experiments"]
        if item["experiment_id"] == "adopted"
    )
    assert row["entries"] == [] and row["excluded"][0]["skill"] == skill
    assert client.get(f"/api/memory/graduated/adopted/{skill}").status_code == 404
    assert (
        client.post(
            f"/api/memory/graduated/adopted/{skill}/exclude", json={}
        ).status_code
        == 409
    )

    restored = client.delete(f"/api/memory/graduated/adopted/{skill}/exclude")
    assert restored.status_code == 200
    assert client.get(f"/api/memory/graduated/adopted/{skill}").status_code == 200
    assert client.delete(f"/api/memory/graduated/adopted/{skill}/exclude").status_code == 404
    assert (
        client.post(
            "/api/memory/graduated/adopted/no-such-skill/exclude", json={}
        ).status_code
        == 404
    )
    assert (
        client.post(
            f"/api/memory/graduated/adopted/Not_Kebab/exclude", json={}
        ).status_code
        == 400
    )


def test_the_write_routes_map_each_refusal_to_its_own_status(tmp_path: Path) -> None:
    _library(tmp_path)
    experiments = tmp_path / "experiments"
    experiments.mkdir()
    client = TestClient(create_app(tmp_path, experiments))

    for name in ("Not_Kebab", "curated", "..", "nested%2Fname"):
        response = client.post(
            "/api/memory/curated", json={"name": name, "content": "# x\n"}
        )
        assert response.status_code == 400, name
    assert (
        client.post(
            "/api/memory/curated",
            json={"name": "pit-read-budget", "content": "# x\n"},
        ).status_code
        == 409
    )
    assert client.put("/api/memory/curated/absent-entry", json={"content": "# x\n"}).status_code == 404
    assert client.delete("/api/memory/curated/absent-entry").status_code == 404
    assert (
        client.post(
            "/api/memory/curated/promoted-entry/promote",
            json={"experiment_id": "unknown", "skill": "same-window-parent-control"},
        ).status_code
        == 404
    )
    # A refusal must not hand the host tree back with the reason.
    detail = client.post(
        "/api/memory/curated",
        json={"name": "too-long", "content": "x" * (MAX_SKILL_CHARS + 1)},
    ).json()["detail"]
    assert "16000" in detail and str(tmp_path) not in detail
    assert _entry_names({"curated": client.get("/api/memory").json()["curated"]}) == [
        "output-dir-hygiene",
        "pit-read-budget",
    ]


def test_the_proxied_surface_reads_and_writes_like_any_other_client(
    tmp_path: Path,
) -> None:
    """Write access is the deployment's question, not this route family's: the
    loopback/Unix-socket bind and the edge's login gate decide who reaches the
    console at all, exactly as they do for the experiment routes. A request
    carrying the edge's forwarded headers is therefore an ordinary client."""

    _library(tmp_path)
    experiments = tmp_path / "experiments"
    experiments.mkdir()
    _experiment(experiments, "adopted")
    client = TestClient(create_app(tmp_path, experiments))

    created = client.post(
        "/api/memory/curated",
        json={"name": "cash-buffer-rule", "content": "# 现金缓冲\n"},
        headers=PUBLIC_EDGE,
    )
    assert created.status_code == 200
    assert "cash-buffer-rule" in _entry_names(created.json())
    assert (
        client.put(
            "/api/memory/curated/cash-buffer-rule",
            json={"content": "# 现金缓冲\n\n改过。\n"},
            headers=PUBLIC_EDGE,
        ).status_code
        == 200
    )
    assert (
        client.post(
            "/api/memory/curated/parent-control-reading/promote",
            json={"experiment_id": "adopted", "skill": "same-window-parent-control"},
            headers=PUBLIC_EDGE,
        ).status_code
        == 200
    )
    assert (
        client.delete(
            "/api/memory/curated/cash-buffer-rule", headers=PUBLIC_EDGE
        ).status_code
        == 200
    )


def test_a_curated_body_crosses_the_route_unchanged(tmp_path: Path) -> None:
    """One body on both surfaces: the editor saves back what it opened."""

    _library(tmp_path)
    body = f"# 路径规则\n\n不要写 {HOST_PATH}。\n"
    _curated(tmp_path, "workspace-path-rules", body)
    client = TestClient(create_app(tmp_path, tmp_path / "experiments"))
    for headers in ({}, PUBLIC_EDGE):
        served = client.get(
            "/api/memory/curated/workspace-path-rules", headers=headers
        ).json()
        assert served["content"] == body
        assert "redacted" not in served


# ---- what sessions reported back ------------------------------------------
#
# Sessions may doubt, ignore and report mounted memory; they never rewrite it.
# The console is where those verdicts are read back together, so the invariants
# are that a verdict is attributed to its experiment, that a note crosses the
# host boundary projected, and that "disputed" means what it says.


def _feedback(entry: str, verdict: str, note: str) -> dict[str, object]:
    return {
        "entry": entry,
        "source": entry.split("/")[0],
        "name": entry.split("/")[1],
        "verdict": verdict,
        "note": note,
        "recorded_at": "2026-01-02T03:04:05+00:00",
    }


def test_feedback_is_aggregated_per_entry_across_experiments(tmp_path: Path) -> None:
    _library(tmp_path)
    experiments = tmp_path / "experiments"
    experiments.mkdir()
    first = _experiment(experiments, "adopted")
    second = _experiment(experiments, "other")
    _mounted_run(
        first,
        memory_feedback=[
            _feedback("curated/pit-read-budget", "confirmed", "按它做结果一致。"),
            _feedback("curated/output-dir-hygiene", "outdated", "目录约定已经变了。"),
        ],
    )
    _mounted_run(
        second,
        memory_feedback=[
            _feedback("curated/pit-read-budget", "outdated", "摘要字段取不到了。")
        ],
    )

    aggregate = memory.memory_feedback(experiments)
    assert aggregate["unreadable"] == []
    entry = aggregate["entries"]["curated/pit-read-budget"]
    assert entry["counts"] == {"confirmed": 1, "outdated": 1, "wrong": 0}
    assert entry["experiments"] == 2 and entry["disputed"] is False
    assert {report["experiment_id"] for report in entry["reports"]} == {
        "adopted",
        "other",
    }
    # The session is named the way every other console surface names it.
    assert {report["session_label"] for report in entry["reports"]} == {
        "epoch_001/2022Q1"
    }
    assert aggregate["entries"]["curated/output-dir-hygiene"]["counts"]["outdated"] == 1


def test_disputed_needs_two_experiments_not_two_sessions(tmp_path: Path) -> None:
    """One session's bad day is not the same signal as two experiments
    independently concluding the entry is wrong."""

    _library(tmp_path)
    experiments = tmp_path / "experiments"
    experiments.mkdir()
    first = _experiment(experiments, "adopted")
    second = _experiment(experiments, "other")
    _mounted_run(
        first,
        memory_feedback=[_feedback("curated/pit-read-budget", "wrong", "不成立。")],
    )
    _mounted_run(
        first,
        run_dir="run_second_session",
        memory_feedback=[_feedback("curated/pit-read-budget", "wrong", "还是不成立。")],
    )
    entry = memory.memory_feedback(experiments)["entries"]["curated/pit-read-budget"]
    assert entry["counts"]["wrong"] == 2 and entry["experiments"] == 1
    assert entry["disputed"] is False

    _mounted_run(
        second,
        memory_feedback=[_feedback("curated/pit-read-budget", "wrong", "这里也不成立。")],
    )
    entry = memory.memory_feedback(experiments)["entries"]["curated/pit-read-budget"]
    assert entry["experiments"] == 2 and entry["disputed"] is True


def test_a_note_crosses_the_host_boundary_projected(tmp_path: Path) -> None:
    _library(tmp_path)
    experiments = tmp_path / "experiments"
    experiments.mkdir()
    directory = _experiment(experiments, "adopted")
    _mounted_run(
        directory,
        memory_feedback=[
            _feedback("curated/pit-read-budget", "wrong", f"它让我去读 {HOST_PATH}。")
        ],
    )
    note = memory.memory_feedback(experiments)["entries"]["curated/pit-read-budget"][
        "reports"
    ][0]["note"]
    assert HOST_PATH not in note and "[host path omitted]" in note


def test_an_experiment_whose_feedback_is_unreadable_is_named_not_dropped(
    tmp_path: Path,
) -> None:
    """A missing verdict changes what an entry looks like, so a partial count
    must not be presented as the whole picture."""

    _library(tmp_path)
    experiments = tmp_path / "experiments"
    experiments.mkdir()
    directory = _experiment(experiments, "adopted")
    _mounted_run(
        directory,
        memory_feedback=[_feedback("curated/pit-read-budget", "confirmed", "成立。")],
    )
    broken = experiments / "broken"
    (broken / "artifacts" / "run_x").mkdir(parents=True)
    (broken / "artifacts" / "run_x" / "host_run_manifest.json").write_text(
        "{truncated", encoding="utf-8"
    )
    aggregate = memory.memory_feedback(experiments)
    assert aggregate["entries"]["curated/pit-read-budget"]["counts"]["confirmed"] == 1
    assert [item["experiment_id"] for item in aggregate["unreadable"]] == ["broken"]
    assert aggregate["unreadable"][0]["error"].endswith("session feedback is unreadable")
    assert str(tmp_path) not in aggregate["unreadable"][0]["error"]


def test_a_verdict_this_console_does_not_know_is_not_counted(tmp_path: Path) -> None:
    _library(tmp_path)
    experiments = tmp_path / "experiments"
    experiments.mkdir()
    directory = _experiment(experiments, "adopted")
    _mounted_run(
        directory,
        memory_feedback=[
            _feedback("curated/pit-read-budget", "invented", "x"),
            {"verdict": "confirmed", "note": "no entry"},
        ],
    )
    assert memory.memory_feedback(experiments)["entries"] == {}


def test_the_routes_serve_the_badges_on_the_page_and_the_reports_in_the_pane(
    tmp_path: Path,
) -> None:
    """Two projections of one aggregate: the page bundle carries counts only,
    and the notes arrive when the reader selects an entry."""

    _library(tmp_path)
    experiments = tmp_path / "experiments"
    experiments.mkdir()
    directory = _experiment(experiments, "adopted")
    _mounted_run(
        directory,
        memory_feedback=[
            _feedback("curated/pit-read-budget", "wrong", "与本窗口观察不符。"),
            _feedback(
                "adopted/same-window-parent-control", "confirmed", "同窗对照同样成立。"
            ),
        ],
    )
    client = TestClient(create_app(tmp_path, experiments))

    overview = client.get("/api/memory").json()
    badges = overview["feedback"]["entries"]["curated/pit-read-budget"]
    assert badges["counts"]["wrong"] == 1 and badges["experiments"] == 1
    assert badges["disputed"] is False
    assert "reports" not in badges
    assert overview["feedback"]["unreadable"] == []

    curated = client.get("/api/memory/curated/pit-read-budget").json()
    assert curated["feedback"]["counts"]["wrong"] == 1
    assert curated["feedback"]["reports"][0]["note"] == "与本窗口观察不符。"
    assert curated["feedback"]["reports"][0]["experiment_id"] == "adopted"

    candidate = client.get(
        "/api/memory/graduated/adopted/same-window-parent-control"
    ).json()
    assert candidate["feedback"]["counts"]["confirmed"] == 1
    # An entry nobody reported on still answers with an empty aggregate.
    other = client.get("/api/memory/curated/output-dir-hygiene").json()
    assert other["feedback"]["counts"] == {"confirmed": 0, "outdated": 0, "wrong": 0}
    assert other["feedback"]["reports"] == []
