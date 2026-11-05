"""Operating-memory console model: the curated library, snapshots, curated CRUD.

The read view's invariants are what it refuses to show and what it refuses to
invent: a malformed library reports an error rather than an empty shelf, an
experiment's block is the snapshot it froze rather than the library now, and
nothing crosses the HTTP boundary carrying a host path or a raw run identity.

The curated writes add two of their own: an entry a session would refuse never
reaches the library, and the library is never left half-written.

Every fixture is synthesized in a tempfile repo root exactly as the pipeline
writes it: a curated library under ``configs/operating_memory/``, experiment
directories with their ledger and console state, and a collected session's
``host_run_manifest.json``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from autotrade.environment.identity import AgentRefStore
from autotrade.pipelines.config import DEFAULT_RESEARCH_GEOMETRY
from autotrade.pipelines.hitl_state import (
    ControlState,
    build_session_plan,
    write_control,
)
from autotrade.pipelines.ledger import ExperimentLedger
from autotrade.pipelines.skills import (
    MAX_SKILL_CHARS,
    OPERATING_MEMORY_LIBRARY,
    RETIRED_MEMORY_ORIGIN,
    RETIRED_OPERATING_MEMORY_MODE,
    build_skills_index,
    create_operating_memory_snapshot,
    operating_memory_snapshot_path,
    operating_memory_snapshot_root,
    read_operating_memory_snapshot,
    validate_skills_tree,
)
from autotrade.webui import memory
from autotrade.webui.server import create_app
from tests.unit.acceptance_fixture import CREATE_RULES

HOST_PATH = "/Data2/lzp/ADMCubeQuant/experiments/adopted/artifacts"
SESSION_KEY = "research"
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


def _experiment(experiments_root: Path, name: str) -> Path:
    """One research arm: its ledger's session record and its console state."""

    directory = experiments_root / name
    # Before any ledger or artifact exists: the store refuses to initialize
    # beside pre-random-ref artifacts, exactly as it does for a real experiment.
    AgentRefStore(directory)
    ExperimentLedger(directory / "ledgers" / "experiment_ledger.jsonl").append(
        {
            "record_type": "research_session",
            "experiment_id": name,
            "epoch_id": "research",
            "fold_id": "s1",
            "run_id": f"run_{name}",
        }
    )
    hitl = directory / "hitl"
    hitl.mkdir(parents=True, exist_ok=True)
    (hitl / "schedule.json").write_text(
        json.dumps(build_session_plan(forward={})), encoding="utf-8"
    )
    write_control(hitl / "control.json", ControlState())
    return directory


def _mounted_run(directory: Path, *, run_dir: str = RUN_ID, **overrides: object) -> Path:
    manifest = directory / "artifacts" / run_dir / "host_run_manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {
        "run_id": run_dir,
        "kind": "fold",
        "session_key": SESSION_KEY,
        "operating_memory": {
            "mode": "curated",
            "sources": [
                {
                    "source": "curated",
                    "origin": "curated",
                    "entries": ["output-dir-hygiene", "pit-read-budget"],
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
    entry = memory.curated_entry(tmp_path, "pit-read-budget")
    assert entry["name"] == "pit-read-budget"
    assert entry["title"] == "PIT 读取预算"
    assert entry["content"].startswith("# PIT 读取预算")
    with pytest.raises(KeyError):
        memory.curated_entry(tmp_path, "not-curated")
    for bad in ("../configs", "PIT", "a/b", ""):
        with pytest.raises(ValueError):
            memory.curated_entry(tmp_path, bad)


def test_a_curated_body_is_served_verbatim_so_an_edit_round_trips(
    tmp_path: Path,
) -> None:
    """The library is researcher-authored repository content that this console
    also edits: rewriting a body for display would silently save the rewrite
    over the real line the next time it is opened and saved."""

    _library(tmp_path)
    body = f"# 路径规则\n\n只写 /mnt/agent/workspace/output，不要写 {HOST_PATH}。\n"
    _curated(tmp_path, "workspace-path-rules", body)
    assert memory.curated_entry(tmp_path, "workspace-path-rules")["content"] == body
    memory.update_curated_entry(tmp_path, "workspace-path-rules", body)
    assert (
        tmp_path / OPERATING_MEMORY_LIBRARY / "workspace-path-rules" / "SKILL.md"
    ).read_text(encoding="utf-8") == body


# ---- the snapshot one experiment froze -------------------------------------
#
# Operating memory is fixed per experiment: the curated library is resolved
# once, when the experiment is created, and every session mounts that same
# copy. So the block answers one question for the whole experiment, and a
# library change after the fact belongs to the next experiment.


def _snapshot(
    repo_root: Path,
    experiments: Path,
    experiment_id: str,
    mode: str = "curated",
) -> dict:
    directory = experiments / experiment_id
    directory.mkdir(parents=True, exist_ok=True)
    return create_operating_memory_snapshot(directory, mode=mode, repo_root=repo_root)


def _retired_tier_snapshot(repo_root: Path, experiments: Path, experiment_id: str) -> None:
    """The snapshot an arm created before 2026-11-05 froze: the curated
    library beside one graduated experiment's skills, under the retired mode."""

    directory = experiments / experiment_id
    _snapshot(repo_root, experiments, experiment_id)
    root = operating_memory_snapshot_root(directory)
    root.chmod(0o755)
    item = root / "adopted" / "same-window-parent-control"
    item.mkdir(parents=True)
    (item / "SKILL.md").write_text("# 同窗父本对照\n\n正文。\n", encoding="utf-8")
    path = operating_memory_snapshot_path(directory)
    record = json.loads(path.read_text(encoding="utf-8"))
    record["mode"] = RETIRED_OPERATING_MEMORY_MODE
    record["entries"].append(
        {
            "origin": RETIRED_MEMORY_ORIGIN,
            "source": "adopted",
            "name": "same-window-parent-control",
            "heldout_end": "20260930",
        }
    )
    path.write_text(json.dumps(record), encoding="utf-8")


def test_the_block_lists_the_snapshot_this_experiment_froze(tmp_path: Path) -> None:
    _library(tmp_path)
    experiments = tmp_path / "experiments"
    experiments.mkdir()
    directory = _experiment(experiments, "current")
    _params(directory, operating_memory="curated")
    _snapshot(tmp_path, experiments, "current")
    _mounted_run(directory)

    payload = memory.experiment_memory(experiments, "current")
    assert payload["mode"] == "curated"
    assert payload["sessions_seen"] == 1
    snapshot = payload["snapshot"]
    assert snapshot["created_from"] == "creation" and snapshot["created_at"]
    assert snapshot["sources"] == [
        {
            "source": "curated",
            "origin": "curated",
            "entries": ["output-dir-hygiene", "pit-read-budget"],
        }
    ]
    # Nothing per session is projected any more, and no host identity leaks.
    assert "sessions" not in payload
    assert RUN_ID not in json.dumps(payload, ensure_ascii=False)


def test_an_arm_of_the_retired_graduated_tier_lists_what_it_still_mounts(
    tmp_path: Path,
) -> None:
    """Every arm created before the graduated tier's retirement names its mode
    and may hold graduated entries in its snapshot: the block opens, reads the
    mode as the curated library it is, keeps the snapshot's own record of the
    mode, and lists only the curated entries its sessions mount."""

    _library(tmp_path)
    experiments = tmp_path / "experiments"
    experiments.mkdir()
    directory = _experiment(experiments, "current")
    _params(directory, operating_memory=RETIRED_OPERATING_MEMORY_MODE)
    _retired_tier_snapshot(tmp_path, experiments, "current")

    payload = memory.experiment_memory(experiments, "current")
    assert "error" not in payload
    assert payload["mode"] == "curated"
    assert payload["snapshot"]["mode"] == RETIRED_OPERATING_MEMORY_MODE
    assert [source["source"] for source in payload["snapshot"]["sources"]] == ["curated"]
    client = TestClient(create_app(tmp_path, experiments))
    assert client.get("/api/experiments/current/memory").status_code == 200
    assert (
        client.get("/api/experiments/current/memory/curated/pit-read-budget").status_code
        == 200
    )
    assert (
        client.get(
            "/api/experiments/current/memory/adopted/same-window-parent-control"
        ).status_code
        == 400
    )


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
        **DEFAULT_RESEARCH_GEOMETRY.to_record(), **CREATE_RULES,
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
    directory = _experiment(experiments, "current")
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
    directory = _experiment(experiments, "legacy")
    _params(directory, operating_memory="curated")
    payload = memory.experiment_memory(experiments, "legacy")
    assert payload["snapshot"] is None and payload["sessions_seen"] == 0
    assert "error" not in payload


def test_an_unreadable_snapshot_is_reported_not_hidden(tmp_path: Path) -> None:
    _library(tmp_path)
    experiments = tmp_path / "experiments"
    experiments.mkdir()
    directory = _experiment(experiments, "current")
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
    directory = _experiment(experiments, "current")
    _params(directory, operating_memory="curated")
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
    _params(directory, operating_memory="curated")
    _mounted_run(directory)
    (experiments / ".staging").mkdir()
    client = TestClient(create_app(tmp_path, experiments))

    overview = client.get("/api/memory")
    assert overview.status_code == 200
    body = overview.json()
    assert body.keys() == {"default_mode", "curated", "experiments"}
    assert body["default_mode"] == "curated"
    assert {"source", "library", "entries"} <= body["curated"].keys()
    assert {"name", "title", "summary", "bytes", "files"} <= body["curated"][
        "entries"
    ][0].keys()
    # The experiments the inboxes read, which the page's filters offer.
    assert body["experiments"] == ["adopted"]

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
    assert memory.curated_entry(tmp_path, "cash-buffer-rule")["content"].startswith(
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
    assert memory.curated_entry(tmp_path, "pit-read-budget")["content"].startswith(
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


def test_a_delete_removes_the_whole_item_and_leaves_running_sessions_alone(
    tmp_path: Path,
) -> None:
    """Running sessions hold their own read-only copy, so the delete is allowed
    while they run and takes the whole item, staging included."""

    _library(tmp_path)
    result = memory.delete_curated_entry(tmp_path, "pit-read-budget")
    assert result["action"] == "deleted"
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


# ---- write routes ---------------------------------------------------------


def test_the_curated_write_routes_carry_one_crud_cycle(tmp_path: Path) -> None:
    _library(tmp_path)
    experiments = tmp_path / "experiments"
    experiments.mkdir()
    client = TestClient(create_app(tmp_path, experiments))

    created = client.post(
        "/api/memory/curated",
        json={"name": "cash-buffer-rule", "content": "# 现金缓冲\n\n留出缓冲。\n"},
    )
    assert created.status_code == 200
    assert {"name", "action", "curated"} <= created.json().keys()
    assert "cash-buffer-rule" in _entry_names(created.json())

    edited = client.put(
        "/api/memory/curated/cash-buffer-rule", json={"content": "# 现金缓冲\n\n改过。\n"}
    )
    assert edited.status_code == 200
    assert client.get("/api/memory/curated/cash-buffer-rule").json()["content"].endswith(
        "改过。\n"
    )

    deleted = client.delete("/api/memory/curated/cash-buffer-rule")
    assert deleted.status_code == 200
    assert "cash-buffer-rule" not in _entry_names(deleted.json())


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


def test_the_memory_page_is_attached_only_once_its_inboxes_have_answered() -> None:
    """Navigating to 演化 painted the two feedback sections as loading boxes
    and then shrank them onto their answers. Both inboxes now fill their still
    detached hosts before the page is attached, so it opens at its own height
    and nothing re-renders them afterwards."""

    app_js = (
        Path(__file__).resolve().parents[2] / "src/autotrade/webui/static/app.js"
    ).read_text(encoding="utf-8")
    body = app_js.split("async function renderMemoryPage(", 1)[1].split("\n}", 1)[0]
    awaited = 'await Promise.all([renderFeedback("issue"), renderFeedback("skill")]);'
    assert body.index(awaited) < body.rindex("$main.replaceChildren(")
    assert body.count("renderFeedback(") == 2
