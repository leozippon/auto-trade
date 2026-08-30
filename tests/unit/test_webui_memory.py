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


def _mounted_run(directory: Path, **overrides: object) -> Path:
    manifest = directory / "artifacts" / RUN_ID / "host_run_manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {
        "run_id": RUN_ID,
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

    rows = {row["experiment_id"]: row for row in memory.graduated_tier(experiments)["experiments"]}
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
    row = memory.graduated_tier(experiments)["experiments"][0]
    assert row["revealed"] is False
    assert row["verdict"] is None
    assert row["admitted"] is False
    assert row["entries"] == []
    # The mount itself is unaffected: this is a display gate only.
    from autotrade.pipelines.skills import graduated_memory_sources

    assert [source.source for source in graduated_memory_sources(experiments)] == [
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
    payload = memory.graduated_tier(experiments)
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
    payload = memory.graduated_tier(experiments)
    assert payload["error"].endswith("graduated memory cannot be resolved")
    assert str(tmp_path) not in payload["error"]
    assert [row["experiment_id"] for row in payload["experiments"]] == ["curated"]
    assert payload["experiments"][0]["admitted"] is None


# ---- what one experiment actually mounted ---------------------------------


def test_the_mounted_block_projects_the_run_manifest(tmp_path: Path) -> None:
    experiments = tmp_path / "experiments"
    experiments.mkdir()
    directory = _experiment(experiments, "adopted")
    _params(directory, operating_memory="curated+graduated")
    _mounted_run(directory)

    payload = memory.experiment_memory(experiments, "adopted")
    assert payload["mode"] == "curated+graduated"
    session = payload["sessions"][0]
    assert session["kind"] == "fold"
    assert session["session_label"] == "epoch_001/2022Q1"
    assert session["mode"] == "curated+graduated"
    assert [source["source"] for source in session["sources"]] == ["curated", "adopted"]
    assert [source["origin"] for source in session["sources"]] == [
        "curated",
        "graduated",
    ]
    assert session["sources"][0]["entries"] == [
        "output-dir-hygiene",
        "pit-read-budget",
    ]


def test_the_mounted_block_never_carries_a_raw_run_id_or_a_host_path(
    tmp_path: Path,
) -> None:
    experiments = tmp_path / "experiments"
    experiments.mkdir()
    directory = _experiment(experiments, "adopted")
    _params(directory, operating_memory="curated+graduated")
    # A key this read model does not know about still has to leave the host
    # boundary through the same projection, not verbatim.
    _mounted_run(
        directory,
        operating_memory={
            "mode": "curated",
            "sources": [
                {
                    "source": "curated",
                    "origin": "curated",
                    "entries": ["pit-read-budget"],
                    "mount_root": HOST_PATH,
                }
            ],
        },
    )
    payload = memory.experiment_memory(experiments, "adopted")
    text = json.dumps(payload, ensure_ascii=False)
    assert RUN_ID not in text
    assert HOST_PATH not in text
    assert str(tmp_path) not in text
    assert payload["sessions"][0]["run_ref"].startswith("run_ref_")
    assert payload["sessions"][0]["sources"][0]["mount_root"] == "[host path omitted]"


def test_an_unreadable_manifest_keeps_the_session_visible(tmp_path: Path) -> None:
    experiments = tmp_path / "experiments"
    experiments.mkdir()
    directory = _experiment(experiments, "adopted")
    _params(directory, operating_memory="curated+graduated")
    manifest = _mounted_run(directory)
    manifest.write_text("{truncated", encoding="utf-8")
    session = memory.experiment_memory(experiments, "adopted")["sessions"][0]
    assert session["run_ref"].startswith("run_ref_")
    assert session["error"].endswith("run manifest is unreadable")


def test_a_session_that_mounted_nothing_says_so(tmp_path: Path) -> None:
    experiments = tmp_path / "experiments"
    experiments.mkdir()
    directory = _experiment(experiments, "adopted")
    _params(directory, operating_memory="none")
    _mounted_run(directory, operating_memory={"mode": "none", "sources": []})
    payload = memory.experiment_memory(experiments, "adopted")
    assert payload["mode"] == "none"
    assert payload["sessions"][0]["sources"] == []


def test_an_experiment_without_params_falls_back_to_the_pipeline_default(
    tmp_path: Path,
) -> None:
    experiments = tmp_path / "experiments"
    experiments.mkdir()
    _experiment(experiments, "adopted")
    payload = memory.experiment_memory(experiments, "adopted")
    assert payload["mode"] == payload["default_mode"]
    assert payload["sessions"] == []


def test_a_mounted_entry_opens_through_the_same_read_routes_until_it_is_gone(
    tmp_path: Path,
) -> None:
    """The mounted block is a historical record, and its entry chips ask today's
    library and tier. Both sources must resolve while the entry still exists,
    and an entry since removed must read as missing rather than as a failure —
    the record itself is never rewritten."""

    _library(tmp_path)
    experiments = tmp_path / "experiments"
    experiments.mkdir()
    directory = _experiment(experiments, "adopted")
    _params(directory, operating_memory="curated+graduated")
    _mounted_run(directory)
    client = TestClient(create_app(tmp_path, experiments))

    session = client.get("/api/experiments/adopted/memory").json()["sessions"][0]
    curated, graduated = session["sources"]
    assert (curated["source"], curated["origin"]) == ("curated", "curated")
    assert (graduated["source"], graduated["origin"]) == ("adopted", "graduated")
    for name in curated["entries"]:
        assert client.get(f"/api/memory/curated/{name}").status_code == 200
    for name in graduated["entries"]:
        assert client.get(f"/api/memory/graduated/adopted/{name}").status_code == 200

    assert client.delete("/api/memory/curated/pit-read-budget").status_code == 200
    assert client.get("/api/memory/curated/pit-read-budget").status_code == 404
    # The manifest still records what that session mounted.
    assert "pit-read-budget" in (
        client.get("/api/experiments/adopted/memory").json()["sessions"][0]["sources"][
            0
        ]["entries"]
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
    assert {"default_mode", "curated", "graduated"} <= body.keys()
    assert {"source", "library", "entries"} <= body["curated"].keys()
    assert {"name", "title", "summary", "bytes", "files"} <= body["curated"][
        "entries"
    ][0].keys()
    assert {"experiment_id", "verdict", "revealed", "admitted", "entries"} <= body[
        "graduated"
    ]["experiments"][0].keys()

    entry = client.get("/api/memory/curated/pit-read-budget")
    assert entry.status_code == 200
    assert entry.json()["content"].startswith("# PIT 读取预算")
    assert client.get("/api/memory/curated/not-curated").status_code == 404
    assert client.get("/api/memory/curated/Not_Kebab").status_code == 400

    mounted = client.get("/api/experiments/adopted/memory")
    assert mounted.status_code == 200
    assert {"experiment_id", "mode", "default_mode", "sessions"} <= mounted.json().keys()
    assert mounted.json()["sessions"][0]["sources"][0]["source"] == "curated"
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
    entry = memory.graduated_entry(experiments, "adopted", "same-window-parent-control")
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
            memory.graduated_entry(experiments, experiment_id, skill)
    with pytest.raises(ValueError):
        memory.graduated_entry(experiments, "adopted", "Not_Kebab")


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
