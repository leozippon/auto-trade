"""Operating-memory console read model: library, tier gate, mounts, routes.

The view is read-only, so the invariants worth testing are what it refuses to
show and what it refuses to invent: a malformed library reports an error rather
than an empty shelf, an experiment that has not revealed its held-out results
publishes no verdict here either, admission is never recomputed beside
``skills.graduated_memory_sources``, and nothing crosses the HTTP boundary
carrying a host path or a raw run identity.

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
    OPERATING_MEMORY_LIBRARY,
    ExperimentSkillsStore,
    build_skills_index,
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


def test_a_host_path_inside_a_curated_body_is_redacted(tmp_path: Path) -> None:
    """The library is repository content, but the console is proxied to a
    public edge: the same path rule applies as to every other served text."""

    _library(tmp_path)
    _curated(
        tmp_path,
        "workspace-path-rules",
        f"# 路径规则\n\n只写 /mnt/agent/workspace/output，不要写 {HOST_PATH}。\n",
    )
    content = memory.curated_entry(tmp_path, "workspace-path-rules")["content"]
    assert HOST_PATH not in content
    assert "[host path omitted]" in content
    # An allow-listed container root stays readable.
    assert "/mnt/agent/workspace/output" in content


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
