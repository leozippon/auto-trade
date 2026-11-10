"""Console surfaces of the research geometry.

The console lists experiments from whatever is on disk, so a ``params.json``
naming parameters no parameter declares must degrade to an unreadable row
instead of taking the listing down; the experiment's verdict reaches the listing once its forward record
exists, with the Paper action only for an experiment created on this host; and the create form is seeded with the research geometry the worker
actually runs.
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from autotrade.common.records import write_json_atomic
from autotrade.environment.identity import AgentRefStore
from autotrade.pipelines.config import DEFAULT_RESEARCH_GEOMETRY
from autotrade.pipelines.hitl_state import (
    WEB_CREATE_DEFAULTS,
    ControlState,
    build_session_plan,
    write_control,
)
from autotrade.pipelines.ledger import ExperimentLedger
from autotrade.webui.registry import experiment_listing, summarize_experiment
from autotrade.webui.server import create_app
from tests.unit.acceptance_fixture import REQUIRED_RULES


def _experiment(root: Path, experiment_id: str, params: dict[str, object]) -> Path:
    directory = root / experiment_id
    AgentRefStore(directory)
    hitl = directory / "hitl"
    hitl.mkdir(parents=True)
    write_json_atomic(hitl / "params.json", {"experiment_id": experiment_id, **params})
    write_control(hitl / "control.json", ControlState())
    write_json_atomic(hitl / "status.json", {"schema_version": 1, "state": "completed"})
    return directory


# What the console stamps: the geometry and every required rule.
_CURRENT = {**DEFAULT_RESEARCH_GEOMETRY.to_record(), **REQUIRED_RULES}
_UNKNOWN = {
    "fold_period": "year",
    "development_first_period": "2022",
    "development_last_period": "2025",
    "heldout_first_period": "20260101..20260630",
    "heldout_last_period": "20260101..20260630",
}


def test_a_params_file_of_unknown_keys_is_flagged_unreadable_and_the_rest_still_list(tmp_path: Path):
    root = tmp_path / "experiments"
    _experiment(root, "legacy", _UNKNOWN)
    _experiment(root, "current", _CURRENT)
    rows = {row["experiment_id"]: row for row in experiment_listing(root)["experiments"]}
    assert rows["legacy"]["state"] == "unreadable"
    assert "UnsupportedParamsError" in str(rows["legacy"]["error"])
    assert rows["current"]["state"] == "completed"
    client = TestClient(create_app(tmp_path))
    listed = client.get("/api/experiments").json()["experiments"]
    assert {row["experiment_id"]: row["state"] for row in listed} == {
        "legacy": "unreadable",
        "current": "completed",
    }
    detail = client.get("/api/experiments/legacy").json()
    assert detail["state"] == "unreadable"
    assert detail["params"] == {}


def _registration(directory: Path) -> dict[str, object]:
    """The research session that registered ``strategy_research_abc``."""

    frozen = directory / "artifacts/strategy/registered/strategy_research_abc/output"
    return {
        "record_type": "research_session",
        "experiment_id": directory.name,
        "epoch_id": "research",
        "fold_id": "research",
        "run_id": "run_research",
        "session_key": "research",
        "outcome": "completed",
        "steps": [],
        "registered": {"artifact_id": "strategy_research_abc", "output_path": str(frozen)},
    }


def _forward(directory: Path) -> dict[str, object]:
    """The forward record of that registration, with its readout."""

    return {
        "record_type": "forward",
        "experiment_id": directory.name,
        "epoch_id": "forward",
        "fold_id": "forward",
        "run_id": "run_forward",
        "session_key": "forward",
        "artifact_id": "strategy_research_abc",
        "status": "ok",
        "readout": {"state": "registered_with_record", "record": {}, "forward": {}},
    }


def test_the_verdict_reaches_the_listing_once_the_forward_record_exists(tmp_path: Path):
    root = tmp_path / "experiments"
    directory = _experiment(root, "registered", _CURRENT)
    write_json_atomic(directory / "hitl/schedule.json", build_session_plan(forward={}))
    ledger = ExperimentLedger(directory / "ledgers/experiment_ledger.jsonl")
    ledger.append(_registration(directory))
    # Frozen, forward still running: no verdict and nothing to open on Paper.
    summary = summarize_experiment(directory)
    assert summary["verdict"] is None
    assert summary["paper_openable"] is False
    ledger.append(_forward(directory))
    summary = summarize_experiment(directory)
    assert summary["verdict"]["status"] == "registered_with_record"
    assert summary["verdict"]["reasons"] == []
    assert summary["paper_openable"] is True
    detail = TestClient(create_app(tmp_path)).get("/api/experiments/registered").json()
    assert detail["verdict"]["status"] == "registered_with_record"


def test_a_registered_strategy_synced_from_another_host_is_listed_without_the_paper_action(tmp_path: Path):
    """Opening a Paper account rebuilds the experiment's settings, which refuse an
    ``experiments_root`` other than the directory's parent: an experiment created
    on the ZGCA instance and synced back is read here, but the console does not
    offer what would fail. Its own experiments, recorded absolute (the console)
    or repository-relative (the command line), still offer it."""

    root = tmp_path / "experiments"
    for experiment_id, recorded in (
        ("console", str(root)),
        ("command_line", "experiments"),
        ("synced", "/personal_fast_storage/auto-trade/experiments"),
    ):
        directory = _experiment(root, experiment_id, {**_CURRENT, "experiments_root": recorded})
        ledger = ExperimentLedger(directory / "ledgers/experiment_ledger.jsonl")
        ledger.append(_registration(directory))
        ledger.append(_forward(directory))
    rows = {row["experiment_id"]: row for row in experiment_listing(root)["experiments"]}
    assert {name: row["ending"]["state"] for name, row in rows.items()} == dict.fromkeys(rows, "registered_with_record")
    assert {name: row["paper_openable"] for name, row in rows.items()} == {
        "console": True,
        "command_line": True,
        "synced": False,
    }
    # The raw artifact id stays on the host side of the listing.
    assert "strategy_research_abc" not in json.dumps(rows)


def test_the_create_form_is_seeded_with_the_research_geometry(tmp_path: Path):
    schema = TestClient(create_app(tmp_path)).get("/api/parameter-schema").json()
    fields = {field["key"]: field for group in schema["groups"] for field in group["fields"]}
    for key, value in DEFAULT_RESEARCH_GEOMETRY.to_record().items():
        assert fields[key]["required"] is True, key
        assert fields[key]["default"] == value, key
    for retired in ("test_stage", "epochs", "fold_period", "development_first_period"):
        assert retired not in fields
    for key, value in (
        ("window_months", 132),
        ("max_research_minutes", 10_080),
        ("max_llm_calls", 6400),
        ("screen_exclude_st", False),
        ("screen_exclude_new_listed_days", 0),
        ("screen_boards", []),
    ):
        assert fields[key]["default"] == value, key
        assert WEB_CREATE_DEFAULTS[key] == (tuple(value) if isinstance(value, list) else value), key
    # The form pre-fills the geometry, but a blank date is still a missing
    # required field, not a silent default.
    response = TestClient(create_app(tmp_path)).post(
        "/api/experiments",
        json={"params": {"experiment_id": "no_calendar", "research_start": ""}},
    )
    assert response.status_code == 400
    assert "research_start" in json.dumps(response.json())
    assert not (tmp_path / "experiments" / "no_calendar").exists()
