"""Console surfaces of the research geometry.

The console lists experiments from whatever is on disk, so a Fold-era
``params.json`` must degrade to an unreadable row instead of taking the
listing down; the arm's verdict reaches the listing once its forward record
exists; and the create form is seeded with the research geometry the worker
actually runs.
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from autotrade.environment.identity import AgentRefStore
from autotrade.environment.runtime import write_json_atomic
from autotrade.pipelines.config import DEFAULT_RESEARCH_GEOMETRY
from autotrade.pipelines.hitl_state import (
    WEB_CREATE_DEFAULTS,
    ControlState,
    build_session_plan,
    write_control,
)
from autotrade.pipelines.ledger import ExperimentLedger
from autotrade.webui.registry import list_experiments, summarize_experiment
from autotrade.webui.server import create_app


def _experiment(root: Path, experiment_id: str, params: dict[str, object]) -> Path:
    directory = root / experiment_id
    AgentRefStore(directory)
    hitl = directory / "hitl"
    hitl.mkdir(parents=True)
    write_json_atomic(hitl / "params.json", {"experiment_id": experiment_id, **params})
    write_control(hitl / "control.json", ControlState(mode="auto"))
    write_json_atomic(hitl / "status.json", {"schema_version": 1, "state": "completed"})
    return directory


_CURRENT = {**DEFAULT_RESEARCH_GEOMETRY.to_record()}
_FOLD_ERA = {
    "fold_period": "year",
    "development_first_period": "2022",
    "development_last_period": "2025",
    "heldout_first_period": "20260101..20260630",
    "heldout_last_period": "20260101..20260630",
}


def test_a_fold_era_params_file_is_flagged_unreadable_and_the_rest_still_list(tmp_path: Path):
    root = tmp_path / "experiments"
    _experiment(root, "legacy", _FOLD_ERA)
    _experiment(root, "current", _CURRENT)
    rows = {row["experiment_id"]: row for row in list_experiments(root)}
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


def test_the_verdict_reaches_the_listing_once_the_forward_record_exists(tmp_path: Path):
    root = tmp_path / "experiments"
    directory = _experiment(root, "graduate", _CURRENT)
    write_json_atomic(directory / "hitl/schedule.json", build_session_plan(forward={}))
    ledger = ExperimentLedger(directory / "ledgers/experiment_ledger.jsonl")
    frozen = directory / "artifacts/strategy/frozen/strategy_research_abc/output"
    ledger.append(
        {
            "record_type": "research_session",
            "experiment_id": "graduate",
            "epoch_id": "research",
            "fold_id": "research",
            "run_id": "run_research",
            "session_key": "research",
            "outcome": "freeze",
            "steps": [],
            "frozen": {"artifact_id": "strategy_research_abc", "output_path": str(frozen)},
        }
    )
    # Frozen, forward still running: no verdict and no Paper candidate.
    summary = summarize_experiment(directory)
    assert summary["verdict"] is None
    assert summary["paper_candidate"] is None
    ledger.append(
        {
            "record_type": "forward",
            "experiment_id": "graduate",
            "epoch_id": "forward",
            "fold_id": "forward",
            "run_id": "run_forward",
            "session_key": "forward",
            "artifact_id": "strategy_research_abc",
            "status": "ok",
            "verdict": {"status": "graduated", "reasons": []},
        }
    )
    summary = summarize_experiment(directory)
    assert summary["verdict"]["status"] == "graduated"
    assert summary["verdict"]["reasons"] == []
    assert summary["paper_candidate"]["artifact_id"] == "strategy_research_abc"
    detail = TestClient(create_app(tmp_path)).get("/api/experiments/graduate").json()
    assert detail["verdict"]["status"] == "graduated"


def test_the_create_form_is_seeded_with_the_research_geometry(tmp_path: Path):
    schema = TestClient(create_app(tmp_path)).get("/api/parameter-schema").json()
    fields = {field["key"]: field for group in schema["groups"] for field in group["fields"]}
    for key, value in DEFAULT_RESEARCH_GEOMETRY.to_record().items():
        assert fields[key]["required"] is True, key
        assert fields[key]["default"] == value, key
    for retired in ("test_stage", "epochs", "fold_period", "development_first_period"):
        assert retired not in fields
    for key, value in (
        ("window_months", 24),
        ("max_research_minutes", 2400),
        ("max_replay_years", 96),
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
        json={"experiment_id": "no_calendar", "research_start": ""},
    )
    assert response.status_code == 400
    assert "research_start" in json.dumps(response.json())
    assert not (tmp_path / "experiments" / "no_calendar").exists()
