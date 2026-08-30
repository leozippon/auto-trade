"""Console surfaces of the development-window design.

The console lists experiments from whatever is on disk, so a legacy
``params.json`` (pre development-window keys) must degrade to an unreadable
row instead of taking the listing down; the graduation verdict must reach the
listing only after the reveal; and the create form must be seeded with the
regular-Fold defaults the worker actually runs.
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from autotrade.environment.identity import AgentRefStore
from autotrade.environment.runtime import write_json_atomic
from autotrade.pipelines.config import AcceptanceRules
from autotrade.pipelines.hitl_state import WEB_CREATE_DEFAULTS, ControlState, write_control
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


_CURRENT = {
    "fold_period": "year",
    "development_first_period": "2022",
    "development_last_period": "2025",
    "heldout_first_period": "20260101..20260630",
    "heldout_last_period": "20260101..20260630",
}
_LEGACY = {
    "fold_period": "year",
    "first_test_period": "2023",
    "last_test_period": "2025",
    "heldout_first_period": "20260101..20260630",
    "heldout_last_period": "20260101..20260630",
}


def test_a_legacy_params_file_is_flagged_unreadable_and_the_rest_still_list(tmp_path: Path):
    root = tmp_path / "experiments"
    _experiment(root, "legacy", _LEGACY)
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


def test_the_verdict_reaches_the_listing_only_after_the_reveal(tmp_path: Path):
    root = tmp_path / "experiments"
    directory = _experiment(root, "graduate", _CURRENT)
    write_json_atomic(
        directory / "hitl/schedule.json",
        {
            "schema_version": 1,
            "sessions": [
                {
                    "key": "epoch_001/fold_20220101..20251231",
                    "kind": "fold",
                    "epoch_id": "epoch_001",
                    "fold_id": "fold_20220101..20251231",
                },
                {
                    "key": "heldout",
                    "kind": "heldout",
                    "epoch_id": "epoch_001",
                    "periods": [
                        {"label": "20260101..20260630", "start": "20260101", "end": "20260630"}
                    ],
                },
            ],
        },
    )
    ledger = ExperimentLedger(directory / "ledgers/experiment_ledger.jsonl")
    ledger.append(
        {
            "record_type": "fold",
            "experiment_id": "graduate",
            "epoch_id": "epoch_001",
            "fold_id": "fold_20220101..20251231",
            "run_id": "run_dev",
            "session_key": "epoch_001/fold_20220101..20251231",
            "fold_status": "frozen",
            "validation_period": "20220101..20251231",
            "test_period": None,
            "validation_result": {"total_return": 0.4},
            "test_result": None,
        }
    )
    # Before any held-out record: no verdict, nothing revealed.
    summary = summarize_experiment(directory)
    assert summary["test_revealed"] is False
    assert summary["verdict"] is None
    result = {
        "total_return": 0.06,
        "sharpe": 1.1,
        "max_drawdown": -0.08,
        "benchmark": {"label": "CSI 300", "benchmark_return": 0.02},
    }
    ledger.append(
        {
            "record_type": "heldout",
            "experiment_id": "graduate",
            "epoch_id": "epoch_001",
            "fold_id": "heldout_20260101..20260630",
            "run_id": "run_ho",
            "session_key": "heldout",
            "period": "20260101..20260630",
            "result": result,
            "verdict": AcceptanceRules().heldout_verdict(result),
        }
    )
    # Every planned held-out period is recorded: auto-revealed, verdict shown.
    summary = summarize_experiment(directory)
    assert summary["test_revealed"] is True
    assert summary["verdict"]["status"] == "graduated"
    assert summary["verdict"]["reasons"] == []
    assert summary["verdict"]["periods"][0]["period"] == "20260101..20260630"
    detail = TestClient(create_app(tmp_path)).get("/api/experiments/graduate").json()
    assert detail["verdict"]["status"] == "graduated"


def test_a_held_out_row_without_a_verdict_shows_no_verdict_in_the_console(tmp_path: Path):
    root = tmp_path / "experiments"
    directory = _experiment(root, "old_row", _CURRENT)
    write_json_atomic(
        directory / "hitl/schedule.json",
        {
            "schema_version": 1,
            "sessions": [
                {
                    "key": "heldout",
                    "kind": "heldout",
                    "epoch_id": "epoch_001",
                    "periods": [{"label": "2026Q2", "start": "20260401", "end": "20260630"}],
                }
            ],
        },
    )
    ExperimentLedger(directory / "ledgers/experiment_ledger.jsonl").append(
        {
            "record_type": "heldout",
            "experiment_id": "old_row",
            "epoch_id": "epoch_001",
            "fold_id": "heldout_2026Q2",
            "run_id": "run_ho",
            "session_key": "heldout",
            "period": "2026Q2",
            "result": {"total_return": 0.01},
        }
    )
    summary = summarize_experiment(directory)
    assert summary["state"] == "completed"
    assert summary["test_revealed"] is True
    assert summary["verdict"] is None


def test_the_create_form_is_seeded_with_the_yearly_fold_design(tmp_path: Path):
    schema = TestClient(create_app(tmp_path)).get("/api/parameter-schema").json()
    fields = {field["key"]: field for group in schema["groups"] for field in group["fields"]}
    assert fields["test_stage"]["type"] == "bool"
    assert fields["test_stage"]["default"] is False
    assert fields["development_first_period"]["required"] is True
    assert fields["development_last_period"]["required"] is True
    assert "first_test_period" not in fields and "last_test_period" not in fields
    for key, value in (
        ("epochs", 3),
        ("meta_learning_fold_interval", 1),
        ("window_months", 24),
        ("max_fold_minutes", 480),
        ("max_steps_per_fold", 20),
        ("max_backtests_per_fold", 20),
        ("max_llm_calls", 1200),
        ("screen_exclude_st", False),
        ("screen_exclude_new_listed_days", 0),
        ("screen_boards", []),
    ):
        assert fields[key]["default"] == value, key
        assert WEB_CREATE_DEFAULTS[key] == (tuple(value) if isinstance(value, list) else value), key
    # The console form pre-fills the calendar from the defaults, but a blank
    # development field is still a missing required field, not a silent default.
    response = TestClient(create_app(tmp_path)).post(
        "/api/experiments",
        json={"experiment_id": "no_calendar", "development_first_period": ""},
    )
    assert response.status_code == 400
    assert "development_first_period" in json.dumps(response.json())
    assert not (tmp_path / "experiments" / "no_calendar").exists()
