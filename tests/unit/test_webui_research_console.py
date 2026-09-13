"""The research console over a research arm's ledger.

An arm researches in sessions, freezes at most once and is then replayed once
over forward and Held-out. The console must say where the arm is from the
ledger alone, keep every number of the replay sealed until the forward record
(and with it the verdict) exists, rank the best experiment on that verdict
only, and refuse a create whose geometry the worker would refuse.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from autotrade.environment.identity import AgentRefStore
from autotrade.environment.step_tree import StepTree
from autotrade.pipelines.config import DEFAULT_RESEARCH_GEOMETRY
from autotrade.pipelines.ledger import ExperimentLedger
from autotrade.webui.registry import (
    best_experiment,
    experiment_detail,
    list_experiments,
    summarize_experiment,
)
from autotrade.webui.server import create_app
from autotrade.webui.steps import step_tree_view
from tests.unit.webui_research_arm import REPLAY, SESSIONS, build_arm


def _records(directory: Path) -> list[dict[str, object]]:
    return ExperimentLedger(directory / "ledgers/experiment_ledger.jsonl").read()


def _result_names(root: Path, experiment_id: str, mode: str) -> list[str]:
    return sorted(
        path.name
        for path in (root / experiment_id / "artifacts/results").glob(f"{mode}_*")
    )


def test_a_researching_arm_shows_its_sessions_and_nothing_frozen(tmp_path: Path) -> None:
    build_arm(tmp_path, "arm", "research")
    detail = experiment_detail(tmp_path, "arm")
    assert detail["stage"] == "research"
    assert (detail["research_recorded"], detail["research_total"]) == (1, SESSIONS)
    assert detail["verdict"] is None
    assert detail["forward"] is None
    assert detail["frozen"] is None
    assert detail["paper_candidate"] is None
    assert [session["key"] for session in detail["sessions"]] == ["s1", "s2", "s3", "forward"]
    assert [session["kind"] for session in detail["sessions"]] == [
        "research",
        "research",
        "research",
        "forward",
    ]
    s1 = detail["sessions"][0]["record"]
    assert s1["outcome"] == "continue"
    assert len(s1["validations"]) == 2
    assert s1["trials_to_date"] == 2
    assert s1["froze"] is False
    assert s1["prior"] == "PRIOR after s1"
    # The best candidate is the full-span Validation with the highest
    # neutralised IR, read off the ledger rows.
    rows = _records(tmp_path / "arm")[0]["steps"]
    best_row = max(rows, key=lambda row: row["neutralized"]["information_ratio"])
    assert s1["best"]["step_id"] == best_row["step_id"]
    assert s1["best"]["neutralized_excess"] == best_row["neutralized"]["neutralized_excess"]
    assert s1["best"]["trials"] == 2
    assert 0.0 <= s1["best"]["deflated_sharpe_probability"] <= 1.0
    assert "record" not in detail["sessions"][1]
    assert detail["sessions"][3]["replay"] == REPLAY
    # Research results are development evidence: readable as they land.
    client = TestClient(create_app(tmp_path, tmp_path))
    name = s1["validations"][0]["result"]
    assert client.get(f"/api/experiments/arm/results/{name}/equity").status_code == 200


def test_the_best_candidate_carries_the_deflated_sharpe_the_gate_gave_the_nominee(
    tmp_path: Path,
) -> None:
    """One source for the number: the session row's probability for its
    nominee equals the one the freeze gate recorded when it froze."""

    build_arm(tmp_path, "arm", "sealed")
    record = experiment_detail(tmp_path, "arm")["sessions"][1]["record"]
    assert record["outcome"] == "freeze"
    assert record["froze"] is True
    assert record["best"]["step_id"] == record["nominated_step_id"]
    ledger_gate = _records(tmp_path / "arm")[1]["freeze_gate"]
    assert record["best"]["deflated_sharpe_probability"] == pytest.approx(
        ledger_gate["deflated_sharpe"]["deflated_sharpe_probability"]
    )
    assert record["freeze_gate"]["deflated_sharpe_probability"] == pytest.approx(
        ledger_gate["deflated_sharpe"]["deflated_sharpe_probability"]
    )
    # A sub-span Validation is listed but never the best full-span candidate.
    assert {row["span"] for row in record["validations"]} == {"full", "Y3"}


def test_a_frozen_arm_is_sealed_until_its_verdict_exists(tmp_path: Path) -> None:
    directory = build_arm(tmp_path, "arm", "sealed", alive=True)
    detail = experiment_detail(tmp_path, "arm")
    assert detail["stage"] == "forward"
    assert detail["current_session"] == "forward"
    assert detail["verdict"] is None
    assert detail["forward"] is None
    assert detail["paper_candidate"] is None
    frozen = detail["frozen"]
    assert frozen["session_key"] == "s2"
    assert frozen["strategy_ref"].startswith("strategy_ref_")
    assert "strategy_s2_abc" not in json.dumps(detail)
    assert frozen["full_span_validations"] == 3
    assert frozen["null_percentile"] == 0.81
    assert frozen["forward_mde"] > 0
    assert frozen["blocks"][0]["label"] == "202407-202506"
    # The running replay's result directory is on disk, and nothing serves it:
    # it answers exactly like a result that does not exist.
    [running] = _result_names(tmp_path, "arm", "heldout")
    client = TestClient(create_app(tmp_path, tmp_path))
    for route in ("equity", "style", "orders", "orders.csv"):
        sealed = client.get(f"/api/experiments/arm/results/{running}/{route}")
        absent = client.get(f"/api/experiments/arm/results/heldout_absent/{route}")
        assert sealed.status_code == absent.status_code == 404, route
        assert sealed.json()["detail"].replace(running, "") == absent.json()[
            "detail"
        ].replace("heldout_absent", "")
    # The listing and status carry the stage only.
    for payload in (
        client.get("/api/experiments").text,
        client.get("/api/experiments/arm").text,
        client.get("/api/experiments/arm/status").text,
    ):
        assert running not in payload
        assert "lower_bound" not in payload
    # The frozen research result stays readable.
    assert client.get(f"/api/experiments/arm/results/{frozen['result']}/style").status_code == 200
    assert (directory / "artifacts/results" / running).is_dir()


@pytest.mark.parametrize("status", ["graduated", "discarded"])
def test_the_verdict_opens_the_forward_replay(tmp_path: Path, status: str) -> None:
    build_arm(tmp_path, "arm", status)
    detail = experiment_detail(tmp_path, "arm")
    assert detail["stage"] == "verdict"
    assert detail["verdict"]["status"] == status
    record = _records(tmp_path / "arm")[-1]
    assert detail["verdict"]["reasons"] == record["verdict"]["reasons"]
    forward = detail["forward"]
    assert forward["replay"] == REPLAY
    assert forward["verdict"]["status"] == status
    assert forward["slices"]["forward"]["lower_bound"] == record["slices"]["forward"]["lower_bound"]
    assert forward["slices"]["heldout"]["tolerance"] == record["slices"]["heldout"]["tolerance"]
    assert forward["null_percentile"] == 0.77
    assert "result_ref" not in json.dumps(detail)
    client = TestClient(create_app(tmp_path, tmp_path))
    equity = client.get(f"/api/experiments/arm/results/{forward['result']}/equity").json()
    [strategy] = equity["series"]
    assert strategy["dates"][0] == REPLAY["start"]
    assert equity["benchmark"]["dates"] == strategy["dates"]
    assert equity["exposure"]["strategy"]["long"][0] == pytest.approx(0.9)
    orders = client.get(f"/api/experiments/arm/results/{forward['result']}/orders").json()
    assert orders["row_count"] == 2
    if status == "graduated":
        assert detail["verdict"]["reasons"] == []
        assert detail["paper_candidate"] == {
            "artifact_id": "strategy_s2_abc",
            "command": (
                "python scripts/paper/run_paper.py init --experiment arm "
                "--artifact strategy_s2_abc"
            ),
        }
    else:
        assert "forward_lower_bound_not_positive" in detail["verdict"]["reasons"]
        assert detail["paper_candidate"] is None


def test_a_research_that_froze_nothing_has_a_verdict_and_no_replay(tmp_path: Path) -> None:
    build_arm(tmp_path, "arm", "no_deliverable")
    detail = experiment_detail(tmp_path, "arm")
    assert detail["stage"] == "verdict"
    assert detail["verdict"] == {
        "status": "no_deliverable",
        "reasons": ["no_edge: nothing survived"],
    }
    assert detail["frozen"] is None
    assert detail["forward"] is None
    assert detail["sessions"][1]["record"]["arm_end"]["status"] == "no_deliverable"
    assert detail["sessions"][1]["record"]["best"] is None


def test_an_arm_with_no_plan_yet_lists_as_created(tmp_path: Path) -> None:
    build_arm(tmp_path, "arm", "created")
    row = summarize_experiment(tmp_path / "arm")
    assert row["state"] == "created"
    assert row["stage"] == "research"
    assert (row["research_recorded"], row["research_total"]) == (0, None)
    assert experiment_detail(tmp_path, "arm")["sessions"] == []


def test_a_fold_era_plan_is_listed_as_unreadable(tmp_path: Path) -> None:
    directory = build_arm(tmp_path, "legacy", "research")
    (directory / "hitl/schedule.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "sessions": [
                    {"session_key": "epoch_001/fold_2024Q1", "kind": "fold", "epoch_id": "epoch_001"}
                ],
            }
        ),
        encoding="utf-8",
    )
    build_arm(tmp_path, "current", "research")
    rows = {row["experiment_id"]: row for row in list_experiments(tmp_path)}
    assert rows["legacy"]["state"] == "unreadable"
    assert rows["current"]["stage"] == "research"


def _lower_bound(row: dict[str, object]) -> float:
    return row["forward"]["slices"]["forward"]["lower_bound"]  # type: ignore[index]


def test_the_best_experiment_is_ranked_on_the_forward_verdict_only(tmp_path: Path) -> None:
    for experiment_id, stage in (
        ("discarded", "discarded"),
        ("researching", "research"),
        ("no_edge", "no_deliverable"),
        ("graduated", "graduated"),
    ):
        build_arm(tmp_path, experiment_id, stage)
    rows = list_experiments(tmp_path)
    assert best_experiment(rows) == {
        "experiment_id": "graduated",
        "basis": "forward_lower_bound",
    }
    # Without a graduate, another verdict-bearing arm ranks by the same number;
    # an arm whose verdict carries none has nothing to rank by.
    without_graduate = [row for row in rows if row["experiment_id"] != "graduated"]
    assert best_experiment(without_graduate)["experiment_id"] == "discarded"
    # A graduate always outranks a discarded arm, whatever the bounds.
    graduated = next(row for row in rows if row["experiment_id"] == "graduated")
    discarded = next(row for row in rows if row["experiment_id"] == "discarded")
    discarded["forward"]["slices"]["forward"]["lower_bound"] = _lower_bound(graduated) + 1
    assert best_experiment([discarded, graduated])["experiment_id"] == "graduated"
    # Two graduates: the higher lower bound wins.
    second = json.loads(json.dumps(graduated))
    second["experiment_id"] = "graduated_better"
    second["forward"]["slices"]["forward"]["lower_bound"] = _lower_bound(graduated) + 0.1
    assert best_experiment([graduated, second])["experiment_id"] == "graduated_better"
    # A stopped arm without a verdict never ranks on research numbers.
    researching = [row for row in rows if row["experiment_id"] == "researching"]
    assert best_experiment(researching) is None


def test_running_arms_rank_by_stage_when_no_verdict_exists(tmp_path: Path) -> None:
    build_arm(tmp_path, "early", "research", alive=True)
    build_arm(tmp_path, "sealed", "sealed", alive=True)
    rows = list_experiments(tmp_path)
    assert best_experiment(rows) == {"experiment_id": "sealed", "basis": "stage"}
    listed = TestClient(create_app(tmp_path, tmp_path)).get("/api/experiments").json()
    assert listed["best"] == {"experiment_id": "sealed", "basis": "stage"}


@pytest.mark.parametrize(
    ("patch", "message"),
    [
        ({"research_start": "20210101"}, "July 1"),
        ({"forward_end": "20270630"}, "forward_end must be 20260630"),
        ({"heldout_end": "20260630"}, "Held-out follows the forward period"),
        ({"research_end": "20250631"}, "not a calendar date"),
    ],
)
def test_the_create_form_refuses_a_geometry_the_worker_refuses(
    tmp_path: Path, patch: dict[str, str], message: str
) -> None:
    response = TestClient(create_app(tmp_path)).post(
        "/api/experiments",
        json={
            "params": {
                "experiment_id": "geometry",
                **DEFAULT_RESEARCH_GEOMETRY.to_record(),
                **patch,
            }
        },
    )
    assert response.status_code == 400
    assert message in response.json()["detail"]
    assert not (tmp_path / "experiments/geometry").exists()


def test_the_step_tree_names_sessions_and_the_frozen_node(tmp_path: Path) -> None:
    directory = build_arm(tmp_path, "arm", "sealed")
    refs = AgentRefStore(directory)
    frozen = _records(directory)[1]["frozen"]
    output = Path(frozen["output_path"])
    tree = StepTree(directory / "steps")
    node_id = tree.record_step(
        output,
        epoch_id="research",
        fold_id=refs.get_or_create("fold", "s2"),
        run_id=refs.get_or_create("run", "run_s2"),
        result_name="valid_000",
        revision_id=refs.get_or_create("strategy", "revision_s2_0"),
        metrics={"total_return": 0.1},
    )
    ledger = ExperimentLedger(directory / "ledgers/experiment_ledger.jsonl")
    records = ledger.read()
    records[1]["frozen"]["source_step_id"] = node_id
    ledger.rewrite(records)
    [node] = step_tree_view(directory)["nodes"]
    assert node["session_key"] == "s2"
    assert node["frozen"] is True
    assert node["fold_ref"].startswith("fold_ref_")
