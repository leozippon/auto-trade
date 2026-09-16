"""The research console over a research arm's ledger.

An arm runs one research session, freezes at most once and is then replayed once
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
from tests.unit.webui_research_arm import REPLAY, _session_record, _step, build_arm


def _records(directory: Path) -> list[dict[str, object]]:
    return ExperimentLedger(directory / "ledgers/experiment_ledger.jsonl").read()


def _result_names(root: Path, experiment_id: str, mode: str) -> list[str]:
    return sorted(
        path.name
        for path in (root / experiment_id / "artifacts/results").glob(f"{mode}_*")
    )


def test_a_researching_arm_shows_its_plan_and_nothing_frozen(tmp_path: Path) -> None:
    build_arm(tmp_path, "arm", "research")
    detail = experiment_detail(tmp_path, "arm")
    assert detail["stage"] == "research"
    assert detail["research_outcome"] is None
    assert detail["verdict"] is None
    assert detail["forward"] is None
    assert detail["frozen"] is None
    assert detail["paper_candidate"] is None
    assert [session["key"] for session in detail["sessions"]] == ["research", "forward"]
    assert [session["kind"] for session in detail["sessions"]] == ["research", "forward"]
    # The session in flight has no ledger record yet; its plan entry stands.
    assert "record" not in detail["sessions"][0]
    assert detail["sessions"][1]["replay"] == REPLAY


def test_the_best_candidate_carries_the_deflated_sharpe_the_gate_gave_the_nominee(
    tmp_path: Path,
) -> None:
    """One source for the number: the session row's probability for its
    nominee equals the one the freeze gate recorded when it froze."""

    build_arm(tmp_path, "arm", "sealed")
    record = experiment_detail(tmp_path, "arm")["sessions"][0]["record"]
    assert record["outcome"] == "freeze"
    assert record["froze"] is True
    assert record["attempts"] == 1
    assert record["budget_used"]["replay_years"] == 8
    assert record["best"]["step_id"] == record["nominated_step_id"]
    assert record["best"]["trials"] == 3
    # Research results are development evidence: readable as they land.
    client = TestClient(create_app(tmp_path, tmp_path))
    name = Path(str(_records(tmp_path / "arm")[0]["steps"][0]["validation_result_ref"])).parent.name
    assert client.get(f"/api/experiments/arm/results/{name}/equity").status_code == 200
    # The record says whether its trace is on disk; this synthetic arm has none.
    assert record["trace"] is False
    ledger_gate = _records(tmp_path / "arm")[0]["freeze_gate"]
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
    assert detail["status"]["session_key"] == "forward"
    assert detail["verdict"] is None
    assert detail["forward"] is None
    assert detail["paper_candidate"] is None
    frozen = detail["frozen"]
    assert frozen["session_key"] == "research"
    assert frozen["strategy_ref"].startswith("strategy_ref_")
    assert "strategy_research_abc" not in json.dumps(detail)
    assert frozen["full_span_validations"] == 2
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
            "artifact_id": "strategy_research_abc",
            "command": (
                "python scripts/paper/run_paper.py init --experiment arm "
                "--artifact strategy_research_abc"
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
    assert detail["sessions"][0]["record"]["arm_end"]["status"] == "no_deliverable"
    assert detail["sessions"][0]["record"]["best"] is None


def test_the_listing_carries_the_budget_and_the_research_outcome(tmp_path: Path) -> None:
    """The budget bars read the arm's ceilings (worker defaults under the
    params) and what the session spent: the ledger's block once recorded,
    else the last block of the live session's trace, else nothing."""

    live = build_arm(tmp_path, "live", "research", alive=True)
    build_arm(tmp_path, "idle", "research")
    build_arm(tmp_path, "judged", "graduated")
    params = json.loads((live / "hitl/params.json").read_text(encoding="utf-8"))
    (live / "hitl/params.json").write_text(json.dumps({**params, "max_research_minutes": 60}), encoding="utf-8")
    traces = live / "artifacts/traces"
    traces.mkdir(parents=True)
    (traces / "run_research_live.jsonl").write_text(
        "\n".join(
            json.dumps(event)
            for event in (
                {"event_type": "session_start"},
                {"event_type": "llm_call", "budget_used": {"inference_seconds": 900.0, "llm_calls": 7, "replay_years": 4, "null_controls": 0}},
            )
        )
        + "\n",
        encoding="utf-8",
    )
    judged_traces = tmp_path / "judged/artifacts/traces"
    judged_traces.mkdir(parents=True)
    (judged_traces / "run_research.jsonl").write_text('{"event_type": "session_start"}\n', encoding="utf-8")
    assert experiment_detail(tmp_path, "judged")["sessions"][0]["record"]["trace"] is True
    rows = {row["experiment_id"]: row for row in list_experiments(tmp_path)}
    assert rows["live"]["budget"] == {"inference_seconds": 3600.0, "llm_calls": 6400.0, "replay_years": 96.0, "null_controls": 12.0}
    assert rows["live"]["budget_used"] == {"inference_seconds": 900.0, "llm_calls": 7, "replay_years": 4, "null_controls": 0}
    assert rows["idle"]["budget_used"] is None
    assert rows["judged"]["budget_used"]["replay_years"] == 8
    assert (rows["live"]["research_outcome"], rows["judged"]["research_outcome"]) == (None, "freeze")
    client = TestClient(create_app(tmp_path.parent, tmp_path))
    assert client.get("/api/experiments/live/status").json()["budget_used"]["llm_calls"] == 7
    assert client.get("/api/experiments/idle/status").json()["budget_used"] is None


def test_the_listing_names_the_research_curve_and_the_replay_its_thresholds(tmp_path: Path) -> None:
    """The one curve per arm starts from the research-period validation of the
    candidate the tiles name — the frozen artifact's own once the arm froze —
    and the replay's step lists the thresholds the verdict will hold it to
    before the record exists, from the arm's parameters and the pipeline's
    constants; the record's own block carries the same keys afterwards."""

    from autotrade.pipelines.hitl_state import WEB_CREATE_DEFAULTS

    build_arm(tmp_path, "researching", "research")
    build_arm(tmp_path, "frozen", "sealed")
    build_arm(tmp_path, "judged", "graduated")
    rows = {row["experiment_id"]: row for row in list_experiments(tmp_path)}
    assert rows["researching"]["research_result"] is None
    frozen_ref = _records(tmp_path / "frozen")[0]["frozen"]["research_result_ref"]
    assert rows["frozen"]["research_result"] == Path(str(frozen_ref)).parent.name
    # This arm nominated its best candidate, so both names agree.
    assert rows["frozen"]["research_best"]["result"] == rows["frozen"]["research_result"]

    preview = experiment_detail(tmp_path, "researching")["sessions"][1]["thresholds"]
    assert preview["max_drawdown"] == pytest.approx(WEB_CREATE_DEFAULTS["max_drawdown"])
    assert preview["cost_stress_multiplier"] == pytest.approx(WEB_CREATE_DEFAULTS["cost_stress_multiplier"])
    assert (preview["min_round_trips"], preview["min_mean_gross"], preview["recency_months"]) == (12, 0.5, 6)
    # The record's own block carries the same keys; its parameter-derived
    # values are the arm's (the synthetic arm was judged at a 90% drawdown cap).
    recorded = experiment_detail(tmp_path, "judged")["forward"]["verdict"]["thresholds"]
    assert set(preview) <= set(recorded)
    for key in ("forward_confidence", "recency_months", "min_round_trips", "min_mean_gross", "heldout_tolerance_z"):
        assert recorded[key] == pytest.approx(preview[key]), key


def _sidecar(directory: Path, row: dict[str, object]) -> Path:
    """The host sidecar the session writes for one recorded Validation."""

    path = directory / ".host/steps" / f"{row['step_id']}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "step_id": row["step_id"],
                "revision_id": row["revision_id"],
                "span": row["span"],
                "summary": row["summary"],
                "result_ref": row["validation_result_ref"],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_a_running_session_serves_its_best_candidate_as_the_record_will(tmp_path: Path) -> None:
    """While the one research session runs, the listing reads the Validations
    it has recorded so far from the host sidecars and names the best
    full-span candidate, its curve and its result by the rule the recorded
    session is read with, so nothing changes when the record lands."""

    directory = build_arm(tmp_path, "arm", "research", alive=True)
    steps = [
        _step(directory, "research", 0, edge=0.0005, seed=1),
        _step(directory, "research", 1, edge=0.002, seed=3),
        _step(directory, "research", 2, edge=0.001, seed=4, span="Y3"),
    ]
    for row in steps:
        _sidecar(directory, row)
    live = summarize_experiment(directory)
    assert live["research_best"]["step_id"] == steps[1]["step_id"]
    assert live["research_best"]["session_key"] == "research"
    assert live["research_best"]["trials"] == 3
    assert live["research_result"] == Path(str(steps[1]["validation_result_ref"])).parent.name
    client = TestClient(create_app(tmp_path, tmp_path))
    assert client.get(f"/api/experiments/arm/results/{live['research_result']}/equity").status_code == 200
    # The replay stays sealed: a result no record or sidecar names is absent.
    assert client.get("/api/experiments/arm/results/heldout_nope/equity").status_code == 404

    ExperimentLedger(directory / "ledgers/experiment_ledger.jsonl").append(
        _session_record(
            "arm",
            outcome="no_edge",
            steps=steps,
            trials_to_date=3,
            arm_end={"status": "no_deliverable", "reason": "no_edge: nothing survived"},
        )
    )
    recorded = summarize_experiment(directory)
    assert recorded["research_best"] == live["research_best"]
    assert recorded["research_result"] == live["research_result"]


def test_a_recorded_node_is_read_once_across_listings(tmp_path: Path) -> None:
    """A Validation is immutable once recorded: its sidecar and style file are
    read on the first listing and never again, and the best candidate is kept
    per node set, so a poll re-reads nothing until another node is recorded."""

    directory = build_arm(tmp_path, "arm", "research", alive=True)
    first = _step(directory, "research", 0, edge=0.002, seed=3)
    _sidecar(directory, first)
    before = summarize_experiment(directory)["research_best"]
    assert before["step_id"] == first["step_id"]
    (Path(str(first["validation_result_ref"])).parent / "style_analysis.json").unlink()
    assert summarize_experiment(directory)["research_best"] == before
    # A newly recorded node is read, and the best is chosen over the new set.
    second = _step(directory, "research", 1, edge=0.004, seed=5)
    _sidecar(directory, second)
    assert summarize_experiment(directory)["research_best"]["step_id"] == second["step_id"]


def test_an_arm_with_no_plan_yet_lists_as_created(tmp_path: Path) -> None:
    build_arm(tmp_path, "arm", "created")
    row = summarize_experiment(tmp_path / "arm")
    assert row["state"] == "created"
    assert row["stage"] == "research"
    assert row["research_outcome"] is None
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
    assert best_experiment(rows) == {"experiment_id": "graduated"}
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


def test_a_running_arm_is_never_the_best_experiment(tmp_path: Path) -> None:
    """Ranking running arms by how far they had got crowned one of a field that
    had produced no out-of-sample evidence at all. With nothing judged the
    homepage names no best experiment."""

    build_arm(tmp_path, "early", "research", alive=True)
    build_arm(tmp_path, "sealed", "sealed", alive=True)
    rows = list_experiments(tmp_path)
    assert best_experiment(rows) is None
    listed = TestClient(create_app(tmp_path, tmp_path)).get("/api/experiments").json()
    assert listed["best"] is None
    # …and the homepage draws no hero for a null answer, rather than an empty
    # crowned panel above the listing.
    script = (
        Path(__file__).resolve().parents[2] / "src/autotrade/webui/static/app.js"
    ).read_text(encoding="utf-8")
    assert "if (!best) return null;" in script.split("function bestRow(", 1)[1]
    home = script.split("function homeView(", 1)[1].split("\nfunction ", 1)[0]
    assert "if (best)" in home and "heroPanel(best)" in home


def test_the_listing_carries_the_freeze_and_the_best_candidate_so_far(
    tmp_path: Path,
) -> None:
    """The experiment card draws its freeze step and its research evidence from
    the listing alone, and must read the same numbers the experiment page does."""

    build_arm(tmp_path, "researching", "research")
    build_arm(tmp_path, "frozen", "sealed")
    build_arm(tmp_path, "fresh", "created")
    rows = {row["experiment_id"]: row for row in list_experiments(tmp_path)}
    assert rows["researching"]["frozen_session"] is None
    assert rows["frozen"]["frozen_session"] == "research"
    # An arm without a recorded session has no record to read a candidate off.
    assert rows["fresh"]["research_best"] is None
    assert rows["researching"]["research_best"] is None
    best = rows["frozen"]["research_best"]
    assert best["session_key"] == "research"
    session_best = experiment_detail(tmp_path, "frozen")["sessions"][0]["record"]["best"]
    # The card's four evidence tiles, the last one a pair: a measurable
    # candidate carries every figure, so the card never draws a partial row.
    evidence = (
        "neutralized_excess",
        "information_ratio",
        "deflated_sharpe_probability",
        "full_span_validations",
        "trials",
    )
    for field in ("step_id", *evidence):
        assert best[field] == session_best[field], field
        assert best[field] is not None, field


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


def test_the_step_tree_names_the_session_and_the_frozen_node(tmp_path: Path) -> None:
    directory = build_arm(tmp_path, "arm", "sealed")
    refs = AgentRefStore(directory)
    frozen = _records(directory)[0]["frozen"]
    output = Path(frozen["output_path"])
    tree = StepTree(directory / "steps")
    node_id = tree.record_step(
        output,
        epoch_id="research",
        session_ref=refs.get_or_create("session", "research"),
        run_id=refs.get_or_create("run", "run_research"),
        result_name="valid_000",
        revision_id=refs.get_or_create("strategy", "revision_research_0"),
        metrics={"total_return": 0.1},
    )
    ledger = ExperimentLedger(directory / "ledgers/experiment_ledger.jsonl")
    records = ledger.read()
    records[0]["frozen"]["source_step_id"] = node_id
    ledger.rewrite(records)
    [node] = step_tree_view(directory)["nodes"]
    assert node["session_key"] == "research"
    assert node["frozen"] is True
    assert node["session_ref"].startswith("session_ref_")
