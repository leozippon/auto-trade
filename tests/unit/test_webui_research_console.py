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
    ENDING_STATES,
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


def test_every_way_an_arm_can_end_reads_as_one_ending_with_a_reason(tmp_path: Path) -> None:
    """The console shows an arm's ending in one vocabulary, so the projection
    must classify every ending the pipeline can produce and give each its own
    reason; an arm that can still run has none at all."""

    for experiment_id, stage in (
        ("graduated", "graduated"),
        ("rejected", "discarded"),
        ("no_edge", "no_deliverable"),
        ("budget_exhausted", "deadline"),
        ("broken", "broken"),
        ("running", "research"),
    ):
        build_arm(tmp_path, experiment_id, stage)
    rows = {row["experiment_id"]: row for row in list_experiments(tmp_path)}
    assert rows["running"]["ending"] is None
    endings = {name: rows[name]["ending"] for name in ENDING_STATES}
    assert {name: ending["state"] for name, ending in endings.items()} == {
        name: name for name in ENDING_STATES
    }
    # A graduate is named by the two numbers that carried it, a refusal by the
    # criteria it failed, and the three endings no replay decided by what the
    # record says: the Agent's first sentence, the budget, the error's first line.
    forward = rows["graduated"]["forward"]["slices"]
    assert endings["graduated"]["reason"] == (
        f"前推 80% 下界 {forward['forward']['lower_bound'] * 100:+.2f}%"
        f" · Held-out 超额 {forward['heldout']['neutralized_excess'] * 100:+.2f}%"
    )
    assert endings["rejected"]["reason"].split(" · ")[0] == "F2"
    assert endings["no_edge"]["reason"] == "没有候选值得冻结"
    assert endings["budget_exhausted"]["reason"] == "模型调用次数用尽"
    assert endings["broken"]["reason"] == "RuntimeError: sandbox image is gone"
    # The homepage lists what can still run first, then the endings in order.
    listed = [row["experiment_id"] for row in list_experiments(tmp_path)]
    assert listed[-len(ENDING_STATES) :] == list(ENDING_STATES)
    assert "running" in listed[: -len(ENDING_STATES)]
    # The detail page reads the same projection as the card.
    assert experiment_detail(tmp_path, "no_edge")["ending"] == endings["no_edge"]


def test_an_arm_whose_nomination_the_freeze_gate_refused_reads_as_rejected(
    tmp_path: Path,
) -> None:
    """The gate refusing a nomination ends the arm without a replay. It is not
    the Agent's own "no candidate was worth freezing", so it reads as the
    refusal it is rather than as 未发现超额."""

    directory = build_arm(tmp_path, "arm", "research")
    ExperimentLedger(directory / "ledgers/experiment_ledger.jsonl").append(
        _session_record(
            "arm",
            outcome="freeze",
            steps=[],
            trials_to_date=1,
            freeze_gate={"passed": False, "reasons": ["freeze_deflated_sharpe_below_threshold"]},
            arm_end={
                "status": "no_deliverable",
                "reason": "freeze refused by the gate (freeze_deflated_sharpe_below_threshold)",
            },
        )
    )
    assert summarize_experiment(directory)["ending"] == {
        "state": "rejected",
        "reason": "提名未通过冻结门",
    }


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
    # This arm's params.json was written before an account's capital derived
    # its limits: it names none of them, and the console attributes none to it.
    assert preview["max_drawdown"] is None
    assert not {"active_max_drawdown", "tracking_error_cap"} & set(preview)
    assert preview["cost_stress_multiplier"] == pytest.approx(WEB_CREATE_DEFAULTS["cost_stress_multiplier"])
    # One created since states all five, as null where it overrides nothing,
    # and is listed with what its capital derives and what it overrides.
    path = tmp_path / "researching/hitl/params.json"
    params = json.loads(path.read_text(encoding="utf-8"))
    derived = dict.fromkeys(
        ("max_drawdown", "active_max_drawdown", "tracking_error_cap", "beta_min", "beta_max")
    )
    path.write_text(
        json.dumps({**params, **derived, "initial_cash": 1_000_000, "max_drawdown": 0.3}),
        encoding="utf-8",
    )
    stated = experiment_detail(tmp_path, "researching")["sessions"][1]["thresholds"]
    assert {key: stated[key] for key in derived} == {
        "max_drawdown": 0.3,
        "active_max_drawdown": 0.15,
        "tracking_error_cap": 0.08,
        "beta_min": 0.85,
        "beta_max": 1.15,
    }
    path.write_text(json.dumps({**params, **derived, "initial_cash": 100_000}), encoding="utf-8")
    small = experiment_detail(tmp_path, "researching")["sessions"][1]["thresholds"]
    assert (small["max_drawdown"], small["active_max_drawdown"]) == (0.45, 0.30)
    assert small["tracking_error_cap"] is None
    path.write_text(json.dumps(params), encoding="utf-8")
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


def test_only_a_graduate_is_the_best_experiment(tmp_path: Path) -> None:
    for experiment_id, stage in (
        ("discarded", "discarded"),
        ("researching", "research"),
        ("no_edge", "no_deliverable"),
        ("graduated", "graduated"),
    ):
        build_arm(tmp_path, experiment_id, stage)
    rows = list_experiments(tmp_path)
    assert best_experiment(rows) == {"experiment_id": "graduated"}
    # Every other ending is an arm the pipeline refused: with no graduate the
    # homepage crowns nobody, whatever bound the refused arm carries.
    without_graduate = [row for row in rows if row["experiment_id"] != "graduated"]
    assert best_experiment(without_graduate) is None
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


def test_the_card_and_the_research_panel_name_the_same_four_figures() -> None:
    """One vocabulary for the arm's evidence: the experiment card and the
    research session panel draw the same four tiles, in the same order, with
    the gate threshold and the deflation denominator explained once."""

    script = (
        Path(__file__).resolve().parents[2] / "src/autotrade/webui/static/app.js"
    ).read_text(encoding="utf-8")
    labels = ("中性化超额", "IR", "DSR", "累计试验")
    for opening in ("function evidenceTiles(", "function researchSessionPanel("):
        body = script.split(opening, 1)[1].split("\nfunction ", 1)[0]
        found = [label for label in labels if f'label: "{label}"' in body or f'"最佳候选{label}"' in body]
        assert found == list(labels), opening
        assert "DSR_TITLE" in body and "TRIALS_TITLE" in body, opening
    # The gloss says what the figure is; the gate's own limits and measured
    # values stay in the checklist, which reads both from the record.
    [dsr_gloss] = [row for row in script.splitlines() if row.startswith("const DSR_TITLE")]
    assert "≥" not in dsr_gloss, dsr_gloss
    assert "全区间验证次数" in script.split("const REASON_LABELS", 1)[1]


def test_a_figure_whose_label_does_not_read_itself_is_glossed_once() -> None:
    """IR is an acronym over a formula, 中性化超额 and 残差跟踪误差 name what was
    regressed out, and the 下界 is a bootstrap percentile: none of the four can
    be read off its own label. IR sat bare on all four of its sites while the
    DSR beside it was fully explained. Each gloss is one shared constant, so no
    site can drift, and it hangs on the same `title` hover the console uses
    everywhere else — including on a checklist row, which had no slot for one.
    """

    script = (
        Path(__file__).resolve().parents[2] / "src/autotrade/webui/static/app.js"
    ).read_text(encoding="utf-8")
    for name in ("IR_TITLE", "NEUTRALIZED_EXCESS_TITLE", "TRACKING_ERROR_TITLE", "LOWER_BOUND_TITLE"):
        assert script.count(f"const {name} = ") == 1, name
    for opening, glosses in (
        ("function evidenceTiles(", ("IR_TITLE",)),
        ("function forwardTiles(", ("LOWER_BOUND_TITLE", "NEUTRALIZED_EXCESS_TITLE")),
        ("function researchSessionPanel(", ("IR_TITLE", "NEUTRALIZED_EXCESS_TITLE")),
        ("function frozenPanel(", ("IR_TITLE", "NEUTRALIZED_EXCESS_TITLE", "TRACKING_ERROR_TITLE")),
        ("function freezeGateChecklist(", ("DSR_TITLE",)),
        ("function forwardCriteria(", ("LOWER_BOUND_TITLE",)),
    ):
        body = script.split(opening, 1)[1].split("\nfunction ", 1)[0]
        for gloss in glosses:
            assert gloss in body, (opening, gloss)
    assert "item.title" in script.split("function checklist(", 1)[1]


def test_an_experiment_card_wears_the_ending_as_a_badge_alone() -> None:
    """The homepage grid stays scannable: a card says how the arm ended in one
    badge and keeps the reason in that badge's tooltip. The sentence itself is
    drawn only on the experiment page — its header and the 裁决 view."""

    script = (
        Path(__file__).resolve().parents[2] / "src/autotrade/webui/static/app.js"
    ).read_text(encoding="utf-8")
    card = script.split("function experimentCard(", 1)[1].split("\nfunction ", 1)[0]
    badge = script.split("function endingBadge(", 1)[1].split("\nfunction ", 1)[0]
    assert "armBadge(item)" in card
    assert "endingReason" not in card
    assert "title: ending.reason || null" in badge
    # The definition and its two call sites, and no third page.
    assert script.count("endingReason(") == 3


def test_the_three_replay_stage_views_speak_one_criteria_vocabulary() -> None:
    """前推回放, Held-out and 裁决 draw the criteria the same way: every view
    renders the shared F1–F6 / H1–H4 rows as a checklist, so a threshold is
    spelled once and the verdict cannot drift back into a chip row of its own
    that states the same limits differently."""

    script = (
        Path(__file__).resolve().parents[2] / "src/autotrade/webui/static/app.js"
    ).read_text(encoding="utf-8")
    bodies = {
        name: script.split(f"function {name}(", 1)[1].split("\nfunction ", 1)[0]
        for name in ("forwardStagePanel", "heldoutStagePanel", "verdictStagePanel")
    }
    assert "checklist(forwardCriteria(" in bodies["forwardStagePanel"]
    assert "checklist(heldoutCriteria(" in bodies["heldoutStagePanel"]
    verdict = bodies["verdictStagePanel"]
    assert "forwardCriteria(" in verdict and "heldoutCriteria(" in verdict
    # The thresholds reach the verdict as those rows' own threshold column.
    assert "chip" not in verdict and "thresholdChips" not in script


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
    # The card's four evidence tiles: a measurable candidate carries every
    # figure, so the card never draws a partial row.
    evidence = (
        "neutralized_excess",
        "information_ratio",
        "deflated_sharpe_probability",
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
