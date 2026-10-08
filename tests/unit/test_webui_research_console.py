"""The research console over a research arm's ledger.

An arm runs one research session, freezes at most once and is then replayed once
over forward and Held-out. The console must say where the arm is from the
ledger alone, keep every number of the replay sealed until the forward record
(and with it the verdict) exists, rank the best experiment on that verdict
only, and refuse a create whose geometry the worker would refuse.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from autotrade.environment.identity import AgentRefStore
from autotrade.environment.runtime import write_json_atomic
from autotrade.environment.step_tree import StepTree
from autotrade.environment.tools.report_issue import (
    append_issue_report,
    append_issue_resolution,
    issue_reports_path,
)
from autotrade.pipelines.config import DEFAULT_RESEARCH_GEOMETRY, RETIRED_SWITCHES
from autotrade.pipelines.ledger import ExperimentLedger, verdict_void
from autotrade.webui import registry
from autotrade.webui.registry import (
    ENDING_STATES,
    best_experiment,
    experiment_detail,
    list_experiments,
    summarize_experiment,
)
from autotrade.webui.server import create_app
from autotrade.webui.steps import step_tree_view
from tests.unit.webui_research_arm import (
    ENVIRONMENT_BLOCKED_REASON,
    REPLAY,
    _session_record,
    _step,
    build_arm,
)


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


def test_a_result_without_a_style_sidecar_is_not_an_unknown_result(tmp_path: Path) -> None:
    """"This result has no style artifact" is an expected state of a real
    result -- one recorded before style analysis existed, one written under an
    older schema -- and the console has to tell it from "no such result". The
    two must differ in the payload, never only in the wording of an error."""

    directory = build_arm(tmp_path, "arm", "graduated")
    client = TestClient(create_app(tmp_path, tmp_path))
    name = experiment_detail(tmp_path, "arm")["frozen"]["result"]
    present = client.get(f"/api/experiments/arm/results/{name}/style")
    assert present.status_code == 200
    assert present.json()["available"] is True

    sidecar = directory / "artifacts/results" / name / "style_analysis.json"
    sidecar.unlink()
    absent = client.get(f"/api/experiments/arm/results/{name}/style")
    assert absent.status_code == 200
    assert absent.json() == {"available": False, "reason": "no_style_artifact"}

    sidecar.write_text(json.dumps({"schema_version": 0, "mode": "valid"}), encoding="utf-8")
    stale = client.get(f"/api/experiments/arm/results/{name}/style")
    assert stale.status_code == 200
    assert stale.json() == {"available": False, "reason": "no_style_artifact"}

    # A sidecar that is there but cannot be parsed is a load failure, and says
    # so rather than reading as a result that never had one.
    sidecar.write_text('{"schema_version": ', encoding="utf-8")
    corrupt = client.get(f"/api/experiments/arm/results/{name}/style")
    assert corrupt.status_code == 200
    assert corrupt.json() == {"available": False, "reason": "unreadable"}

    unknown = client.get("/api/experiments/arm/results/valid_absent/style")
    assert unknown.status_code == 404
    assert unknown.json()["detail"] == "unknown result: valid_absent"


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
        assert detail["paper_candidate"] == {"artifact_id": "strategy_research_abc"}
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
        ("environment_blocked", "environment_blocked"),
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
    # A graduate is named by its forward year as the holder reads it -- his
    # account, the benchmark, the account against the zero-skill panel with
    # no regression, then the neutralised IR the gate graded -- a refusal by
    # the criteria it failed, and the three endings no replay decided by what
    # the record says: the Agent's first sentence, the budget, the error's
    # first line.
    forward = rows["graduated"]["forward"]["slices"]["forward"]
    raw = forward["raw_readings"]
    assert endings["graduated"]["reason"] == (
        f"前推 账户 {raw['strategy_return'] * 100:+.2f}%"
        f" · 基准 {raw['benchmark_return'] * 100:+.2f}%"
        f" · 对面板（未回归）{raw['plain_selection'] * 100:+.2f}%"
        f" · 中性化主动 IR {forward['information_ratio']:.2f}"
    )
    assert endings["rejected"]["reason"].split(" · ")[0] == "F2"
    assert rows["graduated"]["paper_candidate"] is not None
    assert endings["no_edge"]["reason"] == "没有候选值得冻结"
    assert endings["budget_exhausted"]["reason"] == "模型调用次数用尽"
    # The host's reason in full, its error included: that is what to act on.
    assert endings["environment_blocked"]["reason"] == ENVIRONMENT_BLOCKED_REASON
    assert endings["broken"]["reason"] == "RuntimeError: sandbox image is gone"
    # The homepage lists what can still run first, then the endings in order.
    listed = [row["experiment_id"] for row in list_experiments(tmp_path)]
    assert listed[-len(ENDING_STATES) :] == list(ENDING_STATES)
    assert "running" in listed[: -len(ENDING_STATES)]
    # The detail page reads the same projection as the card.
    assert experiment_detail(tmp_path, "no_edge")["ending"] == endings["no_edge"]


def test_the_home_list_flags_a_blocked_arm_and_an_unresolved_issue(tmp_path: Path) -> None:
    """Two things put an attention line on an arm's card and nothing else
    does: the host ending its session on the environment, and an issue report
    its session filed that nobody has resolved while the arm can still run.
    Resolving the report clears the line even on a row the listing keeps
    without re-reading, and an arm that ended no longer carries its reports."""

    build_arm(tmp_path, "blocked", "environment_blocked")
    stopped = build_arm(tmp_path, "stopped", "research")
    ended = build_arm(tmp_path, "ended", "no_deliverable")

    def attention() -> dict[str, object]:
        rows = {row["experiment_id"]: row for row in list_experiments(tmp_path)}
        return {name: rows[name]["attention"] for name in ("blocked", "stopped", "ended")}

    assert attention() == {
        "blocked": {"reason": ENVIRONMENT_BLOCKED_REASON},
        "stopped": None,
        "ended": None,
    }
    summary = "五年种子在 2022-07-01 之后的回放全部失败。证据见下。"
    report = append_issue_report(
        issue_reports_path(stopped),
        {"report_id": "issue_a", "category": "data", "summary": summary, "evidence": "e"},
    )
    append_issue_report(
        issue_reports_path(ended),
        {"report_id": "issue_b", "category": "data", "summary": summary, "evidence": "e"},
    )
    flagged = attention()
    assert flagged["stopped"] == {"reason": "1 条问题报告未处置：五年种子在 2022-07-01 之后的回放全部失败"}
    assert flagged["ended"] is None
    append_issue_resolution(
        issue_reports_path(stopped), report_id=str(report["report_id"]), outcome="fixed", note="seed rebuilt"
    )
    assert attention()["stopped"] is None
    # The card draws the line; the ending keeps its own badge and colour.
    static = Path(__file__).resolve().parents[2] / "src/autotrade/webui/static"
    script = (static / "app.js").read_text(encoding="utf-8")
    card = script.split("function experimentCard(", 1)[1].split("\nfunction ", 1)[0]
    assert "attentionLine(item)" in card
    assert "item.attention" in script.split("function attentionLine(", 1)[1].split("\nfunction ", 1)[0]
    assert ".badge.ending-environment_blocked" in (static / "style.css").read_text(encoding="utf-8")


def test_a_withdrawn_graduation_reads_as_a_failure_everywhere(tmp_path: Path) -> None:
    """A graduate later found to have a material problem is failed, not a kind
    of graduate: it ends as a refusal, sorts among the refusals ahead of the
    endings nobody decided, is never the best experiment, and its 裁决 view
    still has who withdrew it, when, why and on what evidence."""

    for experiment_id, stage, created in (
        ("graduated", "graduated", "2026-09-10"),
        ("refused", "discarded", "2026-09-11"),
        ("withdrawn", "graduated", "2026-09-12"),
        ("no_edge", "no_deliverable", "2026-09-13"),
    ):
        directory = build_arm(tmp_path, experiment_id, stage)
        params = json.loads((directory / "hitl/params.json").read_text(encoding="utf-8"))
        write_json_atomic(directory / "hitl/params.json", {**params, "_created_at": f"{created}T00:00:00+00:00"})
    reason = "复跑计入股息税后研究期主动 IR 为 −1.80。毕业靠的是冻结种子的一次抽签。"
    ExperimentLedger(tmp_path / "withdrawn/ledgers/experiment_ledger.jsonl").append(
        verdict_void(
            "withdrawn",
            voided_by="operator",
            reason=reason,
            evidence_ref="logs/notes/live_readiness_20261004/dividend_tax_rerun",
        )
    )
    arms = {"graduated", "refused", "withdrawn", "no_edge"}
    rows = [row for row in list_experiments(tmp_path) if row["experiment_id"] in arms]
    withdrawn = next(row for row in rows if row["experiment_id"] == "withdrawn")
    # One ending word with the refusals; the reason says it had graduated and
    # why it no longer has, in the void's first sentence.
    assert withdrawn["ending"] == {
        "state": "rejected",
        "reason": "毕业后复核未通过：复跑计入股息税后研究期主动 IR 为 −1.80",
    }
    # Listed in the failures' group, newest first inside it, ahead of the
    # endings nobody decided whatever its date.
    assert [row["experiment_id"] for row in rows] == ["graduated", "withdrawn", "refused", "no_edge"]
    # Never crowned, even over a lower bound the graduate cannot match.
    withdrawn["forward"]["slices"]["forward"]["lower_bound"] = 1.0
    assert best_experiment(rows) == {"experiment_id": "graduated"}
    assert best_experiment([withdrawn]) is None
    # The replay's own record stays, and the 裁决 view has the void in full.
    assert withdrawn["forward"]["slices"]["forward"]["raw_readings"]["raw_excess"] is not None
    assert withdrawn["paper_candidate"] is None
    void = experiment_detail(tmp_path, "withdrawn")["verdict"]["void"]
    assert void["voided_by"] == "operator"
    assert void["reason"] == reason
    assert void["evidence_ref"] == "logs/notes/live_readiness_20261004/dividend_tax_rerun"
    assert void["recorded_at"]
    # The pages have no second word for it: no label, step state or badge
    # colour of its own.
    static = Path(__file__).resolve().parents[2] / "src/autotrade/webui/static"
    script = (static / "app.js").read_text(encoding="utf-8")
    for table in ("const ENDING_LABELS", "const ENDING_STEP_STATES"):
        assert "voided" not in script.split(table, 1)[1].split("};", 1)[0], table
    assert "毕业已作废" not in script
    assert "ending-voided" not in (static / "style.css").read_text(encoding="utf-8")


def test_seed_replicates_are_shown_where_the_record_has_them_and_absent_otherwise(
    tmp_path: Path,
) -> None:
    """An arm judged on its training seeds together shows each seed and their
    mean on the experiment page -- the freeze's seed-mean IR beside the
    nominee's and the bar, and the forward and Held-out readings per seed --
    and the series F8 judges them on together -- from the fields the console
    serves; an arm frozen on one seed serves none of them and draws nothing
    in their place."""

    build_arm(tmp_path, "seeded", "graduated", seeded=True)
    build_arm(tmp_path, "single", "graduated")
    seeded = experiment_detail(tmp_path, "seeded")
    single = experiment_detail(tmp_path, "single")
    gate = seeded["sessions"][0]["record"]["freeze_gate"]["seed_replicates"]
    frozen = seeded["frozen"]["seed_replicates"]
    forward = seeded["forward"]["slices"]["forward"]
    assert frozen == gate and gate["information_ratio_bar"] is not None
    assert gate["mean_information_ratio"] == pytest.approx(
        (seeded["frozen"]["information_ratio"] + gate["replicates"][0]["information_ratio"]) / 2
    )
    assert gate["replicates"][0]["seed_line"] == "main.py: SEED_BASE = 2000"
    for name in ("forward", "heldout"):
        block = seeded["forward"]["slices"][name]
        assert block["seed_mean"]["members"] == 2
        assert block["seed_mean"]["raw_readings"]["plain_selection"] == pytest.approx(
            (block["raw_readings"]["plain_selection"] + block["seed_replicates"][0]["raw_readings"]["plain_selection"]) / 2
        )
    assert forward["seed_mean"]["members"] == 2
    assert isinstance(forward["plain_excess_lower_bound"], float)
    assert registry._CRITERION_CODES["forward_plain_excess_lower_bound_not_positive"] == "F8"
    # One seed: no replicate served, so nothing drawn; F8 reads the book alone.
    assert single["frozen"]["seed_replicates"] is None
    alone = single["sessions"][0]["record"]["freeze_gate"]["seed_replicates"]
    assert (alone["replicates"], alone["mean_information_ratio"]) == ([], None)
    assert "seed_mean" not in single["forward"]["slices"]["forward"]
    assert isinstance(single["forward"]["slices"]["forward"]["plain_excess_lower_bound"], float)

    script = (
        Path(__file__).resolve().parents[2] / "src/autotrade/webui/static/app.js"
    ).read_text(encoding="utf-8")

    def body(name: str) -> str:
        return script.split(f"function {name}(", 1)[1].split("\nfunction ", 1)[0]

    def reads(name: str, var: str) -> set[str]:
        return set(re.findall(rf"\b{var}\.([a-z_]+)", body(name)))

    replicate = set(forward["seed_replicates"][0])
    assert reads("seedReadingsTable", "replicate") <= replicate
    assert reads("seedReadingsTable", "raw") <= set(forward["seed_mean"]["raw_readings"])
    assert reads("seedReadingsTable", "slice") <= set(forward)
    for name in ("frozenPanel", "freezeGateChecklist"):
        assert reads(name, "seeds") <= set(gate), name
        assert reads(name, "entry") <= set(gate["replicates"][0]), name
    assert "if (!shown.length) return null;" in body("seedReadingsTable")
    for panel in ("forwardStagePanel", "heldoutStagePanel", "verdictStagePanel"):
        assert "seedReadingsTable(" in body(panel), panel
    # Never on the home page.
    for home in ("experimentCard", "heroPanel", "evidenceTiles", "forwardTiles"):
        assert "seed" not in body(home), home


def test_the_listing_rederives_a_row_only_when_its_files_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The home page polls the listing every 5 s over every arm ever created:
    an ended arm with no live worker is re-read only when its status, ledger,
    params or identity map changes, while an arm with a live worker is re-read
    on every poll."""

    root = tmp_path / "experiments"
    ended = build_arm(root, "ended", "no_deliverable")
    build_arm(root, "running", "research", alive=True)
    derived: list[str] = []
    summarize = registry.summarize_experiment

    def counting(directory: Path) -> dict[str, object]:
        derived.append(directory.name)
        return summarize(directory)

    monkeypatch.setattr(registry, "summarize_experiment", counting)
    first = {row["experiment_id"]: row for row in list_experiments(root)}
    assert sorted(derived) == ["ended", "running"]
    derived.clear()
    second = {row["experiment_id"]: row for row in list_experiments(root)}
    assert derived == ["running"]
    assert second["ended"] == first["ended"]
    # The status changing under the arm reaches the very next poll.
    write_json_atomic(
        ended / "hitl/status.json",
        {"schema_version": 1, "pid": 999_999_999, "state": "failed", "error": "RuntimeError: boom"},
    )
    derived.clear()
    third = {row["experiment_id"]: row for row in list_experiments(root)}
    assert sorted(derived) == ["ended", "running"]
    assert third["ended"]["ending"] == {"state": "broken", "reason": "RuntimeError: boom"}


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
    # This arm's params.json was written before the five limits reached the
    # create form: it names none of them, and the console attributes none to it.
    assert preview["max_drawdown"] is None
    assert not {"active_max_drawdown", "tracking_error_cap"} & set(preview)
    assert preview["cost_stress_multiplier"] == pytest.approx(WEB_CREATE_DEFAULTS["cost_stress_multiplier"])
    # One created since states all five, and the console lists the rules that
    # arm's own parameters resolve to -- a mandate only where it named a cap.
    path = tmp_path / "researching/hitl/params.json"
    params = json.loads(path.read_text(encoding="utf-8"))
    stated_keys = ("max_drawdown", "active_max_drawdown", "tracking_error_cap", "beta_min", "beta_max")
    derived = dict.fromkeys(stated_keys)
    path.write_text(
        json.dumps(
            {**params, **derived, "initial_cash": 1_000_000,
             "tracking_error_cap": 0.08, "max_drawdown": 0.3}
        ),
        encoding="utf-8",
    )
    stated = experiment_detail(tmp_path, "researching")["sessions"][1]["thresholds"]
    assert {key: stated[key] for key in derived} == {
        "max_drawdown": 0.3,
        "active_max_drawdown": 0.30,
        "tracking_error_cap": 0.08,
        "beta_min": 0.85,
        "beta_max": 1.15,
    }
    # The same arm without a cap: no mandate, on the very same capital.
    path.write_text(
        json.dumps({**params, **derived, "initial_cash": 1_000_000}), encoding="utf-8"
    )
    untracked = experiment_detail(tmp_path, "researching")["sessions"][1]["thresholds"]
    assert (untracked["max_drawdown"], untracked["active_max_drawdown"]) == (0.45, 0.30)
    assert untracked["tracking_error_cap"] is None
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


def test_a_running_session_serves_its_best_candidate_and_the_record_keeps_it(tmp_path: Path) -> None:
    """While the one research session runs, the listing reads the Validations
    it has recorded so far from the host sidecars and names the best
    full-span candidate, its curve and its result by the rule the recorded
    session is read with, with the freeze gate read now. Once the record
    lands the same candidate is read off it alone: its figures stay, and the
    gate the session never recorded over it is shown as none."""

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
    assert live["research_best"]["deflated_sharpe_probability"] is not None
    assert live["research_best"]["panel_return"] is None  # these sidecars carry no panel
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
    gated = ("deflated_sharpe_probability", "trials", "panel_return")
    assert {key: value for key, value in recorded["research_best"].items() if key not in gated} == {
        key: value for key, value in live["research_best"].items() if key not in gated
    }
    assert (recorded["research_best"]["deflated_sharpe_probability"], recorded["research_best"]["trials"]) == (None, None)
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


def test_an_ended_arm_is_read_off_its_ledger_and_never_judged_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A recorded research session and a forward record are what an ended arm
    shows: a cold console lists every such arm and opens its page without
    calling the freeze gate or reading any replay's series, and the numbers it
    serves are the ledger's own -- the nominee's deflated Sharpe as its gate
    recorded it, and no gate at all over a best candidate the session never
    nominated. Only a running session's live Validations are gated, now."""

    root = tmp_path / "experiments"
    for experiment_id, stage in (
        ("frozen", "sealed"),
        ("graduate", "graduated"),
        ("refused", "discarded"),
        ("no_edge", "no_deliverable"),
        # Stopped mid-session: Validations on disk, no session record.
        ("stopped", "research"),
    ):
        build_arm(root, experiment_id, stage)
    _sidecar(root / "stopped", _step(root / "stopped", "research", 0, edge=0.002, seed=3))
    no_edge = root / "no_edge"
    steps = [
        _step(no_edge, "research", 0, edge=0.0005, seed=1),
        _step(no_edge, "research", 1, edge=0.002, seed=3),
    ]
    ledger = ExperimentLedger(no_edge / "ledgers/experiment_ledger.jsonl")
    records = ledger.read()
    records[0]["steps"] = steps
    ledger.rewrite(records)
    for directory in root.iterdir():
        if directory.name == "stopped":
            continue  # its Validations are read off their own files
        for result in (directory / "artifacts/results").glob("*/style_analysis.json"):
            result.unlink()

    calls: list[str] = []
    gate, panel = registry.freeze_gate_for, registry.panel_return

    def gated(*args: object, **kwargs: object) -> object:
        calls.append("freeze_gate_for")
        return gate(*args, **kwargs)

    def paneled(*args: object, **kwargs: object) -> object:
        calls.append("panel_return")
        return panel(*args, **kwargs)

    monkeypatch.setattr(registry, "freeze_gate_for", gated)
    monkeypatch.setattr(registry, "panel_return", paneled)
    monkeypatch.setattr(registry, "_SUMMARY_CACHE", {})
    client = TestClient(create_app(tmp_path, root))
    rows = {row["experiment_id"]: row for row in client.get("/api/experiments").json()["experiments"]}
    details = {name: client.get(f"/api/experiments/{name}").json() for name in rows}
    assert calls == []
    assert all(row.get("state") != "unreadable" for row in rows.values())

    # The nominee is the best candidate: its deflated Sharpe is the gate's.
    stored = _records(root / "frozen")[0]
    best = rows["frozen"]["research_best"]
    assert best["step_id"] == stored["nominated_step_id"]
    assert best["deflated_sharpe_probability"] == stored["freeze_gate"]["deflated_sharpe"]["deflated_sharpe_probability"]
    assert best["trials"] == stored["freeze_gate"]["deflated_sharpe"]["trials"]
    assert best["information_ratio"] == stored["steps"][1]["neutralized"]["information_ratio"]
    assert {"session_key": "research", **details["frozen"]["sessions"][0]["record"]["best"]} == best
    # A session that nominated nothing recorded no gate over its best, and
    # one stopped before its record has none either.
    unnominated = rows["no_edge"]["research_best"]
    assert unnominated["step_id"] == steps[1]["step_id"]
    assert (unnominated["deflated_sharpe_probability"], unnominated["trials"]) == (None, None)
    stopped = rows["stopped"]["research_best"]
    assert stopped["information_ratio"] is not None
    assert (stopped["deflated_sharpe_probability"], stopped["trials"]) == (None, None)
    # The forward slices and verdict are the record's, key for key.
    for name in ("graduate", "refused"):
        forward = _records(root / name)[-1]
        assert details[name]["forward"]["slices"] == forward["slices"]
        assert details[name]["forward"]["verdict"] == forward["verdict"]

    # A running session is gated now, over the Validations it has so far.
    running = build_arm(root, "running", "research", alive=True)
    _sidecar(running, _step(running, "research", 0, edge=0.002, seed=3))
    live = client.get("/api/experiments").json()
    assert calls[:1] == ["freeze_gate_for"]
    assert next(row for row in live["experiments"] if row["experiment_id"] == "running")[
        "research_best"
    ]["deflated_sharpe_probability"] is not None


def test_a_page_poll_keeps_one_request_in_flight() -> None:
    """Every page poll keeps one request in flight, so a slow listing cannot
    stack requests on the server."""

    script = (
        Path(__file__).resolve().parents[2] / "src/autotrade/webui/static/app.js"
    ).read_text(encoding="utf-8")
    helper = script.split("function setPollInterval(", 1)[1].split("\n}\n", 1)[0]
    assert (
        helper.index("if (busy) return;")
        < helper.index("busy = true;")
        < helper.index("await tick();")
        < helper.index("finally {")
        < helper.rindex("busy = false;")
    )
    # Every other timer is a synchronous clock: a callback that awaits the
    # server (or is a named poll function) goes through the guard.
    rest = script.replace(helper, "")
    assert re.search(r"setInterval\((?!\(\) =>)", rest) is None
    # The definition, then the home, status, trace, sub-agent and Paper polls.
    assert rest.count("setPollInterval(") == 6


def test_a_poll_holding_the_kept_digest_receives_only_the_live_rows(tmp_path: Path) -> None:
    """The home page polls the listing every few seconds, and nearly all of it
    is ended arms that change only when their files do. A poll that hands back
    the digest of those rows gets the live rows and the order of every id; any
    change to a kept row sends every row again."""

    root = tmp_path / "experiments"
    ended = build_arm(root, "ended", "no_deliverable")
    build_arm(root, "running", "research", alive=True)
    client = TestClient(create_app(tmp_path, root))
    full = client.get("/api/experiments").json()
    order = [row["experiment_id"] for row in full["experiments"]]
    assert sorted(order) == ["ended", "running"] and "order" not in full
    poll = client.get("/api/experiments", params={"kept": full["kept"]}).json()
    assert [row["experiment_id"] for row in poll["experiments"]] == ["running"]
    assert poll["order"] == order
    assert (poll["kept"], poll["best"], poll["running"]) == (
        full["kept"],
        full["best"],
        full["running"],
    )
    stale = client.get("/api/experiments", params={"kept": "not-the-digest"}).json()
    assert "order" not in stale and len(stale["experiments"]) == 2
    write_json_atomic(
        ended / "hitl/status.json",
        {"schema_version": 1, "pid": 999_999_999, "state": "failed", "error": "RuntimeError: boom"},
    )
    changed = client.get("/api/experiments", params={"kept": full["kept"]}).json()
    assert "order" not in changed and changed["kept"] != full["kept"]
    rows = {row["experiment_id"]: row for row in changed["experiments"]}
    assert rows["ended"]["ending"] == {"state": "broken", "reason": "RuntimeError: boom"}


def _shelve(directory: Path) -> Path:
    marker = directory / "hitl" / registry.ARCHIVED_NAME
    write_json_atomic(marker, {"archived_at": "2026-10-06T00:00:00+00:00", "reason": "historical"})
    return marker


def test_an_archived_arm_is_listed_only_on_request_and_its_page_still_opens(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cold console lists the arms not archived without summarizing an
    archived one at all, and counts the archived ones; asked for them, it
    lists those alone. The archived arm's page opens as before and says it is
    archived."""

    root = tmp_path / "experiments"
    build_arm(root, "ended", "no_deliverable")
    _shelve(build_arm(root, "shelved", "sealed"))
    summarized: list[str] = []
    summarize = registry.summarize_experiment

    def counting(directory: Path) -> dict[str, object]:
        summarized.append(Path(directory).name)
        return summarize(directory)

    monkeypatch.setattr(registry, "summarize_experiment", counting)
    monkeypatch.setattr(registry, "_SUMMARY_CACHE", {})
    client = TestClient(create_app(tmp_path, root))
    listing = client.get("/api/experiments").json()
    assert [row["experiment_id"] for row in listing["experiments"]] == ["ended"]
    assert listing["archived"] == 1 and summarized == ["ended"]
    shelf = client.get("/api/experiments", params={"archived": 1}).json()
    assert [row["experiment_id"] for row in shelf["experiments"]] == ["shelved"]
    assert shelf["archived"] == 1
    assert shelf["experiments"][0]["research_best"]["deflated_sharpe_probability"] is not None
    detail = client.get("/api/experiments/shelved").json()
    assert detail["archived"]["reason"] == "historical" and detail["stage"] == shelf["experiments"][0]["stage"]
    assert client.get("/api/experiments/ended").json()["archived"] is None


def test_archiving_or_restoring_any_arm_changes_the_kept_digest(tmp_path: Path) -> None:
    """A poll holding the digest is told of an archive or a restore by a new
    digest and every row, even when the arm's own row is derived afresh on
    every poll and so is in no kept set."""

    root = tmp_path / "experiments"
    build_arm(root, "ended", "no_deliverable")
    running = build_arm(root, "running", "research", alive=True)
    client = TestClient(create_app(tmp_path, root))
    full = client.get("/api/experiments").json()
    marker = _shelve(running)
    archived = client.get("/api/experiments", params={"kept": full["kept"]}).json()
    assert "order" not in archived and archived["kept"] != full["kept"]
    assert [row["experiment_id"] for row in archived["experiments"]] == ["ended"]
    marker.unlink()
    restored = client.get("/api/experiments", params={"kept": archived["kept"]}).json()
    assert "order" not in restored and restored["kept"] == full["kept"] != archived["kept"]
    assert sorted(row["experiment_id"] for row in restored["experiments"]) == ["ended", "running"]
    assert restored["archived"] == 0


def test_a_card_curve_is_served_on_a_sample_of_its_days(tmp_path: Path) -> None:
    root = tmp_path / "experiments"
    build_arm(root, "arm", "graduated")
    client = TestClient(create_app(tmp_path, root))
    name = client.get("/api/experiments").json()["experiments"][0]["research_result"]
    url = f"/api/experiments/arm/results/{name}/equity"
    full = client.get(url).json()["series"][0]
    thin = client.get(url, params={"points": 20}).json()
    line = thin["series"][0]
    assert len(full["dates"]) > 20
    assert len(line["dates"]) == len(line["cum"]) == len(line["drawdown"]) == 20
    assert (line["dates"][0], line["dates"][-1]) == (full["dates"][0], full["dates"][-1])
    for date, value in zip(line["dates"], line["cum"], strict=True):
        assert full["cum"][full["dates"].index(date)] == value
    assert line["final"] == full["final"]
    assert len(thin["benchmark"]["dates"]) == 20
    assert client.get(url, params={"points": 1}).status_code == 422


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


def test_a_home_card_shows_each_figure_alone_never_as_a_fraction() -> None:
    """A card keeps its figures but shows each as one labelled value: no two
    values joined with a slash, no progress as done over total, no budget
    against its limit (the bar alone carries the share). The figures stay
    full on the experiment page, and the card and the research panel name the
    best candidate's figures alike."""

    script = (
        Path(__file__).resolve().parents[2] / "src/autotrade/webui/static/app.js"
    ).read_text(encoding="utf-8")

    def body(name: str) -> str:
        return script.split(f"function {name}(", 1)[1].split("\nfunction ", 1)[0]

    joined = re.compile(r"\$\{[^}]*\} ?/ ?\$\{")
    for name in ("forwardTiles", "accountTiles", "evidenceTiles"):
        assert not joined.search(body(name)), name
    card = body("experimentCard")
    assert "fraction: false" in card and "{ mini: true }" in card
    assert "evidenceTiles(item)" in card and "forwardTiles(item)" in body("heroPanel")
    # The fraction and the limit are left out of the card's lines only.
    assert "fraction && Number.isFinite(done)" in body("activityNode")
    budget = body("budgetBars")
    assert "mini ? row.spent" in budget and "mini ? `${row.label}已用 ${row.spent}`" in budget
    assert "activityNode(fresh.status)" in body("controlPanel")
    # One vocabulary for the best candidate: the card's four tiles are the
    # experiment page's first four, recorded or live.
    labels = ("最佳候选中性化超额", "IR", "DSR", "累计试验")
    for name in ("evidenceTiles", "bestCandidateTiles"):
        found = [label for label in labels if f'label: "{label}"' in body(name)]
        assert found == list(labels), name
        assert "DSR_TITLE" in body(name) and "TRIALS_TITLE" in body(name), name
    assert "bestCandidateTiles(best, record.trials_to_date)" in body("researchSessionPanel")
    assert "bestCandidateTiles(detail.research_best)" in body("sessionDetailPanel")
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
        ("function forwardTiles(", ("PLAIN_SELECTION_TITLE", "NEUTRAL_IR_TITLE")),
        ("function bestCandidateTiles(", ("IR_TITLE", "DSR_TITLE", "TRIALS_TITLE")),
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


def test_each_validation_shows_its_raw_excess_beside_its_active_figures(tmp_path: Path) -> None:
    build_arm(tmp_path, "arm", "graduated")
    detail = experiment_detail(tmp_path, "arm")
    record = next(entry for entry in detail["sessions"] if entry["kind"] == "research")["record"]
    assert {"excess_return", "raw_excess_at_cost_stress"} <= set(record["validations"][0])
    # The ledger holds no reading of the best candidate's panel.
    assert "panel_return" not in record["best"]


def test_a_forward_record_without_the_holders_readings_shows_none(tmp_path: Path) -> None:
    """A forward record written before slices carried the holder's readings
    is shown as written: no reading is derived again from its replay, and
    the graduate's ending line names the figures it does hold. Its curve
    still draws the replay's panel beside the book, on the book's own days."""

    from autotrade.environment.replay.style import STYLE_ARTIFACT_NAME

    directory = build_arm(tmp_path, "arm", "graduated")
    ledger_path = directory / "ledgers/experiment_ledger.jsonl"
    records = [json.loads(line) for line in ledger_path.read_text(encoding="utf-8").splitlines()]
    record = records[-1]
    for block in record["slices"].values():
        del block["raw_readings"], block["plain_excess"]
    del record["verdict"]["holder_line"]
    ledger_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records), encoding="utf-8"
    )
    detail = experiment_detail(tmp_path, "arm")
    assert detail["forward"]["slices"] == record["slices"]
    ending = summarize_experiment(directory)["ending"]["reason"]
    information_ratio = record["slices"]["forward"]["information_ratio"]
    assert ending == f"前推 账户 — · 基准 — · 对面板（未回归）— · 中性化主动 IR {information_ratio:.2f}"
    sidecar = json.loads(
        (Path(str(record["result_ref"])).parent / STYLE_ARTIFACT_NAME).read_text(encoding="utf-8")
    )
    client = TestClient(create_app(tmp_path, tmp_path))
    equity = client.get(f"/api/experiments/arm/results/{detail['forward']['result']}/equity").json()
    assert equity["panel"]["dates"] == equity["series"][0]["dates"]
    assert equity["panel"]["final"] == pytest.approx(
        float(np.prod([1.0 + value for _day, value in sidecar["panel_daily"]]) - 1.0), abs=1e-6
    )


def test_an_arm_carrying_the_retired_switches_reads_as_one_without_them(tmp_path: Path) -> None:
    """The console accepts the switches every older params.json carries and
    reads nothing from them: the replay plan, the listing row and the page
    are those of the same arm without them. A key no reader knows still makes
    the arm unreadable."""

    def plan(experiment_id: str) -> dict[str, object]:
        return experiment_detail(tmp_path, experiment_id)["sessions"][1]["thresholds"]

    def written(directory: Path, extra: dict[str, object]) -> None:
        path = directory / "hitl/params.json"
        params = json.loads(path.read_text(encoding="utf-8"))
        path.write_text(json.dumps({**params, **extra}), encoding="utf-8")

    build_arm(tmp_path, "current", "research")
    written(build_arm(tmp_path, "legacy", "research"), dict.fromkeys(RETIRED_SWITCHES, False))
    rows = {row["experiment_id"]: row for row in list_experiments(tmp_path)}
    assert rows["legacy"].get("state") != "unreadable"
    assert plan("legacy") == plan("current")
    assert not RETIRED_SWITCHES & set(plan("legacy"))

    written(build_arm(tmp_path, "unknown", "research"), {"require_heldout_magic": True})
    unknown = {row["experiment_id"]: row for row in list_experiments(tmp_path)}["unknown"]
    assert unknown["state"] == "unreadable"
