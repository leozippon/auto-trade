"""The research console over a research experiment's ledger.

An experiment runs one research session, registers at most once and is then replayed once
over forward and Held-out. The console must say where the experiment is from the
ledger alone, keep every number of the replay sealed until the forward record
(and with it the verdict) exists, rank the best experiment on that verdict
only, and refuse a create whose geometry the worker would refuse.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from autotrade.common.records import write_json_atomic
from autotrade.environment.executor import StrategyRaised
from autotrade.environment.identity import AgentRefStore
from autotrade.environment.step_tree import StepTree
from autotrade.environment.tools.report_issue import (
    append_issue_report,
    append_issue_resolution,
    issue_reports_path,
)
from autotrade.pipelines import validation_record
from autotrade.pipelines.ledger import ExperimentLedger, recorded_path
from autotrade.pipelines.research_session import session_role_quotas
from autotrade.pipelines.session_tools import record_validation_period_failure
from autotrade.webui import registry
from autotrade.webui.registry import (
    ENDING_STATES,
    best_experiment,
    experiment_detail,
    experiment_listing,
    summarize_experiment,
)
from autotrade.webui.server import create_app
from autotrade.webui.steps import step_tree_view
from tests.unit.acceptance_fixture import (
    CREATE_RULES,
    NO_SEED,
    REQUIRED_RULES,
    TEST_GEOMETRY,
)
from tests.unit.test_webui_inject_message import _const_block, _fn
from tests.unit.webui_research_experiment import (
    ENVIRONMENT_BLOCKED_REASON,
    REPLAY,
    _step,
    build_experiment,
)


def _records(directory: Path) -> list[dict[str, object]]:
    return ExperimentLedger(directory / "ledgers/experiment_ledger.jsonl").read()


def _result_names(root: Path, experiment_id: str, mode: str) -> list[str]:
    return sorted(
        path.name
        for path in (root / experiment_id / "artifacts/results").glob(f"{mode}_*")
    )


def test_a_researching_experiment_shows_its_plan_and_nothing_registered(tmp_path: Path) -> None:
    build_experiment(tmp_path, "experiment", "research")
    detail = experiment_detail(tmp_path, "experiment")
    assert detail["stage"] == "research"
    assert detail["research_outcome"] is None
    assert detail["verdict"] is None
    assert detail["forward"] is None
    assert detail["registered"] is None
    assert detail["paper_openable"] is False
    assert [session["key"] for session in detail["sessions"]] == ["research", "forward"]
    assert [session["kind"] for session in detail["sessions"]] == ["research", "forward"]
    # The session in flight has no ledger record yet; its plan entry stands.
    assert "record" not in detail["sessions"][0]
    assert detail["sessions"][1]["replay"] == REPLAY


def test_a_registered_experiment_lists_its_validations_and_the_registration_it_registered_under(
    tmp_path: Path,
) -> None:
    """A recorded session serves its outcome, attempts, budget, the registration
    the registration passed (which carries no score) and every Validation it ran;
    research results are development evidence, readable as they land."""

    build_experiment(tmp_path, "experiment", "forward")
    record = experiment_detail(tmp_path, "experiment")["sessions"][0]["record"]
    assert record["outcome"] == "completed"
    assert record["registered"] is True
    assert record["attempts"] == 1
    assert record["budget_used"]["test_replays"] == 8
    assert "best" not in record
    client = TestClient(create_app(tmp_path, tmp_path))
    name = Path(str(_records(tmp_path / "experiment")[0]["steps"][0]["validation_result_ref"])).parent.name
    assert client.get(f"/api/experiments/experiment/results/{name}/equity").status_code == 200
    # The record says whether its trace is on disk; this synthetic experiment has none.
    assert record["trace"] is False
    assert record["registration"]["passed"] is True
    assert {row["span"] for row in record["validations"]} == {"full", "Y3"}


def test_a_registered_experiment_serves_no_replay_until_its_readout_exists(tmp_path: Path) -> None:
    directory = build_experiment(tmp_path, "experiment", "forward", alive=True)
    detail = experiment_detail(tmp_path, "experiment")
    assert detail["stage"] == "forward"
    assert detail["status"]["session_key"] == "forward"
    assert detail["verdict"] is None
    assert detail["forward"] is None
    assert detail["paper_openable"] is False
    frozen = detail["registered"]
    assert frozen["session_key"] == "research"
    assert frozen["strategy_ref"].startswith("strategy_ref_")
    assert "strategy_research_abc" not in json.dumps(detail)
    assert frozen["null_percentile"] == 0.81
    assert frozen["forward_mde"] > 0
    assert frozen["blocks"][0]["label"] == "202407-202506"
    # The running replay's result directory is on disk, and nothing serves it:
    # it answers exactly like a result that does not exist.
    [running] = _result_names(tmp_path, "experiment", "heldout")
    client = TestClient(create_app(tmp_path, tmp_path))
    for route in ("equity", "style", "orders", "orders.csv"):
        unnamed = client.get(f"/api/experiments/experiment/results/{running}/{route}")
        absent = client.get(f"/api/experiments/experiment/results/heldout_absent/{route}")
        assert unnamed.status_code == absent.status_code == 404, route
        assert unnamed.json()["detail"].replace(running, "") == absent.json()[
            "detail"
        ].replace("heldout_absent", "")
    # The listing and status carry the stage only.
    for payload in (
        client.get("/api/experiments").text,
        client.get("/api/experiments/experiment").text,
        client.get("/api/experiments/experiment/status").text,
    ):
        assert running not in payload
        assert "lower_bound" not in payload
    # The frozen research result stays readable.
    assert client.get(f"/api/experiments/experiment/results/{frozen['result']}/style").status_code == 200
    assert (directory / "artifacts/results" / running).is_dir()


def test_a_result_without_a_style_sidecar_is_not_an_unknown_result(tmp_path: Path) -> None:
    """"This result has no style artifact" is an expected state of a real
    result, and the console has to tell it from "no such result". The two must
    differ in the payload, never only in the wording of an error."""

    directory = build_experiment(tmp_path, "experiment", "registered_with_record")
    client = TestClient(create_app(tmp_path, tmp_path))
    name = experiment_detail(tmp_path, "experiment")["registered"]["result"]
    present = client.get(f"/api/experiments/experiment/results/{name}/style")
    assert present.status_code == 200
    assert present.json()["available"] is True

    # A sidecar of another schema is no sidecar of this code's.
    sidecar = directory / "artifacts/results" / name / "style_analysis.json"
    sidecar.write_text(json.dumps({**json.loads(sidecar.read_text(encoding="utf-8")), "schema_version": 2}), encoding="utf-8")
    assert client.get(f"/api/experiments/experiment/results/{name}/style").json() == {
        "available": False,
        "reason": "no_style_artifact",
    }
    sidecar.unlink()
    absent = client.get(f"/api/experiments/experiment/results/{name}/style")
    assert absent.status_code == 200
    assert absent.json() == {"available": False, "reason": "no_style_artifact"}

    sidecar.write_text(json.dumps({"schema_version": 0, "mode": "valid"}), encoding="utf-8")
    stale = client.get(f"/api/experiments/experiment/results/{name}/style")
    assert stale.status_code == 200
    assert stale.json() == {"available": False, "reason": "no_style_artifact"}

    # A sidecar that is there but cannot be parsed is a load failure, and says
    # so rather than reading as a result that never had one.
    sidecar.write_text('{"schema_version": ', encoding="utf-8")
    corrupt = client.get(f"/api/experiments/experiment/results/{name}/style")
    assert corrupt.status_code == 200
    assert corrupt.json() == {"available": False, "reason": "unreadable"}

    unknown = client.get("/api/experiments/experiment/results/valid_absent/style")
    assert unknown.status_code == 404
    assert unknown.json()["detail"] == "unknown result: valid_absent"


def test_a_moved_experiment_serves_its_recorded_paths_from_where_it_now_lives(
    tmp_path: Path,
) -> None:
    """Records name files by the host path they were written at. An experiment moved
    to another checkout or host serves them from its new directory, and a
    recorded file that is not there fails by name, not as an unknown result."""

    old_root = tmp_path / "old/experiments"
    build_experiment(old_root, "experiment", "registered_with_record")
    new_root = tmp_path / "new/experiments"
    new_root.parent.mkdir()
    old_root.rename(new_root)
    assert str(old_root / "experiment") in json.dumps(_records(new_root / "experiment"))
    client = TestClient(create_app(tmp_path / "new", new_root))
    name = experiment_detail(new_root, "experiment")["forward"]["result"]
    for route in ("equity", "style", "orders", "orders.csv"):
        response = client.get(f"/api/experiments/experiment/results/{name}/{route}")
        assert response.status_code == 200, route
    assert client.get("/api/experiments/experiment/registered/strategy.zip").status_code == 200

    (new_root / "experiment/artifacts/results" / name / "result.json").unlink()
    missing = client.get(f"/api/experiments/experiment/results/{name}/equity")
    assert missing.status_code == 404
    assert missing.json()["detail"] == (
        f"experiment: artifacts/results/{name}/result.json is recorded but not on disk"
    )
    with pytest.raises(FileNotFoundError, match="outside the experiment"):
        recorded_path(new_root / "experiment", tmp_path / "elsewhere/experiment/result.json")


def test_a_research_that_registered_nothing_has_a_verdict_and_no_replay(tmp_path: Path) -> None:
    build_experiment(tmp_path, "experiment", "deadline")
    detail = experiment_detail(tmp_path, "experiment")
    assert detail["stage"] == "readout"
    assert detail["verdict"] == {
        "status": "no_deliverable",
        "reasons": [
            (
                "research budget exhausted before any test, with no registrable "
                "full-span candidate (llm_call_budget_exhausted)"
            )
        ],
    }
    assert detail["registered"] is None
    assert detail["forward"] is None
    assert detail["sessions"][0]["record"]["experiment_end"]["status"] == "no_deliverable"
    assert detail["record"] is None


def test_every_way_an_experiment_can_end_reads_as_one_ending_with_a_reason(tmp_path: Path) -> None:
    """The console shows an experiment's ending in one vocabulary, so the projection
    must classify every ending the pipeline can produce and give each its own
    reason; an experiment that can still run has none at all."""

    for experiment_id, stage in (
        ("registered_with_record", "registered_with_record"),
        ("budget_exhausted", "deadline"),
        ("environment_blocked", "environment_blocked"),
        ("broken", "broken"),
        ("running", "research"),
    ):
        build_experiment(tmp_path, experiment_id, stage)
    rows = {row["experiment_id"]: row for row in experiment_listing(tmp_path)["experiments"]}
    assert rows["running"]["ending"] is None
    endings = {name: rows[name]["ending"] for name in ENDING_STATES}
    assert {name: ending["state"] for name, ending in endings.items()} == {
        name: name for name in ENDING_STATES
    }
    # The endings no replay decided are named by what the record says: the
    # budget, the error's first line.
    # A registered strategy is named by its readout: the record's significance test and
    # the forward lower bound.
    readout = rows["registered_with_record"]["forward"]["readout"]
    assert endings["registered_with_record"]["reason"] == (
        f"验证记录 残差 IR {readout['record']['residual_information_ratio']:.2f}"
        f" · p {readout['record']['p_value']:.2f} · 换候选 0 次"
        f" · 前推下界 {readout['forward']['lower_bound'] * 100:+.2f}%"
    )
    assert rows["registered_with_record"]["paper_openable"] is True
    assert endings["budget_exhausted"]["reason"] == "模型调用次数用尽"
    # The host's reason in full, its error included: that is what to act on.
    assert endings["environment_blocked"]["reason"] == ENVIRONMENT_BLOCKED_REASON
    assert endings["broken"]["reason"] == "RuntimeError: sandbox image is gone"
    # The homepage lists what can still run first, then the endings in order.
    listed = [row["experiment_id"] for row in experiment_listing(tmp_path)["experiments"]]
    assert listed[-len(ENDING_STATES) :] == list(ENDING_STATES)
    assert "running" in listed[: -len(ENDING_STATES)]
    # The detail page reads the same projection as the card.
    assert experiment_detail(tmp_path, "budget_exhausted")["ending"] == endings["budget_exhausted"]


@pytest.mark.parametrize(
    ("record", "passed"),
    [
        ({"residual_information_ratio": 1.2, "threshold": 0.9}, True),
        ({"residual_information_ratio": 0.5, "threshold": 0.9}, False),
        ({"residual_information_ratio": 0.9, "threshold": 0.9}, True),
        ({"residual_information_ratio": float("nan"), "threshold": 0.9}, False),
        ({"residual_information_ratio": float("inf"), "threshold": 0.9}, False),
        ({"residual_information_ratio": 1.2, "threshold": float("nan")}, False),
        ({"residual_information_ratio": None, "threshold": 0.9, "unmeasured": "too few days"}, False),
        ({"threshold": 0.9}, False),
        ({"residual_information_ratio": 1.2}, False),
        (None, False),
    ],
)
def test_a_record_passes_when_its_residual_ir_reaches_its_stored_threshold(
    record: dict[str, object] | None, passed: bool
) -> None:
    """通过 / 未通过 is read off the stored readout alone: the record's residual IR
    at or above the threshold stored beside it. A reading that is missing or
    not finite fails rather than passing by default."""

    readout = {} if record is None else {"record": record}
    assert registry.record_passed(readout) is passed


def test_a_registered_experiment_serves_whether_its_record_passed_and_keeps_its_stored_state(
    tmp_path: Path,
) -> None:
    """The listing row and the experiment page carry ``record.passed`` from the
    stored readout, so a record below its threshold reads as failed while its
    stored verdict and ending state stay ``registered_with_record``; before the
    readout exists there is nothing to pass."""

    root = tmp_path / "experiments"
    build_experiment(root, "passed", "registered_with_record")
    build_experiment(root, "failed", "registered_with_record")
    build_experiment(root, "replaying", "forward")
    ledger = root / "failed/ledgers/experiment_ledger.jsonl"
    rows = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines()]
    stored = rows[-1]["readout"]["record"]
    stored["residual_information_ratio"] = stored["threshold"] - 0.01
    ledger.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    listed = {row["experiment_id"]: row for row in experiment_listing(root)["experiments"]}
    assert (listed["passed"]["record"]["passed"], listed["failed"]["record"]["passed"]) == (True, False)
    assert listed["replaying"]["record"]["passed"] is None
    for name in ("passed", "failed"):
        detail = experiment_detail(root, name)
        assert detail["record"]["passed"] is listed[name]["record"]["passed"]
        assert detail["ending"]["state"] == listed[name]["ending"]["state"] == "registered_with_record"
        assert detail["verdict"]["status"] == "registered_with_record"
    assert _records(root / "failed")[-1]["readout"]["record"] == stored


def test_a_registered_experiment_is_labelled_passed_or_failed_wherever_its_ending_shows(
    tmp_path: Path,
) -> None:
    """A registered strategy with its record wears 通过 or 未通过 by its record's
    test, never its stored state's name; a record with no ``passed`` at all is
    未通过; every other ending keeps its own label. The card, the experiment
    page header and the best-experiment panel wear experimentBadge, the 读出
    view endingBadge and the process list endingLabel, all of the whole item."""

    for experiment_id, stage in (
        ("passed", "registered_with_record"),
        ("deadline", "deadline"),
        ("environment_blocked", "environment_blocked"),
        ("broken", "broken"),
    ):
        build_experiment(tmp_path / "experiments", experiment_id, stage)
    rows = {row["experiment_id"]: row for row in experiment_listing(tmp_path / "experiments")["experiments"]}
    failed = json.loads(json.dumps(rows["passed"]))
    failed["record"]["passed"] = False
    rows["failed"] = failed
    rows["unread"] = {**rows["passed"], "record": None}
    script = (
        Path(__file__).resolve().parents[2] / "src/autotrade/webui/static/app.js"
    ).read_text(encoding="utf-8")
    harness = "\n".join(
        [
            "const el = (tag, attrs, ...children) => ({ tag, attrs, children });",
            "const stateBadge = () => null;",
            _const_block(script, "ENDING_LABELS"),
            *(_fn(script, name) for name in ("endingKey", "endingLabel", "endingBadge", "experimentBadge")),
            f"const rows = {json.dumps(rows)};",
            "console.log(JSON.stringify(Object.fromEntries(Object.entries(rows).map(([id, row]) => {",
            "  const badge = experimentBadge(row);",
            "  return [id, [endingLabel(row), badge.attrs.class, badge.children[0]]];",
            "}))));",
        ]
    )
    result = subprocess.run(["node", "-e", harness], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {
        "passed": ["通过", "badge ending-passed", "通过"],
        "failed": ["未通过", "badge ending-failed", "未通过"],
        "unread": ["未通过", "badge ending-failed", "未通过"],
        "deadline": ["预算耗尽", "badge ending-budget_exhausted", "预算耗尽"],
        "environment_blocked": ["环境阻断", "badge ending-environment_blocked", "环境阻断"],
        "broken": ["失败", "badge ending-broken", "失败"],
    }
    assert "registered_with_record" not in _const_block(script, "ENDING_LABELS")
    assert "endingBadge(detail)" in _fn(script, "readoutStagePanel")
    assert "status: endingLabel(item)" in _fn(script, "pipelineTailSteps")
    for name in ("experimentCard", "renderDetailPage", "heroPanel"):
        assert "experimentBadge(" in _fn(script, name), name


def test_the_home_list_flags_a_blocked_experiment_and_an_unresolved_issue(tmp_path: Path) -> None:
    """Two things put an attention line on an experiment's card and nothing else
    does: the host ending its session on the environment, and an issue report
    its session filed that nobody has resolved while the experiment can still run.
    Resolving the report clears the line even on a row the listing keeps
    without re-reading, and an experiment that ended no longer carries its reports."""

    build_experiment(tmp_path, "blocked", "environment_blocked")
    stopped = build_experiment(tmp_path, "stopped", "research")
    ended = build_experiment(tmp_path, "ended", "deadline")

    def attention() -> dict[str, object]:
        rows = {row["experiment_id"]: row for row in experiment_listing(tmp_path)["experiments"]}
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
    # The card reads the line off the row's own field.
    script = (
        Path(__file__).resolve().parents[2] / "src/autotrade/webui/static/app.js"
    ).read_text(encoding="utf-8")
    assert "item.attention" in _fn(script, "attentionLine")


def test_seed_replicates_are_shown_where_the_record_has_them_and_absent_otherwise(
    tmp_path: Path,
) -> None:
    """An experiment judged on its training seeds together shows each seed and their
    mean on the experiment page -- the registration's seed-mean IR beside the
    nominee's and the threshold, and the forward and Held-out readings per seed --
    and the series F8 judges them on together -- from the fields the console
    serves; an experiment frozen on one seed serves none of them and draws nothing
    in their place."""

    build_experiment(tmp_path, "seeded", "registered_with_record", seeded=True)
    build_experiment(tmp_path, "single", "registered_with_record")
    seeded = experiment_detail(tmp_path, "seeded")
    single = experiment_detail(tmp_path, "single")
    gate = seeded["sessions"][0]["record"]["registration"]["seed_replicates"]
    frozen = seeded["registered"]["seed_replicates"]
    forward = seeded["forward"]["slices"]["forward"]
    # Registration reads the replicates and judges no seed mean against a threshold.
    assert frozen == gate
    assert "mean_information_ratio" not in gate
    assert gate["replicates"][0]["seed_line"] == "main.py: SEED_BASE = 2000"
    for name in ("forward", "heldout"):
        block = seeded["forward"]["slices"][name]
        assert block["seed_mean"]["members"] == 2
        assert block["seed_mean"]["raw_readings"]["plain_selection"] == pytest.approx(
            (block["raw_readings"]["plain_selection"] + block["seed_replicates"][0]["raw_readings"]["plain_selection"]) / 2
        )
    assert forward["seed_mean"]["members"] == 2
    assert isinstance(forward["lower_bound"], float)
    # One seed: no replicate served, so nothing drawn; F2 reads the book alone.
    assert single["registered"]["seed_replicates"] is None
    alone = single["sessions"][0]["record"]["registration"]["seed_replicates"]
    assert alone["replicates"] == []
    assert "seed_mean" not in single["forward"]["slices"]["forward"]
    assert isinstance(single["forward"]["slices"]["forward"]["lower_bound"], float)
    # The record row the session's own test wrote names the replicate it
    # replayed beside the nominee with that replicate's active return; the
    # host's completion replays the nominee alone, and so does every step of
    # an experiment on one seed.
    tested = validation_record.test_step_records(_records(tmp_path / "seeded"))[0]
    rows = seeded["record"]["rows"]
    assert rows[0]["seed_replicates"] == [
        {
            "node": gate["replicates"][0]["step_id"],
            "active_return": tested["replicates"][0]["entry"]["active_return"],
            "strategy_error": None,
        }
    ]
    assert all(row["seed_replicates"] == [] for row in rows[1:])
    assert all(row["seed_replicates"] == [] for row in single["record"]["rows"])

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
    for name in ("registeredPanel", "registrationChecklist"):
        assert reads(name, "seeds") <= set(gate), name
        assert reads(name, "entry") <= set(gate["replicates"][0]), name
    assert "row.seed_replicates" in body("recordPanel")
    assert reads("recordPanel", "item") <= set(rows[0]["seed_replicates"][0])
    assert "if (!shown.length) return null;" in body("seedReadingsTable")
    for panel in ("forwardStagePanel", "heldoutStagePanel", "readoutStagePanel"):
        assert "seedReadingsTable(" in body(panel), panel
    # Never on the home page.
    for home in ("experimentCard", "heroPanel", "forwardTiles", "evidenceTiles"):
        assert "seed" not in body(home), home


def test_the_listing_rederives_a_row_only_when_its_files_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The home page polls the listing every 5 s over every experiment ever created:
    an ended experiment with no live worker is re-read only when its status, ledger,
    params or identity map changes, while an experiment with a live worker is re-read
    on every poll."""

    root = tmp_path / "experiments"
    ended = build_experiment(root, "ended", "deadline")
    build_experiment(root, "running", "research", alive=True)
    derived: list[str] = []
    summarize = registry.summarize_experiment

    def counting(directory: Path) -> dict[str, object]:
        derived.append(directory.name)
        return summarize(directory)

    monkeypatch.setattr(registry, "summarize_experiment", counting)
    first = {row["experiment_id"]: row for row in experiment_listing(root)["experiments"]}
    assert sorted(derived) == ["ended", "running"]
    derived.clear()
    second = {row["experiment_id"]: row for row in experiment_listing(root)["experiments"]}
    assert derived == ["running"]
    assert second["ended"] == first["ended"]
    # The status changing under the experiment reaches the very next poll.
    write_json_atomic(
        ended / "hitl/status.json",
        {"schema_version": 1, "pid": 999_999_999, "state": "failed", "error": "RuntimeError: boom"},
    )
    derived.clear()
    third = {row["experiment_id"]: row for row in experiment_listing(root)["experiments"]}
    assert sorted(derived) == ["ended", "running"]
    assert third["ended"]["ending"] == {"state": "broken", "reason": "RuntimeError: boom"}


def test_the_listing_carries_the_budget_and_the_research_outcome(tmp_path: Path) -> None:
    """The budget bars read the experiment's ceilings (worker defaults under the
    params) and what the session spent: the ledger's block once recorded,
    else the last block of the live session's trace, else nothing."""

    live = build_experiment(tmp_path, "live", "research", alive=True)
    build_experiment(tmp_path, "idle", "research")
    build_experiment(tmp_path, "judged", "registered_with_record")
    params = json.loads((live / "hitl/params.json").read_text(encoding="utf-8"))
    (live / "hitl/params.json").write_text(json.dumps({**params, "max_research_minutes": 60}), encoding="utf-8")
    traces = live / "artifacts/traces"
    traces.mkdir(parents=True)
    (traces / "run_research_live.jsonl").write_text(
        "\n".join(
            json.dumps(event)
            for event in (
                {"event_type": "session_start"},
                {"event_type": "llm_call", "budget_used": {"inference_seconds": 900.0, "llm_calls": 7, "test_replays": 4, "null_controls": 0}},
            )
        )
        + "\n",
        encoding="utf-8",
    )
    judged_traces = tmp_path / "judged/artifacts/traces"
    judged_traces.mkdir(parents=True)
    (judged_traces / "run_research.jsonl").write_text('{"event_type": "session_start"}\n', encoding="utf-8")
    assert experiment_detail(tmp_path, "judged")["sessions"][0]["record"]["trace"] is True
    rows = {row["experiment_id"]: row for row in experiment_listing(tmp_path)["experiments"]}
    assert rows["live"]["budget"] == {
        "inference_seconds": 3600.0,
        "llm_calls": 6400.0,
        "null_controls": 12.0,
        "api_spend_usd": 50.0,
        "subagent_calls": float(session_role_quotas(6400)[0]),
    }
    assert rows["live"]["budget_used"] == {"inference_seconds": 900.0, "llm_calls": 7, "test_replays": 4, "null_controls": 0}
    assert rows["idle"]["budget_used"] is None
    assert rows["judged"]["budget_used"]["test_replays"] == 8
    assert (rows["live"]["research_outcome"], rows["judged"]["research_outcome"]) == (None, "completed")
    client = TestClient(create_app(tmp_path.parent, tmp_path))
    assert client.get("/api/experiments/live/status").json()["budget_used"]["llm_calls"] == 7
    assert client.get("/api/experiments/idle/status").json()["budget_used"] is None


def test_a_terminated_session_shows_what_it_spent_and_no_record(tmp_path: Path) -> None:
    """A session the operator terminated before it recorded anything has no
    ledger record, and the console invents none. Its trace still says what it
    spent, so the budget reads off the trace with the worker gone, and the
    status still names the session and its run for the trace replay."""

    directory = build_experiment(tmp_path, "cut", "research")
    block = {"inference_seconds": 1316.2, "llm_calls": 180, "test_replays": 0, "null_controls": 0, "api_spend_usd": 1.8}
    write_json_atomic(
        directory / "hitl/status.json",
        {
            "schema_version": 1,
            "pid": 999_999_999,
            "state": "terminated",
            "session_key": "research",
            "run_id": "run_research_cut",
        },
    )
    trace_dir = directory / "artifacts/traces"
    trace_dir.mkdir(parents=True)
    (trace_dir / "run_research_cut.jsonl").write_text(
        json.dumps({"event_type": "llm_call", "budget_used": block}) + "\n", encoding="utf-8"
    )
    row = {row["experiment_id"]: row for row in experiment_listing(tmp_path)["experiments"]}["cut"]
    detail = experiment_detail(tmp_path, "cut")
    assert (row["state"], row["worker_alive"], row["budget_used"]) == ("terminated", False, block)
    assert detail["budget_used"] == block
    assert detail["status"]["session_key"] == "research" and detail["status"]["run_ref"]
    assert detail["sessions"][0] == {"key": "research", "kind": "research"}
    client = TestClient(create_app(tmp_path.parent, tmp_path))
    assert client.get("/api/experiments/cut/status").json()["budget_used"] == block
    # The same experiment with no trace on disk has spent nothing the console can read.
    (trace_dir / "run_research_cut.jsonl").unlink()
    assert experiment_detail(tmp_path, "cut")["budget_used"] is None


def test_a_trace_at_its_size_cap_ends_where_the_session_stopped(tmp_path: Path) -> None:
    """A session stops on the event that crosses its trace's size cap, so the
    trace's last block is the spend it reached and the status poll reads it as
    it reads any unrecorded session's; the trace view ends on the writer's
    marker, which says the session stopped there."""

    directory = build_experiment(tmp_path / "experiments", "deep", "research", alive=True)
    block = {"inference_seconds": 90000.0, "llm_calls": 4400, "test_replays": 2, "null_controls": 0}
    trace = directory / "artifacts/traces/run_research_live.jsonl"
    trace.parent.mkdir(parents=True)
    marker = {"event_type": "trace_limit_reached", "ts": "2026-11-01T00:00:00+00:00", "max_bytes": 1073741824}
    trace.write_text(
        "".join(json.dumps(event) + "\n" for event in ({"event_type": "llm_call", "budget_used": block}, marker)),
        encoding="utf-8",
    )
    client = TestClient(create_app(tmp_path, tmp_path / "experiments"))
    assert client.get("/api/experiments/deep/status").json()["budget_used"] == block
    [shown] = client.get("/api/experiments/deep/trace/blocks").json()["blocks"]
    assert (shown["kind"], shown["label"]) == ("marker", "Trace 已达上限")
    assert "1073741824" in shown["text"] and "会话在此停止" in shown["text"]


@pytest.mark.skipif(shutil.which("node") is None, reason="node required for the JS check")
def test_the_attempts_before_a_resume_keep_their_traces_in_reach(tmp_path: Path) -> None:
    """A session that failed and resumed holds most of its research in the
    traces of the attempts before the last, which the session's record does not
    name. The detail lists each failed research attempt with its error's first
    line and whether its trace is on disk, and the trace routes serve it by the
    attempt's own run ref; a failed forward attempt is no research attempt.
    A worker that failed or was lost also names its log under the error."""

    directory = build_experiment(tmp_path / "experiments", "broken", "broken")
    ExperimentLedger(directory / "ledgers/experiment_ledger.jsonl").append(
        {
            "record_type": "attempt_failed",
            "experiment_id": "broken",
            "epoch_id": "forward",
            "fold_id": "forward",
            "run_id": "run_forward_1",
            "session_key": "forward",
            "phase": "forward",
            "error": "OSError: replay host lost",
        }
    )
    trace = directory / "artifacts/traces/run_research_2.jsonl"
    trace.parent.mkdir(parents=True)
    trace.write_text(json.dumps({"event_type": "user_message", "content": "second attempt"}) + "\n", encoding="utf-8")
    client = TestClient(create_app(tmp_path, tmp_path / "experiments"))
    detail = client.get("/api/experiments/broken").json()
    attempts = detail["research_attempts"]
    assert [(row["error"], row["trace"]) for row in attempts] == [
        ("RuntimeError: sandbox image is gone", False),
        ("RuntimeError: sandbox image is gone", True),
        ("RuntimeError: sandbox image is gone", False),
    ]
    assert all(row["run_ref"].startswith("run_ref_") and row["recorded_at"] for row in attempts)
    assert "run_research_2" not in json.dumps(detail)
    blocks = client.get("/api/experiments/broken/trace/blocks", params={"run_id": attempts[1]["run_ref"]}).json()
    assert [block["text"] for block in blocks["blocks"]] == ["second attempt"]

    script = (
        Path(__file__).resolve().parents[2] / "src/autotrade/webui/static/app.js"
    ).read_text(encoding="utf-8")
    harness = "\n".join(
        [
            """
const node = (tag) => ({
  tag, nodeType: 1, style: {}, children: [],
  append(...nodes) { this.children.push(...nodes); },
  replaceChildren(...nodes) { this.children = nodes; },
  addEventListener() {}, setAttribute() {},
});
const document = { createElement: node, createTextNode: (text) => ({ nodeType: 3, text }) };
const controlBar = () => node("div");
const experimentBadge = () => null, activityNode = () => null, experimentEquityHost = () => null;
const budgetBars = () => null;
const fmtTs = () => "then";
const traceReplayNode = (id, run) => document.createTextNode(`[trace ${run}]`);
""",
            *(_fn(script, name) for name in ("el", "panelHead", "controlPanel", "earlierAttemptsPanel")),
            f"const detail = {json.dumps(detail)};",
            """
const text = (item) => item.nodeType === 3 ? item.text : item.children.map(text).join("");
const lost = { ...detail, state: "interrupted", status: {} };
console.log(JSON.stringify({
  attempts: text(earlierAttemptsPanel(detail)),
  none: earlierAttemptsPanel({ ...detail, research_attempts: [] }),
  failed: text(controlPanel(detail)),
  lost: text(controlPanel(lost)),
  stopped: text(controlPanel({ ...lost, state: "stopped" })),
}));
""",
        ]
    )
    result = subprocess.run(["node", "-e", harness], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    drawn = json.loads(result.stdout)
    assert drawn["attempts"] == (
        "此前失败的尝试研究在其后的尝试里原地续跑"
        "第 1 次 · then · RuntimeError: sandbox image is gone这次尝试的 Trace 不在磁盘上"
        f"第 2 次 · then · RuntimeError: sandbox image is gone[trace {attempts[1]['run_ref']}]"
        "第 3 次 · then · RuntimeError: sandbox image is gone这次尝试的 Trace 不在磁盘上"
    )
    assert drawn["none"] is None
    # The synthetic repository has no worker log; one that exists is named.
    assert "Worker 日志" not in drawn["failed"] and drawn["failed"].startswith("错误：RuntimeError")
    log = tmp_path / "logs/workers/broken.log"
    log.parent.mkdir(parents=True)
    log.write_text("Traceback\n", encoding="utf-8")
    named = client.get("/api/experiments/broken/status").json()
    assert named["worker_log"] == "logs/workers/broken.log"
    harness = harness.replace(f"const detail = {json.dumps(detail)};", f"const detail = {json.dumps({**detail, **named})};")
    result = subprocess.run(["node", "-e", harness], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    drawn = json.loads(result.stdout)
    assert drawn["failed"].endswith("Worker 日志：logs/workers/broken.log")
    assert drawn["lost"] == "Worker 日志：logs/workers/broken.log"
    assert drawn["stopped"] == ""


def test_the_listing_names_the_research_curve_and_the_replay_its_thresholds(tmp_path: Path) -> None:
    """The one curve per experiment starts from the research-period validation of the
    candidate the tiles name — the registered artifact's own once the experiment registered —
    and the replay's step lists the thresholds the verdict will hold it to
    before the record exists, from the experiment's parameters; the record's own
    block carries the same keys afterwards."""

    build_experiment(tmp_path, "researching", "research")
    build_experiment(tmp_path, "registered", "forward")
    build_experiment(tmp_path, "judged", "registered_with_record")
    rows = {row["experiment_id"]: row for row in experiment_listing(tmp_path)["experiments"]}
    assert rows["researching"]["research_result"] is None
    registered_ref = _records(tmp_path / "registered")[0]["registered"]["research_result_ref"]
    assert rows["registered"]["research_result"] == Path(str(registered_ref)).parent.name

    # The forward step states how the record will be read: the level, the
    # research and validation years, the step length and the threshold over the
    # whole validation period.
    preview = experiment_detail(tmp_path, "researching")["sessions"][1]["thresholds"]
    assert preview == {
        "alpha": REQUIRED_RULES["alpha"],
        "research_years": REQUIRED_RULES["research_years"],
        "validation_years": 3,
        "test_step_months": 3,
        "threshold": pytest.approx(0.9497, abs=1e-4),
    }
    recorded = experiment_detail(tmp_path, "judged")["forward"]["readout"]["record"]
    assert isinstance(recorded["threshold"], float)
    # The readout view words exactly these before the record exists.
    script = (
        Path(__file__).resolve().parents[2] / "src/autotrade/webui/static/app.js"
    ).read_text(encoding="utf-8")
    harness = "\n".join(
        [
            *(_fn(script, name) for name in ("fmtSharpe", "readoutRule")),
            f"console.log(readoutRule({json.dumps(preview)}));",
        ]
    )
    result = subprocess.run(["node", "-e", harness], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert result.stdout.startswith("1 个研究年之后的验证期 3 年按 3 个月一步走完或预算用尽后，")
    assert "门槛（约 0.95，α = 0.05）写作通过，否则未通过" in result.stdout
    assert "readoutRule(thresholds || {})" in _fn(script, "readoutStagePanel")


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


def test_a_running_sessions_validations_are_served_off_their_sidecars(tmp_path: Path) -> None:
    """While the one research session runs, its Validations are known to the
    console only through the host sidecars: their results are readable, no
    research-period figure is served as evidence, and the record stays empty
    until the ledger holds a step."""

    directory = build_experiment(tmp_path, "experiment", "research", alive=True)
    steps = [
        _step(directory, "research", 0, edge=0.0005, seed=1),
        _step(directory, "research", 1, edge=0.002, seed=3),
    ]
    for row in steps:
        _sidecar(directory, row)
    live = summarize_experiment(directory)
    assert live["record"] is None and live["research_result"] is None
    assert "research_best" not in live
    client = TestClient(create_app(tmp_path, tmp_path))
    for row in steps:
        name = Path(str(row["validation_result_ref"])).parent.name
        assert client.get(f"/api/experiments/experiment/results/{name}/equity").status_code == 200
    # A result no record or sidecar names is absent.
    assert client.get("/api/experiments/experiment/results/heldout_nope/equity").status_code == 404


def test_a_recorded_node_is_read_once_across_listings(tmp_path: Path) -> None:
    """A Validation is immutable once recorded: its sidecar and style file are
    read on the first listing and never again, so a poll re-reads nothing
    until another node is recorded."""

    directory = build_experiment(tmp_path, "experiment", "research", alive=True)
    first = _step(directory, "research", 0, edge=0.002, seed=3)
    _sidecar(directory, first)
    [before] = registry._live_steps(directory)
    assert before["step_id"] == first["step_id"]
    (Path(str(first["validation_result_ref"])).parent / "style_analysis.json").unlink()
    assert registry._live_steps(directory) == [before]
    # A newly recorded node is read beside the kept one.
    second = _step(directory, "research", 1, edge=0.004, seed=5)
    _sidecar(directory, second)
    assert [row["step_id"] for row in registry._live_steps(directory)] == [first["step_id"], second["step_id"]]


def test_an_ended_experiment_is_read_off_its_ledger_and_never_judged_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A recorded research session and a forward record are what an ended experiment
    shows: a cold console lists every such experiment and opens its page without
    reading any replay's series again, and the numbers it serves are the
    ledger's own -- the slices key for key, and a readout as the forward record
    states it."""

    root = tmp_path / "experiments"
    for experiment_id, stage in (
        ("registered", "forward"),
        ("recorded", "registered_with_record"),
        ("exhausted", "deadline"),
    ):
        build_experiment(root, experiment_id, stage)
    for directory in root.iterdir():
        for result in (directory / "artifacts/results").glob("*/style_analysis.json"):
            result.unlink()

    calls: list[str] = []
    statistics = registry.validation_statistics

    def counted(*args: object, **kwargs: object) -> object:
        calls.append("validation_statistics")
        return statistics(*args, **kwargs)

    monkeypatch.setattr(registry, "validation_statistics", counted)
    monkeypatch.setattr(registry, "_SUMMARY_CACHE", {})
    client = TestClient(create_app(tmp_path, root))
    rows = {row["experiment_id"]: row for row in client.get("/api/experiments").json()["experiments"]}
    details = {name: client.get(f"/api/experiments/{name}").json() for name in rows}
    assert calls == []
    assert all(row.get("state") != "unreadable" for row in rows.values())
    recorded = _records(root / "recorded")[-1]
    assert details["recorded"]["forward"]["slices"] == recorded["slices"]
    assert details["recorded"]["forward"]["readout"] == recorded["readout"]
    assert rows["recorded"]["record"]["steps"] == recorded["readout"]["record"]["steps"]


def test_the_revision_moves_when_the_session_records_and_only_then(tmp_path: Path) -> None:
    """A page left open on a running session must learn of a new Validation or
    test step: the status poll carries a revision that moves on a ledger
    append and on a step-tree write, equals the detail's while neither
    happened, and ignores what the poll itself already follows (the status)."""

    directory = build_experiment(tmp_path, "experiment", "research", alive=True)
    client = TestClient(create_app(tmp_path, tmp_path))

    def revisions() -> tuple[str, str]:
        return (
            client.get("/api/experiments/experiment").json()["revision"],
            client.get("/api/experiments/experiment/status").json()["revision"],
        )

    first, polled = revisions()
    assert first == polled
    status = directory / "hitl/status.json"
    status.write_text(status.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    assert revisions() == (first, first)
    refs = AgentRefStore(directory)
    StepTree(directory / "steps").record_failed_attempt(
        epoch_id="research",
        session_ref=refs.get_or_create("session", "research"),
        run_id=refs.get_or_create("run", "run_research_live"),
        result_name="valid_001",
        error="BacktestError",
    )
    second, polled = revisions()
    assert second == polled != first
    ExperimentLedger(directory / "ledgers/experiment_ledger.jsonl").append(
        {
            "record_type": "attempt_failed",
            "experiment_id": "experiment",
            "epoch_id": "research",
            "fold_id": "research",
            "run_id": "run_research_live",
            "session_key": "research",
            "phase": "research",
            "error": "OSError: host lost",
        }
    )
    third, polled = revisions()
    assert third == polled and third not in (first, second)


@pytest.mark.skipif(shutil.which("node") is None, reason="node required for the JS check")
def test_a_moved_revision_rereads_the_experiment_in_place() -> None:
    """The status poll re-reads the experiment when its revision moved: the
    page takes the fresh detail and redraws the control panel and the selected
    step. A read that fails leaves the page as it stands for the next poll,
    and one that returns after the reader left touches nothing."""

    script = (
        Path(__file__).resolve().parents[2] / "src/autotrade/webui/static/app.js"
    ).read_text(encoding="utf-8")
    poll = script.split("async function renderDetailPage(", 1)[1].split("}, 4000);", 1)[0]
    assert "fresh.revision !== detailView.detail.revision" in poll and "await rereadDetail();" in poll
    harness = "\n".join(
        [
            # ``_fn`` starts at ``function``.
            "async " + _fn(script, "rereadDetail"),
            """
const drawn = [];
const controlPanel = (detail) => ({ bar: detail.revision });
const selectStep = (key) => drawn.push(key);
const page = () => ({
  experimentId: "experiment", selectedKey: "research", detail: { revision: "a" },
  barHost: { replaceWith(bar) { drawn.push(bar); } },
});
let detailView = page();
let api = async () => ({ revision: "b" });
(async () => {
  await rereadDetail();
  const reread = [detailView.detail.revision, detailView.barHost, [...drawn]];
  api = async () => { throw new Error("409"); };
  const failed = await rereadDetail().then(() => "read", () => "left as it stands");
  api = async () => { detailView = page(); return { revision: "c" }; };
  drawn.length = 0;
  await rereadDetail();
  console.log(JSON.stringify({ reread, failed, left: [detailView.detail.revision, drawn] }));
})();
""",
        ]
    )
    result = subprocess.run(["node", "-e", harness], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {
        "reread": ["b", {"bar": "b"}, [{"bar": "b"}, "research"]],
        "failed": "left as it stands",
        "left": ["a", []],
    }


def test_a_poll_holding_the_kept_digest_receives_only_the_live_rows(tmp_path: Path) -> None:
    """The home page polls the listing every few seconds, and nearly all of it
    is ended experiments that change only when their files do. A poll that hands back
    the digest of those rows gets the live rows and the order of every id; any
    change to a kept row sends every row again."""

    root = tmp_path / "experiments"
    ended = build_experiment(root, "ended", "deadline")
    build_experiment(root, "running", "research", alive=True)
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


def test_an_archived_experiment_is_listed_only_on_request_and_its_page_still_opens(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cold console lists the experiments not archived without summarizing an
    archived one at all, and counts the archived ones; asked for them, it
    lists those alone. The archived experiment's page opens as before and says it is
    archived."""

    root = tmp_path / "experiments"
    build_experiment(root, "ended", "deadline")
    _shelve(build_experiment(root, "shelved", "forward"))
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
    assert shelf["experiments"][0]["registered_session"] == "research"
    detail = client.get("/api/experiments/shelved").json()
    assert detail["archived"]["reason"] == "historical" and detail["stage"] == shelf["experiments"][0]["stage"]
    assert client.get("/api/experiments/ended").json()["archived"] is None


@pytest.mark.skipif(shutil.which("node") is None, reason="node required for the JS check")
def test_the_console_offers_no_start_for_an_archived_experiment() -> None:
    """The manager never starts an archived experiment, so its experiment page offers
    neither 重启 nor 恢复运行 and its card on the archived page no 恢复运行,
    while the same experiment not archived still gets them. The page and the cards
    are rendered by the console's own functions on a minimal DOM."""

    script = (
        Path(__file__).resolve().parents[2] / "src/autotrade/webui/static/app.js"
    ).read_text(encoding="utf-8")
    assert "experimentGrid(rows, { archived: true })" in _fn(script, "renderArchivedPage")
    # What a card shows beside its controls is not under test.
    context = (
        "experimentName",
        "experimentBadges",
        "experimentBadge",
        "fmtTs",
        "attentionLine",
        "pipelineStepper",
        "activityNode",
        "budgetBars",
        "evidenceTiles",
        "cardEquityNode",
    )
    harness = "\n".join(
        [
            """
const document = {
  createElement: (tag) => ({
    tag, nodeType: 1, style: {}, children: [],
    append(...nodes) { this.children.push(...nodes); },
    addEventListener() {}, setAttribute() {},
  }),
  createTextNode: (text) => ({ nodeType: 3, text }),
};
""",
            *(f"const {name} = () => null;" for name in context),
            _const_block(script, "RESUMABLE_STATES"),
            *(_fn(script, name) for name in ("el", "controlBar", "experimentCard", "experimentGrid")),
            """
const buttons = (node) => node.nodeType !== 1 ? [] : node.tag === "button"
  ? [node.children.map((child) => child.text || "").join("")]
  : node.children.flatMap(buttons);
const archived = { archived_at: "2026-10-24T00:00:00+00:00", reason: "not resumed" };
const dead = {
  experiment_id: "experiment", kind: "hitl", state: "interrupted", worker_alive: false,
  control: { request: null },
};
const live = { ...dead, state: "running_session", worker_alive: true };
console.log(JSON.stringify({
  dead: buttons(controlBar(dead)),
  deadArchived: buttons(controlBar({ ...dead, archived })),
  live: buttons(controlBar(live)),
  liveArchived: buttons(controlBar({ ...live, archived })),
  card: buttons(experimentGrid([dead])),
  archivedCard: buttons(experimentGrid([dead], { archived: true })),
}));
""",
        ]
    )
    result = subprocess.run(["node", "-e", harness], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {
        "dead": ["恢复运行"],
        "deadArchived": [],
        "live": ["暂停", "停止", "强制终止", "重启"],
        "liveArchived": ["暂停", "停止", "强制终止"],
        "card": ["恢复运行", "删除"],
        "archivedCard": ["删除"],
    }


def test_archiving_or_restoring_any_experiment_changes_the_kept_digest(tmp_path: Path) -> None:
    """A poll holding the digest is told of an archive or a restore by a new
    digest and every row, even when the experiment's own row is derived afresh on
    every poll and so is in no kept set."""

    root = tmp_path / "experiments"
    build_experiment(root, "ended", "deadline")
    running = build_experiment(root, "running", "research", alive=True)
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
    build_experiment(root, "experiment", "registered_with_record")
    client = TestClient(create_app(tmp_path, root))
    name = client.get("/api/experiments").json()["experiments"][0]["research_result"]
    url = f"/api/experiments/experiment/results/{name}/equity"
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


def test_an_experiment_with_no_plan_yet_lists_as_created(tmp_path: Path) -> None:
    build_experiment(tmp_path, "experiment", "created")
    row = summarize_experiment(tmp_path / "experiment")
    assert row["state"] == "created"
    assert row["stage"] == "research"
    assert row["research_outcome"] is None
    assert experiment_detail(tmp_path, "experiment")["sessions"] == []


def test_a_plan_of_unknown_sessions_is_listed_as_unreadable(tmp_path: Path) -> None:
    directory = build_experiment(tmp_path, "invalid", "research")
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
    build_experiment(tmp_path, "current", "research")
    rows = {row["experiment_id"]: row for row in experiment_listing(tmp_path)["experiments"]}
    assert rows["invalid"]["state"] == "unreadable"
    assert rows["current"]["stage"] == "research"


def test_the_best_experiment_is_the_registered_strategy_with_the_highest_forward_lower_bound(
    tmp_path: Path,
) -> None:
    """Only an experiment that ended ``registered_with_record`` is offered, by the
    forward lower bound its readout states; a strategy that raised in its
    forward replay has no bound and is not offered; no other ending is ever
    the best, whatever bound it carries. Ties keep the listing order."""

    for experiment_id, stage in (
        ("first", "registered_with_record"),
        ("second", "registered_with_record"),
        ("raised", "registered_with_record"),
        ("researching", "research"),
        ("exhausted", "deadline"),
        ("blocked", "environment_blocked"),
    ):
        build_experiment(tmp_path, experiment_id, stage)

    def restate(name: str, **forward: object) -> None:
        ledger = tmp_path / name / "ledgers/experiment_ledger.jsonl"
        rows = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines()]
        rows[-1]["readout"]["forward"].update(forward)
        ledger.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    bound = _records(tmp_path / "first")[-1]["readout"]["forward"]["lower_bound"]
    restate("second", lower_bound=bound + 0.1)
    restate("raised", lower_bound=None, residual_reading=None, strategy_error={"slice": "forward", "error": "boom"})
    rows = experiment_listing(tmp_path)["experiments"]
    assert best_experiment(rows) == {"experiment_id": "second"}
    by_id = {row["experiment_id"]: row for row in rows}
    assert best_experiment([by_id["raised"]]) is None
    tie = json.loads(json.dumps(by_id["second"]))
    tie["experiment_id"] = "tie"
    assert best_experiment([by_id["second"], tie]) == {"experiment_id": "second"}
    # An ending that registered nothing is never crowned, even carrying a bound.
    for name in ("researching", "exhausted", "blocked"):
        other = json.loads(json.dumps(by_id[name]))
        other["forward"] = by_id["second"]["forward"]
        assert best_experiment([other, by_id["first"]]) == {"experiment_id": "first"}, name
    # The home page serves the same answer with its listing.
    listed = TestClient(create_app(tmp_path, tmp_path)).get("/api/experiments").json()
    assert listed["best"] == {"experiment_id": "second"} and "ranking" not in listed


def test_a_running_experiment_is_never_the_best_experiment(tmp_path: Path) -> None:
    """Ranking running experiments by how far they had got would crown one of a
    field that has produced no forward reading at all: with none the listing
    names no best experiment."""

    build_experiment(tmp_path, "early", "research", alive=True)
    build_experiment(tmp_path, "registered", "forward", alive=True)
    rows = experiment_listing(tmp_path)["experiments"]
    assert best_experiment(rows) is None
    listed = TestClient(create_app(tmp_path, tmp_path)).get("/api/experiments").json()
    assert listed["best"] is None


def test_a_home_card_shows_six_figures_with_a_dash_for_a_reading_not_yet_taken(tmp_path: Path) -> None:
    """A card's evidence is one block of six figures in a fixed order -- the
    validation record's steps out of the validation period's, residual IR,
    p-value and candidate changes, then the forward lower bound and the
    Held-out slice's residual IR -- so every card holds the same six: a reading
    the replay has not produced yet is a dash, never a missing tile. An
    experiment still before its first test reads 0 of its steps. The listing
    row carries the Held-out reading as the forward record states it."""

    root = tmp_path / "experiments"
    for stage in ("research", "forward", "registered_with_record"):
        build_experiment(root, stage, stage)
    rows = {row["experiment_id"]: row for row in experiment_listing(root)["experiments"]}
    heldout = _records(root / "registered_with_record")[-1]["slices"]["heldout"]["residual_reading"]
    assert rows["registered_with_record"]["forward"]["slices"]["heldout"]["residual_reading"] == heldout
    script = (
        Path(__file__).resolve().parents[2] / "src/autotrade/webui/static/app.js"
    ).read_text(encoding="utf-8")
    harness = "\n".join(
        [
            *(_const_block(script, name) for name in ("RECORD_IR_TITLE", "P_VALUE_TITLE", "LOWER_BOUND_TITLE")),
            *(_fn(script, name) for name in ("fmtPct", "fmtSharpe", "fmtPValue", "fmtSteps", "signCls", "evidenceTiles")),
            "const statTilesRow = (tiles) => tiles;",
            f"const rows = {json.dumps(rows)};",
            "console.log(JSON.stringify(Object.fromEntries(",
            "  Object.entries(rows).map(([id, row]) => [id, evidenceTiles(row)]))));",
        ]
    )
    result = subprocess.run(["node", "-e", harness], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    tiles = json.loads(result.stdout)
    labels = ["已测步数", "记录残差 IR", "p 值", "换候选", "前推下界", "Held-out 残差 IR"]
    for name in ("research", "forward", "registered_with_record"):
        assert [tile["label"] for tile in tiles[name]] == labels, name
    assert rows["research"]["record"] is None
    assert [tile["value"] for tile in tiles["research"]] == ["0 / 12", "—", "—", "—", "—", "—"]
    record = rows["forward"]["record"]
    assert [tile["value"] for tile in tiles["forward"]] == [
        f"{record['steps']} / 12",
        f"{record['residual_information_ratio']:.2f}",
        f"{record['p_value']:.3f}",
        str(record["candidate_changes"]),
        "—",
        "—",
    ]
    registered = {tile["label"]: tile for tile in tiles["registered_with_record"]}
    assert registered["Held-out 残差 IR"]["value"] == f"{heldout['residual_information_ratio']:.2f}"
    assert registered["前推下界"]["value"] != "—"
    assert all(tile["value"] != "—" for tile in tiles["registered_with_record"])


def test_the_listing_carries_the_registration_and_the_record_so_far(tmp_path: Path) -> None:
    """The experiment card draws its registration step and its evidence from the
    listing alone, and must read the same numbers the experiment page does."""

    build_experiment(tmp_path, "researching", "research")
    build_experiment(tmp_path, "registered", "forward")
    build_experiment(tmp_path, "recorded", "registered_with_record")
    build_experiment(tmp_path, "fresh", "created")
    rows = {row["experiment_id"]: row for row in experiment_listing(tmp_path)["experiments"]}
    assert rows["researching"]["registered_session"] is None
    assert rows["registered"]["registered_session"] == "research"
    # An experiment without a recorded step has no record to read, but states
    # the steps its validation period tiles into: three validation years in
    # quarters, by the pipeline's own tiling.
    assert rows["fresh"]["record"] is None and rows["researching"]["record"] is None
    assert {rows[name]["validation_steps"] for name in ("fresh", "researching", "registered", "recorded")} == {12}
    assert experiment_detail(tmp_path, "researching")["validation_steps"] == 12
    record = rows["recorded"]["record"]
    detail = experiment_detail(tmp_path, "recorded")["record"]
    for field in ("steps", "residual_information_ratio", "p_value", "candidate_changes", "test_horizon", "start", "end"):
        assert record[field] == detail[field], field
        assert record[field] is not None, field
    # The page adds the rows and the curve the card does not draw.
    assert len(detail["rows"]) == record["steps"] == len(detail["curve"]["steps"])
    assert len(detail["curve"]["dates"]) == len(detail["curve"]["book"]) == len(detail["curve"]["baseline"]) > 0
    assert "rows" not in record and "curve" not in record


@pytest.mark.parametrize(
    ("patch", "message"),
    [
        ({"research_start": "20210101"}, "first of a month"),
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
                **TEST_GEOMETRY.to_record(), **CREATE_RULES, **NO_SEED,
                **patch,
            }
        },
    )
    assert response.status_code == 400
    assert message in response.json()["detail"]
    assert not (tmp_path / "experiments/geometry").exists()


def test_the_step_tree_names_the_session_and_the_registered_node(tmp_path: Path) -> None:
    directory = build_experiment(tmp_path, "experiment", "forward")
    refs = AgentRefStore(directory)
    frozen = _records(directory)[0]["registered"]
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
    records[0]["registered"]["source_step_id"] = node_id
    ledger.path.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
    [node] = step_tree_view(directory)["nodes"]
    assert node["session_key"] == "research"
    assert node["registered"] is True
    assert node["session_ref"].startswith("session_ref_")


def test_a_failed_forward_replay_attempt_is_counted(tmp_path: Path) -> None:
    """The verdict view's replay failures are the forward replay's
    ``attempt_failed`` rows, which the pipeline writes under its stage name
    (``forward``), not under the provider phase its replay reads (``heldout``):
    filtering on the latter counted none, ever."""

    directory = build_experiment(tmp_path, "experiment", "registered_with_record")
    assert experiment_detail(tmp_path, "experiment")["replay_attempts"] == {
        "failed": 0,
        "last_error": None,
    }
    ledger = ExperimentLedger(directory / "ledgers/experiment_ledger.jsonl")
    for stage in ("research", "forward"):
        # The attempt keys ``ExperimentPipeline`` writes on its failure path.
        ledger.append(
            {
                "experiment_id": "experiment",
                "epoch_id": stage,
                "fold_id": stage,
                "run_id": f"run_{stage}_failed",
                "session_key": stage,
                "phase": stage,
                "record_type": "attempt_failed",
                "error": f"OSError: {stage} host lost",
            }
        )
    assert experiment_detail(tmp_path, "experiment")["replay_attempts"] == {
        "failed": 1,
        "last_error": "OSError: forward host lost",
    }


@pytest.mark.skipif(shutil.which("node") is None, reason="node required for the JS check")
def test_the_validation_record_names_a_candidate_by_its_handle(tmp_path: Path) -> None:
    """A step's candidate is a node id over a hundred characters long; written
    out it pushed the step's figures off the screen. The table shows the
    node's handle and keeps the whole id in the cell's hover."""

    build_experiment(tmp_path, "experiment", "registered_with_record")
    record = experiment_detail(tmp_path, "experiment")["record"]
    script = (
        Path(__file__).resolve().parents[2] / "src/autotrade/webui/static/app.js"
    ).read_text(encoding="utf-8")
    harness = "\n".join(
        [
            """
const node = (tag) => ({
  tag, nodeType: 1, style: {}, children: [], attrs: {},
  append(...nodes) { this.children.push(...nodes); },
  addEventListener() {}, setAttribute(key, value) { this.attrs[key] = value; },
});
const document = { createElement: node, createTextNode: (text) => ({ nodeType: 3, text }) };
// The router's listeners sit between dataTable and the next function.
const window = { addEventListener() {} };
const route = null;
const equityChart = () => null;
const fmtAmount = String;
""",
            _const_block(script, "SOURCE_LABELS"),
            *(
                _fn(script, name)
                for name in (
                    "el", "panelHead", "dataTable", "fmtSteps", "fmtSharpe", "fmtPValue",
                    "fmtPct", "signCls", "drawdownOf", "recordPanel",
                )
            ),
            f"const record = {json.dumps({**record, 'curve': {}})};",
            """
const text = (item) => item.nodeType === 3 ? item.text : item.children.map(text).join("");
const find = (item, tag) => item.nodeType !== 1 ? [] : [
  ...(item.tag === tag ? [item] : []), ...item.children.flatMap((child) => find(child, tag)),
];
const cell = find(find(recordPanel({ record, validation_steps: 4 }), "tr")[1], "td")[2];
console.log(JSON.stringify([text(cell), cell.attrs.title]));
""",
        ]
    )
    result = subprocess.run(["node", "-e", harness], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    candidate = record["rows"][0]["candidate"]
    assert json.loads(result.stdout) == [candidate.rsplit("__", 1)[1], candidate]
    assert candidate.rsplit("__", 1)[1].startswith("valid_")


def _object_const(script: str, name: str) -> str:
    """One ``const NAME = {…};`` of app.js, whose comments may hold a semicolon."""

    start = script.index(f"const {name} = {{")
    return script[start : script.index("\n};", start) + 3]


def _pipeline_steps(rows: dict[str, dict[str, object]]) -> dict[str, list[list[str]]]:
    """Each listing row's five steps as the console's own stepper draws them:
    key, node state and status word."""

    script = (
        Path(__file__).resolve().parents[2] / "src/autotrade/webui/static/app.js"
    ).read_text(encoding="utf-8")
    harness = "\n".join(
        [
            *(
                _object_const(script, name)
                for name in ("STATE_LABELS", "OUTCOME_LABELS", "ENDING_LABELS", "STEP_LABELS", "STEP_STATUS_LABELS", "ENDING_STEP_STATES")
            ),
            *(
                _fn(script, name)
                for name in ("endingKey", "endingLabel", "researchStep", "pipelineTailSteps", "pipelineSteps")
            ),
            f"const rows = {json.dumps(rows)};",
            "console.log(JSON.stringify(Object.fromEntries(Object.entries(rows).map(([id, row]) =>",
            "  [id, pipelineSteps(row).map((step) => [step.key, step.state, step.status])]))));",
        ]
    )
    result = subprocess.run(["node", "-e", harness], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


@pytest.mark.skipif(shutil.which("node") is None, reason="node required for the JS check")
def test_a_replay_that_failed_is_drawn_failed_not_done_or_pending(tmp_path: Path) -> None:
    """The forward replay can fail two ways and neither is a replay done. The
    registered strategy's own error is recorded with the readout and no slice:
    the ending names the slice it raised in and both replay steps wear the
    cross. A worker that broke on the replay leaves no record: the replay
    steps wear the cross, not the outline of a step still to run. A replay
    that is merely waiting keeps its outline."""

    root = tmp_path / "experiments"
    waiting = build_experiment(root, "waiting", "forward")
    raised = build_experiment(root, "raised", "forward")
    error = {"slice": "heldout", "error": "KeyError: 'close'"}
    ExperimentLedger(raised / "ledgers/experiment_ledger.jsonl").append(
        {
            # As ``ExperimentPipeline`` records a replay its strategy raised in.
            "record_type": "forward",
            "experiment_id": "raised",
            "epoch_id": "forward",
            "fold_id": "forward",
            "run_id": "run_forward",
            "session_key": "forward",
            "artifact_id": "strategy_research_abc",
            "replay": REPLAY,
            "status": "strategy_error",
            "strategy_error": error["slice"],
            "error": error["error"],
            "result_ref": None,
            "slices": None,
            "refits_executed": None,
            "null_control": None,
            "readout": validation_record.readout(_records(raised), alpha=0.05, strategy_error=error),
        }
    )
    write_json_atomic(raised / "hitl/status.json", {"schema_version": 1, "state": "completed"})
    broken = build_experiment(root, "broken", "forward")
    for attempt in (1, 2, 3):
        ExperimentLedger(broken / "ledgers/experiment_ledger.jsonl").append(
            {
                "record_type": "attempt_failed",
                "experiment_id": "broken",
                "epoch_id": "forward",
                "fold_id": "forward",
                "run_id": f"run_forward_{attempt}",
                "session_key": "forward",
                "phase": "forward",
                "error": "OSError: replay host lost",
            }
        )
    write_json_atomic(
        broken / "hitl/status.json",
        {"schema_version": 1, "state": "failed", "session_key": "forward", "error": "OSError: replay host lost"},
    )
    assert waiting.is_dir()
    rows = {row["experiment_id"]: row for row in experiment_listing(root)["experiments"]}
    assert rows["raised"]["ending"]["state"] == "registered_with_record"
    assert rows["raised"]["ending"]["reason"].endswith(" · Held-out 段策略报错")
    assert rows["raised"]["forward"]["error"] == "KeyError: 'close'"
    assert rows["raised"]["paper_openable"] is False
    assert rows["broken"]["ending"] == {"state": "broken", "reason": "OSError: replay host lost"}
    drawn = _pipeline_steps(rows)
    done = [["research", "done", "验证期完成"], ["registered", "done", "已定版"]]
    assert drawn["raised"] == [
        *done,
        ["forward", "failed", "策略报错"],
        ["heldout", "failed", "策略报错"],
        ["readout", "done", "通过" if rows["raised"]["record"]["passed"] else "未通过"],
    ]
    assert drawn["broken"] == [
        *done,
        ["forward", "failed", "失败"],
        ["heldout", "failed", "失败"],
        ["readout", "ended", "失败"],
    ]
    assert drawn["waiting"][2:4] == [["forward", "pending", "待记录"], ["heldout", "pending", "待记录"]]


@pytest.mark.skipif(shutil.which("node") is None, reason="node required for the JS check")
def test_the_steps_after_research_wait_on_what_the_pipeline_waits_on(tmp_path: Path) -> None:
    """While research runs nothing is registered, replayed or read: the
    registration and the replay wait on the registration, which the last test
    or the host makes, and the readout waits on its record. Nothing is
    nominated and nothing is judged."""

    row = summarize_experiment(build_experiment(tmp_path, "experiment", "research", alive=True))
    assert _pipeline_steps({"experiment": row})["experiment"] == [
        ["research", "running", "运行中"],
        ["registered", "pending", "待定版"],
        ["forward", "pending", "待定版"],
        ["heldout", "pending", "待定版"],
        ["readout", "pending", "待记录"],
    ]


def test_failed_research_validations_are_counted_apart_from_the_replay(tmp_path: Path) -> None:
    """A research Validation that fails is no attempt failure: it leaves its
    dead-end node in the step tree and no ledger row, so the replay's attempt
    counter read 0 for an experiment whose four Validations had failed. The detail
    counts them on their own -- how many, how many the environment failed and
    refunded, and the last reason -- and leaves the replay counter alone."""

    directory = build_experiment(tmp_path, "experiment", "registered_with_record")
    assert experiment_detail(tmp_path, "experiment")["research_validation_failures"] == {
        "failed": 0,
        "environment": 0,
        "last_error": None,
        "validation_period_failed": 0,
        "validation_period_last_error": None,
    }
    refs = AgentRefStore(directory)
    tree = StepTree(directory / "steps")
    for name, cause, error in (
        ("valid_001", "strategy", "daily Validation failed: BacktestError: fit failed: label has 2937"),
        ("valid_006", "environment", "daily Validation failed: CUDA out of memory: the device was taken"),
    ):
        # As ``batch_validate`` records a failed candidate.
        tree.record_failed_attempt(
            epoch_id="research",
            session_ref=refs.get_or_create("session", "research"),
            run_id=refs.get_or_create("run", "run_research"),
            result_name=name,
            error=error,
            metadata={"batch_id": "batch_a", "candidate": name, "cause": cause},
        )
    # A failed test leg tells the session its class alone; the host keeps the
    # error, which the operator reads here.
    record_validation_period_failure(
        directory, StrategyRaised("KeyError: 'close' on 20250815"), tool="test", step=1, candidate="node"
    )
    detail = experiment_detail(tmp_path, "experiment")
    assert detail["research_validation_failures"] == {
        "failed": 2,
        "environment": 1,
        "last_error": "daily Validation failed: CUDA out of memory: the device was taken",
        "validation_period_failed": 1,
        "validation_period_last_error": "StrategyRaised: KeyError: 'close' on 20250815",
    }
    assert detail["replay_attempts"] == {"failed": 0, "last_error": None}
    # The step-tree view marks the same nodes failed.
    nodes = step_tree_view(directory)["nodes"]
    assert sum(node["status"] == "failed" for node in nodes) == 2


def test_each_validation_shows_its_raw_excess_beside_its_active_figures(tmp_path: Path) -> None:
    build_experiment(tmp_path, "experiment", "registered_with_record")
    detail = experiment_detail(tmp_path, "experiment")
    record = next(entry for entry in detail["sessions"] if entry["kind"] == "research")["record"]
    assert {"excess_return", "plain_excess", "excess_at_cost_stress"} <= set(record["validations"][0])


def test_the_forward_curve_draws_its_random_baseline_on_the_books_days(tmp_path: Path) -> None:
    """The forward replay's curve draws the replay's random baseline beside the
    book, on the book's own days."""

    from autotrade.environment.replay.style import STYLE_ARTIFACT_NAME

    directory = build_experiment(tmp_path, "experiment", "registered_with_record")
    record = _records(directory)[-1]
    detail = experiment_detail(tmp_path, "experiment")
    sidecar = json.loads(
        (Path(str(record["result_ref"])).parent / STYLE_ARTIFACT_NAME).read_text(encoding="utf-8")
    )
    client = TestClient(create_app(tmp_path, tmp_path))
    equity = client.get(f"/api/experiments/experiment/results/{detail['forward']['result']}/equity").json()
    assert equity["baseline"]["dates"] == equity["series"][0]["dates"]
    assert equity["baseline"]["final"] == pytest.approx(
        float(np.prod([1.0 + value for _day, value in sidecar["baseline_daily"]]) - 1.0), abs=1e-6
    )


def test_an_experiment_naming_an_unknown_parameter_is_unreadable(tmp_path: Path) -> None:
    """A params.json naming a key no parameter declares lists as unreadable,
    beside the experiments that still list."""

    build_experiment(tmp_path, "current", "research")
    path = build_experiment(tmp_path, "unknown", "research") / "hitl/params.json"
    params = json.loads(path.read_text(encoding="utf-8"))
    path.write_text(json.dumps({**params, "ablation_required": True}), encoding="utf-8")
    rows = {row["experiment_id"]: row for row in experiment_listing(tmp_path)["experiments"]}
    assert rows["unknown"]["state"] == "unreadable"
    assert rows["current"]["stage"] == "research"


def test_every_record_of_a_validation_window_is_listed_with_its_p_value_and_count(tmp_path: Path) -> None:
    """The multiplicity readout: an experiment's readout view lists every frozen
    strategy's validation record over the same validation period, archived experiments
    included, with its one-sided null p-value and how many there are, and
    pools the forward-year readings of every registered strategy over the same
    forward months. Nothing is adjusted: each experiment keeps its own reading."""

    from autotrade.pipelines.verdict import pooled_reading

    build_experiment(tmp_path, "good", "registered_with_record")
    build_experiment(tmp_path, "other", "registered_with_record")
    build_experiment(tmp_path, "early", "research")
    readouts = experiment_detail(tmp_path, "good")["readouts"]
    window = readouts["validation_window"]
    assert window["count"] == 2
    by_experiment = {row["experiment_id"]: row for row in window["readings"]}
    assert set(by_experiment) == {"good", "other"}
    for name in ("good", "other"):
        block = _records(tmp_path / name)[-1]["readout"]["record"]
        assert (window["start"], window["end"]) == (block["start"], block["end"])
        assert by_experiment[name]["p_value"] == block["p_value"]
        assert (by_experiment[name]["steps"], by_experiment[name]["candidate_changes"]) == (block["steps"], block["candidate_changes"])
    # Most significant first.
    assert [row["p_value"] for row in window["readings"]] == sorted(row["p_value"] for row in window["readings"])
    pooled = pooled_reading(
        [_records(tmp_path / name)[-1]["slices"]["forward"]["residual_reading"] for name in ("good", "other")]
    )
    assert readouts["forward_pool"] == {
        "start": REPLAY["start"],
        "end": REPLAY["forward_end"],
        **{key: pytest.approx(value) for key, value in pooled.items()},
        "raised": 0,
    }
    # A strategy that raised in its forward replay has no reading to pool and
    # is counted, never left out silently.
    crashed = build_experiment(tmp_path, "crashed", "registered_with_record")
    ledger = crashed / "ledgers/experiment_ledger.jsonl"
    lines = ledger.read_text(encoding="utf-8").splitlines()
    record = json.loads(lines[-1])
    record.update(status="strategy_error", strategy_error="forward", slices=None)
    record["readout"]["forward"] = {
        "lower_bound": None,
        "residual_reading": None,
        "strategy_error": {"slice": "forward", "error": "StrategyRaised: boom"},
    }
    ledger.write_text("\n".join([*lines[:-1], json.dumps(record)]) + "\n", encoding="utf-8")
    pool = experiment_detail(tmp_path, "good")["readouts"]["forward_pool"]
    assert (pool["count"], pool["raised"]) == (2, 1)
    # An experiment without a forward record has no readout.
    assert experiment_detail(tmp_path, "early")["readouts"] is None


