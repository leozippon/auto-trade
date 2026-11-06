"""The research console over a research experiment's ledger.

An experiment runs one research session, freezes at most once and is then replayed once
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
from autotrade.environment.identity import AgentRefStore
from autotrade.environment.step_tree import StepTree
from autotrade.environment.tools.report_issue import (
    append_issue_report,
    append_issue_resolution,
    issue_reports_path,
)
from autotrade.pipelines.config import (
    DEFAULT_RESEARCH_GEOMETRY,
    OBSOLETE_KEYS,
    RETIRED_SWITCHES,
)
from autotrade.pipelines.ledger import ExperimentLedger, recorded_path, verdict_void
from autotrade.pipelines.research_session import session_role_quotas
from autotrade.webui import registry
from autotrade.webui.registry import (
    ENDING_STATES,
    experiment_detail,
    experiment_listing,
    ranking,
    summarize_experiment,
)
from autotrade.webui.server import create_app
from autotrade.webui.steps import step_tree_view
from tests.unit.acceptance_fixture import CREATE_RULES, REQUIRED_RULES
from tests.unit.test_webui_inject_message import _const_block, _fn
from tests.unit.webui_research_experiment import (
    ENVIRONMENT_BLOCKED_REASON,
    REPLAY,
    _session_record,
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


def test_a_researching_experiment_shows_its_plan_and_nothing_frozen(tmp_path: Path) -> None:
    build_experiment(tmp_path, "experiment", "research")
    detail = experiment_detail(tmp_path, "experiment")
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


def test_a_frozen_experiment_lists_its_validations_and_the_registration_it_froze_under(
    tmp_path: Path,
) -> None:
    """A recorded session serves its outcome, attempts, budget, the registration
    the freeze passed (which carries no score) and every Validation it ran;
    research results are development evidence, readable as they land."""

    build_experiment(tmp_path, "experiment", "sealed")
    record = experiment_detail(tmp_path, "experiment")["sessions"][0]["record"]
    assert record["outcome"] == "freeze"
    assert record["froze"] is True
    assert record["attempts"] == 1
    assert record["budget_used"]["test_replays"] == 8
    assert "best" not in record
    client = TestClient(create_app(tmp_path, tmp_path))
    name = Path(str(_records(tmp_path / "experiment")[0]["steps"][0]["validation_result_ref"])).parent.name
    assert client.get(f"/api/experiments/experiment/results/{name}/equity").status_code == 200
    # The record says whether its trace is on disk; this synthetic experiment has none.
    assert record["trace"] is False
    assert record["freeze_gate"]["passed"] is True
    assert record["freeze_gate"]["deflated_sharpe_probability"] is None
    assert {row["span"] for row in record["validations"]} == {"full", "Y3"}


def test_a_frozen_experiment_is_sealed_until_its_verdict_exists(tmp_path: Path) -> None:
    directory = build_experiment(tmp_path, "experiment", "sealed", alive=True)
    detail = experiment_detail(tmp_path, "experiment")
    assert detail["stage"] == "forward"
    assert detail["status"]["session_key"] == "forward"
    assert detail["verdict"] is None
    assert detail["forward"] is None
    assert detail["paper_candidate"] is None
    frozen = detail["frozen"]
    assert frozen["session_key"] == "research"
    assert frozen["strategy_ref"].startswith("strategy_ref_")
    assert "strategy_research_abc" not in json.dumps(detail)
    assert frozen["full_span_validations"] is None
    assert frozen["null_percentile"] == 0.81
    assert frozen["forward_mde"] > 0
    assert frozen["blocks"][0]["label"] == "202407-202506"
    # The running replay's result directory is on disk, and nothing serves it:
    # it answers exactly like a result that does not exist.
    [running] = _result_names(tmp_path, "experiment", "heldout")
    client = TestClient(create_app(tmp_path, tmp_path))
    for route in ("equity", "style", "orders", "orders.csv"):
        sealed = client.get(f"/api/experiments/experiment/results/{running}/{route}")
        absent = client.get(f"/api/experiments/experiment/results/heldout_absent/{route}")
        assert sealed.status_code == absent.status_code == 404, route
        assert sealed.json()["detail"].replace(running, "") == absent.json()[
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
    result -- one recorded before style analysis existed, one written under an
    older schema -- and the console has to tell it from "no such result". The
    two must differ in the payload, never only in the wording of an error."""

    directory = build_experiment(tmp_path, "experiment", "graduated")
    client = TestClient(create_app(tmp_path, tmp_path))
    name = experiment_detail(tmp_path, "experiment")["frozen"]["result"]
    present = client.get(f"/api/experiments/experiment/results/{name}/style")
    assert present.status_code == 200
    assert present.json()["available"] is True

    # A version-1 sidecar, written before the factor regression replaced the
    # regressed blocks, holds the same series and is served alike.
    sidecar = directory / "artifacts/results" / name / "style_analysis.json"
    sidecar.write_text(json.dumps({**json.loads(sidecar.read_text(encoding="utf-8")), "schema_version": 1}), encoding="utf-8")
    assert client.get(f"/api/experiments/experiment/results/{name}/style").json()["available"] is True
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
    build_experiment(old_root, "experiment", "graduated")
    new_root = tmp_path / "new/experiments"
    new_root.parent.mkdir()
    old_root.rename(new_root)
    assert str(old_root / "experiment") in json.dumps(_records(new_root / "experiment"))
    client = TestClient(create_app(tmp_path / "new", new_root))
    name = experiment_detail(new_root, "experiment")["forward"]["result"]
    for route in ("equity", "style", "orders", "orders.csv"):
        response = client.get(f"/api/experiments/experiment/results/{name}/{route}")
        assert response.status_code == 200, route
    assert client.get("/api/experiments/experiment/frozen/strategy.zip").status_code == 200

    (new_root / "experiment/artifacts/results" / name / "result.json").unlink()
    missing = client.get(f"/api/experiments/experiment/results/{name}/equity")
    assert missing.status_code == 404
    assert missing.json()["detail"] == (
        f"experiment: artifacts/results/{name}/result.json is recorded but not on disk"
    )
    with pytest.raises(FileNotFoundError, match="outside the experiment"):
        recorded_path(new_root / "experiment", tmp_path / "elsewhere/experiment/result.json")


@pytest.mark.parametrize("status", ["graduated", "discarded"])
def test_a_sealed_gate_record_still_opens_its_forward_replay(tmp_path: Path, status: str) -> None:
    """An experiment judged under the retired sealed gate keeps its verdict and its
    replay readable: the record is shown as written, and no readout is read
    off it."""

    build_experiment(tmp_path, "experiment", status)
    detail = experiment_detail(tmp_path, "experiment")
    assert detail["stage"] == "readout"
    assert detail["forward"]["readout"] is None and detail["readouts"] is None
    assert detail["verdict"]["status"] == status
    record = _records(tmp_path / "experiment")[-1]
    assert detail["verdict"]["reasons"] == record["verdict"]["reasons"]
    forward = detail["forward"]
    assert forward["replay"] == REPLAY
    assert forward["verdict"]["status"] == status
    assert forward["slices"]["forward"]["lower_bound"] == record["slices"]["forward"]["lower_bound"]
    assert forward["slices"]["heldout"]["residual_reading"] == record["slices"]["heldout"]["residual_reading"]
    assert forward["null_percentile"] == 0.77
    assert "result_ref" not in json.dumps(detail)
    client = TestClient(create_app(tmp_path, tmp_path))
    equity = client.get(f"/api/experiments/experiment/results/{forward['result']}/equity").json()
    [strategy] = equity["series"]
    assert strategy["dates"][0] == REPLAY["start"]
    assert equity["benchmark"]["dates"] == strategy["dates"]
    assert equity["exposure"]["strategy"]["long"][0] == pytest.approx(0.9)
    orders = client.get(f"/api/experiments/experiment/results/{forward['result']}/orders").json()
    assert orders["row_count"] == 2
    if status == "graduated":
        assert detail["verdict"]["reasons"] == []
        assert detail["paper_candidate"] == {"artifact_id": "strategy_research_abc"}
    else:
        # The sealed gate alone decides: this book read below its threshold.
        assert detail["verdict"]["reasons"] == ["sealed_residual_ir_below_bar"]
        assert detail["paper_candidate"] is None


def test_a_research_that_froze_nothing_has_a_verdict_and_no_replay(tmp_path: Path) -> None:
    build_experiment(tmp_path, "experiment", "no_deliverable")
    detail = experiment_detail(tmp_path, "experiment")
    assert detail["stage"] == "readout"
    assert detail["verdict"] == {
        "status": "no_deliverable",
        "reasons": ["no_edge: nothing survived"],
    }
    assert detail["frozen"] is None
    assert detail["forward"] is None
    assert detail["sessions"][0]["record"]["experiment_end"]["status"] == "no_deliverable"
    assert detail["record"] is None


def test_every_way_an_experiment_can_end_reads_as_one_ending_with_a_reason(tmp_path: Path) -> None:
    """The console shows an experiment's ending in one vocabulary, so the projection
    must classify every ending the pipeline can produce and give each its own
    reason; an experiment that can still run has none at all."""

    for experiment_id, stage in (
        ("frozen_with_record", "frozen_with_record"),
        ("graduated", "graduated"),
        ("rejected", "discarded"),
        ("no_edge", "no_deliverable"),
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
    # A graduate is named by its forward year as the holder reads it -- his
    # account, the benchmark, the account against the random baseline, then
    # the active IR the gate graded -- a refusal by
    # the criteria it failed, and the three endings no replay decided by what
    # the record says: the Agent's first sentence, the budget, the error's
    # first line.
    forward = rows["graduated"]["forward"]["slices"]["forward"]
    raw = forward["raw_readings"]
    assert endings["graduated"]["reason"] == (
        f"前推 账户 {raw['strategy_return'] * 100:+.2f}%"
        f" · 基准 {raw['benchmark_return'] * 100:+.2f}%"
        f" · 对面板 {raw['plain_selection'] * 100:+.2f}%"
        f" · 主动 IR {forward['information_ratio']:.2f}"
    )
    # A frozen strategy is named by its readout: the record's significance test and
    # the forward lower bound.
    readout = rows["frozen_with_record"]["forward"]["readout"]
    assert endings["frozen_with_record"]["reason"] == (
        f"滚动记录 残差 IR {readout['record']['residual_information_ratio']:.2f}"
        f" · p {readout['record']['p_value']:.2f} · 换候选 0 次"
        f" · 前推下界 {readout['forward']['lower_bound'] * 100:+.2f}%"
    )
    assert rows["frozen_with_record"]["paper_candidate"] is not None
    # The retired sealed gate's reason token leads a rejected experiment's reasons.
    assert endings["rejected"]["reason"].split(" · ")[0] == "sealed_residual_ir_below_bar"
    assert rows["graduated"]["paper_candidate"] is not None
    assert endings["no_edge"]["reason"] == "没有候选值得冻结"
    assert endings["budget_exhausted"]["reason"] == "模型调用次数用尽"
    # The host's reason in full, its error included: that is what to act on.
    assert endings["environment_blocked"]["reason"] == ENVIRONMENT_BLOCKED_REASON
    assert endings["broken"]["reason"] == "RuntimeError: sandbox image is gone"
    # The homepage lists what can still run first, then the endings in order.
    listed = [row["experiment_id"] for row in experiment_listing(tmp_path)["experiments"]]
    assert listed[-len(ENDING_STATES) :] == list(ENDING_STATES)
    assert "running" in listed[: -len(ENDING_STATES)]
    # The detail page reads the same projection as the card.
    assert experiment_detail(tmp_path, "no_edge")["ending"] == endings["no_edge"]


def test_the_home_list_flags_a_blocked_experiment_and_an_unresolved_issue(tmp_path: Path) -> None:
    """Two things put an attention line on an experiment's card and nothing else
    does: the host ending its session on the environment, and an issue report
    its session filed that nobody has resolved while the experiment can still run.
    Resolving the report clears the line even on a row the listing keeps
    without re-reading, and an experiment that ended no longer carries its reports."""

    build_experiment(tmp_path, "blocked", "environment_blocked")
    stopped = build_experiment(tmp_path, "stopped", "research")
    ended = build_experiment(tmp_path, "ended", "no_deliverable")

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
        directory = build_experiment(tmp_path, experiment_id, stage)
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
    experiments = {"graduated", "refused", "withdrawn", "no_edge"}
    rows = [row for row in experiment_listing(tmp_path)["experiments"] if row["experiment_id"] in experiments]
    withdrawn = next(row for row in rows if row["experiment_id"] == "withdrawn")
    # One ending word with the refusals; the reason says it had graduated and
    # why it no longer has, in the void's first sentence.
    assert withdrawn["ending"] == {
        "state": "rejected",
        "reason": "验收通过后复核未通过：复跑计入股息税后研究期主动 IR 为 −1.80",
    }
    # Listed in the failures' group, newest first inside it, ahead of the
    # endings nobody decided whatever its date.
    assert [row["experiment_id"] for row in rows] == ["graduated", "withdrawn", "refused", "no_edge"]
    # Nothing judged under the sealed gate is ranked: the ranking lists frozen
    # strategies with a readout, and these experiments have none.
    assert ranking(rows) == []
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
    """An experiment judged on its training seeds together shows each seed and their
    mean on the experiment page -- the freeze's seed-mean IR beside the
    nominee's and the threshold, and the forward and Held-out readings per seed --
    and the series F8 judges them on together -- from the fields the console
    serves; an experiment frozen on one seed serves none of them and draws nothing
    in their place."""

    build_experiment(tmp_path, "seeded", "graduated", seeded=True)
    build_experiment(tmp_path, "single", "graduated")
    seeded = experiment_detail(tmp_path, "seeded")
    single = experiment_detail(tmp_path, "single")
    gate = seeded["sessions"][0]["record"]["freeze_gate"]["seed_replicates"]
    frozen = seeded["frozen"]["seed_replicates"]
    forward = seeded["forward"]["slices"]["forward"]
    # Registration reads the replicates and judges no seed mean against a threshold.
    assert frozen == gate
    assert (gate["mean_information_ratio"], gate["information_ratio_bar"]) == (None, None)
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
    assert single["frozen"]["seed_replicates"] is None
    alone = single["sessions"][0]["record"]["freeze_gate"]["seed_replicates"]
    assert (alone["replicates"], alone["mean_information_ratio"]) == ([], None)
    assert "seed_mean" not in single["forward"]["slices"]["forward"]
    assert isinstance(single["forward"]["slices"]["forward"]["lower_bound"], float)

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
    for panel in ("forwardStagePanel", "heldoutStagePanel", "readoutStagePanel"):
        assert "seedReadingsTable(" in body(panel), panel
    # Never on the home page.
    for home in ("experimentCard", "rankingPanel", "evidenceTiles"):
        assert "seed" not in body(home), home


def test_the_listing_rederives_a_row_only_when_its_files_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The home page polls the listing every 5 s over every experiment ever created:
    an ended experiment with no live worker is re-read only when its status, ledger,
    params or identity map changes, while an experiment with a live worker is re-read
    on every poll."""

    root = tmp_path / "experiments"
    ended = build_experiment(root, "ended", "no_deliverable")
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


def test_an_experiment_whose_nomination_the_freeze_gate_refused_reads_as_rejected(
    tmp_path: Path,
) -> None:
    """The gate refusing a nomination ends the experiment without a replay. It is not
    the Agent's own "no candidate was worth freezing", so it reads as the
    refusal it is rather than as 未发现超额."""

    directory = build_experiment(tmp_path, "experiment", "research")
    ExperimentLedger(directory / "ledgers/experiment_ledger.jsonl").append(
        _session_record(
            "experiment",
            outcome="freeze",
            steps=[],
            trials_to_date=1,
            freeze_gate={"passed": False, "reasons": ["freeze_deflated_sharpe_below_threshold"]},
            experiment_end={
                "status": "no_deliverable",
                "reason": "freeze refused by the gate (freeze_deflated_sharpe_below_threshold)",
            },
        )
    )
    assert summarize_experiment(directory)["ending"] == {
        "state": "rejected",
        "reason": "提名未通过定版登记",
    }


def test_the_listing_carries_the_budget_and_the_research_outcome(tmp_path: Path) -> None:
    """The budget bars read the experiment's ceilings (worker defaults under the
    params) and what the session spent: the ledger's block once recorded,
    else the last block of the live session's trace, else nothing."""

    live = build_experiment(tmp_path, "live", "research", alive=True)
    build_experiment(tmp_path, "idle", "research")
    build_experiment(tmp_path, "judged", "graduated")
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
    assert (rows["live"]["research_outcome"], rows["judged"]["research_outcome"]) == (None, "freeze")
    client = TestClient(create_app(tmp_path.parent, tmp_path))
    assert client.get("/api/experiments/live/status").json()["budget_used"]["llm_calls"] == 7
    assert client.get("/api/experiments/idle/status").json()["budget_used"] is None


def test_a_terminated_session_shows_what_it_spent_and_no_record(tmp_path: Path) -> None:
    """A session the operator terminated before it recorded anything has no
    ledger record, and the console invents none. Its trace still says what it
    spent, so the budget reads off the trace with the worker gone, and the
    research step says how it stopped and replays that trace instead of
    reading as a step yet to run."""

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

    script = (
        Path(__file__).resolve().parents[2] / "src/autotrade/webui/static/app.js"
    ).read_text(encoding="utf-8")

    def body(name: str) -> str:
        return script.split(f"function {name}(", 1)[1].split("\nfunction ", 1)[0]

    step = body("researchStep")
    assert 'status.session_key === "research" && status.run_ref' in step
    assert "STATE_LABELS[item.state]" in step and 'STATE_LABELS = {' in script
    assert 'terminated: "已强制终止"' in script
    panel = body("sessionDetailPanel")
    assert "traceReplayNode(detail.experiment_id, runRef, detail)" in panel
    assert "没有研究记录" in panel


def test_the_listing_names_the_research_curve_and_the_replay_its_thresholds(tmp_path: Path) -> None:
    """The one curve per experiment starts from the research-period validation of the
    candidate the tiles name — the frozen artifact's own once the experiment froze —
    and the replay's step lists the thresholds the verdict will hold it to
    before the record exists, from the experiment's parameters; the record's own
    block carries the same keys afterwards."""

    build_experiment(tmp_path, "researching", "research")
    build_experiment(tmp_path, "frozen", "sealed")
    build_experiment(tmp_path, "judged", "graduated")
    rows = {row["experiment_id"]: row for row in experiment_listing(tmp_path)["experiments"]}
    assert rows["researching"]["research_result"] is None
    frozen_ref = _records(tmp_path / "frozen")[0]["frozen"]["research_result_ref"]
    assert rows["frozen"]["research_result"] == Path(str(frozen_ref)).parent.name

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
    recorded = experiment_detail(tmp_path, "judged")["forward"]["verdict"]["thresholds"]
    assert recorded["alpha"] == preview["alpha"]
    # An experiment created before the sealed gate can run no forward stage: none.
    path = tmp_path / "researching/hitl/params.json"
    params = json.loads(path.read_text(encoding="utf-8"))
    path.write_text(json.dumps({key: params[key] for key in params if key not in REQUIRED_RULES}), encoding="utf-8")
    assert experiment_detail(tmp_path, "researching")["sessions"][1]["thresholds"] == {}


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
    ledger's own -- a sealed-gate record's slices and verdict key for key, and
    a readout as the forward record states it."""

    root = tmp_path / "experiments"
    for experiment_id, stage in (
        ("frozen", "sealed"),
        ("recorded", "frozen_with_record"),
        ("graduate", "graduated"),
        ("refused", "discarded"),
        ("no_edge", "no_deliverable"),
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
    for name in ("graduate", "refused"):
        forward = _records(root / name)[-1]
        assert details[name]["forward"]["slices"] == forward["slices"]
        assert details[name]["forward"]["verdict"] == forward["verdict"]
    recorded = _records(root / "recorded")[-1]
    assert details["recorded"]["forward"]["readout"] == recorded["readout"]
    assert rows["recorded"]["record"]["steps"] == recorded["readout"]["record"]["steps"]


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
    is ended experiments that change only when their files do. A poll that hands back
    the digest of those rows gets the live rows and the order of every id; any
    change to a kept row sends every row again."""

    root = tmp_path / "experiments"
    ended = build_experiment(root, "ended", "no_deliverable")
    build_experiment(root, "running", "research", alive=True)
    client = TestClient(create_app(tmp_path, root))
    full = client.get("/api/experiments").json()
    order = [row["experiment_id"] for row in full["experiments"]]
    assert sorted(order) == ["ended", "running"] and "order" not in full
    poll = client.get("/api/experiments", params={"kept": full["kept"]}).json()
    assert [row["experiment_id"] for row in poll["experiments"]] == ["running"]
    assert poll["order"] == order
    assert (poll["kept"], poll["ranking"], poll["running"]) == (
        full["kept"],
        full["ranking"],
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
    build_experiment(root, "ended", "no_deliverable")
    _shelve(build_experiment(root, "shelved", "sealed"))
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
    assert shelf["experiments"][0]["frozen_session"] == "research"
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
    build_experiment(root, "ended", "no_deliverable")
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
    build_experiment(root, "experiment", "graduated")
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


def test_a_fold_era_plan_is_listed_as_unreadable(tmp_path: Path) -> None:
    directory = build_experiment(tmp_path, "legacy", "research")
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
    assert rows["legacy"]["state"] == "unreadable"
    assert rows["current"]["stage"] == "research"


def _lower_bound(row: dict[str, object]) -> float:
    return row["forward"]["slices"]["forward"]["lower_bound"]  # type: ignore[index]


def test_the_ranking_lists_frozen_strategies_by_their_records_residual_ir(tmp_path: Path) -> None:
    """Only a frozen strategy with a readout is ranked, by its validation record's
    residual IR; a record the significance test could not read sorts last; nothing
    judged under the sealed gate and nothing still researching is listed."""

    for experiment_id, stage in (
        ("first", "frozen_with_record"),
        ("second", "frozen_with_record"),
        ("unread", "frozen_with_record"),
        ("graduated", "graduated"),
        ("researching", "research"),
        ("no_edge", "no_deliverable"),
    ):
        build_experiment(tmp_path, experiment_id, stage)

    def restate(name: str, **record: object) -> None:
        ledger = tmp_path / name / "ledgers/experiment_ledger.jsonl"
        rows = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines()]
        rows[-1]["readout"]["record"].update(record)
        ledger.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    restate("second", residual_information_ratio=50.0)
    restate("unread", residual_information_ratio=None, p_value=None, unmeasured="the record has no entry yet")
    rows = experiment_listing(tmp_path)["experiments"]
    ranked = ranking(rows)
    assert [row["experiment_id"] for row in ranked] == ["second", "first", "unread"]
    assert ranked[0]["record"]["residual_information_ratio"] == 50.0
    assert all(row["paper_candidate"] for row in ranked)
    readout = _records(tmp_path / "first")[-1]["readout"]
    assert ranked[1]["record"]["p_value"] == readout["record"]["p_value"]
    assert ranked[1]["forward"] == readout["forward"]
    # The home page serves the same ranking with its listing.
    listed = TestClient(create_app(tmp_path, tmp_path)).get("/api/experiments").json()
    assert listed["ranking"] == ranked


def test_an_experiment_without_a_readout_is_not_ranked(tmp_path: Path) -> None:
    """Ranking running experiments by how far they had got would crown one of a field
    that has produced no out-of-sample reading at all: with no readout the
    ranking is empty, and the homepage draws no ranking panel."""

    build_experiment(tmp_path, "early", "research", alive=True)
    build_experiment(tmp_path, "frozen", "sealed", alive=True)
    rows = experiment_listing(tmp_path)["experiments"]
    assert ranking(rows) == []
    listed = TestClient(create_app(tmp_path, tmp_path)).get("/api/experiments").json()
    assert listed["ranking"] == []
    script = (
        Path(__file__).resolve().parents[2] / "src/autotrade/webui/static/app.js"
    ).read_text(encoding="utf-8")
    assert "if (!rows.length) return null;" in script.split("function rankingPanel(", 1)[1]
    home = script.split("function homeView(", 1)[1].split("\nfunction ", 1)[0]
    assert "rankingPanel(payload)" in home and "heroPanel" not in script


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
    for name in ("accountTiles", "evidenceTiles", "rankingPanel"):
        assert not joined.search(body(name)), name
    card = body("experimentCard")
    assert "fraction: false" in card and "{ mini: true }" in card
    assert "evidenceTiles(item)" in card
    # The fraction and the limit are left out of the card's lines only.
    assert "fraction && Number.isFinite(done)" in body("activityNode")
    budget = re.sub(r"\s+", " ", body("budgetBars"))
    assert "mini ? row.spent" in budget and "mini ? `${row.label}已用 ${row.spent}`" in budget
    assert "activityNode(fresh.status)" in body("controlPanel")
    # The card's evidence is the validation record and the forward reading, never
    # a research-period figure.
    labels = ("已测步数", "记录残差 IR", "p 值", "换候选", "前推下界")
    found = [label for label in labels if f'label: "{label}"' in body("evidenceTiles")]
    assert found == list(labels)
    for retired in ("最佳候选", "DSR", "bestCandidateTiles", "research_best"):
        assert retired not in body("evidenceTiles") and retired not in body("sessionDetailPanel"), retired
    # The gloss says what the figure is; the gate's own limits and measured
    # values stay in the checklist, which reads both from the record.
    [dsr_gloss] = [row for row in script.splitlines() if row.startswith("const DSR_TITLE")]
    assert "≥" not in dsr_gloss, dsr_gloss
    assert "全区间验证次数" in script.split("const REASON_LABELS", 1)[1]


def test_a_figure_whose_label_does_not_read_itself_is_glossed_once() -> None:
    """IR is an acronym over a formula, 主动超额 and 主动跟踪误差 are statistics
    of a series the host builds, and the 下界 is a bootstrap percentile: none
    of the four can be read off its own label. IR sat bare on all four of its sites while the
    DSR beside it was fully explained. Each gloss is one shared constant, so no
    site can drift, and it hangs on the same `title` hover the console uses
    everywhere else — including on a checklist row, which had no slot for one.
    """

    script = (
        Path(__file__).resolve().parents[2] / "src/autotrade/webui/static/app.js"
    ).read_text(encoding="utf-8")
    for name in ("IR_TITLE", "EXCESS_TITLE", "TRACKING_ERROR_TITLE", "LOWER_BOUND_TITLE"):
        assert script.count(f"const {name} = ") == 1, name
    for opening, glosses in (
        ("function evidenceTiles(", ("IR_TITLE",)),
        ("function evidenceTiles(", ("RECORD_IR_TITLE", "P_VALUE_TITLE", "LOWER_BOUND_TITLE")),
        ("function researchSessionPanel(", ("IR_TITLE", "EXCESS_TITLE")),
        ("function frozenPanel(", ("IR_TITLE", "EXCESS_TITLE", "TRACKING_ERROR_TITLE")),
        ("function freezeGateChecklist(", ("DSR_TITLE",)),
    ):
        body = script.split(opening, 1)[1].split("\nfunction ", 1)[0]
        for gloss in glosses:
            assert gloss in body, (opening, gloss)
    assert "item.title" in script.split("function checklist(", 1)[1]


def test_an_experiment_card_wears_the_ending_as_a_badge_alone() -> None:
    """The homepage grid stays scannable: a card says how the experiment ended in one
    badge and keeps the reason in that badge's tooltip. The sentence itself is
    drawn only on the experiment page — its header and the 裁决 view."""

    script = (
        Path(__file__).resolve().parents[2] / "src/autotrade/webui/static/app.js"
    ).read_text(encoding="utf-8")
    card = script.split("function experimentCard(", 1)[1].split("\nfunction ", 1)[0]
    badge = script.split("function endingBadge(", 1)[1].split("\nfunction ", 1)[0]
    assert "experimentBadge(item)" in card
    assert "endingReason" not in card
    assert "title: ending.reason || null" in badge
    # The definition and its two call sites, and no third page.
    assert script.count("endingReason(") == 3


def test_the_three_replay_stage_views_judge_nothing() -> None:
    """前推回测, Held-out and 读出 judge nothing of their own: the two slice
    views say their readings are reported, never judged, and the readout view
    draws the record's significance test, the forward reading and every experiment's record
    over the same validation window; nothing of the sealed gate remains."""

    script = (
        Path(__file__).resolve().parents[2] / "src/autotrade/webui/static/app.js"
    ).read_text(encoding="utf-8")
    bodies = {
        name: script.split(f"function {name}(", 1)[1].split("\nfunction ", 1)[0]
        for name in ("forwardStagePanel", "heldoutStagePanel", "readoutStagePanel")
    }
    readout = bodies["readoutStagePanel"]
    assert "readoutBlock(readout)" in readout and "validationWindowBlock(detail.readouts)" in readout
    assert "forwardPoolBlock(detail.readouts)" in readout and "paperHandoff(detail)" in readout
    for name in ("forwardStagePanel", "heldoutStagePanel"):
        assert "只报告，不作判定" in bodies[name]
    for retired in ("sealedCriterion", "validityCriteria", "sealedReadingBlock", "sealedOnly", "verdictStagePanel"):
        assert retired not in script, retired
    assert "chip" not in readout and "thresholdChips" not in script


def test_the_listing_carries_the_freeze_and_the_record_so_far(tmp_path: Path) -> None:
    """The experiment card draws its freeze step and its evidence from the
    listing alone, and must read the same numbers the experiment page does."""

    build_experiment(tmp_path, "researching", "research")
    build_experiment(tmp_path, "frozen", "sealed")
    build_experiment(tmp_path, "recorded", "frozen_with_record")
    build_experiment(tmp_path, "fresh", "created")
    rows = {row["experiment_id"]: row for row in experiment_listing(tmp_path)["experiments"]}
    assert rows["researching"]["frozen_session"] is None
    assert rows["frozen"]["frozen_session"] == "research"
    # An experiment without a recorded step has no record to read.
    assert rows["fresh"]["record"] is None and rows["researching"]["record"] is None
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
                **DEFAULT_RESEARCH_GEOMETRY.to_record(), **CREATE_RULES,
                **patch,
            }
        },
    )
    assert response.status_code == 400
    assert message in response.json()["detail"]
    assert not (tmp_path / "experiments/geometry").exists()


def test_the_step_tree_names_the_session_and_the_frozen_node(tmp_path: Path) -> None:
    directory = build_experiment(tmp_path, "experiment", "sealed")
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


def test_a_failed_forward_replay_attempt_is_counted(tmp_path: Path) -> None:
    """The verdict view's replay failures are the forward replay's
    ``attempt_failed`` rows, which the pipeline writes under its stage name
    (``forward``), not under the provider phase its replay reads (``heldout``):
    filtering on the latter counted none, ever."""

    directory = build_experiment(tmp_path, "experiment", "graduated")
    assert experiment_detail(tmp_path, "experiment")["replay_attempts"] == {
        "failed": 0,
        "last_error": None,
    }
    ledger = ExperimentLedger(directory / "ledgers/experiment_ledger.jsonl")
    for stage in ("research", "forward"):
        # The attempt keys ``RollingExperimentPipeline`` writes on its failure path.
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


def test_failed_research_validations_are_counted_apart_from_the_replay(tmp_path: Path) -> None:
    """A research Validation that fails is no attempt failure: it leaves its
    dead-end node in the step tree and no ledger row, so the replay's attempt
    counter read 0 for an experiment whose four Validations had failed. The detail
    counts them on their own -- how many, how many the environment failed and
    refunded, and the last reason -- and leaves the replay counter alone."""

    directory = build_experiment(tmp_path, "experiment", "graduated")
    assert experiment_detail(tmp_path, "experiment")["research_validation_failures"] == {
        "failed": 0,
        "environment": 0,
        "last_error": None,
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
    detail = experiment_detail(tmp_path, "experiment")
    assert detail["research_validation_failures"] == {
        "failed": 2,
        "environment": 1,
        "last_error": "daily Validation failed: CUDA out of memory: the device was taken",
    }
    assert detail["replay_attempts"] == {"failed": 0, "last_error": None}
    # The step-tree view marks the same nodes failed.
    nodes = step_tree_view(directory)["nodes"]
    assert sum(node["status"] == "failed" for node in nodes) == 2


def test_each_validation_shows_its_raw_excess_beside_its_active_figures(tmp_path: Path) -> None:
    build_experiment(tmp_path, "experiment", "graduated")
    detail = experiment_detail(tmp_path, "experiment")
    record = next(entry for entry in detail["sessions"] if entry["kind"] == "research")["record"]
    assert {"excess_return", "plain_excess", "excess_at_cost_stress"} <= set(record["validations"][0])


def test_a_forward_record_without_the_holders_readings_shows_none(tmp_path: Path) -> None:
    """A forward record written before slices carried the holder's readings
    is shown as written: no reading is derived again from its replay, and
    the graduate's ending line names the figures it does hold. Its curve
    still draws the replay's random baseline beside the book, on the book's own days."""

    from autotrade.environment.replay.style import STYLE_ARTIFACT_NAME

    directory = build_experiment(tmp_path, "experiment", "graduated")
    ledger_path = directory / "ledgers/experiment_ledger.jsonl"
    records = [json.loads(line) for line in ledger_path.read_text(encoding="utf-8").splitlines()]
    record = records[-1]
    for block in record["slices"].values():
        del block["raw_readings"], block["plain_excess"]
    del record["verdict"]["holder_line"]
    ledger_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records), encoding="utf-8"
    )
    detail = experiment_detail(tmp_path, "experiment")
    assert detail["forward"]["slices"] == record["slices"]
    ending = summarize_experiment(directory)["ending"]["reason"]
    information_ratio = record["slices"]["forward"]["information_ratio"]
    assert ending == f"前推 账户 — · 基准 — · 对面板 — · 主动 IR {information_ratio:.2f}"
    sidecar = json.loads(
        (Path(str(record["result_ref"])).parent / STYLE_ARTIFACT_NAME).read_text(encoding="utf-8")
    )
    client = TestClient(create_app(tmp_path, tmp_path))
    equity = client.get(f"/api/experiments/experiment/results/{detail['forward']['result']}/equity").json()
    assert equity["baseline"]["dates"] == equity["series"][0]["dates"]
    assert equity["baseline"]["final"] == pytest.approx(
        float(np.prod([1.0 + value for _day, value in sidecar["panel_daily"]]) - 1.0), abs=1e-6
    )


def test_a_record_read_on_the_retired_regressed_scale_shows_no_regressed_figure(tmp_path: Path) -> None:
    """An experiment judged before 2026-10-28 stored its step rows, gate, frozen
    block and slices on the regressed scale and, when F8 refused it, a reason
    no condition carries now. No such figure reaches a current label: step
    rows and the frozen block are read off their own sidecars, as the gate
    reads them; a slice keeps its plain readings, its F8 bound stands in F2's
    place and F8's refusal is F2's."""

    directory = build_experiment(tmp_path, "experiment", "graduated", seeded=True)
    current = experiment_detail(tmp_path, "experiment")
    ledger_path = directory / "ledgers/experiment_ledger.jsonl"
    records = [json.loads(line) for line in ledger_path.read_text(encoding="utf-8").splitlines()]
    research = next(row for row in records if row["record_type"] == "research_session")
    for row in research["steps"]:
        row["neutralized"] = {**row.pop("graded"), "information_ratio": 9.0}
    gate = research["freeze_gate"]
    gate.update(neutralized_excess=0.5, information_ratio=9.0)
    gate["seed_replicates"]["mean_information_ratio"] = 9.0
    frozen = research["frozen"]
    frozen.update(neutralized_excess=frozen.pop("plain_excess"), information_ratio=9.0, tracking_error=0.9)
    forward = records[-1]
    block = forward["slices"]["forward"]
    block.update(
        neutralized_excess=0.5,
        lower_bound=0.08,
        plain_excess_lower_bound=-0.05,
        information_ratio=9.0,
        excess_at_cost_stress=0.4,
    )
    forward["verdict"].update(status="discarded", reasons=["forward_plain_excess_lower_bound_not_positive"])
    ledger_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records), encoding="utf-8"
    )

    detail = experiment_detail(tmp_path, "experiment")
    shown = detail["forward"]["slices"]["forward"]
    assert shown["lower_bound"] == -0.05 and shown["plain_excess"] == block["plain_excess"]
    assert not {"neutralized_excess", "information_ratio", "excess_at_cost_stress", "plain_excess_lower_bound"} & set(shown)
    # The verdict's reasons are shown as recorded.
    assert detail["forward"]["verdict"]["reasons"] == ["forward_plain_excess_lower_bound_not_positive"]
    # A retired token has no code: the ending names it as recorded.
    assert summarize_experiment(directory)["ending"] == {
        "state": "rejected",
        "reason": "forward_plain_excess_lower_bound_not_positive",
    }
    figures = ("plain_excess", "tracking_error", "information_ratio")
    assert {key: detail["frozen"][key] for key in figures} == {key: current["frozen"][key] for key in figures}
    assert detail["frozen"]["forward_mde"] is None
    assert detail["frozen"]["seed_replicates"]["mean_information_ratio"] is None
    record, recorded = detail["sessions"][0]["record"], current["sessions"][0]["record"]
    assert record["validations"] == recorded["validations"]
    assert record["freeze_gate"]["information_ratio"] is None
    assert record["freeze_gate"]["seed_replicates"]["mean_information_ratio"] is None


def test_an_experiment_carrying_the_retired_switches_reads_as_one_without_them(tmp_path: Path) -> None:
    """The console accepts the switches every older params.json carries, and
    the ablation rule of every experiment created from 2026-11-03 until its
    retirement, and reads nothing from them: the replay plan, the listing row
    and the page are those of the same experiment without them. A key no reader
    knows still makes the experiment unreadable."""

    def plan(experiment_id: str) -> dict[str, object]:
        return experiment_detail(tmp_path, experiment_id)["sessions"][1]["thresholds"]

    def written(directory: Path, extra: dict[str, object]) -> None:
        path = directory / "hitl/params.json"
        params = json.loads(path.read_text(encoding="utf-8"))
        path.write_text(json.dumps({**params, **extra}), encoding="utf-8")

    build_experiment(tmp_path, "current", "research")
    written(
        build_experiment(tmp_path, "legacy", "research"),
        {**dict.fromkeys(RETIRED_SWITCHES, False), "ablation_required": True},
    )
    rows = {row["experiment_id"]: row for row in experiment_listing(tmp_path)["experiments"]}
    assert rows["legacy"].get("state") != "unreadable"
    assert plan("legacy") == plan("current")
    assert not (RETIRED_SWITCHES | OBSOLETE_KEYS) & set(plan("legacy"))

    written(build_experiment(tmp_path, "unknown", "research"), {"require_heldout_magic": True})
    unknown = {row["experiment_id"]: row for row in experiment_listing(tmp_path)["experiments"]}["unknown"]
    assert unknown["state"] == "unreadable"


def test_every_record_of_a_validation_window_is_listed_with_its_p_value_and_count(tmp_path: Path) -> None:
    """The multiplicity readout: an experiment's readout view lists every frozen
    strategy's validation record over the same validation period, archived experiments
    included, with its one-sided null p-value and how many there are, and
    pools the forward-year readings of every frozen strategy over the same
    forward months. Nothing is adjusted: each experiment keeps its own reading."""

    from autotrade.pipelines.verdict import pooled_reading

    build_experiment(tmp_path, "good", "frozen_with_record")
    build_experiment(tmp_path, "other", "frozen_with_record")
    build_experiment(tmp_path, "legacy", "graduated")
    build_experiment(tmp_path, "early", "research")
    readouts = experiment_detail(tmp_path, "good")["readouts"]
    window = readouts["validation_window"]
    assert window["count"] == 2 and readouts["unreadable_experiments"] == 0
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
    crashed = build_experiment(tmp_path, "crashed", "frozen_with_record")
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
    # An experiment without a forward record, or one judged under the sealed gate, has no readout.
    assert experiment_detail(tmp_path, "early")["readouts"] is None
    assert experiment_detail(tmp_path, "legacy")["readouts"] is None


