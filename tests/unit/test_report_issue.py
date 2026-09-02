"""`report_issue`: the parent sessions' defect channel to the operators.

The contract under test: a report is validated, capped, redacted, and appended
to the owning experiment's ``ledgers/issue_reports.jsonl`` — and nothing else
changes. It never touches the run manifest or any artifact, it never reaches a
mounted workspace, and the console lists it through the same public projection
as every other Agent-authored string.

The answer travels the same log: the researcher's ``resolve_issue.py`` appends
a resolution line, the report itself is never rewritten, and the console joins
the two and hides what is settled.
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from autotrade.environment.identity import AgentRefStore
from autotrade.environment.runtime import RunManifest
from autotrade.environment.tools.base import ToolError, ToolRegistry
from autotrade.environment.tools.report_issue import (
    ISSUE_CATEGORIES,
    ISSUE_REPORT_SCHEMA_VERSION,
    ISSUE_REPORTS_NAME,
    MAX_ISSUE_EVIDENCE_CHARS,
    MAX_ISSUE_REPORTS_PER_SESSION,
    MAX_ISSUE_RESOLUTION_NOTE_CHARS,
    MAX_ISSUE_SUMMARY_CHARS,
    ReportIssueTool,
    append_issue_report,
    append_issue_resolution,
    issue_reports_path,
    read_issue_reports,
)
from autotrade.webui import issues
from autotrade.webui.server import create_app

from .test_operating_memory import (
    _experiment_with_skill,
    _repo_with_library,
    _workspace,
)

SESSION_KEY = "epoch_001/fold_2022Q1"


def _session(tmp_path: Path) -> tuple[ReportIssueTool, RunManifest, Path]:
    """One parent session and the experiment log its reports land in."""

    experiment_dir = tmp_path / "experiments" / "expA"
    experiment_dir.mkdir(parents=True)
    root = tmp_path / "session"
    root.mkdir()
    manifest = RunManifest.create(
        root / "artifacts" / "run_manifest.json",
        {
            "experiment_id": "expA",
            "epoch_id": "epoch_001",
            "fold_id": "fold_2022Q1",
            "run_id": "run_a1",
            "session_key": SESSION_KEY,
            "kind": "fold",
        },
        ref_store=AgentRefStore(root),
    )
    return ReportIssueTool(issue_reports_path(experiment_dir), manifest), manifest, experiment_dir


def _report(tool: ReportIssueTool, **overrides: str):
    arguments = {
        "category": "tool_output",
        "summary": "验收脚本把 classic diff 的 <;> 前缀当成 unified diff 统计。",
        "evidence": "对同一对文件分别跑两种 diff，统计行数相差一倍；见 workspace/notes/diff.md。",
        **overrides,
    }
    return tool.invoke(arguments)


# ---- filing a report -------------------------------------------------------


def test_a_report_is_one_redacted_line_with_the_session_link_keys(
    tmp_path: Path,
) -> None:
    tool, manifest, experiment_dir = _session(tmp_path)
    before = json.dumps(manifest.data, sort_keys=True, default=str)
    result = _report(tool, evidence="curl 带上了 sk-abcdef1234567890 才复现；预期不需要凭据。")
    assert result.ok
    assert str(result.value["report_id"]).startswith("issue_")
    assert result.value["reports_this_session"] == 1
    assert result.value["max_reports_per_session"] == MAX_ISSUE_REPORTS_PER_SESSION
    records = read_issue_reports(issue_reports_path(experiment_dir))
    assert len(records) == 1
    record = records[0]
    assert record["schema_version"] == ISSUE_REPORT_SCHEMA_VERSION
    assert record["recorded_at"] == result.value["recorded_at"]
    assert record["category"] == "tool_output"
    for key, value in {
        "experiment_id": "expA",
        "epoch_id": "epoch_001",
        "fold_id": "fold_2022Q1",
        "run_id": "run_a1",
        "session_key": SESSION_KEY,
        "kind": "fold",
    }.items():
        assert record[key] == value
    # The same redaction path as every trace/ledger write.
    assert "sk-abcdef1234567890" not in json.dumps(record)
    assert "sk-[redacted]" in str(record["evidence"])
    # Pure telemetry: the run manifest (Agent-visible and host copy alike)
    # is untouched, so no session input surface can carry the report back.
    assert json.dumps(manifest.data, sort_keys=True, default=str) == before
    for path in (manifest.path, manifest.host_path):
        assert "diff" not in path.read_text(encoding="utf-8")


def test_reports_append_in_order_and_survive_reader_round_trips(
    tmp_path: Path,
) -> None:
    tool, _manifest, experiment_dir = _session(tmp_path)
    first = _report(tool, summary="第一条。")
    second = _report(tool, summary="第二条。", category="environment")
    assert first.ok and second.ok
    records = read_issue_reports(issue_reports_path(experiment_dir))
    assert [item["summary"] for item in records] == ["第一条。", "第二条。"]
    assert [item["category"] for item in records] == ["tool_output", "environment"]


@pytest.mark.parametrize(
    "overrides",
    [
        {"category": "complaint"},
        {"summary": "   "},
        {"summary": "长" * (MAX_ISSUE_SUMMARY_CHARS + 1)},
        {"evidence": "   "},
        {"evidence": "长" * (MAX_ISSUE_EVIDENCE_CHARS + 1)},
    ],
)
def test_an_invalid_report_is_refused_and_nothing_is_written(
    tmp_path: Path, overrides: dict[str, str]
) -> None:
    tool, _manifest, experiment_dir = _session(tmp_path)
    with pytest.raises(ToolError):
        _report(tool, **overrides)
    assert not issue_reports_path(experiment_dir).exists()


def test_the_session_cap_refuses_the_seventeenth_report_explicitly(
    tmp_path: Path,
) -> None:
    tool, _manifest, experiment_dir = _session(tmp_path)
    for index in range(MAX_ISSUE_REPORTS_PER_SESSION):
        assert _report(tool, summary=f"第 {index} 条。").ok
    with pytest.raises(ToolError, match="cap"):
        _report(tool, summary="多余的一条。")
    assert len(read_issue_reports(issue_reports_path(experiment_dir))) == (
        MAX_ISSUE_REPORTS_PER_SESSION
    )


def test_concurrent_calls_never_exceed_the_cap(tmp_path: Path) -> None:
    """The spec is non-mutating, so one turn may dispatch calls in parallel."""

    tool, _manifest, experiment_dir = _session(tmp_path)
    refused: list[ToolError] = []

    def file_five(offset: int) -> None:
        for index in range(5):
            try:
                _report(tool, summary=f"并发第 {offset}-{index} 条。")
            except ToolError as exc:
                refused.append(exc)

    threads = [threading.Thread(target=file_five, args=(n,)) for n in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(refused) == 20 - MAX_ISSUE_REPORTS_PER_SESSION
    assert len(read_issue_reports(issue_reports_path(experiment_dir))) == (
        MAX_ISSUE_REPORTS_PER_SESSION
    )


def test_the_schema_is_the_contract_the_model_reads(tmp_path: Path) -> None:
    schema = ReportIssueTool.spec.input_schema["properties"]
    assert schema["category"]["enum"] == list(ISSUE_CATEGORIES)
    assert schema["summary"]["maxLength"] == MAX_ISSUE_SUMMARY_CHARS
    assert schema["evidence"]["maxLength"] == MAX_ISSUE_EVIDENCE_CHARS
    assert ReportIssueTool.spec.mutating is False
    # Dispatch through the registry answers a bad shape with the example call.
    tool, _manifest, _experiment_dir = _session(tmp_path)
    result = ToolRegistry([tool]).invoke(
        "report_issue",
        {"category": "complaint", "summary": "x", "evidence": "y"},
    )
    assert not result.ok
    assert "correct call example" in result.error


def test_the_reader_fails_fast_on_a_foreign_schema(tmp_path: Path) -> None:
    path = tmp_path / ISSUE_REPORTS_NAME
    append_issue_report(path, {"category": "other", "summary": "s", "evidence": "e"})
    path.write_text(
        path.read_text(encoding="utf-8")
        + json.dumps({"schema_version": 2, "summary": "future"})
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="schema_version"):
        read_issue_reports(path)


# ---- never an Agent input --------------------------------------------------


def test_reports_never_reach_a_mounted_session_workspace(tmp_path: Path) -> None:
    """The graduated tier mounts other experiments' skills; the reports of an
    admitted experiment must not ride along into any session input."""

    experiments = tmp_path / "experiments"
    experiments.mkdir()
    source = _experiment_with_skill(experiments, "adopted")
    append_issue_report(
        issue_reports_path(source),
        {"category": "environment", "summary": "宿主争用导致回放超时。", "evidence": "复现步骤。"},
    )
    repo = _repo_with_library(tmp_path)
    root = tmp_path / "session"
    root.mkdir()
    workspace, mounted = _workspace(
        root, mode="curated+graduated", repo_root=repo, experiments_root=experiments
    )
    assert mounted  # the source experiment's skills did mount
    assert not list(workspace.rglob(ISSUE_REPORTS_NAME))
    for path in workspace.rglob("*"):
        if path.is_file():
            assert "宿主争用" not in path.read_text(encoding="utf-8", errors="ignore")


# ---- console projection ----------------------------------------------------


def _console_experiment(experiments_root: Path, name: str) -> Path:
    directory = experiments_root / name
    directory.mkdir(parents=True)
    AgentRefStore(directory)
    hitl = directory / "hitl"
    hitl.mkdir()
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
    return directory


def _file_report(directory: Path, **overrides: str) -> dict[str, object]:
    record = {
        "report_id": "issue_abc",
        "experiment_id": directory.name,
        "session_key": SESSION_KEY,
        "run_id": "run_raw",
        "kind": "fold",
        "category": "data",
        "summary": "单位表与实际列不一致。",
        "evidence": "读 snapshot/daily 的 close 列，单位是分而不是元。",
        **overrides,
    }
    return append_issue_report(issue_reports_path(directory), record)


def test_the_listing_projects_labels_and_scrubs_host_paths(tmp_path: Path) -> None:
    experiments = tmp_path / "experiments"
    first = _console_experiment(experiments, "expA")
    second = _console_experiment(experiments, "expB")
    _file_report(first, summary="读 /Data2/lzp/ADMCubeQuant/experiments/expA/artifacts 出错。")
    _file_report(second, category="docs", session_key="epoch_009/fold_ghost")
    payload = issues.issue_reports(experiments)
    assert payload["total"] == 2
    assert payload["unreadable"] == []
    reports = payload["reports"]
    # Newest first across experiments.
    stamps = [str(item["recorded_at"]) for item in reports]
    assert stamps == sorted(stamps, reverse=True)
    by_experiment = {str(item["experiment_id"]): item for item in reports}
    labelled = by_experiment["expA"]
    assert labelled["session_label"] == "epoch_001/2022Q1"
    assert "/Data2" not in str(labelled["summary"])
    assert "[host path omitted]" in str(labelled["summary"])
    # A session key the plan no longer names still lists, just unlabelled.
    assert by_experiment["expB"]["session_label"] == ""
    assert by_experiment["expB"]["category"] == "docs"
    # The experiment filter and the page cap.
    only = issues.issue_reports(experiments, experiment_id="expA")
    assert [item["experiment_id"] for item in only["reports"]] == ["expA"]
    capped = issues.issue_reports(experiments, limit=1)
    assert len(capped["reports"]) == 1 and capped["total"] == 2
    with pytest.raises(ValueError):
        issues.issue_reports(experiments, limit=0)


def test_an_unreadable_log_is_named_not_dropped(tmp_path: Path) -> None:
    experiments = tmp_path / "experiments"
    good = _console_experiment(experiments, "expA")
    broken = _console_experiment(experiments, "expB")
    _file_report(good)
    issue_reports_path(broken).parent.mkdir(parents=True, exist_ok=True)
    issue_reports_path(broken).write_text("not json\n", encoding="utf-8")
    payload = issues.issue_reports(experiments)
    assert [item["experiment_id"] for item in payload["reports"]] == ["expA"]
    assert [item["experiment_id"] for item in payload["unreadable"]] == ["expB"]
    assert "unreadable" in str(payload["unreadable"][0]["error"])


def test_the_endpoint_serves_the_listing_read_only(tmp_path: Path) -> None:
    experiments = tmp_path / "experiments"
    directory = _console_experiment(experiments, "expA")
    _file_report(directory)
    client = TestClient(create_app(tmp_path, experiments))
    payload = client.get("/api/issue-reports").json()
    assert payload["total"] == 1
    report = payload["reports"][0]
    assert report["experiment_id"] == "expA"
    assert report["category"] == "data"
    assert report["summary"] and report["evidence"] and report["recorded_at"]
    # The raw run id never crosses the HTTP boundary.
    assert "run_raw" not in json.dumps(payload)
    assert client.get("/api/issue-reports", params={"experiment_id": "expA"}).json()["total"] == 1
    assert client.get("/api/issue-reports", params={"experiment_id": "missing"}).status_code == 404
    assert client.get("/api/issue-reports", params={"limit": 0}).status_code == 422


# ---- answering a report ----------------------------------------------------

RESOLVE_SCRIPT = Path(__file__).resolve().parents[2] / "scripts/experiments/resolve_issue.py"


def test_a_resolution_is_a_second_line_and_the_report_stays_as_filed(
    tmp_path: Path,
) -> None:
    tool, _manifest, experiment_dir = _session(tmp_path)
    filed = _report(tool, summary="回放超时被报成候选异常。")
    path = issue_reports_path(experiment_dir)
    original = path.read_text(encoding="utf-8")
    resolution = append_issue_resolution(
        path,
        report_id=str(filed.value["report_id"]),
        outcome="fixed",
        note="commit abc1234：失败行改为报告宿主争用。",
    )
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    assert len(lines) == 2
    # Append-only: the session's own line is byte-identical afterwards.
    assert lines[0] == original
    assert json.loads(lines[1])["record_type"] == "resolution"
    assert resolution["schema_version"] == ISSUE_REPORT_SCHEMA_VERSION
    [report] = read_issue_reports(path)
    assert report["report_id"] == filed.value["report_id"]
    assert report["category"] == "tool_output"
    assert report["outcome"] == "fixed"
    assert report["resolution"] == "commit abc1234：失败行改为报告宿主争用。"
    # One stamp, the log's own: the resolution time cannot disagree with it.
    assert report["resolved_at"] == resolution["recorded_at"]


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"report_id": "issue_ghost"}, "unknown issue report"),
        ({"outcome": "wontfix"}, "outcome must be one of"),
        ({"note": "   "}, "note must say"),
        ({"note": "长" * (MAX_ISSUE_RESOLUTION_NOTE_CHARS + 1)}, "exceeds"),
    ],
)
def test_a_resolution_that_cannot_be_trusted_is_refused_and_nothing_is_written(
    tmp_path: Path, overrides: dict[str, str], match: str
) -> None:
    tool, _manifest, experiment_dir = _session(tmp_path)
    filed = _report(tool)
    path = issue_reports_path(experiment_dir)
    before = path.read_text(encoding="utf-8")
    arguments = {
        "report_id": str(filed.value["report_id"]),
        "outcome": "fixed",
        "note": "commit abc1234。",
        **overrides,
    }
    with pytest.raises(ValueError, match=match):
        append_issue_resolution(path, **arguments)
    assert path.read_text(encoding="utf-8") == before


def test_a_report_is_answered_once(tmp_path: Path) -> None:
    tool, _manifest, experiment_dir = _session(tmp_path)
    report_id = str(_report(tool).value["report_id"])
    path = issue_reports_path(experiment_dir)
    append_issue_resolution(
        path, report_id=report_id, outcome="fixed", note="commit abc1234。"
    )
    with pytest.raises(ValueError, match="already resolved as fixed"):
        append_issue_resolution(
            path, report_id=report_id, outcome="not_a_defect", note="改判。"
        )
    assert len(path.read_text(encoding="utf-8").splitlines()) == 2


def test_the_reader_fails_fast_on_a_resolution_naming_no_report(
    tmp_path: Path,
) -> None:
    """Within schema 1 an orphan resolution is corruption, not history."""

    path = tmp_path / ISSUE_REPORTS_NAME
    append_issue_report(
        path,
        {"report_id": "issue_1", "category": "other", "summary": "s", "evidence": "e"},
    )
    append_issue_report(
        path,
        {
            "record_type": "resolution",
            "report_id": "issue_ghost",
            "outcome": "fixed",
            "note": "n",
        },
    )
    with pytest.raises(ValueError, match="no filed issue report"):
        read_issue_reports(path)


def test_resolved_reports_leave_the_default_page_and_come_back_on_request(
    tmp_path: Path,
) -> None:
    experiments = tmp_path / "experiments"
    directory = _console_experiment(experiments, "expA")
    _file_report(directory, report_id="issue_open", summary="未处置的一条。")
    _file_report(directory, report_id="issue_done", summary="已处置的一条。")
    append_issue_resolution(
        issue_reports_path(directory),
        report_id="issue_done",
        outcome="not_a_defect",
        note="按设计：244 是固定年化常数，窗口长度不是年化因子。",
    )
    default = issues.issue_reports(experiments)
    assert [item["report_id"] for item in default["reports"]] == ["issue_open"]
    assert default["total"] == 1 and default["resolved"] == 1
    assert default["reports"][0]["outcome"] == ""
    full = issues.issue_reports(experiments, include_resolved=True)
    assert full["total"] == 2 and full["resolved"] == 1
    done = next(item for item in full["reports"] if item["report_id"] == "issue_done")
    assert done["outcome"] == "not_a_defect"
    assert str(done["resolution"]).startswith("按设计")
    assert done["resolved_at"]


def test_the_endpoint_serves_the_resolution_the_page_reads_and_takes_no_writes(
    tmp_path: Path,
) -> None:
    experiments = tmp_path / "experiments"
    directory = _console_experiment(experiments, "expA")
    _file_report(directory, report_id="issue_done")
    append_issue_resolution(
        issue_reports_path(directory),
        report_id="issue_done",
        outcome="accepted_limitation",
        note="合同如此，代价与影响都已记录。",
    )
    client = TestClient(create_app(tmp_path, experiments))
    default = client.get("/api/issue-reports").json()
    assert default["reports"] == [] and default["total"] == 0
    assert default["resolved"] == 1
    payload = client.get(
        "/api/issue-reports", params={"include_resolved": "true"}
    ).json()
    report = payload["reports"][0]
    assert report["outcome"] == "accepted_limitation"
    assert report["resolution"].startswith("合同如此")
    assert report["resolved_at"]
    # The console reads; only the shell writes.
    assert client.post("/api/issue-reports", json={}).status_code in (404, 405)


def test_the_cli_records_a_resolution_and_refuses_the_rest(tmp_path: Path) -> None:
    """The researcher's only entry point, run the way it is documented."""

    experiments = tmp_path / "experiments"
    directory = _console_experiment(experiments, "expA")
    _file_report(directory, report_id="issue_cli")

    def resolve(*arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                sys.executable,
                str(RESOLVE_SCRIPT),
                "--experiments-root",
                str(experiments),
                *arguments,
            ],
            capture_output=True,
            text=True,
            check=False,
        )

    recorded = resolve(
        "--experiment", "expA",
        "--report", "issue_cli",
        "--outcome", "fixed",
        "--note", "commit 10e541f：Test figure 检查改为整词匹配。",
    )
    assert recorded.returncode == 0, recorded.stderr
    assert json.loads(recorded.stdout)["outcome"] == "fixed"
    [report] = read_issue_reports(issue_reports_path(directory))
    assert report["outcome"] == "fixed"
    assert str(report["resolution"]).startswith("commit 10e541f")

    again = resolve(
        "--experiment", "expA", "--report", "issue_cli",
        "--outcome", "fixed", "--note", "重复一次。",
    )
    assert again.returncode == 2 and "already resolved" in again.stderr
    ghost = resolve(
        "--experiment", "expA", "--report", "issue_nope",
        "--outcome", "fixed", "--note", "不存在的报告。",
    )
    assert ghost.returncode == 2 and "unknown issue report" in ghost.stderr
    elsewhere = resolve(
        "--experiment", "expZ", "--report", "issue_cli",
        "--outcome", "fixed", "--note", "不存在的实验。",
    )
    assert elsewhere.returncode == 2 and "unknown experiment" in elsewhere.stderr
    # Only the one accepted resolution reached the log.
    assert len(issue_reports_path(directory).read_text(encoding="utf-8").splitlines()) == 2

