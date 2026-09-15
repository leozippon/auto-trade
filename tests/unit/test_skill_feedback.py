"""`skill_feedback`: the parent sessions' negative channel on mounted skills.

The contract under test is what the channel it replaces did not have. Only a
mounted entry can be named, and an unknown name is refused with the mounted set
rather than silently filed. Only ``outdated`` and ``wrong`` exist — there is no
way to file a positive verdict, which is what turned 651 calls of the old tool
into 522 useless lines. Evidence has a floor, so a bare verdict cannot be
recorded at all, and one entry takes one report per run.

The rest is the shape ``report_issue`` already established and this channel now
shares: one redacted, version-stamped line in the owning experiment's ledger,
answered by a resolution line the researcher appends from the shell, joined and
projected for the console and never read back by any session.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from autotrade.environment.identity import AgentRefStore
from autotrade.environment.runtime import RunManifest, append_versioned_jsonl
from autotrade.environment.tools.base import ToolError
from autotrade.environment.tools.skill_feedback import (
    MAX_SKILL_EVIDENCE_CHARS,
    MIN_SKILL_EVIDENCE_CHARS,
    SKILL_CLAIMS,
    SKILL_FEEDBACK_SCHEMA_VERSION,
    SkillFeedbackTool,
    append_skill_feedback_resolution,
    read_skill_feedback,
    skill_feedback_path,
)
from autotrade.pipelines.skills import MemorySource, mounted_skill_refs
from autotrade.webui import skill_feedback as projection
from autotrade.webui.server import create_app

SESSION_KEY = "research"
CURATED = "curated/grid-plateau-selection"
GRADUATED = "momentum_20260101/same-window-parent-control"
RESOLVE_SCRIPT = (
    Path(__file__).resolve().parents[2] / "scripts/experiments/resolve_skill_feedback.py"
)
# A real report: what was tried, what the reading was, what it contradicts.
EVIDENCE = (
    "条目要求单参数网格不超过四点；本会话把 5 点网格一次提交 batch_validate（node_7f3a），"
    "replay-year 与单日耗时都在限额内，四点上限来自已废弃的 daily_backtest。"
)


def _mounted() -> dict[str, str]:
    """The mount map exactly as the session builds it from its own sources."""

    return mounted_skill_refs(
        (
            MemorySource(
                "curated", "curated", Path("memory/curated"), ("grid-plateau-selection",)
            ),
            MemorySource(
                "momentum_20260101",
                "graduated",
                Path("memory/momentum_20260101"),
                ("same-window-parent-control",),
            ),
        )
    )


def _session(
    tmp_path: Path, *, mounted: dict[str, str] | None = None
) -> tuple[SkillFeedbackTool, RunManifest, Path]:
    experiment_dir = tmp_path / "experiments" / "expA"
    experiment_dir.mkdir(parents=True)
    root = tmp_path / "session"
    root.mkdir()
    manifest = RunManifest.create(
        root / "artifacts" / "run_manifest.json",
        {
            "experiment_id": "expA",
            "run_id": "run_a1",
            "session_key": SESSION_KEY,
            "kind": "research",
        },
        ref_store=AgentRefStore(root),
    )
    tool = SkillFeedbackTool(
        skill_feedback_path(experiment_dir),
        manifest,
        _mounted() if mounted is None else mounted,
    )
    return tool, manifest, experiment_dir


def _file(tool: SkillFeedbackTool, **overrides: str):
    return tool.invoke(
        {"skill": CURATED, "claim": "outdated", "evidence": EVIDENCE, **overrides}
    )


# ---- filing a report -------------------------------------------------------


def test_the_example_evidence_clears_the_floor_this_channel_is_built_on() -> None:
    """The floor is the redesign: a verdict without a reproduction cannot be filed."""

    assert MIN_SKILL_EVIDENCE_CHARS <= len(EVIDENCE) <= MAX_SKILL_EVIDENCE_CHARS
    example = SkillFeedbackTool.spec.example
    assert MIN_SKILL_EVIDENCE_CHARS <= len(str(example["evidence"]))
    assert example["skill"] in _mounted() or "/" in str(example["skill"])


def test_a_report_is_one_redacted_line_carrying_the_entry_and_its_origin(
    tmp_path: Path,
) -> None:
    tool, manifest, experiment_dir = _session(tmp_path)
    before = json.dumps(manifest.data, sort_keys=True, default=str)
    result = _file(tool, evidence=EVIDENCE + " 复现时 curl 用了 sk-abcdef1234567890。")
    assert result.ok
    assert str(result.value["report_id"]).startswith("skillfb_")
    assert result.value["skill"] == CURATED and result.value["claim"] == "outdated"
    [record] = read_skill_feedback(skill_feedback_path(experiment_dir))
    assert record["schema_version"] == SKILL_FEEDBACK_SCHEMA_VERSION
    assert record["recorded_at"] == result.value["recorded_at"]
    assert record["skill"] == CURATED
    assert record["origin"] == "curated"
    assert record["claim"] == "outdated"
    for key, value in {
        "experiment_id": "expA",
        "run_id": "run_a1",
        "session_key": SESSION_KEY,
        "kind": "research",
    }.items():
        assert record[key] == value
    # The same redaction path as every trace/ledger write.
    assert "sk-abcdef1234567890" not in json.dumps(record)
    assert "sk-[redacted]" in str(record["evidence"])
    # Pure telemetry: the run manifest is untouched, so no session input surface
    # can carry the report back.
    assert json.dumps(manifest.data, sort_keys=True, default=str) == before


def test_a_graduated_entry_files_under_its_own_origin(tmp_path: Path) -> None:
    tool, _, experiment_dir = _session(tmp_path)
    _file(tool, skill=GRADUATED, claim="wrong")
    [record] = read_skill_feedback(skill_feedback_path(experiment_dir))
    assert record["skill"] == GRADUATED and record["origin"] == "graduated"
    assert record["claim"] == "wrong"


# ---- what the channel refuses ----------------------------------------------


def test_an_entry_this_session_does_not_mount_is_refused_with_the_mounted_set(
    tmp_path: Path,
) -> None:
    """The Agent can only speak about entries it can read, and the refusal says
    which those are — a name it guessed is a typo, not a finding."""

    tool, _, experiment_dir = _session(tmp_path)
    with pytest.raises(ToolError) as error:
        _file(tool, skill="curated/no-such-entry")
    assert "unknown mounted skill" in str(error.value)
    assert CURATED in str(error.value) and GRADUATED in str(error.value)
    assert error.value.error_type == "skill_feedback_policy"
    # A bare entry name is not the mounted reference either.
    with pytest.raises(ToolError):
        _file(tool, skill="grid-plateau-selection")
    assert not skill_feedback_path(experiment_dir).exists()


def test_a_session_mounting_nothing_says_so_instead_of_listing_an_empty_set(
    tmp_path: Path,
) -> None:
    tool, _, _ = _session(tmp_path, mounted={})
    with pytest.raises(ToolError, match="mounts no operating memory"):
        _file(tool)


def test_there_is_no_positive_verdict_and_no_other_claim(tmp_path: Path) -> None:
    """The removed tool's 522 confirmations are unfileable here by construction."""

    assert set(SKILL_CLAIMS) == {"outdated", "wrong"}
    assert SkillFeedbackTool.spec.input_schema["properties"]["claim"]["enum"] == [
        "outdated",
        "wrong",
    ]
    tool, _, experiment_dir = _session(tmp_path)
    for claim in ("confirmed", "useful", ""):
        with pytest.raises(ToolError, match="claim must be one of"):
            _file(tool, claim=claim)
    assert not skill_feedback_path(experiment_dir).exists()


def test_evidence_must_be_concrete_and_bounded(tmp_path: Path) -> None:
    tool, _, experiment_dir = _session(tmp_path)
    with pytest.raises(ToolError, match=f"at least {MIN_SKILL_EVIDENCE_CHARS}"):
        _file(tool, evidence="条目不对。")
    with pytest.raises(ToolError, match=f"at least {MIN_SKILL_EVIDENCE_CHARS}"):
        # Padding does not clear the floor: the value is stripped first.
        _file(tool, evidence="条目不对。" + " " * MIN_SKILL_EVIDENCE_CHARS)
    with pytest.raises(ToolError, match="exceeds"):
        _file(tool, evidence="长" * (MAX_SKILL_EVIDENCE_CHARS + 1))
    assert not skill_feedback_path(experiment_dir).exists()
    schema = SkillFeedbackTool.spec.input_schema["properties"]["evidence"]
    assert schema["minLength"] == MIN_SKILL_EVIDENCE_CHARS
    assert schema["maxLength"] == MAX_SKILL_EVIDENCE_CHARS
    assert SkillFeedbackTool.spec.mutating is False


def test_one_entry_takes_one_report_per_run(tmp_path: Path) -> None:
    """A second verdict on the same entry tells the researcher nothing new."""

    tool, _, experiment_dir = _session(tmp_path)
    _file(tool)
    with pytest.raises(ToolError, match="already filed feedback on"):
        _file(tool, claim="wrong")
    # A different entry is still open.
    _file(tool, skill=GRADUATED)
    assert len(read_skill_feedback(skill_feedback_path(experiment_dir))) == 2


def test_the_parent_alone_files_and_the_session_registry_knows_the_tool() -> None:
    from autotrade.agent.runner import _SESSION_TOOLS
    from autotrade.agent.subagent import allowed_subagent_tools

    assert SkillFeedbackTool.spec.name in _SESSION_TOOLS
    for role in ("general-purpose", "Explore"):
        assert SkillFeedbackTool.spec.name not in allowed_subagent_tools(role)


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
                    {"kind": "research", "session_key": SESSION_KEY, "index": 1}
                ],
            }
        ),
        encoding="utf-8",
    )
    return directory


def _filed(directory: Path, **overrides: object) -> dict[str, object]:
    record = {
        "report_id": "skillfb_abc",
        "experiment_id": directory.name,
        "session_key": SESSION_KEY,
        "run_id": "run_raw",
        "kind": "research",
        "skill": CURATED,
        "origin": "curated",
        "claim": "outdated",
        "evidence": EVIDENCE,
        **overrides,
    }
    return append_versioned_jsonl(
        skill_feedback_path(directory),
        record,
        schema_version=SKILL_FEEDBACK_SCHEMA_VERSION,
    )


def test_the_listing_splits_the_reference_labels_the_session_and_scrubs_paths(
    tmp_path: Path,
) -> None:
    experiments = tmp_path / "experiments"
    first = _console_experiment(experiments, "expA")
    second = _console_experiment(experiments, "expB")
    _filed(
        first,
        evidence=EVIDENCE + " 见 /Data2/lzp/ADMCubeQuant/experiments/expA/artifacts。",
    )
    _filed(
        second,
        report_id="skillfb_b",
        skill=GRADUATED,
        origin="graduated",
        claim="wrong",
        session_key="s9",
    )
    payload = projection.skill_feedback(experiments)
    assert payload["total"] == 2 and payload["unreadable"] == []
    stamps = [str(item["recorded_at"]) for item in payload["reports"]]
    assert stamps == sorted(stamps, reverse=True)
    rows = {str(item["experiment_id"]): item for item in payload["reports"]}
    labelled = rows["expA"]
    # The stored reference is one fact; the page gets the mount's own split so
    # a curated row can open the entry it names.
    assert labelled["skill"] == CURATED
    assert labelled["source"] == "curated"
    assert labelled["name"] == "grid-plateau-selection"
    assert labelled["origin"] == "curated"
    assert labelled["session_label"] == SESSION_KEY
    assert "/Data2" not in str(labelled["evidence"])
    assert "[host path omitted]" in str(labelled["evidence"])
    assert rows["expB"]["claim"] == "wrong"
    assert rows["expB"]["source"] == "momentum_20260101"
    assert rows["expB"]["origin"] == "graduated"
    # A session key the plan no longer names still lists, just unlabelled.
    assert rows["expB"]["session_label"] == ""
    only = projection.skill_feedback(experiments, experiment_id="expA")
    assert [item["experiment_id"] for item in only["reports"]] == ["expA"]
    capped = projection.skill_feedback(experiments, limit=1)
    assert len(capped["reports"]) == 1 and capped["total"] == 2
    with pytest.raises(ValueError):
        projection.skill_feedback(experiments, limit=0)
    with pytest.raises(ValueError):
        projection.skill_feedback(
            experiments, limit=projection.MAX_SKILL_FEEDBACK_PAGE + 1
        )


def test_an_unreadable_log_is_named_not_dropped(tmp_path: Path) -> None:
    experiments = tmp_path / "experiments"
    good = _console_experiment(experiments, "expA")
    broken = _console_experiment(experiments, "expB")
    corrupt = _console_experiment(experiments, "expC")
    _filed(good)
    skill_feedback_path(broken).parent.mkdir(parents=True, exist_ok=True)
    skill_feedback_path(broken).write_text("not json\n", encoding="utf-8")
    # A line that passed the version gate but carries a claim no code can label
    # is corruption too, and is named rather than guessed at.
    _filed(corrupt, claim="confirmed")
    payload = projection.skill_feedback(experiments)
    assert [item["experiment_id"] for item in payload["reports"]] == ["expA"]
    named = {str(item["experiment_id"]) for item in payload["unreadable"]}
    assert named == {"expB", "expC"}
    assert "unreadable" in str(payload["unreadable"][0]["error"])
    assert str(tmp_path) not in str(payload["unreadable"][0]["error"])


def test_resolved_reports_leave_the_default_page_and_come_back_on_request(
    tmp_path: Path,
) -> None:
    experiments = tmp_path / "experiments"
    directory = _console_experiment(experiments, "expA")
    _filed(directory, report_id="skillfb_open")
    _filed(directory, report_id="skillfb_done", skill=GRADUATED, origin="graduated")
    append_skill_feedback_resolution(
        skill_feedback_path(directory),
        report_id="skillfb_done",
        outcome="skill_updated",
        note="commit 10e541f：条目改写为按工具上限说明网格点数。",
    )
    default = projection.skill_feedback(experiments)
    assert [item["report_id"] for item in default["reports"]] == ["skillfb_open"]
    assert default["total"] == 1 and default["resolved"] == 1
    full = projection.skill_feedback(experiments, include_resolved=True)
    assert full["total"] == 2 and full["resolved"] == 1
    done = next(item for item in full["reports"] if item["report_id"] == "skillfb_done")
    assert done["outcome"] == "skill_updated"
    assert str(done["resolution"]).startswith("commit 10e541f")
    assert done["resolved_at"]


def test_the_endpoint_serves_the_listing_read_only(tmp_path: Path) -> None:
    experiments = tmp_path / "experiments"
    directory = _console_experiment(experiments, "expA")
    _filed(directory)
    client = TestClient(create_app(tmp_path, experiments))
    payload = client.get("/api/skill-feedback").json()
    assert payload["total"] == 1 and payload["resolved"] == 0
    assert payload["limit"] == projection.MAX_SKILL_FEEDBACK_PAGE
    report = payload["reports"][0]
    assert report["experiment_id"] == "expA"
    assert report["claim"] == "outdated"
    assert report["skill"] == CURATED and report["name"] == "grid-plateau-selection"
    assert report["origin"] == "curated" and report["evidence"]
    # The raw run id never crosses the HTTP boundary.
    assert "run_raw" not in json.dumps(payload, ensure_ascii=False)
    assert client.get("/api/skill-feedback", params={"limit": 0}).status_code == 422
    assert client.get(
        "/api/skill-feedback", params={"experiment_id": "nope"}
    ).status_code == 404
    # The console reads; only the shell writes.
    assert client.post("/api/skill-feedback", json={}).status_code in (404, 405)


# ---- the researcher's answer -----------------------------------------------


def test_the_cli_records_a_resolution_and_refuses_the_rest(tmp_path: Path) -> None:
    experiments = tmp_path / "experiments"
    directory = _console_experiment(experiments, "expA")
    _filed(directory, report_id="skillfb_cli")

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
        "--report", "skillfb_cli",
        "--outcome", "skill_updated",
        "--note", "commit 10e541f：精选库条目已改写。",
    )
    assert recorded.returncode == 0, recorded.stderr
    assert json.loads(recorded.stdout)["outcome"] == "skill_updated"
    [report] = read_skill_feedback(skill_feedback_path(directory))
    assert report["outcome"] == "skill_updated"
    assert str(report["resolution"]).startswith("commit 10e541f")

    again = resolve(
        "--experiment", "expA", "--report", "skillfb_cli",
        "--outcome", "not_a_defect", "--note", "重复一次。",
    )
    assert again.returncode == 2 and "already resolved" in again.stderr
    ghost = resolve(
        "--experiment", "expA", "--report", "skillfb_nope",
        "--outcome", "not_a_defect", "--note", "不存在的报告。",
    )
    assert ghost.returncode == 2 and "unknown skill feedback" in ghost.stderr
    elsewhere = resolve(
        "--experiment", "expZ", "--report", "skillfb_cli",
        "--outcome", "not_a_defect", "--note", "不存在的实验。",
    )
    assert elsewhere.returncode == 2 and "unknown experiment" in elsewhere.stderr
    # Only the one accepted resolution reached the log.
    text = skill_feedback_path(directory).read_text(encoding="utf-8")
    assert len(text.splitlines()) == 2
