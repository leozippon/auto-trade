"""Full safe Agent Trace projection for Meta learning."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

import pytest

from autotrade.agent.prompts import META_PHASE_CONTRACT, build_meta_learning_prompt
from autotrade.environment.artifacts import FilesystemArtifactStore
from autotrade.environment.identity import agent_visible_ref
from autotrade.environment.llm import ScriptedLLM, ToolCall
from autotrade.pipelines.local_backend import LLMMetaLearner
from autotrade.pipelines.meta_inputs import (
    AGENT_TRACE_FULL_CONTENT_CHARS,
    AGENT_TRACE_FULL_RELATIVE_DIR,
    AgentTraceSourceError,
    build_meta_fold_review_bundle,
    build_meta_fold_reviews,
    compact_agent_trace,
    project_full_agent_trace,
    serialize_full_agent_trace,
    write_meta_agent_trace_sidecars,
)
from tests.unit.test_interactive_worker_local import _explore_then


def _as_map(value: object) -> dict[str, object]:
    assert isinstance(value, Mapping)
    return dict(value)


def _event_line(event: dict[str, object]) -> str:
    return json.dumps(event, ensure_ascii=False)


def _write_trace(path: Path, events: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(_event_line(event) for event in events) + "\n",
        encoding="utf-8",
    )


def _fold(
    tmp_path: Path,
    events: list[dict[str, object]] | Sequence[Mapping[str, object]] | None = None,
    *,
    fold_id: str = "fold_2024Q1",
    run_id: str = "run_fold",
    with_ref: bool = True,
) -> dict[str, object]:
    record: dict[str, object] = {
        "record_type": "fold",
        "epoch_id": "epoch_001",
        "fold_id": fold_id,
        "run_id": run_id,
        "fold_status": "frozen",
    }
    if with_ref:
        trace = tmp_path / "traces" / f"{run_id}.jsonl"
        source = [
            dict(event)
            for event in (events or [{"event_type": "session_start", "mode": "fold"}])
        ]
        _write_trace(trace, source)
        record["agent_trace_ref"] = str(trace)
    return record


def test_full_projection_keeps_all_events_in_source_order() -> None:
    events = [
        {"event_type": "session_start", "mode": "fold"},
        {"event_type": "environment_stage", "stage": "sandbox_layout"},
        {"event_type": "llm_call", "content": "plan", "status": "ok"},
        {"event_type": "unknown_lifecycle", "phase": "mid", "status": "running"},
        {"event_type": "session_end", "status": "finished", "llm_calls": 1},
    ]
    projected = project_full_agent_trace(events)
    assert [item["event_type"] for item in projected] == [
        "session_start",
        "environment_stage",
        "llm_call",
        "unknown_lifecycle",
        "session_end",
    ]
    assert projected[1]["stage"] == "sandbox_layout"
    assert projected[3]["phase"] == "mid"
    compact = compact_agent_trace(events)
    assert [item["event_type"] for item in compact] == [
        "session_start",
        "llm_call",
        "session_end",
    ]


def test_full_projection_keeps_bounded_compaction_summary() -> None:
    summary = "process summary " * 1_000
    projected = project_full_agent_trace(
        [{"event_type": "context_compaction", "summary": summary}],
        content_chars=128,
    )
    assert projected[0]["summary"] == summary[:128]


def test_full_projection_keeps_main_and_subagent_lifecycle() -> None:
    events = [
        {"event_type": "session_start", "mode": "fold"},
        {
            "event_type": "tool_call",
            "tool": "explore",
            "parent_call_id": "call_1",
            "arguments": {"role": "auditor", "task": "inspect schema"},
            "result": {"ok": True},
        },
        {
            "event_type": "explore_task",
            "task_id": "explore_abc",
            "parent_call_id": "call_1",
            "role": "auditor",
            "task": "inspect schema",
            "status": "started",
        },
        {
            "event_type": "explore_llm",
            "task_id": "explore_abc",
            "content": "look at columns",
            "status": "ok",
        },
        {
            "event_type": "explore_tool",
            "task_id": "explore_abc",
            "tool": "grep",
            "result": {"ok": True},
        },
        {
            "event_type": "explore",
            "task_id": "explore_abc",
            "parent_call_id": "call_1",
            "role": "auditor",
            "status": "completed",
            "digest": "daily has trade_date",
        },
        {"event_type": "session_end", "llm_calls": 2, "explored_roles": ["auditor"]},
    ]
    projected = project_full_agent_trace(events)
    assert [item["event_type"] for item in projected] == [
        "session_start",
        "tool_call",
        "explore_task",
        "explore_llm",
        "explore_tool",
        "explore",
        "session_end",
    ]
    assert projected[2]["role"] == "auditor"
    assert projected[2]["parent_call_id"] == "call_1"
    assert projected[2]["task"] == {"omitted": True, "chars": len("inspect schema")}
    assert _as_map(projected[1]["arguments"])["task"] == {
        "omitted": True,
        "chars": len("inspect schema"),
    }
    assert projected[3]["content"] == "look at columns"
    assert projected[5]["digest"] == "daily has trade_date"
    assert _as_map(projected[4]["result"])["ok"] is True


def test_full_projection_redacts_prompt_user_ask_user_host_secret_and_leaks() -> None:
    secret = "sk-" + "abcdefghijklmnopqrstuvwxyz"
    body = "def generate_orders(context):\n    return []\n"
    events = [
        {
            "event_type": "session_start",
            "mode": "fold",
            "system_prompt": "FULL SYSTEM PROMPT SENSITIVE",
            "instruction": "USER INSTRUCTION FULL TEXT",
        },
        {
            "event_type": "user_message",
            "interrupt": True,
            "safe_point": "after_tools",
            "content": "do not keep this inbox body /Data2/lzp/secret",
        },
        {
            "event_type": "tool_call",
            "tool": "ask_user",
            "arguments": {
                "question": "which hypothesis?",
                "summary": "fork after auditor",
            },
            "result": {"ok": True, "value": {"reply": "keep the daily floor"}},
        },
        {
            "event_type": "tool_call",
            "tool": "edit_file",
            "arguments": {
                "path": "/Data2/lzp/ADMCubeQuant/experiments/x/output/main.py",
                "old": "old source",
                "new": "new source",
                "old_text": "old source",
                "new_text": "new source",
                "content": body,
                "api_key": secret,
            },
            "result": {
                "ok": True,
                "heldout": {"total_return": 0.99},
                "per_stock": {"000001.SZ": [0.1]},
                "daily_returns": [0.01],
                "weekly_returns": [0.02],
            },
        },
        {
            "event_type": "mystery_event",
            "instruction": "nested instruction",
            "payload": {
                "text": "mystery body",
                "heldout": 1,
                "note": "failed at /home/lzp/hidden",
            },
        },
    ]
    projected = project_full_agent_trace(events)
    rendered = json.dumps(projected, ensure_ascii=False)
    assert "FULL SYSTEM PROMPT SENSITIVE" not in rendered
    assert "USER INSTRUCTION FULL TEXT" not in rendered
    assert "do not keep this inbox body" not in rendered
    assert "which hypothesis?" not in rendered
    assert "fork after auditor" not in rendered
    assert "keep the daily floor" not in rendered
    assert "old source" not in rendered
    assert "new source" not in rendered
    assert body not in rendered
    assert secret not in rendered
    assert "/Data2/" not in rendered
    assert "/home/" not in rendered
    assert "0.99" not in rendered
    assert "000001.SZ" not in rendered
    assert "system_prompt" not in projected[0]
    assert "instruction" not in projected[0]
    assert "content" not in projected[1]
    assert projected[1]["interrupt"] is True
    ask_args = _as_map(projected[2]["arguments"])
    assert ask_args["question"] == {"omitted": True, "chars": len("which hypothesis?")}
    assert ask_args["summary"] == {"omitted": True, "chars": len("fork after auditor")}
    reply = _as_map(_as_map(projected[2]["result"])["value"])["reply"]
    assert reply == {"omitted": True, "chars": len("keep the daily floor")}
    args = _as_map(projected[3]["arguments"])
    assert args["path"] == "[host_path]"
    assert args["content"] == {"omitted": True, "chars": len(body)}
    assert args["api_key"] == "[redacted]"
    result = _as_map(projected[3]["result"])
    assert "heldout" not in result
    assert "per_stock" not in result
    assert "daily_returns" not in result
    assert "weekly_returns" not in result
    mystery = _as_map(projected[4]["payload"])
    assert mystery["text"] == {"omitted": True, "chars": len("mystery body")}
    assert "heldout" not in mystery
    assert mystery["note"] == "failed at [host_path]"


def test_full_projection_caps_llm_and_explore_content() -> None:
    huge = "alpha " * 4000
    events = [
        {"event_type": "llm_call", "content": huge},
        {"event_type": "explore", "digest": huge, "content": huge},
        {"event_type": "explore_llm", "content": huge},
        {
            "event_type": "tool_call",
            "tool": "write_file",
            "arguments": {"content": huge},
        },
    ]
    projected = project_full_agent_trace(events)
    assert isinstance(projected[0]["content"], str)
    assert len(cast(str, projected[0]["content"])) == AGENT_TRACE_FULL_CONTENT_CHARS
    assert isinstance(projected[1]["digest"], str)
    assert len(cast(str, projected[1]["digest"])) == AGENT_TRACE_FULL_CONTENT_CHARS
    assert isinstance(projected[2]["content"], str)
    assert len(cast(str, projected[2]["content"])) == AGENT_TRACE_FULL_CONTENT_CHARS
    assert _as_map(projected[3]["arguments"])["content"] == {
        "omitted": True,
        "chars": len(huge),
    }


def test_missing_trace_ref_is_unavailable(tmp_path: Path) -> None:
    reviews, sidecars = build_meta_fold_review_bundle([_fold(tmp_path, with_ref=False)])
    assert reviews[0]["agent_trace"] == []
    full = _as_map(reviews[0]["agent_trace_full"])
    assert full["available"] is False
    assert full["path"] is None
    assert sidecars[0].available is False
    assert sidecars[0].payload is None


def test_missing_or_corrupt_trace_source_fails(tmp_path: Path) -> None:
    missing = {
        "record_type": "fold",
        "fold_id": "fold_2024Q1",
        "fold_status": "frozen",
        "agent_trace_ref": str(tmp_path / "absent.jsonl"),
    }
    with pytest.raises(AgentTraceSourceError, match="missing"):
        build_meta_fold_reviews([missing])
    bad_json = tmp_path / "bad.jsonl"
    bad_json.write_text("{not-json\n", encoding="utf-8")
    with pytest.raises(AgentTraceSourceError, match="invalid JSON"):
        build_meta_fold_reviews(
            [
                {
                    "record_type": "fold",
                    "fold_id": "fold_2024Q1",
                    "fold_status": "frozen",
                    "agent_trace_ref": str(bad_json),
                }
            ]
        )
    not_object = tmp_path / "list.jsonl"
    not_object.write_text("[1, 2]\n", encoding="utf-8")
    with pytest.raises(AgentTraceSourceError, match="not a JSON object"):
        build_meta_fold_reviews(
            [
                {
                    "record_type": "fold",
                    "fold_id": "fold_2024Q1",
                    "fold_status": "frozen",
                    "agent_trace_ref": str(not_object),
                }
            ]
        )
    bad_utf8 = tmp_path / "bad-utf8.jsonl"
    bad_utf8.write_bytes(b"\xff\xfe\n")
    with pytest.raises(AgentTraceSourceError, match="UTF-8"):
        build_meta_fold_reviews(
            [
                {
                    "record_type": "fold",
                    "fold_id": "fold_2024Q1",
                    "fold_status": "frozen",
                    "agent_trace_ref": str(bad_utf8),
                }
            ]
        )


def test_trace_limit_sets_source_truncated_and_keeps_events(tmp_path: Path) -> None:
    events = [
        {"event_type": "session_start", "mode": "fold"},
        {"event_type": "llm_call", "content": "still visible"},
        {"event_type": "trace_limit_reached", "max_bytes": 32},
        {"event_type": "session_end", "status": "finished"},
    ]
    reviews, sidecars = build_meta_fold_review_bundle([_fold(tmp_path, events)])
    full = _as_map(reviews[0]["agent_trace_full"])
    assert full["source_truncated"] is True
    assert full["available"] is True
    assert full["events"] == 4
    assert sidecars[0].source_truncated is True
    payload = sidecars[0].payload
    assert payload is not None
    types = [
        json.loads(line)["event_type"]
        for line in payload.decode("utf-8").splitlines()
        if line
    ]
    assert types == [
        "session_start",
        "llm_call",
        "trace_limit_reached",
        "session_end",
    ]


def test_sidecar_budget_fails_instead_of_truncating(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import autotrade.pipelines.meta_inputs as meta_inputs

    events = [{"event_type": "llm_call", "content": "x" * 40} for _ in range(3)]
    record = _fold(tmp_path, events)
    with pytest.raises(ValueError, match="sidecar exceeds"):
        build_meta_fold_review_bundle([record], max_file_bytes=32)
    first = _fold(tmp_path, events, fold_id="fold_a", run_id="run_a")
    second = _fold(
        tmp_path,
        [{"event_type": "llm_call", "content": "y" * 40}],
        fold_id="fold_b",
        run_id="run_b",
    )
    with pytest.raises(ValueError, match="window exceeds"):
        build_meta_fold_review_bundle([first, second], max_window_bytes=80)
    monkeypatch.setattr(meta_inputs, "AGENT_TRACE_FULL_MAX_FILE_BYTES", 16)
    with pytest.raises(ValueError, match="sidecar exceeds"):
        build_meta_fold_review_bundle(
            [record],
            max_file_bytes=meta_inputs.AGENT_TRACE_FULL_MAX_FILE_BYTES,
        )


def test_same_fold_id_in_two_epochs_has_distinct_opaque_sidecars(tmp_path: Path) -> None:
    first = _fold(tmp_path, run_id="run_epoch_1")
    second = _fold(tmp_path, run_id="run_epoch_2")
    second["epoch_id"] = "epoch_002"
    _reviews, sidecars = build_meta_fold_review_bundle([first, second])
    assert len({sidecar.relative_path for sidecar in sidecars}) == 2
    assert all("epoch_" not in sidecar.relative_path for sidecar in sidecars)
    assert all("fold_" not in sidecar.relative_path for sidecar in sidecars)


def test_sidecar_metadata_hash_permissions_and_stays_out_of_meta_context(
    tmp_path: Path,
) -> None:
    events = [
        {
            "event_type": "session_start",
            "system_prompt": "PROMPT",
            "mode": "fold",
        },
        {"event_type": "llm_call", "content": "plan next edit"},
        {
            "event_type": "tool_call",
            "tool": "write_file",
            "arguments": {
                "path": "/Data2/lzp/secret/main.py",
                "content": "secret body",
            },
            "result": {"ok": True},
        },
    ]
    fold_id = "fold_2024Q1"
    record = _fold(tmp_path, events, fold_id=fold_id)
    reviews, sidecars = build_meta_fold_review_bundle([record])
    review = reviews[0]
    sidecar = sidecars[0]
    fold_ref = agent_visible_ref(fold_id, prefix="fold_ref")
    trace_ref = agent_visible_ref(f"epoch_001:{fold_id}", prefix="trace_ref")
    assert review["fold_id"] == fold_ref
    assert "2024Q1" not in str(review["agent_trace_full"])
    assert review["agent_trace_full"] == sidecar.metadata()
    assert sidecar.relative_path == f"{AGENT_TRACE_FULL_RELATIVE_DIR}/{trace_ref}.jsonl"
    assert sidecar.payload is not None
    assert sidecar.sha256 == hashlib.sha256(sidecar.payload).hexdigest()
    assert sidecar.bytes == len(sidecar.payload)
    assert sidecar.events == 3
    workspace = tmp_path / "workspace"
    write_meta_agent_trace_sidecars(workspace, sidecars)
    dest = workspace / sidecar.relative_path
    assert dest.is_file()
    assert dest.read_bytes() == sidecar.payload
    assert stat_mode(dest) == 0o444
    assert not list(dest.parent.glob(".*.tmp"))
    public = json.dumps(review, ensure_ascii=False)
    assert "system_prompt" not in public
    assert "secret body" not in public
    assert "/Data2/" not in public
    assert "payload" not in _as_map(review["agent_trace_full"])
    assert sidecar.payload.decode("utf-8") not in public


def test_llm_meta_learner_writes_sidecar_outside_meta_context(tmp_path: Path) -> None:
    events = [
        {"event_type": "session_start", "system_prompt": "KEEP OUT", "mode": "fold"},
        {"event_type": "llm_call", "content": "delegate auditor"},
    ]
    fold_id = "fold_2025Q2"
    reviews, sidecars = build_meta_fold_review_bundle(
        [_fold(tmp_path, events, fold_id=fold_id)]
    )
    baseline = tmp_path / "baseline" / "main.py"
    baseline.parent.mkdir()
    baseline.write_text("def generate_orders(context):\n    return []\n", encoding="utf-8")
    store = FilesystemArtifactStore(tmp_path / "artifacts")
    llm = ScriptedLLM(
        [
            *_explore_then(
                ToolCall("taste", "write_taste", {"taste": "prefer simple signals"}),
                ToolCall("finish_meta", "finish_meta", {"taste_path": "taste.md"}),
            )
        ]
    )
    learner = LLMMetaLearner(
        llm=llm,
        baseline_strategy=baseline,
        artifact_store=store,
        experiment_dir=tmp_path / "experiment",
        runtime_root=tmp_path / "runtime",
        max_llm_calls=2,
        deadline_seconds=30.0,
        use_docker=False,
        rebuild_enabled=False,
    )
    learner(
        {
            "run_id": "run_meta",
            "experiment_id": "exp",
            "epoch_id": "epoch_002",
            "meta_learning_id": "epoch_002_after_fold",
            "development_history": {"fold_reviews": reviews},
            "agent_trace_sidecars": sidecars,
        }
    )
    collected = tmp_path / "run_meta"
    public = json.loads(
        (collected / "workspace" / "inputs" / "meta_context.json").read_text(
            encoding="utf-8"
        )
    )
    assert "agent_trace_sidecars" not in public
    assert "KEEP OUT" not in json.dumps(public)
    reviews = cast(
        list[object], _as_map(public["development_history"])["fold_reviews"]
    )
    full = _as_map(_as_map(reviews[0])["agent_trace_full"])
    live = (
        tmp_path
        / "runtime"
        / "run_meta"
        / "agent"
        / "workspace"
        / str(full["path"])
    )
    dest = collected / "workspace" / str(full["path"])
    assert live.is_file()
    assert dest.is_file()
    assert hashlib.sha256(live.read_bytes()).hexdigest() == full["sha256"]
    assert live.stat().st_size == full["bytes"]
    assert stat_mode(live) == 0o444
    assert stat_mode(live.parent) == 0o555
    assert "2025Q2" not in live.name
    assert "KEEP OUT" not in live.read_text(encoding="utf-8")
    host_manifest = json.loads(
        (collected / "host_run_manifest.json").read_text(encoding="utf-8")
    )
    overview = _as_map(_as_map(host_manifest["development_inputs"])["agent_trace_full"])
    assert overview["directory"] == "/mnt/agent/workspace/inputs/agent_traces"
    assert overview["available"] == 1
    assert overview["fold_count"] == 1
    refs = cast(list[object], overview["refs"])
    assert _as_map(refs[0])["path"] == full["path"]
    assert "/Data2/" not in json.dumps(overview)
    assert not (collected / "output" / "agent_traces").exists()
    output_dir = collected / "output"
    if output_dir.is_dir():
        assert not list(output_dir.rglob("*agent_traces*"))


def test_prompt_contract_reads_index_then_sidecar_and_does_not_inline() -> None:
    prompt = build_meta_learning_prompt({"fold_reviews": [{"agent_trace": ["raw"]}]})
    assert "inputs/meta_context.json" in prompt
    assert "agent_trace_full" in prompt
    assert "process summary" in prompt
    assert "available sidecar" in prompt
    assert "不要把原文堆进 PRIOR" in prompt
    assert "PIT/Test/Held-out" in prompt
    assert "fold_reviews" not in prompt
    assert META_PHASE_CONTRACT.count("完整 sidecar") >= 1
    assert "完整安全投影 sidecar" in META_PHASE_CONTRACT


def test_serialize_matches_sidecar_hash() -> None:
    events = project_full_agent_trace(
        [
            {"event_type": "llm_call", "content": "ok"},
            {"event_type": "session_end", "llm_calls": 1},
        ]
    )
    payload = serialize_full_agent_trace(events)
    assert payload.endswith(b"\n")
    assert hashlib.sha256(payload).hexdigest()
    parsed = [json.loads(line) for line in payload.decode("utf-8").splitlines()]
    assert parsed == events


def stat_mode(path: Path) -> int:
    return path.stat().st_mode & 0o777
