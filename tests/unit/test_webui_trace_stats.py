"""Trace summary aggregation and Agent-visible block projection."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from autotrade.environment.identity import AgentRefStore
from autotrade.pipelines.hitl_state import proc_start_ticks
from autotrade.webui import traces
from autotrade.webui.public_identity import PublicIdentity
from autotrade.webui.server import create_app
from autotrade.webui.traces import (
    DEFAULT_PAGE_BYTES,
    project_trace_blocks,
    trace_stats,
)

APP_JS = Path(__file__).resolve().parents[2] / "src/autotrade/webui/static/app.js"


def _write_trace(path: Path, events: list[dict[str, object]]) -> Path:
    path.write_text(
        "".join(json.dumps(event, ensure_ascii=False) + "\n" for event in events),
        encoding="utf-8",
    )
    return path


def test_trace_stats_counts_unique_subagent_tasks_not_calls(tmp_path: Path) -> None:
    path = _write_trace(
        tmp_path / "run.jsonl",
        [
            {
                "event_type": "subagent_task",
                "task_id": "agent_a",
                "parent_call_id": "call_1",
                "status": "started",
            },
            {"event_type": "subagent_llm", "task_id": "agent_a"},
            {"event_type": "subagent_tool", "task_id": "agent_a"},
            {"event_type": "subagent", "task_id": "agent_a"},
            {"event_type": "subagent_task", "task_id": "agent_b", "status": "started"},
            {"event_type": "subagent_llm", "task_id": "agent_b"},
            {
                "event_type": "llm_call",
                "task_id": "agent_a",
                "usage": {
                    "total_tokens": 9,
                    "prompt_tokens": 6,
                    "completion_tokens": 3,
                },
            },
            {"event_type": "tool_call", "tool": "agent", "task_id": "agent_ignored"},
            {"event_type": "session_start", "task_id": "not_agent"},
        ],
    )
    stats = trace_stats(path)
    counts = stats["counts"]
    assert isinstance(counts, dict)
    assert stats["subagent_tasks"] == 2
    assert counts["llm_call"] == 1
    assert stats["tool_counts"] == {"agent": 1}
    assert stats["llm_total_tokens"] == 9
    assert stats["llm_prompt_tokens"] == 6
    assert stats["llm_completion_tokens"] == 3
    assert stats["last_llm_prompt_tokens"] == 0
    assert stats["compact_ops"] == 0


def test_trace_stats_old_trace_without_start_still_counts_unique_task(
    tmp_path: Path,
) -> None:
    path = _write_trace(
        tmp_path / "old.jsonl",
        [
            {"event_type": "subagent_llm", "task_id": "agent_old"},
            {"event_type": "subagent_tool", "task_id": "agent_old"},
            {
                "event_type": "subagent",
                "task_id": "agent_old",
                "task": "do not parse this as a task id",
            },
            {"event_type": "subagent_llm", "task_id": "  "},
            {"event_type": "subagent_task", "task": "missing id"},
            {"event_type": "subagent", "task_id": 12},
        ],
    )
    assert trace_stats(path)["subagent_tasks"] == 1


def test_trace_stats_ignores_non_subagent_events(tmp_path: Path) -> None:
    path = _write_trace(
        tmp_path / "plain.jsonl",
        [
            {"event_type": "llm_call", "task_id": "agent_fake"},
            {"event_type": "tool_call", "tool": "read_file", "task_id": "agent_fake"},
            {"event_type": "context_compaction", "task_id": "agent_fake"},
        ],
    )
    stats = trace_stats(path)
    counts = stats["counts"]
    assert isinstance(counts, dict)
    assert stats["subagent_tasks"] == 0
    assert counts["llm_call"] == 1
    assert stats["tool_counts"] == {"read_file": 1}


def test_trace_stats_incrementally_dedups_appended_events(tmp_path: Path) -> None:
    path = tmp_path / "live.jsonl"
    _write_trace(path, [{"event_type": "subagent_task", "task_id": "agent_a", "status": "started"}])
    assert trace_stats(path)["subagent_tasks"] == 1
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"event_type": "subagent_llm", "task_id": "agent_a"}) + "\n")
        handle.write(json.dumps({"event_type": "subagent", "task_id": "agent_a"}) + "\n")
    assert trace_stats(path)["subagent_tasks"] == 1
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"event_type": "subagent_tool", "task_id": "agent_b"}) + "\n")
    assert trace_stats(path)["subagent_tasks"] == 2


def test_trace_stats_counts_compact_ops_without_double_counting(
    tmp_path: Path,
) -> None:
    path = tmp_path / "live_ops.jsonl"
    _write_trace(
        path,
        [
            {"event_type": "context_compaction", "status": "ok"},
            {"event_type": "context_compaction", "status": "error"},
            {
                "event_type": "context_edit",
                "reason": "context_window_budget",
                "summarized_tool_results": 2,
            },
            {
                "event_type": "context_edit",
                "reason": "provider_context_overflow_recovery",
                "summarized_tool_results": 1,
            },
            {"event_type": "llm_call"},
        ],
    )
    first = trace_stats(path)
    assert first["compact_ops"] == 2
    assert "trim_ops" not in first
    assert "clear_ops" not in first
    assert "cleared_tool_results" not in first
    counts = first["counts"]
    assert isinstance(counts, dict)
    assert counts["context_edit"] == 2
    second = trace_stats(path)
    assert second["compact_ops"] == 2
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"event_type": "context_compaction"}) + "\n")
        handle.write(
            json.dumps(
                {
                    "event_type": "context_edit",
                    "summarized_tool_results": 9,
                    "reason": "context_window_budget",
                }
            )
            + "\n"
        )
    third = trace_stats(path)
    assert third["compact_ops"] == 3
    later_counts = third["counts"]
    assert isinstance(later_counts, dict)
    assert later_counts["context_edit"] == 3


def test_trace_stats_last_main_llm_prompt_ignores_subagent_calls(
    tmp_path: Path,
) -> None:
    path = _write_trace(
        tmp_path / "main.jsonl",
        [
            {
                "event_type": "llm_call",
                "task_id": "agent_a",
                "usage": {"prompt_tokens": 6, "completion_tokens": 1, "total_tokens": 7},
            },
            {
                "event_type": "llm_call",
                "usage": {
                    "prompt_tokens": 12000,
                    "completion_tokens": 20,
                    "total_tokens": 12020,
                },
            },
            {
                "event_type": "llm_call",
                "usage": {
                    "prompt_tokens": 8000,
                    "completion_tokens": 10,
                    "total_tokens": 8010,
                },
            },
        ],
    )
    stats = trace_stats(path)
    assert stats["last_llm_prompt_tokens"] == 8000
    assert stats["llm_prompt_tokens"] == 20006


def test_the_stats_chips_read_only_fields_the_stats_route_serves(tmp_path: Path) -> None:
    """A renamed counter would render as 0 instead of failing, so every field
    the chip row reads must be one the trace stats route serves: the scan's
    counters plus the model window the route adds from the experiment's model."""

    script = APP_JS.read_text(encoding="utf-8")
    source = script.split("function statsChipsRow(", 1)[1].split("\nfunction ", 1)[0]
    reads = set(re.findall(r"\bstats\.([a-z_]+)", source))
    served = set(trace_stats(_write_trace(tmp_path / "run.jsonl", []))) | {"context_window_tokens"}
    assert {"subagent_tasks", "compact_ops", "last_llm_prompt_tokens"} <= reads
    assert reads <= served, sorted(reads - served)


def _subagent_blocks(blocks: list[dict[str, object]]) -> list[dict[str, object]]:
    return [block for block in blocks if block.get("kind") == "subagent"]


def _experiment_with_trace(
    tmp_path: Path,
    events: list[dict[str, object]],
    *,
    experiment_id: str = "demo",
    run_id: str = "run_001",
) -> PublicIdentity:
    root = tmp_path / "experiments" / experiment_id
    AgentRefStore(root)
    (root / "hitl").mkdir(parents=True)
    (root / "hitl" / "status.json").write_text(
        json.dumps({"schema_version": 1, "state": "created"}),
        encoding="utf-8",
    )
    (root / "hitl" / "schedule.json").write_text(
        json.dumps({"schema_version": 1, "sessions": []}), encoding="utf-8"
    )
    traces_dir = root / "artifacts" / "traces"
    traces_dir.mkdir(parents=True)
    _write_trace(traces_dir / f"{run_id}.jsonl", events)
    return PublicIdentity(root)


def test_project_main_tool_started_and_completed_counts_once() -> None:
    blocks = project_trace_blocks(
        [
            {"event_type": "llm_call", "content": "plan"},
            {
                "event_type": "tool_call_started",
                "tool": "read_file",
                "tool_call_id": "c1",
            },
            {
                "event_type": "tool_call",
                "tool": "read_file",
                "tool_call_id": "c1",
                "result": {"ok": True},
            },
            {"event_type": "llm_call", "content": "done"},
        ]
    )
    assert [block["kind"] for block in blocks] == [
        "agent_output",
        "tool_group",
        "agent_output",
    ]
    group = blocks[1]
    assert group["count"] == 1
    assert group["ok"] == 1
    assert group["failed"] == 0
    assert group["running"] == 0
    assert group["tools"] == [
        {
            "name": "read_file",
            "count": 1,
            "ok": 1,
            "failed": 0,
            "running": 0,
            "summary": "",
        }
    ]


def test_project_failed_running_and_tail_tool_groups() -> None:
    blocks = project_trace_blocks(
        [
            {"event_type": "llm_call", "content": "before"},
            {
                "event_type": "tool_call",
                "tool": "shell",
                "result": {"ok": False, "error": "boom"},
            },
            {"event_type": "llm_call", "content": "after fail"},
            {
                "event_type": "tool_call_started",
                "tool": "read_file",
                "tool_call_id": "open1",
            },
            {"event_type": "llm_call", "content": "after running"},
            {"event_type": "tool_call", "tool": "grep", "result": {"ok": True}},
        ]
    )
    assert [block["kind"] for block in blocks] == [
        "agent_output",
        "tool_group",
        "agent_output",
        "tool_group",
        "agent_output",
        "tool_group",
    ]
    failed, running, tail = (block for block in blocks if block["kind"] == "tool_group")
    assert failed["count"] == 1 and failed["failed"] == 1 and failed["ok"] == 0
    tools = failed["tools"]
    assert isinstance(tools, list)
    assert tools[0]["summary"] == "boom"
    assert running["running"] == 1 and running["ok"] == 0
    assert tail["ok"] == 1 and tail["count"] == 1


def test_project_subagent_emits_one_card_per_task_id() -> None:
    blocks = project_trace_blocks(
        [
            {
                "event_type": "subagent_task",
                "task_id": "agent_new",
                "status": "started",
                "description": "inspect schema",
            },
            {"event_type": "subagent_llm", "task_id": "agent_new"},
            {
                "event_type": "subagent_tool",
                "task_id": "agent_new",
                "tool": "grep",
                "tool_call_id": "g1",
                "result": {"ok": True},
            },
            {
                "event_type": "subagent_task",
                "task_id": "agent_new",
                "status": "completed",
                "summary": "has trade_date",
            },
            {"event_type": "subagent_llm", "task_id": "agent_old"},
            {
                "event_type": "subagent_tool",
                "task_id": "agent_old",
                "tool": "read_file",
                "result": {"ok": True},
            },
            {
                "event_type": "subagent",
                "task_id": "agent_old",
                "summary": "old summary",
            },
        ]
    )
    sub = _subagent_blocks(blocks)
    # One card per task, updated in place: a finished report renders once.
    assert [(block["task_id"], block["phase"], block["status"]) for block in sub] == [
        ("agent_new", "ended", "completed"),
        ("agent_old", "ended", "completed"),
    ]
    new_ended, old_ended = sub
    assert new_ended["description"] == "inspect schema"
    assert "task" not in new_ended
    assert new_ended["summary"] == "has trade_date"
    assert new_ended["tools"] == [
        {
            "name": "grep",
            "count": 1,
            "ok": 1,
            "failed": 0,
            "running": 0,
            "summary": "",
        }
    ]
    assert old_ended["summary"] == "old summary"
    assert old_ended["tools"] == [
        {
            "name": "read_file",
            "count": 1,
            "ok": 1,
            "failed": 0,
            "running": 0,
            "summary": "",
        }
    ]


def test_project_agent_output_exposes_llm_call_model() -> None:
    blocks = project_trace_blocks(
        [
            {
                "event_type": "llm_call",
                "content": "plan",
                "model": "qwen-3.8-27b-fp8",
            },
            {"event_type": "llm_call", "content": "done"},
            {"event_type": "llm_call", "content": "blank", "model": "  "},
            {
                "event_type": "subagent_task",
                "task_id": "agent_a",
                "status": "started",
                "role": "auditor",
                "model": "child-model",
            },
        ]
    )
    outputs = [block for block in blocks if block.get("kind") == "agent_output"]
    assert [block.get("model") for block in outputs] == [
        "qwen-3.8-27b-fp8",
        None,
        None,
    ]
    started = next(
        block
        for block in _subagent_blocks(blocks)
        if block["phase"] == "started"
    )
    assert started["model"] == "child-model"
    assert started["role"] == "auditor"


def test_project_subagent_exposes_model_thinking_and_inherit_context() -> None:
    blocks = project_trace_blocks(
        [
            {
                "event_type": "subagent_task",
                "task_id": "agent_meta",
                "status": "started",
                "role": "auditor",
                "model": "qwen-3.8-27b-fp8",
                "thinking": "low",
                "inherit_context": True,
                "description": "schema audit",
            },
            {
                "event_type": "subagent",
                "task_id": "agent_meta",
                "status": "completed",
                "summary": "ok",
                "role": "auditor",
                "model": "qwen-3.8-27b-fp8",
                "thinking": "low",
                "inherit_context": True,
            },
        ]
    )
    (card,) = _subagent_blocks(blocks)
    assert card["phase"] == "ended"
    assert card["role"] == "auditor"
    assert card["model"] == "qwen-3.8-27b-fp8"
    assert card["thinking"] == "low"
    assert card["inherit_context"] is True
    assert card["description"] == "schema audit"
    assert "turns_limit" not in card


def test_project_subagent_card_carries_the_effective_turn_budget() -> None:
    """The parent-chosen (or defaulted) ``max_turns`` lands on the card as
    ``turns_limit``, the field the console reads."""

    blocks = project_trace_blocks(
        [
            {
                "event_type": "subagent_task",
                "task_id": "agent_turns",
                "status": "started",
                "role": "developer",
                "thinking": "medium",
                "turns_limit": 12,
            }
        ]
    )
    (card,) = _subagent_blocks(blocks)
    assert card["turns_limit"] == 12
    assert "block.turns_limit" in APP_JS.read_text(encoding="utf-8")


def test_project_legacy_subagent_digest_is_ignored() -> None:
    blocks = project_trace_blocks(
        [
            {
                "event_type": "subagent",
                "task_id": "legacy",
                "status": "completed",
                "digest": "must not migrate",
            }
        ]
    )
    ended = next(block for block in _subagent_blocks(blocks) if block["phase"] == "ended")
    assert ended["summary"] == ""
    assert "digest" not in ended


def test_project_subagent_internal_tools_stay_on_subagent_card() -> None:
    blocks = project_trace_blocks(
        [
            {"event_type": "llm_call", "content": "delegate"},
            {
                "event_type": "tool_call_started",
                "tool": "agent",
                "tool_call_id": "e1",
            },
            {"event_type": "subagent_task", "task_id": "agent_a", "status": "started"},
            {
                "event_type": "subagent_tool",
                "task_id": "agent_a",
                "tool": "grep",
                "tool_call_id": "g1",
                "result": {"ok": True},
            },
            {
                "event_type": "subagent_tool",
                "task_id": "agent_a",
                "tool": "read_file",
                "result": {"ok": True},
            },
            {
                "event_type": "subagent",
                "task_id": "agent_a",
                "summary": "done",
            },
            {
                "event_type": "tool_call",
                "tool": "agent",
                "tool_call_id": "e1",
                "result": {"ok": True},
            },
            {"event_type": "llm_call", "content": "back"},
        ]
    )
    groups = [block for block in blocks if block["kind"] == "tool_group"]
    assert len(groups) == 1
    assert groups[0]["count"] == 1
    assert groups[0]["ok"] == 1
    assert [row["name"] for row in groups[0]["tools"]] == ["agent"]
    ended = next(
        block
        for block in _subagent_blocks(blocks)
        if block["phase"] == "ended"
    )
    assert [row["name"] for row in ended["tools"]] == ["grep", "read_file"]
    orphan = project_trace_blocks(
        [
            {
                "event_type": "subagent_tool",
                "task_id": "agent_b",
                "tool": "grep",
                "result": {"ok": True},
            },
            {"event_type": "subagent", "task_id": "agent_b"},
        ]
    )
    assert all(block["kind"] != "tool_group" for block in orphan)
    orphan_ended = next(
        block for block in _subagent_blocks(orphan) if block["phase"] == "ended"
    )
    assert [row["name"] for row in orphan_ended["tools"]] == ["grep"]


def test_project_subagent_terminal_statuses() -> None:
    blocks = project_trace_blocks(
        [
            {"event_type": "subagent_task", "task_id": "ok", "status": "completed"},
            {
                "event_type": "subagent_task",
                "task_id": "err",
                "status": "error",
                "error": "nope",
            },
            {"event_type": "subagent_task", "task_id": "to", "status": "timeout"},
            {
                "event_type": "subagent_task",
                "task_id": "cx",
                "status": "cancelled",
            },
            # A provider refusal must never be drawn as a completed child.
            {"event_type": "subagent", "task_id": "rf", "status": "refused"},
        ]
    )
    ended = {
        block["task_id"]: block["status"]
        for block in _subagent_blocks(blocks)
        if block["phase"] == "ended"
    }
    assert ended == {
        "ok": "completed",
        "err": "error",
        "to": "timeout",
        "cx": "cancelled",
        "rf": "refused",
    }
    err = next(
        block
        for block in _subagent_blocks(blocks)
        if block["task_id"] == "err" and block["phase"] == "ended"
    )
    assert err["error"] == "nope"


def test_project_user_message_flushes_pending_tools() -> None:
    blocks = project_trace_blocks(
        [
            {
                "event_type": "tool_call_started",
                "tool": "read_file",
                "tool_call_id": "c1",
            },
            {"event_type": "user_message", "content": "continue", "ts": "u1"},
            {"event_type": "llm_call", "content": "ok"},
        ]
    )
    assert [block["kind"] for block in blocks] == [
        "tool_group",
        "user",
        "agent_output",
    ]
    assert blocks[1]["text"] == "continue"
    assert blocks[0]["running"] == 1


def test_project_unknown_bad_payload_and_clips_long_text() -> None:
    assert project_trace_blocks(None) == []
    assert project_trace_blocks("nope") == []
    assert project_trace_blocks([None, 1, "x", {"event_type": "mystery"}]) == []
    long_text = "字" * (traces._BLOCK_TEXT_CHARS + 50)
    long_subagent_summary = "摘" * (traces._BLOCK_SUBAGENT_SUMMARY_CHARS + 20)
    long_error = "错" * (traces._BLOCK_ERROR_CHARS + 20)
    long_summary = "e" * (traces._BLOCK_SUMMARY_CHARS + 20)
    blocks = project_trace_blocks(
        [
            {"event_type": "llm_call", "content": long_text},
            {
                "event_type": "tool_call",
                "tool": "shell",
                "result": {"ok": False, "error": long_summary},
            },
            {"event_type": "user_message", "message": long_text},
            {
                "event_type": "subagent",
                "task_id": "t1",
                "summary": long_subagent_summary,
                "error": long_error,
                "status": "error",
            },
        ]
    )
    texts = [block for block in blocks if block["kind"] in {"agent_output", "user"}]
    assert all(len(str(block["text"])) == traces._BLOCK_TEXT_CHARS for block in texts)
    group = next(block for block in blocks if block["kind"] == "tool_group")
    tools = group["tools"]
    assert isinstance(tools, list)
    assert len(str(tools[0]["summary"])) == traces._BLOCK_SUMMARY_CHARS
    ended = next(
        block
        for block in _subagent_blocks(blocks)
        if block["phase"] == "ended"
    )
    assert len(str(ended["summary"])) == traces._BLOCK_SUBAGENT_SUMMARY_CHARS
    assert len(str(ended["error"])) == traces._BLOCK_ERROR_CHARS


def test_project_internal_events_emit_no_blocks() -> None:
    assert (
        project_trace_blocks(
            [
                {
                    "event_type": "session_start",
                    "system_prompt": "sys",
                    "instruction": "do work",
                },
                {"event_type": "system_prompt", "content": "sys"},
                {"event_type": "instruction", "content": "do work"},
                {"event_type": "llm_call", "usage": {"total_tokens": 3}},
                {"event_type": "budget", "remaining": 1},
            ]
        )
        == []
    )


def test_project_compaction_and_notice_blocks() -> None:
    """The parent's compactions and the advisory before them are their own
    quiet blocks: who triggered it, which calls it replaced, the summary that
    now stands for them; a compaction that failed says why."""

    blocks = project_trace_blocks(
        [
            {"event_type": "context_notice", "ts": "t1", "estimated_tokens": 150_000, "token_threshold": 200_000},
            {"event_type": "llm_call", "content": "plan"},
            {
                "event_type": "context_compaction",
                "ts": "t2",
                "trigger": "agent",
                "status": "ok",
                "call_index": 12,
                "replaced_call_range": [1, 12],
                "dropped_messages": 40,
                "summary_chars": 900,
                "summary": "strategy state and open threads",
            },
            {"event_type": "context_compaction", "ts": "t3", "status": "error", "error": "provider timeout"},
            # A child's compaction stays on its own card, not in the parent's flow.
            {"event_type": "subagent_context_compaction", "task_id": "agent_a", "turn": 3, "compaction": {"status": "ok"}},
        ]
    )
    assert [block["kind"] for block in blocks] == ["notice", "agent_output", "compaction", "compaction", "subagent"]
    assert blocks[0] == {"kind": "notice", "ts": "t1", "estimated_tokens": 150_000, "token_threshold": 200_000}
    assert blocks[2] == {
        "kind": "compaction",
        "ts": "t2",
        "trigger": "agent",
        "status": "ok",
        "replaced_call_range": [1, 12],
        "dropped_messages": 40,
        "summary": "strategy state and open threads",
        "summary_chars": 900,
        "error": "",
    }
    assert (blocks[3]["trigger"], blocks[3]["status"], blocks[3]["error"]) == ("runtime", "error", "provider timeout")
    assert blocks[3]["replaced_call_range"] is None
    # The console renders both kinds, as a fold on a rule and as a marker.
    script = APP_JS.read_text(encoding="utf-8")
    assert 'kind === "compaction") renderCompactionBlock(' in script
    assert 'kind === "notice") renderNoticeBlock(' in script


def test_project_subagent_finishing_after_a_compaction_keeps_its_one_card() -> None:
    """The card is created where the child was launched and updated in place
    by every later event of its task id, a compaction of the parent's
    context in between notwithstanding: one card, its terminal status on it."""

    blocks = project_trace_blocks(
        [
            {"event_type": "subagent_task", "task_id": "agent_a", "status": "started", "ts": "t1", "role": "explore"},
            {"event_type": "subagent_llm", "task_id": "agent_a", "turn": 1, "usage": {"prompt_tokens": 10, "completion_tokens": 2}},
            {"event_type": "context_compaction", "ts": "t2", "trigger": "agent", "status": "ok"},
            {"event_type": "subagent", "task_id": "agent_a", "ts": "t3", "status": "completed", "turns": 3, "llm_calls": 3, "tool_calls": 5, "summary": "done"},
        ]
    )
    assert [block["kind"] for block in blocks] == ["subagent", "compaction"]
    card = blocks[0]
    assert (card["task_id"], card["status"], card["phase"]) == ("agent_a", "completed", "ended")
    assert (card["turns"], card["tool_calls"], card["summary"]) == (3, 5, "done")


def test_trace_stats_carries_the_last_budget_block(tmp_path: Path) -> None:
    """Every budgeted event carries the session's cumulative spend; the stats
    projection hands the newest block on, across appends, so the console's
    budget bars follow a live session."""

    path = _write_trace(
        tmp_path / "run.jsonl",
        [
            {"event_type": "session_start"},
            {"event_type": "llm_call", "budget_used": {"inference_seconds": 10.0, "llm_calls": 1, "test_replays": 0, "null_controls": 0}},
            {"event_type": "tool_call", "tool": "shell", "budget_used": {"inference_seconds": 25.5, "llm_calls": 2, "test_replays": 4, "null_controls": 0}},
        ],
    )
    first = trace_stats(path)
    assert first["budget_used"] == {"inference_seconds": 25.5, "llm_calls": 2, "test_replays": 4, "null_controls": 0}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"event_type": "llm_call", "budget_used": {"inference_seconds": 30.0, "llm_calls": 3, "test_replays": 4, "null_controls": 1}}) + "\n")
    assert trace_stats(path)["budget_used"]["llm_calls"] == 3
    assert trace_stats(_write_trace(tmp_path / "bare.jsonl", [{"event_type": "session_start"}]))["budget_used"] is None


def test_trace_blocks_api_projects_whole_trace_without_paging_groups(
    tmp_path: Path,
) -> None:
    events = [
        {"event_type": "session_start", "system_prompt": "sys", "instruction": "go"},
        {"event_type": "llm_call", "content": "plan"},
        {
            "event_type": "tool_call_started",
            "tool": "read_file",
            "tool_call_id": "c1",
        },
        {
            "event_type": "tool_call",
            "tool": "read_file",
            "tool_call_id": "c1",
            "result": {"ok": True},
        },
        {
            "event_type": "context_compaction",
            "trigger": "agent",
            "status": "ok",
            "replaced_call_range": [1, 1],
            "dropped_messages": 3,
            "summary": "x" * (DEFAULT_PAGE_BYTES + 2048),
        },
        {"event_type": "user_message", "content": "ok?"},
        {"event_type": "llm_call", "content": "done"},
    ]
    identity = _experiment_with_trace(tmp_path, events)
    client = TestClient(create_app(tmp_path))
    trace_ref = identity.trace_ref("run_001")
    response = client.get(
        "/api/experiments/demo/trace/blocks", params={"run_id": trace_ref}
    )
    assert response.status_code == 200
    payload = response.json()
    assert "events" not in payload
    assert payload["eof"] is True
    assert payload["blocks"] == project_trace_blocks(events)
    assert [block["kind"] for block in payload["blocks"]] == [
        "agent_output",
        "tool_group",
        "compaction",
        "user",
        "agent_output",
    ]
    group = payload["blocks"][1]
    assert group["count"] == 1 and group["ok"] == 1 and group["running"] == 0


def test_trace_blocks_api_guards_invalid_experiment_and_run(tmp_path: Path) -> None:
    identity = _experiment_with_trace(
        tmp_path, [{"event_type": "llm_call", "content": "x"}]
    )
    client = TestClient(create_app(tmp_path))
    missing = client.get(
        "/api/experiments/nope/trace/blocks", params={"run_id": "run_001"}
    )
    assert missing.status_code == 404
    traversal = client.get(
        "/api/experiments/..secret/trace/blocks", params={"run_id": "run_001"}
    )
    assert traversal.status_code in {400, 404}
    assert "etc" not in traversal.text
    for run_id in ("../run_001", "run_001/../..", "/etc/passwd"):
        response = client.get(
            "/api/experiments/demo/trace/blocks", params={"run_id": run_id}
        )
        assert response.status_code in {400, 404}, run_id
        assert "etc" not in response.text
    missing_run = client.get(
        "/api/experiments/demo/trace/blocks",
        params={"run_id": identity.run_ref("run_missing")},
    )
    assert missing_run.status_code == 404


def _claim_writer(experiment_dir: Path, *, alive: bool, run_id: str | None = None) -> None:
    """Own the trace from this process, or from nobody at all; ``run_id`` is
    the run the status names as the one being written."""

    status: dict[str, object] = {"schema_version": 1, "state": "running", "run_id": run_id}
    if alive:
        status["pid"] = os.getpid()
        status["pid_start_ticks"] = proc_start_ticks(os.getpid())
    (experiment_dir / "hitl" / "status.json").write_text(
        json.dumps(status), encoding="utf-8"
    )


def test_a_torn_tail_ends_pagination_once_its_writer_is_gone(tmp_path: Path) -> None:
    """Trailing bytes with no newline mean "an event is still being written"
    only while the writer lives. A write torn by ENOSPC or a kill leaves the
    same bytes forever, so a dead run's page must reach EOF instead of
    answering zero progress to every poll -- and the fragment is reported as a
    truncation, never decoded into an event."""

    events = [{"event_type": "llm_call", "content": "one"}]
    identity = _experiment_with_trace(tmp_path, events)
    root = tmp_path / "experiments" / "demo"
    trace = root / "artifacts" / "traces" / "run_001.jsonl"
    complete = trace.stat().st_size
    with trace.open("a", encoding="utf-8") as handle:
        handle.write('{"event_type": "llm_call", "content": "tor')
    size = trace.stat().st_size
    marker = f"<truncated final event: {size - complete} bytes>"

    live = traces.read_trace_page(trace, offset=complete, writer_alive=True)
    assert live == {"events": [], "next_offset": complete, "eof": False}
    live_tail = traces.read_trace_tail(trace, max_events=10, writer_alive=True)
    assert live_tail["next_offset"] == complete and live_tail["eof"] is False
    assert [event.get("content") for event in live_tail["events"]] == ["one"]

    dead = traces.read_trace_page(trace, offset=complete, writer_alive=False)
    assert dead == {"events": [{"raw": marker}], "next_offset": size, "eof": True}
    dead_tail = traces.read_trace_tail(trace, max_events=10, writer_alive=False)
    assert dead_tail["next_offset"] == size and dead_tail["eof"] is True
    assert [event.get("content") for event in dead_tail["events"]] == ["one", None]
    assert dead_tail["events"][-1] == {"raw": marker}
    assert "tor" not in json.dumps(dead_tail["events"], ensure_ascii=False)

    # The route decides liveness the way the SSE tail does, from hitl/status.json.
    client = TestClient(create_app(tmp_path))
    params = {"run_id": identity.trace_ref("run_001"), "offset": complete}
    _claim_writer(root, alive=True)
    running = client.get("/api/experiments/demo/trace/blocks", params=params).json()
    assert running["eof"] is False and running["next_offset"] == complete
    assert running["blocks"] == []
    _claim_writer(root, alive=False)
    ended = client.get("/api/experiments/demo/trace/blocks", params=params).json()
    assert ended["eof"] is True and ended["next_offset"] == size
    assert [block["kind"] for block in ended["blocks"]] == ["raw"]
    assert ended["blocks"][0]["text"] == marker


def test_a_dead_runs_torn_tail_ends_every_page_that_reaches_it(tmp_path: Path) -> None:
    """The replay loader re-reads from offset 0 with a growing window, so the
    torn tail must end the trace on whichever page reaches it, not only on a
    page that starts at it. Liveness is the requested run's: a finished run
    is dead while the worker writes a newer run of the same experiment."""

    events = [
        {"event_type": "llm_call", "content": "one"},
        {"event_type": "llm_call", "content": "two"},
    ]
    identity = _experiment_with_trace(tmp_path, events)
    root = tmp_path / "experiments" / "demo"
    trace = root / "artifacts" / "traces" / "run_001.jsonl"
    second = len(json.dumps(events[0]).encode()) + 1
    complete = trace.stat().st_size
    with trace.open("a", encoding="utf-8") as handle:
        handle.write('{"event_type": "llm_call", "content": "tor')
    size = trace.stat().st_size
    marker = {"raw": f"<truncated final event: {size - complete} bytes>"}

    for offset, expected in ((0, events), (second, events[1:])):
        dead = traces.read_trace_page(trace, offset=offset, writer_alive=False)
        assert dead == {"events": [*expected, marker], "next_offset": size, "eof": True}
        assert "tor" not in json.dumps(dead["events"])
        live = traces.read_trace_page(trace, offset=offset, writer_alive=True)
        assert live == {"events": expected, "next_offset": complete, "eof": False}
    # A window that stops short of the tail is an ordinary page.
    short = traces.read_trace_page(trace, offset=0, max_bytes=second, writer_alive=False)
    assert short == {"events": events[:1], "next_offset": second, "eof": False}

    client = TestClient(create_app(tmp_path))
    params = {"run_id": identity.trace_ref("run_001"), "offset": 0}

    def page() -> dict[str, object]:
        return client.get("/api/experiments/demo/trace/blocks", params=params).json()

    _claim_writer(root, alive=True, run_id="run_001")
    assert traces.trace_writer_alive(root, "run_001")
    running = page()
    assert running["eof"] is False and running["next_offset"] == complete
    assert "raw" not in [block["kind"] for block in running["blocks"]]
    # The same worker has moved on to a newer run: this one can no longer grow.
    _claim_writer(root, alive=True, run_id="run_002")
    assert not traces.trace_writer_alive(root, "run_001")
    superseded = page()
    _claim_writer(root, alive=False, run_id="run_001")
    crashed = page()
    for ended in (superseded, crashed):
        assert ended["eof"] is True and ended["next_offset"] == size
        last = ended["blocks"][-1]
        assert (last["kind"], last["text"]) == ("raw", marker["raw"])


def test_a_dead_runs_open_tool_calls_read_as_interrupted(tmp_path: Path) -> None:
    """A call started and never finished is running only while its writer
    lives. Once the writer is gone a read that reaches the end of the trace
    counts it as interrupted, so the replay of a terminated session never
    shows a tool still running; a read that stops short of the end cannot
    tell and leaves it running."""

    events = [
        {"event_type": "llm_call", "content": "plan"},
        {"event_type": "tool_call_started", "tool": "read_file", "tool_call_id": "c1"},
        {"event_type": "tool_call", "tool": "read_file", "tool_call_id": "c1", "result": {"ok": True}},
        {"event_type": "tool_call_started", "tool": "validate", "tool_call_id": "c2"},
        # A content-free round: an event after the open call that draws no block.
        {"event_type": "llm_call"},
    ]
    identity = _experiment_with_trace(tmp_path, events)
    root = tmp_path / "experiments" / "demo"
    trace = root / "artifacts" / "traces" / "run_001.jsonl"
    short = trace.stat().st_size - len(json.dumps(events[-1]).encode()) - 1

    def group(payload: dict[str, object]) -> dict[str, object]:
        [block] = [block for block in payload["blocks"] if block["kind"] == "tool_group"]
        return block

    live = group(traces.read_trace_blocks(trace, writer_alive=True))
    assert (live["running"], "interrupted" in live) == (1, False)
    dead = traces.read_trace_blocks(trace, writer_alive=False)
    assert dead["eof"] is True
    ended = group(dead)
    assert (ended["count"], ended["ok"], ended["running"], ended["interrupted"]) == (2, 1, 0, 1)
    rows = {row["name"]: row for row in ended["tools"]}
    assert (rows["validate"]["running"], rows["validate"]["interrupted"]) == (0, 1)
    assert "interrupted" not in rows["read_file"]
    partial = traces.read_trace_blocks(trace, offset=0, max_bytes=short, writer_alive=False)
    assert partial["eof"] is False and group(partial)["running"] == 1

    # The route reads liveness from hitl/status.json, for a tail as for a page.
    client = TestClient(create_app(tmp_path))
    params = {"run_id": identity.trace_ref("run_001"), "tail_events": 20}
    _claim_writer(root, alive=True, run_id="run_001")
    running = client.get("/api/experiments/demo/trace/blocks", params=params).json()
    assert group(running)["running"] == 1
    _claim_writer(root, alive=False, run_id="run_001")
    terminated = client.get("/api/experiments/demo/trace/blocks", params=params).json()
    assert terminated["eof"] is True
    assert (group(terminated)["running"], group(terminated)["interrupted"]) == (0, 1)


def test_project_subagent_running_card_accumulates_progress() -> None:
    blocks = project_trace_blocks(
        [
            {
                "event_type": "subagent_task",
                "ts": "2026-08-27T10:00:00+00:00",
                "task_id": "agent_live",
                "status": "started",
                "role": "general-purpose",
                "description": "Value 因子研究",
            },
            {
                "event_type": "subagent_llm",
                "task_id": "agent_live",
                "turn": 1,
                "usage": {
                    "prompt_tokens": 1000,
                    "completion_tokens": 100,
                    "total_tokens": 1100,
                },
            },
            {
                "event_type": "subagent_tool",
                "task_id": "agent_live",
                "tool": "grep",
                "result": {"ok": True},
            },
            {
                "event_type": "subagent_tool",
                "task_id": "agent_live",
                "tool": "shell",
                "result": {"ok": False, "error": "exit 1"},
            },
            {
                "event_type": "subagent_llm",
                "task_id": "agent_live",
                "turn": 2,
                "usage": {
                    "prompt_tokens": 2000,
                    "completion_tokens": 200,
                    "total_tokens": 2200,
                },
            },
        ]
    )
    (card,) = _subagent_blocks(blocks)
    assert card["phase"] == "started" and card["status"] == "running"
    assert card["started_at"] == "2026-08-27T10:00:00+00:00"
    assert "ended_at" not in card
    assert card["turns"] == 2 and card["llm_calls"] == 2
    assert card["tool_calls"] == 2
    assert card["usage"] == {
        "prompt_tokens": 3000,
        "completion_tokens": 300,
        "total_tokens": 3300,
    }
    assert card["last_tool"] == {"name": "shell", "status": "failed"}


def test_project_subagent_finished_card_prefers_terminal_totals() -> None:
    """A tail window can miss early rounds, so the terminal event's own
    totals win over whatever was summed from the visible records."""

    blocks = project_trace_blocks(
        [
            {
                "event_type": "subagent_task",
                "ts": "2026-08-27T10:00:00+00:00",
                "task_id": "agent_done",
                "status": "started",
                "role": "Explore",
            },
            {
                "event_type": "subagent_llm",
                "task_id": "agent_done",
                "turn": 1,
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 1,
                    "total_tokens": 11,
                },
            },
            {
                "event_type": "subagent",
                "ts": "2026-08-27T10:30:00+00:00",
                "task_id": "agent_done",
                "status": "completed",
                "summary": "结论",
                "turns": 19,
                "llm_calls": 20,
                "tool_calls": 143,
                "usage_totals": {
                    "prompt_tokens": 1_018_682,
                    "completion_tokens": 63_802,
                    "total_tokens": 1_082_484,
                },
            },
        ]
    )
    (card,) = _subagent_blocks(blocks)
    assert card["phase"] == "ended" and card["status"] == "completed"
    assert card["started_at"] == "2026-08-27T10:00:00+00:00"
    assert card["ended_at"] == "2026-08-27T10:30:00+00:00"
    assert card["turns"] == 19 and card["llm_calls"] == 20
    assert card["tool_calls"] == 143
    assert card["usage"]["total_tokens"] == 1_082_484
    assert card["summary"] == "结论"


def test_project_subagent_usage_total_falls_back_to_its_halves() -> None:
    blocks = project_trace_blocks(
        [
            {
                "event_type": "subagent_llm",
                "task_id": "agent_half",
                "usage": {"prompt_tokens": 40, "completion_tokens": 2},
            }
        ]
    )
    (card,) = _subagent_blocks(blocks)
    assert card["usage"] == {
        "prompt_tokens": 40,
        "completion_tokens": 2,
        "total_tokens": 42,
    }


def test_trace_stats_separates_running_subagents_and_their_tokens(
    tmp_path: Path,
) -> None:
    path = tmp_path / "run.jsonl"
    _write_trace(
        path,
        [
            {
                "event_type": "llm_call",
                "usage": {
                    "prompt_tokens": 500,
                    "completion_tokens": 50,
                    "total_tokens": 550,
                },
            },
            {"event_type": "subagent_task", "task_id": "done", "status": "started"},
            {
                "event_type": "subagent_llm",
                "task_id": "done",
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 1,
                    "total_tokens": 11,
                },
            },
            {
                "event_type": "subagent",
                "task_id": "done",
                "status": "completed",
                "usage_totals": {
                    "prompt_tokens": 900,
                    "completion_tokens": 90,
                    "total_tokens": 990,
                },
            },
            {"event_type": "subagent_task", "task_id": "live", "status": "started"},
            {
                "event_type": "subagent_llm",
                "task_id": "live",
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 10,
                    "total_tokens": 110,
                },
            },
        ],
    )
    stats = trace_stats(path)
    assert stats["subagent_tasks"] == 2
    assert stats["subagent_running"] == 1
    # The finished task reports its own totals; the live one is summed so far.
    assert stats["subagent_prompt_tokens"] == 1000
    assert stats["subagent_completion_tokens"] == 100
    assert stats["subagent_total_tokens"] == 1100
    # Sub-agent spend never leaks into the main-agent figures.
    assert stats["llm_prompt_tokens"] == 500
    assert stats["llm_total_tokens"] == 550


def test_trace_stats_closes_a_running_subagent_on_append(tmp_path: Path) -> None:
    path = tmp_path / "append.jsonl"
    _write_trace(
        path,
        [
            {"event_type": "subagent_task", "task_id": "live", "status": "started"},
            {
                "event_type": "subagent_llm",
                "task_id": "live",
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 10,
                    "total_tokens": 110,
                },
            },
        ],
    )
    first = trace_stats(path)
    assert first["subagent_running"] == 1
    assert first["subagent_total_tokens"] == 110
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "event_type": "subagent",
                    "task_id": "live",
                    "status": "completed",
                    "usage_totals": {
                        "prompt_tokens": 300,
                        "completion_tokens": 30,
                        "total_tokens": 330,
                    },
                }
            )
            + "\n"
        )
    second = trace_stats(path)
    assert second["subagent_tasks"] == 1
    assert second["subagent_running"] == 0
    assert second["subagent_total_tokens"] == 330


def test_project_unreadable_lines_stay_visible_as_raw_blocks() -> None:
    blocks = project_trace_blocks(
        [
            {"event_type": "llm_call", "content": "plan"},
            {"raw": "{not json"},
            {"raw": "<oversized event skipped: 900000 bytes>"},
        ]
    )
    assert [block["kind"] for block in blocks] == ["agent_output", "raw", "raw"]
    assert blocks[1]["text"] == "{not json"
    assert blocks[2]["text"] == "<oversized event skipped: 900000 bytes>"


def test_corrupt_line_is_projected_and_marked_in_the_download(
    tmp_path: Path,
) -> None:
    """One bad line must not blank the projection nor fail the whole file,
    and its unredactable content must never leave the host."""

    identity = _experiment_with_trace(
        tmp_path, [{"event_type": "llm_call", "content": "hi"}]
    )
    trace = tmp_path / "experiments/demo/artifacts/traces/run_001.jsonl"
    with trace.open("a", encoding="utf-8") as handle:
        handle.write("{not json\n")
    client = TestClient(create_app(tmp_path))
    trace_ref = identity.trace_ref("run_001")

    payload = client.get(
        "/api/experiments/demo/trace/blocks", params={"run_id": trace_ref}
    ).json()
    assert [block["kind"] for block in payload["blocks"]] == ["agent_output", "raw"]
    assert payload["blocks"][1]["text"] == "{not json"

    download = client.get(
        "/api/experiments/demo/trace/download", params={"run_id": trace_ref}
    )
    assert download.status_code == 200
    records = [json.loads(line) for line in download.text.strip().splitlines()]
    assert records[0]["content"] == "hi"
    assert records[1] == {"event_type": "unreadable_line", "bytes": 10}
    assert "not json" not in download.text


def test_trace_stream_nudges_with_offsets_and_no_event_payload(
    tmp_path: Path,
) -> None:
    identity = _experiment_with_trace(
        tmp_path, [{"event_type": "llm_call", "content": "secret-content"}]
    )
    client = TestClient(create_app(tmp_path))
    response = client.get(
        "/api/experiments/demo/trace/stream",
        params={"run_id": identity.trace_ref("run_001")},
    )
    assert response.status_code == 200
    # Compression would hold the nudges back until a buffer fills.
    assert "content-encoding" not in response.headers
    body = response.text
    # The console re-reads /trace/blocks; the stream only says "there is more".
    assert "secret-content" not in body
    assert '"offset"' in body
    assert "id: " in body
    assert "event: eof" in body


def test_the_client_block_window_is_the_one_the_route_accepts() -> None:
    # A larger client window would be refused by the blocks route's bound.
    assert "const MAX_TRACE_BLOCK_BYTES = 32 * 1024 * 1024;" in APP_JS.read_text(encoding="utf-8")
    assert traces.MAX_BLOCK_READ_BYTES == 32 * 1024 * 1024


def _child_events(*, terminal: bool = True) -> list[dict[str, object]]:
    """One Fold child's own records, plus a parent event it must ignore."""

    events: list[dict[str, object]] = [
        {"event_type": "llm_call", "content": "parent turn"},
        {
            "event_type": "subagent_task",
            "ts": "2026-09-01T10:00:00+00:00",
            "task_id": "agent_1",
            "status": "started",
            "role": "developer",
            "model": "qwen-3.8-27b-fp8",
            "thinking": "inherit",
            "inherit_context": False,
            "description": "因子实现",
        },
        {
            "event_type": "subagent_llm",
            "ts": "2026-09-01T10:00:20+00:00",
            "task_id": "agent_1",
            "turn": 1,
            "model": "qwen-3.8-27b-fp8",
            "content": "先看数据布局",
            "usage": {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110},
        },
        {
            "event_type": "subagent_tool",
            "ts": "2026-09-01T10:00:25+00:00",
            "task_id": "agent_1",
            "turn": 1,
            "tool": "read_file",
            "arguments": {"path": "/mnt/agent/workspace/output/main.py", "limit": 40},
            "result": {"ok": True, "value": {"lines": 40}},
        },
        {
            "event_type": "subagent_tool",
            "ts": "2026-09-01T10:00:26+00:00",
            "task_id": "agent_1",
            "turn": 1,
            "tool": "grep",
            "arguments": {"pattern": "trade_date"},
            "result": {"ok": False, "error": "no match"},
        },
        {
            "event_type": "subagent_llm",
            "ts": "2026-09-01T10:01:00+00:00",
            "task_id": "agent_1",
            "turn": 2,
            "content": "已确认字段",
            "usage": {"prompt_tokens": 200, "completion_tokens": 20, "total_tokens": 220},
        },
        {"event_type": "tool_call", "tool": "shell", "result": {"ok": True}},
    ]
    if not terminal:
        return events
    events.extend(
        [
            {
                "event_type": "subagent_wrap_up",
                "ts": "2026-09-01T10:01:10+00:00",
                "task_id": "agent_1",
                "turn": 2,
                "turns_limit": 3,
            },
            {
                "event_type": "subagent_steer",
                "ts": "2026-09-01T10:01:15+00:00",
                "task_id": "agent_1",
                "turn": 3,
                "chars": 18,
                "delivery": "delivered",
            },
            {
                # The compaction record is nested: its own status/summary must
                # not read as the child's outcome or report.
                "event_type": "subagent_context_compaction",
                "ts": "2026-09-01T10:01:20+00:00",
                "task_id": "agent_1",
                "turn": 3,
                "compaction": {
                    "status": "error",
                    "error": "compaction response is empty",
                    "summary": "should not become the child's report",
                    "estimated_tokens": 201000,
                    "messages_before": 40,
                    "messages_after": 40,
                },
            },
            {
                "event_type": "subagent",
                "ts": "2026-09-01T10:02:00+00:00",
                "task_id": "agent_1",
                "status": "completed",
                "role": "developer",
                "turns": 2,
                "llm_calls": 3,
                "tool_calls": 2,
                "truncated": True,
                "summary": "因子已实现并通过静态检查。",
                "usage_totals": {
                    "prompt_tokens": 300,
                    "completion_tokens": 30,
                    "total_tokens": 330,
                },
            },
        ]
    )
    return events


def test_project_subagent_trace_orders_rounds_tools_and_summary() -> None:
    projected = traces.project_subagent_trace(_child_events(), "agent_1")
    assert projected["found"] is True
    # Four markers: the wrap-up prompt, the parent's steer, the child's own
    # context compaction, then the output-truncation notice.
    assert [block["kind"] for block in projected["blocks"]] == [
        "agent_output",
        "tool_group",
        "agent_output",
        "marker",
        "marker",
        "marker",
        "marker",
        "summary",
    ]
    first, group, second, marker, steer, compaction, cut_off, summary = projected["blocks"]
    assert steer["label"] == "父代理指令" and steer["text"] == "第 3 回合前送达（18 字符）。"
    assert compaction["label"] == "上下文压缩"
    assert "第 3 回合：error，消息 40→40，估算 201000 tokens" in compaction["text"]
    assert first["turn"] == 1 and first["text"] == "先看数据布局"
    assert second["turn"] == 2
    assert [row["name"] for row in group["tools"]] == ["read_file", "grep"]
    calls = group["calls"]
    assert [call["name"] for call in calls] == ["read_file", "grep"]
    assert calls[0]["arguments"]["path"] == "/mnt/agent/workspace/output/main.py"
    assert calls[0]["arguments"]["limit"] == "40"
    assert calls[0]["status"] == "ok" and "lines" in str(calls[0]["result"])
    assert calls[1]["status"] == "failed" and calls[1]["error"] == "no match"
    assert marker["label"] == "收尾提示"
    assert cut_off["label"] == "输出截断"
    assert summary["text"] == "因子已实现并通过静态检查。"
    # The terminal event's own totals drive the header, as on the outer card.
    header = projected["header"]
    assert header["kind"] == "subagent" and header["phase"] == "ended"
    assert header["status"] == "completed"
    assert header["role"] == "developer"
    assert header["model"] == "qwen-3.8-27b-fp8"
    assert header["thinking"] == "inherit"
    assert (header["turns"], header["llm_calls"], header["tool_calls"]) == (2, 3, 2)
    assert header["usage"]["total_tokens"] == 330
    assert header["started_at"] == "2026-09-01T10:00:00+00:00"
    assert header["ended_at"] == "2026-09-01T10:02:00+00:00"


def test_project_subagent_trace_running_child_has_no_report_yet() -> None:
    projected = traces.project_subagent_trace(_child_events(terminal=False), "agent_1")
    header = projected["header"]
    assert header["phase"] == "started" and header["status"] == "running"
    assert "ended_at" not in header
    # Live accumulation, not terminal totals.
    assert header["llm_calls"] == 2 and header["turns"] == 2
    assert header["usage"]["total_tokens"] == 330
    assert all(block["kind"] != "summary" for block in projected["blocks"])
    assert [block["kind"] for block in projected["blocks"]] == [
        "agent_output",
        "tool_group",
        "agent_output",
    ]


def test_subagent_trace_route_projects_redacts_and_guards(tmp_path: Path) -> None:
    events = _child_events()
    events[3]["arguments"] = {  # type: ignore[index]
        "path": "/Data2/host/secret/experiments/demo/output/main.py"
    }
    identity = _experiment_with_trace(tmp_path, events)
    client = TestClient(create_app(tmp_path))
    trace_ref = identity.trace_ref("run_001")

    payload = client.get(
        "/api/experiments/demo/trace/subagents/agent_1",
        params={"run_id": trace_ref},
    )
    assert payload.status_code == 200
    body = payload.json()
    assert body["task_id"] == "agent_1"
    assert [block["kind"] for block in body["blocks"]] == [
        "agent_output",
        "tool_group",
        "agent_output",
        "marker",
        "marker",
        "marker",
        "marker",
        "summary",
    ]
    assert body["header"]["status"] == "completed"
    text = json.dumps(body, ensure_ascii=False)
    assert "/Data2/host/secret" not in text
    assert str(tmp_path) not in text

    assert (
        client.get(
            "/api/experiments/demo/trace/subagents/agent_missing",
            params={"run_id": trace_ref},
        ).status_code
        == 404
    )
    bad = client.get(
        "/api/experiments/demo/trace/subagents/..%2Fetc",
        params={"run_id": trace_ref},
    )
    assert bad.status_code in {400, 404}
    assert "etc" not in bad.text


def test_a_trace_longer_than_the_block_window_shows_its_newest_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A full-budget trace outgrows the block window. The default read — the
    live view and a replay's first load — projects the newest window from a
    line start and says the history before it is cut, so a live view keeps
    following the session; an explicit window still reads from its offset; and
    a sub-agent that ran before the window is still found."""

    turns = [{"event_type": "llm_call", "content": f"turn {index:02d} " + "x" * 200} for index in range(40)]
    path = _write_trace(tmp_path / "long.jsonl", [*_child_events(), *turns])
    monkeypatch.setattr(traces, "MAX_BLOCK_READ_BYTES", 4_000)

    page = traces.read_trace_blocks(path, writer_alive=False)
    shown = [block["text"][:7] for block in page["blocks"] if block["kind"] == "agent_output"]
    assert shown[-1] == "turn 39" and "turn 00" not in shown and "parent " not in shown
    assert page["history_truncated"] is True
    assert page["eof"] is True and page["next_offset"] == path.stat().st_size

    head = traces.read_trace_blocks(path, offset=0, max_bytes=4_000, writer_alive=False)
    assert head["blocks"][0]["text"] == "parent turn" and head["eof"] is False

    child = traces.read_subagent_trace(path, "agent_1")
    assert child["found"] is True and child["header"]["status"] == "completed"
    assert [block["kind"] for block in child["blocks"]][:2] == ["agent_output", "tool_group"]
    assert traces.read_subagent_trace(path, "agent_9")["found"] is False


def _app_js_fragment(script: str, head: str, end: str = "\n}\n") -> str:
    """One top-level declaration from app.js, by its opening line."""

    start = script.index(head)
    return script[start : script.index(end, start) + len(end)]


def _run_app_js_snippet(tmp_path: Path, body: str, *, heads: tuple[str, ...]) -> str:
    """Run app.js declarations against a minimal DOM stub under node."""

    node = shutil.which("node")
    if node is None:  # pragma: no cover - node is present on the dev host
        pytest.skip("node is required for the DOM-stub render check")
    script = APP_JS.read_text(encoding="utf-8")
    stub = """
const TS_TIME_FMT = new Intl.DateTimeFormat("zh-CN", {
  timeZone: "Asia/Shanghai", hour12: false, month: "2-digit", day: "2-digit",
  hour: "2-digit", minute: "2-digit", second: "2-digit",
});
const liveTimers = [];
function el(tag, attrs = {}, ...children) {
  const node = {
    tag, attrs, children: [], dataset: {}, textContent: "",
    append(...items) {
      for (const item of items.flat())
        if (item !== null && item !== undefined) this.children.push(item);
    },
    get childNodes() { return this.children; },
  };
  node.append(...children);
  return node;
}
const nodeText = (node) =>
  typeof node === "string" ? node : node.textContent || node.children.map(nodeText).join("");
const tickAll = (node) => {
  if (node.dataset && node.dataset.elapsedFrom) tickElapsedClocks(node);
  for (const child of node.children || []) if (typeof child !== "string") tickAll(child);
};
"""
    parts = [
        _app_js_fragment(script, head, "]);\n")
        for head in (
            "const SUBAGENT_STATUS_LABELS = new Map([",
            "const TERMINAL_SUBAGENT_STATUS = new Set([",
            "const TOOL_STATUS_LABELS = new Map([",
        )
    ]
    parts = [stub, *parts] + [_app_js_fragment(script, head) for head in heads]
    source = tmp_path / "render.mjs"
    source.write_text("\n".join(parts) + "\n" + body, encoding="utf-8")
    result = subprocess.run(
        [node, str(source)], capture_output=True, text=True, timeout=60, check=False
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def test_the_running_card_renders_meta_progress_and_last_tool(
    tmp_path: Path,
) -> None:
    """The card's three lines while the child runs: model · 推理 · 上下文 · ⏱ ·
    启动时间, the round/call/token progress, and the latest tool."""

    body = """
const detail = { params: { reasoning_effort: "xhigh" } };
const started = new Date(Date.now() - 188000).toISOString();
const block = {
  kind: "subagent", phase: "started", status: "running", role: "developer",
  description: "因子实现", model: "qwen-3.8-27b-fp8", thinking: "medium",
  inherit_context: false, ts: started, started_at: started,
  turns: 4, llm_calls: 4, tool_calls: 6,
  usage: { prompt_tokens: 16414, completion_tokens: 444, total_tokens: 16858 },
  last_tool: { name: "shell", status: "running" },
};
const head = subagentHeadMetaNode(block, detail);
tickAll(head);
console.log(nodeText(head));
console.log(subagentProgressParts(block).join(" · "));
console.log(subagentLastToolLabel(block));
"""
    lines = _run_app_js_snippet(
        tmp_path,
        body,
        heads=(
            "function fmtDuration(",
            "function fmtTokens(",
            "function fmtTsTime(",
            "function parentReasoningLabel(",
            "function isRunningSubagent(",
            "function elapsedClockNode(",
            "function tickElapsedClocks(",
            "function subagentClockNode(",
            "function subagentThinkingLabel(",
            "function subagentContextLabel(",
            "function subagentMetaLine(",
            "function subagentHeadMetaNode(",
            "function subagentProgressParts(",
            "function subagentUsageTitle(",
            "function subagentLastToolLabel(",
        ),
    ).splitlines()
    assert re.fullmatch(
        r"qwen-3\.8-27b-fp8 · 推理 medium · 独立上下文 ⏱ 3:08 · \d\d-\d\d \d\d:\d\d:\d\d",
        lines[0],
    ), lines[0]
    assert lines[1] == "4 回合 · 模型 4 次 · 工具 6 次 · Σ 17 k tokens"
    assert lines[2] == "最近工具 shell · 进行中"


def test_the_dock_pins_one_block_per_running_child(tmp_path: Path) -> None:
    """Rendered: one pinned block per running child, reading as its card does
    — role · 状态, task, launch meta with the ticking clock, the progress line
    and the control that opens the child's Trace. A finished child is not
    pinned, and with nothing running there is no dock at all."""

    body = """
const detail = { params: { reasoning_effort: "xhigh" } };
const started = new Date(Date.now() - 188000).toISOString();
const blocks = [
  { kind: "subagent", phase: "ended", status: "completed", role: "reader",
    task_id: "agent_0", description: "读文档", ts: started, started_at: started,
    ended_at: new Date().toISOString(), turns: 2, llm_calls: 2 },
  { kind: "agent_output", text: "继续" },
  { kind: "subagent", phase: "started", status: "running", role: "developer",
    task_id: "agent_1", description: "因子实现", model: "qwen-3.8-27b-fp8",
    thinking: "medium", inherit_context: false, ts: started, started_at: started,
    turns: 4, llm_calls: 4, tool_calls: 6,
    usage: { prompt_tokens: 16414, completion_tokens: 444, total_tokens: 16858 },
    last_tool: { name: "shell", status: "running" } },
];
const dock = runningSubagentDock(blocks, detail, "run_1");
console.log(dock.children.length);
const pinned = dock.children[0];
console.log(`${pinned.tag} ${pinned.attrs.class} ${typeof pinned.attrs.onclick}`);
tickAll(pinned);
const summary = pinned.children[0];
console.log(nodeText(summary.children[0]));
console.log(nodeText(summary.children[1]));
console.log(pinned.children.length);
const head = subagentSummaryNode(blocks[2], detail, () => {}).children[0];
console.log(`${head.tag} ${head.attrs.class} ${head.attrs.title} ${typeof head.attrs.onclick}`);
console.log(String(runningSubagentDock(blocks.slice(0, 2), detail, "run_1")));
"""
    lines = _run_app_js_snippet(
        tmp_path,
        body,
        heads=(
            "function fmtDuration(",
            "function fmtTokens(",
            "function fmtTsTime(",
            "function parentReasoningLabel(",
            "function isRunningSubagent(",
            "function elapsedClockNode(",
            "function tickElapsedClocks(",
            "function subagentClockNode(",
            "function subagentThinkingLabel(",
            "function subagentContextLabel(",
            "function subagentMetaLine(",
            "function subagentHeadMetaNode(",
            "function subagentProgressParts(",
            "function subagentUsageTitle(",
            "function subagentLastToolLabel(",
            "function subagentSummaryNode(",
            "function subagentDockBlock(",
            "function runningSubagentDock(",
        ),
    ).splitlines()
    # Only the running child, drawn as a clickable trace block.
    assert lines[0] == "1"
    assert lines[1] == (
        "button trace-block subagent running trace-dock-block function"
    )
    assert re.fullmatch(
        r"🧩 developer · 进行中因子实现qwen-3\.8-27b-fp8 · 推理 medium · "
        r"独立上下文 ⏱ 3:08 · \d\d-\d\d \d\d:\d\d:\d\d",
        lines[2],
    ), lines[2]
    assert lines[3] == "4 回合 · 模型 4 次 · 工具 6 次 · Σ 17 k tokens · 最近工具 shell · 进行中"
    # The pinned block is the readout and nothing else — no label beside it.
    assert lines[4] == "1"
    # The card's head row is a button carrying the same action.
    assert lines[5] == "button head subagent-head 打开子代理 Trace function"
    assert lines[6] == "null"
