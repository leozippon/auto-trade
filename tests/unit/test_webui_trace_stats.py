"""Trace summary aggregation and Agent-visible block projection."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from autotrade.environment.identity import AgentRefStore
from autotrade.webui import traces
from autotrade.webui.public_identity import PublicIdentity
from autotrade.webui.server import create_app
from autotrade.webui.traces import (
    DEFAULT_PAGE_BYTES,
    project_trace_blocks,
    trace_stats,
)

APP_JS = Path(__file__).resolve().parents[2] / "src/autotrade/webui/static/app.js"
STYLE_CSS = Path(__file__).resolve().parents[2] / "src/autotrade/webui/static/style.css"
INDEX_HTML = Path(__file__).resolve().parents[2] / "src/autotrade/webui/static/index.html"


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
                "event_type": "subagent_task_started",
                "task_id": "agent_a",
                "parent_call_id": "call_1",
            },
            {
                "event_type": "subagent_task",
                "task_id": "agent_a",
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
            {"event_type": "subagent_task_started", "task": "missing id"},
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
    _write_trace(path, [{"event_type": "subagent_task_started", "task_id": "agent_a"}])
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


def test_trace_stats_recomputes_when_cached_summary_lacks_subagent_field(
    tmp_path: Path,
) -> None:
    path = _write_trace(
        tmp_path / "legacy_cache.jsonl",
        [{"event_type": "subagent_task_started", "task_id": "agent_a"}],
    )
    key = str(path.resolve())
    traces._STATS_CACHE[key] = {
        "offset": path.stat().st_size,
        "counts": {"subagent_task_started": 1},
        "tool_counts": {},
        "llm_total_tokens": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
    }
    try:
        stats = trace_stats(path)
        counts = stats["counts"]
        assert isinstance(counts, dict)
        assert stats["subagent_tasks"] == 1
        assert counts["subagent_task_started"] == 1
    finally:
        traces._STATS_CACHE.pop(key, None)


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


def test_subagent_trace_card_shows_model_thinking_and_context() -> None:
    script = APP_JS.read_text(encoding="utf-8")
    source = script.split("function subagentSummaryNode(", 1)[1].split(
        "\nfunction ", 1
    )[0]
    assert "🧩" in source
    assert "block.role" in source
    assert "subagentHeadMetaNode" in source
    assert "subagentProgressParts" in source
    assert "subagentLastToolLabel" in source
    # The dead `task` field is gone from both the projection and the UI.
    assert "if (block.task)" not in script
    assert "block.task ||" not in script
    assert "trace-box-scroll" in script
    # Launch metadata is spelled out once, in subagentMetaLine.
    meta = script.split("function subagentMetaLine(", 1)[1].split("\nfunction ", 1)[0]
    assert "block.model" in meta
    assert "subagentThinkingLabel" in meta
    assert "subagentContextLabel" in meta
    assert '.join(" · ")' in meta
    context = script.split("function subagentContextLabel(", 1)[1].split(
        "\nfunction ", 1
    )[0]
    assert "继承上下文" in context and "独立上下文" in context
    assert "续用 ${block.resumed_from}" in context
    # `model · 推理 x · 独立上下文 ⏱ 5:21 · 08-28 15:13:20`: the live clock node
    # and the launch timestamp need explicit separators, not a flex gap.
    head_meta = script.split("function subagentHeadMetaNode(", 1)[1].split(
        "\nfunction ", 1
    )[0]
    assert "subagentMetaLine(block, detail)" in head_meta
    assert 'line.append(" ")' in head_meta
    assert "` · ${launched}`" in head_meta
    assert "fmtTsTime(block.ts)" in head_meta
    # The modal's head is the child's own trace head, not a second copy of
    # the card's launch metadata.
    modal = script.split("async function openSubagentTrace(", 1)[1].split("\nfunction ", 1)[0]
    assert "subagentMetaLine" not in modal and "subagentTraceHead(payload, detail)" in modal
    # Clipping is the backend's job; the console renders what it receives.
    assert ".slice(0, 400)" not in script
    assert ".slice(0, 240)" not in script
    assert ".slice(0, 160)" not in script
    assert "isRunningSubagent" in script


def test_agent_output_title_includes_trace_model_and_reasoning() -> None:
    script = APP_JS.read_text(encoding="utf-8")
    source = script.split("function renderAgentOutputBlock(", 1)[1].split(
        "\nfunction ", 1
    )[0]
    assert "block.model" in source
    assert "parentReasoningLabel" in source
    assert "推理 ${effort}" in source
    assert "params.model" not in source
    assert "qwen-3.8-27b-fp8" not in source
    assert "meta_model" not in source


def test_stats_chips_show_subagent_near_llm_only_when_positive() -> None:
    script = APP_JS.read_text(encoding="utf-8")
    source = script.split("function statsChipsRow(", 1)[1].split("\nfunction ", 1)[0]
    assert "Number(stats.subagent_tasks) || 0" in source
    assert "Number(stats.subagent_running) || 0" in source
    assert "🧩 子代理 ${subagentRunning} 运行 / ${subagentTasks} 累计" in source
    assert 'key === "llm_call" && subagentTasks' in source
    # The main Agent's context is a ring against its model window, the
    # compaction count beside it; the token totals keep their meaning in
    # tooltips rather than in prose.
    assert "ringGauge(" in source
    assert "context_window_tokens" in source
    assert "主 Agent 累计输入 tokens" in source
    assert "主 Agent 累计输出 tokens" in source
    # Child spend is shown beside the parent totals, never folded into them.
    assert "🧩 子代理 Σ ${fmtTokens(subagentTokens)}" in source
    assert "subagent_prompt_tokens" in source
    assert "subagent_completion_tokens" in source
    assert "`⟲ 压缩 ${Number(stats.compact_ops) || 0}`" in source
    assert "compact_ops" in source
    assert "trim_ops" not in source
    assert "clear_ops" not in source
    assert "`Trim ${" not in source
    assert "`Clear ${" not in source
    assert "context_compaction" not in source
    assert "`输入 ${" not in source
    assert "`输出 ${" not in source
    assert "`上下文 ${" not in source
    assert "推理" not in source
    assert "执行中" not in source
    assert "last_llm_prompt_tokens" in source
    assert "instruction" not in source
    assert ".task " not in source


def test_live_trace_panel_claims_stream_before_await() -> None:
    script = APP_JS.read_text(encoding="utf-8")
    source = script.split("function liveTracePanel(", 1)[1].split("\nfunction ", 1)[0]
    refresh = source.split("const refreshBlocks", 1)[1].split("const scheduleRefresh", 1)[0]
    claim_at = refresh.find("streamOpening = true")
    await_at = refresh.find("await ")
    assert 0 <= claim_at < await_at
    assert refresh.count("openStream(") == 2
    # A panel replaced during the fetch opens no stream.
    assert refresh.count("box.isConnected") == 2
    assert source.count("new EventSource") == 1
    assert "refreshBlocks();" in source
    assert "await refreshBlocks();" in source


def test_detail_poll_does_not_rebuild_on_environment_stage() -> None:
    script = APP_JS.read_text(encoding="utf-8")
    source = script.split("async function renderDetailPage(", 1)[1]
    poll = source.split("pollTimer = setInterval", 1)[1].split("}, 4000);", 1)[0]
    assert "route(true)" in poll
    assert "environment_stage" not in poll
    assert "session_key" in poll
    assert "run_ref" in poll


def test_control_panel_shows_skills_only_once_published_and_no_stage_prose() -> None:
    script = APP_JS.read_text(encoding="utf-8")
    source = script.split("function controlPanel(", 1)[1]
    head = source.split("\nfunction ", 1)[0]
    assert "skills ? el(" in head and "`📚 Skills ${skills}`" in head
    assert "generation_id" not in head
    # The stepper says where the arm is; the panel carries the badge, the
    # activity and the budget bars, never a session count.
    assert "研究会话" not in script
    # One activity renderer, drawn here and nowhere else on the page.
    assert "activityNode(" in head
    assert script.count("activityNode(") == 3, "definition, control panel, card"


def test_index_html_loads_app_js_without_inlining_trace_chips() -> None:
    html = INDEX_HTML.read_text(encoding="utf-8")
    assert 'src="/static/app.js"' in html
    assert "🧩 子代理" not in html
    assert "subagent_tasks" not in html


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
                "event_type": "subagent_task_started",
                "task_id": "agent_new",
                "description": "inspect schema",
            },
            {"event_type": "subagent_llm", "task_id": "agent_new"},
            {
                "event_type": "subagent_tool_started",
                "task_id": "agent_new",
                "tool": "grep",
                "tool_call_id": "g1",
            },
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
    assert "rounds_limit" not in card


def test_project_subagent_card_carries_the_effective_turn_budget() -> None:
    """The parent-chosen (or defaulted) ``max_turns`` lands on the card as
    ``rounds_limit`` and the console spells it out next to the thinking level."""

    blocks = project_trace_blocks(
        [
            {
                "event_type": "subagent_task",
                "task_id": "agent_turns",
                "status": "started",
                "role": "developer",
                "thinking": "medium",
                "rounds_limit": 12,
            }
        ]
    )
    (card,) = _subagent_blocks(blocks)
    assert card["rounds_limit"] == 12
    meta = APP_JS.read_text(encoding="utf-8").split("function subagentMetaLine(", 1)[1].split(
        "\nfunction ", 1
    )[0]
    assert "block.rounds_limit" in meta and "上限 ${roundsLimit} 轮" in meta


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
            {"event_type": "subagent_task_started", "task_id": "agent_a"},
            {
                "event_type": "subagent_tool_started",
                "task_id": "agent_a",
                "tool": "grep",
                "tool_call_id": "g1",
            },
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
            {"event_type": "subagent_context_compaction", "task_id": "agent_a", "round": 3, "compaction": {"status": "ok"}},
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
            {"event_type": "subagent_task_started", "task_id": "agent_a", "ts": "t1", "role": "explore"},
            {"event_type": "subagent_llm", "task_id": "agent_a", "round": 1, "usage": {"prompt_tokens": 10, "completion_tokens": 2}},
            {"event_type": "context_compaction", "ts": "t2", "trigger": "agent", "status": "ok"},
            {"event_type": "subagent", "task_id": "agent_a", "ts": "t3", "status": "completed", "rounds": 3, "llm_calls": 3, "tool_calls": 5, "summary": "done"},
        ]
    )
    assert [block["kind"] for block in blocks] == ["subagent", "compaction"]
    card = blocks[0]
    assert (card["task_id"], card["status"], card["phase"]) == ("agent_a", "completed", "ended")
    assert (card["rounds"], card["tool_calls"], card["summary"]) == (3, 5, "done")


def test_trace_stats_carries_the_last_budget_block(tmp_path: Path) -> None:
    """Every budgeted event carries the session's cumulative spend; the stats
    projection hands the newest block on, across appends, so the console's
    budget bars follow a live session."""

    path = _write_trace(
        tmp_path / "run.jsonl",
        [
            {"event_type": "session_start"},
            {"event_type": "llm_call", "budget_used": {"inference_seconds": 10.0, "llm_calls": 1, "replay_years": 0, "null_controls": 0}},
            {"event_type": "tool_call", "tool": "shell", "budget_used": {"inference_seconds": 25.5, "llm_calls": 2, "replay_years": 4, "null_controls": 0}},
        ],
    )
    first = trace_stats(path)
    assert first["budget_used"] == {"inference_seconds": 25.5, "llm_calls": 2, "replay_years": 4, "null_controls": 0}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"event_type": "llm_call", "budget_used": {"inference_seconds": 30.0, "llm_calls": 3, "replay_years": 4, "null_controls": 1}}) + "\n")
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
                "round": 1,
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
                "event_type": "subagent_tool_started",
                "task_id": "agent_live",
                "tool": "shell",
                "tool_call_id": "s1",
            },
            {
                "event_type": "subagent_llm",
                "task_id": "agent_live",
                "round": 2,
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
    assert card["rounds"] == 2 and card["llm_calls"] == 2
    assert card["tool_calls"] == 2
    assert card["usage"] == {
        "prompt_tokens": 3000,
        "completion_tokens": 300,
        "total_tokens": 3300,
    }
    assert card["last_tool"] == {"name": "shell", "status": "running"}


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
                "round": 1,
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
                "rounds": 19,
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
    assert card["rounds"] == 19 and card["llm_calls"] == 20
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
    body = response.text
    # The console re-reads /trace/blocks; the stream only says "there is more".
    assert "secret-content" not in body
    assert '"offset"' in body
    assert "id: " in body
    assert "event: eof" in body


def test_trace_replay_threads_detail_and_clamps_the_block_window() -> None:
    script = APP_JS.read_text(encoding="utf-8")
    assert (
        "traceReplayNode(detail.experiment_id, record.run_ref, detail)"
        in script
    )
    assert "function traceReplayNode(experimentId, runId, detail) {" in script
    source = script.split("function traceReplayNode(", 1)[1].split("\nfunction ", 1)[0]
    assert "detail," in source
    assert "MAX_TRACE_BLOCK_BYTES" in source
    # The client window must never exceed what the route accepts.
    assert f"const MAX_TRACE_BLOCK_BYTES = {32 * 1024 * 1024};" in script.replace(
        "32 * 1024 * 1024", str(32 * 1024 * 1024)
    )
    assert traces.MAX_BLOCK_READ_BYTES == 32 * 1024 * 1024


def _child_events(*, terminal: bool = True, legacy: bool = False) -> list[dict[str, object]]:
    """One Fold child's own records, plus a parent event it must ignore."""

    names = (
        {
            "task": "explore_task",
            "llm": "explore_llm",
            "tool": "explore_tool",
            "end": "explore",
        }
        if legacy
        else {
            "task": "subagent_task",
            "llm": "subagent_llm",
            "tool": "subagent_tool",
            "end": "subagent",
        }
    )
    events: list[dict[str, object]] = [
        {"event_type": "llm_call", "content": "parent turn"},
        {
            "event_type": names["task"],
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
            "event_type": names["llm"],
            "ts": "2026-09-01T10:00:20+00:00",
            "task_id": "agent_1",
            "round": 1,
            "model": "qwen-3.8-27b-fp8",
            "content": "先看数据布局",
            "usage": {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110},
        },
        {
            "event_type": names["tool"],
            "ts": "2026-09-01T10:00:25+00:00",
            "task_id": "agent_1",
            "round": 1,
            "tool": "read_file",
            "arguments": {"path": "/mnt/agent/workspace/output/main.py", "limit": 40},
            "result": {"ok": True, "value": {"lines": 40}},
        },
        {
            "event_type": names["tool"],
            "ts": "2026-09-01T10:00:26+00:00",
            "task_id": "agent_1",
            "round": 1,
            "tool": "grep",
            "arguments": {"pattern": "trade_date"},
            "result": {"ok": False, "error": "no match"},
        },
        {
            "event_type": names["llm"],
            "ts": "2026-09-01T10:01:00+00:00",
            "task_id": "agent_1",
            "round": 2,
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
                "round": 2,
                "rounds_limit": 3,
            },
            {
                "event_type": "subagent_steer",
                "ts": "2026-09-01T10:01:15+00:00",
                "task_id": "agent_1",
                "round": 3,
                "chars": 18,
                "delivery": "delivered",
            },
            {
                # The compaction record is nested: its own status/summary must
                # not read as the child's outcome or report.
                "event_type": "subagent_context_compaction",
                "ts": "2026-09-01T10:01:20+00:00",
                "task_id": "agent_1",
                "round": 3,
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
                "event_type": names["end"],
                "ts": "2026-09-01T10:02:00+00:00",
                "task_id": "agent_1",
                "status": "completed",
                "role": "developer",
                "rounds": 2,
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
    assert steer["label"] == "父代理指令" and steer["text"] == "第 3 轮前送达（18 字符）。"
    assert compaction["label"] == "上下文压缩"
    assert "第 3 轮：error，消息 40→40，估算 201000 tokens" in compaction["text"]
    assert first["round"] == 1 and first["text"] == "先看数据布局"
    assert second["round"] == 2
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
    assert (header["rounds"], header["llm_calls"], header["tool_calls"]) == (2, 3, 2)
    assert header["usage"]["total_tokens"] == 330
    assert header["started_at"] == "2026-09-01T10:00:00+00:00"
    assert header["ended_at"] == "2026-09-01T10:02:00+00:00"


def test_project_subagent_trace_legacy_explore_events_project_the_same() -> None:
    current = traces.project_subagent_trace(_child_events(), "agent_1")
    legacy = traces.project_subagent_trace(_child_events(legacy=True), "agent_1")
    assert [block["kind"] for block in legacy["blocks"]] == [
        block["kind"] for block in current["blocks"]
    ]
    assert legacy["header"]["status"] == "completed"
    assert legacy["header"]["usage"]["total_tokens"] == 330


def test_project_subagent_trace_running_child_has_no_report_yet() -> None:
    projected = traces.project_subagent_trace(_child_events(terminal=False), "agent_1")
    header = projected["header"]
    assert header["phase"] == "started" and header["status"] == "running"
    assert "ended_at" not in header
    # Live accumulation, not terminal totals.
    assert header["llm_calls"] == 2 and header["rounds"] == 2
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


def test_the_running_dock_pins_a_block_per_child_and_opens_its_trace() -> None:
    """A running child is pinned at the bottom edge of the trace box as a
    block reading exactly like its card, and both open the child's own Trace
    through the one control, never a second readout of their own."""

    script = APP_JS.read_text(encoding="utf-8")
    dock = script.split("function runningSubagentDock(", 1)[1].split("\nfunction ", 1)[0]
    assert "filter(isRunningSubagent)" in dock
    assert '"trace-block subagent running trace-dock-block"' in dock
    assert "subagentOpenButton(" in dock
    control = script.split("function subagentOpenButton(", 1)[1].split("\nfunction ", 1)[0]
    assert "subagentSummaryNode(block, detail)" in control
    assert "openSubagentTrace(detail, runRef, block)" in control
    assert "详细 Trace ↗" in control
    render = script.split("function renderTraceBlocks(", 1)[1].split("\nfunction ", 1)[0]
    assert "fragment.append(scroll)" in render
    assert "runningSubagentDock(blocks, detail, runRef)" in render
    # The chip strip and the inline fold it replaced are gone, styles included.
    style = STYLE_CSS.read_text(encoding="utf-8")
    assert "trace-running" not in script and "trace-running" not in style
    assert "subagentInlineTrace" not in script and "subagent-inline" not in style


def test_subagent_trace_opens_as_a_modal_over_the_parent_and_follows_it() -> None:
    """The card and the pinned block open one modal window on the child's own
    Trace, rendered by the parent's block renderer and followed while it runs;
    the parent trace behind it is untouched and cannot scroll."""

    script = APP_JS.read_text(encoding="utf-8")
    assert "async function openSubagentTrace(detail, runRef, block)" in script
    assert "/trace/subagents/${encodeURIComponent(taskId)}" in script
    card = script.split("function renderSubagentBlock(", 1)[1].split("\nfunction ", 1)[0]
    assert 'subagentOpenButton(block, detail, runRef, "subagent-open")' in card
    # A running child's card carries the accent and the live dot; a finished
    # one is the compact card.
    assert 'node.classList.toggle("running", isRunningSubagent(block))' in card
    summary = script.split("function subagentSummaryNode(", 1)[1].split("\nfunction ", 1)[0]
    assert 'running ? el("span", { class: "live-dot"' in summary
    opener = script.split("async function openSubagentTrace(", 1)[1].split("\nfunction ", 1)[0]
    # The child's records go through the same renderer as the parent's, and
    # one box for the modal's lifetime keeps the folds the reader opened.
    assert "renderTraceBlocks(box" in opener
    assert "previous: previousBlocks" in opener
    assert '"subagent-modal"' in opener and "closeModal" in opener
    assert "isRunningSubagent(payload.header || block)" in opener
    assert "clearInterval(poll)" in opener
    # Backdrop, Escape and a still page behind belong to the shared helper.
    show = script.split("function showModal(", 1)[1].split("\nfunction ", 1)[0]
    assert "if (event.target === mask) closeModal();" in show
    assert 'document.addEventListener("keydown", modalEscape)' in show
    assert 'document.body.classList.add("modal-open")' in show
    close = script.split("function closeModal(", 1)[1].split("\nfunction ", 1)[0]
    assert 'document.removeEventListener("keydown", modalEscape)' in close
    assert 'document.body.classList.remove("modal-open")' in close
    assert 'event.key === "Escape"' in script
    assert "body.modal-open #app {" in STYLE_CSS.read_text(encoding="utf-8")
    head = script.split("function subagentTraceHead(", 1)[1].split("\nfunction ", 1)[0]
    assert "subagentHeadMetaNode" in head
    # An inherited level is reported as the effective parent level.
    thinking = script.split("function subagentThinkingLabel(", 1)[1].split(
        "\nfunction ", 1
    )[0]
    assert '"inherit"' in thinking
    assert "（继承）" in thinking
    assert "parentReasoningLabel" in thinking


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
  rounds: 4, llm_calls: 4, tool_calls: 6,
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
    assert lines[1] == "4 轮 · 模型 4 次 · 工具 6 次 · Σ 17 k tokens"
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
    ended_at: new Date().toISOString(), rounds: 2, llm_calls: 2 },
  { kind: "agent_output", text: "继续" },
  { kind: "subagent", phase: "started", status: "running", role: "developer",
    task_id: "agent_1", description: "因子实现", model: "qwen-3.8-27b-fp8",
    thinking: "medium", inherit_context: false, ts: started, started_at: started,
    rounds: 4, llm_calls: 4, tool_calls: 6,
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
console.log(nodeText(pinned.children[1]));
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
            "function subagentOpenButton(",
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
    assert lines[3] == "4 轮 · 模型 4 次 · 工具 6 次 · Σ 17 k tokens · 最近工具 shell · 进行中"
    assert lines[4] == "详细 Trace ↗"
    assert lines[5] == "null"
