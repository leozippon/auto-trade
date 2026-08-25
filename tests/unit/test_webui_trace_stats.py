"""Trace summary aggregation and Agent-visible block projection."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from autotrade.webui import traces
from autotrade.webui.server import create_app
from autotrade.webui.traces import (
    DEFAULT_PAGE_BYTES,
    project_trace_blocks,
    trace_stats,
)

APP_JS = Path(__file__).resolve().parents[2] / "src/autotrade/webui/static/app.js"
INDEX_HTML = Path(__file__).resolve().parents[2] / "src/autotrade/webui/static/index.html"


def _write_trace(path: Path, events: list[dict[str, object]]) -> Path:
    path.write_text(
        "".join(json.dumps(event, ensure_ascii=False) + "\n" for event in events),
        encoding="utf-8",
    )
    return path


def test_trace_stats_counts_unique_explore_tasks_not_calls(tmp_path: Path) -> None:
    path = _write_trace(
        tmp_path / "run.jsonl",
        [
            {
                "event_type": "explore_task_started",
                "task_id": "explore_a",
                "parent_call_id": "call_1",
            },
            {
                "event_type": "explore_task",
                "task_id": "explore_a",
                "status": "started",
            },
            {"event_type": "explore_llm", "task_id": "explore_a"},
            {"event_type": "explore_tool", "task_id": "explore_a"},
            {"event_type": "explore", "task_id": "explore_a"},
            {"event_type": "explore_task", "task_id": "explore_b", "status": "started"},
            {"event_type": "explore_llm", "task_id": "explore_b"},
            {
                "event_type": "llm_call",
                "task_id": "explore_a",
                "usage": {
                    "total_tokens": 9,
                    "prompt_tokens": 6,
                    "completion_tokens": 3,
                },
            },
            {"event_type": "tool_call", "tool": "explore", "task_id": "explore_ignored"},
            {"event_type": "session_start", "task_id": "not_explore"},
        ],
    )
    stats = trace_stats(path)
    counts = stats["counts"]
    assert isinstance(counts, dict)
    assert stats["subagent_tasks"] == 2
    assert counts["llm_call"] == 1
    assert stats["tool_counts"] == {"explore": 1}
    assert stats["llm_total_tokens"] == 9
    assert stats["llm_prompt_tokens"] == 6
    assert stats["llm_completion_tokens"] == 3


def test_trace_stats_old_trace_without_start_still_counts_unique_task(
    tmp_path: Path,
) -> None:
    path = _write_trace(
        tmp_path / "old.jsonl",
        [
            {"event_type": "explore_llm", "task_id": "explore_old"},
            {"event_type": "explore_tool", "task_id": "explore_old"},
            {
                "event_type": "explore",
                "task_id": "explore_old",
                "task": "do not parse this as a task id",
            },
            {"event_type": "explore_llm", "task_id": "  "},
            {"event_type": "explore_task_started", "task": "missing id"},
            {"event_type": "explore", "task_id": 12},
        ],
    )
    assert trace_stats(path)["subagent_tasks"] == 1


def test_trace_stats_ignores_non_explore_events(tmp_path: Path) -> None:
    path = _write_trace(
        tmp_path / "plain.jsonl",
        [
            {"event_type": "llm_call", "task_id": "explore_fake"},
            {"event_type": "tool_call", "tool": "read_file", "task_id": "explore_fake"},
            {"event_type": "context_compaction", "task_id": "explore_fake"},
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
    _write_trace(path, [{"event_type": "explore_task_started", "task_id": "explore_a"}])
    assert trace_stats(path)["subagent_tasks"] == 1
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"event_type": "explore_llm", "task_id": "explore_a"}) + "\n")
        handle.write(json.dumps({"event_type": "explore", "task_id": "explore_a"}) + "\n")
    assert trace_stats(path)["subagent_tasks"] == 1
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"event_type": "explore_tool", "task_id": "explore_b"}) + "\n")
    assert trace_stats(path)["subagent_tasks"] == 2


def test_trace_stats_recomputes_when_cached_summary_lacks_subagent_field(
    tmp_path: Path,
) -> None:
    path = _write_trace(
        tmp_path / "legacy_cache.jsonl",
        [{"event_type": "explore_task_started", "task_id": "explore_a"}],
    )
    key = str(path.resolve())
    traces._STATS_CACHE[key] = {
        "offset": path.stat().st_size,
        "counts": {"explore_task_started": 1},
        "tool_counts": {},
        "llm_total_tokens": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "active_tool": None,
        "last_event_ts": None,
    }
    try:
        stats = trace_stats(path)
        counts = stats["counts"]
        assert isinstance(counts, dict)
        assert stats["subagent_tasks"] == 1
        assert counts["explore_task_started"] == 1
    finally:
        traces._STATS_CACHE.pop(key, None)


def test_stats_chips_show_subagent_near_llm_only_when_positive() -> None:
    script = APP_JS.read_text(encoding="utf-8")
    source = script.split("function statsChipsRow(stats)", 1)[1].split("\nfunction ", 1)[0]
    assert "Number(stats.subagent_tasks) || 0" in source
    assert "🧩 子代理" in source
    assert 'key === "llm_call" && subagentTasks' in source
    assert "instruction" not in source
    assert ".task " not in source


def test_live_trace_panel_claims_stream_before_await() -> None:
    script = APP_JS.read_text(encoding="utf-8")
    source = script.split("function liveTracePanel(", 1)[1].split("\nfunction ", 1)[0]
    refresh = source.split("const refreshBlocks", 1)[1].split("const scheduleRefresh", 1)[0]
    claim_at = refresh.find("streamOpening = true")
    await_at = refresh.find("await ")
    assert 0 <= claim_at < await_at
    assert refresh.count("if (claimStream) openStream") == 2
    assert source.count("new EventSource") == 1
    assert "refreshBlocks();" in source
    assert "await refreshBlocks();" in source


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
) -> None:
    root = tmp_path / "experiments" / experiment_id
    (root / "hitl").mkdir(parents=True)
    (root / "hitl" / "status.json").write_text(
        json.dumps({"schema_version": 1, "state": "created"}),
        encoding="utf-8",
    )
    traces_dir = root / "artifacts" / "traces"
    traces_dir.mkdir(parents=True)
    _write_trace(traces_dir / f"{run_id}.jsonl", events)


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


def test_project_subagent_new_and_old_events_aggregate_by_task_id() -> None:
    blocks = project_trace_blocks(
        [
            {
                "event_type": "explore_task_started",
                "task_id": "explore_new",
                "task": "inspect schema",
            },
            {"event_type": "explore_llm", "task_id": "explore_new"},
            {
                "event_type": "explore_tool_started",
                "task_id": "explore_new",
                "tool": "grep",
                "tool_call_id": "g1",
            },
            {
                "event_type": "explore_tool",
                "task_id": "explore_new",
                "tool": "grep",
                "tool_call_id": "g1",
                "result": {"ok": True},
            },
            {
                "event_type": "explore_task",
                "task_id": "explore_new",
                "status": "completed",
                "digest": "has trade_date",
            },
            {"event_type": "explore_llm", "task_id": "explore_old"},
            {
                "event_type": "explore_tool",
                "task_id": "explore_old",
                "tool": "read_file",
                "result": {"ok": True},
            },
            {
                "event_type": "explore",
                "task_id": "explore_old",
                "task": "old inspect",
                "digest": "old digest",
            },
        ]
    )
    sub = _subagent_blocks(blocks)
    assert [(block["task_id"], block["phase"], block["status"]) for block in sub] == [
        ("explore_new", "started", "running"),
        ("explore_new", "ended", "completed"),
        ("explore_old", "started", "running"),
        ("explore_old", "ended", "completed"),
    ]
    new_ended = next(
        block
        for block in sub
        if block["task_id"] == "explore_new" and block["phase"] == "ended"
    )
    old_ended = next(
        block
        for block in sub
        if block["task_id"] == "explore_old" and block["phase"] == "ended"
    )
    assert new_ended["task"] == "inspect schema"
    assert new_ended["digest"] == "has trade_date"
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
    assert old_ended["task"] == "old inspect"
    assert old_ended["digest"] == "old digest"
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


def test_project_explore_internal_tools_stay_on_subagent_card() -> None:
    blocks = project_trace_blocks(
        [
            {"event_type": "llm_call", "content": "delegate"},
            {
                "event_type": "tool_call_started",
                "tool": "explore",
                "tool_call_id": "e1",
            },
            {
                "event_type": "explore_task_started",
                "task_id": "explore_a",
                "task": "inspect",
            },
            {
                "event_type": "explore_tool_started",
                "task_id": "explore_a",
                "tool": "grep",
                "tool_call_id": "g1",
            },
            {
                "event_type": "explore_tool",
                "task_id": "explore_a",
                "tool": "grep",
                "tool_call_id": "g1",
                "result": {"ok": True},
            },
            {
                "event_type": "explore_tool",
                "task_id": "explore_a",
                "tool": "read_file",
                "result": {"ok": True},
            },
            {
                "event_type": "explore",
                "task_id": "explore_a",
                "digest": "done",
            },
            {
                "event_type": "tool_call",
                "tool": "explore",
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
    assert [row["name"] for row in groups[0]["tools"]] == ["explore"]
    ended = next(
        block
        for block in _subagent_blocks(blocks)
        if block["phase"] == "ended"
    )
    assert [row["name"] for row in ended["tools"]] == ["grep", "read_file"]
    orphan = project_trace_blocks(
        [
            {
                "event_type": "explore_tool",
                "task_id": "explore_b",
                "tool": "grep",
                "result": {"ok": True},
            },
            {"event_type": "explore", "task_id": "explore_b"},
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
            {"event_type": "explore_task", "task_id": "ok", "status": "completed"},
            {
                "event_type": "explore_task",
                "task_id": "err",
                "status": "error",
                "error": "nope",
            },
            {"event_type": "explore_task", "task_id": "to", "status": "timeout"},
            {
                "event_type": "explore_task",
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
    long_task = "任" * (traces._BLOCK_TASK_CHARS + 20)
    long_digest = "摘" * (traces._BLOCK_DIGEST_CHARS + 20)
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
                "event_type": "explore",
                "task_id": "t1",
                "task": long_task,
                "digest": long_digest,
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
    assert len(str(ended["task"])) == traces._BLOCK_TASK_CHARS
    assert len(str(ended["digest"])) == traces._BLOCK_DIGEST_CHARS
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
                {"event_type": "context_compaction", "summary": "compacted"},
                {"event_type": "llm_call", "usage": {"total_tokens": 3}},
                {"event_type": "budget", "remaining": 1},
                {"raw": "not a typed event"},
            ]
        )
        == []
    )


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
        {"event_type": "context_compaction", "blob": "x" * (DEFAULT_PAGE_BYTES + 2048)},
        {
            "event_type": "tool_call",
            "tool": "read_file",
            "tool_call_id": "c1",
            "result": {"ok": True},
        },
        {"event_type": "user_message", "content": "ok?"},
        {"event_type": "llm_call", "content": "done"},
    ]
    _experiment_with_trace(tmp_path, events)
    client = TestClient(create_app(tmp_path))
    raw = client.get("/api/experiments/demo/trace", params={"run_id": "run_001"})
    assert raw.status_code == 200
    raw_payload = raw.json()
    assert "blocks" not in raw_payload
    assert [event.get("event_type") for event in raw_payload["events"]] == [
        "session_start",
        "llm_call",
        "tool_call_started",
    ]
    assert raw_payload["eof"] is False

    response = client.get(
        "/api/experiments/demo/trace/blocks", params={"run_id": "run_001"}
    )
    assert response.status_code == 200
    payload = response.json()
    assert "events" not in payload
    assert payload["eof"] is True
    assert payload["event_count"] == len(events)
    assert payload["blocks"] == project_trace_blocks(events)
    assert [block["kind"] for block in payload["blocks"]] == [
        "agent_output",
        "tool_group",
        "user",
        "agent_output",
    ]
    group = payload["blocks"][1]
    assert group["count"] == 1 and group["ok"] == 1 and group["running"] == 0


def test_trace_blocks_api_guards_invalid_experiment_and_run(tmp_path: Path) -> None:
    _experiment_with_trace(tmp_path, [{"event_type": "llm_call", "content": "x"}])
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
        "/api/experiments/demo/trace/blocks", params={"run_id": "run_missing"}
    )
    assert missing_run.status_code == 404


def test_raw_trace_endpoint_still_returns_unprojected_events(tmp_path: Path) -> None:
    events = [
        {"event_type": "session_start", "system_prompt": "sys", "instruction": "go"},
        {"event_type": "llm_call", "content": "hi"},
        {"event_type": "tool_call", "tool": "shell", "result": {"ok": True}},
    ]
    _experiment_with_trace(tmp_path, events)
    client = TestClient(create_app(tmp_path))
    raw = client.get("/api/experiments/demo/trace", params={"run_id": "run_001"}).json()
    assert "blocks" not in raw
    assert [event.get("event_type") for event in raw["events"]] == [
        "session_start",
        "llm_call",
        "tool_call",
    ]
    assert raw["events"][1]["content"] == "hi"
    assert raw["eof"] is True
    blocks = client.get(
        "/api/experiments/demo/trace/blocks", params={"run_id": "run_001"}
    ).json()
    assert "events" not in blocks
    assert [block["kind"] for block in blocks["blocks"]] == [
        "agent_output",
        "tool_group",
    ]
