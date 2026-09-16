"""The Agent-readable transcript beside the host trace.

The JSONL trace stays host-only (raw run id in its name, 0600); the writer
appends one text block per event to ``artifacts/transcripts/<run_ref>.txt``,
rotated into parts a ``read_file`` can hold, and the session reads it through
the ``trace`` root. The transcript carries what the Agent already saw, in a
shape ``read_file``'s line paging and ``grep`` land on.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from autotrade.environment import runtime
from autotrade.environment.runtime import (
    AgentTraceWriter,
    agent_transcript_dir,
    render_transcript_block,
)
from autotrade.environment.tools import GrepTool, ReadFileTool, SafeWorkspace, SearchRoots, ToolRegistry

IDS = {
    "experiment_id": "exp",
    "epoch_id": "research",
    "session_ref": "session_ref_ab",
    "run_id": "run_ref_x",
    "session_kind": "research",
}


def _writer(tmp_path: Path, **kwargs) -> AgentTraceWriter:
    artifacts = tmp_path / "artifacts"
    return AgentTraceWriter(
        artifacts / "traces" / "run_raw.jsonl",
        ids=IDS,
        transcript_dir=agent_transcript_dir(artifacts),
        **kwargs,
    )


def test_every_event_lands_as_one_readable_block_named_by_the_run_ref(tmp_path: Path):
    writer = _writer(tmp_path)
    writer.emit("session_start", {"system_prompt": "SYSTEM PROMPT TEXT", "instruction": "开始本研究会话"})
    writer.emit(
        "llm_call",
        {"call_index": 1, "status": "ok", "content": "先读数据\n再冒烟", "usage": {"prompt_tokens": 12}},
    )
    writer.emit(
        "tool_call",
        {
            "call_index": 1,
            "tool_call_id": "call_1",
            "tool": "shell",
            "arguments": {"argv": ["python", "notes/probe.py"]},
            "result": {"ok": True, "value": {"stdout": "ic=0.031\n", "exit_code": 0}},
        },
    )
    writer.emit("subagent_tool", {"task_id": "task_1", "round": 2, "tool": "grep", "result": {"ok": True}})

    transcript = tmp_path / "artifacts" / "transcripts" / "run_ref_x.txt"
    text = transcript.read_text(encoding="utf-8")
    blocks = [block for block in text.split("\n\n") if block.strip()]
    assert [block.splitlines()[0].split()[2] for block in blocks] == [
        "session_start",
        "llm_call",
        "tool_call",
        "subagent_tool",
    ]
    # The system prompt is the Agent's own context already: not repeated.
    assert "SYSTEM PROMPT TEXT" not in text
    assert "instruction: 开始本研究会话" in text
    assert "content:\n先读数据\n再冒烟" in text
    assert "=== " in blocks[2] and " tool_call call=1 tool=shell id=call_1" in blocks[2].splitlines()[0]
    # Structured fields are indented JSON, one value per line.
    assert '"argv": [' in blocks[2] and '"stdout": "ic=0.031\\n"' in blocks[2]
    assert "subagent_tool tool=grep task=task_1 round=2" in blocks[3].splitlines()[0]
    # Host bookkeeping stays in the JSONL: no raw run id, no event ids.
    assert "run_raw" not in text and "event_id" not in text and "session_ref_ab" not in text
    jsonl = (tmp_path / "artifacts" / "traces" / "run_raw.jsonl").read_text(encoding="utf-8")
    assert len(jsonl.splitlines()) == 4


def test_long_fields_are_clipped_and_the_transcript_rotates_into_readable_parts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(runtime, "TRANSCRIPT_PART_MAX_BYTES", 2_000)
    writer = _writer(tmp_path, max_event_bytes=runtime.TRACE_MAX_EVENT_BYTES)
    for index in range(6):
        writer.emit("llm_call", {"call_index": index, "content": f"turn {index} " + "x" * 600})
    parts = sorted((tmp_path / "artifacts" / "transcripts").glob("run_ref_x*.txt"))
    # A part is full once it passes the cap, so no part exceeds the cap by more
    # than one block, every block lands whole in one part, and the first part
    # keeps the run ref's plain name.
    assert parts[-1].name == "run_ref_x.txt" and len(parts) >= 2
    assert all(part.stat().st_size < 2_000 + 800 for part in parts)
    texts = [part.read_text(encoding="utf-8") for part in parts]
    assert sum(text.count("=== ") for text in texts) == 6
    assert all(text.startswith("=== ") and text.endswith("\n\n") for text in texts)
    # Every part is a whole number of blocks, so no event is cut in two.
    for part in parts:
        assert part.read_text(encoding="utf-8").startswith("=== ")
    huge = "y" * (runtime.TRACE_CONTENT_PREVIEW_CHARS + 50)
    block = render_transcript_block({"ts": "t", "event_type": "llm_call", "content": huge})
    assert "[... 50 more characters in the host trace]" in block
    assert block.count("y") == runtime.TRACE_CONTENT_PREVIEW_CHARS


def test_the_trace_and_its_transcript_stop_together_at_the_session_cap(tmp_path: Path):
    """One bounded record: the JSONL stops at ``max_bytes`` with its terminal
    marker, and the transcript ends on the same marker instead of growing on
    with events no host reader will ever see."""

    writer = _writer(tmp_path, max_bytes=6_000, max_event_bytes=3_000)
    for index in range(12):
        writer.emit("llm_call", {"call_index": index, "content": f"turn {index} " + "x" * 500})

    jsonl = (tmp_path / "artifacts" / "traces" / "run_raw.jsonl").read_text(encoding="utf-8")
    events = [json.loads(line) for line in jsonl.splitlines()]
    assert events[-1]["event_type"] == "trace_limit_reached"
    assert events[-1]["max_bytes"] == 6_000
    recorded = [event["call_index"] for event in events if event["event_type"] == "llm_call"]
    assert 0 < len(recorded) < 12, "the cap must actually cut this run short"

    parts = sorted((tmp_path / "artifacts" / "transcripts").glob("run_ref_x*.txt"))
    text = "".join(part.read_text(encoding="utf-8") for part in parts)
    assert text.count("=== ") == len(recorded) + 1
    assert "trace_limit_reached" in text
    assert f"turn {recorded[-1]} " in text
    assert f"turn {recorded[-1] + 1} " not in text
    # The marker is written once: later events change neither record.
    writer.emit("session_end", {"status": "ok"})
    assert (tmp_path / "artifacts" / "traces" / "run_raw.jsonl").read_text(encoding="utf-8") == jsonl
    assert "".join(part.read_text(encoding="utf-8") for part in parts) == text


def test_the_session_reads_its_transcript_through_the_trace_root(tmp_path: Path):
    writer = _writer(tmp_path)
    writer.emit("tool_call", {"call_index": 7, "tool": "batch_validate", "result": {"neutralized_excess": 0.041}})
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    roots = SearchRoots(SafeWorkspace(workspace), trace_root=agent_transcript_dir(tmp_path / "artifacts"))
    registry = ToolRegistry([ReadFileTool(roots), GrepTool(roots)])
    assert "trace" in roots.names
    read = registry.invoke("read_file", {"root": "trace", "path": "run_ref_x.txt"})
    assert read.ok, read.error
    assert "batch_validate" in read.value["content"] and '"neutralized_excess": 0.041' in read.value["content"]
    found = registry.invoke(
        "grep",
        {"root": "trace", "path": "run_ref_x.txt", "pattern": "neutralized_excess", "output_mode": "content"},
    )
    assert found.ok, found.error
    assert "0.041" in json.dumps(found.value)
    # Without a transcript directory the root is simply not offered.
    assert "trace" not in SearchRoots(SafeWorkspace(workspace)).names


def test_a_transcript_needs_the_runs_opaque_ref(tmp_path: Path):
    with pytest.raises(ValueError, match="opaque ref"):
        AgentTraceWriter(
            tmp_path / "trace.jsonl",
            ids={"experiment_id": "exp"},
            transcript_dir=tmp_path / "transcripts",
        )


def test_host_paths_never_reach_the_transcript_and_its_files_stay_private(tmp_path: Path):
    """The transcript is the Agent-visible boundary of the trace: an exception
    or a result that names an absolute host path is redacted there while the
    host JSONL keeps it, sandbox mounts under /mnt stay, and the directory and
    its parts are private like the JSONL beside them."""

    import stat

    writer = _writer(tmp_path)
    host_path = f"{tmp_path}/experiments/exp/artifacts/run_raw/result.json"
    writer.emit(
        "session_error",
        {"status": "error", "error": f"FileNotFoundError: [Errno 2] No such file or directory: '{host_path}'"},
    )
    writer.emit(
        "tool_call",
        {
            "call_index": 2,
            "tool_call_id": "call_2",
            "tool": "read_file",
            "arguments": {"root": "output", "path": "main.py"},
            "result": {"ok": False, "error": f"host says {host_path}", "hint": "/mnt/agent/workspace/output/main.py"},
        },
    )
    transcript = tmp_path / "artifacts" / "transcripts" / "run_ref_x.txt"
    text = transcript.read_text(encoding="utf-8")
    assert str(tmp_path) not in text
    assert "[host_path]" in text
    assert "/mnt/agent/workspace/output/main.py" in text
    # The host trace is untouched evidence.
    assert host_path in (tmp_path / "artifacts" / "traces" / "run_raw.jsonl").read_text(encoding="utf-8")
    assert stat.S_IMODE(transcript.parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(transcript.stat().st_mode) == 0o600
